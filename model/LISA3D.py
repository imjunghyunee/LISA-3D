"""
LISA-3D model: geometry-aware LISA++ with manual LoRA injection.

Extends LISAForCausalLM (LISA++) with:
  - LoRALinear: rank-r adapter wrapping a frozen nn.Linear
  - inject_lora(): path-guarded LoRA injection into CLIP + LLaMA attention,
                   explicitly skipping SAM's visual_model layers
  - LISA3DForCausalLM: subclass adding model_forward_3d() for the two-view
                       geometry-consistent training step

Reference: LISA-3D arXiv 2512.01008, Section 2.2.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# LISA repo must be on PYTHONPATH (see run.sh / run_train.sh)
from model.LISA import LISAForCausalLM
from model.warping import warp_mask


# ── LoRA building blocks ──────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank adapter.

    output = base(x) + (alpha / r) * lora_B(lora_A(dropout(x)))

    lora_A: (r, in_features) — initialised with kaiming_uniform
    lora_B: (out_features, r) — initialised to zero (so adapter starts at 0)
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.linear = linear          # frozen original weight
        self.r = r
        self.scale = alpha / r
        in_f, out_f = linear.in_features, linear.out_features

        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialise
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * self.lora_B(self.lora_A(self.lora_dropout(x)))

    def extra_repr(self) -> str:
        return (
            f"in={self.linear.in_features}, out={self.linear.out_features}, "
            f"r={self.r}, scale={self.scale:.3f}"
        )


# ── LoRA injection ────────────────────────────────────────────────────────

# Linear layer names targeted for LoRA in LLaMA and CLIP attention.
# NOTE: SAM's mask_decoder transformer uses the same names, but is excluded
#       via the path guard below.
_LORA_TARGET_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "out_proj"}


def _should_inject(full_path: str, child_name: str) -> bool:
    """Return True iff this module should receive a LoRA adapter.

    Inclusion rules (must satisfy one):
      • path is inside CLIP vision tower encoder layers
      • path is inside LLaMA self-attention layers

    Exclusion rule (overrides inclusion):
      • path is inside SAM's visual_model (mask decoder, image encoder)
    """
    if child_name not in _LORA_TARGET_NAMES:
        return False
    if "visual_model" in full_path:
        return False
    clip_path = "vision_tower.vision_tower.encoder.layers" in full_path
    llama_path = ("model.layers" in full_path) and ("self_attn" in full_path)
    return clip_path or llama_path


def inject_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> nn.Module:
    """Inject LoRA adapters into CLIP and LLaMA attention layers.

    Steps:
      1. Freeze all parameters in the model.
      2. Walk named_modules(); for each (parent, child) pair where
         _should_inject(full_path, child_name) is True, replace the
         child nn.Linear with a LoRALinear (which has grad-enabled params).

    Returns the model in-place (also returned for chaining).
    """
    # Step 1: freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Step 2: inject LoRA via setattr on parent modules
    # We iterate over named_modules and track parent references manually.
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_path = f"{name}.{child_name}" if name else child_name
            if _should_inject(full_path, child_name) and isinstance(child, nn.Linear):
                lora_layer = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_layer)
                replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[inject_lora] Replaced {replaced} Linear layers. "
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100.0 * trainable / max(total, 1):.2f}%)"
    )
    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA parameters (lora_A / lora_B weights)."""
    return {
        name: param
        for name, param in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }


def load_lora_weights(model: nn.Module, path: str) -> None:
    """Load saved LoRA state dict back into the model (non-strict)."""
    ckpt = torch.load(path, map_location="cpu")
    lora_sd = ckpt.get("lora_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    # Only LoRA keys should be loaded; base model keys are expected as missing
    lora_missing = [k for k in missing if "lora_" in k]
    if lora_missing:
        print(f"[load_lora_weights] WARNING: missing LoRA keys: {lora_missing}")
    if unexpected:
        print(f"[load_lora_weights] WARNING: unexpected keys: {unexpected}")
    print(f"[load_lora_weights] Loaded LoRA weights from {path}")


# ── Loss helpers ──────────────────────────────────────────────────────────

def _bce_dice_loss(
    pred_masks: List[torch.Tensor],
    gt_masks: List[torch.Tensor],
    bce_weight: float = 2.0,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """BCE + Dice segmentation loss over a list of (N_i, H, W) tensors."""
    total_loss = torch.tensor(0.0, device=pred_masks[0].device, dtype=pred_masks[0].dtype)
    n_total = 0
    for pred, gt in zip(pred_masks, gt_masks):
        n = gt.shape[0]
        if n == 0:
            continue

        # BCE
        bce = F.binary_cross_entropy_with_logits(pred, gt, reduction="none")
        bce = bce.flatten(1).mean(1).sum()

        # Dice
        p = pred.sigmoid().flatten(1)
        g = gt.flatten(1)
        dice = 1.0 - (2.0 * (p * g).sum(1) + 1.0) / (p.sum(1) + g.sum(1) + 1.0)
        dice = dice.sum()

        total_loss = total_loss + bce_weight * bce + dice_weight * dice
        n_total += n

    return total_loss / max(n_total, 1)


def _geo_consistency_loss(
    P_a: torch.Tensor,
    P_b: torch.Tensor,
    P_tilde_a2b: torch.Tensor,
    P_tilde_b2a: torch.Tensor,
) -> torch.Tensor:
    """L_geo = ||P_b - sg(P̃_{a→b})||_1 + ||P_a - sg(P̃_{b→a})||_1.

    The stop-gradient is applied externally (callers detach the warped tensors).
    """
    loss_a2b = (P_b - P_tilde_a2b).abs().mean()
    loss_b2a = (P_a - P_tilde_b2a).abs().mean()
    return loss_a2b + loss_b2a


# ── LISA-3D model ─────────────────────────────────────────────────────────

class LISA3DForCausalLM(LISAForCausalLM):
    """LISA++ augmented with a geometry-aware two-view training step.

    Inherits all inference capabilities from LISAForCausalLM (evaluate(),
    model_forward(), etc.) and adds model_forward_3d() for training with
    the multi-view consistency loss from LISA-3D.
    """

    def model_forward_3d(
        self,
        # ── View A ──────────────────────────────────────────────────────
        images_a: torch.FloatTensor,
        images_clip_a: torch.FloatTensor,
        masks_list_a: List[torch.FloatTensor],
        label_list_a: List[torch.Tensor],
        resize_list_a: List[tuple],
        depth_a: torch.FloatTensor,
        K_a: torch.FloatTensor,
        E_a: torch.FloatTensor,
        # ── View B ──────────────────────────────────────────────────────
        images_b: torch.FloatTensor,
        images_clip_b: torch.FloatTensor,
        masks_list_b: List[torch.FloatTensor],
        label_list_b: List[torch.Tensor],
        resize_list_b: List[tuple],
        depth_b: torch.FloatTensor,
        K_b: torch.FloatTensor,
        E_b: torch.FloatTensor,
        # ── Shared text tokens ───────────────────────────────────────────
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        # ── Hyper-parameters ─────────────────────────────────────────────
        geo_lambda: float = 0.4,
        bce_loss_weight: float = 2.0,
        dice_loss_weight: float = 0.5,
        ce_loss_weight: float = 1.0,
        **kwargs,
    ) -> dict:
        """Two-view geometry-aware forward pass for training.

        Computes:
          L_total = L_seg_a + L_seg_b + geo_lambda * L_geo

        where:
          L_seg_x = BCE + Dice segmentation loss for view x
          L_geo   = ||P_b - sg(P̃_{a→b})||_1 + ||P_a - sg(P̃_{b→a})||_1

        Returns dict with loss components for logging.
        """
        # ── View A forward ────────────────────────────────────────────────
        out_a = self.model_forward(
            images=images_a,
            images_clip=images_clip_a,
            input_ids=input_ids,
            labels=labels,
            attention_masks=attention_masks,
            offset=offset,
            masks_list=masks_list_a,
            label_list=label_list_a,
            resize_list=resize_list_a,
            inference=False,
            **kwargs,
        )

        # ── View B forward ────────────────────────────────────────────────
        out_b = self.model_forward(
            images=images_b,
            images_clip=images_clip_b,
            input_ids=input_ids,
            labels=labels,
            attention_masks=attention_masks,
            offset=offset,
            masks_list=masks_list_b,
            label_list=label_list_b,
            resize_list=resize_list_b,
            inference=False,
            **kwargs,
        )

        # ── Segmentation losses ───────────────────────────────────────────
        # model_forward already returns combined ce + bce + dice loss.
        seg_loss_a = out_a["loss"]
        seg_loss_b = out_b["loss"]

        # ── Geometric consistency loss ────────────────────────────────────
        # We need per-image sigmoid probabilities at the original resolution.
        # Re-run a minimal inference-mode pass to extract raw logit maps.
        with torch.no_grad():
            inf_a = self.model_forward(
                images=images_a,
                images_clip=images_clip_a,
                input_ids=input_ids,
                labels=labels,
                attention_masks=attention_masks,
                offset=offset,
                masks_list=masks_list_a,
                label_list=label_list_a,
                resize_list=resize_list_a,
                inference=True,
            )
            inf_b = self.model_forward(
                images=images_b,
                images_clip=images_clip_b,
                input_ids=input_ids,
                labels=labels,
                attention_masks=attention_masks,
                offset=offset,
                masks_list=masks_list_b,
                label_list=label_list_b,
                resize_list=resize_list_b,
                inference=True,
            )

        # pred_masks: list of (N_seg, H, W) per batch item; take first seg token
        pred_a = inf_a["pred_masks"]   # list[(N_seg, H, W)]
        pred_b = inf_b["pred_masks"]

        geo_loss = torch.tensor(0.0, device=images_a.device, dtype=images_a.dtype)
        n_pairs = 0

        for i in range(len(pred_a)):
            if pred_a[i].shape[0] == 0 or pred_b[i].shape[0] == 0:
                continue

            # Take the first segmentation mask per sample (primary object)
            P_a_i = pred_a[i][0].sigmoid()   # (H, W)
            P_b_i = pred_b[i][0].sigmoid()   # (H, W)
            H, W = P_a_i.shape

            # Camera params for this batch item
            K_a_i = K_a[i:i+1]   # (1, 3, 3)
            E_a_i = E_a[i:i+1]
            K_b_i = K_b[i:i+1]
            E_b_i = E_b[i:i+1]
            d_a_i = depth_a[i:i+1]   # (1, H_orig, W_orig)
            d_b_i = depth_b[i:i+1]

            # Resize depth to match mask resolution if needed
            if d_a_i.shape[-2:] != (H, W):
                d_a_i = F.interpolate(
                    d_a_i.unsqueeze(1).float(), size=(H, W), mode="nearest"
                ).squeeze(1)
                d_b_i = F.interpolate(
                    d_b_i.unsqueeze(1).float(), size=(H, W), mode="nearest"
                ).squeeze(1)

            P_a_batch = P_a_i.unsqueeze(0)  # (1, H, W)
            P_b_batch = P_b_i.unsqueeze(0)

            # Warp and stop-gradient
            P_tilde_a2b = warp_mask(
                P_a_batch,
                d_a_i.to(images_a.device),
                K_a_i.to(images_a.device),
                E_a_i.to(images_a.device),
                K_b_i.to(images_a.device),
                E_b_i.to(images_a.device),
            ).detach()  # stop-gradient on warped target

            P_tilde_b2a = warp_mask(
                P_b_batch,
                d_b_i.to(images_a.device),
                K_b_i.to(images_a.device),
                E_b_i.to(images_a.device),
                K_a_i.to(images_a.device),
                E_a_i.to(images_a.device),
            ).detach()

            geo_loss = geo_loss + _geo_consistency_loss(
                P_a_batch, P_b_batch, P_tilde_a2b, P_tilde_b2a
            )
            n_pairs += 1

        if n_pairs > 0:
            geo_loss = geo_loss / n_pairs

        total_loss = seg_loss_a + seg_loss_b + geo_lambda * geo_loss

        return {
            "loss": total_loss,
            "seg_loss_a": seg_loss_a.detach(),
            "seg_loss_b": seg_loss_b.detach(),
            "geo_loss": geo_loss.detach(),
            "ce_loss_a": out_a["ce_loss"].detach(),
            "ce_loss_b": out_b["ce_loss"].detach(),
        }
