"""
Shared utilities for LISA-3D: constants, loss functions, camera helpers,
image preprocessing, and checkpoint I/O.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

# ── Re-export LISA constants (LISA repo must be on PYTHONPATH) ────────────
# Use importlib to load LISA's utils/utils.py directly, avoiding the
# circular import that arises when both repos share a 'utils' package name.
import importlib.util as _ilu
import os as _os

_lisa_utils_path = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "..", "LISA", "utils", "utils.py",
)
_lisa_utils_spec = _ilu.spec_from_file_location("_lisa_utils", _lisa_utils_path)
_lisa_utils = _ilu.module_from_spec(_lisa_utils_spec)
_lisa_utils_spec.loader.exec_module(_lisa_utils)

IGNORE_INDEX          = _lisa_utils.IGNORE_INDEX           # noqa: F401
IMAGE_TOKEN_INDEX     = _lisa_utils.IMAGE_TOKEN_INDEX       # noqa: F401
DEFAULT_IMAGE_TOKEN   = _lisa_utils.DEFAULT_IMAGE_TOKEN     # noqa: F401
DEFAULT_IM_START_TOKEN = _lisa_utils.DEFAULT_IM_START_TOKEN # noqa: F401
DEFAULT_IM_END_TOKEN  = _lisa_utils.DEFAULT_IM_END_TOKEN    # noqa: F401
DEFAULT_IMAGE_PATCH_TOKEN = _lisa_utils.DEFAULT_IMAGE_PATCH_TOKEN  # noqa: F401
AverageMeter          = _lisa_utils.AverageMeter            # noqa: F401
dict_to_cuda          = _lisa_utils.dict_to_cuda            # noqa: F401

# ── LISA-3D hyper-parameters ──────────────────────────────────────────────
LORA_R       = 16
LORA_ALPHA   = 32.0
LORA_DROPOUT = 0.05
GEO_LAMBDA   = 0.4
DEPTH_MIN_MM = 200.0
DEPTH_MAX_MM = 5000.0

# GraspClutter6D camera types → image-ID offset within 4-camera interleave.
# Frame layout: img_id = ann_id * 4 + cam_offset  (ann_ids 0–12, 13 per scene)
CAM_OFFSETS = {
    "realsense-d415": 1,
    "realsense-d435": 2,
    "azure-kinect":   3,
    "zivid":          4,
}
# Divide raw uint16 depth by this to get metres
CAM_DEPTH_SCALES = {
    "realsense-d415": 1000.0,
    "realsense-d435": 1000.0,
    "azure-kinect":   10000.0,
    "zivid":          10000.0,
}

# ── SAM image normalisation constants (from segment_anything) ─────────────
SAM_PIXEL_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
SAM_PIXEL_STD  = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


# ── Image preprocessing ───────────────────────────────────────────────────

def preprocess_sam(
    x: torch.Tensor,
    img_size: int = 1024,
) -> torch.Tensor:
    """Normalise pixel values and pad to a square SAM input.

    Args:
        x:        (3, H, W) float tensor in [0, 255].
        img_size: Target square size (default 1024).

    Returns:
        (3, img_size, img_size) float tensor, normalised and padded.
    """
    x = (x - SAM_PIXEL_MEAN.to(x)) / SAM_PIXEL_STD.to(x)
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


# ── Camera helpers ────────────────────────────────────────────────────────

def build_intrinsic(cam_K_nested) -> np.ndarray:
    """Build (3, 3) float32 intrinsic matrix from nested list."""
    return np.array(cam_K_nested, dtype=np.float32).reshape(3, 3)


def build_extrinsic(R_w2c_flat, t_w2c_mm) -> np.ndarray:
    """Build (4, 4) float32 world-to-camera matrix.

    Args:
        R_w2c_flat: 9-element sequence (row-major rotation).
        t_w2c_mm:   3-element sequence, translation in millimetres.
                    Converted to metres for consistency with depth.

    Returns:
        (4, 4) float32:
            [[R,    t/1000],
             [0, 0, 0,   1]]
    """
    R = np.array(R_w2c_flat, dtype=np.float32).reshape(3, 3)
    t = np.array(t_w2c_mm,   dtype=np.float32).reshape(3, 1) / 1000.0  # mm → m
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = R
    E[:3, 3:4] = t
    return E


def get_eval_img_ids(
    camera: str = "realsense-d415",
    n_total: int = 13,
    n_select: int = 8,
) -> list:
    """Return the image IDs used during evaluation.

    Mirrors the selection in graspclutter6dAPI/utils/eval_seg_3d_iou.py:
      ann_ids = np.linspace(0, n_total-1, n_select, dtype=int)
      img_ids = [ann_id * 4 + cam_offset  for ann_id in ann_ids]

    For realsense-d415 (offset=1), n_total=13, n_select=8:
      ann_ids → [0, 1, 2, 4, 5, 7, 9, 11]
      img_ids → [1, 5, 9, 17, 21, 29, 37, 45]
    """
    cam_offset = CAM_OFFSETS[camera]
    ann_ids = np.linspace(0, n_total - 1, n_select, dtype=int)
    ann_ids = sorted(set(ann_ids.tolist()))
    return [int(ann_id) * 4 + cam_offset for ann_id in ann_ids]


# ── Loss functions ────────────────────────────────────────────────────────

def bce_loss(pred_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pixel-wise binary cross-entropy averaged over all pixels and masks."""
    return F.binary_cross_entropy_with_logits(
        pred_logit, target, reduction="mean"
    )


def dice_loss(
    pred_logit: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss (operates on sigmoid of logits).

    Args:
        pred_logit: (..., H, W) raw logits.
        target:     (..., H, W) binary ground-truth in [0, 1].

    Returns:
        Scalar Dice loss.
    """
    p = pred_logit.sigmoid().flatten(-2)   # (..., H*W)
    g = target.flatten(-2)
    num = 2.0 * (p * g).sum(-1) + eps
    den = p.sum(-1) + g.sum(-1) + eps
    return (1.0 - num / den).mean()


def seg_loss(
    pred_logit: torch.Tensor,
    target: torch.Tensor,
    bce_w: float = 2.0,
    dice_w: float = 0.5,
) -> torch.Tensor:
    """L_seg = bce_w * BCE + dice_w * Dice for one view's mask predictions."""
    return bce_w * bce_loss(pred_logit, target) + dice_w * dice_loss(pred_logit, target)


def geo_consistency_loss(
    P_a: torch.Tensor,
    P_b: torch.Tensor,
    P_tilde_a2b: torch.Tensor,
    P_tilde_b2a: torch.Tensor,
) -> torch.Tensor:
    """L_geo = ||P_b - sg(P̃_{a→b})||_1 + ||P_a - sg(P̃_{b→a})||_1.

    Callers must pass already-detached warped tensors for stop-gradient.
    """
    return (P_b - P_tilde_a2b).abs().mean() + (P_a - P_tilde_b2a).abs().mean()


# ── Checkpoint I/O ────────────────────────────────────────────────────────

def save_lora_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
) -> None:
    """Save LoRA weights + optimizer state."""
    from model.LISA3D import get_lora_state_dict
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "lora_state_dict": get_lora_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"[checkpoint] Saved LoRA weights to {path}")


def load_lora_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> int:
    """Load LoRA weights + optimizer state. Returns the saved epoch."""
    ckpt = torch.load(path, map_location="cpu")
    from model.LISA3D import load_lora_weights
    load_lora_weights(model, path)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    print(f"[checkpoint] Resumed from epoch {epoch}, path: {path}")
    return epoch
