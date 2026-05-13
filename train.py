"""
LISA-3D Stage-1 training: per-object geometry-aware fine-tuning.

Train granularity is one (scene, frame-pair, Object Name) anchor: the model
is asked "Please segment the {name} in this image." and supervised on the
mask_visib union of every obj_id sharing that Object Name in the frame.

Three trainable-parameter modes (CLI ``--unfreeze_mode``):
  A   : LoRA only (CLIP + LLaMA self-attn).  Paper default.
  B   : A + ``text_hidden_fcs`` + SAM ``visual_model.mask_decoder``.
  B+  : B + ``lm_head`` + ``embed_tokens``.

Multi-GPU is driven by ``torchrun --nproc_per_node=$NUM_GPUS``: the script
reads ``RANK/LOCAL_RANK/WORLD_SIZE`` from env, wraps the model in DDP, and
guards all logging / checkpoint writes on rank-0.  Single-GPU (``torchrun
--nproc_per_node=1`` or plain python) also works through the no-op fallback.

See ``run_train_A.sh`` / ``run_train_B.sh`` / ``run_train_Bplus.sh``.
"""

import argparse
import csv
import json
import os
import sys
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA3D import LISA3DForCausalLM, inject_lora, unfreeze_seg_heads
from utils.category_map import (
    dump_objects_json,
    load_obj_id_to_name,
    name_to_obj_ids,
)
from utils.dataset import GraspClutter6DPairDataset, collate_fn_train
from utils.utils import (
    DEFAULT_IMAGE_TOKEN,
    save_lora_checkpoint,
    load_lora_checkpoint,
)


# ── Argument parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LISA-3D per-object training")
    p.add_argument("--version",           default="Senqiao/LISA_Plus_7b",
                   help="HuggingFace model ID for LISA++")
    p.add_argument("--vision_pretrained", required=True,
                   help="Path to SAM ViT-H checkpoint (sam_vit_h_*.pth)")
    p.add_argument("--data_root",         default="/home/jhwang/grasp/graspclutter6dAPI/GraspClutter6D")
    p.add_argument("--csv_path",          default="",
                   help="obj_id<->name CSV. Default: "
                        "<graspclutter6dAPI>/GraspClutter6D/graspclutter6d_object_id.csv")
    p.add_argument("--target_names",      default="",
                   help="Comma-separated Object Names to restrict training to. "
                        "Empty = all 194 names from the CSV.")
    p.add_argument("--output_dir",        default="./runs/lisa3d")
    p.add_argument("--precision",         default="bf16",
                   choices=["fp32", "bf16", "fp16"])
    p.add_argument("--image_size",        default=1024,  type=int)
    p.add_argument("--model_max_length",  default=512,   type=int)
    p.add_argument("--lora_r",            default=16,    type=int)
    p.add_argument("--lora_alpha",        default=32.0,  type=float)
    p.add_argument("--lora_dropout",      default=0.05,  type=float)
    p.add_argument("--unfreeze_mode",     default="A",
                   choices=["A", "B", "B+"],
                   help="Which segmentation heads to unfreeze on top of LoRA.")
    p.add_argument("--head_lr_scale",     default=0.3,   type=float,
                   help="LR multiplier for unfrozen head params (vs --lr).")
    p.add_argument("--geo_lambda",        default=0.4,   type=float)
    p.add_argument("--lr",                default=3e-4,  type=float)
    p.add_argument("--weight_decay",      default=0.05,  type=float)
    p.add_argument("--epochs",            default=10,    type=int)
    p.add_argument("--batch_size",        default=1,     type=int)
    p.add_argument("--grad_accum_steps",  default=2,     type=int)
    p.add_argument("--workers",           default=4,     type=int)
    p.add_argument("--resume",            default="",    type=str)
    p.add_argument("--print_freq",        default=50,    type=int)
    p.add_argument("--save_freq",         default=1,     type=int)
    p.add_argument("--save_steps",        default=0,     type=int,
                   help="If >0, save an intra-epoch checkpoint every N "
                        "optimizer steps to <output_dir>/lora_<mode>_step_"
                        "latest.pth (atomic rename). 0 = epoch-boundary "
                        "saves only (legacy behaviour).")
    p.add_argument("--vision_tower",
                   default="openai/clip-vit-large-patch14")
    p.add_argument("--conv_type",         default="llava_v1")
    p.add_argument("--camera",            default="realsense-d415")
    p.add_argument("--min_visib_fract",   default=0.1,   type=float)
    p.add_argument("--min_px_visib",      default=200,   type=int)
    p.add_argument("--load_in_4bit",      action="store_true")
    p.add_argument("--load_in_8bit",      action="store_true")

    # ── Sweep / HP-tuning integration (tune.py talks to train.py through
    #    these flags + the trial output directory).  Default values keep
    #    standalone training unchanged.
    p.add_argument("--val_scene_ids_path", default="", type=str,
                   help="JSON list of scene_ids reserved for validation. "
                        "If empty, no validation pass is run.")
    p.add_argument("--train_scene_ids_path", default="", type=str,
                   help="Override path to the train scene_id JSON. "
                        "Default = <data_root>/split_info/"
                        "grasp_train_scene_ids.json.")
    p.add_argument("--max_anchors_per_epoch", default=0, type=int,
                   help="If >0, take a random Subset of train_dataset of "
                        "this size per epoch (sweep-time budget cap).")
    p.add_argument("--trial_id", default="", type=str,
                   help="Optional identifier written into trial_result.json.")
    p.add_argument("--prune_sentinel", default="", type=str,
                   help="Path that, if present at an epoch boundary, makes "
                        "rank-0 dump a 'pruned' trial_result.json and exit.")
    return p.parse_args()


# ── Distributed setup ─────────────────────────────────────────────────────

def setup_ddp() -> Tuple[int, int, int]:
    """Read LOCAL_RANK/RANK/WORLD_SIZE from env and initialise NCCL.

    Returns (local_rank, global_rank, world_size).  Falls back to a 0/0/1
    no-op when ``torchrun`` env vars are absent (single-process / plain
    ``python train.py``).
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return local_rank, global_rank, world_size


# ── Model loading ─────────────────────────────────────────────────────────

def build_model(args, local_rank: int, is_rank0: bool):
    """Load LISA++, inject LoRA, optionally unfreeze segmentation heads."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif args.load_in_8bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["visual_model"],
            ),
        })

    if is_rank0:
        print(f"Loading LISA++ from {args.version} ...")
    model = LISA3DForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=seg_token_idx,
        vision_pretrained=args.vision_pretrained,
        # B / B+ keep the mask decoder trainable; A reaches it via
        # unfreeze_seg_heads anyway, so leave config flag at False here.
        train_mask_decoder=False,
        out_dim=256,
        # Loss weights consumed by LISA's model_forward; without these the
        # training-mode path raises ``Tensor * NoneType``.
        ce_loss_weight=1.0,
        bce_loss_weight=2.0,
        dice_loss_weight=0.5,
        **kwargs,
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device(f"cuda:{local_rank}")
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()

    if args.precision == "bf16":
        model = model.bfloat16().to(device)
        vision_tower.to(dtype=torch.bfloat16, device=device)
    elif args.precision == "fp16":
        model = model.half().to(device)
        vision_tower.to(dtype=torch.half, device=device)
    else:
        model = model.float().to(device)
        vision_tower.to(dtype=torch.float32, device=device)

    if is_rank0:
        print(f"Injecting LoRA (r={args.lora_r}, α={args.lora_alpha}) ...")
    inject_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                dropout=args.lora_dropout)
    unfreeze_seg_heads(model, args.unfreeze_mode)

    # inject_lora creates new nn.Linear adapters on CPU/fp32; bring them onto
    # the same device & dtype as the base model so DDP sees a single device
    # type across all parameters.
    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.half,
        "fp32": torch.float32,
    }[args.precision]
    model.to(device=device, dtype=target_dtype)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    return model, tokenizer, seg_token_idx, device


# ── Loss curve logging ────────────────────────────────────────────────────

class LossLogger:
    """CSV-backed loss curve recorder (rank-0 only).

    Writes two files under ``output_dir``:
      * ``loss_log.csv``   — one row per training step (instantaneous values).
      * ``loss_epoch.csv`` — one row per epoch (averages over the epoch).

    Both files are append-friendly so ``--resume`` runs extend the curve
    rather than clobbering it. Plot rendering happens via ``render_plots``.
    """

    STEP_HEADER  = ["epoch", "step", "global_step",
                    "loss", "seg_a", "seg_b", "geo", "lr"]
    EPOCH_HEADER = ["epoch", "num_steps",
                    "loss", "seg_a", "seg_b", "geo", "lr_end",
                    "val_loss", "val_seg", "val_geo"]

    def __init__(self, output_dir: str):
        self.step_path  = os.path.join(output_dir, "loss_log.csv")
        self.epoch_path = os.path.join(output_dir, "loss_epoch.csv")
        self._step_f = self._open(self.step_path, self.STEP_HEADER)
        self._epoch_f = self._open(self.epoch_path, self.EPOCH_HEADER)
        self._step_w  = csv.writer(self._step_f)
        self._epoch_w = csv.writer(self._epoch_f)

    @staticmethod
    def _open(path: str, header):
        is_new = (not os.path.exists(path)) or os.path.getsize(path) == 0
        f = open(path, "a", newline="")
        if is_new:
            csv.writer(f).writerow(header)
            f.flush()
        return f

    def log_step(self, epoch: int, step: int, global_step: int,
                 loss: float, seg_a: float, seg_b: float, geo: float,
                 lr: float, flush: bool = False) -> None:
        self._step_w.writerow([
            epoch, step, global_step,
            f"{loss:.6f}", f"{seg_a:.6f}", f"{seg_b:.6f}",
            f"{geo:.6f}",  f"{lr:.6e}",
        ])
        if flush:
            self._step_f.flush()

    @staticmethod
    def _fmt(v):
        """Empty cell for None, fixed-point float otherwise."""
        return "" if v is None else f"{float(v):.6f}"

    def log_epoch(self, epoch: int, num_steps: int,
                  loss: float, seg_a: float, seg_b: float, geo: float,
                  lr_end: float,
                  val_loss=None, val_seg=None, val_geo=None) -> None:
        self._epoch_w.writerow([
            epoch, num_steps,
            f"{loss:.6f}", f"{seg_a:.6f}", f"{seg_b:.6f}",
            f"{geo:.6f}",  f"{lr_end:.6e}",
            self._fmt(val_loss), self._fmt(val_seg), self._fmt(val_geo),
        ])
        self._epoch_f.flush()
        self._step_f.flush()

    def close(self) -> None:
        try:
            self._step_f.close()
            self._epoch_f.close()
        except Exception:
            pass

    def render_plots(self, output_dir: str) -> None:
        """Render loss_curve.png / loss_curve_epoch.png from the CSV files.

        Silently skips if matplotlib is unavailable or the CSV is empty.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[loss curve] matplotlib not available, skipping PNG render.")
            return

        def _read(path: str):
            if not os.path.exists(path):
                return None
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            return rows if rows else None

        def _to_floats(rows, col):
            out = []
            for r in rows:
                try:
                    out.append(float(r[col]))
                except (KeyError, ValueError):
                    out.append(float("nan"))
            return out

        def _plot(rows, x_col, out_path, title_suffix):
            x = _to_floats(rows, x_col)
            fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
            for ax, col, ttl in [
                (axes[0, 0], "loss",  "total loss"),
                (axes[0, 1], "seg_a", "L_seg (view a)"),
                (axes[1, 0], "seg_b", "L_seg (view b)"),
                (axes[1, 1], "geo",   "L_geo"),
            ]:
                ax.plot(x, _to_floats(rows, col), linewidth=1.0)
                ax.set_title(ttl)
                ax.grid(alpha=0.3)
            axes[1, 0].set_xlabel(x_col)
            axes[1, 1].set_xlabel(x_col)
            fig.suptitle(f"LISA-3D training loss curves ({title_suffix})")
            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            plt.close(fig)

        step_rows = _read(self.step_path)
        if step_rows:
            _plot(step_rows, "global_step",
                  os.path.join(output_dir, "loss_curve.png"),
                  "per step")
            print(f"[loss curve] wrote {output_dir}/loss_curve.png "
                  f"({len(step_rows)} steps)")

        epoch_rows = _read(self.epoch_path)
        if epoch_rows and len(epoch_rows) > 1:
            _plot(epoch_rows, "epoch",
                  os.path.join(output_dir, "loss_curve_epoch.png"),
                  "per epoch")
            print(f"[loss curve] wrote {output_dir}/loss_curve_epoch.png "
                  f"({len(epoch_rows)} epochs)")


# ── Training loop ─────────────────────────────────────────────────────────

def _move_tensor(t: torch.Tensor, device: torch.device,
                 float_dtype: torch.dtype) -> torch.Tensor:
    """Move tensor to device and, for float tensors, cast to ``float_dtype``.
    Integer / bool tensors keep their original dtype."""
    if t.is_floating_point():
        return t.to(device=device, dtype=float_dtype, non_blocking=True)
    return t.to(device=device, non_blocking=True)


def train_one_epoch(
    model,
    raw_model,
    loader,
    optimizer,
    scheduler,
    writer,
    args,
    epoch: int,
    device: torch.device,
    global_step: int,
    is_rank0: bool,
    world_size: int = 1,
    loss_logger: Optional["LossLogger"] = None,
    start_step: int = 0,
) -> Tuple[int, dict]:
    """Train for one epoch.

    Args:
        start_step: Number of batches to skip at the start of this epoch
                    (used for mid-epoch resume).  Pre-skip batches are still
                    pulled from the DataLoader workers (so DistributedSampler's
                    deterministic shuffle order advances correctly) but are
                    discarded without forward/backward.
    """
    model.train()
    total_loss   = 0.0
    total_seg_a  = 0.0
    total_seg_b  = 0.0
    total_geo    = 0.0
    n_steps_done = 0   # count of un-skipped batches (for running averages)
    optimizer.zero_grad()

    float_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.half,
        "fp32": torch.float32,
    }[args.precision]

    if start_step > 0 and is_rank0:
        print(f"[resume] epoch {epoch}: skipping first {start_step} batches "
              f"to reach saved position.")

    for step, batch in enumerate(loader):
        # Mid-epoch resume: discard batches we already trained on.
        if step < start_step:
            continue

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = _move_tensor(v, device, float_dtype)
            elif isinstance(v, list):
                batch[k] = [
                    _move_tensor(x, device, float_dtype)
                    if isinstance(x, torch.Tensor) else x
                    for x in v
                ]

        out = raw_model.model_forward_3d(
            images_a=batch["images_a"],
            images_clip_a=batch["images_clip_a"],
            masks_list_a=batch["masks_list_a"],
            label_list_a=batch["label_list_a"],
            resize_list_a=batch["resize_list_a"],
            depth_a=batch["depth_a"],
            K_a=batch["K_a"],
            E_a=batch["E_a"],
            images_b=batch["images_b"],
            images_clip_b=batch["images_clip_b"],
            masks_list_b=batch["masks_list_b"],
            label_list_b=batch["label_list_b"],
            resize_list_b=batch["resize_list_b"],
            depth_b=batch["depth_b"],
            K_b=batch["K_b"],
            E_b=batch["E_b"],
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            attention_masks=batch["attention_masks"],
            offset=batch["offset"],
            geo_lambda=args.geo_lambda,
        )

        loss = out["loss"] / args.grad_accum_steps
        loss.backward()

        loss_v  = out["loss"].item()
        seg_a_v = out["seg_loss_a"].item()
        seg_b_v = out["seg_loss_b"].item()
        geo_v   = out["geo_loss"].item()
        total_loss  += loss_v
        total_seg_a += seg_a_v
        total_seg_b += seg_b_v
        total_geo   += geo_v
        n_steps_done += 1

        if is_rank0 and loss_logger is not None:
            loss_logger.log_step(
                epoch=epoch,
                step=step + 1,
                global_step=global_step,
                loss=loss_v,
                seg_a=seg_a_v,
                seg_b=seg_b_v,
                geo=geo_v,
                lr=scheduler.get_last_lr()[0],
                flush=((step + 1) % args.print_freq == 0),
            )

        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Intra-epoch step checkpoint.  Save at a clean boundary —
            # gradients zeroed, optimizer/scheduler advanced — so resume
            # observes the same state we observed here.  Rank-0 writes;
            # all ranks wait at a barrier so they don't race ahead while
            # disk IO is in flight.
            if (args.save_steps > 0
                    and global_step % args.save_steps == 0):
                if is_rank0:
                    ckpt_path = os.path.join(
                        args.output_dir,
                        f"lora_{args.unfreeze_mode}_step_latest.pth",
                    )
                    save_lora_checkpoint(
                        raw_model, optimizer, epoch, ckpt_path,
                        scheduler=scheduler,
                        step_in_epoch=step + 1,
                        global_step=global_step,
                        atomic=True,
                    )
                if world_size > 1:
                    dist.barrier()

        if is_rank0 and (step + 1) % args.print_freq == 0:
            n = max(1, n_steps_done)
            print(
                f"Epoch [{epoch}/{args.epochs}] Step [{step+1}/{len(loader)}] "
                f"loss={total_loss/n:.4f}  "
                f"seg_a={total_seg_a/n:.4f}  "
                f"seg_b={total_seg_b/n:.4f}  "
                f"geo={total_geo/n:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            if writer is not None:
                writer.add_scalar("train/loss",     total_loss / n, global_step)
                writer.add_scalar("train/seg_a",    total_seg_a / n, global_step)
                writer.add_scalar("train/seg_b",    total_seg_b / n, global_step)
                writer.add_scalar("train/geo_loss", total_geo  / n, global_step)
                writer.add_scalar("train/lr",
                                  scheduler.get_last_lr()[0], global_step)

    n_steps = max(1, n_steps_done)
    epoch_avg = {
        "num_steps": n_steps,
        "loss":  total_loss  / n_steps,
        "seg_a": total_seg_a / n_steps,
        "seg_b": total_seg_b / n_steps,
        "geo":   total_geo   / n_steps,
        "lr_end": scheduler.get_last_lr()[0],
    }
    return global_step, epoch_avg


# ── Validation pass (sweep / HP-tuning) ───────────────────────────────────

@torch.no_grad()
def validate_one_epoch(
    model,
    raw_model,
    loader,
    args,
    epoch: int,
    device: torch.device,
    world_size: int,
    is_rank0: bool,
) -> dict:
    """Forward-only pass on the held-out val set; returns averaged metrics.

    Mirrors ``train_one_epoch`` minus backward/optimizer; computes
    ``val_seg = seg_a + seg_b`` and ``val_geo`` per batch, accumulates over
    the val loader, then all-reduces sums across DDP ranks before averaging
    so every rank sees the same numbers (rank-0 logs).
    """
    model.eval()
    float_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.half,
        "fp32": torch.float32,
    }[args.precision]

    sum_seg = torch.zeros(1, device=device)
    sum_geo = torch.zeros(1, device=device)
    n_batches = torch.zeros(1, device=device)

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = _move_tensor(v, device, float_dtype)
            elif isinstance(v, list):
                batch[k] = [
                    _move_tensor(x, device, float_dtype)
                    if isinstance(x, torch.Tensor) else x
                    for x in v
                ]

        out = raw_model.model_forward_3d(
            images_a=batch["images_a"],
            images_clip_a=batch["images_clip_a"],
            masks_list_a=batch["masks_list_a"],
            label_list_a=batch["label_list_a"],
            resize_list_a=batch["resize_list_a"],
            depth_a=batch["depth_a"],
            K_a=batch["K_a"],
            E_a=batch["E_a"],
            images_b=batch["images_b"],
            images_clip_b=batch["images_clip_b"],
            masks_list_b=batch["masks_list_b"],
            label_list_b=batch["label_list_b"],
            resize_list_b=batch["resize_list_b"],
            depth_b=batch["depth_b"],
            K_b=batch["K_b"],
            E_b=batch["E_b"],
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            attention_masks=batch["attention_masks"],
            offset=batch["offset"],
            geo_lambda=args.geo_lambda,
        )

        sum_seg += (out["seg_loss_a"] + out["seg_loss_b"]).detach().float()
        sum_geo += out["geo_loss"].detach().float()
        n_batches += 1

    if world_size > 1:
        dist.all_reduce(sum_seg,   op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_geo,   op=dist.ReduceOp.SUM)
        dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)

    n = max(1.0, n_batches.item())
    val_seg  = sum_seg.item() / n
    val_geo  = sum_geo.item() / n
    val_loss = val_seg + args.geo_lambda * val_geo

    model.train()
    if is_rank0:
        print(f"[val   {epoch:03d}]  val_loss={val_loss:.4f}  "
              f"val_seg={val_seg:.4f}  val_geo={val_geo:.4f}  "
              f"(λ={args.geo_lambda})")
    return {"val_loss": val_loss, "val_seg": val_seg, "val_geo": val_geo}


def _write_trial_result(args, status: str,
                        best_val_loss=None, best_epoch=None,
                        final_val_loss=None, final_train_loss=None,
                        last_epoch=None) -> None:
    """Dump trial_result.json into args.output_dir (rank-0 only).

    ``tune.py`` reads this file to recover the objective value.  Status
    is one of {"completed", "pruned"}; the driver detects "failed" by
    the file being absent after a non-zero subprocess exit code.
    """
    payload = {
        "status": status,
        "trial_id": args.trial_id,
        "unfreeze_mode": args.unfreeze_mode,
        "geo_lambda": args.geo_lambda,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_val_loss": final_val_loss,
        "final_train_loss": final_train_loss,
        "last_epoch": last_epoch,
    }
    path = os.path.join(args.output_dir, "trial_result.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _check_prune_sentinel(args, device, world_size: int,
                          is_rank0: bool) -> bool:
    """Return True on every rank if the sentinel file exists.

    Rank-0 reads the FS; the result is broadcast over NCCL so every rank
    agrees and breaks the loop together (otherwise DDP hangs).
    """
    flag = torch.zeros(1, device=device)
    if is_rank0 and args.prune_sentinel and os.path.exists(args.prune_sentinel):
        flag += 1
    if world_size > 1:
        dist.broadcast(flag, src=0)
    return flag.item() > 0


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    local_rank, global_rank, world_size = setup_ddp()
    is_rank0 = (global_rank == 0)

    if is_rank0:
        os.makedirs(args.output_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()  # ensure output_dir exists on every rank

    # Optional tensorboard writer (rank-0 only)
    writer = None
    if is_rank0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))
            print(f"TensorBoard logs: {args.output_dir}/tb")
        except ImportError:
            print("tensorboard not available, logging to stdout only.")

    # CSV-backed loss curve logger (rank-0 only)
    loss_logger: Optional[LossLogger] = None
    if is_rank0:
        loss_logger = LossLogger(args.output_dir)
        print(f"[loss curve] CSV logs at {loss_logger.step_path} "
              f"and {loss_logger.epoch_path}")

    # ── Object-name mapping ──────────────────────────────────────────────
    csv_path = args.csv_path or os.path.join(
        os.path.dirname(args.data_root.rstrip("/")),
        "graspclutter6dAPI", "GraspClutter6D", "graspclutter6d_object_id.csv",
    )
    # Fallback: try the canonical sibling-repo path used in this monorepo.
    if not os.path.exists(csv_path):
        csv_path = "/home/jhwang/grasp/graspclutter6dAPI/GraspClutter6D/graspclutter6d_object_id.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"obj_id<->name CSV not found.  Pass --csv_path explicitly. "
            f"Tried: {csv_path}"
        )

    id_to_name = load_obj_id_to_name(csv_path)
    nmap = name_to_obj_ids(id_to_name)

    target_names = None
    if args.target_names:
        target_names = [n.strip() for n in args.target_names.split(",") if n.strip()]
        nmap = {n: ids for n, ids in nmap.items() if n in set(target_names)}

    if is_rank0:
        dump_objects_json(
            id_to_name,
            os.path.join(args.output_dir, "objects.json"),
            target_names=target_names,
        )
        print(f"[mapping] objects.json written to {args.output_dir}/objects.json "
              f"({len(nmap)} names)")

    # ── Build model ───────────────────────────────────────────────────────
    model, tokenizer, seg_token_idx, device = build_model(args, local_rank, is_rank0)
    model.seg_token_idx = seg_token_idx

    raw_model = model  # for direct method access; DDP wrapping happens below

    # ── Build dataset ─────────────────────────────────────────────────────
    clip_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)

    # Train scene_ids: --train_scene_ids_path overrides the default split
    # (sweep mode carves a train/val partition; see utils/val_split.py).
    train_scene_ids_path = (
        args.train_scene_ids_path
        or os.path.join(args.data_root, "split_info",
                        "grasp_train_scene_ids.json")
    )
    with open(train_scene_ids_path) as f:
        train_scene_ids = [int(s) for s in json.load(f)]

    train_dataset = GraspClutter6DPairDataset(
        data_root=args.data_root,
        tokenizer=tokenizer,
        clip_image_processor=clip_processor,
        scene_ids=train_scene_ids,
        name_to_obj_ids=nmap,
        image_size=args.image_size,
        conv_type=args.conv_type,
        camera=args.camera,
        min_visib_fract=args.min_visib_fract,
        min_px_visib=args.min_px_visib,
        target_names=target_names,
    )
    if is_rank0:
        print(f"Training dataset: {len(train_dataset)} (scene, frame, name) anchors "
              f"from {len(train_scene_ids)} scenes, "
              f"{len(set(n for _, _, n in train_dataset._items))} unique names.")

    # Sweep budget cap: deterministic subsample of the full anchor list.
    # One fixed permutation for the whole run is sufficient for HP ranking;
    # DistributedSampler then re-shuffles within the subset each epoch.
    if (args.max_anchors_per_epoch > 0
            and args.max_anchors_per_epoch < len(train_dataset)):
        full_n = len(train_dataset)
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(full_n, generator=g)[
            :args.max_anchors_per_epoch
        ].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, perm)
        if is_rank0:
            print(f"[sweep] train_dataset subsampled to "
                  f"{len(train_dataset)} / {full_n} anchors "
                  f"(--max_anchors_per_epoch={args.max_anchors_per_epoch}).")

    if world_size > 1:
        sampler: object = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=global_rank,
            shuffle=True, drop_last=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_fn_train,
        pin_memory=True,
        drop_last=True,
    )

    # ── Validation loader (sweep mode) ────────────────────────────────────
    # When --val_scene_ids_path is supplied (typically by tune.py), build a
    # deterministic-order loader over the held-out scenes; validate after
    # every epoch so the sweep driver can prune based on val_loss.
    val_loader = None
    if args.val_scene_ids_path:
        if not os.path.exists(args.val_scene_ids_path):
            raise FileNotFoundError(
                f"--val_scene_ids_path not found: {args.val_scene_ids_path}"
            )
        with open(args.val_scene_ids_path) as f:
            val_scene_ids = [int(s) for s in json.load(f)]
        val_dataset = GraspClutter6DPairDataset(
            data_root=args.data_root,
            tokenizer=tokenizer,
            clip_image_processor=clip_processor,
            scene_ids=val_scene_ids,
            name_to_obj_ids=nmap,
            image_size=args.image_size,
            conv_type=args.conv_type,
            camera=args.camera,
            min_visib_fract=args.min_visib_fract,
            min_px_visib=args.min_px_visib,
            target_names=target_names,
        )
        if is_rank0:
            print(f"Validation dataset: {len(val_dataset)} anchors "
                  f"from {len(val_scene_ids)} scenes.")
        if world_size > 1:
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=global_rank,
                shuffle=False, drop_last=False,
            )
        else:
            val_sampler = None
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.workers,
            collate_fn=collate_fn_train,
            pin_memory=True,
            drop_last=False,
        )

    # ── DDP wrap ──────────────────────────────────────────────────────────
    if world_size > 1:
        # The two inference-mode passes inside model_forward_3d touch a
        # disjoint param subset on each backward, so find_unused_parameters
        # is required.
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        raw_model = model.module

    # ── Optimizer (two param groups: LoRA + unfrozen heads) ──────────────
    lora_params, head_params = [], []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_A" in name or "lora_B" in name:
            lora_params.append(p)
        else:
            head_params.append(p)
    param_groups = [{"params": lora_params, "lr": args.lr}]
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": args.lr * args.head_lr_scale,
        })
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    if is_rank0:
        n_lora = sum(p.numel() for p in lora_params)
        n_head = sum(p.numel() for p in head_params)
        print(f"[optimizer] LoRA params: {n_lora:,} (lr={args.lr:.2e})  "
              f"| Head params: {n_head:,} "
              f"(lr={args.lr * args.head_lr_scale:.2e})  "
              f"| mode={args.unfreeze_mode}")

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01
    )

    # ── Resume ────────────────────────────────────────────────────────────
    # ``load_lora_checkpoint`` returns {epoch, step_in_epoch, global_step}.
    # step_in_epoch > 0 ⇒ mid-epoch save: re-enter that epoch and skip the
    # first step_in_epoch batches.  step_in_epoch == 0 ⇒ epoch-boundary
    # save (legacy or end-of-epoch): start the NEXT epoch fresh.
    start_epoch = 0
    start_step_in_epoch = 0
    resumed_global_step = 0
    if args.resume:
        state = load_lora_checkpoint(
            raw_model, optimizer, args.resume, scheduler=scheduler,
        )
        if state["step_in_epoch"] > 0:
            # Re-enter the in-progress epoch.  The loop is `for epoch in
            # range(start_epoch + 1, ...)`, so set start_epoch one below
            # the saved epoch so the first iteration uses state["epoch"].
            start_epoch = state["epoch"] - 1
            start_step_in_epoch = state["step_in_epoch"]
        else:
            start_epoch = state["epoch"]
        resumed_global_step = state["global_step"]

    # ── Training loop ─────────────────────────────────────────────────────
    # Sweep bookkeeping (None in standalone runs without val_loader).
    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    last_val_loss: Optional[float] = None
    last_train_loss: Optional[float] = None
    pruned = False
    sweep_mode = bool(args.val_scene_ids_path or args.trial_id
                      or args.prune_sentinel)

    # Prefer the resumed counter when available; fall back to the legacy
    # epoch-derived estimate for old epoch-only checkpoints.
    global_step = (resumed_global_step
                   if resumed_global_step > 0
                   else start_epoch * steps_per_epoch)

    first_epoch_after_resume = True
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        skip_n = start_step_in_epoch if first_epoch_after_resume else 0
        first_epoch_after_resume = False

        global_step, epoch_avg = train_one_epoch(
            model, raw_model, train_loader, optimizer, scheduler,
            writer, args, epoch, device, global_step, is_rank0,
            world_size=world_size,
            loss_logger=loss_logger,
            start_step=skip_n,
        )
        last_train_loss = epoch_avg["loss"]

        # Validation pass (sweep mode only; standalone runs skip).
        val_metrics = {"val_loss": None, "val_seg": None, "val_geo": None}
        if val_loader is not None:
            val_metrics = validate_one_epoch(
                model, raw_model, val_loader, args, epoch, device,
                world_size, is_rank0,
            )
            last_val_loss = val_metrics["val_loss"]
            if best_val_loss is None or last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                best_epoch = epoch

        if is_rank0 and loss_logger is not None:
            loss_logger.log_epoch(
                epoch=epoch,
                num_steps=epoch_avg["num_steps"],
                loss=epoch_avg["loss"],
                seg_a=epoch_avg["seg_a"],
                seg_b=epoch_avg["seg_b"],
                geo=epoch_avg["geo"],
                lr_end=epoch_avg["lr_end"],
                val_loss=val_metrics["val_loss"],
                val_seg=val_metrics["val_seg"],
                val_geo=val_metrics["val_geo"],
            )
            print(f"[epoch {epoch:03d}] avg loss={epoch_avg['loss']:.4f}  "
                  f"seg_a={epoch_avg['seg_a']:.4f}  "
                  f"seg_b={epoch_avg['seg_b']:.4f}  "
                  f"geo={epoch_avg['geo']:.4f}")
            if writer is not None and val_metrics["val_loss"] is not None:
                writer.add_scalar("val/loss",  val_metrics["val_loss"], epoch)
                writer.add_scalar("val/seg",   val_metrics["val_seg"],  epoch)
                writer.add_scalar("val/geo",   val_metrics["val_geo"],  epoch)

        if is_rank0 and epoch % args.save_freq == 0:
            ckpt_path = os.path.join(
                args.output_dir,
                f"lora_{args.unfreeze_mode}_epoch{epoch:03d}.pth",
            )
            save_lora_checkpoint(
                raw_model, optimizer, epoch, ckpt_path,
                scheduler=scheduler,
                step_in_epoch=0,        # epoch boundary: resume next epoch
                global_step=global_step,
            )

        # Sweep prune signal (rank-0 reads sentinel, broadcasts to others).
        if _check_prune_sentinel(args, device, world_size, is_rank0):
            if is_rank0:
                print(f"[sweep] prune sentinel detected at epoch {epoch}; "
                      f"writing trial_result.json (status=pruned) and exiting.")
                _write_trial_result(
                    args, status="pruned",
                    best_val_loss=best_val_loss, best_epoch=best_epoch,
                    final_val_loss=last_val_loss,
                    final_train_loss=last_train_loss,
                    last_epoch=epoch,
                )
            pruned = True
            break

    if is_rank0:
        if not pruned:
            save_lora_checkpoint(
                raw_model, optimizer, args.epochs,
                os.path.join(args.output_dir,
                             f"lora_{args.unfreeze_mode}_final.pth"),
                scheduler=scheduler,
                step_in_epoch=0,
                global_step=global_step,
            )
        if writer is not None:
            writer.close()
        if loss_logger is not None:
            loss_logger.render_plots(args.output_dir)
            loss_logger.close()
        if sweep_mode and not pruned:
            _write_trial_result(
                args, status="completed",
                best_val_loss=best_val_loss, best_epoch=best_epoch,
                final_val_loss=last_val_loss,
                final_train_loss=last_train_loss,
                last_epoch=args.epochs,
            )
        print("Training complete." if not pruned else "Training pruned.")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
