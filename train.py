"""
LISA-3D Stage-1 training: geometry-aware LoRA fine-tuning.

Injects LoRA adapters (r=16, α=32) into the CLIP vision encoder and LLaMA
language decoder of LISA++, then trains them with a two-view multi-view
consistency loss on GraspClutter6D RGB-D scenes.

Only LoRA parameters (~11.6M) are updated; the base LISA++ weights (including
SAM image encoder and mask decoder) stay frozen throughout.

Usage:
    See run_train.sh for a complete launcher.
    Direct call:
        python train.py \
            --version Senqiao/LISA_Plus_7b \
            --vision_pretrained /path/to/sam_vit_h.pth \
            --data_root /home/jhwang/grasp/GraspClutter6D \
            --output_dir ./runs/lisa3d \
            --precision bf16
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA3D import LISA3DForCausalLM, inject_lora
from utils.dataset import GraspClutter6DPairDataset, collate_fn_train
from utils.utils import (
    DEFAULT_IMAGE_TOKEN,
    save_lora_checkpoint,
    load_lora_checkpoint,
)


# ── Argument parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LISA-3D Stage-1 training")
    p.add_argument("--version",           default="Senqiao/LISA_Plus_7b",
                   help="HuggingFace model ID for LISA++")
    p.add_argument("--vision_pretrained", required=True,
                   help="Path to SAM ViT-H checkpoint (sam_vit_h_*.pth)")
    p.add_argument("--data_root",         default="/home/jhwang/grasp/GraspClutter6D")
    p.add_argument("--output_dir",        default="./runs/lisa3d")
    p.add_argument("--precision",         default="bf16",
                   choices=["fp32", "bf16", "fp16"])
    p.add_argument("--image_size",        default=1024,  type=int)
    p.add_argument("--model_max_length",  default=512,   type=int)
    p.add_argument("--lora_r",            default=16,    type=int)
    p.add_argument("--lora_alpha",        default=32.0,  type=float)
    p.add_argument("--lora_dropout",      default=0.05,  type=float)
    p.add_argument("--geo_lambda",        default=0.4,   type=float)
    p.add_argument("--lr",                default=3e-4,  type=float)
    p.add_argument("--weight_decay",      default=0.05,  type=float)
    p.add_argument("--epochs",            default=10,    type=int)
    p.add_argument("--batch_size",        default=1,     type=int)
    p.add_argument("--grad_accum_steps",  default=2,     type=int)
    p.add_argument("--workers",           default=4,     type=int)
    p.add_argument("--resume",            default="",    type=str)
    p.add_argument("--gpu",               default=0,     type=int)
    p.add_argument("--print_freq",        default=50,    type=int)
    p.add_argument("--save_freq",         default=1,     type=int)
    p.add_argument("--vision_tower",
                   default="openai/clip-vit-large-patch14")
    p.add_argument("--conv_type",         default="llava_v1")
    p.add_argument("--camera",            default="realsense-d415")
    p.add_argument("--min_visib_fract",   default=0.1,   type=float)
    p.add_argument("--min_px_visib",      default=200,   type=int)
    p.add_argument("--load_in_4bit",      action="store_true")
    p.add_argument("--load_in_8bit",      action="store_true")
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────

def build_model(args):
    """Load LISA++ and inject geometry-aware LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Add [SEG] segmentation token
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # Precision / quantisation
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

    print(f"Loading LISA++ from {args.version} ...")
    model = LISA3DForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=seg_token_idx,
        vision_pretrained=args.vision_pretrained,
        train_mask_decoder=False,   # SAM mask decoder stays frozen
        out_dim=256,
        **kwargs,
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    # Initialise vision modules
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()

    # Cast and move to GPU
    device = torch.device(f"cuda:{args.gpu}")
    if args.precision == "bf16":
        model = model.bfloat16().to(device)
        vision_tower.to(dtype=torch.bfloat16, device=device)
    elif args.precision == "fp16":
        model = model.half().to(device)
        vision_tower.to(dtype=torch.half, device=device)
    else:
        model = model.float().to(device)
        vision_tower.to(dtype=torch.float32, device=device)

    # Inject geometry-aware LoRA (freezes all params, then adds LoRA)
    print(f"Injecting LoRA (r={args.lora_r}, α={args.lora_alpha}) ...")
    inject_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    # Enable gradient checkpointing to reduce activation memory
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    return model, tokenizer, seg_token_idx


# ── Training loop ─────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    writer,
    args,
    epoch: int,
    device: torch.device,
    global_step: int,
) -> int:
    model.train()
    total_loss      = 0.0
    total_seg_a     = 0.0
    total_seg_b     = 0.0
    total_geo       = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        # Move to GPU
        for k in batch:
            v = batch[k]
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, list):
                batch[k] = [
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in v
                ]

        # Forward pass
        out = model.model_forward_3d(
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

        total_loss  += out["loss"].item()
        total_seg_a += out["seg_loss_a"].item()
        total_seg_b += out["seg_loss_b"].item()
        total_geo   += out["geo_loss"].item()

        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        if (step + 1) % args.print_freq == 0:
            n = step + 1
            print(
                f"Epoch [{epoch}/{args.epochs}] Step [{step+1}/{len(loader)}] "
                f"loss={total_loss/n:.4f}  "
                f"seg_a={total_seg_a/n:.4f}  "
                f"seg_b={total_seg_b/n:.4f}  "
                f"geo={total_geo/n:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            if writer is not None:
                writer.add_scalar("train/loss",    total_loss  / n, global_step)
                writer.add_scalar("train/seg_a",   total_seg_a / n, global_step)
                writer.add_scalar("train/seg_b",   total_seg_b / n, global_step)
                writer.add_scalar("train/geo_loss",total_geo   / n, global_step)
                writer.add_scalar("train/lr",      scheduler.get_last_lr()[0], global_step)

    return global_step


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Optional tensorboard writer
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))
        print(f"TensorBoard logs: {args.output_dir}/tb")
    except ImportError:
        print("tensorboard not available, logging to stdout only.")

    # ── Build model ───────────────────────────────────────────────────────
    model, tokenizer, seg_token_idx = build_model(args)
    model.seg_token_idx = seg_token_idx

    # ── Build dataset ─────────────────────────────────────────────────────
    clip_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)

    with open(
        os.path.join(args.data_root, "split_info", "grasp_train_scene_ids.json")
    ) as f:
        train_scene_ids = [int(s) for s in json.load(f)]

    train_dataset = GraspClutter6DPairDataset(
        data_root=args.data_root,
        tokenizer=tokenizer,
        clip_image_processor=clip_processor,
        scene_ids=train_scene_ids,
        image_size=args.image_size,
        conv_type=args.conv_type,
        camera=args.camera,
        min_visib_fract=args.min_visib_fract,
        min_px_visib=args.min_px_visib,
    )
    print(f"Training dataset: {len(train_dataset)} samples "
          f"from {len(train_scene_ids)} scenes.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn_train,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer and scheduler ───────────────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_params, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = args.epochs * len(train_loader) // args.grad_accum_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_lora_checkpoint(model, optimizer, args.resume)

    # ── Training loop ─────────────────────────────────────────────────────
    global_step = start_epoch * len(train_loader) // args.grad_accum_steps
    for epoch in range(start_epoch + 1, args.epochs + 1):
        global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            writer, args, epoch, device, global_step,
        )

        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"lora_epoch{epoch:03d}.pth"
            )
            save_lora_checkpoint(model, optimizer, epoch, ckpt_path)

    # Save final checkpoint
    save_lora_checkpoint(
        model, optimizer, args.epochs,
        os.path.join(args.output_dir, "lora_final.pth"),
    )
    if writer is not None:
        writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
