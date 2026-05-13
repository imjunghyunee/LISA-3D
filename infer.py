"""
LISA-3D Stage-2 inference: per-Object-Name segmentation + depth-based 3D lifting.

For each scene in the test split (or a user-specified list):
  1. Select the 8 evaluation frames (per Clutt3R-Seg protocol).
  2. For each Object Name (default: all 194 names from the CSV), prompt
     LISA++ with "Please segment the {name} in this image.", obtain a binary
     mask, back-project masked depth pixels to camera-frame 3D coordinates,
     and label every resulting point with that Object Name's sentinel obj_id.
  3. Concatenate all (name, frame) point clouds and save them as
     predictions/{scene_id:06d}/seg_3d.npz for evaluation by
     graspclutter6dAPI/utils/eval_seg_3d_iou.py with the auto-generated
     ``objects.json`` produced alongside the LoRA checkpoint at train time.

Usage:
    See run.sh for a complete launcher.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA3D import LISA3DForCausalLM, inject_lora, load_lora_weights
from utils.category_map import (
    dump_objects_json,
    load_obj_id_to_name,
    name_to_obj_ids,
)
from utils.dataset import GraspClutter6DInferDataset, _build_conversation
from utils.lifting import (
    accumulate_views,
    make_rgba_prompt,
    save_seg_3d,
    unproject_depth,
    write_ply_ascii,
)
from utils.utils import CAM_DEPTH_SCALES


_PROMPT_TEMPLATE = "Please segment the {name} in this image."
_ANSWER_TEMPLATE = "Sure, [SEG]."
_MIN_FG_PIXELS = 50   # below this, skip a name's mask as noise


# ── Argument parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LISA-3D per-Object-Name inference on GraspClutter6D",
    )
    p.add_argument("--version",           default="Senqiao/LISA_Plus_7b")
    p.add_argument("--vision_pretrained", required=True,
                   help="Path to SAM ViT-H checkpoint")
    p.add_argument("--lora_weights",      default="",
                   help="Path to trained LISA-3D .pth file (A / B / B+).")
    p.add_argument("--data_root",         default="/home/jhwang/grasp/graspclutter6dAPI/GraspClutter6D")
    p.add_argument("--csv_path",          default="",
                   help="obj_id<->name CSV.  Default: graspclutter6dAPI one.")
    p.add_argument("--target_names",      default="",
                   help="Comma-separated Object Names to query.  Empty=all.")
    p.add_argument("--output_dir",        default="./predictions")
    p.add_argument("--scene_ids",         nargs="+", type=int, default=None,
                   help="Scene IDs to process. Default: all test scenes.")
    p.add_argument("--precision",         default="bf16",
                   choices=["fp32", "bf16", "fp16"])
    p.add_argument("--image_size",        default=1024, type=int)
    p.add_argument("--model_max_length",  default=512,  type=int)
    p.add_argument("--gpu",               default=0,    type=int)
    p.add_argument("--n_views",           default=8,    type=int)
    p.add_argument("--camera",            default="realsense-d415",
                   choices=["realsense-d415", "realsense-d435", "azure-kinect", "zivid"])
    p.add_argument("--vis_save_path",     default="",
                   help="If set, save overlay images and PLY files here.")
    p.add_argument("--vis_ply",           action="store_true",
                   help="Also write .ply files (ASCII, no open3d needed).")
    p.add_argument("--max_new_tokens",    default=32,   type=int)
    p.add_argument("--conv_type",         default="llava_v1")
    p.add_argument("--vision_tower",
                   default="openai/clip-vit-large-patch14")
    p.add_argument("--load_in_4bit",      action="store_true")
    p.add_argument("--load_in_8bit",      action="store_true")
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────

def build_model(args):
    """Load LISA++ with optional LoRA / unfrozen-head weights."""
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

    print(f"Loading LISA++ from {args.version} ...")
    model = LISA3DForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=seg_token_idx,
        vision_pretrained=args.vision_pretrained,
        train_mask_decoder=False,
        out_dim=256,
        **kwargs,
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
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

    if args.lora_weights:
        print(f"Injecting LoRA and loading weights from {args.lora_weights} ...")
        inject_lora(model, r=16, alpha=32.0)
        load_lora_weights(model, args.lora_weights)
    else:
        print("No LoRA weights — using base LISA++ for inference.")

    model.eval()
    return model, tokenizer, seg_token_idx, device


# ── Single-frame inference ────────────────────────────────────────────────

@torch.no_grad()
def infer_frame(
    model,
    tokenizer,
    seg_token_idx: int,
    item: dict,
    input_ids: torch.LongTensor,
    device: torch.device,
    args,
) -> np.ndarray:
    """Run LISA++ inference for one (frame, prompt) pair. Returns (H, W) bool."""
    img_sam  = item["images_sam"].unsqueeze(0).to(device)
    img_clip = item["images_clip"].unsqueeze(0).to(device)
    input_ids = input_ids.unsqueeze(0).to(device)

    if args.precision == "bf16":
        img_sam  = img_sam.bfloat16()
        img_clip = img_clip.bfloat16()
    elif args.precision == "fp16":
        img_sam  = img_sam.half()
        img_clip = img_clip.half()

    resize_list        = [item["resize_shape"]]
    original_size_list = [item["original_size"]]

    output_ids, pred_masks = model.evaluate(
        images_clip=img_clip,
        images=img_sam,
        input_ids=input_ids,
        resize_list=resize_list,
        original_size_list=original_size_list,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
    )

    if not pred_masks or pred_masks[0].shape[0] == 0:
        H, W = item["original_size"]
        return np.zeros((H, W), dtype=bool)

    mask_logit = pred_masks[0][0]   # (H, W)
    binary_mask = (mask_logit.sigmoid() > 0.5).cpu().numpy().astype(bool)
    return binary_mask


# ── Visualisation helpers ─────────────────────────────────────────────────

def save_overlay(rgb_path: str, binary_mask: np.ndarray, save_path: str) -> None:
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        return
    overlay = rgb.copy()
    overlay[binary_mask] = (
        rgb[binary_mask].astype(np.float32) * 0.4
        + np.array([0, 0, 200], dtype=np.float32) * 0.6
    ).astype(np.uint8)
    cv2.imwrite(save_path, overlay)


def save_rgba(rgb_path: str, binary_mask: np.ndarray, save_path: str) -> None:
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        return
    rgb_np = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgba = make_rgba_prompt(rgb_np, binary_mask)
    rgba_bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(save_path, rgba_bgra)


# ── Per-scene inference ───────────────────────────────────────────────────

def run_scene(
    model,
    tokenizer,
    seg_token_idx: int,
    dataset: GraspClutter6DInferDataset,
    scene_id: int,
    name_to_ids: Dict[str, List[int]],
    name_prompts: Dict[str, torch.Tensor],
    args,
    device: torch.device,
) -> None:
    depth_scale = CAM_DEPTH_SCALES[args.camera]
    scene_dir   = os.path.join(args.data_root, "scenes", f"{scene_id:06d}")
    out_dir     = os.path.join(args.output_dir, f"{scene_id:06d}")
    os.makedirs(out_dir, exist_ok=True)

    scene_items = [
        dataset[i]
        for i in range(len(dataset))
        if dataset._items[i][0] == scene_id
    ]
    if not scene_items:
        print(f"  Scene {scene_id:06d}: no frames found, skipping.")
        return

    all_points: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    n_names = len(name_prompts)
    for item in scene_items:
        img_id   = item["img_id"]
        depth_mm = item["depth"].numpy()
        K        = item["K"].numpy()

        kept = 0
        for name, input_ids in name_prompts.items():
            binary_mask = infer_frame(
                model, tokenizer, seg_token_idx, item, input_ids, device, args,
            )
            if int(binary_mask.sum()) < _MIN_FG_PIXELS:
                continue
            sentinel = int(name_to_ids[name][0])
            pts, lbls = unproject_depth(
                depth_mm=depth_mm,
                K=K,
                mask=binary_mask,
                label=sentinel,
                depth_scale=depth_scale,
            )
            if len(pts) == 0:
                continue
            all_points.append(pts)
            all_labels.append(lbls)
            kept += 1

            if args.vis_save_path:
                rgb_path = os.path.join(scene_dir, "rgb", f"{img_id:06d}.png")
                vis_dir  = os.path.join(args.vis_save_path, f"{scene_id:06d}")
                os.makedirs(vis_dir, exist_ok=True)
                safe_name = name.replace(" ", "_").replace("/", "_")
                save_overlay(
                    rgb_path, binary_mask,
                    os.path.join(vis_dir,
                                 f"{img_id:06d}_{safe_name}_overlay.jpg"),
                )
                save_rgba(
                    rgb_path, binary_mask,
                    os.path.join(vis_dir,
                                 f"{img_id:06d}_{safe_name}_rgba.png"),
                )

        print(f"  Scene {scene_id:06d} frame {img_id:06d}: "
              f"{kept}/{n_names} names produced foreground")

    if not all_points:
        # Save an empty npz so the evaluator can still load this scene.
        empty_pts  = np.zeros((0, 3), dtype=np.float32)
        empty_lbls = np.zeros((0,),   dtype=np.int32)
        save_seg_3d(os.path.join(out_dir, "seg_3d.npz"), empty_pts, empty_lbls)
        print(f"  Scene {scene_id:06d}: no foreground at all — saved empty seg_3d.npz")
        return

    points, labels = accumulate_views(all_points, all_labels)
    npz_path = os.path.join(out_dir, "seg_3d.npz")
    save_seg_3d(npz_path, points, labels)
    print(f"  Saved {len(points)} total points → {npz_path}")

    if args.vis_save_path and args.vis_ply and len(points) > 0:
        ply_path = os.path.join(args.vis_save_path, f"{scene_id:06d}", "seg_3d.ply")
        write_ply_ascii(ply_path, points, labels=labels)
        print(f"  PLY saved → {ply_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def _resolve_csv_path(args) -> str:
    if args.csv_path:
        return args.csv_path
    return "/home/jhwang/grasp/graspclutter6dAPI/GraspClutter6D/graspclutter6d_object_id.csv"


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Object-name mapping ──────────────────────────────────────────────
    csv_path = _resolve_csv_path(args)
    id_to_name = load_obj_id_to_name(csv_path)
    nmap = name_to_obj_ids(id_to_name)

    if args.target_names:
        wanted = {n.strip() for n in args.target_names.split(",") if n.strip()}
        nmap = {n: ids for n, ids in nmap.items() if n in wanted}
    print(f"Inference over {len(nmap)} Object Names.")

    # Write objects.json next to predictions for the evaluator.
    objects_json = os.path.join(args.output_dir, "objects.json")
    dump_objects_json(id_to_name, objects_json,
                      target_names=list(nmap.keys()))
    print(f"Wrote evaluator-format objects.json → {objects_json}")

    # ── Model & tokenizer ────────────────────────────────────────────────
    model, tokenizer, seg_token_idx, device = build_model(args)
    clip_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)

    # Pre-tokenise one prompt per Object Name (text only depends on the name,
    # not on the image, so we cache this once).
    name_prompts: Dict[str, torch.Tensor] = {}
    for name in nmap.keys():
        prompt = _PROMPT_TEMPLATE.format(name=name)
        ids, _, _ = _build_conversation(
            prompt, _ANSWER_TEMPLATE, tokenizer, args.conv_type,
        )
        name_prompts[name] = ids

    # ── Scene list ───────────────────────────────────────────────────────
    if args.scene_ids is not None:
        scene_ids = args.scene_ids
    else:
        with open(
            os.path.join(args.data_root, "split_info", "grasp_test_scene_ids.json")
        ) as f:
            scene_ids = [int(s) for s in json.load(f)]

    print(f"Running inference on {len(scene_ids)} scenes ...")
    print(f"Camera: {args.camera} | Views/scene: {args.n_views}")
    print("─" * 60)

    # Single dataset reused for all scenes (the prompt baked into the
    # dataset is a placeholder; the actual prompts are injected per-name in
    # ``run_scene`` via ``name_prompts``).
    dataset = GraspClutter6DInferDataset(
        data_root=args.data_root,
        tokenizer=tokenizer,
        clip_image_processor=clip_processor,
        scene_ids=scene_ids,
        prompt=_PROMPT_TEMPLATE.format(name="object"),  # placeholder
        camera=args.camera,
        image_size=args.image_size,
        conv_type=args.conv_type,
        n_views=args.n_views,
    )

    for scene_id in scene_ids:
        print(f"Scene {scene_id:06d}:")
        run_scene(
            model, tokenizer, seg_token_idx,
            dataset, scene_id, nmap, name_prompts, args, device,
        )

    print("=" * 60)
    print(f"Predictions saved to: {args.output_dir}")
    print(
        "Evaluate with:\n"
        f"  python /home/jhwang/grasp/graspclutter6dAPI/utils/eval_seg_3d_iou.py \\\n"
        f"      --gc6d_root {args.data_root} \\\n"
        f"      --pred_dir {args.output_dir} \\\n"
        f"      --camera {args.camera} \\\n"
        f"      --category_file {objects_json}"
    )


if __name__ == "__main__":
    main()
