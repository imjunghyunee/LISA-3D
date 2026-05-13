# LISA-3D: Lifting Language-Image Segmentation to 3D via Multi-View Consistency

This repository contains a reference implementation of **LISA-3D** (arXiv: 2512.01008) adapted to the [GraspClutter6D](https://github.com/SeungyeonKim/GraspClutter6D) RGB-D benchmark.

LISA-3D retrofits the instruction-following segmentation model [LISA](https://github.com/dvlab-research/LISA) with **geometry-aware Low-Rank Adaptation (LoRA)** layers and trains them with a **differentiable multi-view reprojection consistency loss**. Only ~11.6 M LoRA parameters are updated; the LISA backbone and the downstream 3D module remain frozen.

The paper's Stage 2 reconstruction module is **SAM-3D** (not publicly released). In this implementation the predicted 2D masks are instead **lifted to camera-frame 3D point clouds via depth back-projection** so predictions can be evaluated with the Clutt3R-Seg `eval_seg_3d_iou.py` protocol.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Stage 1 (training)                                │
│  (I_a, T) ──► Φ_θ (LISA + LoRA) ──► P_a ─┐                               │
│                                          ├─► L_seg (BCE + Dice)          │
│  (I_b, T) ──► Φ_θ (LISA + LoRA) ──► P_b ─┘                               │
│       │                                                                  │
│       │  D_a, K_a, E_a, K_b, E_b                                         │
│       └─► W(P_a) = P̃_{a→b}  ──► L_geo = || P_b − sg(P̃_{a→b}) ||_1 + …  │
│                                                                          │
│  L_total = L_seg_a + L_seg_b + λ · L_geo       (λ = 0.4)                 │
└──────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────┐
│                        Stage 2 (inference)                               │
│  (I, T) ──► Φ_θ ──► M ──► unproject_depth(M, D, K) ──► points + labels   │
│                                                          (camera frame)  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Installation](#installation)
3. [Data preparation](#data-preparation)
4. [Training](#training)
5. [Inference](#inference)
6. [Evaluation](#evaluation)
7. [Visualisation](#visualisation)
8. [Citation](#citation)
9. [Acknowledgements](#acknowledgements)

---

## Repository layout

```
LISA-3D/
├── model/
│   ├── LISA3D.py         # LISA3DForCausalLM, LoRA injection, multi-view forward
│   └── warping.py        # Differentiable depth-based reprojection W
├── utils/
│   ├── dataset.py        # GraspClutter6D pair (train) and per-frame (infer) datasets
│   ├── lifting.py        # Mask back-projection, RGBA prompt, .npz / .ply writers
│   └── utils.py          # Constants, losses, camera helpers, checkpoint I/O
├── train.py              # Stage-1 LoRA training entry point
├── infer.py              # Stage-2 inference entry point
├── run_train.sh          # Conda-aware training launcher
├── run.sh                # Conda-aware inference launcher
├── lisa.yaml             # Conda environment specification
└── requirements_extra.txt
```

## Installation

### 1. Clone this repository **and its sibling LISA repository**

The LISA-3D code re-uses LISA's `model.LISA`, `model.llava1p5`, `model.segment_anything`, and `utils.utils` modules. They must be importable, so the original LISA repository must live next to this one:

```bash
git clone https://github.com/dvlab-research/LISA.git
git clone https://github.com/imjunghyunee/LISA-3D.git
```

`run.sh` / `run_train.sh` automatically prepend `../LISA` to `PYTHONPATH`. If you place the repos elsewhere, edit `LISA_DIR` near the top of those scripts.

### 2. Create the conda environment

```bash
cd LISA-3D
conda env create -f lisa.yaml      # creates the 'lisa' environment
conda activate lisa
pip install -r requirements_extra.txt   # tensorboard, tqdm (optional)
```

The environment pins **Python 3.9**, **PyTorch 1.13.1 + CUDA 11.7**, **transformers 4.31.0**, and **bitsandbytes 0.41.1** to remain binary-compatible with LISA++. Newer PyTorch/transformers versions are known to break the LISA model loader.

### 3. Download the SAM ViT-H checkpoint

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

You can place it anywhere; pass the path via `--vision_pretrained` in every command below.

### 4. (Optional) Open3D for `.ply` visualisation

Not required — the lifting module writes ASCII PLY with plain numpy. Install only if you want to view point clouds with Open3D:

```bash
pip install open3d
```

## Data preparation

This implementation trains and evaluates on **GraspClutter6D** in its native BOP-style directory layout:

```
<DATA_ROOT>/
├── scenes/
│   └── {scene_id:06d}/
│       ├── rgb/{img_id:06d}.png
│       ├── depth/{img_id:06d}.png        # uint16, mm (RealSense)  or  scaled (Kinect/Zivid)
│       ├── mask_visib/{img_id:06d}_{obj_idx:06d}.png
│       ├── scene_camera.json             # per-frame K and world→camera (R, t) in mm
│       ├── scene_gt.json                 # per-frame list of {obj_id, cam_R_m2c, cam_t_m2c}
│       └── scene_gt_info.json            # per-object visibility statistics
└── split_info/
    ├── grasp_train_scene_ids.json
    └── grasp_test_scene_ids.json
```

GraspClutter6D interleaves 4 cameras into a single frame index: `img_id = ann_id * 4 + cam_offset` with offsets `realsense-d415=1`, `realsense-d435=2`, `azure-kinect=3`, `zivid=4`. Frames where `img_id < 1` are metadata; the loaders skip them automatically.

Set `DATA_ROOT` once and pass it everywhere via `--data_root`. The defaults in both shell launchers expect `/home/jhwang/grasp/graspclutter6dAPI/GraspClutter6D` — edit them or override with the flag.

## Training

Stage-1 trains only the LoRA adapters (~11.6 M params); LISA's CLIP encoder, LLaMA decoder, SAM image encoder, and SAM mask decoder all remain frozen.

### Quick start

```bash
./run_train.sh \
    --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
    --data_root         /path/to/GraspClutter6D \
    --output_dir        ./runs/lisa3d
```

This launches `train.py` with the paper's defaults: AdamW (`lr=3e-4`, `wd=0.05`, `β=(0.9, 0.95)`), cosine LR schedule, batch size 1 with grad accumulation 2, 10 epochs, LoRA `r=16` / `α=32` / `dropout=0.05`, geometric loss weight `λ=0.4`, bf16 precision.

### Common overrides

```bash
./run_train.sh \
    --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
    --data_root         /path/to/GraspClutter6D \
    --output_dir        ./runs/lisa3d_long \
    --epochs            15 \
    --lr                2e-4 \
    --geo_lambda        0.5 \
    --lora_r            16 \
    --lora_alpha        32 \
    --camera            realsense-d415 \
    --gpu               0
```

### Resuming a run

```bash
./run_train.sh \
    --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
    --resume            runs/lisa3d/lora_epoch005.pth
```

Checkpoints are written every `--save_freq` epochs as `lora_epoch{NNN}.pth` plus a final `lora_final.pth`. They contain only the LoRA `A`/`B` weights and the optimiser state — typically a few tens of MB.

### Hardware notes

- A single 48 GB GPU (RTX 6000 Ada / A6000 / L40) is sufficient for `batch_size=1, grad_accum=2` at 1024×1024 resolution with gradient checkpointing enabled.
- For 24 GB cards, add `--load_in_8bit` (slower) or reduce `--image_size 768`. The geometric loss requires `depth` and intrinsics at the network's output resolution; `model_forward_3d` automatically nearest-neighbour-resizes depth to match — no other changes needed.
- TensorBoard logs are written to `<output_dir>/tb/` whenever `tensorboard` is importable.

## Inference

Stage-2 takes a (possibly LoRA-fine-tuned) LISA model, predicts a binary mask per evaluation view, back-projects masked depth pixels into camera-frame 3D, and writes one `seg_3d.npz` per scene.

### With trained LoRA weights

```bash
./run.sh \
    --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
    --lora_weights      runs/lisa3d/lora_final.pth \
    --data_root         /path/to/GraspClutter6D \
    --output_dir        ./predictions
```

### Without LoRA (vanilla LISA++ baseline)

```bash
./run.sh \
    --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
    --data_root         /path/to/GraspClutter6D \
    --output_dir        ./predictions_baseline
```

### Targeted inference

```bash
./run.sh \
    --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
    --lora_weights      runs/lisa3d/lora_final.pth \
    --scene_ids         2 3 6 7 8 \
    --prompt            "Segment the blue cylindrical container." \
    --label             66
```

- `--scene_ids` accepts a space-separated list; omit it to process every scene in `grasp_test_scene_ids.json`.
- `--prompt` is the natural-language instruction passed to LISA. The default segments *all* foreground objects.
- `--label` is the integer `obj_id` written into every foreground point in the output. Use `1` for class-agnostic predictions or the category ID when issuing a category-specific prompt.
- `--n_views` controls how many of the 13 per-camera annotation frames are used (default 8, matching the Clutt3R-Seg evaluation protocol).

### Output format

```
predictions/
└── {scene_id:06d}/
    └── seg_3d.npz       # {"points": (N, 3) float32 metres,
                         #  "labels": (N,)   int32   obj_id per point}
```

## Evaluation

3D-IoU evaluation lives in the sibling `graspclutter6dAPI` repository:

```bash
python ../graspclutter6dAPI/utils/eval_seg_3d_iou.py \
    --gc6d_root     /path/to/GraspClutter6D \
    --pred_dir      ./predictions \
    --camera        realsense-d415 \
    --category_file /path/to/categories.json
```

The evaluator builds ground-truth point clouds by sampling object meshes; the predictions written by `infer.py` are already in the camera frame and units (metres) it expects.

## Visualisation

To save per-frame mask overlays, RGBA prompt PNGs, and (optionally) lifted point clouds:

```bash
./run.sh \
    --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
    --lora_weights      runs/lisa3d/lora_final.pth \
    --vis_save_path     ./vis_output \
    --vis_ply
```

This adds the following alongside the regular `.npz` predictions:

```
vis_output/
└── {scene_id:06d}/
    ├── {img_id:06d}_overlay.jpg   # RGB image with mask shaded red
    ├── {img_id:06d}_rgba.png      # I^prompt = [I, M] (RGB + alpha=mask*255)
    └── seg_3d.ply                 # ASCII point cloud (open3d not required)
```

## Citation

If you use this code, please cite the original LISA-3D paper:

```bibtex
@article{guo2025lisa3d,
  title   = {LISA-3D: Lifting Language-Image Segmentation to 3D via Multi-View Consistency},
  author  = {Guo, Zhongbin and Liu, Jiahe and Gao, Wenyu and Li, Yushan and Li, Chengzhi and Jian, Ping},
  journal = {arXiv preprint arXiv:2512.01008},
  year    = {2025}
}
```

## Acknowledgements

This implementation builds directly on:

- **[LISA](https://github.com/dvlab-research/LISA)** — the base reasoning segmentation model and its `model.llava1p5` / `model.segment_anything` modules.
- **[Segment Anything](https://github.com/facebookresearch/segment-anything)** — SAM ViT-H image encoder and mask decoder.
- **[LLaVA](https://github.com/haotian-liu/LLaVA)** — vision-language conversation framework that LISA inherits from.
- **[GraspClutter6D](https://github.com/SeungyeonKim/GraspClutter6D)** — RGB-D scenes, camera calibration, visibility annotations.
- **LISA-3D (arXiv 2512.01008)** — the methodology this repository implements.
