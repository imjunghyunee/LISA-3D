#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# LISA-3D: Language-grounded 3D segmentation inference on GraspClutter6D
#
# Runs LISA++ (with optional geometry-aware LoRA weights) on a set of scenes,
# back-projects predicted masks into camera-frame 3D point clouds, and saves
# predictions/{scene_id}/seg_3d.npz for evaluation with the Clutt3R-Seg
# evaluation protocol.
#
# Usage:
#   ./run.sh --vision_pretrained /path/to/sam_vit_h.pth [options]
#
# Examples:
#   # Base LISA++ (no LoRA) on all test scenes, label all foreground as obj_id=1:
#   ./run.sh --vision_pretrained /path/to/sam_vit_h.pth
#
#   # With trained LoRA weights on specific scenes:
#   ./run.sh \
#       --vision_pretrained /path/to/sam_vit_h.pth \
#       --lora_weights runs/lisa3d/lora_final.pth \
#       --scene_ids 2 3 6 7 8
#
#   # Category-specific query with correct obj_id label:
#   ./run.sh \
#       --vision_pretrained /path/to/sam_vit_h.pth \
#       --lora_weights runs/lisa3d/lora_final.pth \
#       --prompt "Segment the blue cylindrical container." \
#       --label 66 \
#       --scene_ids 2 3 6
#
#   # Save mask overlays and RGBA prompt images:
#   ./run.sh \
#       --vision_pretrained /path/to/sam_vit_h.pth \
#       --lora_weights runs/lisa3d/lora_final.pth \
#       --vis_save_path ./vis_output \
#       --vis_ply
#
# Evaluation (after running inference):
#   python ../graspclutter6dAPI/utils/eval_seg_3d_iou.py \
#       --gc6d_root /home/jhwang/grasp/GraspClutter6D \
#       --pred_dir ./predictions \
#       --camera realsense-d415 \
#       --category_file /path/to/categories.json
# ──────────────────────────────────────────────────────────────────────────────
set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
VERSION="Senqiao/LISA_Plus_7b"
VISION_PRETRAINED=""
LORA_WEIGHTS=""
DATA_ROOT="/home/jhwang/grasp/GraspClutter6D"
OUTPUT_DIR="./predictions"
VIS_SAVE_PATH=""
GPU="0"
PRECISION="bf16"
N_VIEWS="8"
CAMERA="realsense-d415"
SCENE_IDS=""
LABEL="1"
PROMPT="Please segment all objects in this image."
VISION_TOWER="openai/clip-vit-large-patch14"
MAX_NEW_TOKENS="32"
VIS_PLY="false"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)           VERSION="$2";           shift 2 ;;
        --vision_pretrained) VISION_PRETRAINED="$2"; shift 2 ;;
        --lora_weights)      LORA_WEIGHTS="$2";      shift 2 ;;
        --data_root)         DATA_ROOT="$2";         shift 2 ;;
        --output_dir)        OUTPUT_DIR="$2";        shift 2 ;;
        --vis_save_path)     VIS_SAVE_PATH="$2";     shift 2 ;;
        --gpu)               GPU="$2";               shift 2 ;;
        --precision)         PRECISION="$2";         shift 2 ;;
        --n_views)           N_VIEWS="$2";           shift 2 ;;
        --camera)            CAMERA="$2";            shift 2 ;;
        --scene_ids)
            # Collect all following non-flag arguments as scene IDs
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SCENE_IDS="$SCENE_IDS $1"
                shift
            done
            ;;
        --label)             LABEL="$2";             shift 2 ;;
        --prompt)            PROMPT="$2";            shift 2 ;;
        --vision_tower)      VISION_TOWER="$2";      shift 2 ;;
        --max_new_tokens)    MAX_NEW_TOKENS="$2";    shift 2 ;;
        --vis_ply)           VIS_PLY="true";         shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z "$VISION_PRETRAINED" ]]; then
    echo "ERROR: --vision_pretrained is required (path to sam_vit_h_*.pth)"
    echo ""
    echo "Download SAM ViT-H checkpoint:"
    echo "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    exit 1
fi

if [[ ! -f "$VISION_PRETRAINED" ]]; then
    echo "ERROR: SAM checkpoint not found at: $VISION_PRETRAINED"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
CONDA_ENV="lisa"
PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Make sure the 'lisa' conda environment is installed."
    exit 1
fi

# Add both LISA-3D and LISA directories to PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LISA_DIR="$(dirname "$SCRIPT_DIR")/LISA"

# LISA must precede LISA-3D so that model.llava1p5 / model.segment_anything
# are resolved from LISA's model/ namespace package before LISA-3D's.
export PYTHONPATH="${LISA_DIR}:${SCRIPT_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CONDA_BASE}/envs/${CONDA_ENV}/lib:${CONDA_BASE}/envs/${CONDA_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

# ── Build argument string ─────────────────────────────────────────────────────
EXTRA_ARGS=""
[[ -n "$LORA_WEIGHTS"  ]] && EXTRA_ARGS="$EXTRA_ARGS --lora_weights $LORA_WEIGHTS"
[[ -n "$VIS_SAVE_PATH" ]] && EXTRA_ARGS="$EXTRA_ARGS --vis_save_path $VIS_SAVE_PATH"
[[ -n "$SCENE_IDS"     ]] && EXTRA_ARGS="$EXTRA_ARGS --scene_ids $SCENE_IDS"
[[ "$VIS_PLY" == "true" ]] && EXTRA_ARGS="$EXTRA_ARGS --vis_ply"

# ── Print configuration ───────────────────────────────────────────────────────
echo "LISA-3D Inference"
echo "─────────────────────────────────────────────────────────────────────"
echo "Model version      : $VERSION"
echo "SAM checkpoint     : $VISION_PRETRAINED"
echo "LoRA weights       : ${LORA_WEIGHTS:-none (base LISA++)}"
echo "Data root          : $DATA_ROOT"
echo "Output dir         : $OUTPUT_DIR"
echo "Camera             : $CAMERA"
echo "Views per scene    : $N_VIEWS"
echo "Prompt             : $PROMPT"
echo "Label (obj_id)     : $LABEL"
echo "GPU                : $GPU"
echo "Precision          : $PRECISION"
[[ -n "$VIS_SAVE_PATH" ]] && echo "Visualisation dir  : $VIS_SAVE_PATH"
[[ -n "$SCENE_IDS"     ]] && echo "Scene IDs          : $SCENE_IDS"
echo "─────────────────────────────────────────────────────────────────────"

# ── Run inference ─────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "${SCRIPT_DIR}/infer.py" \
    --version           "$VERSION" \
    --vision_pretrained "$VISION_PRETRAINED" \
    --data_root         "$DATA_ROOT" \
    --output_dir        "$OUTPUT_DIR" \
    --precision         "$PRECISION" \
    --n_views           "$N_VIEWS" \
    --camera            "$CAMERA" \
    --prompt            "$PROMPT" \
    --label             "$LABEL" \
    --vision_tower      "$VISION_TOWER" \
    --max_new_tokens    "$MAX_NEW_TOKENS" \
    $EXTRA_ARGS
