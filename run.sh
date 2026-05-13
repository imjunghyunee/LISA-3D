#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# LISA-3D: per-Object-Name segmentation inference on GraspClutter6D.
#
# For every Object Name in the CSV (or the subset passed via --target_names),
# queries LISA++ with "Please segment the {name} in this image.", lifts the
# resulting mask to 3D via depth back-projection, and writes
# predictions/{scene_id}/seg_3d.npz for evaluation with
# graspclutter6dAPI/utils/eval_seg_3d_iou.py.  The matching
# {output_dir}/objects.json is written alongside.
#
# Usage:
#   ./run.sh --vision_pretrained /path/to/sam_vit_h.pth                # base LISA++
#
#   ./run.sh \
#       --vision_pretrained /path/to/sam_vit_h.pth \
#       --lora_weights runs/lisa3d_B/lora_B_final.pth                  # trained run
#
#   ./run.sh \
#       --vision_pretrained /path/to/sam_vit_h.pth \
#       --lora_weights runs/lisa3d_B/lora_B_final.pth \
#       --scene_ids 2 3 6 7 --target_names "banana,apple"              # subset
#
#   ./run.sh \
#       --vision_pretrained /path/to/sam_vit_h.pth \
#       --lora_weights runs/lisa3d_B/lora_B_final.pth \
#       --vis_save_path ./vis_output --vis_ply                         # visualise
#
# Evaluation (after running inference):
#   python /home/jhwang/grasp/graspclutter6dAPI/utils/eval_seg_3d_iou.py \
#       --gc6d_root /home/jhwang/grasp/graspclutter6dAPI/GraspClutter6D \
#       --pred_dir ./predictions \
#       --camera realsense-d415 \
#       --category_file ./predictions/objects.json
# ──────────────────────────────────────────────────────────────────────────────
set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
VERSION="Senqiao/LISA_Plus_7b"
VISION_PRETRAINED=""
LORA_WEIGHTS=""
DATA_ROOT="/home/jhwang/grasp/graspclutter6dAPI/GraspClutter6D"
CSV_PATH="/home/jhwang/grasp/graspclutter6dAPI/GraspClutter6D/graspclutter6d_object_id.csv"
OUTPUT_DIR="./predictions"
VIS_SAVE_PATH=""
GPU="0"
PRECISION="bf16"
N_VIEWS="8"
CAMERA="realsense-d415"
SCENE_IDS=""
TARGET_NAMES=""
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
        --csv_path)          CSV_PATH="$2";          shift 2 ;;
        --output_dir)        OUTPUT_DIR="$2";        shift 2 ;;
        --vis_save_path)     VIS_SAVE_PATH="$2";     shift 2 ;;
        --gpu)               GPU="$2";               shift 2 ;;
        --precision)         PRECISION="$2";         shift 2 ;;
        --n_views)           N_VIEWS="$2";           shift 2 ;;
        --camera)            CAMERA="$2";            shift 2 ;;
        --scene_ids)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SCENE_IDS="$SCENE_IDS $1"
                shift
            done
            ;;
        --target_names)      TARGET_NAMES="$2";      shift 2 ;;
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

if [[ ! -f "$CSV_PATH" ]]; then
    echo "ERROR: Object-name CSV not found at: $CSV_PATH"
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
[[ -n "$TARGET_NAMES"  ]] && EXTRA_ARGS="$EXTRA_ARGS --target_names $TARGET_NAMES"
[[ "$VIS_PLY" == "true" ]] && EXTRA_ARGS="$EXTRA_ARGS --vis_ply"

# ── Print configuration ───────────────────────────────────────────────────────
echo "LISA-3D Inference (per Object Name)"
echo "─────────────────────────────────────────────────────────────────────"
echo "Model version      : $VERSION"
echo "SAM checkpoint     : $VISION_PRETRAINED"
echo "LoRA weights       : ${LORA_WEIGHTS:-none (base LISA++)}"
echo "Data root          : $DATA_ROOT"
echo "Object-name CSV    : $CSV_PATH"
echo "Output dir         : $OUTPUT_DIR"
echo "Camera             : $CAMERA"
echo "Views per scene    : $N_VIEWS"
echo "GPU                : $GPU"
echo "Precision          : $PRECISION"
[[ -n "$VIS_SAVE_PATH" ]] && echo "Visualisation dir  : $VIS_SAVE_PATH"
[[ -n "$SCENE_IDS"     ]] && echo "Scene IDs          : $SCENE_IDS"
[[ -n "$TARGET_NAMES"  ]] && echo "Target names       : $TARGET_NAMES"
echo "─────────────────────────────────────────────────────────────────────"

# ── Run inference ─────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "${SCRIPT_DIR}/infer.py" \
    --version           "$VERSION" \
    --vision_pretrained "$VISION_PRETRAINED" \
    --data_root         "$DATA_ROOT" \
    --csv_path          "$CSV_PATH" \
    --output_dir        "$OUTPUT_DIR" \
    --precision         "$PRECISION" \
    --n_views           "$N_VIEWS" \
    --camera            "$CAMERA" \
    --vision_tower      "$VISION_TOWER" \
    --max_new_tokens    "$MAX_NEW_TOKENS" \
    $EXTRA_ARGS
