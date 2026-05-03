#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# LISA-3D Stage-1 training: geometry-aware LoRA fine-tuning
#
# Injects LoRA adapters (r=16, α=32) into CLIP + LLaMA attention layers and
# trains them with a two-view multi-view consistency loss on GraspClutter6D.
# Only LoRA parameters (~11.6M) are updated; everything else stays frozen.
#
# Usage:
#   ./run_train.sh --vision_pretrained /path/to/sam_vit_h.pth [options]
#
# Examples:
#   # Basic training with defaults:
#   ./run_train.sh --vision_pretrained /path/to/sam_vit_h.pth
#
#   # Custom settings:
#   ./run_train.sh \
#       --vision_pretrained /path/to/sam_vit_h.pth \
#       --output_dir ./runs/lisa3d_v2 \
#       --epochs 15 \
#       --lr 2e-4 \
#       --gpu 0
#
#   # Resume from checkpoint:
#   ./run_train.sh \
#       --vision_pretrained /path/to/sam_vit_h.pth \
#       --resume runs/lisa3d/lora_epoch005.pth
# ──────────────────────────────────────────────────────────────────────────────
set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
VERSION="Senqiao/LISA_Plus_7b"
VISION_PRETRAINED=""
DATA_ROOT="/home/jhwang/grasp/GraspClutter6D"
OUTPUT_DIR="./runs/lisa3d"
GPU="0"
PRECISION="bf16"
EPOCHS="10"
BATCH_SIZE="1"
GRAD_ACCUM="2"
LR="3e-4"
WEIGHT_DECAY="0.05"
GEO_LAMBDA="0.4"
LORA_R="16"
LORA_ALPHA="32.0"
LORA_DROPOUT="0.05"
CAMERA="realsense-d415"
WORKERS="4"
PRINT_FREQ="50"
SAVE_FREQ="1"
RESUME=""
VISION_TOWER="openai/clip-vit-large-patch14"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)           VERSION="$2";           shift 2 ;;
        --vision_pretrained) VISION_PRETRAINED="$2"; shift 2 ;;
        --data_root)         DATA_ROOT="$2";         shift 2 ;;
        --output_dir)        OUTPUT_DIR="$2";        shift 2 ;;
        --gpu)               GPU="$2";               shift 2 ;;
        --precision)         PRECISION="$2";         shift 2 ;;
        --epochs)            EPOCHS="$2";            shift 2 ;;
        --batch_size)        BATCH_SIZE="$2";        shift 2 ;;
        --grad_accum)        GRAD_ACCUM="$2";        shift 2 ;;
        --lr)                LR="$2";                shift 2 ;;
        --weight_decay)      WEIGHT_DECAY="$2";      shift 2 ;;
        --geo_lambda)        GEO_LAMBDA="$2";        shift 2 ;;
        --lora_r)            LORA_R="$2";            shift 2 ;;
        --lora_alpha)        LORA_ALPHA="$2";        shift 2 ;;
        --lora_dropout)      LORA_DROPOUT="$2";      shift 2 ;;
        --camera)            CAMERA="$2";            shift 2 ;;
        --workers)           WORKERS="$2";           shift 2 ;;
        --print_freq)        PRINT_FREQ="$2";        shift 2 ;;
        --save_freq)         SAVE_FREQ="$2";         shift 2 ;;
        --resume)            RESUME="$2";            shift 2 ;;
        --vision_tower)      VISION_TOWER="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z "$VISION_PRETRAINED" ]]; then
    echo "ERROR: --vision_pretrained is required (path to sam_vit_h_*.pth)"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LISA_DIR="$(dirname "$SCRIPT_DIR")/LISA"

# LISA must precede LISA-3D so that model.llava1p5 / model.segment_anything
# are resolved from LISA's model/ namespace package before LISA-3D's.
export PYTHONPATH="${LISA_DIR}:${SCRIPT_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CONDA_BASE}/envs/${CONDA_ENV}/lib:${CONDA_BASE}/envs/${CONDA_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

# Install optional tensorboard if not present
"$PYTHON" -m pip install tensorboard tqdm --quiet 2>/dev/null || true

# ── Build argument string ─────────────────────────────────────────────────────
EXTRA_ARGS=""
[[ -n "$RESUME" ]] && EXTRA_ARGS="$EXTRA_ARGS --resume $RESUME"

# ── Print configuration ───────────────────────────────────────────────────────
echo "LISA-3D Stage-1 Training"
echo "─────────────────────────────────────────────────────────────────────"
echo "Model version      : $VERSION"
echo "SAM checkpoint     : $VISION_PRETRAINED"
echo "Data root          : $DATA_ROOT"
echo "Output dir         : $OUTPUT_DIR"
echo "GPU                : $GPU"
echo "Precision          : $PRECISION"
echo "Epochs             : $EPOCHS"
echo "Batch size         : $BATCH_SIZE (grad accum: $GRAD_ACCUM)"
echo "Learning rate      : $LR  (weight decay: $WEIGHT_DECAY)"
echo "Geo lambda         : $GEO_LAMBDA"
echo "LoRA               : r=$LORA_R  α=$LORA_ALPHA  dropout=$LORA_DROPOUT"
echo "Camera             : $CAMERA"
[[ -n "$RESUME"    ]] && echo "Resuming from      : $RESUME"
echo "─────────────────────────────────────────────────────────────────────"

# ── Run training ──────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "${SCRIPT_DIR}/train.py" \
    --version           "$VERSION" \
    --vision_pretrained "$VISION_PRETRAINED" \
    --data_root         "$DATA_ROOT" \
    --output_dir        "$OUTPUT_DIR" \
    --precision         "$PRECISION" \
    --epochs            "$EPOCHS" \
    --batch_size        "$BATCH_SIZE" \
    --grad_accum_steps  "$GRAD_ACCUM" \
    --lr                "$LR" \
    --weight_decay      "$WEIGHT_DECAY" \
    --geo_lambda        "$GEO_LAMBDA" \
    --lora_r            "$LORA_R" \
    --lora_alpha        "$LORA_ALPHA" \
    --lora_dropout      "$LORA_DROPOUT" \
    --camera            "$CAMERA" \
    --workers           "$WORKERS" \
    --print_freq        "$PRINT_FREQ" \
    --save_freq         "$SAVE_FREQ" \
    --vision_tower      "$VISION_TOWER" \
    $EXTRA_ARGS
