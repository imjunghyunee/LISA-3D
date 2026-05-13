#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# LISA-3D Path A — LoRA-only training on GraspClutter6D (per-object names).
#
# Trains ONLY CLIP + LLaMA self-attn LoRA adapters (~11.6 M params).
# Mask decoder, text_hidden_fcs, lm_head and embed_tokens stay frozen.
# This is the lightest of the three comparison runs.
#
# Multi-GPU: set CUDA_VISIBLE_DEVICES before invoking; the script auto-detects
# the number of GPUs and launches via `torchrun --nproc_per_node=$NUM_GPUS`.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./run_train_A.sh \
#       --vision_pretrained /path/to/sam_vit_h_4b8939.pth
# ──────────────────────────────────────────────────────────────────────────────
set -e

# ── Defaults (Path A) ─────────────────────────────────────────────────────────
VERSION="Senqiao/LISA_Plus_7b"
VISION_PRETRAINED=""
DATA_ROOT="/home/jhwang/grasp/graspclutter6dAPI/GraspClutter6D"
CSV_PATH="/home/jhwang/grasp/graspclutter6dAPI/GraspClutter6D/graspclutter6d_object_id.csv"
OUTPUT_DIR="./runs/lisa3d_A"
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
UNFREEZE_MODE="A"
HEAD_LR_SCALE="0.3"     # unused in A but harmless
CAMERA="realsense-d415"
WORKERS="4"
PRINT_FREQ="50"
SAVE_FREQ="1"
RESUME=""
VISION_TOWER="openai/clip-vit-large-patch14"
TARGET_NAMES=""
MASTER_PORT="${MASTER_PORT:-29500}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)           VERSION="$2";           shift 2 ;;
        --vision_pretrained) VISION_PRETRAINED="$2"; shift 2 ;;
        --data_root)         DATA_ROOT="$2";         shift 2 ;;
        --csv_path)          CSV_PATH="$2";          shift 2 ;;
        --output_dir)        OUTPUT_DIR="$2";        shift 2 ;;
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
        --head_lr_scale)     HEAD_LR_SCALE="$2";     shift 2 ;;
        --camera)            CAMERA="$2";            shift 2 ;;
        --workers)           WORKERS="$2";           shift 2 ;;
        --print_freq)        PRINT_FREQ="$2";        shift 2 ;;
        --save_freq)         SAVE_FREQ="$2";         shift 2 ;;
        --resume)            RESUME="$2";            shift 2 ;;
        --vision_tower)      VISION_TOWER="$2";      shift 2 ;;
        --target_names)      TARGET_NAMES="$2";      shift 2 ;;
        --master_port)       MASTER_PORT="$2";       shift 2 ;;
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
if [[ ! -f "$CSV_PATH" ]]; then
    echo "ERROR: obj_id<->name CSV not found at: $CSV_PATH"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
CONDA_ENV="lisa"
PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
TORCHRUN="${CONDA_BASE}/envs/${CONDA_ENV}/bin/torchrun"

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    exit 1
fi
if [[ ! -f "$TORCHRUN" ]]; then
    # Fallback: use python -m torch.distributed.run
    TORCHRUN="$PYTHON -m torch.distributed.run"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LISA_DIR="$(dirname "$SCRIPT_DIR")/LISA"

# LISA must precede LISA-3D so that model.llava1p5 / model.segment_anything
# are resolved from LISA's model/ namespace package before LISA-3D's.
export PYTHONPATH="${LISA_DIR}:${SCRIPT_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CONDA_BASE}/envs/${CONDA_ENV}/lib:${CONDA_BASE}/envs/${CONDA_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

# Install optional packages on the launching node only
"$PYTHON" -m pip install tensorboard tqdm --quiet 2>/dev/null || true

# ── GPU auto-detection ────────────────────────────────────────────────────────
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
    NUM_GPUS=$("$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

# ── Build extra args ──────────────────────────────────────────────────────────
EXTRA_ARGS=""
[[ -n "$RESUME"        ]] && EXTRA_ARGS="$EXTRA_ARGS --resume $RESUME"
[[ -n "$TARGET_NAMES"  ]] && EXTRA_ARGS="$EXTRA_ARGS --target_names $TARGET_NAMES"

# ── Print configuration ───────────────────────────────────────────────────────
echo "LISA-3D Path A — LoRA-only training"
echo "─────────────────────────────────────────────────────────────────────"
echo "GPUs                : $CUDA_VISIBLE_DEVICES  (nproc_per_node=$NUM_GPUS)"
echo "Model version       : $VERSION"
echo "SAM checkpoint      : $VISION_PRETRAINED"
echo "Data root           : $DATA_ROOT"
echo "Object-name CSV     : $CSV_PATH"
echo "Output dir          : $OUTPUT_DIR"
echo "Precision           : $PRECISION"
echo "Unfreeze mode       : $UNFREEZE_MODE (LoRA only)"
echo "Epochs              : $EPOCHS"
echo "Batch size          : $BATCH_SIZE (grad accum: $GRAD_ACCUM)"
echo "Learning rate       : $LR  (head_lr_scale: $HEAD_LR_SCALE)"
echo "Geo lambda          : $GEO_LAMBDA"
echo "LoRA                : r=$LORA_R  α=$LORA_ALPHA  dropout=$LORA_DROPOUT"
echo "Camera              : $CAMERA"
[[ -n "$RESUME"        ]] && echo "Resuming from       : $RESUME"
[[ -n "$TARGET_NAMES"  ]] && echo "Target names        : $TARGET_NAMES"
echo "─────────────────────────────────────────────────────────────────────"

# ── Run training (torchrun handles DDP env setup) ─────────────────────────────
$TORCHRUN \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    "${SCRIPT_DIR}/train.py" \
    --version           "$VERSION" \
    --vision_pretrained "$VISION_PRETRAINED" \
    --data_root         "$DATA_ROOT" \
    --csv_path          "$CSV_PATH" \
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
    --unfreeze_mode     "$UNFREEZE_MODE" \
    --head_lr_scale     "$HEAD_LR_SCALE" \
    --camera            "$CAMERA" \
    --workers           "$WORKERS" \
    --print_freq        "$PRINT_FREQ" \
    --save_freq         "$SAVE_FREQ" \
    --vision_tower      "$VISION_TOWER" \
    $EXTRA_ARGS
