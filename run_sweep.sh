#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# LISA-3D Optuna hyperparameter sweep launcher.
#
# Per-path budget defaults (~1 day on 6× RTX 8000, bf16, MedianPruner):
#   A   : 30 trials × 2 epochs × max_anchors 2000   (~12 h)
#   B   : 25 trials × 2 epochs × max_anchors 2000   (~14 h)
#   B+  : 20 trials × 2 epochs × max_anchors 1500   (~14 h)
# All override-able via the matching --flag.
#
# Each sweep runs trials sequentially on the visible GPUs (no inter-trial
# parallelism by default; see plan).  Run separate studies for separate
# unfreeze_modes — A / B / B+ are study-level axes, not trial-level.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./run_sweep.sh \
#       --study_name lisa3d_A_v1 --unfreeze_mode A \
#       --vision_pretrained /path/to/sam_vit_h_4b8939.pth
# ──────────────────────────────────────────────────────────────────────────────
set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
STUDY_NAME=""
UNFREEZE_MODE=""
VISION_PRETRAINED=""
DATA_ROOT="/home/jhwang/grasp/graspclutter6dAPI/GraspClutter6D"
CSV_PATH="/home/jhwang/grasp/graspclutter6dAPI/GraspClutter6D/graspclutter6d_object_id.csv"
TRAIN_JSON="${DATA_ROOT}/split_info/grasp_train_scene_ids.json"
SWEEP_DIR=""                # filled in after STUDY_NAME is known
N_TRIALS=""
SWEEP_EPOCHS=""
MAX_ANCHORS=""
N_VAL_SCENES="40"
VAL_SPLIT_SEED="1337"
SEARCH_SPACE="core"
PRECISION="bf16"
TARGET_NAMES=""
POLL_SECS="30"
MASTER_PORT_BASE="29600"
PRUNER_WARMUP="1"
SEED="42"
VERSION="Senqiao/LISA_Plus_7b"
VISION_TOWER="openai/clip-vit-large-patch14"
CAMERA="realsense-d415"
WORKERS="4"
GEO_LAMBDA="0.4"
FORCE_SPLIT=""              # set to --force to overwrite existing split

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --study_name)         STUDY_NAME="$2";        shift 2 ;;
        --unfreeze_mode)      UNFREEZE_MODE="$2";     shift 2 ;;
        --vision_pretrained)  VISION_PRETRAINED="$2"; shift 2 ;;
        --data_root)          DATA_ROOT="$2";         shift 2 ;;
        --csv_path)           CSV_PATH="$2";          shift 2 ;;
        --train_json)         TRAIN_JSON="$2";        shift 2 ;;
        --sweep_dir)          SWEEP_DIR="$2";         shift 2 ;;
        --n_trials)           N_TRIALS="$2";          shift 2 ;;
        --sweep_epochs)       SWEEP_EPOCHS="$2";      shift 2 ;;
        --max_anchors)        MAX_ANCHORS="$2";       shift 2 ;;
        --n_val_scenes)       N_VAL_SCENES="$2";      shift 2 ;;
        --val_split_seed)     VAL_SPLIT_SEED="$2";    shift 2 ;;
        --search_space)       SEARCH_SPACE="$2";      shift 2 ;;
        --precision)          PRECISION="$2";         shift 2 ;;
        --target_names)       TARGET_NAMES="$2";      shift 2 ;;
        --poll_secs)          POLL_SECS="$2";         shift 2 ;;
        --master_port_base)   MASTER_PORT_BASE="$2";  shift 2 ;;
        --pruner_warmup)      PRUNER_WARMUP="$2";     shift 2 ;;
        --seed)               SEED="$2";              shift 2 ;;
        --version)            VERSION="$2";           shift 2 ;;
        --vision_tower)       VISION_TOWER="$2";      shift 2 ;;
        --camera)             CAMERA="$2";            shift 2 ;;
        --workers)            WORKERS="$2";           shift 2 ;;
        --geo_lambda)         GEO_LAMBDA="$2";        shift 2 ;;
        --force_split)        FORCE_SPLIT="--force";  shift 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z "$STUDY_NAME" ]]; then
    echo "ERROR: --study_name is required"; exit 1
fi
if [[ -z "$UNFREEZE_MODE" ]]; then
    echo "ERROR: --unfreeze_mode is required (A | B | B+)"; exit 1
fi
case "$UNFREEZE_MODE" in
    A|B|B+) ;;
    *) echo "ERROR: --unfreeze_mode must be one of A | B | B+"; exit 1 ;;
esac
if [[ -z "$VISION_PRETRAINED" ]]; then
    echo "ERROR: --vision_pretrained is required (path to sam_vit_h_*.pth)"
    exit 1
fi
if [[ ! -f "$VISION_PRETRAINED" ]]; then
    echo "ERROR: SAM checkpoint not found at: $VISION_PRETRAINED"; exit 1
fi
if [[ ! -f "$CSV_PATH" ]]; then
    echo "ERROR: obj_id<->name CSV not found at: $CSV_PATH"; exit 1
fi
if [[ ! -f "$TRAIN_JSON" ]]; then
    echo "ERROR: train scene_id JSON not found at: $TRAIN_JSON"; exit 1
fi

# ── Per-mode budget defaults (only fill if user didn't override) ──────────────
case "$UNFREEZE_MODE" in
    A)
        N_TRIALS="${N_TRIALS:-30}"
        SWEEP_EPOCHS="${SWEEP_EPOCHS:-2}"
        MAX_ANCHORS="${MAX_ANCHORS:-2000}"
        ;;
    B)
        N_TRIALS="${N_TRIALS:-25}"
        SWEEP_EPOCHS="${SWEEP_EPOCHS:-2}"
        MAX_ANCHORS="${MAX_ANCHORS:-2000}"
        ;;
    B+)
        N_TRIALS="${N_TRIALS:-20}"
        SWEEP_EPOCHS="${SWEEP_EPOCHS:-2}"
        MAX_ANCHORS="${MAX_ANCHORS:-1500}"
        ;;
esac

# ── Environment ───────────────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
CONDA_ENV="lisa"
PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
TORCHRUN="${CONDA_BASE}/envs/${CONDA_ENV}/bin/torchrun"

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"; exit 1
fi
if [[ ! -f "$TORCHRUN" ]]; then
    # Fallback: use python -m torch.distributed.run
    TORCHRUN="$PYTHON -m torch.distributed.run"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LISA_DIR="$(dirname "$SCRIPT_DIR")/LISA"

# LISA must precede LISA-3D so model.llava1p5 / model.segment_anything resolve
# from LISA's model/ namespace package before LISA-3D's.
export PYTHONPATH="${LISA_DIR}:${SCRIPT_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CONDA_BASE}/envs/${CONDA_ENV}/lib:${CONDA_BASE}/envs/${CONDA_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
export TORCHRUN  # consumed by tune.py:_resolve_torchrun if set

# Install optional packages on the launching node only
"$PYTHON" -m pip install optuna tensorboard tqdm --quiet 2>/dev/null || true

# ── GPU auto-detection ────────────────────────────────────────────────────────
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
    NUM_GPUS=$("$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

# ── Resolve sweep_dir ─────────────────────────────────────────────────────────
SWEEP_DIR="${SWEEP_DIR:-${SCRIPT_DIR}/runs/sweep/${STUDY_NAME}}"
mkdir -p "$SWEEP_DIR"
TRAIN_IDS_OUT="${SWEEP_DIR}/train_ids.json"
VAL_IDS_OUT="${SWEEP_DIR}/val_ids.json"

# ── 1) Build train/val split (once per study) ─────────────────────────────────
"$PYTHON" -m utils.val_split \
    --in_path        "$TRAIN_JSON" \
    --out_train_path "$TRAIN_IDS_OUT" \
    --out_val_path   "$VAL_IDS_OUT" \
    --n_val          "$N_VAL_SCENES" \
    --seed           "$VAL_SPLIT_SEED" \
    $FORCE_SPLIT

# ── Print configuration ───────────────────────────────────────────────────────
echo "─────────────────────────────────────────────────────────────────────"
echo "LISA-3D Optuna sweep"
echo "─────────────────────────────────────────────────────────────────────"
echo "Study               : $STUDY_NAME"
echo "Unfreeze mode       : $UNFREEZE_MODE"
echo "Search space        : $SEARCH_SPACE"
echo "GPUs                : $CUDA_VISIBLE_DEVICES  (nproc_per_node=$NUM_GPUS)"
echo "Sweep dir           : $SWEEP_DIR"
echo "Train/val split     : seed=$VAL_SPLIT_SEED  n_val=$N_VAL_SCENES"
echo "  train_ids.json    : $TRAIN_IDS_OUT"
echo "  val_ids.json      : $VAL_IDS_OUT"
echo "Trials              : $N_TRIALS"
echo "Sweep epochs        : $SWEEP_EPOCHS"
echo "Max anchors / epoch : $MAX_ANCHORS"
echo "Precision           : $PRECISION"
echo "geo_lambda (fixed)  : $GEO_LAMBDA"
echo "Pruner warmup       : $PRUNER_WARMUP"
echo "Poll seconds        : $POLL_SECS"
echo "Master port base    : $MASTER_PORT_BASE"
[[ -n "$TARGET_NAMES" ]] && echo "Target names        : $TARGET_NAMES"
echo "─────────────────────────────────────────────────────────────────────"

# ── 2) Launch the Optuna driver ───────────────────────────────────────────────
EXTRA_TUNE_ARGS=""
[[ -n "$TARGET_NAMES" ]] && EXTRA_TUNE_ARGS="$EXTRA_TUNE_ARGS --target_names $TARGET_NAMES"

"$PYTHON" "${SCRIPT_DIR}/tune.py" \
    --study_name             "$STUDY_NAME" \
    --sweep_dir              "$SWEEP_DIR" \
    --n_trials               "$N_TRIALS" \
    --unfreeze_mode          "$UNFREEZE_MODE" \
    --sweep_epochs           "$SWEEP_EPOCHS" \
    --max_anchors            "$MAX_ANCHORS" \
    --search_space           "$SEARCH_SPACE" \
    --vision_pretrained      "$VISION_PRETRAINED" \
    --data_root              "$DATA_ROOT" \
    --csv_path               "$CSV_PATH" \
    --train_scene_ids_path   "$TRAIN_IDS_OUT" \
    --val_scene_ids_path     "$VAL_IDS_OUT" \
    --version                "$VERSION" \
    --vision_tower           "$VISION_TOWER" \
    --precision              "$PRECISION" \
    --camera                 "$CAMERA" \
    --workers                "$WORKERS" \
    --geo_lambda             "$GEO_LAMBDA" \
    --poll_secs              "$POLL_SECS" \
    --master_port_base       "$MASTER_PORT_BASE" \
    --pruner_warmup          "$PRUNER_WARMUP" \
    --seed                   "$SEED" \
    $EXTRA_TUNE_ARGS
