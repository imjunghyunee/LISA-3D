#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# DEPRECATED — the old single-script launcher.
#
# Training granularity changed to per-Object-Name with three trainable
# parameter modes (A / B / B+).  Use the dedicated launchers instead:
#
#   ./run_train_A.sh        # LoRA only         (paper default)
#   ./run_train_B.sh        # LoRA + heads
#   ./run_train_Bplus.sh    # LoRA + heads + lm_head + embed_tokens
#
# This stub forwards to run_train_A.sh for backwards compatibility, since
# A is the closest match to the old behaviour (LoRA-only fine-tuning).
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[run_train.sh] DEPRECATED → forwarding to run_train_A.sh"
echo "[run_train.sh] Use run_train_{A,B,Bplus}.sh directly for new training runs."
exec "${SCRIPT_DIR}/run_train_A.sh" "$@"
