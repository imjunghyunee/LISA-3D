"""
LISA-3D Optuna-driven hyperparameter sweep.

Each trial = one full ``torchrun ... train.py`` subprocess with HP overrides
applied as CLI flags.  Trials are isolated in per-trial output directories
under ``runs/sweep/<study_name>/trial_<NNN>/`` so checkpoints, CSV logs,
and ``trial_result.json`` don't collide.

Driver ↔ trial communication
  • Forward (trial → driver) : the trial's rank-0 appends to
    ``loss_epoch.csv``; the driver polls it every ``--poll_secs`` and
    relays new (epoch, val_loss) rows to Optuna via ``trial.report``.
  • Backward (driver → trial) : when ``trial.should_prune()`` fires the
    driver creates an empty ``.prune`` file inside the trial directory.
    ``train.py`` checks for it at every epoch boundary (rank-0 reads,
    broadcasts over NCCL) and bails out with ``status="pruned"`` in its
    ``trial_result.json``.

Storage : ``sqlite:///<sweep_dir>/optuna.db`` — local, resumable
(``load_if_exists=True``), and writable from multiple driver processes
on disjoint CUDA_VISIBLE_DEVICES if you want to parallelise studies.

See ``run_sweep.sh`` for the shell entry point and the documented
defaults per unfreeze_mode.
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


# ── Argument parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Study identity
    p.add_argument("--study_name",       required=True, type=str)
    p.add_argument("--sweep_dir",        required=True, type=str,
                   help="Directory holding optuna.db and trial_NNN/ subdirs.")
    p.add_argument("--n_trials",         default=30,    type=int)

    # Trial-time training config
    p.add_argument("--unfreeze_mode",    required=True,
                   choices=["A", "B", "B+"])
    p.add_argument("--sweep_epochs",     default=2,     type=int,
                   help="Per-trial epoch budget (reduced vs production 10).")
    p.add_argument("--max_anchors",      default=2000,  type=int,
                   help="Forwarded as --max_anchors_per_epoch to train.py.")
    p.add_argument("--search_space",     default="core",
                   choices=["fast", "core"])

    # Required passthroughs for train.py
    p.add_argument("--vision_pretrained", required=True, type=str)
    p.add_argument("--data_root",         required=True, type=str)
    p.add_argument("--csv_path",          required=True, type=str)
    p.add_argument("--train_scene_ids_path", required=True, type=str)
    p.add_argument("--val_scene_ids_path",   required=True, type=str)

    # Optional passthroughs
    p.add_argument("--version",      default="Senqiao/LISA_Plus_7b")
    p.add_argument("--vision_tower", default="openai/clip-vit-large-patch14")
    p.add_argument("--precision",    default="bf16",
                   choices=["fp32", "bf16", "fp16"])
    p.add_argument("--camera",       default="realsense-d415")
    p.add_argument("--workers",      default=4,  type=int)
    p.add_argument("--target_names", default="", type=str)
    p.add_argument("--geo_lambda",   default=0.4, type=float,
                   help="Fixed across trials (sweep objective requires "
                        "comparable units; see plan).")

    # Driver mechanics
    p.add_argument("--poll_secs",        default=30,    type=int)
    p.add_argument("--master_port_base", default=29600, type=int)
    p.add_argument("--pruner_warmup",    default=1,     type=int,
                   help="MedianPruner n_warmup_steps; report from epoch 1 "
                        "but only prune from this epoch onward.")
    p.add_argument("--seed",             default=42,    type=int)
    return p.parse_args()


# ── HP sampling ───────────────────────────────────────────────────────────

def sample_hps(trial: optuna.Trial, preset: str) -> Dict[str, str]:
    """Return CLI-ready string-valued HP overrides for one trial."""
    hp: Dict[str, str] = {}

    # Common to both presets
    hp["lr"] = f"{trial.suggest_float('lr', 5e-5, 1e-3, log=True):.6e}"
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
    hp["lora_r"]       = str(lora_r)
    hp["lora_dropout"] = f"{trial.suggest_float('lora_dropout', 0.0, 0.2):.4f}"
    hp["weight_decay"] = f"{trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True):.6e}"

    if preset == "fast":
        # α tied to r (= 2r) — keeps adapter scaling at the canonical ratio.
        hp["lora_alpha"] = f"{2.0 * lora_r:.1f}"
    else:  # core
        hp["lora_alpha"] = str(
            trial.suggest_categorical("lora_alpha", [8, 16, 32, 64, 128])
        )
        hp["head_lr_scale"] = (
            f"{trial.suggest_float('head_lr_scale', 0.1, 1.0):.4f}"
        )
        hp["grad_accum_steps"] = str(
            trial.suggest_categorical("grad_accum_steps", [1, 2, 4])
        )
    return hp


# ── Trial subprocess builder ──────────────────────────────────────────────

def _resolve_torchrun() -> List[str]:
    """Pick torchrun binary if present, else fall back to module form."""
    for candidate in (
        os.environ.get("TORCHRUN"),
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin", "torchrun"),
        "torchrun",
    ):
        if candidate and (os.path.isfile(candidate) or candidate == "torchrun"):
            return [candidate]
    return [sys.executable, "-m", "torch.distributed.run"]


def _num_gpus() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return max(1, len([x for x in cvd.split(",") if x.strip() != ""]))
    return 1


def build_trial_cmd(args, trial_dir: Path, hp: Dict[str, str],
                    master_port: int) -> List[str]:
    """Compose the torchrun + train.py invocation for one trial."""
    script_dir = Path(__file__).resolve().parent
    train_py = str(script_dir / "train.py")

    cmd: List[str] = _resolve_torchrun() + [
        "--standalone",
        f"--nproc_per_node={_num_gpus()}",
        f"--master_port={master_port}",
        train_py,
        "--version",            args.version,
        "--vision_pretrained",  args.vision_pretrained,
        "--data_root",          args.data_root,
        "--csv_path",           args.csv_path,
        "--train_scene_ids_path", args.train_scene_ids_path,
        "--val_scene_ids_path",   args.val_scene_ids_path,
        "--output_dir",         str(trial_dir),
        "--precision",          args.precision,
        "--epochs",             str(args.sweep_epochs),
        "--batch_size",         "1",
        "--unfreeze_mode",      args.unfreeze_mode,
        "--camera",             args.camera,
        "--workers",            str(args.workers),
        "--vision_tower",       args.vision_tower,
        "--geo_lambda",         f"{args.geo_lambda}",
        "--max_anchors_per_epoch", str(args.max_anchors),
        "--prune_sentinel",     str(trial_dir / ".prune"),
        "--save_freq",          "999",   # don't save per-epoch ckpts in sweep
        "--print_freq",         "100",
    ]
    if args.target_names:
        cmd += ["--target_names", args.target_names]

    # Sampled HPs as overrides
    for k, v in hp.items():
        cmd += [f"--{k}", v]

    return cmd


# ── CSV poll helpers ──────────────────────────────────────────────────────

def _read_epoch_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _parse_val_loss(row: Dict[str, str]) -> Optional[Tuple[int, float]]:
    """Return (epoch, val_loss) if both present; else None."""
    try:
        v = row.get("val_loss", "")
        if v in ("", None):
            return None
        return int(float(row["epoch"])), float(v)
    except (KeyError, ValueError, TypeError):
        return None


# ── Objective ─────────────────────────────────────────────────────────────

def make_objective(args):
    """Closure over args; returns the optuna objective callable."""
    sweep_dir = Path(args.sweep_dir).resolve()

    def objective(trial: optuna.Trial) -> float:
        hp = sample_hps(trial, args.search_space)

        trial_dir = sweep_dir / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        sentinel  = trial_dir / ".prune"
        epoch_csv = trial_dir / "loss_epoch.csv"
        result_js = trial_dir / "trial_result.json"

        master_port = args.master_port_base + (trial.number % 200)
        cmd = build_trial_cmd(args, trial_dir, hp, master_port)

        # Persist the exact command for debugging / repro
        (trial_dir / "trial_cmd.txt").write_text(
            " ".join(shlex.quote(x) for x in cmd) + "\n"
        )

        log_path = trial_dir / "trial.log"
        log_f = open(log_path, "w")
        print(f"[trial {trial.number:03d}] launching: master_port={master_port}  "
              f"hp={hp}")
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, env=os.environ.copy()
        )

        reported_epochs: set = set()
        try:
            while proc.poll() is None:
                time.sleep(args.poll_secs)
                for row in _read_epoch_csv(epoch_csv):
                    pe = _parse_val_loss(row)
                    if pe is None:
                        continue
                    ep, val = pe
                    if ep in reported_epochs:
                        continue
                    reported_epochs.add(ep)
                    trial.report(val, step=ep)
                    if trial.should_prune():
                        print(f"[trial {trial.number:03d}] prune at epoch {ep} "
                              f"(val_loss={val:.4f}); sentinel={sentinel}")
                        sentinel.touch()
                        # Let the trial exit gracefully so the DDP group
                        # tears down cleanly and trial_result.json is written.
                        try:
                            proc.wait(timeout=600)
                        except subprocess.TimeoutExpired:
                            proc.terminate()
                            proc.wait(timeout=60)
                        raise optuna.TrialPruned()
            # Subprocess ended on its own
            proc.wait()
        finally:
            log_f.close()

        if proc.returncode != 0:
            raise RuntimeError(
                f"trial {trial.number} failed: returncode={proc.returncode}; "
                f"see {log_path}"
            )

        if not result_js.exists():
            raise RuntimeError(
                f"trial {trial.number} produced no trial_result.json; "
                f"see {log_path}"
            )
        result = json.loads(result_js.read_text())
        best = result.get("best_val_loss")
        if best is None:
            raise RuntimeError(
                f"trial {trial.number} trial_result.json missing "
                f"best_val_loss: {result}"
            )
        print(f"[trial {trial.number:03d}] done  best_val_loss={best:.4f}  "
              f"best_epoch={result.get('best_epoch')}")
        return float(best)

    return objective


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{sweep_dir / 'optuna.db'}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_warmup_steps=args.pruner_warmup),
        load_if_exists=True,
    )

    print(f"[sweep] study={args.study_name}  storage={storage}")
    print(f"[sweep] preset={args.search_space}  "
          f"unfreeze_mode={args.unfreeze_mode}  "
          f"n_trials={args.n_trials}  sweep_epochs={args.sweep_epochs}  "
          f"max_anchors={args.max_anchors}  geo_lambda={args.geo_lambda}")
    print(f"[sweep] sweep_dir={sweep_dir}")

    study.optimize(
        make_objective(args),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    # Final summary
    print("─" * 70)
    print(f"[sweep] complete.  trials={len(study.trials)}")
    try:
        print(f"[sweep] best value = {study.best_value:.6f}")
        print(f"[sweep] best params = {study.best_params}")
    except ValueError:
        print("[sweep] no completed trials.")

    df = study.trials_dataframe()
    results_csv = sweep_dir / "results.csv"
    df.to_csv(results_csv, index=False)
    print(f"[sweep] trials dataframe → {results_csv}")

    try:
        best = study.best_trial
        with open(sweep_dir / "best_params.json", "w") as f:
            json.dump({
                "study_name":    args.study_name,
                "unfreeze_mode": args.unfreeze_mode,
                "search_space":  args.search_space,
                "best_value":    best.value,
                "best_trial":    best.number,
                "best_params":   best.params,
            }, f, indent=2)
        print(f"[sweep] best_params.json → {sweep_dir / 'best_params.json'}")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
