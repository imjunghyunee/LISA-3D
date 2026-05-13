"""
Deterministic train/val scene-id split for LISA-3D hyperparameter sweeps.

`tune.py` runs many trials against a held-out validation set carved from
`split_info/grasp_train_scene_ids.json` (the GraspClutter6D test split is
reserved for final evaluation, not for HP search). All trials in a single
sweep MUST share the same train/val partition so their objective values
are directly comparable; `make_val_split` derives that partition from a
seed for reproducibility.

Usage (library):
    train_ids, val_ids = make_val_split(
        "/path/to/grasp_train_scene_ids.json", n_val=40, seed=1337)

Usage (CLI, called once by run_sweep.sh):
    python -m utils.val_split \
        --in_path  /path/to/grasp_train_scene_ids.json \
        --out_train_path runs/sweep/<study>/train_ids.json \
        --out_val_path   runs/sweep/<study>/val_ids.json \
        --n_val 40 --seed 1337
"""

import argparse
import json
import os
import random
from typing import List, Tuple


def make_val_split(train_json_path: str, n_val: int,
                   seed: int = 1337) -> Tuple[List[int], List[int]]:
    """Return (train_ids, val_ids) for a deterministic seeded split.

    Args:
        train_json_path: path to a JSON list of integer scene_ids (e.g.,
            ``grasp_train_scene_ids.json``).
        n_val: number of scenes to reserve for validation.
        seed: RNG seed; the same (file, n_val, seed) always returns the
            same partition.

    Raises:
        FileNotFoundError: if ``train_json_path`` does not exist.
        ValueError: if ``n_val`` is not in (0, len(ids)).
    """
    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"Scene id JSON not found: {train_json_path}")
    with open(train_json_path) as f:
        ids = sorted(int(x) for x in json.load(f))
    if not (0 < n_val < len(ids)):
        raise ValueError(
            f"n_val must be in (0, {len(ids)}); got {n_val}."
        )
    rng = random.Random(seed)
    val = sorted(rng.sample(ids, n_val))
    train = sorted(set(ids) - set(val))
    return train, val


def _write_json(path: str, ids: List[int]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(ids, f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in_path",        required=True,
                   help="Path to grasp_train_scene_ids.json.")
    p.add_argument("--out_train_path", required=True,
                   help="Output JSON for the train partition.")
    p.add_argument("--out_val_path",   required=True,
                   help="Output JSON for the val partition.")
    p.add_argument("--n_val",          type=int, default=40)
    p.add_argument("--seed",           type=int, default=1337)
    p.add_argument("--force",          action="store_true",
                   help="Overwrite existing output files.")
    args = p.parse_args()

    if (not args.force
            and os.path.exists(args.out_train_path)
            and os.path.exists(args.out_val_path)):
        print(f"[val_split] outputs already exist, skipping "
              f"(--force to overwrite):\n"
              f"  {args.out_train_path}\n"
              f"  {args.out_val_path}")
        return

    train, val = make_val_split(args.in_path, args.n_val, args.seed)
    _write_json(args.out_train_path, train)
    _write_json(args.out_val_path,   val)
    print(f"[val_split] seed={args.seed}  n_val={args.n_val}  "
          f"train={len(train)}  val={len(val)}")
    print(f"  → {args.out_train_path}")
    print(f"  → {args.out_val_path}")


if __name__ == "__main__":
    main()
