"""
GraspClutter6D obj_id ↔ Object Name mapping helpers.

The dataset ships a CSV (``graspclutter6d_object_id.csv``) whose columns are:
    Object ID, Object Name, Domain, Category, Source, Purchase Link

We map ``Object ID`` (1–200) → ``Object Name`` (e.g. ``"banana"``,
``"cooking_skillet_with_glass lid"``).  Per-object segmentation training in
LISA-3D uses the Object Name as the natural-language prompt key
(``"Please segment the {name} in this image."``) and writes ``obj_id`` as the
per-point label in ``seg_3d.npz`` so the Clutt3R-Seg evaluator scores per
object.

Note: the CSV is UTF-8-BOM and several rows quote commas inside the
"Purchase Link" field, so we parse it with :mod:`csv`, not naive split.
"""

import csv
import json
import os
from typing import Dict, List, Optional


def load_obj_id_to_name(csv_path: str) -> Dict[int, str]:
    """Parse the CSV and return ``{obj_id (int): object_name (str)}``."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Object-id CSV not found: {csv_path}")

    mapping: Dict[int, str] = {}
    # ``utf-8-sig`` strips the BOM if present.
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                obj_id = int(row["Object ID"].strip())
            except (KeyError, ValueError):
                continue
            name = (row.get("Object Name") or "").strip()
            if not name:
                continue
            mapping[obj_id] = name
    return mapping


def name_to_obj_ids(id_to_name: Dict[int, str]) -> Dict[str, List[int]]:
    """Invert the mapping. Each Object Name maps to a list of obj_ids.

    In the official GraspClutter6D CSV the names are unique, so each list has
    length 1 — but we keep the list form so the output schema matches the one
    expected by ``graspclutter6dAPI/utils/eval_seg_3d_iou.py --category_file``.
    """
    out: Dict[str, List[int]] = {}
    for obj_id, name in id_to_name.items():
        out.setdefault(name, []).append(obj_id)
    for name in out:
        out[name].sort()
    return out


def dump_objects_json(
    id_to_name: Dict[int, str],
    out_path: str,
    target_names: Optional[List[str]] = None,
) -> str:
    """Write ``{name: [obj_id]}`` JSON for the per-object IoU evaluator.

    If ``target_names`` is given, only those names are dumped (others skipped).
    """
    name_map = name_to_obj_ids(id_to_name)
    if target_names is not None:
        target_set = set(target_names)
        name_map = {n: ids for n, ids in name_map.items() if n in target_set}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(name_map, f, indent=2, ensure_ascii=False)
    return out_path
