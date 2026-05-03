"""
Depth-based 3D lifting for LISA-3D inference.

Converts per-frame (predicted mask + depth + camera intrinsics) into a
labelled 3D point cloud in camera frame (metres), matching the prediction
format expected by graspclutter6dAPI/utils/eval_seg_3d_iou.py:

  predictions/{scene_id:06d}/seg_3d.npz
    "points": (N, 3) float32  — camera-frame coordinates in metres
    "labels": (N,)  int32     — obj_id for each point (0 = background)

SAM-3D interface note:
  The paper's Stage 2 feeds RGBA prompts to a frozen SAM-3D reconstructor
  to produce Gaussian splats or meshes.  Because SAM-3D (Meta internal) is
  not publicly available, we instead use depth-based back-projection which
  is directly supported by GraspClutter6D's RGB-D data and is compatible
  with the Clutt3R-Seg evaluation protocol.
"""

import os
import struct
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── Core back-projection ──────────────────────────────────────────────────

def unproject_depth(
    depth_mm: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    label: int,
    depth_scale: float = 1000.0,
    depth_min_m: float = 0.2,
    depth_max_m: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project masked depth pixels into camera-frame 3D coordinates.

    Args:
        depth_mm:    (H, W) uint16 or float32 depth in millimetres.
        K:           (3, 3) camera intrinsic matrix.
        mask:        (H, W) bool — True for foreground pixels to lift.
        label:       Integer obj_id assigned to all returned points.
        depth_scale: Divisor to convert depth_mm values to metres
                     (1000.0 for realsense cameras where values are in mm).
        depth_min_m: Minimum valid depth in metres (default 0.2 m).
        depth_max_m: Maximum valid depth in metres (default 5.0 m).

    Returns:
        points: (N, 3) float32 — [x, y, z] in camera frame, metres.
        labels: (N,)  int32   — all equal to `label`.
    """
    H, W = depth_mm.shape
    z = depth_mm.astype(np.float64) / depth_scale  # → metres

    # Combine foreground mask with valid depth range
    valid = mask & (z >= depth_min_m) & (z <= depth_max_m)

    if not valid.any():
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,),   dtype=np.int32),
        )

    # Pixel coordinate grids
    us, vs = np.meshgrid(np.arange(W), np.arange(H))   # (H, W) each
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z_v = z[valid]
    x_v = (us[valid] - cx) / fx * z_v
    y_v = (vs[valid] - cy) / fy * z_v

    points = np.stack([x_v, y_v, z_v], axis=-1).astype(np.float32)
    labels = np.full(points.shape[0], label, dtype=np.int32)
    return points, labels


# ── RGBA prompt creation (for SAM-3D interface) ───────────────────────────

def make_rgba_prompt(
    rgb: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """Concatenate RGB image with binary mask as alpha channel.

    Creates the RGBA prompt I^{prompt} = [I, M] as described in the paper.
    SAM-3D uses the alpha channel to identify the reconstruction target.

    Args:
        rgb:         (H, W, 3) uint8 RGB image.
        binary_mask: (H, W) bool or uint8 binary mask.

    Returns:
        (H, W, 4) uint8 RGBA image (alpha = 255 for foreground).
    """
    alpha = (binary_mask.astype(np.uint8) * 255)
    return np.concatenate([rgb, alpha[:, :, None]], axis=-1)


# ── Save utilities ────────────────────────────────────────────────────────

def save_seg_3d(
    save_path: str,
    points: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Save 3D segmentation prediction in eval-compatible format.

    Args:
        save_path: Path to .npz file (parent directory created automatically).
        points:    (N, 3) float32 — camera-frame coordinates in metres.
        labels:    (N,)  int32   — obj_id per point.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(
        save_path,
        points=points.astype(np.float32),
        labels=labels.astype(np.int32),
    )


def write_ply_ascii(
    filepath: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> None:
    """Write a labelled point cloud as ASCII PLY (no open3d dependency).

    Args:
        filepath: Output .ply path.
        points:   (N, 3) float32 — xyz coordinates.
        colors:   (N, 3) uint8  — RGB colours (optional).
        labels:   (N,)  int32  — per-point labels (optional).
    """
    N = len(points)
    has_color = colors is not None and len(colors) == N
    has_label = labels is not None and len(labels) == N

    properties = ["property float x", "property float y", "property float z"]
    if has_color:
        properties += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    if has_label:
        properties.append("property int label")

    header_lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
    ] + properties + ["end_header"]

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        for i in range(N):
            row = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if has_color:
                row += f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
            if has_label:
                row += f" {int(labels[i])}"
            f.write(row + "\n")


# ── Multi-view accumulation ───────────────────────────────────────────────

def accumulate_views(
    all_points: List[np.ndarray],
    all_labels: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate per-frame point clouds.

    Each frame's points are already in their own camera frame, consistent
    with the Clutt3R-Seg evaluation protocol which evaluates per-camera-frame
    predictions and accumulates via concatenation.

    Args:
        all_points: List of (N_i, 3) float32 arrays.
        all_labels: List of (N_i,)  int32  arrays.

    Returns:
        (N_total, 3) float32, (N_total,) int32
    """
    valid_pts = [p for p in all_points if len(p) > 0]
    valid_lbl = [l for l in all_labels if len(l) > 0]
    if not valid_pts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.concatenate(valid_pts, axis=0), np.concatenate(valid_lbl, axis=0)
