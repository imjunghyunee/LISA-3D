"""
Differentiable depth-based reprojection warping for LISA-3D.

Implements the warping operator W that maps per-pixel logits/probabilities
from view A into the coordinate frame of view B using depth maps and
camera intrinsics/extrinsics.

Reference: LISA-3D paper, Section 2.2 (Differentiable reprojection).
"""

import torch
import torch.nn.functional as F


def inv_w2c(E: torch.Tensor) -> torch.Tensor:
    """Analytical inverse of a batch of world-to-camera matrices.

    For E = [R | t ; 0 1] with R orthogonal,
    E^{-1} = [R^T | -R^T t ; 0 1].

    Args:
        E: (..., 4, 4) world-to-camera matrices, t in meters.

    Returns:
        (..., 4, 4) camera-to-world matrices.
    """
    R = E[..., :3, :3]          # (..., 3, 3)
    t = E[..., :3, 3:4]         # (..., 3, 1)
    R_T = R.transpose(-1, -2)   # (..., 3, 3)
    t_inv = -R_T @ t             # (..., 3, 1)

    E_inv = torch.zeros_like(E)
    E_inv[..., :3, :3] = R_T
    E_inv[..., :3, 3:4] = t_inv
    E_inv[..., 3, 3] = 1.0
    return E_inv


def warp_mask(
    P_a: torch.Tensor,
    depth_a: torch.Tensor,
    K_a: torch.Tensor,
    E_a: torch.Tensor,
    K_b: torch.Tensor,
    E_b: torch.Tensor,
    depth_min_mm: float = 200.0,
    depth_max_mm: float = 5000.0,
) -> torch.Tensor:
    """Warp soft probability mask from view A into view B's frame.

    Implements the warping operator W from the LISA-3D paper:
      x_3D = D_a(u) * K_a^{-1} * [u, v, 1]^T          (backproject)
      x̃    = E_b * E_a^{-1} * [x_3D; 1]               (transform)
      u'   = K_b * [x̃, ỹ, z̃]^T / z̃                   (reproject)
      P̃_{a→b} = bilinear_sample(P_a, u')               (sample)

    Args:
        P_a:        (B, H, W) — soft probability map for view A in [0, 1].
        depth_a:    (B, H, W) — depth map in millimetres (uint16-compatible
                    float32, raw GraspClutter6D values).
        K_a:        (B, 3, 3) — camera intrinsics for view A.
        E_a:        (B, 4, 4) — world-to-camera for view A (t in metres).
        K_b:        (B, 3, 3) — camera intrinsics for view B.
        E_b:        (B, 4, 4) — world-to-camera for view B (t in metres).
        depth_min_mm: Minimum valid depth in mm (default 200 mm = 0.2 m).
        depth_max_mm: Maximum valid depth in mm (default 5000 mm = 5.0 m).

    Returns:
        P_tilde_a2b: (B, H, W) — warped probability map aligned to view B.
                     Out-of-bounds and invalid-depth regions are 0.
    """
    B, H, W = P_a.shape
    device = P_a.device
    dtype = P_a.dtype

    # ── 1. Build pixel grid for view A ────────────────────────────────────
    # (u, v) — pixel coordinates, origin top-left
    vs, us = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )  # (H, W)

    # ── 2. Convert depth to metres; mask invalid pixels ───────────────────
    z_m = depth_a / 1000.0  # mm → m, (B, H, W)
    valid = (depth_a >= depth_min_mm) & (depth_a <= depth_max_mm)  # (B, H, W)

    # ── 3. Unproject: camera-frame 3D coordinates in view A ──────────────
    fx_a = K_a[:, 0, 0].view(B, 1, 1)   # (B, 1, 1)
    fy_a = K_a[:, 1, 1].view(B, 1, 1)
    cx_a = K_a[:, 0, 2].view(B, 1, 1)
    cy_a = K_a[:, 1, 2].view(B, 1, 1)

    us_b = us.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    vs_b = vs.unsqueeze(0).expand(B, -1, -1)

    x_a = (us_b - cx_a) / fx_a * z_m   # (B, H, W)
    y_a = (vs_b - cy_a) / fy_a * z_m
    # pts_cam_a: (B, H, W, 3)
    pts_cam_a = torch.stack([x_a, y_a, z_m], dim=-1)

    # Zero out invalid points so they land far outside the image after projection
    pts_cam_a = pts_cam_a * valid.unsqueeze(-1).to(dtype)

    # ── 4. Homogeneous coordinates: (B, H*W, 4) ──────────────────────────
    ones = torch.ones(B, H, W, 1, device=device, dtype=dtype)
    pts_hom = torch.cat([pts_cam_a, ones], dim=-1)          # (B, H, W, 4)
    pts_hom_flat = pts_hom.view(B, H * W, 4).transpose(1, 2)  # (B, 4, N)

    # ── 5. Transform from camera A to camera B ────────────────────────────
    # T_{a→b} = E_b @ inv(E_a)  (both are w2c; inv(E_a) is c2w)
    T = E_b @ inv_w2c(E_a)  # (B, 4, 4)
    pts_cam_b = (T @ pts_hom_flat)[:, :3, :]  # (B, 3, N)
    pts_cam_b = pts_cam_b.transpose(1, 2).view(B, H, W, 3)  # (B, H, W, 3)

    x_b = pts_cam_b[..., 0]  # (B, H, W)
    y_b = pts_cam_b[..., 1]
    z_b = pts_cam_b[..., 2].clamp(min=1e-6)  # avoid /0

    # ── 6. Reproject into view B pixel coordinates ────────────────────────
    fx_b = K_b[:, 0, 0].view(B, 1, 1)
    fy_b = K_b[:, 1, 1].view(B, 1, 1)
    cx_b = K_b[:, 0, 2].view(B, 1, 1)
    cy_b = K_b[:, 1, 2].view(B, 1, 1)

    u_b = fx_b * x_b / z_b + cx_b   # (B, H, W)
    v_b = fy_b * y_b / z_b + cy_b

    # ── 7. Normalise to [-1, 1] for grid_sample ───────────────────────────
    # grid_sample expects (x, y) = (col, row) normalised to [-1, 1]
    grid_x = 2.0 * u_b / (W - 1) - 1.0   # (B, H, W)
    grid_y = 2.0 * v_b / (H - 1) - 1.0

    # grid_sample grid shape: (B, H_out, W_out, 2) where last dim is (x, y)
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, H, W, 2)

    # ── 8. Bilinear sample ────────────────────────────────────────────────
    # P_a: (B, H, W) → (B, 1, H, W) for grid_sample
    P_a_4d = P_a.unsqueeze(1)
    warped = F.grid_sample(
        P_a_4d,
        grid,
        mode="bilinear",
        padding_mode="zeros",   # out-of-bounds → 0
        align_corners=True,
    )  # (B, 1, H, W)

    return warped.squeeze(1)  # (B, H, W)
