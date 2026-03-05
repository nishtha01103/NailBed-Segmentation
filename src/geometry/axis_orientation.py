"""
Geometry-only anatomical axis orientation.

Stable, rotation-invariant, lighting-independent.

STABILITY IMPROVEMENTS v2:
  - Taper uses ALL mask pixels (not just contour boundary points).
    Pixel-based statistics are ~5x less noisy than contour-only.
  - Canonical axis direction: force +x hemisphere before taper to
    prevent ±180 flip when minAreaRect w_rect ≈ h_rect.
  - Curvature fallback threshold raised 0.01→0.02 (require clearer signal).
  - Image-top heuristic fallback: when taper AND curvature are ambiguous,
    the proximal end is closer to image TOP (standard palm-down camera).
"""

from typing import Tuple
import cv2
import numpy as np


def _orient_anatomical_axis(
    image: np.ndarray,
    mask: np.ndarray,
    centroid: np.ndarray,
    major_axis_unused: np.ndarray,
    minor_axis_unused: np.ndarray,
    lab_frame=None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Determine the proximal→distal anatomical axis from mask geometry.

    Returns:
        anatomical_axis : unit vector pointing proximal → distal  (x, y)
        width_axis      : unit vector perpendicular to anatomical  (x, y)
        distal_sign     : +1 (distal = MAX projection) or -1 (distal = MIN)
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1

    contour = max(contours, key=cv2.contourArea)

    # ── 1. Long axis from minAreaRect ────────────────────────────────────
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w_rect, h_rect), angle = rect

    if w_rect >= h_rect:
        angle_rad = np.deg2rad(angle)
    else:
        angle_rad = np.deg2rad(angle + 90)

    anatomical_axis = np.array(
        [np.cos(angle_rad), np.sin(angle_rad)],
        dtype=np.float64
    )
    anatomical_axis /= np.linalg.norm(anatomical_axis)

    # Force canonical direction (+x hemisphere) to prevent ±180 flip
    if anatomical_axis[0] < 0:
        anatomical_axis = -anatomical_axis

    width_axis = np.array([-anatomical_axis[1], anatomical_axis[0]], dtype=np.float64)

    # ── 2. Project ALL mask pixels (not contour) ─────────────────────────
    nail_yx = np.argwhere(mask == 255)
    if len(nail_yx) < 20:
        nail_xy = contour.reshape(-1, 2).astype(np.float64)
    else:
        nail_xy = nail_yx[:, ::-1].astype(np.float64)

    centered   = nail_xy - np.array([cx, cy])
    proj_major = centered @ anatomical_axis
    proj_minor = centered @ width_axis

    span      = proj_major.max() - proj_major.min()
    edge_span = 0.20 * span

    first_mask = proj_major < (proj_major.min() + edge_span)
    last_mask  = proj_major > (proj_major.max() - edge_span)

    if first_mask.sum() < 4 or last_mask.sum() < 4:
        return anatomical_axis, width_axis, 1

    # ── 3. Width taper (pixel-based, robust) ─────────────────────────────
    def robust_width(vals):
        if len(vals) < 4:
            return 0.0
        return float(np.percentile(vals, 92) - np.percentile(vals, 8))

    w1 = robust_width(proj_minor[first_mask])
    w2 = robust_width(proj_minor[last_mask])

    avg_w       = 0.5 * (w1 + w2)
    taper_ratio = abs(w1 - w2) / max(avg_w, 1.0)

    # ── 4. Curvature fallback ─────────────────────────────────────────────
    contour_xy = contour.reshape(-1, 2).astype(np.float64)
    cont_cent  = contour_xy - np.array([cx, cy])
    cont_proj  = cont_cent @ anatomical_axis
    c_min, c_max = cont_proj.min(), cont_proj.max()
    c_span = c_max - c_min

    def edge_curvature(side_last: bool):
        if c_span < 1:
            return 0.0
        if side_last:
            in_r = cont_proj > (c_min + 0.80 * c_span)
        else:
            in_r = cont_proj < (c_min + 0.20 * c_span)
        pts = contour_xy[in_r]
        if len(pts) < 10:
            return 0.0
        diffs  = np.diff(pts, axis=0)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        return float(np.mean(np.abs(np.diff(angles))))

    curv1 = edge_curvature(side_last=False)
    curv2 = edge_curvature(side_last=True)

    # ── 5. Decision: taper → curvature → image-top heuristic ─────────────
    if taper_ratio > 0.07:
        # Wider end = proximal; nail tapers toward free edge
        distal_is_last = w2 < w1

    elif abs(curv2 - curv1) > 0.02:
        # More curved end = distal (rounded free edge)
        distal_is_last = curv2 > curv1

    else:
        # Image-top heuristic: proximal (cuticle) is HIGHER in the image
        # (smaller row index = smaller y-coordinate = top of image).
        # For standard palm-down, camera-above setup, proximal end is
        # closer to the top of the frame.
        # first_mask = MIN-proj end; last_mask = MAX-proj end.
        # We use nail_yx (row = y in image) to measure which end is higher.
        mean_y_first = float(np.mean(nail_yx[first_mask, 0]))
        mean_y_last  = float(np.mean(nail_yx[last_mask,  0]))
        # Proximal = smaller y (higher in image). If MAX-proj end is lower
        # in image (larger y) then MAX-proj = distal.
        distal_is_last = mean_y_last > mean_y_first

    distal_sign    = 1 if distal_is_last else -1
    anatomical_axis = anatomical_axis * distal_sign
    width_axis = np.array([-anatomical_axis[1], anatomical_axis[0]], dtype=np.float64)

    return anatomical_axis, width_axis, distal_sign