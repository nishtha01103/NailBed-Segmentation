"""Curved nail-bed mask builder using parabolic boundary approximation.

Constructs a per-pixel boolean mask that separates the nail bed from the
free edge, following the natural curvature of the hyponychium arc.
"""

import numpy as np


def _build_curved_bed_mask(
    nail_yx: np.ndarray,
    centroid: np.ndarray,
    major_axis: np.ndarray,
    minor_axis: np.ndarray,
    boundary_proj: float,
    nail_width_px: float,
    end_is_distal: bool,
) -> np.ndarray:
    """Return boolean array (len = len(nail_yx)): True = nail bed side.

    Constructs the nail bed mask as all pixels on the proximal side of the boundary.
    If no boundary is found (boundary_proj is None), returns all True (full mask as bed).
    Parabola depth is clamped to ≤8% of nail width.
    """
    if boundary_proj is None or nail_width_px < 5.0:
        # No boundary: treat full mask as bed (trimmed nail)
        print("[BedMaskBuilder] No boundary found. Returning full mask as bed.")
        return np.ones(len(nail_yx), dtype=bool)

    nail_xy  = nail_yx[:, ::-1].astype(np.float64)
    centered = nail_xy - centroid
    p_maj    = centered @ major_axis
    p_min    = centered @ minor_axis

    half_w = nail_width_px / 2.0
    # Clamp parabola depth to ≤8% of width
    max_depth = 0.08 * nail_width_px
    depth = min(0.06 * half_w, max_depth)

    # Parabolic offset: boundary shifts proximally at lateral edges
    parabola_offset = depth * (p_min / half_w) ** 2

    distal_sign = +1 if end_is_distal else -1
    if distal_sign == +1:
        curved_boundary = boundary_proj - parabola_offset
        in_bed = p_maj <= curved_boundary
    else:
        curved_boundary = boundary_proj + parabola_offset
        in_bed = p_maj >= curved_boundary
    bed_fraction = np.sum(in_bed) / len(in_bed)
    print("[BedMaskBuilder] min_proj:", p_maj.min())
    print("[BedMaskBuilder] max_proj:", p_maj.max())
    print("[BedMaskBuilder] boundary_proj:", boundary_proj)
    print("[BedMaskBuilder] distal_sign:", distal_sign)
    print("[BedMaskBuilder] bed_fraction:", bed_fraction)
    return in_bed
