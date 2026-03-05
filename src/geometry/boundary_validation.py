"""Boundary validation helpers for cross-checking candidate boundaries.

These functions do NOT discover boundaries themselves — they validate or
score a boundary projection found by another method (gradient, K-means, etc.)
by comparing it against an independent signal (Otsu brightness, Canny edges).
"""

from typing import Optional
import cv2
import numpy as np


def _otsu_validate_color_boundary(
    nail_yx: np.ndarray,
    proj_major: np.ndarray,
    lab_img: np.ndarray,
    boundary_proj: float,
    window_px: int = 8,
    debug: bool = False,
) -> float:
    """Score how well an existing boundary candidate matches Otsu expectations.

    Instead of finding a new boundary, this validates an EXISTING boundary
    (from gradient or K-means) by checking whether Otsu's threshold would
    agree that there is a brightness jump at that location.

    Returns confidence multiplier (0.5 to 1.3):
    - > 1.0: Otsu confirms the boundary (boost confidence)
    - 1.0:   Neutral (Otsu inconclusive)
    - < 1.0: Otsu contradicts boundary (reduce confidence)
    """
    before_mask = (proj_major >= boundary_proj - window_px) & \
                  (proj_major < boundary_proj)
    after_mask  = (proj_major >= boundary_proj) & \
                  (proj_major < boundary_proj + window_px)

    if before_mask.sum() < 5 or after_mask.sum() < 5:
        return 1.0  # not enough pixels to judge

    L_before = float(np.median(
        lab_img[:, :, 0][nail_yx[before_mask, 0], nail_yx[before_mask, 1]]
    ))
    L_after = float(np.median(
        lab_img[:, :, 0][nail_yx[after_mask, 0], nail_yx[after_mask, 1]]
    ))

    delta = L_after - L_before  # positive = brighter after boundary = free edge

    if debug:
        print(f"  [OtsuValidate] L_before={L_before:.1f}  L_after={L_after:.1f}  "
              f"delta={delta:.1f}")

    if delta > 12.0:
        return 1.3   # Strong confirmation
    elif delta > 6.0:
        return 1.15  # Moderate confirmation
    elif delta > 0.0:
        return 1.05  # Weak confirmation
    elif delta > -5.0:
        return 1.0   # Neutral
    else:
        return 0.7   # Contradicts boundary (L drops = wrong direction)


def _canny_validate_edge_at_boundary(
    edges_img: np.ndarray,
    nail_yx: np.ndarray,
    proj_major: np.ndarray,
    proj_minor: np.ndarray,
    boundary_proj: float,
    window_px: int = 6,
) -> float:
    """Check Canny edge density at a known boundary projection.

    Used to validate/score a boundary found by gradient or Otsu method.
    High edge density at the candidate location confirms a physical boundary.

    Returns confidence multiplier (0.6 to 1.25):
    - High edge density at boundary -> multiplier > 1.0 (confirmed)
    - Low edge density -> multiplier < 1.0 (not a physical edge)
    """
    near_boundary = (
        (proj_major >= boundary_proj - window_px) &
        (proj_major <= boundary_proj + window_px)
    )
    far_from_boundary = (
        (proj_major < boundary_proj - window_px * 3) |
        (proj_major > boundary_proj + window_px * 3)
    )

    if near_boundary.sum() < 3 or far_from_boundary.sum() < 3:
        return 1.0

    near_yx = nail_yx[near_boundary]
    far_yx  = nail_yx[far_from_boundary]

    try:
        near_edge_density = float(np.mean(
            edges_img[near_yx[:, 0], near_yx[:, 1]] > 0
        ))
        far_edge_density = float(np.mean(
            edges_img[far_yx[:, 0], far_yx[:, 1]] > 0
        ))
    except (IndexError, ValueError):
        return 1.0

    if far_edge_density < 1e-6:
        return 1.0  # avoid division by zero

    density_ratio = near_edge_density / far_edge_density

    if density_ratio > 3.0:
        return 1.25   # Strong physical edge confirmed
    elif density_ratio > 1.8:
        return 1.12   # Moderate confirmation
    elif density_ratio > 1.0:
        return 1.05   # Weak confirmation
    elif density_ratio > 0.5:
        return 1.0    # Neutral
    else:
        return 0.75   # No edge here -- likely false positive


def validate_distal_position(
    boundary_proj: float,
    min_proj: float,
    max_proj: float,
    end_is_distal: bool = True,
    debug: bool = False,
) -> bool:
    """Anatomical distal-position constraint for free-edge boundary.

    Free edge must lie near distal tip; prevents nail bed misclassification.

    Applied AFTER brightness, gradient, and variance checks pass — this is a
    final O(1) sanity gate only.  It does not alter the scoring system.

    The check is orientation-aware: ``end_is_distal=True`` means the distal
    tip is at ``max_proj``; ``end_is_distal=False`` means it is at
    ``min_proj``.  ``boundary_ratio`` is always computed as distance from the
    proximal end toward the distal tip, so the threshold is consistent
    regardless of which physical end is distal.

    Args:
        boundary_proj:  Boundary projection value along the major axis.
        min_proj:       Minimum projection value across all nail pixels.
        max_proj:       Maximum projection value across all nail pixels.
        end_is_distal:  True  = distal tip is at max_proj (default).
                        False = distal tip is at min_proj.
        debug:          When True, prints reason for rejection.

    Returns:
        True  — boundary passes the anatomical position constraint.
        False — boundary is rejected (free_edge_present = False,
                boundary_proj should be set to None by the caller).
    """
    span = max_proj - min_proj + 1e-6

    # Normalise boundary position as distance from proximal → distal (0 → 1).
    if end_is_distal:
        boundary_ratio = (boundary_proj - min_proj) / span
    else:
        boundary_ratio = (max_proj - boundary_proj) / span

    # Step 2: Reject if free edge is not in the distal 25% of nail length.
    if boundary_ratio < 0.75:
        if debug:
            print(
                f"[DistalCheck] REJECTED: boundary_ratio={boundary_ratio:.3f} < 0.75 "
                "(not in distal 25% — likely nail-bed misclassification)"
            )
        return False  # free_edge_present = False, boundary_proj = None

    # Reject boundaries extremely close to tip (likely mask edge artefact).
    if boundary_ratio > 0.98:
        if debug:
            print(
                f"[DistalCheck] REJECTED: boundary_ratio={boundary_ratio:.3f} > 0.98 "
                "(too close to distal tip — likely mask edge artefact)"
            )
        return False  # free_edge_present = False

    return True
