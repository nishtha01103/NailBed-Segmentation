"""Mask cleaning utilities for nail segmentation geometry pipeline.

CHANGES vs original:
- Step 1.5 (new): Remove components that touch the image border AND are
  smaller than 20% of the main component.  These are almost always
  background/skin bleed-through that YOLO includes at the mask edge.
  Kept only when there is no large main component yet (graceful fallback).
- Step 5 minimum-area threshold raised from 5% to 8% — aggressive enough
  to remove the small background islands described in Case C.
- Comments tightened; no logic changes to steps 2-4, 6.
"""

import cv2
import numpy as np
import logging
from config import CLEAN_MASK_BEFORE_GEOMETRY, MORPHOLOGY_KERNEL_SIZE

logger = logging.getLogger(__name__)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in binary mask using flood-fill from image border.

    Finds all background pixels connected to the image border, then marks
    everything else as foreground.  This reliably closes holes caused by
    specular glare or skin folds inside the nail region without enlarging
    the mask boundary.

    Args:
        mask: Binary mask (255 = nail, 0 = background)

    Returns:
        Mask with interior holes filled.
    """
    h, w = mask.shape[:2]
    flood = mask.copy()
    pad   = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def _clean_mask_morphology(mask: np.ndarray) -> np.ndarray:
    """Remove outliers and noise from mask using morphological operations.

    Pipeline (order is important):
    1.    Keep only the largest connected component.
    1.5   Remove border-touching components that are < 20% of main component.
    2.    Morphological closing (5×5) — fills small gaps inside the nail.
    3.    Flood-fill hole filling — closes specular-glare holes.
    4.    Morphological opening (kernel) — removes thin edge protrusions.
    5.    Remove fragments < 8% of total remaining area.
    6.    Slight erosion — removes distal skin contamination.

    Args:
        mask: Binary mask where 255 = nail region, 0 = background

    Returns:
        Cleaned binary mask with outliers removed
    """
    if not CLEAN_MASK_BEFORE_GEOMETRY:
        return mask

    # ------------------------------------------------------------------ #
    # Step 1 — Keep only the largest connected component                  #
    # ------------------------------------------------------------------ #
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = int(np.argmax(areas)) + 1
        main_area = int(areas[largest_label - 1])
        mask = np.where(labels == largest_label, np.uint8(255), np.uint8(0))
    else:
        mask = mask.copy()
        main_area = int(np.sum(mask > 0))

    # ------------------------------------------------------------------ #
    # Step 1.5 — Remove border-touching background fragments              #
    # Border bleed suppression for segmentation artifacts.                #
    # Rationale: YOLO often bleeds a small skin/background patch into the #
    # mask at the image boundary.  These corrupt the axis projection and  #
    # mimic a bright free edge (Case C).  We remove a component when:     #
    #   (a) it touches any image border, AND                              #
    #   (b) EITHER it is < 20% of the main component area                 #
    #       OR  > 30% of its bounding box lies outside the 90% central   #
    #       image region (border-bleed geometry test)                     #
    # Safety: the largest surviving component (main nail) is never        #
    # removed regardless of position or bounding box.                     #
    # ------------------------------------------------------------------ #
    h, w = mask.shape[:2]
    border_strip = np.zeros_like(mask)
    border_strip[0, :]  = 255
    border_strip[-1, :] = 255
    border_strip[:, 0]  = 255
    border_strip[:, -1] = 255

    # Central 90% region bounds
    _cx_min = 0.05 * w
    _cx_max = 0.95 * w
    _cy_min = 0.05 * h
    _cy_max = 0.95 * h

    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # Identify the largest component so it is never removed
    _largest2 = (
        int(np.argmax(stats2[1:, cv2.CC_STAT_AREA])) + 1
        if num_labels2 > 1 else 1
    )

    clean_border = mask.copy()
    for i in range(1, num_labels2):
        # Never remove the main nail
        if i == _largest2:
            continue

        comp_mask = (labels2 == i).astype(np.uint8) * 255
        comp_area = int(stats2[i, cv2.CC_STAT_AREA])

        # Condition (a): must touch the image border
        if not np.any(cv2.bitwise_and(comp_mask, border_strip) > 0):
            continue

        # Condition (b1): smaller than 20% of main nail
        _small = comp_area < 0.20 * max(main_area, 1)

        # Condition (b2): >30% of bounding box lies outside central 90% region
        _bx  = int(stats2[i, cv2.CC_STAT_LEFT])
        _by  = int(stats2[i, cv2.CC_STAT_TOP])
        _bw  = int(stats2[i, cv2.CC_STAT_WIDTH])
        _bh  = int(stats2[i, cv2.CC_STAT_HEIGHT])
        _bbox_area = max(_bw * _bh, 1)
        _ox  = max(0.0, min(_bx + _bw, _cx_max) - max(_bx, _cx_min))
        _oy  = max(0.0, min(_by + _bh, _cy_max) - max(_by, _cy_min))
        _outside_frac = (_bbox_area - _ox * _oy) / _bbox_area
        _mostly_outside = _outside_frac > 0.30

        if _small or _mostly_outside:
            clean_border[labels2 == i] = 0
            logger.debug(
                "[MaskProcessing] Removed border fragment: area=%d (main=%d) "
                "outside_frac=%.2f small=%s mostly_outside=%s",
                comp_area, main_area, _outside_frac, _small, _mostly_outside,
            )
    mask = clean_border

    # ------------------------------------------------------------------ #
    # Step 2 — Morphological closing with explicit 5×5 kernel             #
    # ------------------------------------------------------------------ #
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # ------------------------------------------------------------------ #
    # Step 3 — Flood-fill hole filling                                    #
    # ------------------------------------------------------------------ #
    mask = _fill_holes(mask)

    # ------------------------------------------------------------------ #
    # Step 4 — Morphological opening (configurable kernel)                #
    # ------------------------------------------------------------------ #
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    # ------------------------------------------------------------------ #
    # Step 5 — Remove small fragments left after opening (raised to 8%)   #
    # ------------------------------------------------------------------ #
    min_area = 0.08 * float(np.sum(mask > 0))
    num_labels3, labels3, stats3, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    clean = np.zeros_like(mask)
    for i in range(1, num_labels3):
        if stats3[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels3 == i] = 255
    mask = clean

    # ------------------------------------------------------------------ #
    # Step 6 — Slight erosion to remove distal skin contamination         #
    # ------------------------------------------------------------------ #
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, erode_kernel, iterations=1)

    return mask


def check_distal_curvature(
    mask: np.ndarray,
    major_axis: np.ndarray,
    curvature_threshold: float = 1e-4,
) -> str:
    """Check the curvature of the distal contour to detect segmentation clipping.

    Flat distal contour often indicates segmentation truncation.

    Projects contour points onto the major axis, selects the distal 10%,
    fits a quadratic  p_major = a * p_minor² + b  via numpy.polyfit, and
    flags low confidence when |a| < curvature_threshold (nearly flat edge).

    The mask is never modified; the function only flags possible clipping.

    Args:
        mask:                Binary mask (255 = nail). Not modified.
        major_axis:          Unit vector of the nail's major (length) axis.
        curvature_threshold: |a| below this means distal contour is flat.
                             Default 1e-4.

    Returns:
        "low" — distal contour is nearly flat, possible segmentation clipping.
        "ok"  — curvature is sufficient, no clipping detected.
    """
    # Step 1: Extract contour points from cleaned mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return "ok"

    # Use the largest contour (the nail outline)
    contour = max(contours, key=cv2.contourArea)
    pts = contour[:, 0, :].astype(np.float64)  # (N, 2)  x, y

    if len(pts) < 10:
        return "ok"

    # Step 2: Project contour points onto major and minor axes
    minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float64)
    p_major = pts @ major_axis  # (N,)
    p_minor = pts @ minor_axis  # (N,)

    # Step 3: Select the distal 10% of contour points (highest p_major values)
    distal_threshold = np.percentile(p_major, 90.0)
    distal_mask = p_major >= distal_threshold
    if distal_mask.sum() < 5:
        return "ok"

    d_major = p_major[distal_mask]
    d_minor = p_minor[distal_mask]

    # Step 4: Fit quadratic  p_major = a * p_minor² + (linear) + const
    try:
        coeffs = np.polyfit(d_minor, d_major, 2)
    except (np.linalg.LinAlgError, ValueError):
        return "ok"

    # Step 5: Curvature coefficient |a| (leading term of degree-2 fit)
    a = coeffs[0]

    # Step 6–7: Flat distal contour often indicates segmentation truncation.
    if abs(a) < curvature_threshold:
        geometry_confidence = "low"
        logger.debug(
            "[MaskProcessing] Distal curvature |a|=%.2e < threshold %.2e — "
            "geometry_confidence set to 'low' (possible clipping)",
            abs(a), curvature_threshold,
        )
        return geometry_confidence

    return "ok"


def check_border_contact(
    mask: np.ndarray,
    debug: bool = False,
) -> str:
    """Detect whether the nail mask touches any image border edge.

    Nails touching frame edge may have truncated geometry.

    Runs in O(N) time by checking only the four border rows/columns of the
    mask array.  The mask is never modified.

    Args:
        mask:  Binary mask (255 = nail). Not modified.
        debug: When True, prints a warning if border contact is found.

    Returns:
        "low" — mask touches at least one image border; geometry may be
                truncated.
        "ok"  — mask does not touch any image border.
    """
    # Step 1: Check each of the four border strips in O(N) time.
    # np.any() short-circuits on the first non-zero value found.
    border_contact = (
        np.any(mask[0, :])    # top row
        or np.any(mask[-1, :])  # bottom row
        or np.any(mask[:, 0])   # left column
        or np.any(mask[:, -1])  # right column
    )

    # Step 2–3: Flag low geometry confidence when border is contacted.
    if border_contact:
        geometry_confidence = "low"
        if debug:
            print("[MaskProcessing] WARNING: nail mask touches image border — "
                  "geometry may be truncated (geometry_confidence='low')")
        logger.debug(
            "[MaskProcessing] Nail mask touches image border — "
            "geometry_confidence set to 'low'"
        )
        return geometry_confidence

    return "ok"
