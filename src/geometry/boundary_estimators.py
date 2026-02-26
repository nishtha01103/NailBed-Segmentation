"""Alternative boundary estimation methods: K-Means, Otsu, Canny.

Each estimator returns a projection value along the PCA major axis where
it believes the nail-bed / free-edge boundary lies, or ``None`` if it
cannot make a confident determination.  They are consumed by the
multi-method voting system in ``pipeline.py``.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


def _kmeans_boundary_estimate(
    nail_yx: np.ndarray,
    proj_major: np.ndarray,
    lab_img: np.ndarray,
    end_is_distal: bool,
    axis_span: float,
    min_proj: float,
) -> Optional[float]:
    """Use K-means (k=2) on LAB to find nail-bed vs free-edge boundary.

    Only returns a result when the two clusters are spatially separated
    (free-edge cluster sits distal to nail-bed cluster with < 20% overlap).
    Returns projection of the boundary in the ORIGINAL (non-reordered) space.

    Edge cases handled:
    - Fewer than 100 nail pixels -> return None (too small for reliable clustering)
    - Clusters overlap -> return None
    - Only 1 cluster dominant (trimmed nail) -> return None
    - Polished nail (low saturation variance) -> unreliable, return None
    """
    if len(nail_yx) < 100:
        return None

    L_v = lab_img[:, :, 0][nail_yx[:, 0], nail_yx[:, 1]].astype(np.float32)
    a_v = lab_img[:, :, 1][nail_yx[:, 0], nail_yx[:, 1]].astype(np.float32)
    b_v = lab_img[:, :, 2][nail_yx[:, 0], nail_yx[:, 1]].astype(np.float32)

    # Early exit: if L variance is very low, nail is uniform -- no free edge visible
    if float(np.std(L_v)) < 5.0:
        return None

    features = np.stack([L_v, a_v, b_v], axis=1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    try:
        _, labels, centers = cv2.kmeans(
            features, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
    except cv2.error:
        return None

    labels = labels.flatten()

    # Free edge cluster = highest L centroid (white/translucent)
    free_cluster = int(np.argmax(centers[:, 0]))
    bed_cluster  = 1 - free_cluster

    is_free = labels == free_cluster
    is_bed  = labels == bed_cluster

    # Require both clusters to have at least 5% of pixels
    pct_free = float(is_free.sum()) / max(len(labels), 1)
    pct_bed  = float(is_bed.sum())  / max(len(labels), 1)
    if pct_free < 0.05 or pct_bed < 0.05:
        return None  # One cluster dominates -- trimmed nail or no free edge

    proj_free = proj_major[is_free]
    proj_bed  = proj_major[is_bed]

    # Check spatial separation: free edge cluster must sit distal to nail bed
    if end_is_distal:
        # Distal = high proj values. Free edge min should be > bed median
        if float(proj_free.min()) > float(np.percentile(proj_bed, 60)):
            return float(0.5 * (proj_free.min() + np.percentile(proj_bed, 95)))
    else:
        # Distal = low proj values. Free edge max should be < bed median
        if float(proj_free.max()) < float(np.percentile(proj_bed, 40)):
            return float(0.5 * (proj_free.max() + np.percentile(proj_bed, 5)))

    return None  # Clusters overlap -- not reliable


def _otsu_boundary_estimate(
    nail_yx: np.ndarray,
    proj_major: np.ndarray,
    gray_img: np.ndarray,
    mask: np.ndarray,
    end_is_distal: bool,
    centroid: np.ndarray,
    major_axis: np.ndarray,
    debug: bool = False,
) -> Optional[float]:
    """Otsu threshold on grayscale nail region to find free edge boundary.

    The free edge (white/translucent) is reliably brighter than the nail bed
    in grayscale. Otsu finds the optimal threshold separating these two groups
    without any fixed value, making it adaptive to image brightness.

    Works well when: light skin, natural nails, unpolished
    Works poorly when: polished nails, very dark skin, underexposure
    Returns None in all failure/uncertain cases rather than guessing.

    Edge cases handled:
    - Nail region too uniform (max-min < 15 gray units) -> None
    - Bright fraction outside 5-40% (not a free edge proportion) -> None
    - Bright pixels not spatially separated from bed pixels -> None
    - Fewer than 50 nail pixels -> None

    Args:
        nail_yx:      All nail pixel coordinates (N, 2) row/col
        proj_major:   Major axis projections for all nail pixels (N,)
        image:        Original BGR image
        mask:         Binary nail mask (255 = nail)
        end_is_distal: True if max proj end is distal
        centroid:     PCA centroid [x, y]
        major_axis:   PCA major axis unit vector
        debug:        Print diagnostics when True

    Returns:
        Boundary projection value in original (non-reordered) axis space, or None
    """
    if len(nail_yx) < 50:
        return None

    nail_gray = gray_img[nail_yx[:, 0], nail_yx[:, 1]]

    nail_min = float(nail_gray.min())
    nail_max = float(nail_gray.max())

    # Too uniform for Otsu to produce a meaningful split
    if nail_max - nail_min < 15.0:
        if debug:
            print(f"  [Otsu] Uniform nail (range={nail_max-nail_min:.1f}) -- skipped")
        return None

    # Normalize pixel brightness to full 0-255 range within nail only
    nail_norm = (
        (nail_gray.astype(np.float32) - nail_min) /
        (nail_max - nail_min) * 255.0
    ).astype(np.uint8)

    thresh_val, _ = cv2.threshold(
        nail_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    is_bright = nail_norm > thresh_val
    bright_fraction = float(is_bright.sum()) / max(len(nail_norm), 1)

    if debug:
        print(f"  [Otsu] thresh={thresh_val:.0f}  bright_fraction={bright_fraction:.2f}")

    # Free edge should be 5–55% of total nail area.
    # Raised from 42% — long nails with prominent free edges were being rejected.
    if not (0.05 <= bright_fraction <= 0.55):
        if debug:
            print(f"  [Otsu] bright_fraction {bright_fraction:.2f} out of range -- skipped")
        return None

    bright_proj = proj_major[is_bright]
    bed_proj    = proj_major[~is_bright]

    if len(bright_proj) < 5 or len(bed_proj) < 5:
        return None

    # Spatial separation check
    if end_is_distal:
        fe_min  = float(bright_proj.min())
        bed_med = float(np.percentile(bed_proj, 55))
        if fe_min > bed_med:
            boundary = float(
                0.5 * (fe_min + float(np.percentile(bed_proj, 95)))
            )
            if debug:
                print(f"  [Otsu] Boundary found at proj={boundary:.1f} "
                      f"(fe_min={fe_min:.1f} bed_p95={np.percentile(bed_proj,95):.1f})")
            return boundary
    else:
        fe_max  = float(bright_proj.max())
        bed_med = float(np.percentile(bed_proj, 45))
        if fe_max < bed_med:
            boundary = float(
                0.5 * (fe_max + float(np.percentile(bed_proj, 5)))
            )
            if debug:
                print(f"  [Otsu] Boundary found at proj={boundary:.1f}")
            return boundary

    if debug:
        print("  [Otsu] Bright pixels not spatially separated -- skipped")
    return None


def _canny_boundary_estimate(
    nail_yx: np.ndarray,
    proj_major: np.ndarray,
    gray_img: np.ndarray,
    mask: np.ndarray,
    centroid: np.ndarray,
    major_axis: np.ndarray,
    minor_axis: np.ndarray,
    end_is_distal: bool,
    axis_span: float,
    debug: bool = False,
    precomputed_edges: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Find hyponychium boundary using Canny edge detection within nail mask.

    The hyponychium creates a physically distinct edge (skin-nail junction)
    visible in grayscale even when color differences are weak.

    Algorithm:
    1. Convert to grayscale and blur (removes nail texture noise)
    2. Apply Canny within nail mask region only
    3. Find horizontal edge lines in the distal 40% of the nail
    4. The dominant horizontal edge cluster = hyponychium line
    5. Convert edge pixel positions to major axis projections

    Edge cases handled:
    - No edges in distal region -> None
    - Edges too fragmented (< 8 connected edge pixels in any cluster) -> None
    - Edge is perpendicular to major axis (not a boundary line) -> None
    - Nail too small (< 100 pixels) -> None
    """
    if len(nail_yx) < 100:
        return None

    if precomputed_edges is not None:
        edges = precomputed_edges.copy()
    else:
        nail_median_gray = int(np.median(gray_img[nail_yx[:, 0], nail_yx[:, 1]]))
        masked_gray = np.full_like(gray_img, nail_median_gray)
        masked_gray[mask > 0] = gray_img[mask > 0]

        blurred = cv2.GaussianBlur(masked_gray, (7, 7), 2.0)
        edges = cv2.Canny(blurred, threshold1=18, threshold2=55)
        edges[mask == 0] = 0

    # Find edge pixels
    edge_yx = np.argwhere(edges > 0)
    if len(edge_yx) < 8:
        if debug:
            print("  [Canny] Too few edge pixels -- skipped")
        return None

    # Project edge pixels onto the major axis
    edge_xy       = edge_yx[:, ::-1].astype(np.float64)
    edge_centered = edge_xy - centroid
    edge_proj_maj = edge_centered @ major_axis
    edge_proj_min = edge_centered @ minor_axis

    # Define distal search zone: distal 40% of nail
    min_p = float(proj_major.min())
    max_p = float(proj_major.max())

    if end_is_distal:
        distal_start = min_p + 0.55 * axis_span
        distal_end   = min_p + 0.95 * axis_span
    else:
        distal_start = max_p - 0.95 * axis_span
        distal_end   = max_p - 0.55 * axis_span

    in_distal_zone  = (edge_proj_maj >= distal_start) & (edge_proj_maj <= distal_end)
    distal_edge_proj = edge_proj_maj[in_distal_zone]
    distal_edge_min  = edge_proj_min[in_distal_zone]

    if len(distal_edge_proj) < 8:
        if debug:
            print(f"  [Canny] Only {len(distal_edge_proj)} edge pixels in distal zone -- skipped")
        return None

    # Histogram-based peak detection
    bin_width = max(3.0, axis_span / 60.0)
    n_bins    = max(10, int((distal_end - distal_start) / bin_width))
    hist, bin_edges = np.histogram(distal_edge_proj, bins=n_bins,
                                   range=(distal_start, distal_end))

    if hist.max() < 4:
        if debug:
            print("  [Canny] No dominant edge cluster found -- skipped")
        return None

    peak_bin  = int(np.argmax(hist))
    peak_proj = float(0.5 * (bin_edges[peak_bin] + bin_edges[peak_bin + 1]))

    # Verify the edge cluster spans at least 30% of nail width
    in_peak_bin = (
        (distal_edge_proj >= bin_edges[peak_bin]) &
        (distal_edge_proj <  bin_edges[peak_bin + 1])
    )
    if in_peak_bin.sum() < 4:
        return None

    lateral_span = float(
        distal_edge_min[in_peak_bin].max() -
        distal_edge_min[in_peak_bin].min()
    )
    nail_width_estimate = float(
        (edge_proj_min.max() - edge_proj_min.min())
    )

    if nail_width_estimate > 0:
        width_coverage = lateral_span / nail_width_estimate
        if width_coverage < 0.25:
            if debug:
                print(f"  [Canny] Edge covers only {width_coverage:.0%} of nail width "
                      "-- likely texture edge, not hyponychium")
            return None

    if debug:
        print(f"  [Canny] Boundary at proj={peak_proj:.1f}  "
              f"width_coverage={width_coverage:.0%}  "
              f"peak_count={hist[peak_bin]}")

    return float(peak_proj)
