"""Curved nail-bed mask builder using parabolic boundary approximation.

Constructs a per-pixel boolean mask that separates the nail bed from the
free edge, following the natural curvature of the hyponychium arc.

KEY FIX (v3):
─────────────────────────────────────────────────────────────────────
The pipeline sometimes passes the PCA-derived distal_sign (+1) even when
boundary_detection has internally flipped end_is_distal (e.g. axis-flip
correction).  This causes bed_fraction ~0.20 instead of ~0.74.

Fix: after computing the bed with the provided distal_sign, check whether
bed_fraction is plausible (0.50–0.98).  If not, try the OPPOSITE sign.
If that gives a plausible fraction, use it silently.  This makes the
builder robust to end_is_distal mismatches from any upstream source
without requiring changes to the pipeline glue code.
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def _ransac_curved_boundary(
    p_maj: np.ndarray,
    p_min: np.ndarray,
    boundary_proj: float,
    half_w: float,
    depth: float,
    distal_sign: int,
) -> np.ndarray:
    """Estimate curved boundary using RANSAC quadratic fit on near-boundary pixels.

    RANSAC improves robustness against specular highlights and mask noise.

    Fits the model  p_major = a * (p_minor ** 2) + b  to pixels within
    ±3 projection units of ``boundary_proj``.  Outliers (glare, mask
    artifacts) are automatically discarded by RANSAC before the final
    quadratic is evaluated on all nail pixels.

    Falls back to the original deterministic parabola if:
      - fewer than 20 pixels lie within the boundary band, or
      - sklearn is not installed, or
      - RANSAC raises any exception.

    Args:
        p_maj         : (N,) major-axis projections of all nail pixels
        p_min         : (N,) minor-axis projections of all nail pixels
        boundary_proj : scalar position of the boundary along the major axis
        half_w        : half nail width in projection units
        depth         : deterministic parabola depth (fallback)
        distal_sign   : +1 (distal at max) or −1 (distal at min)

    Returns:
        curved_boundary : (N,) float array representing the curved boundary
                          position for every nail pixel along the major axis.
                          Compare against p_maj using distal_sign to decide
                          bed membership.
    """
    # ── Deterministic fallback (original fixed parabola) ─────────────────────
    parabola_offset = depth * (p_min / half_w) ** 2
    fallback_curve  = (boundary_proj - parabola_offset
                       if distal_sign == 1
                       else boundary_proj + parabola_offset)

    # ── Collect pixels within the boundary band (±3 px) ────────────────────
    BAND_PX     = 3.0
    MIN_INLIERS = 20
    near        = np.abs(p_maj - boundary_proj) <= BAND_PX
    if near.sum() < MIN_INLIERS:
        return fallback_curve

    # ── RANSAC quadratic fit: p_major = a⋅p_minor² + b ────────────────────
    try:
        from sklearn.linear_model import RANSACRegressor

        X_near = (p_min[near] ** 2).reshape(-1, 1)  # feature : p_minor²
        y_near = p_maj[near]                         # target  : p_major

        ransac = RANSACRegressor(
            min_samples=max(0.5, MIN_INLIERS / int(near.sum())),
            residual_threshold=2.0,
            random_state=42,
            max_trials=100,
        )
        ransac.fit(X_near, y_near)

        a = float(ransac.estimator_.coef_[0])
        b = float(ransac.estimator_.intercept_)
        return a * (p_min ** 2) + b

    except Exception:
        return fallback_curve


def _compute_bed(
    p_maj: np.ndarray,
    p_min: np.ndarray,
    boundary_proj: float,
    half_w: float,
    depth: float,
    distal_sign: int,
) -> np.ndarray:
    """Return boolean bed mask for given distal_sign (pure geometry, no I/O).

    Uses RANSAC-fitted quadratic boundary when sufficient near-boundary pixels
    are available; falls back to the deterministic parabola otherwise.
    """
    curved_boundary = _ransac_curved_boundary(
        p_maj, p_min, boundary_proj, half_w, depth, distal_sign
    )
    if distal_sign == 1:
        return p_maj <= curved_boundary
    else:
        return p_maj >= curved_boundary


def _build_curved_bed_mask(
    nail_yx: np.ndarray,
    centroid: np.ndarray,
    major_axis: np.ndarray,
    minor_axis: np.ndarray,
    boundary_proj: float,
    nail_width_px: float,
    end_is_distal,               # bool OR int (+1/-1)
) -> np.ndarray:
    """Return boolean array (len = len(nail_yx)): True = nail bed side.

    Args:
        nail_yx       : (N, 2) pixel coordinates [row, col]
        centroid      : [x, y] PCA centroid
        major_axis    : proximal→distal unit vector
        minor_axis    : lateral unit vector
        boundary_proj : inner edge of the free-edge band in proj_major units
        nail_width_px : full width of the nail in pixels
        end_is_distal : True / +1 if max(proj_major) is the distal tip.
                        May be wrong if the pipeline doesn't propagate the
                        axis-flip correction from boundary_detection —
                        auto-corrected internally.
    """
    # ── Normalise to int sign ────────────────────────────────────────────────
    if isinstance(end_is_distal, bool):
        distal_sign = 1 if end_is_distal else -1
    else:
        distal_sign = 1 if int(end_is_distal) >= 0 else -1

    if boundary_proj is None or nail_width_px < 5.0:
        print("[BedMaskBuilder] No boundary found. Returning full mask as bed.")
        return np.ones(len(nail_yx), dtype=bool)

    nail_xy  = nail_yx[:, ::-1].astype(np.float64)
    centered = nail_xy - centroid
    p_maj    = centered @ major_axis
    p_min    = centered @ minor_axis

    half_w    = nail_width_px / 2.0
    max_depth = 0.08 * nail_width_px
    depth     = min(0.06 * half_w, max_depth)

    # ── Try given sign ───────────────────────────────────────────────────────
    in_bed       = _compute_bed(p_maj, p_min, boundary_proj, half_w, depth, distal_sign)
    bed_fraction = float(np.sum(in_bed)) / max(len(in_bed), 1)

    logger.debug("[BedMaskBuilder] min_proj: %s", p_maj.min())
    logger.debug("[BedMaskBuilder] max_proj: %s", p_maj.max())
    logger.debug("[BedMaskBuilder] boundary_proj: %s", boundary_proj)
    logger.debug("[BedMaskBuilder] distal_sign: %s", distal_sign)
    logger.debug("[BedMaskBuilder] bed_fraction: %s", bed_fraction)

    # ── Auto-correct if orientation is wrong ─────────────────────────────────
    # A bed_fraction < 0.50 means the boundary ended up on the proximal side,
    # which is only possible if distal_sign is flipped relative to the
    # boundary_proj coordinate.  Silently try the opposite sign.
    if not (0.50 <= bed_fraction <= 0.98):
        flipped_sign  = -distal_sign
        in_bed_flip   = _compute_bed(p_maj, p_min, boundary_proj, half_w, depth, flipped_sign)
        bed_frac_flip = float(np.sum(in_bed_flip)) / max(len(in_bed_flip), 1)

        if 0.50 <= bed_frac_flip <= 0.98:
            print(
                f"[BedMaskBuilder] Auto-corrected distal_sign: "
                f"{distal_sign} → {flipped_sign}  "
                f"(bed_fraction {bed_fraction:.2f} → {bed_frac_flip:.2f})"
            )
            return in_bed_flip

        # Neither sign gives a valid fraction → full mask fallback
        print(
            f"[BedMaskBuilder] WARNING: both signs give implausible bed_fraction "
            f"({bed_fraction:.2f} / {bed_frac_flip:.2f}). "
            "Returning full mask as bed."
        )
        return np.ones(len(nail_yx), dtype=bool)

    return in_bed


def compute_bed_geometry(
    nail_yx: np.ndarray,
    in_bed: np.ndarray,
    centroid: np.ndarray,
    major_axis: np.ndarray,
    minor_axis: np.ndarray,
    slice_step: float = 2.0,
    free_edge_present: bool = False,
    plate_length_px: float = 0.0,
    plate_width_px: float = 0.0,
) -> dict:
    """Compute robust nail bed length, width, and ratio from the bed mask.

    Median slice width provides stable nail bed width estimation.
    Projecting onto PCA axes and taking the median across axis-aligned slices
    prevents nail curvature and segmentation noise from shrinking the width
    estimate, which would otherwise inflate the bed ratio falsely.

    Proximal anchoring avoids distal taper bias in nail width measurement.
    Because nail beds taper toward the distal end, width is computed only from
    the 10 %–40 % band of the bed (min_proj + 10 % … min_proj + 40 % of bed_length).
    The first 10 % is excluded to avoid cuticle segmentation artifacts.
    The distal portion is still used for length calculation.

    Ensures correct geometry when free edge exists:
    - When free_edge_present is True  → nail_bed_ratio  is the primary ratio
      (computed from the bed mask only, excuding free-edge pixels).
    - When free_edge_present is False → nail_plate_ratio is the primary ratio
      (computed from the full nail plate dimensions supplied by the caller).
    Both values are always returned for debugging.

    Extreme width slices caused by segmentation noise are rejected before the
    median is computed: any slice narrower than 50 % of the raw median is
    discarded, then the median is recomputed on the clean set.

    Args:
        nail_yx           : (N, 2) pixel coordinates [row, col] for the full nail mask.
        in_bed            : (N,) boolean array — True where pixel belongs to the nail bed.
                            Must be the same length as nail_yx.
        centroid          : [x, y] PCA centroid (same coordinate system as nail_yx col/row).
        major_axis        : unit vector along the nail's long axis (proximal→distal).
        minor_axis        : unit vector perpendicular to major_axis.
        slice_step        : width of each projection slice in pixels (default 2 px).
        free_edge_present : True when a free edge has been detected; selects which
                            ratio is used as the primary measurement.
        plate_length_px   : full nail plate length in pixels (used when not free_edge_present).
        plate_width_px    : full nail plate width  in pixels (used when not free_edge_present).

    Returns:
        dict with keys:
            bed_length_px   – span of bed pixels along the major axis (float, px)
            bed_width_px    – median slice width after noise rejection   (float, px)
            nail_bed_ratio  – bed_length / (bed_width + 1e-6)
            nail_plate_ratio– plate_length / (plate_width + 1e-6)
            primary_ratio   – nail_bed_ratio if free_edge_present else nail_plate_ratio
            slice_count     – number of valid slices used for the width estimate
        Returns all-zero dict if fewer than 10 bed pixels are present.
    """
    _ZERO = {
        "bed_length_px":    0.0,
        "bed_width_px":     0.0,
        "nail_bed_ratio":   0.0,
        "nail_plate_ratio": 0.0,
        "primary_ratio":    0.0,
        "slice_count":      0,
    }

    # ── Nail plate ratio (always available when caller supplies dimensions) ───
    nail_plate_ratio = plate_length_px / (plate_width_px + 1e-6) if plate_width_px > 0 else 0.0

    # ── Select only bed pixels ────────────────────────────────────────────────
    bed_yx = nail_yx[in_bed]
    if len(bed_yx) < 10:
        _ZERO["nail_plate_ratio"] = round(nail_plate_ratio, 6)
        _ZERO["primary_ratio"]    = round(nail_plate_ratio, 6)
        return _ZERO

    # Convert (row, col) → (x, y) and centre on the PCA centroid
    bed_xy  = bed_yx[:, ::-1].astype(np.float64)
    centred = bed_xy - centroid

    # ── Project onto PCA axes (fully vectorised) ──────────────────────────────
    proj_major = centred @ major_axis   # shape (M,)
    proj_minor = centred @ minor_axis   # shape (M,)

    # ── Nail bed length ───────────────────────────────────────────────────────
    min_proj   = float(proj_major.min())
    max_proj   = float(proj_major.max())
    bed_length = max_proj - min_proj

    if bed_length < 1.0:
        _ZERO["nail_plate_ratio"] = round(nail_plate_ratio, 6)
        _ZERO["primary_ratio"]    = round(nail_plate_ratio, 6)
        return _ZERO

    # ── Proximal-region width estimation ─────────────────────────────────────
    # Avoids segmentation noise near cuticle while capturing stable nail width.
    # Width slices are computed only from the 10 %–40 % band of the bed:
    #   • The first 10 % (proximal edge) is excluded to avoid cuticle artifacts.
    #   • Beyond 40 % the nail bed begins to taper distally, biasing width down.
    # The full proj_major range is still used for bed_length.
    proximal_start = min_proj + 0.10 * bed_length
    proximal_end   = min_proj + 0.40 * bed_length
    prox_mask      = (proj_major >= proximal_start) & (proj_major <= proximal_end)
    prox_major     = proj_major[prox_mask]
    prox_minor     = proj_minor[prox_mask]

    # Slice the 10–40 % major-axis band into bins of `slice_step` pixels;
    # for each bin with ≥10 pixels, record the minor-axis span (slice width).
    # np.searchsorted assigns each pixel to a bin in O(M log B) — fully vectorised.
    bin_edges   = np.arange(proximal_start, proximal_end + slice_step, slice_step)
    bin_indices = np.searchsorted(bin_edges, prox_major)   # (K,) int

    slice_widths = []
    for b in range(len(bin_edges)):
        sel = prox_minor[bin_indices == b]
        if len(sel) < 10:
            continue  # ignore sparse / empty slices
        slice_widths.append(float(sel.max() - sel.min()))

    if not slice_widths:
        _ZERO["nail_plate_ratio"] = round(nail_plate_ratio, 6)
        _ZERO["primary_ratio"]    = round(nail_plate_ratio, 6)
        return _ZERO

    # ── Outlier rejection ─────────────────────────────────────────────────────
    # Reject extreme width slices caused by segmentation noise.
    # Any slice narrower than 50 % of the raw median is an artifact of a
    # ragged mask edge or a thin corruption strip; dropping it before the final
    # median prevents it from pulling the estimate downward.
    raw_median    = float(np.median(slice_widths))
    valid_widths  = [w for w in slice_widths if w > 0.5 * raw_median]
    if valid_widths:
        slice_widths = valid_widths  # recompute median on clean set

    # ── Robust bed width ──────────────────────────────────────────────────────
    bed_width = float(np.median(slice_widths))

    # ── Nail bed ratio ────────────────────────────────────────────────────────
    nail_bed_ratio = bed_length / (bed_width + 1e-6)

    # ── Primary ratio selection ───────────────────────────────────────────────
    # Ensures correct geometry when free edge exists:
    # use bed ratio when a free edge separates the bed from the plate tip;
    # fall back to the full-plate ratio when no free edge is detected.
    primary_ratio = nail_bed_ratio if free_edge_present else nail_plate_ratio

    return {
        "bed_length_px":    round(bed_length,        3),
        "bed_width_px":     round(bed_width,          3),
        "nail_bed_ratio":   round(nail_bed_ratio,     6),
        "nail_plate_ratio": round(nail_plate_ratio,   6),
        "primary_ratio":    round(primary_ratio,      6),
        "slice_count":      len(slice_widths),
    }