"""End-to-end nail-bed extraction pipeline.

Orchestrates PCA axis computation, boundary detection, and final measurement.

All public entry points:
- extract_geometry_nail_bed
- extract_geometry_nail_bed_with_diagnostics
- extract_nail_bed_overlay_data

Internal helper:
- _extract_nail_bed_internal

Stage ordering (matches production architecture)
-------------------------------------------------
STAGE 1  mask_processing._clean_mask_morphology
             largest-CC  →  MORPH_CLOSE  →  fill_holes  →  MORPH_OPEN
             →  remove-small-fragments  →  slight-erode

STAGE 2  axis_orientation._orient_anatomical_axis
             minAreaRect long axis, percentile-width taper, geometry only.

STAGE 3  Full-nail geometry computed FIRST (guaranteed fallback).
             length / width / ratio from full mask saved before any
             boundary detection is attempted.  Boundary failure can never
             collapse these values.

STAGE 4  boundary_detection.detect_free_edge_boundary
             Distal REGION classification (distal-30% vs central-40%),
             not a derivative scan.  Returns early with (None, 0.0) when
             the distal band is not structurally different from the nail
             body (trimmed nail).

STAGE 5  (inside detect_free_edge_boundary)
             Boundary placement restricted to distal band only.

What was removed and why
------------------------
- Canny, K-Means, Otsu boundary estimators: all fail under the same
  lighting conditions; voting agreement can be met by chance; better to
  use one robust method than three correlated weak ones.
- Centroid brightness check: whole-nail glare moves centroid to nail
  centre and the real free edge was rejected as glare.
- sigma=2 smoothing: shifts peak inward by ~5 slices; reduced to 1.
- delta_L > 0.25: too strict for naturally pink nails; lowered to 0.18.
"""

from typing import Tuple, Optional
import cv2
import numpy as np

from config import MIN_CONTOUR_AREA, NAIL_BED_VISUALIZE_BOUNDARY

from .mask_processing import _clean_mask_morphology
from .pca_utils import _pca_on_mask_pixels
from .axis_orientation import _orient_anatomical_axis
from .boundary_detection import detect_free_edge_boundary
from .bed_mask_builder import _build_curved_bed_mask


# ---------------------------------------------------------------------------
# Lateral nail fold detector
# ---------------------------------------------------------------------------

def _detect_lateral_folds(
    image: np.ndarray,
    nail_yx: np.ndarray,
    proj_major: np.ndarray,
    proj_minor: np.ndarray,
    centroid: np.ndarray,
    major_axis: np.ndarray,
    minor_axis: np.ndarray,
    mask: np.ndarray,
    min_proj: float,
    bed_length: float,
    gradient_threshold: float = 10.0,
    min_edge_points: int = 20,
    slice_step: float = 2.0,
) -> Optional[float]:
    """Detect lateral nail folds and return anatomical nail width.

    Lateral nail folds define anatomical nail width more accurately than mask
    edges because the Sobel response captures the true dermis–nail boundary
    independent of segmentation mask quality.

    Edge pixels are restricted to a 10–20 px ring **outside** the nail mask
    boundary.  This prevents internal nail texture (ridges, glare, polish
    patterns) from being mistaken for lateral fold edges.

    Algorithm
    ---------
    1. Compute horizontal Sobel gradient (vertical edge response) on the
       grayscale / L-channel image.
    2. Build a boundary band: pixels 10–20 px outside the nail mask by
       dilating twice and subtracting (outer_dil − inner_dil).
    3. Project boundary-band pixels onto the PCA major / minor axes.
    4. Restrict analysis to the proximal 10 %–40 % band along the major axis
       (avoids cuticle noise at the proximal extreme and distal taper).
    5. For each 2-px slice in this band, locate the left- and right-most pixel
       whose gradient magnitude exceeds *gradient_threshold*.
    6. Require at least *min_edge_points* on each side; fall back to None
       (mask-based width) if too few are found.
    7. Fit a least-squares line through each side’s points (minor ≈ a*major + b).
    8. Return the mean distance between the two fitted lines evaluated at every
       sampled major-axis position.

    Args:
        image              : BGR input image (uint8).
        nail_yx            : (N, 2) pixel coordinates [row, col] for the full nail mask
                             (used only for its spatial extent / band-limit check).
        proj_major         : (N,) projections of nail mask pixels onto the major axis.
        proj_minor         : (N,) projections of nail mask pixels onto the minor axis.
        centroid           : [x, y] PCA centroid used to project boundary pixels.
        major_axis         : unit vector along the nail’s long axis.
        minor_axis         : unit vector perpendicular to major_axis.
        mask               : binary nail mask (uint8, 255 = nail).
        min_proj           : minimum value of proj_major (proximal end).
        bed_length         : span of proj_major used to define the proximal band.
        gradient_threshold : minimum absolute Sobel-x magnitude to count as an edge.
        min_edge_points    : minimum valid edge points required per side.
        slice_step         : major-axis slice width in pixels (default 2 px).

    Returns:
        Anatomical nail width in pixels, or None if detection fails.
    """
    # ── Step 1: Sobel-x gradient on grayscale / L channel ────────────────────
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    sobel_x  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_abs = np.abs(sobel_x)   # absolute horizontal gradient magnitude

    # ── Step 2: Build boundary band 10–20 px outside the nail mask ───────────
    # Only accept edge pixels within 10–20 pixels outside the nail mask so that
    # internal nail texture (ridges, glare) cannot be mistaken for fold edges.
    _k_inner = np.ones((21, 21), np.uint8)   # dilate by ~10 px
    _k_outer = np.ones((41, 41), np.uint8)   # dilate by ~20 px
    inner_dil      = cv2.dilate(mask, _k_inner, iterations=1)
    outer_dil      = cv2.dilate(mask, _k_outer, iterations=1)
    boundary_band  = cv2.subtract(outer_dil, inner_dil)   # ring 10–20 px outside

    # Get coordinates of boundary-band pixels and project onto PCA axes.
    band_yx  = np.argwhere(boundary_band > 0)
    if len(band_yx) < min_edge_points * 2:
        return None

    band_xy  = band_yx[:, ::-1].astype(np.float64)
    band_cen = band_xy - centroid
    b_major  = band_cen @ major_axis
    b_minor  = band_cen @ minor_axis
    b_grad   = grad_abs[band_yx[:, 0], band_yx[:, 1]]

    # ── Step 4: Restrict to proximal 10 %–40 % band ─────────────────────────
    prox_start = min_proj + 0.10 * bed_length
    prox_end   = min_proj + 0.40 * bed_length
    band_mask  = (b_major >= prox_start) & (b_major <= prox_end)

    b_major = b_major[band_mask]
    b_minor = b_minor[band_mask]
    b_grad  = b_grad[band_mask]

    if len(b_major) < min_edge_points * 2:
        return None   # not enough pixels in the proximal band

    # ── Step 5: Find strongest left and right edges per slice ──────────────
    bin_edges   = np.arange(prox_start, prox_end + slice_step, slice_step)
    bin_indices = np.searchsorted(bin_edges, b_major)

    left_pts:  list = []   # (major, minor) of left-fold edge per slice
    right_pts: list = []   # (major, minor) of right-fold edge per slice

    for b in range(len(bin_edges)):
        sel = bin_indices == b
        if sel.sum() < 10:
            continue   # too sparse

        sl_major = b_major[sel]
        sl_minor = b_minor[sel]
        sl_grad  = b_grad[sel]

        strong = sl_grad >= gradient_threshold
        if strong.sum() < 2:
            continue   # no strong edges in this slice

        s_minor = sl_minor[strong]
        s_major = sl_major[strong]

        # Representative major-axis position for this slice
        maj_pos = float(np.mean(s_major))

        # Left edge = smallest minor-axis position with strong gradient
        # Right edge = largest minor-axis position with strong gradient
        left_pts.append((maj_pos, float(s_minor.min())))
        right_pts.append((maj_pos, float(s_minor.max())))

    if len(left_pts) < min_edge_points or len(right_pts) < min_edge_points:
        return None   # ── fall back to mask-based width

    # ── Step 7: Fit least-squares lines through left/right edge points ──────
    # Model: minor = a * major + b  →  solved via np.polyfit(major, minor, 1)
    l_maj = np.array([p[0] for p in left_pts])
    l_min = np.array([p[1] for p in left_pts])
    r_maj = np.array([p[0] for p in right_pts])
    r_min = np.array([p[1] for p in right_pts])

    l_coef = np.polyfit(l_maj, l_min, 1)   # [slope, intercept] for left fold
    r_coef = np.polyfit(r_maj, r_min, 1)   # [slope, intercept] for right fold

    # ── Step 8: Average distance between fold lines ──────────────────────────
    # Evaluate both lines at every sampled major-axis position and average.
    all_maj   = np.concatenate([l_maj, r_maj])
    maj_range = np.linspace(all_maj.min(), all_maj.max(), 50)

    l_vals = np.polyval(l_coef, maj_range)
    r_vals = np.polyval(r_coef, maj_range)

    widths = np.abs(r_vals - l_vals)
    fold_width = float(np.mean(widths))

    return fold_width if fold_width > 1.0 else None


def _extract_nail_bed_internal(
    image: np.ndarray,
    mask: np.ndarray,
    is_polished: bool = False,
    debug: bool = False,
    lab_frame: Optional[np.ndarray] = None,
    precomputed_axes: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[
    Optional[float], Optional[float], Optional[float], Optional[bool],
    float, Optional[float], int, Optional[float], bool,
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool
]:
    """Core nail bed extraction pipeline.

    Returns
    -------
    length_px, width_px, area_px, free_edge_present,
    free_edge_confidence, mid_width_px, n_methods,
    boundary_proj, end_is_distal,
    centroid, major_axis, minor_axis, brightness_used
    """
    if mask is None or mask.sum() == 0:
        return None, None, None, None, 0.0, None, 0, None, True, None, None, None, False

    # ── STAGE 1: mask cleaning ──────────────────────────────────────────────
    mask_clean = _clean_mask_morphology(mask)

    contours, _ = cv2.findContours(
        mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None, None, None, 0.0, None, 0, None, True, None, None, None, False
    if cv2.contourArea(max(contours, key=cv2.contourArea)) < MIN_CONTOUR_AREA:
        return None, None, None, None, 0.0, None, 0, None, True, None, None, None, False

    # ── STAGE 2: stable anatomical axes ────────────────────────────────────
    if precomputed_axes is not None:
        centroid, major_axis, minor_axis = precomputed_axes
    else:
        centroid, major_axis, minor_axis = _pca_on_mask_pixels(mask_clean, sample_rate=0.3)
        major_axis, minor_axis, _ = _orient_anatomical_axis(
            image, mask_clean, centroid, major_axis, minor_axis, lab_frame=lab_frame
        )

    nail_yx    = np.argwhere(mask_clean == 255)
    nail_xy    = nail_yx[:, ::-1].astype(np.float64)
    centered   = nail_xy - centroid
    proj_major = centered @ major_axis
    proj_minor = centered @ minor_axis

    # Precompute LAB once.
    lab_img = (lab_frame if lab_frame is not None
               else cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32))

    _full_nail_width_px = float(proj_minor.max() - proj_minor.min())

    # ── STAGE 3: full-nail geometry FIRST (guaranteed fallback) ────────────
    #
    # Compute length / width / area from the FULL mask right now, before any
    # boundary detection.  If boundary detection fails or is skipped, these
    # values are returned unchanged.  Boundary failure can never collapse the
    # ratio to zero or NaN.
    full_length_px = float(proj_major.max() - proj_major.min())
    full_width_px  = _full_nail_width_px
    full_area_px   = float(nail_yx.shape[0])

    if debug:
        print(
            f"  [NailBed] Full-nail geometry (fallback):  "
            f"length={full_length_px:.1f}px  width={full_width_px:.1f}px  "
            f"ratio={full_length_px / max(full_width_px, 1):.2f}"
        )

    # Initialise outputs to full-nail values; will be overwritten on success.
    length_px            = full_length_px
    width_px             = full_width_px
    area_px              = full_area_px
    n_methods            = 0
    boundary_proj        = None
    end_is_distal        = True
    free_edge_present    = False
    free_edge_confidence = 0.0
    in_bed               = np.ones(len(proj_major), dtype=bool)

    # ── STAGE 4 + 5: boundary detection ────────────────────────────────────
    if is_polished:
        # Polished nail: use brightness to orient, then apply 87% anatomical prior.
        _L_pol    = lab_img[:, :, 0][nail_yx[:, 0], nail_yx[:, 1]]
        _sort_idx = np.argsort(proj_major)
        _q        = max(2, len(_sort_idx) // 4)
        _L_start  = float(np.mean(_L_pol[_sort_idx[:_q]]))
        _L_end    = float(np.mean(_L_pol[_sort_idx[-_q:]]))
        end_is_distal = _L_end >= _L_start

        _span = full_length_px
        if end_is_distal:
            _cutoff = float(proj_major.min()) + 0.87 * _span
            in_bed  = proj_major <= _cutoff
        else:
            _cutoff = float(proj_major.max()) - 0.87 * _span
            in_bed  = proj_major >= _cutoff

        free_edge_present    = None
        free_edge_confidence = 0.3
        if debug:
            print(
                f"  [NailBed] Polished nail — anatomical prior (87%) applied  "
                f"(distal at {'max' if end_is_distal else 'min'} proj)"
            )

    else:
        _bd_proj, _bd_conf, _end_is_distal = detect_free_edge_boundary(
            nail_yx, proj_major, proj_minor, lab_img, debug=debug,
            major_axis=major_axis,
        )

        if _bd_proj is not None:
            boundary_proj        = _bd_proj
            free_edge_confidence = _bd_conf
            free_edge_present    = True
            end_is_distal        = _end_is_distal
            n_methods            = 1

            in_bed = _build_curved_bed_mask(
                nail_yx, centroid, major_axis, minor_axis,
                boundary_proj, _full_nail_width_px, end_is_distal,
            )

            # Sanity: if curved mask removed almost everything, revert to full mask.
            bed_fraction = float(in_bed.sum()) / max(len(in_bed), 1)
            if bed_fraction < 0.10:
                if debug:
                    print(
                        f"  [NailBed] bed_fraction={bed_fraction:.3f} < 0.10 "
                        "— boundary rejected, reverting to full-nail geometry"
                    )
                in_bed               = np.ones(len(proj_major), dtype=bool)
                free_edge_present    = False
                free_edge_confidence = 0.0
                boundary_proj        = None
            elif in_bed.sum() < 10:
                if debug:
                    print("  [NailBed] Curved boundary too aggressive — full mask used")
                in_bed               = np.ones(len(proj_major), dtype=bool)
                free_edge_confidence *= 0.5
        else:
            # No boundary: full mask is the nail bed (trimmed nail).
            # in_bed and length/width/area already set to full-nail values above.
            if debug:
                print("  [NailBed] No free edge detected — full nail mask used (trimmed nail)")

    # ── Geometry measurement on nail-bed pixels ─────────────────────────────
    nb_yx       = nail_yx[in_bed]
    nb_xy       = nb_yx[:, ::-1].astype(np.float64)
    nb_centered = nb_xy - centroid

    anatomical_axis = major_axis
    perp_axis       = np.array([-anatomical_axis[1], anatomical_axis[0]])
    proj_len        = nb_centered @ anatomical_axis
    proj_wid        = nb_centered @ perp_axis

    length_px = float(proj_len.max() - proj_len.min())
    width_px  = float(proj_wid.max() - proj_wid.min())
    # ratio is computed by callers from length/width

    # ── Lateral fold width refinement ────────────────────────────────────────
    # Lateral nail folds define anatomical nail width more accurately than mask
    # edges.  Attempt fold detection on the proximal 10-40% band of the nail.
    # Fall back silently to the mask-based width_px when detection fails.
    _min_nail_proj = float(proj_major.min())
    _nail_length   = float(proj_major.max() - proj_major.min())
    _fold_width = _detect_lateral_folds(
        image, nail_yx,
        proj_major, proj_minor,
        centroid, major_axis, minor_axis,
        mask_clean,
        _min_nail_proj, _nail_length,
    )
    if _fold_width is not None:
        if debug:
            print(
                f"  [NailBed] Lateral fold width: {_fold_width:.1f}px "
                f"(mask width was {width_px:.1f}px)"
            )
        width_px = _fold_width

    # Midpoint width (P2-P98 in central 10% band).
    _mid_proj     = float(np.median(proj_len))
    _axis_span_nb = float(proj_len.max() - proj_len.min())
    _in_mid_band  = np.abs(proj_len - _mid_proj) < (_axis_span_nb * 0.05)
    if _in_mid_band.sum() >= 10:
        _mid_width_px = float(
            np.percentile(proj_wid[_in_mid_band], 98)
            - np.percentile(proj_wid[_in_mid_band], 2)
        )
    else:
        _mid_width_px = width_px

    # Area by pixel count.
    nail_bed_mask                            = np.zeros_like(mask_clean)
    nail_bed_mask[nb_yx[:, 0], nb_yx[:, 1]] = 255
    area_px = float(np.sum(nail_bed_mask == 255))

    # Optional boundary visualisation.
    if NAIL_BED_VISUALIZE_BOUNDARY and image is not None and not is_polished:
        if in_bed.sum() < len(proj_len):
            boundary_xy = centroid + float(proj_len.max()) * anatomical_axis
            half_w      = width_px / 2.0
            pt1 = tuple((boundary_xy + half_w * perp_axis).astype(int))
            pt2 = tuple((boundary_xy - half_w * perp_axis).astype(int))
            cv2.line(image, pt1, pt2, (0, 255, 255), 2)

    if debug:
        print(
            f"  [NailBed] length={length_px:.1f}px  width={width_px:.1f}px  "
            f"ratio={length_px / max(width_px, 1):.2f}  area={area_px:.0f}px²  "
            f"free_edge={free_edge_present}  conf={free_edge_confidence:.2f}"
        )

    return (
        length_px, width_px, area_px,
        free_edge_present, free_edge_confidence,
        _mid_width_px, n_methods,
        boundary_proj, end_is_distal,
        centroid, major_axis, minor_axis,
        False,   # brightness_used field kept for API compatibility
    )


# ---------------------------------------------------------------------------
# Public entry points (unchanged API)
# ---------------------------------------------------------------------------

def extract_geometry_nail_bed(
    image: np.ndarray,
    mask: np.ndarray,
    debug: bool = False,
    is_polished: bool = False,
    lab_frame: Optional[np.ndarray] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract nail bed geometry.  Backward-compatible wrapper."""
    length_px, width_px, area_px, _, _, _, _, _, _, _, _, _, _ = (
        _extract_nail_bed_internal(
            image, mask, is_polished=is_polished, debug=debug, lab_frame=lab_frame
        )
    )
    return length_px, width_px, area_px


def extract_geometry_nail_bed_with_diagnostics(
    image: np.ndarray,
    mask: np.ndarray,
    verbose: bool = False,
    is_polished: bool = False,
    lab_frame: Optional[np.ndarray] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[dict]]:
    """Extract nail bed geometry with a full diagnostics dictionary."""
    (length_px, width_px, area_px, free_edge_present, free_edge_confidence,
     _mid_width_px, _n_methods, _boundary_proj, _end_is_distal,
     _centroid, _major_axis, _minor_axis, _brightness_used) = (
        _extract_nail_bed_internal(
            image, mask, is_polished=is_polished, debug=verbose, lab_frame=lab_frame
        )
    )
    diagnostics = {
        "method":                  "region_classify_then_local_boundary",
        "length_px":               length_px,
        "width_px":                width_px,
        "width_midpoint_px":       _mid_width_px,
        "area_px":                 area_px,
        "aspect_ratio":            (
            round(length_px / width_px, 3)
            if (length_px and width_px and width_px > 0)
            else None
        ),
        "free_edge_present":       free_edge_present,
        "free_edge_confidence":    free_edge_confidence,
        "is_polished":             is_polished,
        "boundary_methods_agreed": _n_methods,
        "boundary_confidence":     free_edge_confidence,
        "boundary_proj":           _boundary_proj,
        "end_is_distal":           _end_is_distal,
    }
    return length_px, width_px, area_px, diagnostics


def extract_nail_bed_overlay_data(
    image: np.ndarray,
    mask: np.ndarray,
    is_polished: bool = False,
    debug: bool = False,
    lab_frame: Optional[np.ndarray] = None,
    precomputed_axes: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> dict:
    """Return everything needed to draw a nail-bed overlay for debugging."""
    result = {
        "nail_bed_mask":        mask.copy(),
        "boundary_proj":        None,
        "end_is_distal":        True,
        "free_edge_present":    None,
        "free_edge_confidence": 0.0,
        "centroid":             None,
        "major_axis":           None,
        "minor_axis":           None,
        "length_px":            0.0,
        "width_px":             0.0,
        "area_px":              0.0,
        "width_midpoint_px":    None,
        "full_nail_width_px":   0.0,
        "n_methods":            0,
    }

    if mask is None or mask.sum() == 0:
        return result

    (length_px, width_px, area_px, free_edge_present,
     free_edge_confidence, mid_width_px, n_methods,
     boundary_proj, end_is_distal,
     centroid, major_axis, minor_axis,
     brightness_method_used) = _extract_nail_bed_internal(
        image, mask,
        is_polished=is_polished,
        debug=debug,
        lab_frame=lab_frame,
        precomputed_axes=precomputed_axes,
    )

    if centroid is None or major_axis is None:
        return result

    mask_clean = _clean_mask_morphology(mask)
    nail_yx    = np.argwhere(mask_clean == 255)
    nail_xy    = nail_yx[:, ::-1].astype(np.float64)
    centered   = nail_xy - centroid
    proj_minor = centered @ minor_axis

    full_nail_width_px = float(proj_minor.max() - proj_minor.min())

    if boundary_proj is not None:
        in_bed = _build_curved_bed_mask(
            nail_yx, centroid, major_axis, minor_axis,
            boundary_proj, full_nail_width_px, end_is_distal,
        )
        nb_mask = np.zeros_like(mask_clean)
        if in_bed.sum() >= 10:
            nb_mask[nail_yx[in_bed][:, 0], nail_yx[in_bed][:, 1]] = 255
            # Morphological erosion to remove sliver after boundary nudge
            import cv2
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            nb_mask = cv2.erode(nb_mask, erode_kernel, iterations=1)
        else:
            nb_mask       = mask_clean.copy()
            boundary_proj = None
    else:
        nb_mask = mask_clean.copy()

    result.update({
        "nail_bed_mask":          nb_mask,
        "boundary_proj":          boundary_proj,
        "end_is_distal":          end_is_distal,
        "free_edge_present":      free_edge_present,
        "free_edge_confidence":   free_edge_confidence,
        "centroid":               centroid,
        "major_axis":             major_axis,
        "minor_axis":             minor_axis,
        "length_px":              length_px or 0.0,
        "width_px":               width_px or full_nail_width_px,
        "area_px":                area_px or 0.0,
        "width_midpoint_px":      mid_width_px,
        "full_nail_width_px":     full_nail_width_px,
        "n_methods":              n_methods,
        "brightness_method_used": brightness_method_used,
    })
    return result