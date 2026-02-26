from .boundary_detection import _detect_nail_bed_boundary_canny
"""End-to-end nail-bed extraction pipeline.

Orchestrates PCA axis computation, multi-method boundary voting, and
final measurement.  All public entry points live here:

- ``extract_geometry_nail_bed``
- ``extract_geometry_nail_bed_with_diagnostics``
- ``extract_nail_bed_overlay_data``

Internal helpers:
- ``_detect_free_edge_by_brightness``
- ``_boundary_from_free_edge_mask``
- ``_extract_nail_bed_internal``
"""

from typing import Tuple, Optional
import cv2
import numpy as np

from config import MIN_CONTOUR_AREA, NAIL_BED_VISUALIZE_BOUNDARY

from .mask_processing import _clean_mask_morphology
from .pca_utils import _pca_on_mask_pixels
from .axis_orientation import _orient_anatomical_axis
from .boundary_estimators import (
    _kmeans_boundary_estimate,
    _otsu_boundary_estimate,
    _canny_boundary_estimate,
)
from .boundary_validation import (
    _otsu_validate_color_boundary,
    _canny_validate_edge_at_boundary,
)
from .bed_mask_builder import _build_curved_bed_mask


# ---------------------------------------------------------------------------
# Brightness-based free-edge detection
# ---------------------------------------------------------------------------

def _detect_free_edge_by_brightness(
    nail_yx: np.ndarray,
    mask_clean: np.ndarray,
    lab_img: np.ndarray,
    debug: bool = False,
) -> Tuple[Optional[np.ndarray], float]:
    """Detect free edge as the brightest connected region at one end of the nail.

    The free edge is anatomically and optically distinct from the nail bed:
    - Free edge: white/translucent keratin, no vasculature, high LAB-L (>165)
    - Nail bed: pink/beige, vascularised, moderate LAB-L (120-165)
        # --- Polish detection gate (before free edge logic) ---
        S_vals = hsv_img[:, :, 1][nail_yx[:, 0], nail_yx[:, 1]]
        L_vals = lab_img[:, :, 0][nail_yx[:, 0], nail_yx[:, 1]]
        a_vals = lab_img[:, :, 1][nail_yx[:, 0], nail_yx[:, 1]]
        mean_S = float(np.mean(S_vals))
        std_L = float(np.std(L_vals))
        mean_a = float(np.mean(a_vals))
        is_polished_detected = (
            mean_S > 80 or abs(mean_a) > 20 or (std_L < 5 and mean_S > 50)
        )
        if is_polished_detected:
            if debug:
                print(f"  [PolishDetect] mean_S={mean_S:.1f} std_L={std_L:.1f} mean_a={mean_a:.1f} — classified as polished, skipping boundary detection.")
            free_edge_present = False
            free_edge_confidence = 0.0
            boundary_proj = None
            end_is_distal = True
            n_methods = 0
            nail_bed_mask = mask_clean.copy()
            length_px = float(np.sum(mask_clean == 255))
            width_px = float(np.sum(mask_clean == 255))
            area_px = float(np.sum(mask_clean == 255))
            _mid_width_px = width_px
            centroid = centroid
            major_axis = major_axis
            minor_axis = minor_axis
            brightness_used = False
            return length_px, width_px, area_px, free_edge_present, free_edge_confidence, _mid_width_px, n_methods, boundary_proj, end_is_distal, centroid, major_axis, minor_axis, True
            if _final_boundary is not None:
                # Cross-validate with Otsu colour and Canny edge checks
                _otsu_mult = _otsu_validate_color_boundary(
                    nail_yx, proj_major, lab_img, _final_boundary, debug=debug
                )
                _canny_mult = _canny_validate_edge_at_boundary(
                    _canny_edges_img, nail_yx, proj_major, proj_minor,
                    _final_boundary,
                )
                free_edge_confidence = min(
                    1.0, free_edge_confidence * _otsu_mult * _canny_mult
                )

                in_bed = _build_curved_bed_mask(
                    nail_yx, centroid, major_axis, minor_axis,
                    _final_boundary, _full_nail_width_px, end_is_distal,
                )
                bed_fraction = np.sum(in_bed) / len(in_bed)
                free_edge_fraction = 1.0 - bed_fraction
                # Final stable classification rule
                if (
                    free_edge_confidence >= 0.5
                    and free_edge_fraction >= 0.02
                    and not is_polished_detected
                ):
                    free_edge_present = True
                    boundary_proj     = _final_boundary
                else:
                    free_edge_present = False
                    boundary_proj     = None
                    in_bed = np.ones(len(proj_major), dtype=bool)
                    if debug:
                        print(f"  [Voting] Free edge rejected: conf={free_edge_confidence:.2f}, free_edge_fraction={free_edge_fraction:.3f}, is_polished={is_polished_detected}")
    """
    if len(nail_yx) < 50:
        return None, 0.0

    L_vals = lab_img[:, :, 0][nail_yx[:, 0], nail_yx[:, 1]].astype(np.float32)

    # --- Step 1: Adaptive brightness threshold ---
    L_uint8 = np.clip(L_vals, 0, 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(
        L_uint8.reshape(-1, 1),
        0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thresh_val = float(thresh_val)

    FREE_EDGE_MIN_L = np.percentile(L_vals, 65)
    thresh_val = max(thresh_val, FREE_EDGE_MIN_L)

    is_bright = L_vals > thresh_val
    bright_fraction = float(is_bright.sum()) / max(len(L_vals), 1)

    if debug:
        print(f"  [BrightFE] Otsu thresh={thresh_val:.0f}  "
              f"bright_fraction={bright_fraction:.2f}")

    # Free edge is 5–55% of nail area.
    # The original 40% cap rejected long nails where the free edge is prominent.
    # Raised to 55% to handle nails where the free edge dominates the visible area.
    if not (0.05 <= bright_fraction <= 0.70):
        if debug:
            print(f"  [BrightFE] bright_fraction {bright_fraction:.2f} "
                  "out of [0.05, 0.55] range -- no free edge")
        return None, 0.0

    # --- Step 2: Spatial constraint — bright region must be at ONE END ---
    min_y = int(nail_yx[:, 0].min())
    min_x = int(nail_yx[:, 1].min())
    max_y = int(nail_yx[:, 0].max())
    max_x = int(nail_yx[:, 1].max())

    local_h = max_y - min_y + 1
    local_w = max_x - min_x + 1

    bright_local = np.zeros((local_h, local_w), dtype=np.uint8)
    bright_yx = nail_yx[is_bright]
    bright_local[bright_yx[:, 0] - min_y, bright_yx[:, 1] - min_x] = 255

    _k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_closed = cv2.morphologyEx(bright_local, cv2.MORPH_CLOSE, _k)

    n_comp, labels_img, stats, _ = cv2.connectedComponentsWithStats(
        bright_closed, connectivity=8
    )

    if n_comp <= 1:
        return None, 0.0

    largest_comp = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp_area = stats[largest_comp, cv2.CC_STAT_AREA]

    if comp_area < 0.60 * bright_fraction * len(L_vals):
        if debug:
            print(f"  [BrightFE] Bright region fragmented -- likely glare, not free edge")
        return None, 0.0

    comp_mask_local = (labels_img == largest_comp)
    comp_yx_local = np.argwhere(comp_mask_local)
    comp_centroid_local = comp_yx_local.mean(axis=0)

    nail_centroid_local = np.array([
        nail_yx[:, 0].mean() - min_y,
        nail_yx[:, 1].mean() - min_x,
    ])

    nail_extent_y = float(local_h)
    nail_extent_x = float(local_w)

    dist_from_nail_center_y = abs(
        comp_centroid_local[0] - nail_centroid_local[0]
    ) / max(nail_extent_y / 2.0, 1.0)
    dist_from_nail_center_x = abs(
        comp_centroid_local[1] - nail_centroid_local[1]
    ) / max(nail_extent_x / 2.0, 1.0)
    dist_from_center = max(dist_from_nail_center_y, dist_from_nail_center_x)

    if dist_from_center < 0.30:
        if debug:
            print(f"  [BrightFE] Bright region is central "
                  f"(dist={dist_from_center:.2f}) -- glare, not free edge")
        return None, 0.0

    # --- Step 3: Map bright component back to nail_yx indices ---
    free_edge_mask = np.zeros(len(nail_yx), dtype=bool)
    for idx, (r, c) in enumerate(nail_yx):
        local_r = r - min_y
        local_c = c - min_x
        if (0 <= local_r < local_h and 0 <= local_c < local_w
                and comp_mask_local[local_r, local_c]):
            free_edge_mask[idx] = True

    if free_edge_mask.sum() < 10:
        return None, 0.0

    # --- Step 4: Confidence score ---
    L_free = float(np.mean(L_vals[free_edge_mask]))
    L_bed  = float(np.mean(L_vals[~free_edge_mask]))
    L_contrast = L_free - L_bed

    contrast_score = min(1.0, L_contrast / 40.0)
    size_score = 1.0 - abs(bright_fraction - 0.20) / 0.20
    size_score = max(0.0, min(1.0, size_score))
    spatial_score = min(1.0, (dist_from_center - 0.30) / 0.40)

    confidence = 0.5 * contrast_score + 0.3 * spatial_score + 0.2 * size_score

    if debug:
        print(f"  [BrightFE] L_free={L_free:.0f}  L_bed={L_bed:.0f}  "
              f"contrast={L_contrast:.0f}  dist={dist_from_center:.2f}  "
              f"conf={confidence:.2f}")

    # --- Step 4b: Angular constraint ---
    _nail_mask_local = np.zeros((local_h, local_w), dtype=np.uint8)
    _nail_mask_local[
        nail_yx[:, 0] - min_y,
        nail_yx[:, 1] - min_x
    ] = 255
    _contours, _ = cv2.findContours(
        _nail_mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if _contours:
        _cnt = max(_contours, key=cv2.contourArea)
        _rect = cv2.minAreaRect(_cnt)
        _rw, _rh = _rect[1]
        _geo_ratio = max(_rw, _rh) / max(min(_rw, _rh), 1.0)

        if _geo_ratio > 1.15:
            _angle_deg = _rect[2]
            if _rw < _rh:
                _angle_deg += 90.0
            _angle_rad = np.deg2rad(_angle_deg)
            _geo_long_axis = np.array([np.cos(_angle_rad), np.sin(_angle_rad)])

            _nail_centroid_local_xy = np.array([
                nail_centroid_local[1],
                nail_centroid_local[0],
            ])
            _free_centroid_xy = np.array([
                comp_centroid_local[1],
                comp_centroid_local[0],
            ])
            _fe_dir = _free_centroid_xy - _nail_centroid_local_xy
            _fe_dir_len = float(np.linalg.norm(_fe_dir))

            if _fe_dir_len > 1.0:
                _fe_dir_unit = _fe_dir / _fe_dir_len
                _alignment = abs(float(np.dot(_fe_dir_unit, _geo_long_axis)))
                if _alignment < 0.70:
                    if debug:
                        print(f"  [BrightFE] Free edge direction misaligned with "
                              f"nail long axis (alignment={_alignment:.2f} < 0.70) "
                              f"-- likely side highlight, rejected")
                    return None, 0.0
                confidence *= (0.5 + 0.5 * _alignment)

    return free_edge_mask, float(confidence)


# ---------------------------------------------------------------------------
# Convert free-edge pixel mask to a single boundary projection
# ---------------------------------------------------------------------------

def _boundary_from_free_edge_mask(
    nail_yx: np.ndarray,
    free_edge_mask: np.ndarray,
    centroid: np.ndarray,
    major_axis: np.ndarray,
    minor_axis: np.ndarray,
    debug: bool = False,
) -> Tuple[Optional[float], bool, np.ndarray, np.ndarray]:
    """Convert a free edge pixel mask to a single boundary projection value.

    Once _detect_free_edge_by_brightness has identified WHICH pixels are free
    edge, this function converts that to a single scalar boundary_proj value
    compatible with the rest of the pipeline (_build_curved_bed_mask, etc.).

    Instead of relying on the PCA major axis (which can flip), the function
    derives a stable anatomical axis from the vector pointing from the nail bed
    centroid toward the free edge centroid (proximal -> distal).

    Returns:
        boundary_proj     : float or None
        end_is_distal     : bool
        anatomical_major  : np.ndarray -- corrected major axis (proximal->distal)
        anatomical_minor  : np.ndarray -- corrected minor axis (width)
    """
    nail_xy  = nail_yx[:, ::-1].astype(np.float64)
    centered = nail_xy - centroid

    proj_bed_region  = nail_xy[~free_edge_mask]
    proj_free_region = nail_xy[free_edge_mask]

    if len(proj_free_region) < 5 or len(proj_bed_region) < 5:
        return None, True, major_axis, minor_axis

    # -- Step 1: Compute true proximal->distal axis from region centroids --
    bed_centroid  = proj_bed_region.mean(axis=0)
    free_centroid = proj_free_region.mean(axis=0)

    axis_vec = free_centroid - bed_centroid
    axis_len = float(np.linalg.norm(axis_vec))

    if axis_len < 5.0:
        return None, True, major_axis, minor_axis

    anatomical_major = axis_vec / axis_len
    anatomical_minor = np.array([-anatomical_major[1], anatomical_major[0]])

    # -- Step 2: Verify this axis makes geometric sense --
    proj_all  = centered @ anatomical_major
    proj_free = proj_all[free_edge_mask]
    proj_bed  = proj_all[~free_edge_mask]

    free_median = float(np.median(proj_free))
    bed_median  = float(np.median(proj_bed))
    separation  = abs(free_median - bed_median)
    nail_span   = float(proj_all.max() - proj_all.min())

    if separation < 0.15 * nail_span:
        if debug:
            print(f"  [BoundaryFE] Separation too small ({separation:.1f}px < "
                  f"15% of {nail_span:.1f}px) -- axis rejected")
        return None, True, major_axis, minor_axis

    # -- Step 3: end_is_distal --
    end_is_distal = True  # by construction: axis points proximal->distal

    # -- Step 4: Boundary projection = proximal edge of free edge region --
    boundary_proj = float(np.percentile(proj_free, 10))

    if debug:
        print(f"  [BoundaryFE] bed_centroid={bed_centroid}  "
              f"free_centroid={free_centroid}  "
              f"boundary_proj={boundary_proj:.1f}  "
              f"separation={separation:.1f}px  "
              f"anatomical_axis={anatomical_major}")

    return boundary_proj, end_is_distal, anatomical_major, anatomical_minor


# ---------------------------------------------------------------------------
# Core internal pipeline
# ---------------------------------------------------------------------------

def _extract_nail_bed_internal(
    image: np.ndarray,
    mask: np.ndarray,
    is_polished: bool = False,
    debug: bool = False,
    lab_frame: Optional[np.ndarray] = None,
    precomputed_axes: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[bool], float, Optional[float], int, Optional[float], bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Core nail bed extraction pipeline.

    Performs PCA, free-edge classification, optional boundary crop, and
    measurement in a single pass.  Never applies a fixed-percentage fallback
    crop -- decisions are driven entirely by LAB gradient classification.

    Returns:
        length_px, width_px, area_px, free_edge_present,
        free_edge_confidence, mid_width_px, n_methods,
        boundary_proj, end_is_distal,
        centroid, major_axis, minor_axis, brightness_used
    """
    if mask is None or mask.sum() == 0:
        return None, None, None, None, 0.0, None, 0, None, True, None, None, None

    mask_clean = _clean_mask_morphology(mask)

    contours, _ = cv2.findContours(
        mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None, None, None, 0.0, None, 0, None, True, None, None, None
    if cv2.contourArea(max(contours, key=cv2.contourArea)) < MIN_CONTOUR_AREA:
        return None, None, None, None, 0.0, None, 0, None, True, None, None, None

    # PCA on all mask pixels for stable anatomical axes.
    if precomputed_axes is not None:
        centroid, major_axis, minor_axis = precomputed_axes
    else:
        centroid, major_axis, minor_axis = _pca_on_mask_pixels(mask_clean)
        major_axis, minor_axis, _ = _orient_anatomical_axis(
            image, mask_clean, centroid, major_axis, minor_axis, lab_frame=lab_frame
        )

    nail_yx    = np.argwhere(mask_clean == 255)
    nail_xy    = nail_yx[:, ::-1].astype(np.float64)
    centered   = nail_xy - centroid
    proj_major = centered @ major_axis
    proj_minor = centered @ minor_axis

    # Precompute LAB, HSV, GRAY once per frame
    lab_img = lab_frame if lab_frame is not None else cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pre-compute Canny edge map once -- reused by both _canny_boundary_estimate and _canny_validate_edge_at_boundary
    _nail_med_gray   = int(np.median(gray_img[nail_yx[:, 0], nail_yx[:, 1]]))
    _masked_gray     = np.full_like(gray_img, _nail_med_gray)
    _masked_gray[mask_clean > 0] = gray_img[mask_clean > 0]
    _blurred_canny   = cv2.GaussianBlur(_masked_gray, (7, 7), 2.0)
    _canny_edges_img = cv2.Canny(_blurred_canny, threshold1=18, threshold2=55)
    _canny_edges_img[mask_clean == 0] = 0

    # Initialise voting metadata
    n_methods       = 0
    boundary_proj   = None
    end_is_distal   = True
    _method_results = {}   # always defined (prevents NameError on polished path)

    # --- Polish hook: skip boundary detection ---
    if is_polished:
        _lab_pol = (
            lab_frame
            if lab_frame is not None
            else cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        )
        _L_pol = _lab_pol[:, :, 0][nail_yx[:, 0], nail_yx[:, 1]]
        _sort_idx_pol = np.argsort(proj_major)
        _q_pol = max(2, len(_sort_idx_pol) // 4)
        _L_start_pol = float(np.mean(_L_pol[_sort_idx_pol[:_q_pol]]))
        _L_end_pol   = float(np.mean(_L_pol[_sort_idx_pol[-_q_pol:]]))
        _end_is_distal_pol = _L_end_pol >= _L_start_pol

        _span_pol = float(proj_major.max() - proj_major.min())
        if _end_is_distal_pol:
            _pol_cutoff = float(proj_major.min()) + 0.87 * _span_pol
            in_bed = proj_major <= _pol_cutoff
        else:
            _pol_cutoff = float(proj_major.max()) - 0.87 * _span_pol
            in_bed = proj_major >= _pol_cutoff
        free_edge_present    = None
        free_edge_confidence = 0.3
        if debug:
            print(f"  [NailBed] Polished nail -- anatomical prior (87%) applied"
                  f"  (distal at {'max' if _end_is_distal_pol else 'min'} proj)")

    else:
        # --- Classify and optionally detect boundary ---
        boundary_proj, end_is_distal, free_edge_present, free_edge_confidence = (
            _detect_nail_bed_boundary_canny(
                mask_clean, major_axis, minor_axis, centroid, image, debug=debug
            )
        )
        _gradient_confidence = free_edge_confidence   # preserve before voting modifies it
        _full_nail_width_px = float(proj_minor.max() - proj_minor.min())

        # =========================================================
        # MULTI-METHOD BOUNDARY VOTING SYSTEM
        # =========================================================

        _axis_span_full = float(proj_major.max() - proj_major.min())
        _method_results = {}  # method_name -> (boundary_proj, weight)

        # --- Method 1: LAB Gradient ---
        if boundary_proj is not None:
            _method_results['gradient'] = (boundary_proj, 1.0)

        # --- Method 2: K-Means ---
        # Rule 1: skip KMeans & Otsu when gradient detector is confident.
        _skip_weak_estimators = (
            boundary_proj is not None and _gradient_confidence >= 0.75
        )
        # --- KMeans skip if std(L_all) < 6.0 ---
        L_all = lab_img[:, :, 0][nail_yx[:, 0], nail_yx[:, 1]]
        if not is_polished and not _skip_weak_estimators:
            if float(np.std(L_all)) >= 6.0:
                _km_bd = _kmeans_boundary_estimate(
                    nail_yx, proj_major, lab_img, end_is_distal,
                    _axis_span_full, float(proj_major.min()),
                )
                if _km_bd is not None:
                    _method_results['kmeans'] = (_km_bd, 0.8)
            elif debug:
                print(f"  [Voting] Skipping KMeans: std(L_all)={np.std(L_all):.2f} < 6.0")
        elif _skip_weak_estimators and debug:
            print("  [Voting] Gradient confidence >= 0.75 — skipping KMeans & Otsu")

        # --- Method 3: Otsu ---
        # --- Otsu skip if grayscale range < 20 ---
        nail_gray = gray_img[nail_yx[:, 0], nail_yx[:, 1]]
        gray_range = float(nail_gray.max() - nail_gray.min()) if len(nail_gray) > 0 else 0.0
        if not is_polished and not _skip_weak_estimators:
            if gray_range >= 20.0:
                _otsu_bd = _otsu_boundary_estimate(
                    nail_yx, proj_major, gray_img, mask_clean,
                    end_is_distal, centroid, major_axis,
                    debug=debug,
                )
                if _otsu_bd is not None:
                    _method_results['otsu'] = (_otsu_bd, 0.9)
            elif debug:
                print(f"  [Voting] Skipping Otsu: gray range={gray_range:.2f} < 20")

        # --- Method 4: Canny ---
        if not is_polished:
            _canny_bd = _canny_boundary_estimate(
                nail_yx, proj_major, gray_img, mask_clean,
                centroid, major_axis, minor_axis,
                end_is_distal, _axis_span_full,
                debug=debug,
                precomputed_edges=_canny_edges_img,
            )
            if _canny_bd is not None:
                _method_results['canny'] = (_canny_bd, 0.85)

        # --- Method 5: Brightness Segmentation ---
        if not is_polished:
            _fe_mask, _fe_conf = _detect_free_edge_by_brightness(
                nail_yx, mask_clean, lab_img, debug=debug
            )
            if _fe_mask is not None and _fe_conf > 0.35:
                _bright_bd, _bright_eid, _bright_major, _bright_minor = (
                    _boundary_from_free_edge_mask(
                        nail_yx, _fe_mask, centroid,
                        major_axis, minor_axis,
                        debug=debug,
                    )
                )
                if _bright_bd is not None:
                    _align_major = abs(float(np.dot(_bright_major, major_axis)))
                    _align_minor = abs(float(np.dot(_bright_major, minor_axis)))

                    if _align_major >= _align_minor:
                        _method_results['brightness'] = (_bright_bd, 1.2)
                        end_is_distal = _bright_eid
                        major_axis = _bright_major
                        minor_axis = _bright_minor
                        proj_major = (nail_xy - centroid) @ major_axis
                        proj_minor = (nail_xy - centroid) @ minor_axis
                        if debug:
                            print(f"  [Brightness] Axes corrected from free edge centroids. "
                                  f"boundary={_bright_bd:.1f}  end_is_distal={_bright_eid}  "
                                  f"align_major={_align_major:.2f}")
                    else:
                        _method_results['brightness'] = (_bright_bd, 0.5)
                        if debug:
                            print(f"  [Brightness] Axis replacement REJECTED (would flip axes). "
                                  f"align_major={_align_major:.2f}  align_minor={_align_minor:.2f}  "
                                  f"boundary registered at reduced weight.")

        if debug:
            print(f"  [Voting] Methods with results: {list(_method_results.keys())}")
            for _mn, (_mb, _mw) in _method_results.items():
                print(f"    {_mn}: proj={_mb:.1f}  weight={_mw:.2f}")

        # --- New boundary acceptance logic ---
        n_methods = len(_method_results)
        _final_boundary = None

        # Rule A: Strong Gradient Wins

        AGREE_THRESH = 0.03 * _axis_span_full  # 3% only

        _final_boundary = None

        # --- Rule 1: High-confidence brightness only ---
        if 'brightness' in _method_results and _method_results['brightness'][1] > 0.70:
            candidate = _method_results['brightness'][0]
            _final_boundary = candidate
            free_edge_confidence = min(1.0, free_edge_confidence * 1.15)
            if debug:
                print(f"[Voting] Brightness confidence > 0.70 — accepting boundary at {candidate:.1f}")

        # --- Rule 2: Agreement between NON-CANNY methods only ---
        elif len(_method_results) >= 2:
            method_names = list(_method_results.keys())
            for i in range(len(method_names)):
                for j in range(i+1, len(method_names)):
                    m1, m2 = method_names[i], method_names[j]

                    # ❌ Do NOT allow canny to form agreement
                    if 'canny' in (m1, m2):
                        continue

                    b1, _ = _method_results[m1]
                    b2, _ = _method_results[m2]

                    if abs(b1 - b2) < AGREE_THRESH:
                        candidate = 0.5 * (b1 + b2)
                        _final_boundary = candidate
                        free_edge_confidence = min(1.0, free_edge_confidence * 1.10)
                        if debug:
                            print(f"[Voting] {m1} & {m2} agree — accepting boundary at {candidate:.1f}")
                        break
                if _final_boundary is not None:
                    break

        # --- Rule 3: Revalidate boundary anatomically ---
        if _final_boundary is not None:
            _min_proj_full = float(proj_major.min())
            _max_proj_full = float(proj_major.max())
            rel_pos = (_final_boundary - _min_proj_full) / _axis_span_full
            free_edge_fraction = (_max_proj_full - _final_boundary) / _axis_span_full

            if rel_pos < 0.55 or rel_pos > 0.90:
                if debug:
                    print("[Voting] Boundary failed rel_pos validation — rejecting.")
                _final_boundary = None

            elif free_edge_fraction > 0.20:
                if debug:
                    print("[Voting] Boundary too large to be anatomical free edge — rejecting.")
                _final_boundary = None

        # --- Final decision ---
        if _final_boundary is None and debug:
            print("[Voting] No valid agreement — rejecting boundary.")
            STRONG_WEIGHT = 0.95
            MEDIUM_WEIGHT = 0.70
            AGREE_THRESH  = 0.05 * _axis_span_full   # 5 % of axis span

            _strong = [
                (n, b, w) for n, (b, w) in _method_results.items()
                if w >= STRONG_WEIGHT
            ]
            _medium = [
                (n, b, w) for n, (b, w) in _method_results.items()
                if MEDIUM_WEIGHT <= w < STRONG_WEIGHT
            ]

            if _strong:
                _best = max(_strong, key=lambda x: x[2])
                _final_boundary = _best[1]
                free_edge_confidence = min(1.0, free_edge_confidence * 1.10)
                if debug:
                    print(f"  [Voting] Fallback: Strong method '{_best[0]}' — boundary={_final_boundary:.1f}")
            elif len(_medium) >= 2:
                _medium_sorted = sorted(_medium, key=lambda x: -x[2])
                _found_pair = False
                for _mi in range(len(_medium_sorted)):
                    for _mj in range(_mi + 1, len(_medium_sorted)):
                        if abs(_medium_sorted[_mi][1] - _medium_sorted[_mj][1]) < AGREE_THRESH:
                            _pair_vals = [_medium_sorted[_mi][1], _medium_sorted[_mj][1]]
                            _pair_wts  = [_medium_sorted[_mi][2], _medium_sorted[_mj][2]]
                            _final_boundary = float(
                                np.average(_pair_vals, weights=_pair_wts)
                            )
                            free_edge_confidence = min(
                                1.0, free_edge_confidence * 1.05
                            )
                            _found_pair = True
                            if debug:
                                print(
                                    f"  [Voting] Fallback: Two agreeing medium methods "
                                    f"('{_medium_sorted[_mi][0]}', "
                                    f"'{_medium_sorted[_mj][0]}'): "
                                    f"boundary={_final_boundary:.1f}"
                                )
                            break
                    if _found_pair:
                        break
                if not _found_pair and debug:
                    print(f"  [Voting] Fallback: {len(_medium)} medium methods disagree — no boundary fabricated")
            else:
                if debug:
                    _avail = [n for n, _, _ in _medium] if _medium else list(_method_results.keys())
                    print(f"  [Voting] Fallback: Insufficient evidence ({_avail}) — no boundary")

        # --- Apply final boundary or fall back to full mask ---
        if _final_boundary is not None:
            # Cross-validate with Otsu colour and Canny edge checks
            _otsu_mult = _otsu_validate_color_boundary(
                nail_yx, proj_major, lab_img, _final_boundary, debug=debug
            )
            _canny_mult = _canny_validate_edge_at_boundary(
                _canny_edges_img, nail_yx, proj_major, proj_minor,
                _final_boundary,
            )
            free_edge_confidence = min(
                1.0, free_edge_confidence * _otsu_mult * _canny_mult
            )

            in_bed = _build_curved_bed_mask(
                nail_yx, centroid, major_axis, minor_axis,
                _final_boundary, _full_nail_width_px, end_is_distal,
            )
            bed_fraction = np.sum(in_bed) / len(in_bed)
            free_edge_fraction = 1.0 - bed_fraction
            if free_edge_fraction < 0.02:
                free_edge_present = False
                boundary_proj = None
                if debug:
                    print(f"  [Voting] Free edge fraction {free_edge_fraction:.3f} < 0.02, rejecting free edge.")
            else:
                free_edge_present = True
                boundary_proj     = _final_boundary

            if in_bed.sum() < 10:
                if debug:
                    print("  [Voting] Curved boundary too aggressive — full mask used")
                in_bed = np.ones(len(proj_major), dtype=bool)
                free_edge_confidence *= 0.5
        else:
            # No boundary — use full nail mask, never fabricate
            in_bed = np.ones(len(proj_major), dtype=bool)
            free_edge_present    = False
            free_edge_confidence = max(free_edge_confidence, 0.30)
            boundary_proj        = None



    # --- Anatomical geometry measurement (NO PCA, NO swaps) ---
    # Project bed mask pixels onto anatomical axis and perpendicular
    nb_yx = nail_yx[in_bed]
    nb_xy = nb_yx[:, ::-1].astype(np.float64)
    nb_centered = nb_xy - centroid
    anatomical_axis = major_axis  # already anatomical from minAreaRect logic
    perp_axis = np.array([-anatomical_axis[1], anatomical_axis[0]])
    proj_len = nb_centered @ anatomical_axis
    proj_wid = nb_centered @ perp_axis

    length_px = float(proj_len.max() - proj_len.min())
    width_px  = float(proj_wid.max() - proj_wid.min())
    ratio = length_px / width_px if width_px > 0 else 0.0

    # Midpoint width (P2–P98 in central 10% band)
    _mid_proj = float(np.median(proj_len))
    _axis_span_nb = float(proj_len.max() - proj_len.min())
    _in_mid_band = np.abs(proj_len - _mid_proj) < (_axis_span_nb * 0.05)
    if _in_mid_band.sum() >= 10:
        _mid_width_px = float(
            np.percentile(proj_wid[_in_mid_band], 98)
            - np.percentile(proj_wid[_in_mid_band], 2)
        )
    else:
        _mid_width_px = width_px

    # Area by pixel count
    nail_bed_mask = np.zeros_like(mask_clean)
    nail_bed_mask[nb_yx[:, 0], nb_yx[:, 1]] = 255
    area_px = float(np.sum(nail_bed_mask == 255))

    # --- Optional: visualise boundary line ---
    if NAIL_BED_VISUALIZE_BOUNDARY and image is not None and not is_polished:
        if in_bed.sum() < len(proj_len):
            boundary_xy = centroid + float(proj_len.max()) * anatomical_axis
            half_w      = width_px / 2.0
            pt1 = tuple((boundary_xy + half_w * perp_axis).astype(int))
            pt2 = tuple((boundary_xy - half_w * perp_axis).astype(int))
            cv2.line(image, pt1, pt2, (0, 255, 255), 2)

    # --- Sanity checks on geometry ---
    # REMOVED: Geometry-based overrides and trimmed fallback. Free-edge presence is determined only by boundary detection result.

    if debug:
        print(
            f"  [NailBed] length={length_px:.1f}px  width={width_px:.1f}px  "
            f"ratio={ratio:.2f}  "
            f"area={area_px:.0f}px\u00b2  "
            f"free_edge={free_edge_present}  conf={free_edge_confidence:.2f}"
        )

    _brightness_used = 'brightness' in _method_results

    return (length_px, width_px, area_px, free_edge_present,
        free_edge_confidence, _mid_width_px, n_methods,
        boundary_proj, end_is_distal,
        centroid, major_axis, minor_axis,
        _brightness_used)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def extract_geometry_nail_bed(
    image: np.ndarray,
    mask: np.ndarray,
    debug: bool = False,
    is_polished: bool = False,
    lab_frame: Optional[np.ndarray] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract nail bed geometry using adaptive LAB gradient boundary detection.

    Backward-compatible wrapper around ``_extract_nail_bed_internal``.
    """
    length_px, width_px, area_px, _, _, _mid_width_px, _, _, _, _, _, _, _ = _extract_nail_bed_internal(
        image, mask, is_polished=is_polished, debug=debug, lab_frame=lab_frame
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
        _extract_nail_bed_internal(image, mask, is_polished=is_polished, debug=verbose,
                                   lab_frame=lab_frame)
    )
    diagnostics = {
        "method": "adaptive_lab_gradient_pca_mask_pixels",
        "length_px": length_px,
        "width_px": width_px,
        "width_midpoint_px": _mid_width_px,
        "area_px": area_px,
        "aspect_ratio": (
            round(length_px / width_px, 3)
            if (length_px and width_px and width_px > 0)
            else None
        ),
        "free_edge_present": free_edge_present,
        "free_edge_confidence": free_edge_confidence,
        "is_polished": is_polished,
        "boundary_methods_agreed": _n_methods,
        "boundary_confidence": free_edge_confidence,
        "boundary_proj": _boundary_proj,
        "end_is_distal": _end_is_distal,
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
    proj_major = centered @ major_axis
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
        else:
            nb_mask = mask_clean.copy()
            boundary_proj = None
    else:
        # No boundary detected — use full nail mask (no fabricated boundary)
        nb_mask = mask_clean.copy()

    result.update({
        "nail_bed_mask":        nb_mask,
        "boundary_proj":        boundary_proj,
        "end_is_distal":        end_is_distal,
        "free_edge_present":    free_edge_present,
        "free_edge_confidence": free_edge_confidence,
        "centroid":             centroid,
        "major_axis":           major_axis,
        "minor_axis":           minor_axis,
        "length_px":            length_px or 0.0,
        "width_px":             width_px or full_nail_width_px,
        "area_px":              area_px or 0.0,
        "width_midpoint_px":    mid_width_px,
        "full_nail_width_px":   full_nail_width_px,
        "n_methods":            n_methods,
        "brightness_method_used": brightness_method_used,
    })
    return result