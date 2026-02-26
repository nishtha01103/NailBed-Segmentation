import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
"""Adaptive nail bed boundary detection via LAB gradient analysis.

Contains the primary free-edge boundary detector (local derivative method),
along with helper utilities for profile smoothing and spatial coherence scoring.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from config import FREE_EDGE_K_SIGMA


def _smooth_profile(arr: np.ndarray, window: int = 3) -> np.ndarray:
    """Vectorised centred moving-average using np.convolve (no Python loop).

    NaN values are handled by replacing them with the local non-NaN mean via
    a separate weight convolution (equivalent to masked averaging).  This is
    ~100× faster than the previous per-element Python loop and produces
    identical results.
    """
    if window <= 1:
        return arr.copy()
    kernel  = np.ones(window, dtype=np.float64)
    # Replace NaN with 0 for the value convolution; track valid counts separately.
    valid   = (~np.isnan(arr)).astype(np.float64)
    filled  = np.where(np.isnan(arr), 0.0, arr)
    val_sum = np.convolve(filled, kernel, mode="same")
    cnt_sum = np.convolve(valid,  kernel, mode="same")
    # Where no valid neighbours exist keep original (NaN) value.
    out = np.where(cnt_sum > 0, val_sum / cnt_sum, arr)
    return out



def _detect_free_edge_simple(
    mask: np.ndarray,
    anatomical_axis: np.ndarray,
    width_axis: np.ndarray,
    centroid: np.ndarray,
    lab_img: np.ndarray,
    image: np.ndarray,
    debug: bool = False
) -> Tuple[Optional[float], Optional[bool], float]:
    """
    Simplified, robust free-edge detector for nail bed boundary.
    Projects mask pixels onto anatomical axis, analyzes LAB gradient, and returns boundary projection.
    No strict gating, bins, or multi-condition chains.
    """
    nail_yx = np.argwhere(mask == 255)
    nail_xy = nail_yx[:, ::-1].astype(np.float64)
    centered = nail_xy - centroid
    proj_anat = centered @ anatomical_axis
    min_proj = float(proj_anat.min())
    max_proj = float(proj_anat.max())
    axis_span = max_proj - min_proj
    if axis_span < 10.0:
        if debug:
            print("[FreeEdgeSimple] Axis span too small, rejecting.")
        return None, False, 0.0

    # STEP B: Slice longitudinal profile
    n_slices = int(np.clip(round(axis_span / 3.0), 40, 60))
    edges = np.linspace(min_proj, max_proj, n_slices + 1)
    # Lighting normalization (inside mask only)
    L_channel = lab_img[:, :, 0]
    L_vals = L_channel[nail_yx[:, 0], nail_yx[:, 1]].astype(np.float32)
    mean_L = np.mean(L_vals)
    std_L = np.std(L_vals) + 1e-6
    L_norm_img = (L_channel - mean_L) / std_L
    L_norm_vals = L_norm_img[nail_yx[:, 0], nail_yx[:, 1]]
    a_vals = lab_img[:, :, 1][nail_yx[:, 0], nail_yx[:, 1]].astype(np.float32)
    slice_L = np.zeros(n_slices)
    slice_a = np.zeros(n_slices)
    for i in range(n_slices):
        in_slice = (proj_anat >= edges[i]) & (proj_anat < edges[i + 1])
        if in_slice.sum() < 3:
            slice_L[i] = np.nan
            slice_a[i] = np.nan
            continue
        slice_L[i] = np.percentile(L_norm_vals[in_slice], 80)
        slice_a[i] = np.percentile(a_vals[in_slice], 50)

    # Smooth slice_L
    slice_L = _smooth_profile(slice_L, window=3)

    # STEP C: Compute derivative
    delta_L = np.diff(slice_L)
    delta_a = np.diff(slice_a)
    distal_start = n_slices // 2
    search_idx = np.arange(distal_start, n_slices - 1)

    # STEP D: Data-driven threshold for distal spike
    distal_half = delta_L[search_idx]
    valid_distal = distal_half[~np.isnan(distal_half)]
    if len(valid_distal) == 0:
        # STEP G: Trimmed nail detection
        distal_30 = slice_L[int(0.7 * n_slices):]
        distal_std = float(np.nanstd(distal_30))
        if debug:
            print(f"[FreeEdgeSimple] No valid distal derivatives. Distal std={distal_std:.2f}")
        if distal_std < 4.0:
            return None, False, 0.3  # Trimmed nail
        else:
            return None, False, 0.5  # Uncertain, no boundary
    adaptive_thresh = float(np.mean(valid_distal) + 1.5 * np.std(valid_distal))
    candidates = np.where((delta_L[search_idx] > adaptive_thresh) & (delta_a[search_idx] < 0))[0]
    if len(candidates) == 0:
        distal_30 = slice_L[int(0.7 * n_slices):]
        distal_std = float(np.nanstd(distal_30))
        if debug:
            print(f"[FreeEdgeSimple] No candidate. Adaptive threshold={adaptive_thresh:.2f}, Distal std={distal_std:.2f}")
        if distal_std < 4.0:
            return None, False, 0.3  # Trimmed nail
        else:
            return None, False, 0.5  # Uncertain, no boundary

    # Select slice with maximum delta_L
    best_idx = search_idx[candidates[np.argmax(delta_L[search_idx][candidates])]]
    boundary_proj = float(0.5 * (edges[best_idx] + edges[best_idx + 1]))
    max_delta_L = float(delta_L[best_idx])

    # Color Continuity Check
    bed_mask = (proj_anat < boundary_proj)
    free_mask = (proj_anat >= boundary_proj)
    if np.sum(bed_mask) > 10 and np.sum(free_mask) > 10:
        mean_a_bed = np.mean(lab_img[:, :, 1][nail_yx[bed_mask, 0], nail_yx[bed_mask, 1]])
        mean_a_free = np.mean(lab_img[:, :, 1][nail_yx[free_mask, 0], nail_yx[free_mask, 1]])
        mean_b_bed = np.mean(lab_img[:, :, 2][nail_yx[bed_mask, 0], nail_yx[bed_mask, 1]])
        mean_b_free = np.mean(lab_img[:, :, 2][nail_yx[free_mask, 0], nail_yx[free_mask, 1]])
        if abs(mean_a_free - mean_a_bed) > 20 or abs(mean_b_free - mean_b_bed) > 20:
            if debug:
                print(f"[FreeEdgeSimple] Color continuity check failed: |a*|={abs(mean_a_free - mean_a_bed):.1f}, |b*|={abs(mean_b_free - mean_b_bed):.1f} > 20, rejecting boundary.")
            return None, False, 0.0

    # Free edge region width stability check (CRITICAL)
    free_mask = (proj_anat >= boundary_proj)
    if np.sum(free_mask) > 0:
        # Project free region points onto width_axis (minor axis)
        free_widths = (centered[free_mask] @ width_axis)
        free_width = free_widths.max() - free_widths.min() if free_widths.size > 0 else 0.0
        # Full nail width (all mask points)
        full_widths = (centered @ width_axis)
        full_nail_width = full_widths.max() - full_widths.min() if full_widths.size > 0 else 1e-6
        if free_width < 0.5 * full_nail_width:
            if debug:
                print(f"[FreeEdgeSimple] Free edge width too narrow: {free_width:.2f} < 0.5 * full_nail_width {full_nail_width:.2f}, rejecting.")
            return None, False, 0.0
    # Hard free edge size constraint
    free_edge_fraction = (max_proj - boundary_proj) / axis_span
    if free_edge_fraction > 0.20:
        if debug:
            print(f"[FreeEdgeSimple] Free edge fraction {free_edge_fraction:.2f} > 0.20, rejecting boundary.")
        return None, False, 0.0
    # 4️⃣ Boundary position guard
    rel_pos = (boundary_proj - min_proj) / axis_span
    if rel_pos < 0.55 or rel_pos > 0.90:
        if debug:
            print(f"[FreeEdgeSimple] Boundary rel_pos={rel_pos:.2f} out of range [0.55, 0.90], rejecting.")
        return None, False, 0.0

    # Boundary distance from mask extreme constraint
    # Determine which extreme is distal (assume distal is at max_proj for now)
    # If you have distal sign logic, adjust accordingly
    distance_to_extreme = abs(boundary_proj - max_proj)
    if distance_to_extreme < 0.08 * axis_span:
        if debug:
            print(f"[FreeEdgeSimple] Boundary too close to mask extreme: distance={distance_to_extreme:.2f} < 8% axis_span, rejecting.")
        return None, False, 0.0

    # Glare validation: check mean L before vs after boundary
    before_idx = max(0, best_idx - 3)
    after_idx = min(n_slices - 1, best_idx + 3)
    L_before = np.nanmean(slice_L[before_idx:best_idx])
    L_after = np.nanmean(slice_L[best_idx:after_idx])
    delta_L = L_after - L_before
    # 1️⃣ Reject Extreme Contrast
    if abs(delta_L) > 35:
        if debug:
            print(f"[FreeEdgeSimple] Extreme contrast: |L_after - L_before| = {abs(delta_L):.2f} > 35, rejecting boundary.")
        return None, False, 0.0
    if delta_L < 4:
        if debug:
            print(f"[FreeEdgeSimple] Glare check failed: L_after - L_before = {delta_L:.2f} < 4, rejecting boundary.")
        return None, False, 0.0

    # STEP E: Optional Canny reinforcement
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray, 20, 60)
    band_mask = (proj_anat >= edges[best_idx] - 5) & (proj_anat <= edges[best_idx] + 5)
    if band_mask.sum() > 0:
        edge_density = float(np.mean(edges_img[nail_yx[band_mask, 0], nail_yx[band_mask, 1]] > 0))
    else:
        edge_density = 0.0
    confidence = min(1.0, max_delta_L / 12.0)
    if edge_density > 0.05:
        confidence = min(1.0, confidence + 0.15)

    # STEP F: Confidence calculation
        if max_delta_L < adaptive_thresh:
            if debug:
                print(f"[FreeEdgeSimple] max_delta_L < adaptive_thresh ({adaptive_thresh:.2f}), rejecting boundary.")
            return None, False, 0.0

    if debug:
        print(f"[FreeEdgeSimple] boundary_proj={boundary_proj:.1f} confidence={confidence:.2f} edge_density={edge_density:.2f}")
    return boundary_proj, True, confidence


def _detect_nail_bed_boundary_canny(
    mask: np.ndarray,
    anatomical_axis: np.ndarray,
    width_axis: np.ndarray,
    centroid: np.ndarray,
    image: np.ndarray,
    debug: bool = False,
) -> Tuple[Optional[float], Optional[bool], float]:
    # Utility: Enforce Small Free Edge Size after bed mask creation
    def reject_if_large_free_edge(bed_fraction, debug=False):
        free_edge_fraction = 1 - bed_fraction
        if free_edge_fraction > 0.25:
            if debug:
                print(f"[BoundaryReject] Free edge fraction {free_edge_fraction:.2f} > 0.25, rejecting boundary.")
            return True
        return False
    """Detect free-edge boundary using Canny edge coverage along anatomical axis."""
    nail_yx = np.argwhere(mask == 255)
    nail_xy = nail_yx[:, ::-1].astype(np.float64)
    centered = nail_xy - centroid
    proj_anat = centered @ anatomical_axis
    proj_width = centered @ width_axis
    min_proj = float(proj_anat.min())
    max_proj = float(proj_anat.max())
    axis_span = max_proj - min_proj
    if axis_span < 1.0:
        return None, True, None, 0.3
    n_slices = int(np.clip(round(axis_span / 5.0), 30, 70))
    edges = np.linspace(min_proj, max_proj, n_slices + 1)
    # Lighting normalization
    L_channel = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]
    L_mask_vals = L_channel[mask > 0]
    mean_L = np.mean(L_mask_vals)
    std_L = np.std(L_mask_vals) + 1e-6
    L_norm = (L_channel - mean_L) / std_L
    # Run Canny edge detection once
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray_img, threshold1=18, threshold2=55)
    # For each slice in distal 50%, count edge pixels perpendicular to axis
    distal_start = n_slices // 2
    width_coverage_list = []
    edge_peaks = []
    slice_widths = []
    for i in range(distal_start, n_slices):
        in_slice = (proj_anat >= edges[i]) & (proj_anat < edges[i + 1])
        if in_slice.sum() < 3:
            width_coverage_list.append(0.0)
            edge_peaks.append(0.0)
            slice_widths.append(np.nan)
            continue
        slice_yx = nail_yx[in_slice]
        slice_proj_width = proj_width[in_slice]
        min_w = float(slice_proj_width.min())
        max_w = float(slice_proj_width.max())
        n_bins = max(10, int(max_w - min_w))
        bins = np.linspace(min_w, max_w, n_bins + 1)
        edge_counts = np.zeros(n_bins)
        for j in range(n_bins):
            in_bin = (slice_proj_width >= bins[j]) & (slice_proj_width < bins[j + 1])
            if in_bin.sum() == 0:
                continue
            bin_yx = slice_yx[in_bin]
            edge_count = 0
            for y, x in bin_yx:
                if canny_edges[y, x] > 0:
                    edge_count += 1
            edge_counts[j] = edge_count
        # Compute width coverage: percentage of bins with edge pixels
        width_coverage = float(np.sum(edge_counts > 0)) / max(n_bins, 1)
        width_coverage_list.append(width_coverage)
        slice_widths.append(max_w - min_w)

    # External protrusion boundary rejection and boundary position guard
    boundary_idx = None
    for idx, coverage in enumerate(width_coverage_list):
        if coverage > 0.6:
            boundary_idx = idx + distal_start
            break
    if boundary_idx is not None:
        width_current = slice_widths[boundary_idx - distal_start]
        width_prev = slice_widths[boundary_idx - distal_start - 1] if boundary_idx - distal_start - 1 >= 0 else width_current
        if abs(width_current - width_prev) / max(width_current, 1e-6) > 0.35:
            if debug:
                print(f"  [BoundaryReject] External protrusion detected: width change {width_current:.2f} vs {width_prev:.2f}, rejecting boundary.")
            return None, True, False, 0.5
    if boundary_idx is not None:
        boundary_proj = float(0.5 * (edges[boundary_idx] + edges[boundary_idx + 1]))
        rel_pos = (boundary_proj - min_proj) / axis_span
        if rel_pos < 0.55 or rel_pos > 0.90:
            if debug:
                print(f"  [BoundaryReject] Boundary rel_pos={rel_pos:.2f} out of range [0.55, 0.90], rejecting.")
            return None, True, False, 0.5
        edge_peaks.append(np.max(edge_counts))
    # Canny is only used as a reinforcement signal, not for auto-acceptance.
    # Remove auto-acceptance based on width_coverage. Always return None for boundary.
    if debug:
        print("  [CannyBoundary] Canny cannot accept boundary alone. Returning None.")
    return None, True, False, 0.5
    _distal_25_start = int(0.75 * n_slices)
    _distal_25_L     = s_L[_distal_25_start:]
    _distal_25_valid = _distal_25_L[~np.isnan(_distal_25_L)]

    if len(_distal_25_valid) >= 2:
        distal_L_std    = float(np.std(_distal_25_valid))
        _distal_25_diffs = np.diff(_distal_25_valid)
        max_abs_L_drop  = (
            float(np.max(np.abs(_distal_25_diffs)))
            if len(_distal_25_diffs) > 0
            else 0.0
        )
        # New: also require distal region to be short
        distal_region_length = float(len(_distal_25_valid)) / float(n_slices)
        if (distal_L_std < 4.0 and max_abs_L_drop < 6.0 and distal_region_length < 0.20):
            if debug:
                print(
                    f"  [NailBed] Short-nail early exit: distal 25% "
                    f"L std={distal_L_std:.2f} < 4.0, "
                    f"max |L drop|={max_abs_L_drop:.2f} < 6.0, "
                    f"distal_region_length={distal_region_length:.2f} < 0.20 "
                    f"— classified as trimmed"
                )
            return None, end_is_distal, False, 0.95

    # =========================================================
    # LOCAL DERIVATIVE — free-edge detection
    # =========================================================

    # STEP 1 — Compute slice-to-slice L derivatives
    delta_L_arr = np.diff(s_L)
    delta_a_arr = np.diff(s_a)
    delta_b_arr = np.diff(s_b)
    delta_S_arr = np.diff(s_S)  # negative = saturation dropping = toward free edge
    delta_V_arr = np.diff(s_V)  # positive = brightness rising = toward free edge

    # Second derivative of L — sharp transitions have high curvature;
    # gradual lighting gradients do not.
    second_delta_L = np.diff(delta_L_arr)

    # STEP 2 — Single-pass search window: distal 60 %–95 %.
    # Strictly restricting to this region avoids false positives from
    # gradual lighting gradients in the proximal/mid nail.
    SEARCH_START = int(0.60 * n_slices)
    SEARCH_END   = int(0.95 * n_slices)

    # FIX 3 — Adaptive threshold computed over DISTAL HALF only.
    # Computing stats over all slices inflates std when the proximal half
    # has large cuticle/skin-fold transitions, making the threshold too loose.
    distal_start_idx = int(0.50 * n_slices)
    valid_delta_distal = delta_L_arr[distal_start_idx:][~np.isnan(delta_L_arr[distal_start_idx:])]
    valid_delta_full   = delta_L_arr[~np.isnan(delta_L_arr)]

    if len(valid_delta_distal) < 4:   # not enough distal slices — use full axis
        valid_delta = valid_delta_full
    else:
        valid_delta = valid_delta_distal

    # Hardened boundary acceptance: require width_coverage > 0.6 and abs(L_after - L_before) >= 6
    if canny_width_coverage > 0.6:
        slice_idx = np.argmin(np.abs(edges - canny_boundary))
        before_idx = max(0, slice_idx - 3)
        after_idx = min(n_slices - 1, slice_idx + 3)
        L_channel = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]
        mask_idx = (proj_anat >= edges[before_idx]) & (proj_anat < edges[after_idx])
        L_vals = L_channel[nail_yx[:, 0], nail_yx[:, 1]]
        L_before = np.nanmean(L_vals[mask_idx & (proj_anat < canny_boundary)])
        L_after = np.nanmean(L_vals[mask_idx & (proj_anat >= canny_boundary)])
        if abs(L_after - L_before) >= 6:
            _final_boundary = canny_boundary
            free_edge_confidence = min(1.0, free_edge_confidence * 1.10)
            if debug:
                print(f"  [Voting] Hardened: width_coverage > 0.6 and |L_after-L_before| >= 6 — accepting boundary at {canny_boundary:.1f}")

    if len(valid_delta) == 0:
        if debug:
            print("  [NailBed] No valid derivatives — full mask used")
        return None, end_is_distal, False, 0.9

    local_mean = float(np.mean(valid_delta))
    local_std  = float(np.std(valid_delta))
    adaptive_threshold = local_mean - FREE_EDGE_K_SIGMA * local_std

    # FIX 4 — Minimum absolute L-drop floor.
    # Prevents noise from passing the relative threshold on uniform nails.
    # A real nail-bed / free-edge transition is always ≥ 7 LAB-L units.
    MIN_ABS_DROP = -7.0
    adaptive_threshold = min(adaptive_threshold, MIN_ABS_DROP)

    if debug:
        print(
            f"  [NailBed] local_mean={local_mean:.3f}  local_std={local_std:.3f}  "
            f"adaptive_threshold={adaptive_threshold:.3f}"
        )

    def _scan_candidates(start: int, end: int) -> list:
        """Return candidate slice indices (distal-to-proximal) in [start, end).

        IMPORTANT: The free edge is BRIGHTER than the nail bed (white/translucent
        tip vs pink vascularised bed).  Going proximal→distal across the boundary,
        L INCREASES (positive spike), a* DECREASES (less pink), saturation DROPS.

        The original code only looked for a NEGATIVE delta_combined, which would
        only fire if the free edge were darker than the nail bed — the opposite of
        reality for top-down nail photography.

        Fix: detect ANY sharp spike (positive or negative) using abs().
        The redness check in the evaluation loop confirms the anatomical direction:
          a_before (nail bed, pink) > a_after (free edge, white) → correct boundary.
        """
        cands = []
        abs_threshold = abs(adaptive_threshold)
        for idx in range(end, start, -1):
            if np.isnan(delta_L_arr[idx]):
                continue
            _dS = delta_S_arr[idx] if idx < len(delta_S_arr) else 0.0
            _db = delta_b_arr[idx] if idx < len(delta_b_arr) else 0.0
            _da = delta_a_arr[idx] if idx < len(delta_a_arr) else 0.0

            # Combined multi-channel gradient signal.
            # At the nail-bed → free-edge transition (proximal→distal):
            #   delta_L > 0  (L rises — free edge is brighter)
            #   delta_a < 0  (a* drops — less pink/red)
            #   delta_S < 0  (saturation drops — free edge is white/neutral)
            # We build a signed "free-edge score" and threshold its ABSOLUTE value.
            # Sign convention: positive = free-edge-like transition.
            delta_combined = (
                delta_L_arr[idx]       # +ve when L rises toward free edge
                - 0.5 * _da            # +ve when a* drops (less red)
                - 0.3 * _db            # +ve when b* drops (less yellow)
                - 0.004 * _dS          # +ve when saturation drops
            )
            # Accept if the magnitude exceeds the adaptive threshold.
            # The redness check below then confirms anatomical direction.
            if abs(delta_combined) > abs_threshold:
                cands.append(idx)
        return cands

    # STEP 3 — Gather candidates (single pass: 60 %–95 %)
    candidates = _scan_candidates(SEARCH_START, SEARCH_END)

    # STEP 4 — Evaluate each candidate until one is confirmed
    free_edge_present    = False
    free_edge_confidence = 0.9
    boundary_proj        = None

    WIN        = 5
    A_TOLERANCE = 0.5

    # Relaxed boundary position guard: allow 0.55–0.95
    MIN_BED_FRACTION = 0.55
    MAX_BED_FRACTION = 0.95

    for i in candidates:
        # Boundary position guard: the cut point must lie in [60 %, 95 %].
        rel_pos = (i + 0.5) / n_slices   # 0 = proximal, 1 = distal
        if rel_pos < MIN_BED_FRACTION or rel_pos > MAX_BED_FRACTION:
            if debug:
                print(f"  [NailBed] Candidate {i} rejected by position guard "
                      f"(rel_pos={rel_pos:.2f} outside [{MIN_BED_FRACTION}, {MAX_BED_FRACTION}])")
            continue

        a_before = (
            float(np.nanmean(s_a[max(0, i - WIN):i]))
            if i > 0 else float(s_a[0])
        )
        a_after = float(s_a[min(i + 1, n_slices - 1)])

        redness_confirmed = (a_before - a_after) >= -A_TOLERANCE

        if redness_confirmed:
            # Relaxed second derivative threshold
            if i >= 1 and (i - 1) < len(second_delta_L):
                if abs(second_delta_L[i - 1]) <= 2.0:
                    if debug:
                        print(f"  [NailBed] Candidate {i} rejected: second derivative "
                              f"{abs(second_delta_L[i - 1]):.2f} <= 2.0")
                    continue

            # Saturation/brightness confirmation.
            # At the free edge: saturation drops (white tip) OR L changes sharply.
            # Accept if saturation drops meaningfully OR |delta_L| is large.
            if i < len(delta_S_arr):
                if not (delta_S_arr[i] < -4.0 or abs(delta_L_arr[i]) > 10.0):
                    if debug:
                        print(f"  [NailBed] Candidate {i} rejected: saturation/brightness "
                              f"delta_S={delta_S_arr[i]:.2f}, delta_L={delta_L_arr[i]:.2f} (not desaturated or bright enough)")
                    continue

            next_i        = min(i + 1, n_slices - 1)
            boundary_proj_candidate = float(0.5 * (s_centers[i] + s_centers[next_i]))
            # Lowered coherence threshold: require >= 0.55
            coherence = _boundary_coherence_score(
                mask, centroid, major_axis, minor_axis,
                proj_major, proj_minor, nail_yx,
                boundary_proj_candidate, lab_img, width=10
            )
            if coherence < 0.55:
                if debug:
                    print(f"  [NailBed] Candidate {i} rejected by coherence ({coherence:.2f} < 0.55)")
                continue
            # Lateral coverage check: boundary must span ≥ 35 % of nail width
            _near_bd = (
                (proj_major >= boundary_proj_candidate - 5) &
                (proj_major <= boundary_proj_candidate + 5)
            )
            if _near_bd.sum() >= 3:
                _w_at_bd   = float(
                    np.percentile(proj_minor[_near_bd], 95) -
                    np.percentile(proj_minor[_near_bd], 5)
                )
                _total_w   = float(proj_minor.max() - proj_minor.min())
                if _total_w > 0 and (_w_at_bd / _total_w) < 0.35:
                    if debug:
                        print(f"  [NailBed] Candidate {i} rejected: lateral coverage "
                              f"{_w_at_bd / _total_w:.0%} < 35%")
                    continue
            boundary_proj = boundary_proj_candidate
            free_edge_present    = True
            # Confidence based on magnitude of the brightness jump, not signed value
            free_edge_confidence = min(
                1.0, abs(delta_L_arr[i]) / max(abs(adaptive_threshold), 1e-6)
            )
            if debug:
                print(
                    f"  [NailBed] Free edge detected: slice={i}/{n_slices}  "
                    f"boundary_proj={boundary_proj:.1f}  "
                    f"confidence={free_edge_confidence:.2f}  "
                    f"delta_L={delta_L_arr[i]:.2f}  "
                    f"a_before={a_before:.1f}  a_after={a_after:.1f}"
                )
            break
        else:
            if debug:
                print(
                    f"  [NailBed] Candidate slice {i} failed redness "
                    f"(a_before={a_before:.1f}  a_after={a_after:.1f}) — trying next"
                )

    # STEP 5 — No qualifying candidate → short / trimmed nail
    if not free_edge_present:
        # Check if this is a genuinely short/trimmed nail (L profile flat in distal region)
        _distal_L_std = float(np.nanstd(s_L[int(0.6 * n_slices):]))
        if _distal_L_std < 5.0:
            # Very uniform distal region = genuinely trimmed nail, not a detection failure
            # Report high confidence that free edge is ABSENT
            free_edge_confidence = 0.88
            if debug:
                print(f"  [NailBed] Confirmed short/trimmed nail "
                      f"(distal L std={_distal_L_std:.2f})")
        else:
            # Uncertain: gradient exists but no candidate passed all checks
            free_edge_confidence = 0.45
        if debug:
            print(
                f"  [NailBed] No confirmed boundary from {len(candidates)} candidate(s) "
                "— short nail, full mask used"
            )    

    return boundary_proj, end_is_distal, free_edge_present, free_edge_confidence