"""Free-edge boundary detection via profile analysis.

HISTORY OF BUGS AND FIXES (to understand the choices here):
══════════════════════════════════════════════════════════════════════

v6 ISSUE A (fixed here): Band extends to lunula via dip tolerance
  Fixed by: no dip tolerance (stop on first below-threshold BWC slice).

v6 ISSUE B (fixed here): Spurious L-drop on < 1 L-unit noise
  v6 used band_entry_L with budget = 0.8 × L_std_slices.
  When slices are nearly uniform, L_std_slices ≈ 0.4 → budget ≈ 0.3.
  This fired on float noise (0.3-unit difference).
  Fixed by: floor of 4.0: budget = max(4.0, 0.8 × L_std_slices).

v7 ISSUE (introduced and fixed here): Wrong budget base (pixel_L_std)
  v7 changed to pixel_L_std with 1.2 multiplier → budget = 21-34 units.
  But actual free-edge → nail-bed slice-mean drop is only 3-8 units.
  Budget > drop → L-drop never fires → band extends to 89% → REJECTED.
  Fixed by: reverting to slice-mean L_std (not pixel_L_std).

v7 ISSUE (fixed here): Wrong reference for L-drop (band_entry vs peak)
  band_entry_L is at the first qualified slice = the transition zone,
  where slice mean L ≈ mix of bright/dark = lower than peak.
  This made the limit too low → spurious triggers (v6 Nail 3: 0.3 drop).
  Fixed by: band_peak_L = rolling max L seen during band traversal.
  Peak is always at the brightest slice (white tip), well above nail bed.

v7/v8 ISSUE (fixed here): End selection fails when lunula BWC ≈ free edge BWC
  When both ends have equal BWC (e.g., 0.70 vs 0.70), taper fallback
  picks wrong end. Band enters at lunula → extends to 89%.
  Fixed by: combined score = bc × mean_L per end. Free edge = high L + high BWC.
  Lunula = moderate BWC + much lower mean L. Score correctly separates them.

══════════════════════════════════════════════════════════════════════
CORRECT FORMULA DERIVATION (from v6 debug data):

  Actual slice-mean L drops at free-edge → nail-bed boundary:
    Nail 1: 3.7 units,  Nail 4: 5.1 units,  Nail 5: 5.5 units
  
  L_std_slices for these nails: 3.0 – 4.0
  
  L_drop_budget (SNR-adaptive):
    SNR = (peak_L − mean_L) / (L_std_slices + 1e-6)
    SNR >= 1.5  → max(4.0, 0.8 × L_std_slices)   [original, avoids float noise]
    SNR <  1.5  → max(3.0, 0.6 × L_std_slices)   [softer, for low-contrast nails]
    - budget = 4.0 for most well-lit nails (floor dominates)
    - 3.7 drop: band_peak > band_entry, so peak-referenced drop > 4.0 ✓
    - 0.3 spurious drop: does NOT trigger because 0.3 << 3.0 ✓
    - dark/flat nails: softer floor detects real 2–3 unit drops ✓
══════════════════════════════════════════════════════════════════════
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _region_bright_cov(
    proj_major: np.ndarray,
    proj_minor: np.ndarray,
    L_raw: np.ndarray,
    bright_thresh: float,
    r_start: float,
    r_end: float,
) -> Tuple[float, float]:
    """Return (bright_width_cov, mean_L) for pixels in [r_start, r_end)."""
    mask = (proj_major >= r_start) & (proj_major < r_end)
    if mask.sum() < 4:
        return 0.0, 0.0
    minor_v = proj_minor[mask]
    L_vals  = L_raw[mask]
    bright  = L_vals > bright_thresh
    mL      = float(np.mean(L_vals))
    if bright.sum() < 4:
        return 0.0, mL
    bwc = float(
        (minor_v[bright].max() - minor_v[bright].min()) /
        (minor_v.max() - minor_v.min() + 1e-6)
    )
    return bwc, mL


def detect_free_edge_boundary(
    nail_yx: np.ndarray,
    proj_major: np.ndarray,
    proj_minor: np.ndarray,
    lab_img: np.ndarray,
    debug: bool = False,
    major_axis: Optional[np.ndarray] = None,
) -> Tuple[Optional[float], float, bool]:
    """Detect the nail-bed / free-edge boundary.

    Gradient energy improves boundary localization under uniform lighting.

    Returns:
        boundary_proj : inner edge of the bright band (projection units), or None.
        score         : confidence in [0, 1].
        end_is_distal : True = free edge at MAX end, False = free edge at MIN end.
    """

    min_proj  = float(np.min(proj_major))
    max_proj  = float(np.max(proj_major))
    axis_span = max_proj - min_proj

    if axis_span < 20.0 or len(nail_yx) < 100:
        return None, 0.0, True

    # ── Slice profiles ────────────────────────────────────────────────────────
    N       = int(np.clip(round(axis_span / 5.0), 40, 60))
    edges   = np.linspace(min_proj, max_proj, N + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    L_raw     = lab_img[:, :, 0][nail_yx[:, 0], nail_yx[:, 1]].astype(np.float32)

    # ── Vectorised slice profiles (no Python loop) ────────────────────────────
    _eps     = 1e-6
    _bin_w   = axis_span / N
    _bin_idx = np.clip(
        np.floor((proj_major - min_proj) / _bin_w).astype(np.intp),
        0, N - 1,
    )

    _count   = np.bincount(_bin_idx, minlength=N).astype(np.float64)
    _sum_L   = np.bincount(_bin_idx, weights=L_raw.astype(np.float64), minlength=N)
    _valid6  = _count >= 6
    _valid4  = _count >= 4

    L_profile  = np.where(_valid6, _sum_L / (_count + _eps), np.nan)

    _minor_max = np.full(N, -np.inf)
    _minor_min = np.full(N,  np.inf)
    np.maximum.at(_minor_max, _bin_idx, proj_minor)
    np.minimum.at(_minor_min, _bin_idx, proj_minor)
    W_profile  = np.where(_valid6, _minor_max - _minor_min, np.nan)

    L_s = gaussian_filter1d(L_profile, sigma=1.0, mode="nearest")
    W_s = gaussian_filter1d(W_profile, sigma=1.0, mode="nearest")

    L_mean      = float(np.nanmean(L_s))
    L_std_s     = float(np.nanstd(L_s)) + 1e-6   # slice-mean std (small, 2–5 units)
    L_norm      = (L_s - L_mean) / L_std_s
    W_norm      = W_s / (np.nanmax(W_s) + 1e-6)

    # Raw per-slice mean L for the scan loop (fallback to L_mean for sparse slices)
    _l_mean_raw = np.where(_valid4, _sum_L / (_count + _eps), L_mean)

    # ── SNR-adaptive L_drop_budget ────────────────────────────────────────────
    # SNR = (peak slice-mean L − overall mean) / slice-mean std.
    # High-SNR nails (bright free edge against a darker nail bed) exhibit a
    # large, reliable L-drop at the free-edge boundary → standard floor=4.0 is
    # appropriate and avoids spurious triggers from float noise.
    # Low-contrast nails (dark or flat lighting, SNR < 1.5) have a compressed
    # luminance range — the true free-edge drop is only 2–3 units, so the
    # standard floor would prevent detection entirely.  Reducing the floor to
    # 3.0 and the coefficient to 0.6 preserves sensitivity without raising
    # false positives because BWC gating (min_slice_bwc) still enforces
    # geometric evidence of a bright band.
    # Biological rationale: low-contrast nails require softer drop thresholds.
    _peak_L = float(np.nanmax(L_s))
    _snr    = (_peak_L - L_mean) / (L_std_s + 1e-6)

    if _snr < 1.5:
        # Low-contrast nail: soften the L-drop requirement to avoid missing
        # a real free edge that produces only a shallow luminance gradient.
        L_drop_budget = max(3.0, min(0.6 * L_std_s, 12.0))
    else:
        # Standard case: floor=4.0 blocks spurious sub-unit float noise.
        # Using band_peak_L (not entry) as reference ensures the drop is
        # measured from the brightest free-edge slice, giving a reliable margin.
        L_drop_budget = max(4.0, min(0.8 * L_std_s, 12.0))

    # ── Inclusive bright_thresh ───────────────────────────────────────────────
    bright_thresh = float(min(np.percentile(L_raw, 70),
                              L_mean + 0.45 * L_std_s))

    # ── Vectorised per-slice BWC (precomputed; replaces _slice_bwc closure) ───
    _bright      = L_raw > bright_thresh
    _b_idx       = _bin_idx[_bright]
    _b_minor     = proj_minor[_bright]
    _b_max       = np.full(N, -np.inf)
    _b_min       = np.full(N,  np.inf)
    if _b_idx.size > 0:
        np.maximum.at(_b_max, _b_idx, _b_minor)
        np.minimum.at(_b_min, _b_idx, _b_minor)
    _b_count     = np.bincount(_b_idx, minlength=N).astype(np.float64) if _b_idx.size > 0 else np.zeros(N)
    _span_all    = np.where(_valid6, _minor_max - _minor_min, 0.0)
    bwc_profile  = np.where(
        (_b_count >= 4) & _valid6 & (_span_all > _eps),
        (_b_max - _b_min) / (_span_all + _eps),
        0.0,
    )

    # ── Per-slice intensity variance profile (vectorised) ──────────────────────
    # Free edge shows lower intensity variance due to translucent keratin.
    # Variance per slice: Var = E[X²] − E[X]²  (fully vectorised via bincount).
    _sum_L2      = np.bincount(
        _bin_idx, weights=(L_raw.astype(np.float64) ** 2), minlength=N
    )
    _mean_L_arr  = np.where(_valid6, _sum_L   / (_count + _eps), np.nan)
    _mean_L2_arr = np.where(_valid6, _sum_L2  / (_count + _eps), np.nan)
    var_profile  = np.maximum(_mean_L2_arr - _mean_L_arr ** 2, 0.0)

    # Baseline variance: median of proximal half (stable nail-bed region).
    # Floor at 1.0 to avoid division noise on uniform/dark regions.
    _prox_half   = var_profile[: N // 2]
    _base_finite = _prox_half[np.isfinite(_prox_half)]
    baseline_var = float(np.median(_base_finite)) if _base_finite.size > 0 else 1.0
    baseline_var = max(baseline_var, 1.0)
    # Per-slice ratio: free-edge slices should be < 0.6 (translucent keratin)
    variance_ratio = var_profile / (baseline_var + _eps)

    # ── LAB gradient energy per slice (vectorised) ────────────────────────────
    # Gradient energy improves boundary localization under uniform lighting.
    # Compute Sobel gradients on L channel and project onto major axis direction.
    # Used only as a supporting signal (never standalone).
    grad_profile = np.zeros(N, dtype=np.float64)
    if major_axis is not None:
        L_img = lab_img[:, :, 0].astype(np.float32)
        grad_x = cv2.Sobel(L_img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(L_img, cv2.CV_32F, 0, 1, ksize=3)

        # Project gradient vector onto major axis for each nail pixel
        gx_vals = grad_x[nail_yx[:, 0], nail_yx[:, 1]]
        gy_vals = grad_y[nail_yx[:, 0], nail_yx[:, 1]]
        # Gradient magnitude projected onto major axis
        grad_mag = np.abs(gx_vals * major_axis[0] + gy_vals * major_axis[1])

        # Aggregate gradient magnitude per slice using same binning
        _grad_sum = np.bincount(_bin_idx, weights=grad_mag.astype(np.float64), minlength=N)
        grad_profile = np.where(_valid6, _grad_sum / (_count + _eps), 0.0)
        grad_profile = gaussian_filter1d(grad_profile, sigma=1.0, mode="nearest")

    # Normalize gradient energy to 0–1
    _grad_max = float(np.max(grad_profile))
    grad_norm = grad_profile / (_grad_max + _eps) if _grad_max > _eps else grad_profile

    # Mean gradient energy near each end (used as tie-breaker when BWC is ambiguous).
    # Slices cover the outer 15 % of the nail at each end so the regions are
    # safely clear of the central nail body.
    _grad_has_signal = _grad_max > _eps
    if _grad_has_signal and N > 0:
        _distal_start = int(0.85 * N)
        _prox_end     = int(0.15 * N)
        grad_distal = float(np.nanmean(grad_norm[_distal_start:])) if _distal_start < N else 0.0
        grad_prox   = float(np.nanmean(grad_norm[:_prox_end]))     if _prox_end > 0   else 0.0
    else:
        grad_distal = grad_prox = 0.0

    # ── Combined-score end selection ──────────────────────────────────────────
    # score = BWC × mean_L per end.  Free edge = high BWC AND high L.
    # Lunula = similar BWC but meaningfully lower L.  Score separates them.
    distal_frac = 0.30
    bc_max, mL_max = _region_bright_cov(proj_major, proj_minor, L_raw, bright_thresh,
                                        max_proj - distal_frac * axis_span, max_proj)
    bc_min, mL_min = _region_bright_cov(proj_major, proj_minor, L_raw, bright_thresh,
                                        min_proj, min_proj + distal_frac * axis_span)

    score_max = bc_max * mL_max
    score_min = bc_min * mL_min

    # Taper orientation as fallback when both ends indistinct
    ef           = max(3, int(0.20 * N))
    taper_diff   = np.nanmean(W_norm[:ef]) - np.nanmean(W_norm[-ef:])
    taper_distal = taper_diff > 0

    bc_diff = abs(bc_max - bc_min)
    # Override taper ONLY when BWC clearly differs between ends (>= 0.12).
    # Mean-L alone is unreliable: the lunula can be as bright as the free edge.
    # A 0.12 BWC gap is a real geometric signal; a 0.01-0.02 gap is noise.
    if bc_diff >= 0.12:
        end_is_distal = score_max > score_min
        if debug and end_is_distal != taper_distal:
            print(f"[BoundaryDetect] Brightness overrides taper: "
                  f"bc_max={bc_max:.2f}(mL={mL_max:.0f}) "
                  f"bc_min={bc_min:.2f}(mL={mL_min:.0f})  "
                  f"scores: {score_max:.0f} vs {score_min:.0f}")
    else:
        # BWC is ambiguous (<0.12 gap): use gradient energy near the nail ends
        # as a supporting signal before falling back to taper orientation.
        # The free edge typically shows a stronger intensity transition than the
        # lunula / proximal fold, so higher end-gradient implies the distal end.
        if _grad_has_signal and abs(grad_distal - grad_prox) > 0.08:
            end_is_distal = grad_distal > grad_prox
            if debug:
                print(f"[BoundaryDetect] Gradient tie-breaker: "
                      f"grad_distal={grad_distal:.3f}  grad_prox={grad_prox:.3f}  "
                      f"→ end_is_distal={end_is_distal}")
        else:
            end_is_distal = taper_distal

    # ── Region classification ─────────────────────────────────────────────────
    if end_is_distal:
        ds, de     = max_proj - distal_frac * axis_span, max_proj
        bright_cov = bc_max
    else:
        ds, de     = min_proj, min_proj + distal_frac * axis_span
        bright_cov = bc_min

    cs = min_proj + 0.30 * axis_span
    ce = cs + 0.40 * axis_span
    in_d = (proj_major >= ds) & (proj_major < de)
    in_c = (proj_major >= cs) & (proj_major < ce)

    if in_d.sum() < 10 or in_c.sum() < 10:
        return None, 0.0, end_is_distal

    delta_L_region = float(np.mean(L_raw[in_d]) - np.mean(L_raw[in_c]))

    # NOTE: No delta_L guard here. A negative delta_L on the selected end is
    # NOT reliable evidence of wrong end selection. When the free edge occupies
    # only 15-20% of the nail, the 30% classification window mixes free-edge
    # and nail-bed pixels, dragging the regional mean_L below the central mean.
    # A threshold-based flip on delta_L < -8 breaks correctly-oriented nails
    # (tested: image 2 nails 3 and 5 both had taper=correct but guard flipped
    # them to the wrong end). Nails where the guard appeared to help (image 2
    # nail 1) return None with or without the guard — the scan still exceeds
    # the free_edge_frac limit in either orientation. Safe to omit entirely.

    # ── Region score ──────────────────────────────────────────────────────────
    adaptive_denom = max(2.5, 1.5 * L_std_s)
    delta_score    = np.clip(delta_L_region / adaptive_denom, 0, 1)
    bwc_score      = np.clip(bright_cov / 0.55, 0, 1)

    # Gradient energy in the distal band (normalised 0–1)
    _d_mask_grad = (proj_major >= ds) & (proj_major < de)
    if _d_mask_grad.sum() >= 6 and _grad_max > _eps:
        _d_bin_idx = _bin_idx[_d_mask_grad]
        _d_grad_vals = grad_norm[_d_bin_idx]
        gradient_score = float(np.clip(np.mean(_d_grad_vals), 0, 1))
    else:
        gradient_score = 0.0

    # Variance score: fraction of distal-band slices with variance_ratio < 0.6.
    # Translucent free-edge keratin shows markedly lower intensity variance.
    _d_mask_var = (proj_major >= ds) & (proj_major < de)
    if _d_mask_var.sum() >= 6:
        _d_bin_var   = _bin_idx[_d_mask_var]
        _d_var_vals  = variance_ratio[np.clip(_d_bin_var, 0, N - 1)]
        _finite_mask = np.isfinite(_d_var_vals)
        if _finite_mask.sum() >= 3:
            variance_score = float(np.clip(
                np.mean(_d_var_vals[_finite_mask] < 0.6), 0, 1
            ))
        else:
            variance_score = 0.5  # neutral — insufficient data
    else:
        variance_score = 0.5  # neutral — insufficient data

    # Combined score: L-drop + BWC + gradient energy + variance signal
    region_score = (
        0.4 * delta_score +
        0.3 * bwc_score +
        0.2 * gradient_score +
        0.1 * variance_score
    )

    if debug:
        print(f"[BoundaryDetect] region_score={region_score:.2f}  "
              f"delta_L={delta_L_region:.1f}  bright_cov={bright_cov:.2f}  "
              f"gradient={gradient_score:.2f}  variance={variance_score:.2f}  "
              f"bc_max={bc_max:.2f}(mL={mL_max:.0f}) bc_min={bc_min:.2f}(mL={mL_min:.0f})  "
              f"bright_thresh={bright_thresh:.1f}  end_is_distal={end_is_distal}")

    if region_score < 0.28:
        return None, 0.0, end_is_distal

    # ── Adaptive per-slice BWC thresholds ─────────────────────────────────────
    min_slice_bwc   = float(np.clip(0.82 * bright_cov, 0.35, 0.55))
    inner_threshold = float(max(0.33, 0.60 * bright_cov))

    if debug:
        print(f"[BoundaryDetect] min_slice_bwc={min_slice_bwc:.2f}  "
              f"inner_threshold={inner_threshold:.3f}  "
              f"L_drop_budget={L_drop_budget:.2f}  L_std_s={L_std_s:.2f}")

    # ── Slice search ──────────────────────────────────────────────────────────
    margin       = 3
    search_range = (range(N - margin - 1, margin, -1) if end_is_distal
                    else range(margin, N - margin - 1))

    band_inner_idx    = None
    band_outer_idx    = None
    best_score        = 0.0
    in_band           = False
    band_peak_L       = -np.inf   # rolling max slice-mean L (confirmed free-edge slices only)
    last_good_idx     = None
    l_drop_consec     = 0         # consecutive slices below L-drop limit

    for i in search_range:
        if i < 4 or i + 4 >= N:
            continue

        delta_L_s  = np.nanmean(L_norm[i:i + 4]) - np.nanmean(L_norm[i - 4:i])
        width_drop = (W_norm[i - 4] - W_norm[i]) / (W_norm[i - 4] + 1e-6)

        slice_score = (
            0.65 * np.clip(delta_L_s  / 0.6,  0, 1) +
            0.35 * np.clip(width_drop / 0.25, 0, 1)
        )
        bwc_here = float(bwc_profile[i])
        slice_L  = float(_l_mean_raw[i])

        if not in_band:
            if slice_score < 0.20 or bwc_here < min_slice_bwc:
                continue
            in_band        = True
            band_outer_idx = i
            band_inner_idx = i
            last_good_idx  = i
            band_peak_L    = slice_L
            l_drop_consec  = 0
            best_score     = slice_score
        else:
            # L-drop guard: requires 2 consecutive below-limit slices.
            # A single dip (shadow, curvature, noise) within the free edge
            # drops below peak-budget but recovers on the next slice.
            # A real nail-bed entry stays dark for 2+ slices.
            # Peak is only updated on confirmed non-dip slices to keep the
            # reference anchored at the brightest true free-edge point.
            if slice_L < band_peak_L - L_drop_budget:
                l_drop_consec += 1
                if l_drop_consec >= 2:
                    if debug:
                        print(f"[BoundaryDetect] L-drop stop at slice {i}: "
                              f"slice_L={slice_L:.1f} < peak {band_peak_L:.1f} - {L_drop_budget:.2f}")
                    if last_good_idx is not None:
                        band_inner_idx = last_good_idx
                    break
                # Single dip — don't stop, don't update peak
                continue
            else:
                # Recovered from dip (or never dipped)
                l_drop_consec = 0
                band_peak_L   = max(band_peak_L, slice_L)

            # No dip tolerance on BWC: stop on first below-threshold slice.
            if bwc_here >= inner_threshold:
                band_inner_idx = i
                last_good_idx  = i
                best_score     = max(best_score, slice_score)
            else:
                if last_good_idx is not None:
                    band_inner_idx = last_good_idx
                break

    if band_inner_idx is None:
        if region_score > 0.75:
            shift    = 0.05 * axis_span
            fallback = max_proj - shift if end_is_distal else min_proj + shift
            return float(fallback), 0.45, end_is_distal
        return None, 0.0, end_is_distal

    # ── Variance 3-consecutive-slices confidence check ────────────────────────
    # Ensure variance condition holds for at least 3 consecutive slices
    # before confirming free-edge boundary.
    # A single low-variance slice can be noise; sustained low variance over
    # 3+ slices is consistent with translucent keratin.
    if band_outer_idx is not None and band_inner_idx is not None:
        _blo = min(band_outer_idx, band_inner_idx)
        _bhi = max(band_outer_idx, band_inner_idx) + 1
        _band_vr  = variance_ratio[_blo:_bhi]
        _low_mask = np.isfinite(_band_vr) & (_band_vr < 0.6)
        # Find max consecutive run of True values (vectorised)
        if _low_mask.size > 0 and _low_mask.any():
            _padded  = np.concatenate(([False], _low_mask, [False]))
            _diff    = np.diff(_padded.astype(np.int8))
            _starts  = np.where(_diff == 1)[0]
            _ends    = np.where(_diff == -1)[0]
            _max_run = int(np.max(_ends - _starts)) if _starts.size > 0 else 0
        else:
            _max_run = 0
        if _max_run < 3:
            # Variance criterion not sustained — treat as weaker detection
            best_score = best_score * 0.7
            if debug:
                print(f"[BoundaryDetect] Variance run={_max_run} < 3 — "
                      f"score penalised to {best_score:.2f}")

    # ── Band continuity check (single-slice band) ─────────────────────────────
    if band_outer_idx == band_inner_idx:
        check_idxs = (
            [band_inner_idx - 1, band_inner_idx - 2, band_inner_idx - 3]
            if end_is_distal else
            [band_inner_idx + 1, band_inner_idx + 2, band_inner_idx + 3]
        )
        neighbour_hits = sum(
            1 for j in check_idxs if 0 <= j < N and bwc_profile[j] > 0.22
        )
        if neighbour_hits < 1 and region_score < 0.65:
            return None, 0.0, end_is_distal

    boundary_proj = float(centers[band_inner_idx])

    # ── Minimum free-edge length validation ───────────────────────────────────
    # Rejects thin reflection bands falsely detected as free edge.
    distal_proj      = max_proj if end_is_distal else min_proj
    nail_length      = max_proj - min_proj
    free_edge_length = abs(distal_proj - boundary_proj)
    free_edge_ratio  = free_edge_length / (nail_length + 1e-6)

    if free_edge_ratio < 0.05:
        if debug:
            print(f"[BoundaryDetect] REJECTED: free_edge_ratio={free_edge_ratio:.3f} < 0.05 "
                  "(thin reflection band)")
        free_edge_present = False  # noqa: F841  (documents intent for callers)
        boundary_proj     = None
        return None, 0.0, end_is_distal

    # ── Free-edge-fraction constraint ────────────────────────────────────────
    free_edge_frac = (
        (max_proj - boundary_proj) / axis_span if end_is_distal
        else (boundary_proj - min_proj) / axis_span
    )

    if free_edge_frac > 0.45:
        if free_edge_frac <= 0.55 and region_score >= 0.40:
            if debug:
                print(f"[BoundaryDetect] Clamping free_edge_frac {free_edge_frac:.2f} → 0.45")
            boundary_proj = (max_proj - 0.45 * axis_span if end_is_distal
                             else min_proj + 0.45 * axis_span)
            free_edge_frac = 0.45
        else:
            if debug:
                print(f"[BoundaryDetect] REJECTED: free_edge_frac={free_edge_frac:.2f} > 0.55"
                      " (band reached proximal region)")
            return None, 0.0, end_is_distal

    elif free_edge_frac < 0.05:
        if free_edge_frac >= 0.02 and region_score >= 0.40:
            boundary_proj = (max_proj - 0.05 * axis_span if end_is_distal
                             else min_proj + 0.05 * axis_span)
            free_edge_frac = 0.05
        else:
            return None, 0.0, end_is_distal

    if debug:
        print(f"[BoundaryDetect] ACCEPTED boundary={boundary_proj:.2f}  "
              f"free_edge_frac={free_edge_frac:.2f}  score={best_score:.2f}")

    return float(boundary_proj), float(min(best_score, 1.0)), end_is_distal