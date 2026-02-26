#!/usr/bin/env python3
"""
Real-time nail bed extraction debug viewer.

Shows a live webcam feed with:
  - BLUE  overlay  : full nail mask (from YOLO)
  - GREEN overlay  : extracted nail bed region
  - YELLOW line    : detected free-edge boundary
  - CYAN  line     : PCA major axis
  - HUD text       : free_edge_present, confidence, ratios

Usage:
    python debug_nail_bed_realtime.py           # default webcam (index 0)
    python debug_nail_bed_realtime.py 1         # webcam index 1

Controls:
    Q        - quit
    S        - save screenshot  (debug_screenshot_N.jpg)
    D        - toggle per-frame debug prints
    P        - toggle polished-nail mode (skips boundary detection)
"""

import sys
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

from src.geometry_utils import (extract_nail_bed_overlay_data,
                                 _clean_mask_morphology,
                                 _build_curved_bed_mask,
                                 _orient_anatomical_axis)
from src.color_utils import _detect_polished_nail, _check_lighting_quality
from config import (
    MODEL_PATH,
    REALTIME_MASK_THRESHOLD,
    REALTIME_MIN_MASK_PIXELS,
    REALTIME_DETECTION_CONFIDENCE,
    MIN_NAIL_ASPECT_RATIO,
    MAX_NAIL_ASPECT_RATIO,
    NAIL_LAB_L_MIN, NAIL_LAB_L_MAX,
    NAIL_LAB_A_MIN, NAIL_LAB_A_MAX,
    NAIL_LAB_B_MIN, NAIL_LAB_B_MAX,
)

# ── Overlay colours (BGR) ─────────────────────────────────────────────────────
FULL_COLOR     = (200,  80,   0)   # blue-ish  — full nail
BED_COLOR      = (  0, 200,  60)   # green     — nail bed
BOUNDARY_COLOR = (  0, 255, 255)   # yellow    — free-edge boundary
AXIS_COLOR     = (255, 255,   0)   # cyan      — PCA major axis
TEXT_COLOR     = (255, 255, 255)
SHADOW_COLOR   = (  0,   0,   0)

FULL_ALPHA = 0.30
BED_ALPHA  = 0.50
AXIS_HALF  = 100   # px, half-length of PCA axis line

# Per-detection YOLO confidence gate (on top of the model-level threshold).
# Masks below this score are discarded even if YOLO outputs them.
MIN_DETECTION_CONF = 0.70


# ── Nail validator ────────────────────────────────────────────────────────────

def _is_valid_nail(frame: np.ndarray, nail_mask: np.ndarray,
                  debug: bool = False,
                  lab_frame: np.ndarray = None) -> tuple:
    """Return (is_valid: bool, reason: str) for a candidate nail mask.

    Checks (in order, cheapest first):
    1. Minimum pixel count  (REALTIME_MIN_MASK_PIXELS)
    2. Solidity             (area / convex-hull area >= 0.55)  rejects fragments
    3. Aspect ratio         (PCA length/width within nail range)
    4. LAB colour median    (within nail-plate colour bounds)
    """
    px = int(nail_mask.sum() // 255)
    if px < REALTIME_MIN_MASK_PIXELS:
        return False, f"too small ({px} px < {REALTIME_MIN_MASK_PIXELS})"

    clean = _clean_mask_morphology(nail_mask)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "no contour after cleaning"

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    # --- Solidity: rejects fragmented / C-shaped detections ---
    hull  = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity  = area / hull_area if hull_area > 0 else 0.0
    if solidity < 0.55:
        return False, f"low solidity ({solidity:.2f} < 0.55)"

    # --- Aspect ratio via PCA ---
    nail_yx = np.argwhere(clean == 255)
    nail_xy = nail_yx[:, ::-1].astype(np.float64)
    centroid = nail_xy.mean(axis=0)
    cov = np.cov((nail_xy - centroid).T)
    eigvals = np.linalg.eigvalsh(cov)      # ascending
    if eigvals[0] <= 0:
        return False, "degenerate covariance (line mask)"
    aspect = float(np.sqrt(eigvals[-1] / eigvals[0]))  # eigenvalue ratio ≈ (length/width)²
    # eigenvalue ratio is (length/width)^2, convert to linear
    aspect_linear = float(np.sqrt(max(eigvals[-1], 1e-9) / max(eigvals[0], 1e-9)))
    if aspect_linear < MIN_NAIL_ASPECT_RATIO or aspect_linear > MAX_NAIL_ASPECT_RATIO:
        return False, f"bad aspect ratio ({aspect_linear:.2f})"

    # --- LAB colour check (median of masked region) ---
    lab = lab_frame if lab_frame is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_vals = lab[:, :, 0][nail_yx[:, 0], nail_yx[:, 1]]
    a_vals = lab[:, :, 1][nail_yx[:, 0], nail_yx[:, 1]]
    b_vals = lab[:, :, 2][nail_yx[:, 0], nail_yx[:, 1]]
    med_L  = float(np.median(L_vals))
    med_a  = float(np.median(a_vals))
    med_b  = float(np.median(b_vals))

    if not (NAIL_LAB_L_MIN <= med_L <= NAIL_LAB_L_MAX):
        return False, f"L out of range ({med_L:.0f} not in [{NAIL_LAB_L_MIN},{NAIL_LAB_L_MAX}])"
    if not (NAIL_LAB_A_MIN <= med_a <= NAIL_LAB_A_MAX):
        return False, f"a* out of range ({med_a:.0f} not in [{NAIL_LAB_A_MIN},{NAIL_LAB_A_MAX}])"
    if not (NAIL_LAB_B_MIN <= med_b <= NAIL_LAB_B_MAX):
        return False, f"b* out of range ({med_b:.0f} not in [{NAIL_LAB_B_MIN},{NAIL_LAB_B_MAX}])"

    return True, "ok"

def blend_mask(canvas: np.ndarray, mask: np.ndarray,
               color: tuple, alpha: float) -> None:
    """Blend a solid-colour region onto canvas in-place."""
    overlay = canvas.copy()
    overlay[mask == 255] = color
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)


def put_text(img, text, pos, scale=0.6, color=TEXT_COLOR, thickness=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, SHADOW_COLOR, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),     cv2.FONT_HERSHEY_SIMPLEX,
                scale, color,          thickness,     cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, nail_mask: np.ndarray,
                 data: dict, nail_id: int, text_y: int) -> int:
    """Draw full-nail + bed overlays and HUD text. Returns updated text_y."""
    LINE = 26

    # Full nail — blue
    blend_mask(frame, nail_mask, FULL_COLOR, FULL_ALPHA)

    # Nail bed — green
    bed_mask = data["nail_bed_mask"]
    blend_mask(frame, bed_mask, BED_COLOR, BED_ALPHA)

    # PCA axis
    centroid   = data["centroid"]
    major_axis = data["major_axis"]
    minor_axis = data["minor_axis"]
    if centroid is not None and major_axis is not None:
        cx, cy = int(centroid[0]), int(centroid[1])
        dx, dy = major_axis
        pt1 = (int(cx - dx * AXIS_HALF), int(cy - dy * AXIS_HALF))
        pt2 = (int(cx + dx * AXIS_HALF), int(cy + dy * AXIS_HALF))
        cv2.line(frame, pt1, pt2, AXIS_COLOR, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, AXIS_COLOR, -1)

        # Draw arrow toward detected distal end
        _end_is_distal = data.get("end_is_distal", True)
        _arrow_dir = major_axis if _end_is_distal else -major_axis
        arrow_tip = (
            int(cx + _arrow_dir[0] * AXIS_HALF * 0.8),
            int(cy + _arrow_dir[1] * AXIS_HALF * 0.8),
        )
        arrow_base = (
            int(cx + _arrow_dir[0] * AXIS_HALF * 0.5),
            int(cy + _arrow_dir[1] * AXIS_HALF * 0.5),
        )
        cv2.arrowedLine(frame, arrow_base, arrow_tip, (0, 255, 255), 2,
                        tipLength=0.4)
        # Label which end is free edge
        put_text(frame, "FE", arrow_tip, scale=0.4, color=(0, 255, 255))

    # Free-edge boundary line
    bp = data["boundary_proj"]
    _fe_present = data.get("free_edge_present", None)
    # Only draw boundary line when free edge was actually detected.
    # When free_edge_present is False (trimmed nail) or None (uncertain),
    # boundary_proj may still exist from prior frames via smoothing —
    # do not draw it, as it will be misleading.
    if bp is not None and _fe_present is True and centroid is not None:
        half_w = data.get("full_nail_width_px", data["width_px"]) / 2.0
        bxy    = centroid + bp * major_axis

        # Sanity check: boundary should be on the distal side of centroid
        # proj of boundary relative to centroid = bp itself
        # If end_is_distal=True, boundary should have bp > 0 (distal = high proj)
        # If end_is_distal=False, boundary should have bp < 0 (distal = low proj)
        _end_is_distal_bd = data.get("end_is_distal", True)
        _bp_correct_side = (bp > 0) if _end_is_distal_bd else (bp < 0)

        # Color: yellow = correct side, red = wrong side (boundary on proximal end)
        _line_color = BOUNDARY_COLOR if _bp_correct_side else (0, 0, 255)

        pt1b = tuple((bxy + half_w * minor_axis).astype(int))
        pt2b = tuple((bxy - half_w * minor_axis).astype(int))
        cv2.line(frame, pt1b, pt2b, _line_color, 2, cv2.LINE_AA)

        if not _bp_correct_side:
            put_text(frame, "BOUNDARY ON WRONG SIDE", (14, text_y),
                     scale=0.5, color=(0, 0, 255))
            text_y += LINE

    # HUD text
    fe   = data["free_edge_present"]
    conf = data["free_edge_confidence"]
    label = "A: free edge" if fe else ("B: short nail" if fe is False else "polished")
    color = (0, 255, 100) if fe else (180, 180, 180)

    put_text(frame, f"Nail {nail_id} — {label}  conf={conf:.2f}",
             (14, text_y), color=color)
    text_y += LINE
    # Show which method determined the orientation
    _n_meth_debug = data.get("n_methods", 0)
    _has_brightness = data.get("brightness_method_used", False)
    _orient_source = "brightness" if _has_brightness else (
        "geometry" if _n_meth_debug >= 2 else "color-cue"
    )
    _eid = data.get("end_is_distal")
    _eid_str = "distal=MAX proj" if _eid else "distal=MIN proj"
    put_text(frame, f"  axis orient: {_eid_str}  [{_orient_source}]",
             (14, text_y), scale=0.45, color=(200, 200, 100))
    text_y += LINE
    # Voting method count
    n_meth = data.get("n_methods", 0)
    if n_meth > 0:
        meth_color = (0, 255, 100) if n_meth >= 3 else (0, 200, 255) if n_meth >= 2 else (100, 100, 255)
        put_text(frame, f"  methods agreed: {n_meth}/5",
                 (14, text_y), scale=0.5, color=meth_color)
        text_y += LINE
    # Bed vs full pixel ratio
    total_px = int(nail_mask.sum() // 255)
    bed_px   = int(bed_mask.sum() // 255)
    ratio    = bed_px / total_px if total_px > 0 else 0.0
    put_text(frame, f"  bed/full = {bed_px}/{total_px} ({ratio:.1%})",
             (14, text_y), scale=0.5)
    text_y += LINE

    # Nail bed shape: length, width, aspect ratio
    _nb_len = data.get("length_px", 0.0)
    _nb_wid = data.get("width_px", 0.0)
    if _nb_len > 0 and _nb_wid > 0:
        _nb_ratio = _nb_len / _nb_wid
        _ratio_color = (0, 255, 100) if 0.8 <= _nb_ratio <= 2.0 else (0, 128, 255)
        put_text(frame, f"  bed L={_nb_len:.0f}  W={_nb_wid:.0f}  ratio={_nb_ratio:.2f}",
                 (14, text_y), scale=0.5, color=_ratio_color)
        text_y += LINE

    # Midpoint width (anatomically standardised measurement)
    mid_w = data.get("width_midpoint_px")
    if mid_w is not None:
        put_text(frame, f"  width_mid = {mid_w:.1f} px",
                 (14, text_y), scale=0.5)
        text_y += LINE

    return text_y


# ── Main loop ─────────────────────────────────────────────────────────────────

# Temporal smoothing: keep last N boundary_proj values per nail slot.
# Using median to reject outlier frames.
SMOOTH_FRAMES = 7


# ── Axis smoother ─────────────────────────────────────────────────────────────

class AxisSmoother:
    """Temporal EMA smoother for PCA axes.

    Eliminates frame-to-frame axis jitter caused by:
      - Near-square nails (eigenvalue ambiguity → 90° flips)
      - YOLO mask boundary noise
      - Arbitrary eigenvector sign changes

    Strategy:
      1. Exponential moving-average on the 2×2 covariance matrix.
         This smooths the eigenvalues, preventing the near-equal crossover
         that causes eigenvector flips.
      2. On the first WARMUP frames, accumulate covariance without
         producing axes (lets the EMA converge).
      3. After warmup, apply _orient_anatomical_axis EVERY frame on the
         smoothed eigenvectors.  EMA provides stable vectors; orientation
         corrects their direction.  These are independent operations —
         orientation is never cached or locked.
      4. Resolve sign and swap ambiguity by comparing to the previous
         frame's reference (dot-product tracking).
    """

    WARMUP_FRAMES = 5

    def __init__(self, cov_alpha: float = 0.25, centroid_alpha: float = 0.30):
        self.cov_alpha = cov_alpha
        self.centroid_alpha = centroid_alpha
        self.cov_smooth: np.ndarray | None = None
        self.centroid_smooth: np.ndarray | None = None
        self.ref_major: np.ndarray | None = None
        self.ref_minor: np.ndarray | None = None
        self.n_updates = 0

    def update(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        lab_frame: np.ndarray | None = None,
    ) -> tuple:
        """Return (centroid, major_axis, minor_axis) or triple-None during warmup."""
        clean = _clean_mask_morphology(mask)
        nail_yx = np.argwhere(clean == 255)
        if len(nail_yx) < 50:
            # Too few pixels — return last stable axes if available
            if self.ref_major is not None:
                return (self.centroid_smooth.copy(),
                        self.ref_major.copy(),
                        self.ref_minor.copy())
            return None, None, None

        nail_xy = nail_yx[:, ::-1].astype(np.float64)
        centroid_raw = nail_xy.mean(axis=0)
        centered = nail_xy - centroid_raw
        cov_raw = np.cov(centered.T)  # 2×2 symmetric

        # ── EMA on covariance matrix and centroid ─────────────────────────
        if self.cov_smooth is None:
            self.cov_smooth = cov_raw.copy()
            self.centroid_smooth = centroid_raw.copy()
        else:
            a = self.cov_alpha
            self.cov_smooth = a * cov_raw + (1.0 - a) * self.cov_smooth
            b = self.centroid_alpha
            self.centroid_smooth = b * centroid_raw + (1.0 - b) * self.centroid_smooth

        self.n_updates += 1

        # During warmup just accumulate — don't produce axes yet
        if self.n_updates < self.WARMUP_FRAMES:
            return None, None, None

        # ── Eigen-decomposition on the smoothed covariance ────────────────
        eigvals, eigvecs = np.linalg.eigh(self.cov_smooth)
        major = eigvecs[:, -1].astype(np.float64)  # largest eigenvalue
        minor = eigvecs[:, 0].astype(np.float64)

        # ── Resolve swap ambiguity (eigenvalue crossover) ─────────────────
        if self.ref_major is not None:
            dot_mm = abs(float(np.dot(major, self.ref_major)))
            dot_mn = abs(float(np.dot(major, self.ref_minor)))
            if dot_mn > dot_mm:
                major, minor = minor.copy(), major.copy()

        # ── Resolve sign ambiguity ────────────────────────────────────────
        if self.ref_major is not None:
            if np.dot(major, self.ref_major) < 0:
                major = -major
            if np.dot(minor, self.ref_minor) < 0:
                minor = -minor

        # ── Apply anatomical orientation EVERY frame ──────────────────────
        # EMA gives stable vectors.  Orientation corrects direction.
        # These are independent operations — never cache or lock this.
        major, minor = _orient_anatomical_axis(
            image, clean, self.centroid_smooth, major, minor, lab_frame
        )

        self.ref_major = major.copy()
        self.ref_minor = minor.copy()

        return self.centroid_smooth.copy(), major.copy(), minor.copy()

    def reset(self):
        """Clear all state (e.g. when mode changes)."""
        self.cov_smooth = None
        self.centroid_smooth = None
        self.ref_major = None
        self.ref_minor = None
        self.n_updates = 0


def main():
    cam_index  = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    debug_mode = False
    polished   = False
    screenshot = 0

    # Per-nail-slot boundary history  {slot_index: deque([proj, ...])}
    boundary_history: dict = {}
    # Per-nail-slot orientation history for temporal smoothing of end_is_distal
    orientation_history: dict = {}  # slot → deque of bool
    # Per-nail-slot last stable orientation  {slot → bool}
    stable_orientation: dict = {}
    # Per-nail-slot PCA axis smoother (covariance-matrix EMA)
    axis_smoothers: dict = {}       # slot → AxisSmoother

    print("\n" + "=" * 60)
    print("  Nail Bed DEBUG — Real-Time Viewer")
    print(f"  Camera index : {cam_index}")
    print("  Q = quit  |  S = screenshot  |  D = debug prints")
    print("  P = toggle polished mode (skips boundary detection)")
    print("=" * 60 + "\n")

    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"✗ Cannot open camera {cam_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Frame read failed — exiting")
            break

        canvas = frame.copy()

        # Compute LAB once per frame; reused by validation and overlay extraction
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        # ── Run YOLO ──────────────────────────────────────────────────────────
        results = model(frame, conf=REALTIME_DETECTION_CONFIDENCE,
                        verbose=False)[0]

        text_y = 35
        valid_nails = 0
        active_slots = set()

        if results.masks is not None:
            h, w = frame.shape[:2]

            for idx, (seg, conf_score) in enumerate(
                    zip(results.masks.data, results.boxes.conf), start=1):

                # Build binary mask
                mask_f = seg.cpu().numpy()
                mask_r = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LINEAR)
                nail_mask = (mask_r > REALTIME_MASK_THRESHOLD).astype(np.uint8) * 255

                # Gate 1: per-detection YOLO confidence
                if float(conf_score) < MIN_DETECTION_CONF:
                    if debug_mode:
                        print(f"  [Filter] det {idx} skipped — conf {float(conf_score):.2f} < {MIN_DETECTION_CONF}")
                    continue

                # Gate 2: minimum pixel count (fast, before expensive checks)
                if nail_mask.sum() // 255 < REALTIME_MIN_MASK_PIXELS:
                    if debug_mode:
                        print(f"  [Filter] det {idx} skipped — mask too small")
                    continue

                # Gate 3: shape + colour validation
                valid, reason = _is_valid_nail(frame, nail_mask, debug=debug_mode,
                                               lab_frame=lab_frame)
                if not valid:
                    if debug_mode:
                        print(f"  [Filter] det {idx} rejected — {reason}")
                    continue

                # ── Nail bed extraction ───────────────────────────────────────
                # Auto-detect polish from LAB pixels (overrides manual toggle)
                _auto_polished = False
                _polish_conf = 0.0
                _clean_for_polish = _clean_mask_morphology(nail_mask)
                _polish_yx = np.argwhere(_clean_for_polish == 255)
                if len(_polish_yx) > 20:
                    _polish_lab = lab_frame[_polish_yx[:, 0], _polish_yx[:, 1]].astype(np.uint8)
                    _auto_polished, _polish_conf = _detect_polished_nail(_polish_lab)

                # Check lighting quality
                _light_ok, _light_reason = True, "ok"
                if len(_polish_yx) > 20:
                    _light_ok, _light_reason = _check_lighting_quality(
                        lab_frame[_polish_yx[:, 0], _polish_yx[:, 1]].astype(np.uint8)
                    )

                _effective_polished = polished or _auto_polished

                # ── Compute temporally-smoothed PCA axes ──────────────────
                slot = idx
                active_slots.add(slot)
                if slot not in axis_smoothers:
                    axis_smoothers[slot] = AxisSmoother()

                _sm_c, _sm_maj, _sm_min = axis_smoothers[slot].update(
                    nail_mask, frame, lab_frame
                )

                # Pass smoothed axes when available (after warmup),
                # otherwise let the pipeline compute its own.
                _precomp = (_sm_c, _sm_maj, _sm_min) if _sm_c is not None else None

                data = extract_nail_bed_overlay_data(
                    frame, nail_mask, is_polished=_effective_polished,
                    debug=debug_mode, lab_frame=lab_frame,
                    precomputed_axes=_precomp,
                )

                # ── Temporal smoothing of boundary_proj ───────────────────
                if slot not in boundary_history:
                    boundary_history[slot] = deque(maxlen=SMOOTH_FRAMES)

                raw_proj    = data["boundary_proj"]
                raw_end_dis = data["end_is_distal"] if data["end_is_distal"] is not None else True
                _fe_now = data.get("free_edge_present", None)

                # ── Temporal smoothing of end_is_distal ────────────────────
                # Use majority vote over recent frames to prevent orientation
                # flipping from frame-to-frame noise in the scoring cues.
                # Hysteresis: require supermajority (7/9) to FLIP the current
                # stable orientation. Simple majority keeps the current one.
                ORIENT_FRAMES = 9
                FLIP_THRESHOLD = 7   # need 7 of 9 votes to flip orientation
                if slot not in orientation_history:
                    orientation_history[slot] = deque(maxlen=ORIENT_FRAMES)
                    # Pre-seed: fill the entire history with the first frame's
                    # result so orientation is immediately stable from frame 1.
                    for _ in range(ORIENT_FRAMES):
                        orientation_history[slot].append(raw_end_dis)
                else:
                    orientation_history[slot].append(raw_end_dis)
                _eid_votes = list(orientation_history[slot])
                _eid_true  = sum(1 for v in _eid_votes if v is True)
                _eid_false = len(_eid_votes) - _eid_true
                # Current stable value from previous frame (or raw for first)
                _prev_stable = stable_orientation.get(slot, raw_end_dis)
                # Hysteresis: only flip if the OTHER side has supermajority
                if _prev_stable:
                    # Currently True; only flip to False if False has >= FLIP_THRESHOLD
                    stable_end_dis = False if _eid_false >= FLIP_THRESHOLD else True
                else:
                    # Currently False; only flip to True if True has >= FLIP_THRESHOLD
                    stable_end_dis = True if _eid_true >= FLIP_THRESHOLD else False
                if stable_end_dis != raw_end_dis and debug_mode:
                    print(f"  [Smooth] Slot {slot}: orientation overridden by "
                          f"hysteresis ({_eid_true}T/{_eid_false}F, need {FLIP_THRESHOLD} to flip) "
                          f"raw={raw_end_dis} → stable={stable_end_dis}")
                # Use the stabilised orientation for everything downstream
                data["end_is_distal"] = stable_end_dis
                stable_orientation[slot] = stable_end_dis

                if _fe_now is True and raw_proj is not None:
                    # Only store boundary when free edge is confirmed present.
                    # Storing uncertain/absent frames contaminates the median.
                    boundary_history[slot].append((stable_end_dis, raw_proj))
                elif _fe_now is False:
                    # Trimmed nail confirmed — clear history so stale boundary
                    # from previous frames doesn't appear as a yellow line.
                    boundary_history[slot].clear()

                hist = boundary_history[slot]

                # If orientation flipped from previous frame, clear history
                # (mixing opposite-orientation projections cancels the boundary)
                if len(hist) > 0:
                    last_eid = hist[-1][0]
                    if last_eid != stable_end_dis:
                        boundary_history[slot].clear()
                        if debug_mode:
                            print(f"  [Smooth] Slot {slot}: orientation flipped "
                                  f"({last_eid}→{stable_end_dis}) — history cleared")

                matching = [bp for (eid, bp) in boundary_history[slot]
                            if eid == stable_end_dis]
                if len(matching) >= 2:
                    smoothed_proj = float(np.median(matching))
                    data["boundary_proj"] = smoothed_proj
                    if data["centroid"] is not None and data["major_axis"] is not None:
                        nail_yx = np.argwhere(_clean_mask_morphology(nail_mask) == 255)
                        _width_sm = data.get("full_nail_width_px", data["width_px"])
                        in_bed = _build_curved_bed_mask(
                            nail_yx, data["centroid"],
                            data["major_axis"], data["minor_axis"],
                            smoothed_proj, _width_sm, stable_end_dis
                        )
                        if in_bed.sum() >= 10:
                            nb_mask = np.zeros_like(nail_mask)
                            nb_mask[nail_yx[in_bed][:, 0],
                                    nail_yx[in_bed][:, 1]] = 255
                            data["nail_bed_mask"] = nb_mask

                text_y = draw_overlay(canvas, nail_mask, data, idx, text_y)

                # Show polish/lighting diagnostics on HUD
                if _auto_polished:
                    put_text(canvas, f"  \u26a0 Polish auto-detected (conf={_polish_conf:.2f})",
                             (14, text_y), scale=0.45, color=(0, 200, 255))
                    text_y += 20
                if not _light_ok:
                    _lq_msgs = {
                        'overexposed':     '\u26a0 Too bright',
                        'underexposed':    '\u26a0 Too dark',
                        'glare':           '\u26a0 Glare detected',
                        'uneven_lighting': '\u26a0 Uneven lighting',
                    }
                    put_text(canvas, f"  {_lq_msgs.get(_light_reason, _light_reason)}",
                             (14, text_y), scale=0.45, color=(0, 128, 255))
                    text_y += 20

                valid_nails += 1

        # Prune stale slots (nails that left the frame)
        for slot in list(boundary_history):
            if slot not in active_slots:
                boundary_history.pop(slot, None)
                orientation_history.pop(slot, None)
                stable_orientation.pop(slot, None)
                axis_smoothers.pop(slot, None)

        if valid_nails == 0:
            put_text(canvas, "No nails detected", (14, 50),
                     scale=0.9, color=(0, 50, 255))

        # ── Footer ────────────────────────────────────────────────────────────
        h_c = canvas.shape[0]
        mode_str = "[POLISHED]" if polished else "[NORMAL]"
        dbg_str  = "[DEBUG ON]" if debug_mode else ""
        put_text(canvas,
                 f"Q=quit  S=screenshot  D=debug  P=polished  {mode_str} {dbg_str}",
                 (10, h_c - 15), scale=0.5, color=(180, 180, 180))

        # ── Legend ────────────────────────────────────────────────────────────
        items = [
            (FULL_COLOR,     "Full nail"),
            (BED_COLOR,      "Nail bed"),
            (BOUNDARY_COLOR, "Free-edge boundary"),
            (AXIS_COLOR,     "PCA axis"),
        ]
        ly = h_c - 15 - len(items) * 22
        for color, label in items:
            cv2.rectangle(canvas, (10, ly - 10), (24, ly + 4), color, -1)
            put_text(canvas, label, (30, ly), scale=0.48, color=color)
            ly += 22

        cv2.imshow("Nail Bed Debug — Realtime", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('s'), ord('S')):
            screenshot += 1
            fname = f"debug_screenshot_{screenshot}.jpg"
            cv2.imwrite(fname, canvas)
            print(f"  Screenshot saved → {fname}")
        elif key in (ord('d'), ord('D')):
            debug_mode = not debug_mode
            print(f"  Debug prints: {'ON' if debug_mode else 'OFF'}")
        elif key in (ord('p'), ord('P')):
            polished = not polished
            boundary_history.clear()  # reset smoothing when mode changes
            for _s in axis_smoothers.values():
                _s.reset()
            axis_smoothers.clear()
            print(f"  Polished mode: {'ON' if polished else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("  Camera closed.")


if __name__ == "__main__":
    main()
