#!/usr/bin/env python3
"""Real-time nail segmentation and analysis using webcam."""

import cv2
import numpy as np
import logging
from collections import deque
from ultralytics import YOLO

from src.geometry_utils import extract_geometry, extract_geometry_nail_bed_with_diagnostics
from src.color_utils import extract_nail_color
from src.calibration import MeasurementCalibrator
from config import (
    MODEL_PATH,
    DEFAULT_PIXELS_PER_MM,
    LOG_LEVEL,
    MIN_NAIL_ASPECT_RATIO,
    MAX_NAIL_ASPECT_RATIO,
    MIN_NAIL_LENGTH_MM,
    MAX_NAIL_LENGTH_MM,
    MIN_NAIL_AREA_MM2,
    NAIL_LAB_L_MIN,
    NAIL_LAB_L_MAX,
    NAIL_LAB_A_MIN,
    NAIL_LAB_A_MAX,
    NAIL_LAB_B_MIN,
    NAIL_LAB_B_MAX,
    REALTIME_DETECTION_CONFIDENCE,
    REALTIME_MASK_THRESHOLD,
    REALTIME_MIN_MASK_PIXELS,
    CENTROID_DISTANCE_THRESHOLD,
    ASPECT_RATIO_SIMILARITY_THRESHOLD,
)

# Nail width cap (imported separately so older configs without it don't crash)
try:
    from config import MAX_NAIL_WIDTH_MM
except ImportError:
    MAX_NAIL_WIDTH_MM = 22.0

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class RealtimeNailAnalyzer:
    """Real-time nail analysis from webcam feed."""
    
    def __init__(self, model_path: str = MODEL_PATH, pixels_per_mm: float = DEFAULT_PIXELS_PER_MM):
        """Initialize real-time analyzer.
        
        Args:
            model_path: Path to YOLOv8 model
            pixels_per_mm: Calibration factor for mm conversion
        """
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.calibrator = MeasurementCalibrator(pixels_per_mm=pixels_per_mm)
        self.nail_count = 0
        self.current_nails = []
        self.seen_nails = set()  # Track which nails have been printed

        # ── Rolling confidence print gate ─────────────────────────────────
        # Tracks per-nail-slot boundary confidence over recent frames.
        # We print when the rolling MEAN confidence (over 5 frames) exceeds
        # the threshold. This is far more tolerant of small hand movements
        # than requiring N consecutive perfect frames: a single shaky frame
        # just dips the rolling mean slightly rather than resetting a counter.
        # Slot key = centroid snapped to 40px grid (stable across small moves).
        self._print_conf_history: dict = {}   # slot → deque of conf values
        self._printed_slots: set = set()      # slots already printed this session
        self._PRINT_CONF_WINDOW   = 5          # frames to average over
        self._PRINT_CONF_THRESHOLD = 0.40      # mean confidence needed to print
        self._PRINT_MIN_FRAMES    = 3          # minimum frames before printing
        
        # Temporal smoothing — 5-frame window balances stability with responsiveness.
        self.nail_history = deque(maxlen=5)  # 5 frames (~0.17 s at 30 fps)

        # --- Performance caches (Goals 1, 4, 5) ---
        # Goal 1: last raw YOLO masks; reused for the 2 frames between YOLO runs.
        self._cached_raw_masks = None
        # Per-mask YOLO confidence (parallel to _cached_raw_masks).
        # Applied as a per-mask floor on every frame, including cached ones.
        self._cached_confs: np.ndarray = np.array([], dtype=np.float32)
        # Goal 4 & 5: per-nail measurement cache keyed by (snap_cx, snap_cy).
        # Stores last computed nail-bed values; reused when mask area changes < 10%.
        # This freezes the boundary projection on stable frames and avoids
        # redundant PCA + slice analysis every frame.
        self._nail_perf_cache: dict = {}
        self._nail_perf_cache_max_size = 50  # max cached nail entries

        # Boundary-projection temporal smoothing.
        # Prevents frame-to-frame jitter in boundary detection.
        # Per-slot deque(maxlen=5) of accepted boundary_proj values;
        # keyed by the same centroid-snap key as the perf cache.
        self._boundary_proj_history: dict = {}  # slot → deque of float

        # Orientation temporal smoothing — 5-frame circular-mean buffer.
        # Temporal smoothing reduces orientation jitter.
        # Keyed by the same centroid-snap key as the perf cache.
        self._orientation_history: dict = {}  # slot → deque of float (angles in radians)

        # Geometry measurement temporal stabilization.
        # Temporal smoothing reduces measurement jitter from segmentation noise.
        # Per-slot deque(maxlen=7) of length_px / width_px values;
        # keyed by the same centroid-snap key as the perf cache.
        self._length_history: dict = {}   # slot → deque(maxlen=7) of length_px values
        self._width_history: dict = {}    # slot → deque(maxlen=7) of width_px values
        self._area_history: dict = {}     # slot → deque(maxlen=5) of area_px values

        # PERFORMANCE OPTIMIZATION: Pre-create morphology kernels (avoid repeated creation)
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Per-slot temporal consensus stabilization.
        # One deque per nail slot so multiple nails never mix measurements.
        # Rolling maxlen=8; no hard clear on failure so a single jittery
        # YOLO frame falls off naturally instead of resetting the count.
        self._stability_windows: dict = {}
        self.is_stable = False

        # Pose-lock stabilization. Thresholds loosened for real YOLO jitter:
        #   centroid <= 40 px, area <= 20 %, angle <= 15 deg, ratio <= 0.15
        self.pose_locked = False
        self.locked_pose = None

        # ROI tracking — after a nail is detected and stabilized, use an
        # OpenCV CSRT tracker to follow it instead of running YOLO every
        # frame.  Falls back to YOLO when the tracker loses the target or
        # when the pose lock breaks.
        self.tracker = None
        self.tracker_active = False
        self.tracked_bbox = None

        # Temporal mask fusion — per-slot deques keyed by centroid snap grid.
        # Storing masks per slot prevents masks from different nails or frames
        # being mixed into the same fusion window.
        self._mask_history: dict = {}  # slot_key -> deque of binary masks

        # Free-edge detection temporal stabilization.
        # Requires boundary presence to persist across multiple frames before
        # it is accepted, preventing single-frame detections from switching
        # measurement mode and causing geometry jumps.
        self._boundary_presence_history: dict = {}  # slot → deque(maxlen=5) of bool
        
    def is_valid_nail(self, nail_data: dict) -> bool:
        """Validate that detection is actually a nail (not fingers, glasses, skin, etc).
        
        Checks:
        - Aspect ratio is reasonable (1.2-3.0x)
        - Size is within nail range (3-30 mm length)
        - Width is within realistic nail range
        - Area is sufficient
        - Color is consistent with nail plate
        - Shape characteristics (circularity, solidity, convexity)
        - Ellipse-fit quality (nails are elliptical; blobs are not)
        - Grayscale variance (rejects flat shadows/backgrounds)
        
        Args:
            nail_data: Dictionary with nail measurements and color
        
        Returns:
            True if valid nail, False otherwise
        """
        # Check aspect ratio using symmetric (orientation-independent) ratio.
        # aspect_ratio is now directional and can be < 1 for wide/short nails;
        # max(r, 1/r) normalises to always ≥1 so MAX_NAIL_ASPECT_RATIO still
        # catches needle-like artefacts while wide-but-valid thumbnails pass.
        _ar = nail_data['aspect_ratio']
        sym_ratio = max(_ar, 1.0 / _ar) if _ar > 0 else 0
        if not (MIN_NAIL_ASPECT_RATIO <= sym_ratio <= MAX_NAIL_ASPECT_RATIO):
            return False
        
        # Check length constraints (3-30 mm typical)
        if not (MIN_NAIL_LENGTH_MM <= nail_data['length_mm'] <= MAX_NAIL_LENGTH_MM):
            return False
        
        # Check width constraint — reject wide skin patches / background
        if nail_data['width_mm'] > MAX_NAIL_WIDTH_MM:
            logger.debug(f"Rejected: width {nail_data['width_mm']:.1f}mm > {MAX_NAIL_WIDTH_MM}mm")
            return False
        
        # Check area constraint
        if nail_data['area_mm2'] < MIN_NAIL_AREA_MM2:
            return False
        
        # SHAPE-BASED FILTERING: Reject non-nail objects (fingers, glasses, etc.)
        mask = nail_data.get('mask')
        if mask is not None and mask.sum() > 0:
            # Reuse the contour already computed in process_frame(); only fall
            # back to cv2.findContours if for some reason it is absent.
            cnt = nail_data.get('contour')
            if cnt is None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = max(contours, key=cv2.contourArea) if contours else None
            if cnt is not None:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                
                # Circularity: Nails are somewhat elliptical (0.45-0.9)
                # Fingers/long objects have low circularity (<0.4)
                # Long nails (aspect ratio > 1.8) are naturally less circular
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    _min_circ = 0.28 if sym_ratio > 1.8 else 0.40
                    if circularity < _min_circ or circularity > 0.95:  # Too elongated or too circular
                        logger.debug(f"Rejected: circularity {circularity:.2f} (min={_min_circ:.2f}, sym_ratio={sym_ratio:.2f})")
                        return False
                
                # Solidity: Ratio of contour area to convex hull area
                # Nails are solid (>0.85), fingers with gaps have lower solidity
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity < 0.82:  # Too many gaps/concavities
                        logger.debug(f"Rejected: solidity {solidity:.2f} (irregular shape)")
                        return False
                
                # Extent: Ratio of contour area to bounding rectangle area
                # Nails fill their bounding box well (>0.65)
                x, y, w, h = cv2.boundingRect(cnt)
                rect_area = w * h
                if rect_area > 0:
                    extent = area / rect_area
                    if extent < 0.60:  # Too sparse in bounding box
                        logger.debug(f"Rejected: extent {extent:.2f} (sparse/thin object)")
                        return False
                
                # Ellipse-fit quality: Nails match an ellipse closely.
                # Random objects, shadows, and fingers do not.
                if len(cnt) >= 5:  # fitEllipse needs at least 5 points
                    ellipse = cv2.fitEllipse(cnt)
                    ell_mask = np.zeros_like(mask)
                    cv2.ellipse(ell_mask, ellipse, 255, -1)
                    # IoU between mask and best-fit ellipse
                    intersection = float(np.sum((mask > 0) & (ell_mask > 0)))
                    union = float(np.sum((mask > 0) | (ell_mask > 0)))
                    if union > 0:
                        ell_iou = intersection / union
                        if ell_iou < 0.65:
                            logger.debug(f"Rejected: ellipse IoU {ell_iou:.2f} (not nail-shaped)")
                            return False
        
        # Grayscale variance: real nails have a specular highlight giving
        # std(gray) ~18-40. Flat skin patches are ~6-12.
        # Floor raised 4.0->8.0 to reject skin; 8.0 not 12.0 to be safe.
        if mask is not None and mask.sum() > 0 and nail_data.get('_frame_ref') is not None:
            frame_ref = nail_data['_frame_ref']
            gray_frame = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
            nail_pixels_gray = gray_frame[mask > 0]
            if len(nail_pixels_gray) > 30:
                gray_std = float(np.std(nail_pixels_gray))
                if gray_std < 8.0:
                    logger.debug(f"Rejected: gray std {gray_std:.1f} < 8.0 (flat skin/shadow)")
                    return False
        
        # Check color is nail-like (not lips/skin/glasses)
        if nail_data['nail_color_LAB']:
            L, a, b = nail_data['nail_color_LAB']
            # Nails should be in a specific color range
            if not (NAIL_LAB_L_MIN <= L <= NAIL_LAB_L_MAX):
                return False  # Too bright (lips) or too dark
            if not (NAIL_LAB_A_MIN <= a <= NAIL_LAB_A_MAX):
                return False  # Wrong redness
            if not (NAIL_LAB_B_MIN <= b <= NAIL_LAB_B_MAX):
                return False  # Wrong warmth
            
            # Chroma floor: shadows/dark surfaces have near-zero chroma
            real_a = a - 128
            real_b = b - 128
            chroma = float(np.sqrt(real_a**2 + real_b**2))
            if chroma < 3.0:
                logger.debug(f"Rejected: chroma {chroma:.1f} (achromatic — shadow/object)")
                return False
        
        return True
    
    def estimate_nail_tilt(self, image: np.ndarray, mask: np.ndarray,
                           lab_frame: np.ndarray = None) -> dict:
        """Estimate if nail is tilted out-of-plane relative to camera.
        
        Returns:
            dict with:
            - likely_tilted: bool - True if likely tilted
            - confidence: float - 0-1 confidence level (0 = perpendicular, 1 = definitely tilted)
            - indicators: list - which checks flagged tilt
            - details: dict - detailed measurements
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'likely_tilted': False, 'confidence': 0, 'indicators': [], 'details': {}}
        
        cnt = max(contours, key=cv2.contourArea)
        
        # Get rotated bounding box
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (width, height), angle = rect
        
        # Calculate various measurements
        area = cv2.contourArea(cnt)
        rect_area = width * height
        
        # Get edge pixel analysis
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Reuse pre-computed LAB frame when available (Goal 2)
        if lab_frame is not None:
            L_channel = lab_frame[:, :, 0]
        else:
            L_channel = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        
        # Extract nail region
        nail_gray = img_gray[mask > 0]
        
        indicators = []
        details = {}
        
        # CHECK 1: Edge pixel brightness variation
        edge_brightness_std = np.std(nail_gray)
        details['brightness_std'] = float(edge_brightness_std)
        if edge_brightness_std > 40:  # High variation suggests tilt
            indicators.append("High brightness variation")
        
        # CHECK 2: Circularity deviation
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        details['circularity'] = float(circularity)
        if circularity < 0.5:  # Low circularity suggests tilt
            indicators.append("Low circularity")
        
        # CHECK 3: Aspect ratio extremes (bounding box)
        bbox_aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
        details['bbox_aspect_ratio'] = float(bbox_aspect_ratio)
        if bbox_aspect_ratio > 2.5:
            indicators.append(f"Extreme aspect ratio")
        
        # CHECK 3b: Use symmetric ratio for PCA vs bbox comparison
        # (avoids wide-nail false positives where PCA ratio < 1 is correct)
        from src.geometry_utils import extract_geometry
        pca_length, pca_width, _ = extract_geometry(mask)
        if pca_length and pca_width:
            pca_sym_ratio = max(pca_length, pca_width) / max(min(pca_length, pca_width), 1e-6)
            bbox_sym_ratio = max(width, height) / max(min(width, height), 1e-6)
            details['pca_aspect_ratio'] = float(pca_sym_ratio)
            details['bbox_aspect_ratio'] = float(bbox_sym_ratio)
            aspect_diff = abs(pca_sym_ratio - bbox_sym_ratio) / max(bbox_sym_ratio, 1e-6)
            details['aspect_ratio_diff_percent'] = float(aspect_diff * 100)
            # Raise threshold for wide nails (where ratios naturally differ more)
            _tilt_ar_threshold = 0.20 if bbox_sym_ratio < 1.4 else 0.30
            if aspect_diff > _tilt_ar_threshold:
                indicators.append("PCA vs bbox aspect mismatch")
        
        # CHECK 4: Contour vs bounding box area mismatch
        area_fill_ratio = area / rect_area if rect_area > 0 else 0
        details['area_fill_ratio'] = float(area_fill_ratio)
        if area_fill_ratio < 0.70:  # Lowered threshold from 0.75 to 0.70
            indicators.append("Irregular shape")
        
        # CHECK 5: Intensity gradient
        nail_L = L_channel[mask > 0]
        intensity_gradient = np.std(nail_L) if len(nail_L) > 0 else 0
        details['intensity_gradient'] = float(intensity_gradient)
        # Threshold raised 15->25: healthy nails routinely have std(L)~18-22
        # from the natural specular highlight. Old threshold fired on real nails
        # and combined with Check 1 gave score 0.50 -> false "tilted" rejection.
        if intensity_gradient > 25:
            indicators.append("Perspective shading")
        
        # Weighted indicator scoring.
        # "Perspective shading" weight lowered 0.25->0.15.
        # Normal nail max score: Check1(0.25) + Check5(0.15) = 0.40 < 0.50.
        # Genuinely tilted nail triggers 3+ checks -> score >= 0.55.
        indicator_weights = {
            "High brightness variation": 0.25,
            "Perspective shading": 0.15,
            "Low circularity": 0.20,
            "Extreme aspect ratio": 0.20,
            "PCA vs bbox aspect mismatch": 0.15,
            "Irregular shape": 0.05,
        }
        weighted_score = sum(indicator_weights.get(ind, 0.1) for ind in indicators)
        confidence = min(1.0, weighted_score)
        
        return {
            'likely_tilted': confidence > 0.5,  # Increased threshold from 0.4 to 0.5
            'confidence': confidence,
            'indicators': indicators,
            'details': details
        }
        
    def process_frame(self, frame: np.ndarray, frame_count: int = 0) -> tuple:
        """Process a single frame for nails with validation.
        
        Workflow:
        1. Input: Full-resolution frame from camera (1280x960)
        2. Resize: Down to 640x480 for inference (speed optimization)
        3. Inference: YOLO model produces masks at 640x480
        4. Resize masks: Back to original frame dimensions (1280x960)
        5. Extract geometry: On full-resolution masks for accurate measurements
        
        Args:
            frame: BGR frame from webcam (full resolution from camera)
            frame_count: Current frame number (for skipping expensive operations)
        
        Returns:
            Tuple of (processed_frame, nails_list)
        """
        # --- Goal 2: Convert BGR→LAB once; reused by geometry and tilt estimation ---
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        # ── ROI tracker fast-path ─────────────────────────────────────────
        # When the CSRT tracker is active we skip YOLO entirely and re-use
        # the cached masks translated to the tracked bounding box.  If the
        # tracker loses the target we fall through to YOLO below.
        _tracker_ok = False
        if self.tracker_active and self.tracker is not None:
            success, bbox = self.tracker.update(frame)
            if success:
                self.tracked_bbox = tuple(int(v) for v in bbox)
                _tracker_ok = True
            else:
                # Tracker lost — reset and let YOLO reacquire
                self.tracker_active = False
                self.tracker = None
                self.tracked_bbox = None
                # Clear all per-slot mask histories so stale frames don't
                # contaminate the fusion window after reacquisition.
                self._mask_history.clear()

        # Dedicated inference frame — INTER_AREA minimises aliasing on downscale.
        # Created once per call regardless of whether YOLO runs this frame.
        inference_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

        # --- Goal 1: Run YOLO only every 3rd frame; reuse cached masks otherwise ---
        # Skip YOLO entirely when the ROI tracker is actively following the nail.
        if not _tracker_ok:
            if frame_count % 3 == 0 or self._cached_raw_masks is None:
                results = self.model(
                    inference_frame,
                    conf=REALTIME_DETECTION_CONFIDENCE,
                    device="cpu",
                    imgsz=512,
                    verbose=False,
                    stream=False,
                )[0]
                if results.masks is not None:
                    self._cached_raw_masks = results.masks.data.cpu().numpy()
                    if results.boxes is not None and results.boxes.conf is not None:
                        self._cached_confs = results.boxes.conf.cpu().numpy()
                    else:
                        self._cached_confs = np.ones(
                            len(self._cached_raw_masks), dtype=np.float32
                        )
                else:
                    self._cached_raw_masks = None
                    self._cached_confs = np.array([], dtype=np.float32)

        nails = []

        if self._cached_raw_masks is None:
            return frame, nails

        masks = self._cached_raw_masks
        
        for i, mask in enumerate(masks):
            # Per-mask confidence floor — applied every frame including cached ones.
            # 0.75 sits above the 0.70 model gate but below where real nails land.
            # (0.82 was too high and silently dropped valid detections.)
            _MASK_CONF_FLOOR = 0.75
            if (len(self._cached_confs) > i and
                    float(self._cached_confs[i]) < _MASK_CONF_FLOOR):
                logger.debug(
                    f"Mask {i} dropped: conf={self._cached_confs[i]:.3f}"
                    f" < floor {_MASK_CONF_FLOOR}"
                )
                continue
            # Resize mask to original frame size.
            # INTER_LINEAR gives smooth sub-pixel edges; threshold afterwards
            # avoids blocky staircase artefacts that destabilise PCA axes.
            mask_resized = cv2.resize(
               mask,
               (frame.shape[1], frame.shape[0]),
               interpolation=cv2.INTER_LINEAR
            ) 

            # Use stricter threshold for real-time
            mask_binary = (mask_resized > REALTIME_MASK_THRESHOLD).astype(np.uint8) * 255

            # Use stricter minimum size requirements
            if mask_binary.sum() < REALTIME_MIN_MASK_PIXELS:
                continue

            # ── Per-slot temporal mask fusion ──────────────────────────────
            # Reduce frame-to-frame mask flicker via pixel-wise majority vote.
            # History is stored per nail slot so masks from different nails are
            # never mixed.  A lightweight centroid key is derived directly from
            # mask_binary here (same 20-px snap grid as _cache_key below).
            _fuse_yx = np.argwhere(mask_binary == 255)
            if len(_fuse_yx):
                _fuse_key = (
                    round(float(np.mean(_fuse_yx[:, 1])) / 20),
                    round(float(np.mean(_fuse_yx[:, 0])) / 20),
                )
            else:
                _fuse_key = None

            if not _tracker_ok and _fuse_key is not None:
                # Fresh YOLO detection — clear this slot's history so stale
                # masks from a previous detection don't contaminate the new one.
                self._mask_history.pop(_fuse_key, None)

            if _fuse_key is not None:
                _mask_hist = self._mask_history.setdefault(_fuse_key, deque(maxlen=5))
                _mask_hist.append(mask_binary)

                if _tracker_ok and len(_mask_hist) >= 3:
                    _mstack = np.stack(list(_mask_hist), axis=0).astype(np.float32) / 255.0
                    _fused = (np.mean(_mstack, axis=0) > 0.5).astype(np.uint8) * 255
                    _fused = cv2.morphologyEx(_fused, cv2.MORPH_CLOSE, self.kernel_small)
                    if _fused.sum() >= REALTIME_MIN_MASK_PIXELS:
                        mask_binary = _fused
                    # else: fused mask too sparse — keep the raw mask

            # ── ROI cropping when tracker is active ─────────────────────────
            # When the CSRT tracker is following the nail, crop all
            # analysis inputs to the tracked bounding box (+ padding)
            # so geometry, color, and nail-bed routines process a
            # smaller region.  Coordinates are offset back to
            # full-frame before storage.
            _roi_ox = 0
            _roi_oy = 0
            _work_frame = frame
            _work_lab   = lab_frame
            _work_mask  = mask_binary
            _using_roi  = False

            if _tracker_ok and self.tracked_bbox is not None:
                _tb_x, _tb_y, _tb_w, _tb_h = self.tracked_bbox
                if _tb_w < 30 or _tb_h < 30:
                    # ROI too small — disable tracker, YOLO will reacquire
                    self.tracker_active = False
                    self.tracker = None
                    self.tracked_bbox = None
                else:
                    _pad = int(max(_tb_w, _tb_h) * 0.2)
                    _roi_x1 = max(0, _tb_x - _pad)
                    _roi_y1 = max(0, _tb_y - _pad)
                    _roi_x2 = min(frame.shape[1], _tb_x + _tb_w + _pad)
                    _roi_y2 = min(frame.shape[0], _tb_y + _tb_h + _pad)
                    _roi_crop_mask = mask_binary[_roi_y1:_roi_y2, _roi_x1:_roi_x2]
                    if _roi_crop_mask.sum() < REALTIME_MIN_MASK_PIXELS:
                        # Insufficient mask inside ROI — disable tracker
                        self.tracker_active = False
                        self.tracker = None
                        self.tracked_bbox = None
                    else:
                        _roi_ox = _roi_x1
                        _roi_oy = _roi_y1
                        _work_frame = frame[_roi_y1:_roi_y2, _roi_x1:_roi_x2]
                        _work_lab   = lab_frame[_roi_y1:_roi_y2, _roi_x1:_roi_x2]
                        _work_mask  = _roi_crop_mask
                        # Restrict mask_binary to the ROI so that all downstream
                        # consumers (nail_data['mask'], is_valid_nail, etc.) operate
                        # only on the tracked region and never see fused background.
                        mask_binary = _roi_crop_mask
                        _using_roi  = True

            contours, _ = cv2.findContours(_work_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)

            # Offset contour to full-frame coordinates when using ROI
            if _using_roi:
                cnt = cnt + np.array([_roi_ox, _roi_oy])

            # Precompute mask coordinates once — reused for centroid, cov, and cache key.
            nail_yx = np.argwhere(_work_mask == 255)
            nail_xy = nail_yx[:, ::-1].astype(np.float64)  # (N, 2) in (x, y) order
            # Offset to full-frame coordinates when using ROI
            if _using_roi:
                nail_xy[:, 0] += _roi_ox
                nail_xy[:, 1] += _roi_oy
            centroid = nail_xy.mean(axis=0)
            centered = nail_xy - centroid
            cov = np.cov(centered.T)

            # Morphology is handled inside geometry_utils.
            # Avoid applying it here to prevent mask erosion.

            # Extract geometry (pass frame for anatomical axis orientation)
            geom = extract_geometry(_work_mask, image=_work_frame)
            if geom[0] is None:  # geometry extraction failed
                continue
            
            length_px, width_px, area_px = geom
            
            # EARLY REJECTION: Quick aspect ratio check before expensive operations
            # Calculate preliminary aspect ratio (avoid expensive operations if will fail)
            prelim_aspect_ratio = max(length_px, width_px) / min(length_px, width_px) if min(length_px, width_px) > 1e-6 else 0
            if not (MIN_NAIL_ASPECT_RATIO <= prelim_aspect_ratio <= MAX_NAIL_ASPECT_RATIO):
                logger.debug(f"Early reject: aspect ratio {prelim_aspect_ratio:.2f} out of range")
                continue

            # ── Mask completeness check ───────────────────────────────────────
            # Reject truncated YOLO masks (e.g. nail only partially in frame or
            # occluded) before any expensive computation.  Strategy:
            #   1. Rotated bounding-box major span  (minAreaRect)
            #   2. Mask projection span along PCA major axis
            #   3. coverage = span_mask / span_bbox  — should be ≥ 0.60
            # The PCA major axis is derived from the covariance matrix already
            # computed above, so no extra pass over the data is required.
            try:
                _, (_bb_w, _bb_h), _ = cv2.minAreaRect(cnt)
                _bbox_major_span = float(max(_bb_w, _bb_h))
                if _bbox_major_span > 1e-6:
                    # Eigen-decompose cov to get PCA major axis
                    _evals, _evecs = np.linalg.eigh(cov)
                    _major_axis_cov = _evecs[:, np.argmax(_evals)]  # (2,) unit vector
                    _proj = nail_xy @ _major_axis_cov
                    _mask_span = float(_proj.max() - _proj.min())
                    _coverage = _mask_span / (_bbox_major_span + 1e-6)
                    if _coverage < 0.60:
                        logger.debug(
                            f"Rejected nail {i+1}: truncated mask "
                            f"coverage={_coverage:.2f} (<0.60)"
                        )
                        continue
            except Exception as _mc_err:
                logger.debug(f"Mask completeness check failed: {_mc_err}")
                # Non-fatal: let the nail through if the check itself errors

            # Estimate out-of-plane tilt — pass pre-computed LAB to avoid re-conversion
            tilt_info = self.estimate_nail_tilt(_work_frame, _work_mask, lab_frame=_work_lab)
            
            # Reject measurement if tilt confidence > 0.5 (before expensive color extraction)
            if tilt_info.get('confidence', 0) > 0.5:
                logger.debug(f"Rejecting nail {i+1}: tilt confidence {tilt_info['confidence']:.2f} > 0.5")
                continue
            
            # --- Goals 4 & 5: Nail-bed boundary / PCA caching ---
            # Compute a coarse centroid key (40-px grid) for cache lookup.
            # A 40 px grid tolerates typical YOLO centroid jitter (~5-15 px)
            # while still distinguishing adjacent nails (~80+ px apart at normal
            # camera distances), so the same physical nail always maps to the
            # same slot across frames.
            _GRID_SIZE = 40
            _cx = float(centroid[0])
            _cy = float(centroid[1])
            _cache_key = (round(_cx / _GRID_SIZE), round(_cy / _GRID_SIZE))

            # Per-slot temporal consensus stabilization.
            # Each nail slot has its own deque (maxlen=8). Stability passes
            # when >= 6 of the last 8 frames are within bounds (75% vote).
            # No hard clear — a bad frame just rolls off naturally.
            measurement = {
                "cx": _cx, "cy": _cy, "area": area_px, "ratio": prelim_aspect_ratio,
            }
            _stab_win = self._stability_windows.setdefault(
                _cache_key, deque(maxlen=8)
            )
            _stab_win.append(measurement)

            _CENTROID_THR = 50   # px deviation from median (was 25 — too tight)
            _AREA_THR     = 0.25  # fraction (was 0.15)
            _RATIO_THR    = 0.20  # (was 0.10)
            _MIN_STABLE   = 6    # of 8 frames must pass (75% vote)

            if len(_stab_win) >= 8:
                _cx_v = [m["cx"]    for m in _stab_win]
                _cy_v = [m["cy"]    for m in _stab_win]
                _ar_v = [m["area"]  for m in _stab_win]
                _rt_v = [m["ratio"] for m in _stab_win]
                _cx_med = float(np.median(_cx_v))
                _cy_med = float(np.median(_cy_v))
                _ar_med = float(np.median(_ar_v))
                _rt_med = float(np.median(_rt_v))
                _passing = sum(
                    1 for m in _stab_win
                    if (abs(m["cx"] - _cx_med) + abs(m["cy"] - _cy_med)) < _CENTROID_THR
                    and abs(m["area"]  - _ar_med) / max(_ar_med, 1) < _AREA_THR
                    and abs(m["ratio"] - _rt_med) < _RATIO_THR
                )
                if _passing >= _MIN_STABLE:
                    self.is_stable = True
                else:
                    self.is_stable = False
                    # Do NOT clear the window — let it roll.
                    self.pose_locked = False
                    self.locked_pose = None
                    self.tracker_active = False
                    self.tracker = None
                    self.tracked_bbox = None

            # ── Geometry temporal stabilization ─────────────────────────────────────
            # Temporal smoothing reduces measurement jitter from segmentation noise.
            # Maintains per-slot deque(maxlen=7) of raw length_px / width_px values
            # and replaces raw measurements with their running mean.  Works correctly
            # even when the buffer is not yet full (np.mean handles partial deques).
            #
            # Outlier rejection: prevents corrupted frames from affecting stabilized
            # measurements.  Before appending, the current value is compared against
            # the median of the existing history.  Values that deviate by more than
            # 15 % of the median are silently dropped; the stabilized output then
            # falls back to the current history mean, keeping continuity.
            if _cache_key is not None:
                _len_hist  = self._length_history.setdefault(_cache_key, deque(maxlen=7))
                _wid_hist  = self._width_history.setdefault(_cache_key, deque(maxlen=7))
                _area_hist = self._area_history.setdefault(_cache_key, deque(maxlen=5))

                # Length outlier check
                if len(_len_hist) > 0:
                    _median_length = float(np.median(_len_hist))
                    if abs(length_px - _median_length) <= 0.15 * _median_length:
                        _len_hist.append(length_px)
                    # else: corrupted frame — skip append, reuse existing history
                else:
                    _len_hist.append(length_px)  # buffer empty: always seed

                # Width outlier check
                if len(_wid_hist) > 0:
                    _median_width = float(np.median(_wid_hist))
                    if abs(width_px - _median_width) <= 0.15 * _median_width:
                        _wid_hist.append(width_px)
                    # else: corrupted frame — skip append, reuse existing history
                else:
                    _wid_hist.append(width_px)   # buffer empty: always seed

                # Area outlier check (20 % threshold — area variance is larger than
                # linear dimensions because it scales with the square of the span).
                if len(_area_hist) > 0:
                    _median_area = float(np.median(_area_hist))
                    if abs(area_px - _median_area) <= 0.20 * _median_area:
                        _area_hist.append(area_px)
                    # else: corrupted frame — skip append, reuse existing history
                else:
                    _area_hist.append(area_px)   # buffer empty: always seed

                length_stable = float(np.mean(_len_hist))
                width_stable  = float(np.mean(_wid_hist))
                area_stable   = float(np.median(_area_hist))
            else:
                length_stable = length_px
                width_stable  = width_px
                area_stable   = float(area_px)

            # Nail orientation from second-order central moments (fast, no extra OpenCV call).
            # Used by texture analysis for orientation-aware ridge detection.
            _moments = cv2.moments(_work_mask)
            _mu20 = _moments.get('mu20', 0.0)
            _mu02 = _moments.get('mu02', 0.0)
            _mu11 = _moments.get('mu11', 0.0)
            _nail_orientation = (
                0.5 * np.arctan2(2.0 * _mu11, _mu20 - _mu02)
                if _moments['m00'] > 0 else None
            )

            # ── Pose-lock: lock on first stability, validate on subsequent frames ──
            if self.is_stable and not self.pose_locked:
                # First stable frame — lock the current pose
                self.locked_pose = {
                    "cx":    _cx,
                    "cy":    _cy,
                    "angle": _nail_orientation,   # may be None; handled below
                    "area":  area_px,
                    "ratio": prelim_aspect_ratio,
                }
                self.pose_locked = True

            if self.pose_locked and self.locked_pose is not None:
                # Compare current frame to the locked reference pose
                _pl = self.locked_pose
                _pose_centroid_drift = np.sqrt(
                    (_cx - _pl["cx"]) ** 2 + (_cy - _pl["cy"]) ** 2
                )
                _pose_area_change = (
                    abs(area_px - _pl["area"]) / _pl["area"]
                    if _pl["area"] > 0 else 0.0
                )
                _pose_ratio_change = abs(prelim_aspect_ratio - _pl["ratio"])

                # Angle deviation (circular-safe)
                if _nail_orientation is not None and _pl["angle"] is not None:
                    _pose_angle_diff = abs(_nail_orientation - _pl["angle"])
                    if _pose_angle_diff > np.pi:
                        _pose_angle_diff = 2 * np.pi - _pose_angle_diff
                else:
                    _pose_angle_diff = 0.0  # can't check angle — pass

                _pose_centroid_thr = 40      # was 20 — YOLO centroid jitters +-10-15 px
                _pose_angle_thr    = np.deg2rad(15)  # was 8 deg
                _pose_area_thr     = 0.20    # was 0.10
                _pose_ratio_thr    = 0.15    # was 0.08

                if (_pose_centroid_drift > _pose_centroid_thr or
                        _pose_angle_diff   > _pose_angle_thr or
                        _pose_area_change  > _pose_area_thr or
                        _pose_ratio_change > _pose_ratio_thr):
                    # Pose drifted — break lock and reset stabilization
                    self.pose_locked = False
                    self.locked_pose = None
                    self.is_stable   = False
                    if _cache_key in self._stability_windows:
                        self._stability_windows[_cache_key].clear()
                    self.tracker_active = False
                    self.tracker = None
                    self.tracked_bbox = None

            # ── Temporal smoothing of major axis orientation ───────────────────────
            # Temporal smoothing reduces orientation jitter.
            # Maintains a deque of the last 5 orientation angles (radians) per
            # nail slot and computes the circular mean to handle angle wraparound.
            _major_axis_smooth = None
            _minor_axis_smooth = None
            _axis_confidence   = 0.0
            if _nail_orientation is not None and _cache_key is not None:
                _ori_hist = self._orientation_history.setdefault(
                    _cache_key, deque(maxlen=5)
                )
                _ori_hist.append(_nail_orientation)

                # Circular mean: average sin and cos components separately to
                # correctly handle wraparound at ±π without angular discontinuity.
                _angles = np.array(list(_ori_hist), dtype=np.float64)
                _theta_mean = float(np.arctan2(
                    np.mean(np.sin(_angles)),
                    np.mean(np.cos(_angles)),
                ))

                # Recompute stabilized major/minor axis vectors from smoothed angle.
                _major_axis_smooth = np.array(
                    [np.cos(_theta_mean), np.sin(_theta_mean)], dtype=np.float64
                )
                _minor_axis_smooth = np.array(
                    [-np.sin(_theta_mean), np.cos(_theta_mean)], dtype=np.float64
                )

                # Axis confidence: circular resultant length (1 = perfectly stable,
                # 0 = maximum dispersion).  High values mean the orientation history
                # is tightly clustered; low values indicate orientation jitter.
                _resultant_len = float(abs(np.mean(np.exp(1j * _angles))))
                _axis_confidence = round(_resultant_len, 3)

                # Replace raw orientation with smoothed angle for downstream use.
                _nail_orientation = _theta_mean

            _nb_cached = self._nail_perf_cache.get(_cache_key) if _cache_key else None
            _use_cached_nb = False
            _boundary_methods = 0
            _boundary_conf    = 0.0
            nail_bed_free_edge_present    = None
            nail_bed_free_edge_confidence = 0.0
            nail_bed_length_px = nail_bed_width_px = nail_bed_area_px = None
            if _nb_cached is not None:
                prev_cx, prev_cy = _cache_key
                movement = np.sqrt((prev_cx * 20 - _cx)**2 + (prev_cy * 20 - _cy)**2)
                if movement > 40:
                    _use_cached_nb = False
            if _nb_cached is not None and area_px > 0:
                # Skip cache if stored area is zero (corrupt entry) or current area is zero
                if _nb_cached['area_px'] <= 0 or area_px <= 0:
                    _nb_cached = None
            if _nb_cached is not None and area_px > 0:
                _area_delta = abs(area_px - _nb_cached['area_px']) / max(_nb_cached['area_px'], 1.0)
                _prev_methods = _nb_cached.get('boundary_methods', 0)
                _cache_is_low_confidence = _prev_methods <= 1
                if _area_delta < 0.10 and not _cache_is_low_confidence and _nb_cached.get('nb_length') is not None:
                    nail_bed_length_px = _nb_cached['nb_length']
                    nail_bed_width_px  = _nb_cached['nb_width']
                    nail_bed_area_px   = _nb_cached['nb_area']
                    _boundary_methods  = _prev_methods
                    _boundary_conf     = 0.0  # not recomputed from cache
                    _use_cached_nb = True

            if not _use_cached_nb:
                # Only run expensive color analysis once the pose is locked;
                # before stabilization the result would be discarded anyway.
                if self.pose_locked:
                    lab_color, bgr_color, color_analysis, hex_color = extract_nail_color(
                        _work_frame, _work_mask,
                        nail_id=i + 1,
                        nail_orientation=_nail_orientation,
                        lab_frame=_work_lab,
                    )
                else:
                    lab_color = None
                    bgr_color = None
                    hex_color = None
                    color_analysis = None

                _is_polished_rt = bool(
                    color_analysis and color_analysis.get("is_polished_detected", False)
                )

                if color_analysis and color_analysis.get("skipped") == "polished":
                    nail_bed_length_px = nail_bed_width_px = nail_bed_area_px = None
                    nb_diag                    = None
                    nail_bed_free_edge_present = None
                    nail_bed_free_edge_confidence = 0.0
                    _boundary_methods = 0
                    _boundary_conf    = 0.0
                else:
                    nail_bed_length_px, nail_bed_width_px, nail_bed_area_px, nb_diag = (
                        extract_geometry_nail_bed_with_diagnostics(
                            _work_frame, _work_mask,
                            lab_frame=_work_lab,
                            is_polished=_is_polished_rt
                        )
                    )

                    nail_bed_free_edge_present    = nb_diag.get("free_edge_present") if nb_diag else None
                    nail_bed_free_edge_confidence = nb_diag.get("free_edge_confidence", 0.0) if nb_diag else 0.0
                    _boundary_methods = nb_diag.get("boundary_methods_agreed", 0) if nb_diag else 0
                    _boundary_conf    = nb_diag.get("boundary_confidence", 0.0) if nb_diag else 0.0

                    if (
                        _cache_key is not None
                        and nail_bed_length_px is not None
                        and _boundary_methods >= 1
                        and nail_bed_free_edge_confidence >= 0.55
                    ):
                        if len(self._nail_perf_cache) >= self._nail_perf_cache_max_size:
                            oldest_key = next(iter(self._nail_perf_cache))
                            del self._nail_perf_cache[oldest_key]
                        self._nail_perf_cache[_cache_key] = {
                            'area_px':   area_px,
                            'nb_length': nail_bed_length_px,
                            'nb_width':  nail_bed_width_px,
                            'nb_area':   nail_bed_area_px,
                            'boundary_methods': _boundary_methods,
                            'boundary_confidence': _boundary_conf,
                        }

            # Extract color on cached path — only when pose is locked.
            if _use_cached_nb:
                if self.pose_locked:
                    lab_color, bgr_color, color_analysis, hex_color = extract_nail_color(
                        _work_frame, _work_mask,
                        nail_id=i + 1,
                        nail_orientation=_nail_orientation,
                        lab_frame=_work_lab,
                    )
                else:
                    lab_color = None
                    bgr_color = None
                    hex_color = None
                    color_analysis = None

            # ── Temporal smoothing of boundary_proj ───────────────────────────────────
            # Prevents frame-to-frame jitter in boundary detection.
            # Maintains a deque of the last 5 accepted boundary_proj values per
            # nail slot.  Weighted average (older→newer = 1:2:3:4:5) gives a
            # smooth output that tracks slow boundary movement while ignoring
            # single-frame outliers (glare, transient occlusion, hand tremor).
            # Low-confidence frames reset history so stale anchors are not
            # carried forward when the nail re-enters the frame.
            _raw_boundary_proj    = nb_diag.get("boundary_proj") if nb_diag else None
            _smoothed_boundary_proj = _raw_boundary_proj  # default: pass-through

            if _cache_key is not None:
                _hist = self._boundary_proj_history.setdefault(
                    _cache_key, deque(maxlen=5)
                )


                # Reset history on low confidence — don’t anchor to bad detections
                if _boundary_conf < 0.45 and len(_hist) > 0:
                    _hist.clear()

                if _raw_boundary_proj is not None:
                    if len(_hist) > 0:
                        _prev_mean = float(np.mean(list(_hist)))
                        if abs(_raw_boundary_proj - _prev_mean) > 8.0:
                            # Outlier: too far from recent mean — keep history,
                            # return previous smooth value for temporal consistency
                            _raw_boundary_proj = None
                        else:
                            _hist.append(_raw_boundary_proj)
                    else:
                        _hist.append(_raw_boundary_proj)

                if len(_hist) > 0:
                    _smoothed_boundary_proj = float(np.median(list(_hist)))
                else:
                    _smoothed_boundary_proj = None
            
            # ── Nail-bed geometry fallback & safety clamp ─────────────────────
            # When boundary detection fails (no methods agreed) or returned no
            # measurement, copy full-nail geometry so downstream code always has
            # a finite, self-consistent nail-bed value and no anatomically
            # impossible ratio can be produced.
            if _boundary_methods == 0 or nail_bed_length_px is None:
                nail_bed_length_px         = length_px
                nail_bed_width_px          = width_px
                nail_bed_area_px           = area_px
                nail_bed_free_edge_present = False

            # Safety clamp: nail-bed length can never physically exceed the
            # full nail length (e.g. projection rounding or anatomical prior
            # overshoot).  Width and area are similarly bounded.
            if nail_bed_length_px is not None and length_px and length_px > 0:
                nail_bed_length_px = min(nail_bed_length_px, length_px)
            if nail_bed_width_px is not None and width_px and width_px > 0:
                nail_bed_width_px = min(nail_bed_width_px, width_px)
            if nail_bed_area_px is not None and area_px and area_px > 0:
                nail_bed_area_px = min(nail_bed_area_px, area_px)

            # Convert to mm — use temporally stabilized pixel measurements
            length_mm = self.calibrator.pixel_to_mm(length_stable)
            width_mm = self.calibrator.pixel_to_mm(width_stable)
            area_mm2 = self.calibrator.pixel_area_to_mm2(area_stable)
            
            # Convert nail bed to mm
            nail_bed_length_mm = self.calibrator.pixel_to_mm(nail_bed_length_px) if nail_bed_length_px else None
            nail_bed_width_mm = self.calibrator.pixel_to_mm(nail_bed_width_px) if nail_bed_width_px else None
            nail_bed_area_mm2 = self.calibrator.pixel_area_to_mm2(nail_bed_area_px) if nail_bed_area_px else None
            
            # ── Free-edge detection temporal stabilization ─────────────────────
            # Append detection results to per-slot history and require a
            # majority vote (3/5 frames) before treating the boundary as
            # present.  This prevents a single noisy frame from switching the
            # measurement source between nail-bed and full-nail geometry.
            # Only boolean results (True/False) are appended; None (cache hit
            # or polished nail) is intentionally excluded to avoid diluting the
            # vote on frames where detection was not actually run.
            if _cache_key is not None and nail_bed_free_edge_present is not None:
                _bp_hist = self._boundary_presence_history.setdefault(
                    _cache_key, deque(maxlen=5)
                )
                _bp_hist.append(bool(nail_bed_free_edge_present))
                stable_boundary = sum(_bp_hist) >= 3
            elif _cache_key is not None and _cache_key in self._boundary_presence_history:
                # Cache hit: reuse the last majority vote without adding a new sample.
                _bp_hist = self._boundary_presence_history[_cache_key]
                stable_boundary = sum(_bp_hist) >= 3
            else:
                # No history yet — treat as unstable.
                stable_boundary = False
            # Replace the raw per-frame flag with the temporally stable value.
            nail_bed_free_edge_present = stable_boundary

            # Calculate full-nail aspect ratio from stabilized geometry (directional:
            # length/width, can be < 1).  Using stabilized values reduces ratio jitter
            # caused by per-frame segmentation noise.
            aspect_ratio = length_stable / (width_stable + 1e-6)
            # Directional nail-bed ratio: length/width (> 1 = portrait, < 1 = wide/thumb)
            nail_bed_aspect_ratio = (
                (nail_bed_length_px / nail_bed_width_px)
                if (nail_bed_length_px and nail_bed_width_px and nail_bed_width_px > 1e-6)
                else 0
            )
            # Stable ratio policy: always prefer nail-bed ratio for medical measurements
            # so that a temporary free-edge detection failure never causes a measurement
            # jump by silently switching to full-nail geometry.
            if nail_bed_length_px and nail_bed_width_px and nail_bed_width_px > 1e-6:
                shape_ratio = nail_bed_length_px / nail_bed_width_px
            else:
                shape_ratio = length_stable / (width_stable + 1e-6)
            
            nail_data = {
                'nail_id': i + 1,
                'length_px': length_px,
                'width_px': width_px,
                'area_px': area_px,
                'length_mm': length_mm,
                'width_mm': width_mm,
                'area_mm2': area_mm2,
                'aspect_ratio': aspect_ratio,   # kept for debug / tracking only
                'shape_ratio': shape_ratio,        # primary: nail-bed when available
                'nail_bed_length_px': nail_bed_length_px,
                'nail_bed_width_px': nail_bed_width_px,
                'nail_bed_area_px': nail_bed_area_px,
                'nail_bed_length_mm': nail_bed_length_mm,
                'nail_bed_width_mm': nail_bed_width_mm,
                'nail_bed_area_mm2': nail_bed_area_mm2,
                'nail_bed_aspect_ratio': nail_bed_aspect_ratio,
                'nail_bed_free_edge_present':    nail_bed_free_edge_present,
                'nail_bed_free_edge_confidence': round(float(nail_bed_free_edge_confidence), 3),
                'boundary_methods_count': _boundary_methods,
                'boundary_confidence':    round(float(_boundary_conf), 3),
                'boundary_proj':          _smoothed_boundary_proj,
                'nail_color_LAB': lab_color,
                'nail_color_BGR': bgr_color,
                'nail_color_HEX': hex_color,
                'color_analysis': color_analysis,
                'polished': bool(color_analysis and color_analysis.get("skipped") == "polished"),
                'mask': mask_binary,
                'contour': cnt,
                'tilt_info': tilt_info,  # Add tilt detection info
                'major_axis': (
                    [round(float(_major_axis_smooth[0]), 6), round(float(_major_axis_smooth[1]), 6)]
                    if _major_axis_smooth is not None else None
                ),
                'minor_axis': (
                    [round(float(_minor_axis_smooth[0]), 6), round(float(_minor_axis_smooth[1]), 6)]
                    if _minor_axis_smooth is not None else None
                ),
                'axis_confidence':     _axis_confidence,
                'nail_xy':   nail_xy,
                'centroid':  centroid,
                'covariance': cov,
                '_frame_ref': frame,     # Temporary: used by is_valid_nail, removed after
            }
            
            # Always validate nails properly for accuracy
            if self.is_valid_nail(nail_data):
                nail_data.pop('_frame_ref', None)  # Don't keep full frame in history
                nails.append(nail_data)

                # ── Initialise / refresh ROI tracker on valid YOLO detection ──
                # Once a valid nail is found (whether freshly detected or still
                # pose-locked), seed a CSRT tracker so subsequent frames can
                # skip YOLO.  Re-initialise every time YOLO actually runs to
                # keep the tracker's internal model aligned with the latest
                # segmentation mask.
                if not _tracker_ok:
                    _trk_bbox = cv2.boundingRect(cnt)
                    try:
                        self.tracker = cv2.TrackerCSRT_create()
                        self.tracker.init(frame, _trk_bbox)
                        self.tracker_active = True
                        self.tracked_bbox = _trk_bbox
                    except Exception as _trk_err:
                        logger.debug(f"Tracker init failed: {_trk_err}")
                        self.tracker_active = False
                        self.tracker = None
            else:
                nail_data.pop('_frame_ref', None)
        
        # Only append to history if we have valid detections.
        # Appending empty frames dilutes the smoothing window and causes
        # the display to flicker when nails briefly leave the frame.
        if nails:
            self.nail_history.append(nails)
        elif len(self.nail_history) > 0:
            # Partial flush: remove oldest entry so stale data doesn't persist
            # more than 5 frames after nails disappear from view.
            if len(self.nail_history) == self.nail_history.maxlen:
                self.nail_history.popleft()
        self.current_nails = nails
        if len(self.nail_history) > 1:
           nails = self._smooth_nails(self.nail_history)

        # ── Per-frame cleanup of stabilization dictionaries ────────────────
        # Evict the oldest slot once any dict exceeds MAX_TRACKED_SLOTS to
        # prevent unbounded growth during long sessions.  Insertion order is
        # preserved by dict, so next(iter(d)) always yields the oldest key.
        _MAX_TRACKED_SLOTS = 30
        for _cache_dict in [
            self._stability_windows,
            self._boundary_proj_history,
            self._orientation_history,
            self._length_history,
            self._width_history,
            self._area_history,
            self._boundary_presence_history,
        ]:
            if len(_cache_dict) > _MAX_TRACKED_SLOTS:
                _oldest_key = next(iter(_cache_dict))
                del _cache_dict[_oldest_key]

        return frame, nails
    
    def _smooth_nails(self, history):
        """Apply temporal smoothing using centroid distance + aspect ratio matching.
        
        Improved matching strategy:
        1. Calculate centroid for each nail mask
        2. Match nails across frames by: centroid distance + aspect ratio similarity
        3. Average measurements across matched detections
        
        This provides robust tracking even with hand movement or rotation.
        
        Returns:
            List of nail data with temporally averaged measurements
        """
        if not history or len(history) == 0:
            return []
        
        # Convert deque to list for slicing operations
        history_list = list(history)
        
        # Use the most recent frame's nails as the base
        current_nails = history_list[-1]
        
        # If only one frame of history, no smoothing possible
        if len(history_list) < 2 or not current_nails:
            return current_nails
        
        # Smooth each nail by averaging with previous detections
        smoothed_nails = []
        for nail in current_nails:
            # Calculate centroid from mask
            mask = nail.get('mask')
            if mask is None or mask.sum() == 0:
                smoothed_nails.append(nail)
                continue
            
            moments = cv2.moments(mask)
            if moments['m00'] == 0:
                smoothed_nails.append(nail)
                continue
            
            current_cx = moments['m10'] / moments['m00']
            current_cy = moments['m01'] / moments['m00']
            current_aspect = nail['aspect_ratio']
            
            # Collect measurements from all frames in history for this nail
            measurements = {
                'length_mm': [nail['length_mm']],
                'width_mm': [nail['width_mm']],
                'area_mm2': [nail['area_mm2']],
                'area_px': [nail['area_px']],  # Track pixel area for distance stability
                'aspect_ratio': [nail['aspect_ratio']],  # Track ratios for distance stability
                'nail_bed_length_mm': [nail['nail_bed_length_mm']] if nail['nail_bed_length_mm'] else [],
                'nail_bed_width_mm': [nail['nail_bed_width_mm']] if nail['nail_bed_width_mm'] else [],
                'nail_bed_area_mm2': [nail['nail_bed_area_mm2']] if nail['nail_bed_area_mm2'] else [],
            }
            
            # Look back through history to find matching nails
            # Match by centroid distance + aspect ratio similarity
            for prev_frame_nails in history_list[:-1]:
                best_match = None
                best_match_score = float('inf')
                
                for prev_nail in prev_frame_nails:
                    prev_mask = prev_nail.get('mask')
                    if prev_mask is None or prev_mask.sum() == 0:
                        continue
                    
                    # Calculate previous centroid
                    prev_moments = cv2.moments(prev_mask)
                    if prev_moments['m00'] == 0:
                        continue
                    
                    prev_cx = prev_moments['m10'] / prev_moments['m00']
                    prev_cy = prev_moments['m01'] / prev_moments['m00']
                    prev_aspect = prev_nail['aspect_ratio']

                    centroid_dist = np.sqrt((current_cx - prev_cx)**2 + (current_cy - prev_cy)**2)

                    # Symmetric aspect ratio for matching so that a ratio near
                    # 1.0 doesn't bias matching when directional ratios slightly
                    # flip between frames (e.g. 0.88 vs 1.12 for the same nail).
                    sym_current = max(current_aspect, 1.0 / current_aspect) if current_aspect > 0 else 0
                    sym_prev    = max(prev_aspect,    1.0 / prev_aspect)    if prev_aspect    > 0 else 0
                    aspect_diff = abs(sym_current - sym_prev)

                    # Match if within thresholds
                    if (centroid_dist < CENTROID_DISTANCE_THRESHOLD and 
                        aspect_diff < ASPECT_RATIO_SIMILARITY_THRESHOLD):
                        
                        # Use combined score for best match
                        match_score = centroid_dist + (aspect_diff * 100)  # Weight aspect diff higher
                        
                        if match_score < best_match_score:
                            best_match = prev_nail
                            best_match_score = match_score
                
                # Add best match measurements
                if best_match:
                    measurements['length_mm'].append(best_match['length_mm'])
                    measurements['width_mm'].append(best_match['width_mm'])
                    measurements['area_mm2'].append(best_match['area_mm2'])
                    measurements['area_px'].append(best_match['area_px'])  # Track pixel area
                    measurements['aspect_ratio'].append(best_match['aspect_ratio'])  # Track ratios
                    
                    if best_match['nail_bed_length_mm']:
                        measurements['nail_bed_length_mm'].append(best_match['nail_bed_length_mm'])
                    if best_match['nail_bed_width_mm']:
                        measurements['nail_bed_width_mm'].append(best_match['nail_bed_width_mm'])
                    if best_match['nail_bed_area_mm2']:
                        measurements['nail_bed_area_mm2'].append(best_match['nail_bed_area_mm2'])
            
            # Calculate smoothed values — use MEDIAN not mean so that a single
            # outlier frame (different YOLO mask → wrong boundary → wild ratio)
            # cannot shift the displayed value.
            nail_smoothed = nail.copy()
            nail_smoothed['length_mm'] = np.median(measurements['length_mm']) if measurements['length_mm'] else nail['length_mm']
            nail_smoothed['width_mm'] = np.median(measurements['width_mm']) if measurements['width_mm'] else nail['width_mm']
            nail_smoothed['area_mm2'] = np.median(measurements['area_mm2']) if measurements['area_mm2'] else nail['area_mm2']
            
            if measurements['nail_bed_length_mm']:
                nail_smoothed['nail_bed_length_mm'] = np.median(measurements['nail_bed_length_mm'])
            if measurements['nail_bed_width_mm']:
                nail_smoothed['nail_bed_width_mm'] = np.median(measurements['nail_bed_width_mm'])
            if measurements['nail_bed_area_mm2']:
                nail_smoothed['nail_bed_area_mm2'] = np.median(measurements['nail_bed_area_mm2'])
            
            # DISTANCE STABILITY CHECK: Detect if nail is moving closer/farther
            # If area changes >30% from rolling median, likely distance change - don't update ratio
            area_stable = True
            if len(measurements['area_px']) >= 3:  # Need at least 3 samples for reliable median
                median_area = np.median(measurements['area_px'])
                current_area = nail['area_px']
                area_change_ratio = abs(current_area - median_area) / median_area
                
                if area_change_ratio > 0.30:  # >30% change = likely distance change
                    area_stable = False
                    # Use median ratio instead of current (prevents jitter from distance changes)
                    if len(measurements['aspect_ratio']) >= 2:
                        # Preserve sign (directional) by using median of the directional values
                        nail_smoothed['aspect_ratio'] = np.median(measurements['aspect_ratio'])
                        nail_smoothed['distance_unstable'] = True  # Mark for debugging
            
            # Recalculate ratios based on smoothed values — directional (length/width)
            if area_stable:
                # Use median of historical directional ratios — avoids sign flip artefacts
                # when the directional ratio oscillates near 1.0 across frames
                if len(measurements['aspect_ratio']) >= 2:
                    nail_smoothed['aspect_ratio'] = float(np.median(measurements['aspect_ratio']))
                elif nail_smoothed['width_mm'] > 0:
                    nail_smoothed['aspect_ratio'] = nail_smoothed['length_mm'] / nail_smoothed['width_mm']
                nail_smoothed['distance_unstable'] = False

            if nail_smoothed.get('nail_bed_length_mm') and nail_smoothed.get('nail_bed_width_mm'):
                nb_w = nail_smoothed['nail_bed_width_mm']
                nail_smoothed['nail_bed_aspect_ratio'] = (
                    nail_smoothed['nail_bed_length_mm'] / nb_w if nb_w > 0 else 0
                )

            # Recompute shape_ratio from smoothed values (stable ratio policy):
            # keep nail-bed as primary source; fall back to full-nail geometry only
            # when nail-bed measurements are genuinely absent this frame.
            if nail_smoothed.get('nail_bed_length_mm') and nail_smoothed.get('nail_bed_width_mm'):
                _nb_w_sm = nail_smoothed['nail_bed_width_mm']
                nail_smoothed['shape_ratio'] = (
                    nail_smoothed['nail_bed_length_mm'] / _nb_w_sm
                    if _nb_w_sm > 1e-6 else nail_smoothed.get('shape_ratio', 0)
                )
            else:
                _w_sm = nail_smoothed.get('width_mm') or 0
                nail_smoothed['shape_ratio'] = (
                    nail_smoothed['length_mm'] / (_w_sm + 1e-6)
                    if nail_smoothed.get('length_mm') else nail_smoothed.get('shape_ratio', 0)
                )

            smoothed_nails.append(nail_smoothed)
        
        return smoothed_nails

    def draw_detections(self, frame: np.ndarray, nails: list, frame_count: int = 0) -> np.ndarray:
        """Draw nail detections and info on frame.
        
        Args:
            frame: Input frame
            nails: List of nail detection dictionaries
            frame_count: Current frame number (for skipping expensive drawing ops)
        
        Returns:
            Frame with drawn annotations
        """
        output = frame.copy()
        h, w = output.shape[:2]
        
        if not nails:
            # When pose not locked, prompt to hold steady
            if not self.pose_locked:
                cv2.putText(output, "Hold finger steady...", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
            else:
                # Display "No nails detected"
                cv2.putText(output, "No nails detected", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            return output

        # Stability / pose-lock status header — shown when nails are detected
        if not self.pose_locked:
            cv2.putText(output, "Hold finger steady...", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        else:
            cv2.putText(output, "Analyzing nail", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)

        # Draw each nail
        for i, nail in enumerate(nails):
            mask = nail['mask']
            contours = [nail['contour']]

            # Draw contour in green
            cv2.drawContours(output, contours, -1, (0, 255, 0), 3)
            
            # Get bounding box for text placement
            x, y, box_w, box_h = cv2.boundingRect(contours[0]) if contours else (10, 60 + i*120, 300, 110)
            
            # Text position
            text_y = max(50, y - 20)
            line_height = 28

            # Polished nail: draw orange outline and prompt to remove polish;
            # skip all health metrics — results would be meaningless.
            if nail.get('polished'):
                cv2.drawContours(output, contours, -1, (0, 165, 255), 3)  # orange
                cv2.putText(output, f"Nail {nail['nail_id']} - Polish detected",
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                           0.9, (0, 165, 255), 2)
                text_y += line_height
                cv2.putText(output, "Remove polish for health screening",
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                           0.65, (0, 200, 255), 1)
                continue
            
            # Display nail number with tilt status
            tilt_info = nail.get('tilt_info', {})
            tilt_color = (0, 165, 255) if tilt_info.get('likely_tilted', False) else (0, 255, 0)
            tilt_label = "TILTED" if tilt_info.get('likely_tilted', False) else "DETECTED"
            cv2.putText(output, f"Nail {nail['nail_id']}  [{tilt_label}]",
                       (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, tilt_color, 2)
            text_y += line_height

            # --- Ratios ---
            _bed_ratio = nail.get('nail_bed_aspect_ratio')
            if _bed_ratio:
                cv2.putText(output, f"Bed Shape:  {_bed_ratio:.2f}x",
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
                text_y += line_height
            cv2.putText(output, f"Nail Shape: {nail.get('shape_ratio', nail['aspect_ratio']):.2f}x",
                       (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 1)
            text_y += line_height

            # --- Color (CIE LAB) ---
            if nail['nail_color_LAB']:
                L, a, b = nail['nail_color_LAB']
                L_cie = L * 100.0 / 255.0
                a_cie = a - 128.0
                b_cie = b - 128.0
                cv2.putText(output, f"LAB: L*={L_cie:.0f} a*={a_cie:.0f} b*={b_cie:.0f}",
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 1)
                text_y += line_height

                color = tuple(int(c) for c in nail['nail_color_BGR'])

                # Color patch top-right
                patch_x = max(0, w - 250)
                patch_y = 50 + i * 120 + 35
                cv2.rectangle(output,
                              (patch_x, patch_y),
                              (patch_x + 80, patch_y + 80),
                              color, -1)
                cv2.rectangle(output,
                              (patch_x, patch_y),
                              (patch_x + 80, patch_y + 80),
                              (255, 255, 255), 2)
                        # Show zoomed masked nail preview (skip drawing every 3rd frame for speed)
            if frame_count % 3 != 0:
                mask = nail['mask']
                cnt = nail['contour']

                if cnt is not None:
                    # Get bounding box of the nail with padding for context
                    x, y, box_w, box_h = cv2.boundingRect(cnt)
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + box_w + padding)
                    y2 = min(frame.shape[0], y + box_h + padding)
                    
                    # Crop the region from original frame and mask
                    cropped_frame = frame[y1:y2, x1:x2].copy()
                    cropped_mask = mask[y1:y2, x1:x2].copy()
                    
                    # Apply mask to cropped region
                    thumbnail = cv2.bitwise_and(cropped_frame, cropped_frame, mask=cropped_mask)
                    # Resize to larger zoom (150x150 for better visibility)
                    thumbnail = cv2.resize(thumbnail, (150, 150))
                else:
                    # Fallback if no contour found
                    thumbnail = cv2.bitwise_and(frame, frame, mask=nail['mask'])
                    thumbnail = cv2.resize(thumbnail, (150, 150))

                # Zoomed thumbnail positioned to the right of color patch
                thumb_x = max(0, w - 160)
                thumb_y = 50 + i * 120

                # Ensure we don't exceed frame boundaries
                if thumb_y + 150 < h:
                    output[thumb_y:thumb_y+150, thumb_x:thumb_x+150] = thumbnail        
        
        # Display footer info
        footer_y = h - 30
        cal_info = self.calibrator.get_calibration_info()
        footer_text = f"DPI: {cal_info['dpi']:.0f} | Nails: {len(nails)} | Press Q to quit | S to save screenshot"
        cv2.putText(output, footer_text, (10, footer_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        # Show calibration warning if not properly calibrated
        if hasattr(self.calibrator, "is_calibrated") and not self.calibrator.is_calibrated():
           cv2.putText(output,
                "WARNING: No real calibration applied",
                (10, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1)

        return output
    
    def _find_available_cameras(self, max_index: int = 10) -> list:
        """Auto-detect available cameras with multiple backends.
        
        Args:
            max_index: Maximum camera index to check
        
        Returns:
            List of available camera indices
        """
        available_cameras = []
        
        # Try with default backend first
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        # If no cameras found, try DirectShow backend for Windows
        if not available_cameras:
            logger.info("Trying DirectShow backend...")
            for i in range(max_index):
                # Use DirectShow backend (Windows-specific)
                cap = cv2.VideoCapture(i + cv2.CAP_DSHOW)
                if cap.isOpened():
                    available_cameras.append(i)
                    cap.release()
        
        return available_cameras
    
    def run(self, camera_id: int = -1):
        """Run real-time nail analysis from webcam.
        
        Args:
            camera_id: ID of camera to use (-1 = auto-detect, 0+ = specific camera)
        """
        import time
        
        # Auto-detect camera if not specified
        if camera_id == -1:
            available = self._find_available_cameras()
            
            if not available:
                logger.error("No cameras found.")
                print("\n❌ No cameras detected!")
                print("Troubleshooting:")
                print("1. Check if camera is connected and powered on")
                print("2. Close other applications using the camera")
                print("3. Check Device Manager for camera driver issues")
                print("4. Try plugging camera into a different USB port")
                return
            
            camera_id = available[0]  # Use first available camera
            logger.info(f"Auto-detected camera at index {camera_id}")
            if len(available) > 1:
                print(f"\n📷 Found {len(available)} camera(s): {available}")
                print(f"Using camera {camera_id}.\n")
        
        prev_time = time.time()

        # Try with default backend first
        cap = cv2.VideoCapture(camera_id)
        
        # If default fails on Windows, try DirectShow backend
        if not cap.isOpened():
            logger.info("Trying DirectShow backend...")
            cap = cv2.VideoCapture(camera_id + cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            print(f"\n❌ Could not open camera {camera_id}")
            print("Troubleshooting:")
            print("1. Check if camera is connected")
            print("2. Close applications using the camera (Skype, Teams, Zoom, etc)")
            print("3. Try a different camera index (0, 1, 2, etc)")
            return
        
        # Set camera properties - capture at full resolution for accurate measurements
        # Inference will resize to 640x480, but geometry extraction uses full frame resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Full resolution for accuracy
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)  # Full resolution for accuracy
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame
        
        logger.info("Started real-time nail analysis. Press 'Q' to quit, 'S' to screenshot")
        
        frame_count = 0
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame, attempting to continue...")
                    continue  # Try to continue instead of breaking
                
                frame_count += 1
                
                try:
                    frame, nails = self.process_frame(frame, frame_count)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}")
                    # Continue with original frame on error
                    nails = []
                
                # Reset tracking when no nails in frame — allow fresh print on return
                if not nails:
                    self.seen_nails.clear()
                    self._print_conf_history.clear()
                    self._printed_slots.clear()
                
                # Draw detections (pass frame_count to skip expensive drawing on some frames)
                frame = self.draw_detections(frame, nails, frame_count)
                current_time = time.time()
                fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time)) if 'fps' in locals() else (1 / (current_time - prev_time))

                prev_time = current_time

                cv2.putText(frame, f"FPS: {fps:.1f}",
                     (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX,
                     0.6,
                     (0, 255, 0),
                     2)
 
                # Display frame
                cv2.imshow("Real-Time Nail Analyzer", frame)
                
                # ── Rolling-confidence print gate ───────────────────────────────
                # The camera display always shows detections (no gate needed here).
                # The console print fires when rolling mean confidence >= threshold
                # over the last _PRINT_CONF_WINDOW frames.  This tolerates 1-2
                # shaky/low-confidence frames without resetting a counter.
                if nails:
                    for nail in nails:
                        nail_id  = nail['nail_id']
                        # Compute a slot key (40px snapped centroid) stable across
                        # small hand movements so the same nail maps to same slot.
                        _pmask = nail.get('mask')
                        _pslot = nail_id  # fallback to nail_id
                        if _pmask is not None:
                            _pmom = cv2.moments(_pmask)
                            if _pmom['m00'] > 0:
                                _s40 = 40
                                _pcx = int(_pmom['m10'] / _pmom['m00'] / _s40) * _s40
                                _pcy = int(_pmom['m01'] / _pmom['m00'] / _s40) * _s40
                                _pslot = (_pcx, _pcy)
                        # Skip if already printed for this slot
                        if _pslot in self._printed_slots:
                            continue
                        # Update rolling confidence for this slot.
                        # boundary_confidence reflects free-edge detection quality
                        # (0.3 for polished, 0.0 for trimmed / no free-edge found).
                        # These are both below _PRINT_CONF_THRESHOLD=0.40, so use
                        # geometry validity as the primary print-gate signal:
                        # any nail with valid length+width measurements is treated
                        # as confidently detected (0.6) regardless of free-edge
                        # quality.  Free-edge confidence is still used when higher.
                        _geom_valid = (
                            nail.get('nail_bed_length_mm') is not None
                            and nail.get('length_mm') is not None
                        )
                        _conf_now = max(
                            nail.get('boundary_confidence', 0.0),
                            0.6 if _geom_valid else 0.0,
                        )
                        if _pslot not in self._print_conf_history:
                            self._print_conf_history[_pslot] = deque(
                                maxlen=self._PRINT_CONF_WINDOW)
                        self._print_conf_history[_pslot].append(_conf_now)
                        _hist = self._print_conf_history[_pslot]
                        # Only print when we have enough frames AND mean conf is good
                        if (len(_hist) < self._PRINT_MIN_FRAMES or
                                float(np.mean(_hist)) < self._PRINT_CONF_THRESHOLD):
                            continue
                        # ── Stable enough — print once ───────────────────────
                        self._printed_slots.add(_pslot)
                        self.seen_nails.add(nail_id)
                        print(f"\n{'='*70}")
                        print(f"NEW NAIL DETECTED (Nail {nail_id}) - Frame {frame_count}")
                        print(f"{'='*70}")
                        
                        # Tilt status
                        tilt_info = nail.get('tilt_info', {})
                        if tilt_info.get('likely_tilted', False):
                            print(f"  Status: TILTED (confidence: {tilt_info.get('confidence', 0)*100:.0f}%)")
                            print(f"  Action: Keep nail PARALLEL and FLAT to camera")
                            
                        else:
                            print(f"  Measurements are ACCURATE")
                        
                        # Print ONLY nail bed details
                        if nail.get('nail_bed_length_mm') is not None:
                            print(f"\nNail Bed Geometry:")
                            print(f"  Shape Ratio: {nail.get('shape_ratio', nail['nail_bed_aspect_ratio']):.2f}x "
                                  "(length/breadth)")
                            print(f"  Full Nail Ratio: {nail['aspect_ratio']:.2f}x "
                                  "(length/breadth) [debug]")
                            print(f"  Area:        {nail['nail_bed_area_mm2']:.1f} mm²")

                            # Boundary detection diagnostics
                            _bmc   = nail.get('boundary_methods_count', 0)
                            _bconf = nail.get('boundary_confidence', 0.0)
                            if _bmc == 0:
                                print("  Boundary:    Anatomical prior (no methods detected free edge)")
                            elif _bmc == 1:
                                print(f"  Boundary:    Single method  "
                                      f"(conf={_bconf:.0%}) — verify visually")
                            elif _bmc == 2:
                                print(f"  Boundary:    2 methods agree  (conf={_bconf:.0%})")
                            else:
                                print(f"  Boundary:    {_bmc} methods agree  "
                                      f"(conf={_bconf:.0%}) \u2713 High confidence")

                        # Color sample quality
                        if nail.get('color_analysis'):
                            _purity = nail['color_analysis'].get('color_sample_purity', None)
                            _lq     = nail['color_analysis'].get('lighting_quality', 'ok')
                            if _purity is not None:
                                _purity_label = (
                                    "High"   if _purity >= 0.70 else
                                    "Medium" if _purity >= 0.45 else
                                    "Low \u2014 reposition finger"
                                )
                                print(f"\nColor Sample Quality: {_purity_label} ({_purity:.0%})")
                            if _lq != 'ok':
                                _lq_labels = {
                                    'overexposed':     'Too bright \u2014 reduce lighting',
                                    'underexposed':    'Too dark \u2014 improve lighting',
                                    'glare':           'Glare detected \u2014 reposition',
                                    'uneven_lighting': 'Uneven lighting',
                                }
                                print(f"Lighting:  \u26a0 {_lq_labels.get(_lq, _lq)}")

                        # Color information (standard CIE LAB: L* 0-100, a*/b* -128..+127)
                        if nail['nail_color_LAB']:
                            L, a, b = nail['nail_color_LAB']
                            L_cie = L * 100.0 / 255.0
                            a_cie = a - 128.0
                            b_cie = b - 128.0
                            print(f"\n🎨 Color:")
                            print(f"  LAB: L*={L_cie:.0f}, a*={a_cie:.0f}, b*={b_cie:.0f}")
                        
                        if nail['nail_color_BGR']:
                            b_val, g_val, r_val = nail['nail_color_BGR']
                            hex_str = nail['nail_color_HEX'] if nail['nail_color_HEX'] else "N/A"
                            print(f"  RGB: R={r_val:.0f}, G={g_val:.0f}, B={b_val:.0f} ({hex_str})")
                        
                        # Medical color analysis (L* deviation converted to CIE scale)
                        if nail['color_analysis']:
                            dev = nail['color_analysis']['deviation_from_normal']
                            _dev_L_cie = dev['L_deviation'] * 100.0 / 255.0
                            print(f"  Deviation: L*{_dev_L_cie:+.0f}, a*{dev['a_deviation']:+.0f}, b*{dev['b_deviation']:+.0f}")
                            
                            # Display nail-skin comparison if available
                            has_skin_ref = nail['color_analysis'].get('has_skin_reference', False)
                            if has_skin_ref and 'relative_metrics' in nail['color_analysis']:
                                rel = nail['color_analysis']['relative_metrics']
                                # L values are OpenCV L (0-255); convert to CIE L* (0-100)
                                _skin_L_cie  = rel['skin_reference_L'] * 100.0 / 255.0
                                _delta_L_cie = rel['delta_L'] * 100.0 / 255.0
                                print(f"\n📊 Nail vs Skin:")
                                print(f"  Skin L*: {_skin_L_cie:.0f}")
                                print(f"  ΔL* (lightness): {_delta_L_cie:+.0f}")
                                print(f"  Δb* (yellowness): {rel['delta_b']:+.0f}")
                            
                            # Display color screening status
                            if 'screening_flags' in nail['color_analysis']:
                                color_flags = [f for f in nail['color_analysis']['screening_flags'] 
                                             if 'color' in f.get('condition', '').lower() or 
                                             'pale' in f.get('condition', '').lower() or
                                             'yellow' in f.get('condition', '').lower() or
                                             'blue' in f.get('condition', '').lower()]
                                
                                if color_flags:
                                    color_flag = color_flags[0]
                                    condition = color_flag.get('condition', 'Unknown')
                                    severity = color_flag.get('severity', 'unknown')
                                    
                                    if severity == 'none':
                                        print(f"\n🎨 Color: Normal")
                                    else:
                                        # Remove "Color:" prefix if present and simplify
                                        display_condition = condition.replace('Color: ', '').replace('Color screening: ', '')
                                        print(f"\n🎨 Color: {display_condition}")
                            
                            # Display texture status
                            if 'texture_screening' in nail['color_analysis']:
                                tex = nail['color_analysis']['texture_screening']
                                if tex.get('status') == 'completed':
                                    result = tex.get('screening_result', 'unknown')
                                    if result == 'normal':
                                        print(f"\n🔬 Texture: Normal")
                                    else:
                                        print(f"\n🔬 Texture: {result.replace('_', ' ').title()}")
                            
                            # Display screening summary
                            summary = nail['color_analysis'].get('screening_summary', 'normal')
                            if summary == 'recommend_health_check':
                                print(f"\n⚠️  Screening: Recommend routine health check")
                            else:
                                print(f"\n✓ Screening: Normal")
                        
                        print()

            
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    logger.info("Exiting...")
                    break
                elif key == ord('s') or key == ord('S'):
                    screenshot_count += 1
                    filename = f"nail_screenshot_{screenshot_count}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Screenshot saved: {filename}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera closed")


def main():
    """Main entry point for real-time analysis."""
    try:
        analyzer = RealtimeNailAnalyzer()
        
        # Display calibration info
        cal_info = analyzer.calibrator.get_calibration_info()
        print("\n" + "="*70)
        print("REAL-TIME NAIL ANALYZER")
        print("="*70)
        print(f"Calibration: {cal_info['dpi']:.1f} DPI ({cal_info['pixels_per_mm']:.4f} px/mm)")
        print("\nControls:")
        print("  Q - Quit")
        print("  S - Save screenshot")
        print("="*70 + "\n")
        
        analyzer.run()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()