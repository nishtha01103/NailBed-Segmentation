#!/usr/bin/env python3
"""Real-time nail segmentation and analysis using webcam."""

import cv2
import numpy as np
import logging
from pathlib import Path
from collections import deque
from ultralytics import YOLO

from src.geometry_utils import extract_geometry, extract_geometry_nail_bed, extract_geometry_nail_bed_with_diagnostics
from src.color_utils import extract_nail_color
from src.calibration import MeasurementCalibrator
from config import (
    MODEL_PATH,
    MASK_THRESHOLD,
    MIN_MASK_PIXELS,
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
        
        # Temporal smoothing — 5-frame window balances stability with responsiveness.
        self.nail_history = deque(maxlen=5)  # 5 frames (~0.17 s at 30 fps)

        # --- Performance caches (Goals 1, 4, 5) ---
        # Goal 1: last raw YOLO masks; reused for the 2 frames between YOLO runs.
        self._cached_raw_masks = None
        # Goal 4 & 5: per-nail measurement cache keyed by (snap_cx, snap_cy).
        # Stores last computed nail-bed values; reused when mask area changes < 10%.
        # This freezes the boundary projection on stable frames and avoids
        # redundant PCA + slice analysis every frame.
        self._nail_perf_cache: dict = {}
        self._nail_perf_cache_max_size = 50  # max cached nail entries

        # PERFORMANCE OPTIMIZATION: Pre-create morphology kernels (avoid repeated creation)
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
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
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
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
        
        # Grayscale variance check — flat backgrounds / shadows have very
        # low intensity variance; real nails have visible texture.
        if mask is not None and mask.sum() > 0 and nail_data.get('_frame_ref') is not None:
            frame_ref = nail_data['_frame_ref']
            gray_frame = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
            nail_pixels_gray = gray_frame[mask > 0]
            if len(nail_pixels_gray) > 30:
                gray_std = float(np.std(nail_pixels_gray))
                if gray_std < 4.0:
                    logger.debug(f"Rejected: gray std {gray_std:.1f} (flat shadow/background)")
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
        if intensity_gradient > 15:
            indicators.append("Perspective shading")
        
        # Weighted indicator scoring
        indicator_weights = {
            "High brightness variation": 0.25,
            "Perspective shading": 0.25,
            "Low circularity": 0.15,
            "Extreme aspect ratio": 0.15,
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

        # --- Goal 1: Run YOLO only every 3rd frame; reuse cached masks otherwise ---
        if frame_count % 3 == 0 or self._cached_raw_masks is None:
            inference_frame = cv2.resize(frame, (640, 480))
            results = self.model(
                inference_frame,
                conf=REALTIME_DETECTION_CONFIDENCE,
                device="cpu",
                imgsz=480,
                verbose=False,
            )[0]
            if results.masks is not None:
                self._cached_raw_masks = results.masks.data.cpu().numpy()
            else:
                self._cached_raw_masks = None

        nails = []

        if self._cached_raw_masks is None:
            return frame, nails

        masks = self._cached_raw_masks
        
        for i, mask in enumerate(masks):
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

            # NOTE: Do NOT apply morphological ops here.
            # extract_geometry and extract_geometry_nail_bed both call
            # _clean_mask_morphology internally (controlled by config).
            # Applying morph ops twice over-erodes the mask, shrinks the
            # nail shape, and shifts the PCA axes — causing wrong ratios.
            
            # Extract geometry (pass frame for anatomical axis orientation)
            geom = extract_geometry(mask_binary, image=frame)
            if geom[0] is None:  # geometry extraction failed
                continue
            
            length_px, width_px, area_px = geom
            
            # EARLY REJECTION: Quick aspect ratio check before expensive operations
            # Calculate preliminary aspect ratio (avoid expensive operations if will fail)
            prelim_aspect_ratio = max(length_px, width_px) / min(length_px, width_px) if min(length_px, width_px) > 1e-6 else 0
            if not (MIN_NAIL_ASPECT_RATIO <= prelim_aspect_ratio <= MAX_NAIL_ASPECT_RATIO):
                logger.debug(f"Early reject: aspect ratio {prelim_aspect_ratio:.2f} out of range")
                continue
            
            # Estimate out-of-plane tilt — pass pre-computed LAB to avoid re-conversion
            tilt_info = self.estimate_nail_tilt(frame, mask_binary, lab_frame=lab_frame)
            
            # Reject measurement if tilt confidence > 0.5 (before expensive color extraction)
            if tilt_info.get('confidence', 0) > 0.5:
                logger.debug(f"Rejecting nail {i+1}: tilt confidence {tilt_info['confidence']:.2f} > 0.5")
                continue
            
            # --- Goals 4 & 5: Nail-bed boundary / PCA caching ---
            # Compute a coarse centroid key (20-px grid) for cache lookup.
            _moments = cv2.moments(mask_binary)
            _cache_key = None
            if _moments['m00'] > 0:
                _snap = 20
                _cx = int(_moments['m10'] / _moments['m00'] / _snap) * _snap
                _cy = int(_moments['m01'] / _moments['m00'] / _snap) * _snap
                _snap_area = int(area_px / 500) * 500  # snap to nearest 500 px²
                _cache_key = (_cx, _cy, _snap_area)

            _nb_cached = self._nail_perf_cache.get(_cache_key) if _cache_key else None
            _use_cached_nb = False
            _boundary_methods = 0
            _boundary_conf    = 0.0
            nail_bed_free_edge_present    = None
            nail_bed_free_edge_confidence = 0.0
            nail_bed_length_px = nail_bed_width_px = nail_bed_area_px = None
            if _nb_cached is not None and area_px > 0:
                # Skip cache if stored area is zero (corrupt entry) or current area is zero
                if _nb_cached['area_px'] <= 0 or area_px <= 0:
                    _nb_cached = None
            if _nb_cached is not None and area_px > 0:
                _area_delta = abs(area_px - _nb_cached['area_px']) / max(_nb_cached['area_px'], 1.0)
                _prev_methods = _nb_cached.get('boundary_methods', 0)
                _cache_is_low_confidence = _prev_methods <= 1
                if _area_delta < 0.10 and not _cache_is_low_confidence:
                    nail_bed_length_px = _nb_cached['nb_length']
                    nail_bed_width_px  = _nb_cached['nb_width']
                    nail_bed_area_px   = _nb_cached['nb_area']
                    _boundary_methods  = _prev_methods
                    _boundary_conf     = 0.0  # not recomputed from cache
                    _use_cached_nb = True

            if not _use_cached_nb:
                # Extract color FIRST — needed for polish detection before nail bed
                # Pass pre-computed uint8 LAB frame to avoid re-conversion inside extract_nail_color
                # Note: lab_frame is float32; extract_nail_color expects uint8 for cvtColor,
                # so convert only if extract_nail_color is updated to accept lab_frame kwarg.
                # TODO: pass lab_frame=lab_frame once extract_nail_color signature is updated
                lab_color, bgr_color, color_analysis, hex_color = extract_nail_color(frame, mask_binary)

                # Detect polish from color analysis to disable boundary detection
                # (polished nails disrupt LAB gradients used by the boundary algorithm)
                _is_polished_rt = False
                if color_analysis and color_analysis.get("is_polished_detected", False):
                    _is_polished_rt = True

                # Full pipeline: use diagnostics variant to get voting metadata
                nail_bed_length_px, nail_bed_width_px, nail_bed_area_px, nb_diag = (
                    extract_geometry_nail_bed_with_diagnostics(
                        frame, mask_binary,
                        lab_frame=lab_frame,
                        is_polished=_is_polished_rt
                    )
                )

                # Extract voting diagnostics from nb_diag
                nail_bed_free_edge_present    = nb_diag.get("free_edge_present") if nb_diag else None
                nail_bed_free_edge_confidence = nb_diag.get("free_edge_confidence", 0.0) if nb_diag else 0.0
                # New voting diagnostics
                _boundary_methods = nb_diag.get("boundary_methods_agreed", 0) if nb_diag else 0
                _boundary_conf    = nb_diag.get("boundary_confidence", 0.0) if nb_diag else 0.0

                # Store result if high-quality (Goals 4 & 5: freeze boundary)
                if _cache_key is not None and nail_bed_length_px is not None:
                    if len(self._nail_perf_cache) >= self._nail_perf_cache_max_size:
                        # Evict oldest entry (first key in insertion-ordered dict)
                        oldest_key = next(iter(self._nail_perf_cache))
                        del self._nail_perf_cache[oldest_key]
                    self._nail_perf_cache[_cache_key] = {
                        'area_px':   area_px,
                        'nb_length': nail_bed_length_px,
                        'nb_width':  nail_bed_width_px,
                        'nb_area':   nail_bed_area_px,
                        'boundary_methods': _boundary_methods,
                    }
            
            # Extract color only if not already done above (cached path)
            if _use_cached_nb:
                lab_color, bgr_color, color_analysis, hex_color = extract_nail_color(frame, mask_binary)
            
            # Convert to mm
            length_mm = self.calibrator.pixel_to_mm(length_px)
            width_mm = self.calibrator.pixel_to_mm(width_px)
            area_mm2 = self.calibrator.pixel_area_to_mm2(area_px)
            
            # Convert nail bed to mm
            nail_bed_length_mm = self.calibrator.pixel_to_mm(nail_bed_length_px) if nail_bed_length_px else None
            nail_bed_width_mm = self.calibrator.pixel_to_mm(nail_bed_width_px) if nail_bed_width_px else None
            nail_bed_area_mm2 = self.calibrator.pixel_area_to_mm2(nail_bed_area_px) if nail_bed_area_px else None
            
            # Calculate full-nail aspect ratio (directional: length/width, can be < 1)
            aspect_ratio = length_px / width_px if width_px > 1e-6 else 0
            # Directional nail-bed ratio: length/width (> 1 = portrait, < 1 = wide/thumb)
            nail_bed_aspect_ratio = (
                (nail_bed_length_px / nail_bed_width_px)
                if (nail_bed_length_px and nail_bed_width_px and nail_bed_width_px > 1e-6)
                else 0
            )
            
            nail_data = {
                'nail_id': i + 1,
                'length_px': length_px,
                'width_px': width_px,
                'area_px': area_px,
                'length_mm': length_mm,
                'width_mm': width_mm,
                'area_mm2': area_mm2,
                'aspect_ratio': aspect_ratio,
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
                'nail_color_LAB': lab_color,
                'nail_color_BGR': bgr_color,
                'nail_color_HEX': hex_color,
                'color_analysis': color_analysis,
                'mask': mask_binary,
                'tilt_info': tilt_info,  # Add tilt detection info
                '_frame_ref': frame,     # Temporary: used by is_valid_nail, removed after
            }
            
            # Always validate nails properly for accuracy
            if self.is_valid_nail(nail_data):
                nail_data.pop('_frame_ref', None)  # Don't keep full frame in history
                nails.append(nail_data)
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
            # Display "No nails detected"
            cv2.putText(output, "No nails detected", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            return output
        

        # Draw each nail
        for i, nail in enumerate(nails):
            mask = nail['mask']
            
            # Draw contour in green
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (0, 255, 0), 3)
            
            # Get bounding box for text placement
            x, y, box_w, box_h = cv2.boundingRect(contours[0]) if contours else (10, 60 + i*120, 300, 110)
            
            # Text position
            text_y = max(50, y - 20)
            line_height = 28
            
            # Display nail number with tilt status
            tilt_info = nail.get('tilt_info', {})
            tilt_status = "⚠️ TILTED" if tilt_info.get('likely_tilted', False) else "✓"
            tilt_color = (0, 165, 255) if tilt_info.get('likely_tilted', False) else (0, 255, 0)  # Orange if tilted, green if perpendicular
            
            cv2.putText(output, f"Nail {nail['nail_id']} - {tilt_status}", 
                       (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, tilt_color, 2)
            
            text_y += line_height
            # Display tilt warning if nail is tilted
            if tilt_info.get('likely_tilted', False):
                cv2.putText(output, f"⚠️  Keep nail PARALLEL to camera", 
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                text_y += line_height
                cv2.putText(output, f"Confidence: {tilt_info.get('confidence', 0)*100:.0f}%", 
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 1)
                text_y += line_height
            
            # Calculate and display length to breadth ratio.
            # Full nail: PCA major/minor — always ≥ 1 (longest / shortest).
            length_breadth_ratio = nail['length_mm'] / nail['width_mm'] if nail['width_mm'] > 0 else 0
            ratio_color = (200, 100, 100) if tilt_info.get('likely_tilted', False) else (255, 255, 0)
            cv2.putText(output, f"Nail L/B: {length_breadth_ratio:.2f}  (>=1, longer=higher)",
                       (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ratio_color, 1)

            text_y += line_height
            # Nail bed ratio: CAN be <1 (bed wider than long) or >1 (bed longer than wide).
            if nail.get('nail_bed_length_mm') is not None:
                nb_w = nail.get('nail_bed_width_mm') or 0
                nail_bed_ratio = nail['nail_bed_length_mm'] / nb_w if nb_w > 0 else 0
                label_suffix = ">1 long" if nail_bed_ratio >= 1 else "<1 wide"
                cv2.putText(output, f"Bed  L/B: {nail_bed_ratio:.2f}  ({label_suffix})",
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 100), 1)
                text_y += line_height

            # Show how many methods agreed on the boundary
            _bmc = nail.get('boundary_methods_count', 0)
            _bconf = nail.get('boundary_confidence', 0.0)
            if _bmc > 0:
                _conf_color = (
                    (0, 255, 0)   if _bmc >= 3 else   # green: 3-4 methods agree
                    (0, 200, 255) if _bmc == 2 else    # yellow: 2 methods agree
                    (0, 128, 255)                       # orange: 1 method only
                )
                _bm_label = {1: "1 method", 2: "2 methods", 3: "3 methods", 4: "4 methods"}
                cv2.putText(
                    output,
                    f"Bed boundary: {_bm_label.get(_bmc, f'{_bmc}')} "
                    f"conf={_bconf:.0%}",
                    (20, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, _conf_color, 1
                )
                text_y += line_height
            elif nail.get('nail_bed_length_mm') is not None:
                # Boundary came from anatomical prior
                cv2.putText(
                    output,
                    "Bed boundary: anatomical prior",
                    (20, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (128, 128, 128), 1
                )
                text_y += line_height

            cv2.putText(output, f"Area: {nail['area_mm2']:.0f} mm²", 
                       (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
            
            text_y += line_height
            cv2.putText(output, f"Shape: {nail['aspect_ratio']:.2f}x", 
                       (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
            
            # Color info
            if nail['nail_color_LAB']:
                L, a, b = nail['nail_color_LAB']
                text_y += line_height
                cv2.putText(output, f"Color LAB: L={L:.0f} a={a:.0f} b={b:.0f}", 
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 1)
            
            # RGB color info
            if nail['nail_color_BGR']:
                b_val, g_val, r_val = nail['nail_color_BGR']
                text_y += line_height
                cv2.putText(output, f"Color RGB: R={r_val:.0f} G={g_val:.0f} B={b_val:.0f}", 
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (int(b_val), int(g_val), int(r_val)), 1)
            
            # Color analysis deviations
            if nail['color_analysis']:
                dev = nail['color_analysis']['deviation_from_normal']
                text_y += line_height
                dev_str = f"Dev: L{dev['L_deviation']:+.0f} a{dev['a_deviation']:+.0f} b{dev['b_deviation']:+.0f}"
                cv2.putText(output, dev_str, 
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 150, 255), 1)
                
                # Display screening flags on video
                if 'screening_flags' in nail['color_analysis']:
                    flags = nail['color_analysis']['screening_flags']
                    confidence = nail['color_analysis'].get('screening_confidence', 0)
                    
                    # Show first non-normal flag (most important)
                    screening_text = ""
                    screening_color = (0, 255, 0)  # Green default
                    
                    for flag in flags:
                        condition = flag.get('condition', '').lower()
                        severity = flag.get('severity', 'mild')
                        
                        if severity != 'none':  # Not normal
                            # Found concerning flag - color code by severity
                            if severity == 'moderate':
                                emoji = "⚠️"
                                base_color = (0, 150, 255)  # Orange for moderate
                            elif severity == 'uncertain':
                                emoji = "?"
                                base_color = (200, 200, 200)  # Gray for uncertain
                            else:  # mild
                                emoji = "⚠"
                                base_color = (0, 200, 255)  # Yellow for mild
                            
                            # Condition-specific display
                            if 'pallor' in condition:
                                screening_text = f"{emoji} Pallor"
                                screening_color = (100, 150, 255)  # Light red
                            elif 'yellow' in condition:
                                screening_text = f"{emoji} Yellow"
                                screening_color = base_color
                            elif 'blue' in condition:
                                screening_text = f"{emoji} Bluish"
                                screening_color = (255, 150, 100)  # Light blue
                            elif 'pale' in condition:
                                screening_text = f"{emoji} Pale"
                                screening_color = (180, 180, 255)  # Light
                            elif 'redness' in condition:
                                screening_text = f"{emoji} Red"
                                screening_color = (100, 100, 255)  # Red
                            else:
                                screening_text = f"{emoji} {flag.get('condition', 'Unknown')[:15]}"
                                screening_color = base_color
                            break
                    else:
                        # No concerning flags
                        screening_text = "✓ Normal"
                        screening_color = (0, 255, 0)  # Green
                    
                    if screening_text:
                        text_y += line_height
                        cv2.putText(output, screening_text, 
                                   (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, screening_color, 2)
                
                # Show lighting quality warning if applicable
                if nail.get('color_analysis'):
                    _lq = nail['color_analysis'].get('lighting_quality', 'ok')
                    if _lq != 'ok':
                        _lq_msg = {
                            'overexposed':     '\u26a0 Too bright \u2014 reduce lighting',
                            'underexposed':    '\u26a0 Too dark \u2014 improve lighting',
                            'glare':           '\u26a0 Glare detected \u2014 reposition',
                            'uneven_lighting': '\u26a0 Uneven lighting',
                        }.get(_lq, f'\u26a0 Lighting: {_lq}')
                        text_y += line_height
                        cv2.putText(output, _lq_msg,
                                   (20, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.6, (0, 128, 255), 1)

                    # Polish detection warning
                    if nail['color_analysis'].get('is_polished_detected', False):
                        text_y += line_height
                        cv2.putText(output, '\u26a0 Polish detected \u2014 remove for screening',
                                   (20, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.6, (0, 200, 255), 1)
                
                color = tuple(int(c) for c in nail['nail_color_BGR'])
    
                # Color patch positioned to the left of the thumbnail
                patch_x = max(0, w - 250)
                patch_y = 50 + i * 120 + 35  # Center vertically with thumbnail
    
                cv2.rectangle(output,
                  (patch_x, patch_y),
                  (patch_x + 80, patch_y + 80),
                  color,
                  -1)
    
    
    # Border
                cv2.rectangle(output,
                  (patch_x, patch_y),
                  (patch_x + 80, patch_y + 80),
                  (255, 255, 255),
                  2)
                        # Show zoomed masked nail preview (skip drawing every 3rd frame for speed)
            if frame_count % 3 != 0:
                mask = nail['mask']
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Get bounding box of the nail with padding for context
                    x, y, box_w, box_h = cv2.boundingRect(contours[0])
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
                
                # Reset tracking if no nails detected (allows re-detection when nails reappear)
                if not nails:
                    self.seen_nails.clear()
                
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
                
                # Print details for NEWLY detected nails (print only once per nail)
                if nails:
                    for nail in nails:
                        nail_id = nail['nail_id']
                        if nail_id not in self.seen_nails:
                            # NEW NAIL DETECTED - Print details
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
                                print(f"  Shape Ratio: {nail['nail_bed_aspect_ratio']:.2f}x "
                                      "(length/breadth)")
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

                            # Color information
                            if nail['nail_color_LAB']:
                                L, a, b = nail['nail_color_LAB']
                                print(f"\n🎨 Color:")
                                print(f"  LAB: L={L:.0f}, a={a:.0f}, b={b:.0f}")
                            
                            if nail['nail_color_BGR']:
                                b_val, g_val, r_val = nail['nail_color_BGR']
                                hex_str = nail['nail_color_HEX'] if nail['nail_color_HEX'] else "N/A"
                                print(f"  RGB: R={r_val:.0f}, G={g_val:.0f}, B={b_val:.0f} ({hex_str})")
                            
                            # Medical color analysis
                            if nail['color_analysis']:
                                dev = nail['color_analysis']['deviation_from_normal']
                                print(f"  Deviation: L{dev['L_deviation']:+.0f}, a{dev['a_deviation']:+.0f}, b{dev['b_deviation']:+.0f}")
                                
                                # Display nail-skin comparison if available
                                has_skin_ref = nail['color_analysis'].get('has_skin_reference', False)
                                if has_skin_ref and 'relative_metrics' in nail['color_analysis']:
                                    rel = nail['color_analysis']['relative_metrics']
                                    print(f"\n📊 Nail vs Skin:")
                                    print(f"  Skin L: {rel['skin_reference_L']:.0f}")
                                    print(f"  Δ Lightness: {rel['delta_L']:+.0f}")
                                    print(f"  Δ Yellowness: {rel['delta_b']:+.0f}")
                                
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
