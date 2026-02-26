"""Main nail analysis pipeline using YOLOv8 instance segmentation."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from .color_utils import extract_nail_color, normalize_white_balance
from .geometry_utils import extract_geometry, extract_geometry_nail_bed_with_diagnostics
from .calibration import MeasurementCalibrator, batch_convert_measurements, estimate_dpi_from_known_nail
from config import (
    MASK_THRESHOLD,
    MIN_MASK_PIXELS,
    LOG_LEVEL,
    LOG_FORMAT,
    DEFAULT_PIXELS_PER_MM,
    USE_AUTOMATIC_CALIBRATION,
    TYPICAL_NAIL_LENGTH_MM,
    INCLUDE_REAL_WORLD_MEASUREMENTS,
    NORMALIZE_WHITE_BALANCE,
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class NailAnalyzer:
    """Multi-nail instance segmentation and analysis pipeline.
    
    Uses YOLOv8-seg model to detect and segment nails, then extracts
    geometric measurements and natural color information per nail.
    Supports optional calibration for real-world unit conversion.
    """

    def __init__(self, model_path: str, calibrator: Optional[MeasurementCalibrator] = None):
        """Initialize the nail analyzer with a YOLOv8 segmentation model.
        
        Args:
            model_path: Path to YOLOv8 .pt model file
            calibrator: Optional MeasurementCalibrator for real-world measurements.
                       If None, uses default or auto-estimation.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model fails to load
        """
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e

        # Initialize calibrator
        self.calibrator = calibrator or MeasurementCalibrator(DEFAULT_PIXELS_PER_MM)
        cal_info = self.calibrator.get_calibration_info()
        logger.info(f"Calibration: {cal_info['dpi']:.1f} DPI ({cal_info['pixels_per_mm']:.4f} px/mm)")

    def analyze(self, image_path: str) -> List[Dict[str, Any]]:
        """Analyze nails in an image and extract measurements and colors.
        
        Args:
            image_path: Path to input image file
        
        Returns:
            List of dictionaries with nail results (nail_id, length_px, width_px,
            area_px, nail_color_LAB, nail_color_BGR)
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be read
        """
        # Validate input
        img_file = Path(image_path)
        if not img_file.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        logger.info(f"Analyzing image: {image_path}")

        # Apply white balance normalization if enabled (reduces lighting bias)
        if NORMALIZE_WHITE_BALANCE:
            image = normalize_white_balance(image)
            logger.debug("White balance normalization applied")

        # Run YOLOv8 segmentation
        try:
            results = self.model(image_path)[0]
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return []

        if results.masks is None:
            logger.warning("No nails detected in image")
            return []

        # Extract per-nail results
        masks = results.masks.data.cpu().numpy()
        nail_results = []
        valid_nail_count = 0

        for i, mask in enumerate(masks):
            # Resize mask to original image dimensions
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask = (mask > MASK_THRESHOLD).astype(np.uint8) * 255

            # Filter out very small masks (likely noise)
            if mask.sum() < MIN_MASK_PIXELS:
                logger.debug(f"Skipping mask {i+1}: too small ({mask.sum()} pixels)")
                continue

            # Extract geometry (pass image for anatomical axis orientation)
            length_px, width_px, area_px = extract_geometry(mask, image=image)
            if length_px is None:
                logger.debug(f"Skipping mask {i+1}: geometry extraction failed (invalid contour)")
                continue

            # Extract geometry for nail bed only (free edge excluded)
            nail_bed_length_px = nail_bed_width_px = nail_bed_area_px = None
            nail_bed_free_edge_present = None
            nail_bed_free_edge_confidence = 0.0
            try:
                nail_bed_length_px, nail_bed_width_px, nail_bed_area_px, nb_diag = (
                    extract_geometry_nail_bed_with_diagnostics(image, mask)
                )
                nail_bed_free_edge_present    = nb_diag.get("free_edge_present")
                nail_bed_free_edge_confidence = nb_diag.get("free_edge_confidence", 0.0)
            except Exception as e:
                logger.debug(f"Nail bed extraction failed for mask {i+1}: {e}")

            # Extract color with medical analysis
            lab_color, bgr_color, color_analysis, hex_color = extract_nail_color(image, mask)
            if lab_color is None:
                logger.warning(f"Nail {valid_nail_count + 1}: color extraction failed")
                color_analysis = None

            # Full-nail directional ratio: length (PCA major axis) / width (PCA minor axis).
            # PCA always assigns the longest dimension to the major axis, so this is
            # always ≥ 1.0.  A ratio of 1.0 means square-ish (wide thumbnail); higher
            # values mean a longer, more slender nail.
            aspect_ratio = length_px / width_px if width_px > 0 else 0
            image_area_px = image.shape[0] * image.shape[1]
            relative_size = (area_px / image_area_px) * 100  # As percentage

            # Calculate nail bed aspect ratio (directional: length / width)
            # length = extent along PCA major axis (proximal-distal)
            # width  = extent along PCA minor axis (lateral breadth)
            # Ratio > 1 → longer than wide;  Ratio < 1 → wider than long (e.g. thumb)
            nail_bed_aspect_ratio = (
                (nail_bed_length_px / nail_bed_width_px)
                if (nail_bed_length_px and nail_bed_width_px and nail_bed_width_px > 0)
                else 0
            )
            nail_bed_relative_size = (nail_bed_area_px / image_area_px) * 100 if nail_bed_area_px else 0  # As percentage

            # Build nail result
            nail_result = {
                "nail_id": valid_nail_count + 1,
                "length_px": round(float(length_px), 2),
                "width_px": round(float(width_px), 2),
                "area_px": round(float(area_px), 2),
                "aspect_ratio": round(float(aspect_ratio), 2),
                "relative_size_percent": round(relative_size, 2),
                # Nail bed measurements (free edge excluded)
                "nail_bed_length_px": round(float(nail_bed_length_px), 2) if nail_bed_length_px else None,
                "nail_bed_width_px": round(float(nail_bed_width_px), 2) if nail_bed_width_px else None,
                "nail_bed_area_px": round(float(nail_bed_area_px), 2) if nail_bed_area_px else None,
                "nail_bed_aspect_ratio": round(float(nail_bed_aspect_ratio), 3) if nail_bed_aspect_ratio else None,
                "nail_bed_relative_size_percent": round(nail_bed_relative_size, 2) if nail_bed_relative_size else None,
                # Free-edge detection diagnostics
                "nail_bed_free_edge_present": nail_bed_free_edge_present,
                "nail_bed_free_edge_confidence": round(float(nail_bed_free_edge_confidence), 3),
                "nail_color_LAB": lab_color,
                "nail_color_BGR": bgr_color,
                "nail_color_HEX": hex_color,
            }
            
            # Add medical color analysis if available
            if color_analysis:
                nail_result["color_analysis"] = color_analysis
            
            nail_results.append(nail_result)
            valid_nail_count += 1

        logger.info(f"Successfully analyzed {valid_nail_count} nails")

        # Apply automatic calibration if enabled
        if USE_AUTOMATIC_CALIBRATION and nail_results:
            first_nail_length = nail_results[0].get("length_px")
            if first_nail_length and first_nail_length > 0:
                auto_ppm = estimate_dpi_from_known_nail(
                    first_nail_length, TYPICAL_NAIL_LENGTH_MM
                )
                self.calibrator.set_calibration_from_reference(
                    first_nail_length, TYPICAL_NAIL_LENGTH_MM, "mm"
                )

        # Add real-world measurements if enabled
        if INCLUDE_REAL_WORLD_MEASUREMENTS and nail_results:
            nail_results = batch_convert_measurements(nail_results, self.calibrator)

        return nail_results

