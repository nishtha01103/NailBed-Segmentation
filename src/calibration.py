"""Real-world measurement calibration utilities.

Provides functions to convert pixel measurements to physical units (mm, cm)
using calibration reference markers or manual DPI input.
"""

from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MeasurementCalibrator:
    """Converts pixel measurements to real-world units (mm, cm)."""

    def __init__(self, pixels_per_mm: Optional[float] = None):
        """Initialize calibrator with conversion factor.
        
        Args:
            pixels_per_mm: Pixels per millimeter. If None, uses default estimation.
        """
        self.pixels_per_mm = pixels_per_mm or 5.0  # Default: ~200 DPI
        logger.debug(f"Initialized calibrator: {self.pixels_per_mm:.2f} px/mm")

    def pixel_to_mm(self, pixels: float) -> float:
        """Convert pixel measurement to millimeters.
        
        Args:
            pixels: Measurement in pixels
        
        Returns:
            Measurement in millimeters
        """
        return float(pixels / self.pixels_per_mm)

    def pixel_to_cm(self, pixels: float) -> float:
        """Convert pixel measurement to centimeters.
        
        Args:
            pixels: Measurement in pixels
        
        Returns:
            Measurement in centimeters
        """
        return float(pixels / (self.pixels_per_mm * 10))

    def pixel_area_to_mm2(self, area_px: float) -> float:
        """Convert pixel area to square millimeters.
        
        Args:
            area_px: Area in pixels²
        
        Returns:
            Area in mm²
        """
        return float(area_px / (self.pixels_per_mm ** 2))

    def set_calibration_from_reference(
        self, reference_pixel_length: float, reference_real_length: float, unit: str = "mm"
    ) -> None:
        """Set calibration using a known reference object.
        
        Args:
            reference_pixel_length: Measured pixel length of reference object
            reference_real_length: Actual real-world length of reference object
            unit: Unit of reference_real_length ('mm' or 'cm')
        
        Example:
            calibrator.set_calibration_from_reference(
                reference_pixel_length=150,  # pixels
                reference_real_length=30,    # mm
            )
        """
        if unit.lower() == "cm":
            reference_real_length *= 10  # Convert to mm

        self.pixels_per_mm = reference_pixel_length / reference_real_length
        logger.info(
            f"Calibration set from reference: {self.pixels_per_mm:.4f} px/mm "
            f"({reference_pixel_length}px = {reference_real_length}mm)"
        )

    def get_calibration_info(self) -> dict:
        """Get current calibration information.
        
        Returns:
            Dictionary with calibration metrics
        """
        dpi = self.pixels_per_mm * 25.4  # DPI = px/mm * 25.4
        return {
            "pixels_per_mm": round(self.pixels_per_mm, 4),
            "pixels_per_cm": round(self.pixels_per_mm * 10, 4),
            "dpi": round(dpi, 1),
            "ppi": round(dpi, 1),  # Pixels per inch
        }


def estimate_dpi_from_known_nail(
    nail_length_pixels: float, typical_nail_length_mm: float = 15.0
) -> float:
    """Estimate DPI using typical fingernail length as reference.
    
    Average adult nail length is ~15mm. This heuristic can estimate
    DPI when no explicit calibration object is available.
    
    Args:
        nail_length_pixels: Detected nail length in pixels
        typical_nail_length_mm: Assumed real-world nail length (default: 15mm)
    
    Returns:
        Estimated pixels per millimeter
    
    Note:
        This is a rough estimate. Actual calibration is recommended for
        precise measurements.
    """
    if nail_length_pixels <= 0:
        logger.warning("Invalid nail length for DPI estimation")
        return 5.0  # Default fallback

    ppm = nail_length_pixels / typical_nail_length_mm
    logger.debug(f"Estimated PPM from nail: {ppm:.2f} px/mm ({ppm*25.4:.1f} DPI)")
    return ppm


def batch_convert_measurements(
    results: list, calibrator: MeasurementCalibrator
) -> list:
    """Add real-world measurements to nail analysis results.
    
    Converts both full nail and nail bed measurements from pixels to mm/cm.
    
    Args:
        results: List of nail analysis dictionaries
        calibrator: MeasurementCalibrator instance
    
    Returns:
        Results with added real-world measurements
    """
    for nail in results:
        # Full nail measurements
        if nail["length_px"] is not None:
            nail["length_mm"] = round(calibrator.pixel_to_mm(nail["length_px"]), 2)

        if nail["width_px"] is not None:
            nail["width_mm"] = round(calibrator.pixel_to_mm(nail["width_px"]), 2)

        if nail["area_px"] is not None:
            nail["area_mm2"] = round(calibrator.pixel_area_to_mm2(nail["area_px"]), 2)

        # Nail bed measurements (free edge excluded)
        if nail.get("nail_bed_length_px") is not None:
            nail["nail_bed_length_mm"] = round(calibrator.pixel_to_mm(nail["nail_bed_length_px"]), 2)

        if nail.get("nail_bed_width_px") is not None:
            nail["nail_bed_width_mm"] = round(calibrator.pixel_to_mm(nail["nail_bed_width_px"]), 2)

        if nail.get("nail_bed_area_px") is not None:
            nail["nail_bed_area_mm2"] = round(calibrator.pixel_area_to_mm2(nail["nail_bed_area_px"]), 2)

    return results
def is_calibrated(self) -> bool:
    return False  # calibration not required — ratio output only
