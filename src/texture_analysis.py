"""Simplified nail texture screening for surface analysis.

DISCLAIMER: This is a screening tool only, not a medical diagnostic system.
Results indicate potential texture deviations but require professional evaluation.

Focus areas:
- Longitudinal ridge prominence (vertical texture)
- Surface roughness (overall texture uniformity)
- Image quality validation

This module uses statistical normalization and temporal smoothing to reduce false positives.
"""

from typing import Dict, Optional, List
from collections import deque
import cv2
import numpy as np


# Global temporal smoothing buffers (for video processing)
_ridge_history = deque(maxlen=5)
_roughness_history = deque(maxlen=5)
_baseline_ridge_values = []
_baseline_roughness_values = []


def analyze_nail_texture(
    image: np.ndarray,
    mask: np.ndarray,
    nail_orientation: Optional[float] = None
) -> Dict:
    """Simplified texture screening for nail surface analysis.
    
    DISCLAIMER: This is a screening tool only, not a medical diagnostic system.
    Results indicate potential texture deviations requiring professional evaluation.
    
    This function:
    1. Validates image quality (blur, resolution, contrast)
    2. Measures longitudinal ridge strength (Sobel vertical gradients)
    3. Measures surface roughness (local variance)
    4. Uses statistical normalization to reduce false positives
    5. Applies temporal smoothing for video streams
    
    Args:
        image: BGR image
        mask: Binary nail mask (255 = nail, 0 = background)
        nail_orientation: Not used (reserved for future orientation correction)
    
    Returns:
        Dictionary with quality score, metrics, and screening result
    """
    # Extract nail region
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nail_region = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Get bounding box for focused analysis
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            'image_quality_score': 0.0,
            'screening_result': 'insufficient_quality',
            'message': 'No valid nail region detected'
        }
    
    x, y, w, h = cv2.boundingRect(contours[0])
    nail_crop = nail_region[y:y+h, x:x+w]
    mask_crop = mask[y:y+h, x:x+w]
    
    if nail_crop.size == 0 or np.sum(mask_crop > 0) < 100:
        return {
            'image_quality_score': 0.0,
            'screening_result': 'insufficient_quality',
            'message': 'Nail region too small for analysis'
        }
    
    # Apply CLAHE for lighting normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    nail_crop_normalized = clahe.apply(nail_crop)
    
    # 1. IMAGE QUALITY ASSESSMENT
    quality_score, quality_details = _assess_image_quality(nail_crop_normalized, mask_crop)
    
    # Early exit if quality insufficient
    if quality_score < 0.6:
        return {
            'image_quality_score': quality_score,
            'quality_details': quality_details,
            'screening_result': 'insufficient_quality',
            'message': 'Image quality too low for reliable texture screening',
            'recommendation': 'Use better lighting, ensure focus, minimum 50x50px nail region',
            'disclaimer': 'This is a screening tool only, not a medical diagnosis. Consult a healthcare professional for evaluation.'
        }
    
    # 2. RIDGE STRENGTH MEASUREMENT (Sobel vertical gradient)
    ridge_strength = _measure_ridge_strength(nail_crop_normalized, mask_crop)
    
    # 3. SURFACE ROUGHNESS MEASUREMENT (local variance)
    surface_roughness = _measure_surface_roughness(nail_crop_normalized, mask_crop)
    
    # 4. TEMPORAL SMOOTHING (average over recent frames)
    ridge_strength_smoothed = _apply_temporal_smoothing(ridge_strength, _ridge_history)
    roughness_smoothed = _apply_temporal_smoothing(surface_roughness, _roughness_history)
    
    # 5. STATISTICAL NORMALIZATION (z-score based detection)
    ridge_deviation = _calculate_deviation(ridge_strength_smoothed, _baseline_ridge_values)
    roughness_deviation = _calculate_deviation(roughness_smoothed, _baseline_roughness_values)
    
    # Update baselines (maintain rolling window of normal values)
    _update_baseline(_baseline_ridge_values, ridge_strength_smoothed, max_size=30)
    _update_baseline(_baseline_roughness_values, roughness_smoothed, max_size=30)
    
    # 6. SCREENING DECISION (flag only persistent, significant deviations)
    screening_result, message = _determine_screening_result(
        ridge_deviation, 
        roughness_deviation,
        quality_score
    )
    
    return {
        'image_quality_score': round(quality_score, 2),
        'quality_details': quality_details,
        'ridge_strength': round(ridge_strength_smoothed, 2),
        'surface_roughness': round(roughness_smoothed, 2),
        'ridge_deviation_zscore': round(ridge_deviation, 2),
        'roughness_deviation_zscore': round(roughness_deviation, 2),
        'screening_result': screening_result,
        'message': message,
        'disclaimer': 'This is a screening tool only, not a medical diagnosis. Consult a healthcare professional for evaluation.'
    }


def _assess_image_quality(nail_crop: np.ndarray, mask_crop: np.ndarray) -> tuple:
    """Assess image quality for reliable texture analysis.
    
    Checks:
    - Sharpness (Laplacian variance - detects blur)
    - Resolution (minimum pixel count)
    - Contrast (intensity range)
    
    Args:
        nail_crop: Grayscale nail region
        mask_crop: Binary mask
    
    Returns:
        (quality_score, details_dict): Score 0-1 and breakdown
    """
    valid_pixels = mask_crop > 0
    pixel_count = np.sum(valid_pixels)
    
    if pixel_count < 100:
        return 0.0, {'error': 'Region too small'}
    
    # 1. Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(nail_crop, cv2.CV_64F)
    laplacian_variance = np.var(laplacian[valid_pixels])
    sharpness_score = min(laplacian_variance / 200.0, 1.0)
    
    # 2. Resolution (pixel count)
    resolution_score = min(pixel_count / 2500.0, 1.0)  # 50x50 = 2500px minimum
    
    # 3. Contrast (intensity range)
    pixels = nail_crop[valid_pixels]
    intensity_range = float(np.max(pixels) - np.min(pixels))
    contrast_score = min(intensity_range / 120.0, 1.0)  # Need at least 120 range
    
    # Overall quality (weighted)
    quality_score = (
        0.5 * sharpness_score +    # Blur is most important
        0.3 * resolution_score +    # Size matters
        0.2 * contrast_score        # Contrast helps but less critical
    )
    
    details = {
        'sharpness': round(laplacian_variance, 1),
        'resolution_pixels': int(pixel_count),
        'contrast_range': round(intensity_range, 1),
        'sharpness_score': round(sharpness_score, 2),
        'resolution_score': round(resolution_score, 2),
        'contrast_score': round(contrast_score, 2)
    }
    
    return quality_score, details


def _measure_ridge_strength(nail_crop: np.ndarray, mask_crop: np.ndarray) -> float:
    """Measure longitudinal ridge prominence using Sobel vertical gradients.
    
    Vertical gradients capture the strength of longitudinal (vertical) ridges.
    Higher values indicate more prominent ridge patterns.
    
    Args:
        nail_crop: Grayscale nail region (CLAHE normalized)
        mask_crop: Binary mask
    
    Returns:
        Ridge strength (mean absolute vertical gradient)
    """
    valid_pixels = mask_crop > 0
    
    # Sobel vertical gradient (detects horizontal edges = longitudinal ridges)
    sobel_y = cv2.Sobel(nail_crop, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y_abs = np.abs(sobel_y)
    
    # Average gradient strength over valid region
    ridge_strength = float(np.mean(sobel_y_abs[valid_pixels]))
    
    return ridge_strength


def _measure_surface_roughness(nail_crop: np.ndarray, mask_crop: np.ndarray) -> float:
    """Measure surface roughness using local variance.
    
    Local variance captures texture irregularity. Higher values indicate
    rougher, less uniform surface.
    
    Args:
        nail_crop: Grayscale nail region (CLAHE normalized)
        mask_crop: Binary mask
    
    Returns:
        Surface roughness (mean local variance)
    """
    valid_pixels = mask_crop > 0
    nail_float = nail_crop.astype(np.float32)
    
    # Local mean and variance (5x5 window)
    window_size = 5
    mean_filtered = cv2.blur(nail_float, (window_size, window_size))
    sq_filtered = cv2.blur(nail_float**2, (window_size, window_size))
    variance = sq_filtered - mean_filtered**2
    variance = np.maximum(variance, 0)  # Handle numerical errors
    
    # Average variance over valid region
    roughness = float(np.mean(variance[valid_pixels]))
    
    return roughness


def _apply_temporal_smoothing(current_value: float, history_buffer: deque) -> float:
    """Apply temporal smoothing to reduce frame-to-frame noise.
    
    Averages current value with recent history (up to 5 frames).
    This prevents single-frame spikes from triggering false positives.
    
    Args:
        current_value: Current frame measurement
        history_buffer: Deque containing recent values
    
    Returns:
        Smoothed value (average over buffer)
    """
    history_buffer.append(current_value)
    return float(np.mean(history_buffer))


def _calculate_deviation(value: float, baseline_values: List[float]) -> float:
    """Calculate z-score deviation from baseline.
    
    Z-score = (value - mean) / std_dev
    
    A z-score > 2.0 indicates value is more than 2 standard deviations
    above baseline (statistically significant deviation).
    
    Args:
        value: Current measurement
        baseline_values: Historical baseline measurements
    
    Returns:
        Z-score (standard deviations from baseline mean)
    """
    if len(baseline_values) < 5:
        # Not enough baseline data yet - assume normal
        return 0.0
    
    baseline_mean = np.mean(baseline_values)
    baseline_std = np.std(baseline_values)
    
    if baseline_std < 0.1:
        # Very low variance - avoid division by near-zero
        return 0.0
    
    z_score = (value - baseline_mean) / baseline_std
    return float(z_score)


def _update_baseline(baseline_values: List[float], new_value: float, max_size: int = 30) -> None:
    """Update baseline with new value (rolling window).
    
    Maintains a rolling window of recent "normal" measurements.
    Only adds values that aren't extreme outliers to avoid poisoning baseline.
    
    Args:
        baseline_values: List of baseline measurements (modified in place)
        new_value: New measurement to potentially add
        max_size: Maximum baseline window size
    """
    # Only add to baseline if not an extreme outlier
    if len(baseline_values) >= 5:
        mean = np.mean(baseline_values)
        std = np.std(baseline_values)
        z_score = abs((new_value - mean) / (std + 0.1))
        
        # Don't add extreme outliers to baseline (z > 3)
        if z_score > 3.0:
            return
    
    baseline_values.append(new_value)
    
    # Maintain rolling window
    if len(baseline_values) > max_size:
        baseline_values.pop(0)


def _determine_screening_result(
    ridge_z: float,
    roughness_z: float,
    quality_score: float
) -> tuple:
    """Determine screening result based on deviations.
    
    Uses conservative thresholds:
    - Z-score > 2.0 = statistically significant (2 std deviations)
    - Requires persistent elevation (temporal smoothing already applied)
    
    Args:
        ridge_z: Ridge strength z-score
        roughness_z: Surface roughness z-score
        quality_score: Image quality (0-1)
    
    Returns:
        (screening_result, message): Result code and explanation
    """
    # Both elevated (most concerning)
    if ridge_z > 2.0 and roughness_z > 2.0:
        return (
            'elevated_ridges_and_roughness',
            'Both ridge prominence and surface roughness elevated above baseline. '
            'May indicate texture changes requiring further evaluation.'
        )
    
    # Ridge elevation only
    if ridge_z > 2.0:
        return (
            'elevated_ridges',
            'Longitudinal ridge prominence elevated above baseline. '
            'Normal with aging, but consider evaluation if new or progressive.'
        )
    
    # Roughness elevation only
    if roughness_z > 2.0:
        return (
            'rough_surface',
            'Surface roughness elevated above baseline. '
            'May indicate texture changes.'
        )
    
    # Normal
    return (
        'normal',
        'Texture metrics within normal baseline range.'
    )


def reset_baselines():
    """Reset baseline tracking (useful when switching between different nails/subjects).
    
    Call this when starting analysis of a new nail or new subject to avoid
    cross-contamination of baseline statistics.
    """
    global _baseline_ridge_values, _baseline_roughness_values
    _baseline_ridge_values.clear()
    _baseline_roughness_values.clear()
    _ridge_history.clear()
    _roughness_history.clear()


def get_baseline_stats() -> Dict:
    """Get current baseline statistics (for debugging/monitoring).
    
    Returns:
        Dictionary with baseline means, stds, and sample counts
    """
    ridge_stats = {
        'count': len(_baseline_ridge_values),
        'mean': float(np.mean(_baseline_ridge_values)) if _baseline_ridge_values else 0.0,
        'std': float(np.std(_baseline_ridge_values)) if _baseline_ridge_values else 0.0
    }
    
    roughness_stats = {
        'count': len(_baseline_roughness_values),
        'mean': float(np.mean(_baseline_roughness_values)) if _baseline_roughness_values else 0.0,
        'std': float(np.std(_baseline_roughness_values)) if _baseline_roughness_values else 0.0
    }
    
    return {
        'ridge_baseline': ridge_stats,
        'roughness_baseline': roughness_stats
    }


# Utility function for visualization (simplified)
def visualize_texture_analysis(
    image: np.ndarray,
    mask: np.ndarray,
    results: Dict
) -> np.ndarray:
    """Create simple visualization of texture screening results.
    
    Args:
        image: Original BGR image
        mask: Binary nail mask
        results: Texture analysis results dictionary
    
    Returns:
        Annotated image with screening result overlay
    """
    vis = image.copy()
    
    # Find nail region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return vis
    
    # Draw contour
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    
    # Add result text
    result = results.get('screening_result', 'unknown')
    quality = results.get('image_quality_score', 0)
    
    # Color code by result
    if result == 'normal':
        color = (0, 255, 0)  # Green
        text = "Normal"
    elif result == 'insufficient_quality':
        color = (128, 128, 128)  # Gray
        text = "Low Quality"
    else:
        color = (0, 165, 255)  # Orange
        text = "Elevated"
    
    # Add text overlay
    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.putText(vis, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(vis, f"Q: {quality:.2f}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis
