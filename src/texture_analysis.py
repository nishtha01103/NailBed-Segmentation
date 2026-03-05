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
_ridge_history    = deque(maxlen=5)
_roughness_history = deque(maxlen=5)

# Per-nail baseline tracking — prevents statistical contamination.
# Per-nail baseline prevents statistical contamination.
# Key: nail_id (any hashable) or '_default' for backward-compatible calls.
# Value: {'ridge': List[float], 'roughness': List[float]}
_baselines: dict = {}


def analyze_nail_texture(
    image: np.ndarray,
    mask: np.ndarray,
    nail_orientation: Optional[float] = None,
    nail_id=None,
    use_structure_tensor: bool = False,
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
        nail_orientation: Nail major-axis angle in radians.  When provided,
            ridge detection projects Sobel gradients onto the minor axis for
            orientation-aware measurement.  Pass None to use the legacy
            vertical-Sobel fallback.
        nail_id: Optional hashable identifier for the nail being analysed.
            When provided, baseline statistics are tracked per nail to prevent
            cross-subject contamination.  Defaults to a shared ``'_default'``
            bucket for backward-compatible callers that don't pass an ID.
        use_structure_tensor: When True, uses the structure tensor coherence
            metric instead of the gradient projection metric for ridge
            strength.  Requires more computation but measures directional
            texture consistency more robustly.  Default False.
    
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
    
    # ── Specular glare suppression ────────────────────────────────────────────
    # Suppressing specular highlights improves texture reliability.
    # Extreme specular pixels (near-white, achromatic) falsely inflate both
    # gradient and variance metrics.  We identify them in LAB space and zero
    # them out in a local copy of the mask so the original is never mutated.
    lab_crop = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2LAB)
    L_ch  = lab_crop[:, :, 0].astype(np.int16)   # 0..255 in OpenCV uint8 LAB
    a_ch  = lab_crop[:, :, 1].astype(np.int16)
    b_ch  = lab_crop[:, :, 2].astype(np.int16)
    glare_mask = (
        (L_ch > 240) &
        (np.abs(a_ch - 128) < 10) &
        (np.abs(b_ch - 128) < 10)
    )
    # Local copy — never modifies the caller's mask array
    mask_texture = mask_crop.copy()
    mask_texture[glare_mask] = 0

    # 2. RIDGE STRENGTH MEASUREMENT (orientation-aware Sobel gradient)
    ridge_strength = _measure_ridge_strength(
        nail_crop_normalized, mask_texture, nail_orientation,
        use_structure_tensor=use_structure_tensor,
    )

    # 3. SURFACE ROUGHNESS MEASUREMENT (local variance)
    surface_roughness = _measure_surface_roughness(nail_crop_normalized, mask_texture)
    
    # 4. TEMPORAL SMOOTHING (average over recent frames)
    ridge_strength_smoothed = _apply_temporal_smoothing(ridge_strength, _ridge_history)
    roughness_smoothed = _apply_temporal_smoothing(surface_roughness, _roughness_history)
    
    # ── Per-nail baseline lookup (create slot on first visit) ────────────────
    # Per-nail baseline prevents statistical contamination.
    _nail_key = nail_id if nail_id is not None else '_default'
    if _nail_key not in _baselines:
        _baselines[_nail_key] = {'ridge': [], 'roughness': []}
    _nail_baseline = _baselines[_nail_key]

    # 5. STATISTICAL NORMALIZATION (z-score based detection)
    ridge_deviation    = _calculate_deviation(ridge_strength_smoothed,  _nail_baseline['ridge'])
    roughness_deviation = _calculate_deviation(roughness_smoothed,       _nail_baseline['roughness'])

    # Update per-nail baselines (maintain rolling window of normal values)
    _update_baseline(_nail_baseline['ridge'],    ridge_strength_smoothed, max_size=30)
    _update_baseline(_nail_baseline['roughness'], roughness_smoothed,     max_size=30)
    
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
    
    # 3. Contrast (IQR-based — robust to specular highlights)
    # IQR-based contrast metric is robust to specular highlights.
    pixels = nail_crop[valid_pixels]
    q25, q75   = float(np.percentile(pixels, 25)), float(np.percentile(pixels, 75))
    iqr        = q75 - q25
    contrast_score = min(iqr / 60.0, 1.0)

    # Overall quality (weighted)
    quality_score = (
        0.5 * sharpness_score +    # Blur is most important
        0.3 * resolution_score +    # Size matters
        0.2 * contrast_score        # Contrast helps but less critical
    )

    details = {
        'sharpness': round(laplacian_variance, 1),
        'resolution_pixels': int(pixel_count),
        'contrast_iqr': round(iqr, 1),
        'sharpness_score': round(sharpness_score, 2),
        'resolution_score': round(resolution_score, 2),
        'contrast_score': round(contrast_score, 2)
    }
    
    return quality_score, details


def _measure_ridge_coherence(
    nail_crop: np.ndarray,
    mask_crop: np.ndarray,
    smooth_sigma: float = 2.0,
) -> float:
    """Compute mean structure tensor coherence over valid mask pixels.

    Structure tensor provides advanced ridge orientation coherence metric.

    Coherence ranges from 0 (isotropic / no preferred direction) to 1
    (perfectly unidirectional ridges).  It is invariant to absolute
    gradient magnitude and therefore robust to lighting changes.

    Args:
        nail_crop:    Grayscale nail region (float or uint8).
        mask_crop:    Binary mask (>0 = valid).
        smooth_sigma: Sigma for Gaussian smoothing of tensor components.

    Returns:
        Mean coherence in [0, 1] over valid pixels.
    """
    nail_f = nail_crop.astype(np.float32)
    Ix = cv2.Sobel(nail_f, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(nail_f, cv2.CV_32F, 0, 1, ksize=3)

    # Structure tensor components
    Jxx = Ix * Ix
    Jyy = Iy * Iy
    Jxy = Ix * Iy

    # Gaussian smoothing of each component (neighbourhood integration)
    ksize = 0  # let OpenCV derive kernel size from sigma
    Jxx_s = cv2.GaussianBlur(Jxx, (ksize, ksize), smooth_sigma)
    Jyy_s = cv2.GaussianBlur(Jyy, (ksize, ksize), smooth_sigma)
    Jxy_s = cv2.GaussianBlur(Jxy, (ksize, ksize), smooth_sigma)

    # Analytic eigenvalues of a 2×2 symmetric matrix:
    #   λ1,2 = 0.5 * ( (Jxx+Jyy) ± sqrt((Jxx-Jyy)^2 + 4*Jxy^2) )
    trace = Jxx_s + Jyy_s
    diff  = Jxx_s - Jyy_s
    disc  = np.sqrt(np.maximum(diff * diff + 4.0 * Jxy_s * Jxy_s, 0.0))
    lambda1 = 0.5 * (trace + disc)   # larger eigenvalue
    lambda2 = 0.5 * (trace - disc)   # smaller eigenvalue

    # Coherence = (λ1 − λ2) / (λ1 + λ2 + ε)  ∈ [0, 1]
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)

    valid = mask_crop > 0
    return float(np.mean(coherence[valid]))


def _measure_ridge_strength(
    nail_crop: np.ndarray,
    mask_crop: np.ndarray,
    nail_orientation: Optional[float] = None,
    use_structure_tensor: bool = False,
) -> float:
    """Measure longitudinal ridge prominence using orientation-aware Sobel gradients.

    Orientation-aware ridge detection improves robustness to rotated nails.

    When ``use_structure_tensor`` is True, delegates to
    ``_measure_ridge_coherence`` (structure tensor eigenvalue coherence),
    which measures directional texture consistency rather than raw gradient
    magnitude.  Structure tensor provides advanced ridge orientation
    coherence metric.

    When ``use_structure_tensor`` is False (default), uses the intensity-
    normalised Sobel projection method — faster and sufficient for most cases.

    When ``nail_orientation`` (radians) is provided (and
    ``use_structure_tensor`` is False), gradients are projected onto the
    minor anatomical axis.  When None, falls back to Sobel_y.

    Args:
        nail_crop:            Grayscale nail region (CLAHE normalized)
        mask_crop:            Binary mask
        nail_orientation:     Nail major-axis angle in radians, or None.
        use_structure_tensor: Use structure tensor coherence metric.

    Returns:
        Ridge strength as a float (normalised gradient or coherence score).
    """
    valid_pixels = mask_crop > 0

    # Structure tensor path: coherence metric (flag-gated, not default)
    if use_structure_tensor:
        return _measure_ridge_coherence(nail_crop, mask_crop)

    if nail_orientation is not None:
        # Orientation-aware: project gradient onto the minor (cross-ridge) axis.
        # major_axis = [cos(θ), sin(θ)]  →  along the nail length
        # minor_axis = [-sin(θ), cos(θ)] →  across the nail width (ridge direction)
        cos_t = np.cos(nail_orientation)
        sin_t = np.sin(nail_orientation)
        minor_axis_x = -sin_t   # x-component of minor axis
        minor_axis_y =  cos_t   # y-component of minor axis

        grad_x = cv2.Sobel(nail_crop, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(nail_crop, cv2.CV_64F, 0, 1, ksize=3)

        # Project gradient vector onto minor axis (fully vectorized)
        ridge_response = np.abs(
            grad_x * minor_axis_x + grad_y * minor_axis_y
        )
    else:
        # Fallback: Sobel vertical gradient (assumes vertically oriented nail)
        ridge_response = np.abs(cv2.Sobel(nail_crop, cv2.CV_64F, 0, 1, ksize=3))

    # Intensity-normalized ridge strength improves cross-lighting stability.
    # Dividing mean gradient by mean pixel intensity removes the dependency on
    # overall exposure level and CLAHE output magnitude, making the metric
    # comparable across frames with different lighting conditions.
    mean_gradient  = float(np.mean(ridge_response[valid_pixels]))
    mean_intensity = float(np.mean(nail_crop[valid_pixels])) + 1e-6
    ridge_strength = mean_gradient / mean_intensity
    return ridge_strength


def _measure_surface_roughness(nail_crop: np.ndarray, mask_crop: np.ndarray) -> float:
    """Measure surface roughness using normalised local variance (coefficient of variation).

    Normalized roughness reduces illumination sensitivity.

    Dividing local variance by the squared local mean (coefficient of variation)
    removes the dependency on absolute pixel intensity, making the metric
    comparable across frames with different exposure levels.

    Args:
        nail_crop: Grayscale nail region (CLAHE normalized)
        mask_crop: Binary mask

    Returns:
        Surface roughness (mean normalised local variance over valid pixels)
    """
    valid_pixels = mask_crop > 0
    nail_float = nail_crop.astype(np.float32)

    # Local mean and variance (5x5 window)
    window_size = 5
    mean_filtered = cv2.blur(nail_float, (window_size, window_size))
    sq_filtered   = cv2.blur(nail_float ** 2, (window_size, window_size))
    variance = sq_filtered - mean_filtered ** 2
    variance = np.maximum(variance, 0)  # Handle numerical errors

    # Normalised variance = variance / (mean² + ε)  →  illumination invariant
    normalized_variance = variance / (mean_filtered ** 2 + 1e-6)

    roughness = float(np.mean(normalized_variance[valid_pixels]))
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


def reset_baselines(nail_id=None):
    """Reset baseline tracking.

    When ``nail_id`` is provided, clears only that nail's baseline so its
    history starts fresh without affecting other nails.  When called with no
    argument (or ``nail_id=None``), clears ALL per-nail baselines and the
    shared temporal smoothing buffers — equivalent to the previous behaviour.

    Args:
        nail_id: Optional nail identifier.  Pass the same value used in
            ``analyze_nail_texture`` to reset a specific nail's baseline.
    """
    global _baselines
    if nail_id is not None:
        _nail_key = nail_id
        if _nail_key in _baselines:
            _baselines[_nail_key]['ridge'].clear()
            _baselines[_nail_key]['roughness'].clear()
    else:
        # Full reset — backward-compatible behaviour
        _baselines.clear()
        _ridge_history.clear()
        _roughness_history.clear()


def get_baseline_stats(nail_id=None) -> Dict:
    """Get current baseline statistics (for debugging/monitoring).

    Args:
        nail_id: When provided, returns stats for that specific nail only.
            When None, returns stats for all tracked nails.

    Returns:
        Dictionary with baseline means, stds, and sample counts
    """
    def _stats(values: list) -> dict:
        return {
            'count': len(values),
            'mean':  float(np.mean(values)) if values else 0.0,
            'std':   float(np.std(values))  if values else 0.0,
        }

    if nail_id is not None:
        _nail_key = nail_id if nail_id is not None else '_default'
        bucket = _baselines.get(_nail_key, {'ridge': [], 'roughness': []})
        return {
            'nail_id':          _nail_key,
            'ridge_baseline':   _stats(bucket['ridge']),
            'roughness_baseline': _stats(bucket['roughness']),
        }

    # Return stats for every tracked nail
    return {
        nk: {
            'ridge_baseline':    _stats(bkt['ridge']),
            'roughness_baseline': _stats(bkt['roughness']),
        }
        for nk, bkt in _baselines.items()
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
