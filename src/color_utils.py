"""Color extraction utilities for nail analysis."""

from typing import Tuple, Optional, List
import cv2
import numpy as np
from config import (
    MIN_PIXELS_FOR_COLOR,
    MIN_PIXELS_AFTER_FILTERING,
    LAB_L_MIN,
    LAB_L_MAX,
    LAB_A_OUTLIER_PERCENTILE,
    LAB_B_OUTLIER_PERCENTILE,
    NORMALIZE_WHITE_BALANCE,
    ENABLE_TEXTURE_ANALYSIS,
    TEXTURE_ANALYSIS_DETAIL_LEVEL,
    TEXTURE_MIN_IMAGE_QUALITY,
)

# Import texture analysis (optional, graceful degradation)
try:
    from .texture_analysis import analyze_nail_texture
    TEXTURE_AVAILABLE = True
except ImportError:
    TEXTURE_AVAILABLE = False
    print("Warning: Advanced texture analysis not available. Install scipy for full functionality.")


def normalize_white_balance(image: np.ndarray) -> np.ndarray:
    """Normalize white balance using gray-world assumption.
    
    Reduces lighting bias by equalizing color channel means.
    Assumes that on average, the image should have equal R, G, B values.
    
    Implementation:
    - Calculate mean of each color channel (B, G, R)
    - Calculate overall gray mean
    - Scale each channel to match the gray mean
    - Clip values to valid range [0, 255]
    
    Args:
        image: BGR image array (uint8)
    
    Returns:
        White-balanced image (uint8)
    
    Reference:
        Inspired by gray-world color constancy assumption.
        Effective for normalizing different lighting conditions.
    """
    result = image.copy().astype(np.float32)

    # Calculate mean for each channel
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    # Calculate overall gray mean
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    # Avoid division by zero
    if avg_b > 0:
        result[:, :, 0] *= (avg_gray / avg_b)
    if avg_g > 0:
        result[:, :, 1] *= (avg_gray / avg_g)
    if avg_r > 0:
        result[:, :, 2] *= (avg_gray / avg_r)

    # Clip to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def _remove_color_outliers(pixels: np.ndarray) -> np.ndarray:
    """Remove chromaticity outliers from LAB pixels using percentile clipping.
    
    Args:
        pixels: Array of LAB pixels (N, 3)
    
    Returns:
        Filtered LAB pixel array
    """
    if len(pixels) == 0:
        return pixels
    
    # Work with a and b channels
    a_channel = pixels[:, 1]
    b_channel = pixels[:, 2]

    # Compute percentile bounds
    a_lower, a_upper = np.percentile(
        a_channel, [LAB_A_OUTLIER_PERCENTILE, 100 - LAB_A_OUTLIER_PERCENTILE]
    )
    b_lower, b_upper = np.percentile(
        b_channel, [LAB_B_OUTLIER_PERCENTILE, 100 - LAB_B_OUTLIER_PERCENTILE]
    )

    # Keep only pixels within bounds
    mask = (
        (a_channel >= a_lower)
        & (a_channel <= a_upper)
        & (b_channel >= b_lower)
        & (b_channel <= b_upper)
    )

    return pixels[mask]


def _extract_skin_reference(image: np.ndarray, mask: np.ndarray, dilation_pixels: int = 12) -> Optional[List[float]]:
    """Extract reference skin color from area surrounding nail.
    
    Creates a ring around the nail mask and samples the median LAB color
    from that region. This provides a per-person skin tone reference for
    relative color analysis, making the system adaptive across ethnicities.
    
    Args:
        image: BGR image
        mask: Binary nail mask
        dilation_pixels: How many pixels to dilate for skin ring (default: 12)
    
    Returns:
        Median LAB values of surrounding skin, or None if insufficient pixels
    """
    # Create skin ring mask by dilating and subtracting original
    kernel = np.ones((dilation_pixels, dilation_pixels), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    skin_ring_mask = dilated_mask - mask
    
    # Apply CLAHE to image for consistent skin sampling
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    lab = cv2.merge([L, A, B])
    
    # Extract skin pixels from ring area
    skin_pixels = lab[skin_ring_mask > 0]
    
    # Need sufficient pixels for reliable skin reference (minimum 50)
    if len(skin_pixels) < 50:
        return None
    
    # Use median for robustness against outliers
    skin_median_lab = np.median(skin_pixels, axis=0)
    
    return skin_median_lab.tolist()


def _detect_polished_nail(pixels_lab: np.ndarray) -> Tuple[bool, float]:
    """Detect if a nail is polished based on LAB saturation profile.

    Polished nails have HIGH chroma (a*² + b*² > threshold) and LOW L variance.
    Natural nails have moderate chroma and higher L variance.

    Returns:
        (is_polished, confidence): bool and float 0-1
    """
    if len(pixels_lab) < 20:
        return False, 0.0

    a_real = pixels_lab[:, 1].astype(np.float32) - 128.0
    b_real = pixels_lab[:, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a_real**2 + b_real**2)

    median_chroma = float(np.median(chroma))
    L_std         = float(np.std(pixels_lab[:, 0]))

    # High chroma (>20 LAB units) + low L std (<12) = polished
    is_polished = (median_chroma > 20.0) and (L_std < 12.0)
    # Confidence scales with how far above/below thresholds
    chroma_score = min(1.0, median_chroma / 35.0)
    l_score      = min(1.0, max(0.0, (12.0 - L_std) / 12.0))
    confidence   = 0.5 * chroma_score + 0.5 * l_score if is_polished else 0.0

    return is_polished, float(confidence)


def _check_lighting_quality(pixels_lab: np.ndarray) -> Tuple[bool, str]:
    """Check if lighting conditions are adequate for reliable color analysis.

    Detects:
    - Overexposure: median L > 220 (blown highlights)
    - Underexposure: median L < 60 (too dark)
    - High glare: > 15% pixels at L > 230
    - Mixed/uneven lighting: L std > 35

    Returns:
        (is_adequate, reason_if_not)
    """
    if len(pixels_lab) == 0:
        return False, "no_pixels"

    L = pixels_lab[:, 0].astype(np.float32)
    median_L = float(np.median(L))
    std_L    = float(np.std(L))
    glare_fraction = float(np.mean(L > 230))

    if median_L > 220:
        return False, "overexposed"
    if median_L < 60:
        return False, "underexposed"
    if glare_fraction > 0.15:
        return False, "glare"
    if std_L > 38:
        return False, "uneven_lighting"

    return True, "ok"


def _otsu_filter_nail_pixels(
    pixels_lab: np.ndarray,
    pixels_gray: np.ndarray,
) -> np.ndarray:
    """Use Otsu thresholding to remove specular highlights from color sampling.

    Specular highlights (bright spots from camera flash or window light)
    appear as white/near-white pixels that inflate the L channel and shift
    the median color toward white, making the nail appear paler than it is.

    Also removes deep shadows (sub-threshold dark pixels) that can shift
    color toward dark/gray and confuse the screening flags.

    Two-pass Otsu:
    Pass 1 — upper threshold: removes specular highlights (top brightness cluster)
    Pass 2 — lower threshold: removes deep shadows (bottom brightness cluster)

    Args:
        pixels_lab:  LAB pixel array (N, 3) — already filtered by mask
        pixels_gray: Grayscale values for same pixels (N,) — same order as pixels_lab

    Returns:
        Filtered LAB pixel array with highlights and shadows removed
    """
    if len(pixels_gray) < 30:
        return pixels_lab

    gray_norm = pixels_gray.astype(np.uint8)

    # Pass 1: Upper Otsu — separate highlights from rest
    thresh_high, _ = cv2.threshold(
        gray_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Only apply if highlights are a small fraction (< 30%) — otherwise
    # Otsu is splitting within the nail bed, not removing highlights
    highlight_fraction = float(np.mean(gray_norm > thresh_high))
    if highlight_fraction < 0.30:
        keep_upper = gray_norm <= thresh_high
    else:
        keep_upper = np.ones(len(gray_norm), dtype=bool)

    # Pass 2: Lower Otsu — separate shadows from rest
    # Invert so shadows become the high cluster for Otsu
    gray_inverted = (255 - gray_norm).astype(np.uint8)
    thresh_low, _ = cv2.threshold(
        gray_inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    shadow_fraction = float(np.mean(gray_inverted > thresh_low))
    if shadow_fraction < 0.25:
        keep_lower = gray_inverted <= thresh_low
    else:
        keep_lower = np.ones(len(gray_norm), dtype=bool)

    keep = keep_upper & keep_lower
    if keep.sum() < 20:
        return pixels_lab  # Too aggressive — return unfiltered

    return pixels_lab[keep]


def extract_nail_color(
    image: np.ndarray, mask: np.ndarray, enable_texture: bool = None
) -> Tuple[Optional[List[float]], Optional[List[int]], Optional[dict], Optional[str]]:
    """Extract natural nail plate color with skin-adaptive medical screening.
    
    Applies CLAHE for lighting stability, extracts LAB color metrics,
    and performs relative color analysis compared to surrounding skin tone.
    Optionally includes advanced texture analysis for ridge/pitting detection.
    This makes the system work robustly across all skin tones and ethnicities.
    
    Args:
        image: BGR image
        mask: Binary mask of nail region
        enable_texture: Override config for texture analysis (None = use config)
    
    Returns:
        Tuple of (median_lab, bgr_color, color_analysis_dict, hex_color)
    """

    # Optional white balance normalization (apply once per frame ideally)
    # NOTE: Currently disabled - gray-world normalization overcorrects static images
    # and shifts nail colors towards greenish-blue, making them appear grayish.
    # For static images with fixed lighting, white balance is not needed.
    if NORMALIZE_WHITE_BALANCE:
        image = normalize_white_balance(image)

    # Extract skin reference BEFORE erosion (need full mask for skin ring)
    # This provides per-person baseline for relative color analysis
    skin_reference_lab = _extract_skin_reference(image, mask)

    # Reduced erosion to prevent over-shrinking small nails
    # Use only central region of nail mask to avoid skin contamination
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.erode(mask, kernel, iterations=1)

    # Apply CLAHE to full image FIRST (correct tile statistics),
    # then mask to nail region
    lab_full = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L_full, A_full, B_full = cv2.split(lab_full)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L_full)
    lab_eq = cv2.merge([L_eq, A_full, B_full])

    pixels = lab_eq[refined_mask == 255]

    # Extract grayscale values for the same pixels (same order as pixels)
    _gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _pixels_gray = _gray_full[refined_mask == 255]

    # Apply Otsu-based highlight and shadow rejection BEFORE L range filter
    # This removes specular spots that would inflate L and whiten the color
    pixels = _otsu_filter_nail_pixels(pixels, _pixels_gray)
    # Note: after _otsu_filter_nail_pixels, pixels is already filtered

    if len(pixels) < MIN_PIXELS_FOR_COLOR:
        return None, None, None, None

    # Remove extreme brightness (deep shadows & specular highlights)
    L_channel = pixels[:, 0]
    pixels = pixels[(L_channel > LAB_L_MIN) & (L_channel < LAB_L_MAX)]

    if len(pixels) < MIN_PIXELS_AFTER_FILTERING:
        return None, None, None, None

    # Remove chromatic outliers (skin contamination, reflections)
    pixels = _remove_color_outliers(pixels)

    if len(pixels) < MIN_PIXELS_AFTER_FILTERING:
        return None, None, None, None

    # Detect polish — affects downstream interpretation
    _is_polished, _polish_conf = _detect_polished_nail(pixels)

    # Check lighting quality — flags low confidence if poor
    _lighting_ok, _lighting_reason = _check_lighting_quality(pixels)
    if not _lighting_ok:
        # Still compute color but flag low confidence
        # Do not return None — partial data is more useful than no data
        pass  # will be added to color_analysis dict below

    # Restrict color sampling to central 60% of nail (proximal 20% and distal 20% excluded)
    # This avoids cuticle and free-edge contamination of the color reading
    # We need the mask to compute the PCA major axis extent for this
    nail_yx_color = np.argwhere(refined_mask == 255)
    if len(nail_yx_color) > 20:
        nail_xy_color = nail_yx_color[:, ::-1].astype(np.float64)
        centroid_color = nail_xy_color.mean(axis=0)
        cov_color = np.cov((nail_xy_color - centroid_color).T)
        _, evecs = np.linalg.eigh(cov_color)
        major_color = evecs[:, -1]
        proj_color = (nail_xy_color - centroid_color) @ major_color
        p20 = float(np.percentile(proj_color, 20))
        p80 = float(np.percentile(proj_color, 80))
        # Map pixel indices: pixels array rows correspond to nail_yx_color rows
        proj_all = proj_color  # same order as nail_yx_color
        central_mask_color = (proj_all >= p20) & (proj_all <= p80)
        if central_mask_color.sum() >= MIN_PIXELS_AFTER_FILTERING:
            # Rebuild pixels from central region only
            central_yx = nail_yx_color[central_mask_color]
            pixels = lab_eq[central_yx[:, 0], central_yx[:, 1]]
            # Re-apply existing L range and outlier filters
            L_ch = pixels[:, 0]
            pixels = pixels[(L_ch > LAB_L_MIN) & (L_ch < LAB_L_MAX)]
            if len(pixels) >= MIN_PIXELS_AFTER_FILTERING:
                pixels = _remove_color_outliers(pixels)

    # Use median for robustness against remaining outliers
    median_lab = np.median(pixels, axis=0)

    # Convert for display
    lab_uint8 = np.uint8([[median_lab]])
    bgr_color = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)[0][0]
    
    # Compute hex color for visual reference
    hex_color = '#%02x%02x%02x' % (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))

    # Perform skin-adaptive medical color analysis
    color_analysis = analyze_lab_distribution(
        pixels, 
        skin_reference_lab=skin_reference_lab,
        image=image,
        mask=mask,
        enable_texture=enable_texture,
        is_polished_detected=_is_polished,
        lighting_quality=_lighting_reason if not _lighting_ok else "ok",
    )

    return median_lab.tolist(), bgr_color.tolist(), color_analysis, hex_color




def analyze_lab_distribution(
    pixels: np.ndarray,
    reference_lab: Optional[Tuple[float, float, float]] = None,
    skin_reference_lab: Optional[List[float]] = None,
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    enable_texture: bool = None,
    is_polished_detected: bool = False,
    lighting_quality: str = "ok",
) -> dict:
    """Analyze LAB color distribution with skin-adaptive medical screening.
    
    MEDICAL APPROACH:
    Relative comparison (nail vs skin) is the gold standard for medical screening.
    Clinicians naturally compare nail color to surrounding skin tone to detect:
    - Anemia (pallor): Nail paler than expected
    - Cyanosis: Bluish discoloration (hypoxia)
    - Jaundice: Yellowish discoloration (liver/bile)
    - Erythema: Redness (inflammation/infection)
    
    Optionally includes advanced texture analysis:
    - Ridge detection (longitudinal/horizontal)
    - Surface roughness (fungal infection, eczema)
    - Pitting (psoriasis, alopecia)
    - Edge quality (trauma, splitting)
    
    This approach works equitably across all ethnicities and skin tones,
    unlike absolute thresholds which bias towards light skin.
    
    Args:
        pixels: Array of LAB pixels (N, 3) after filtering
        reference_lab: Optional reference LAB values for healthy nail (absolute)
                      Default: (180, 138, 140) - typical healthy nail in OpenCV LAB
        skin_reference_lab: Surrounding skin LAB values for relative comparison
        image: Original BGR image (required for texture analysis)
        mask: Binary nail mask (required for texture analysis)
        enable_texture: Override config for texture analysis (None = use config)
    
    Returns:
        Dictionary with color analysis, adaptive screening, texture, and confidence score
    """
    if len(pixels) == 0:
        return None
    

    if reference_lab is None:
        # Reference in OpenCV LAB scale (healthy nail approx)
        # OpenCV a and b are centered at 128
        reference_lab = (180.0, 138.0, 140.0)

    L_pixels = pixels[:, 0]
    a_pixels = pixels[:, 1]
    b_pixels = pixels[:, 2]

    median_L = float(np.median(L_pixels))
    median_a = float(np.median(a_pixels))
    median_b = float(np.median(b_pixels))

    std_L = float(np.std(L_pixels))
    std_a = float(np.std(a_pixels))
    std_b = float(np.std(b_pixels))

    ref_L, ref_a, ref_b = reference_lab

    dev_L = float(median_L - ref_L)

    # Convert to real LAB deviation (center at 128)
    real_a = median_a - 128
    real_b = median_b - 128
    ref_real_a = ref_a - 128
    ref_real_b = ref_b - 128

    dev_a = float(real_a - ref_real_a)
    dev_b = float(real_b - ref_real_b)

    chroma = float(np.sqrt(real_a**2 + real_b**2))
    
    # Texture metric (higher = uneven surface or lighting variation)
    texture_std = float(np.std(L_pixels))
    
    # Skin-adaptive relative metrics and screening
    relative_metrics = None
    screening_flags = []
    confidence_score = 0.0  # 0-100 scale
    
    if skin_reference_lab is not None:
        # ===== RELATIVE COMPARISON (HEALTH INDICATOR SCREENING) =====
        # Compute relative color differences (nail vs surrounding skin)
        skin_L, skin_a, skin_b = skin_reference_lab
        skin_real_a = skin_a - 128
        skin_real_b = skin_b - 128
        
        delta_L = float(median_L - skin_L)
        delta_a = float(real_a - skin_real_a)
        delta_b = float(real_b - skin_real_b)
        
        relative_metrics = {
            "delta_L": round(delta_L, 1),  # Nail lightness vs skin
            "delta_a": round(delta_a, 1),  # Nail redness vs skin
            "delta_b": round(delta_b, 1),  # Nail yellowness vs skin
            "skin_reference_L": round(skin_L, 1),  # Skin tone indicator
        }
        
        # High confidence - we have skin reference for relative comparison
        confidence_score = 85.0
        
        # CONSERVATIVE THRESHOLDS (2-std approach for statistical significance)
        # These thresholds are set to reduce false positives while catching significant deviations
        # Based on typical nail-skin color differences across populations
        
        # Adaptive thresholds based on skin tone — 4-tier instead of 3
        # Empirically calibrated to reduce false positives across all skin tones
        if skin_L < 85:
            # Very dark skin (Fitzpatrick V-VI)
            pale_threshold   = 20   # nails naturally lighter than skin
            yellow_threshold = 25   # more yellow acceptable
            blue_threshold   = -18
            redness_threshold = 12  # nails naturally redder vs very dark skin
        elif skin_L < 110:
            # Dark skin (Fitzpatrick IV-V)
            pale_threshold   = 25
            yellow_threshold = 20
            blue_threshold   = -15
            redness_threshold = 10
        elif skin_L < 140:
            # Medium skin (Fitzpatrick III-IV)
            pale_threshold   = 28
            yellow_threshold = 17
            blue_threshold   = -13
            redness_threshold = 8
        else:
            # Light skin (Fitzpatrick I-II)
            pale_threshold   = 32
            yellow_threshold = 14
            blue_threshold   = -11
            redness_threshold = 6
        
        # ===== CONSERVATIVE HEALTH INDICATOR SCREENING =====
        # Only flag statistically significant deviations (2+ std from normal ranges)
        # No disease diagnosis - only general health indicators
        
        # 1. PALE APPEARANCE (unusually light compared to skin)
        if delta_L > pale_threshold:
            screening_flags.append({
                "condition": "Unusually pale appearance",
                "severity": "mild",
                "note": "Nail significantly lighter than surrounding skin",
                "requires_persistence": True  # Should persist multiple frames
            })
        
        # 2. YELLOWISH TONE (elevated yellow compared to skin)
        if delta_b > yellow_threshold:
            screening_flags.append({
                "condition": "Yellowish discoloration",
                "severity": "mild",
                "note": "Nail yellower than typical range",
                "requires_persistence": True
            })
        
        # 3. BLUISH TONE (blue tint compared to skin)
        if delta_b < blue_threshold:
            screening_flags.append({
                "condition": "Bluish discoloration",
                "severity": "mild",
                "note": "Nail showing blue tint",
                "requires_persistence": True
            })
        
        # 4. REDNESS / ERYTHEMA
        if delta_a > redness_threshold:
            screening_flags.append({
                "condition": "Redness / erythema",
                "severity": "mild",
                "note": "Nail redder than surrounding skin",
                "requires_persistence": True
            })
        
        # 5. High texture variation removed - handled by texture_analysis module
        
        # Add explicit normal color flag if no color abnormalities found
        color_abnormalities_found = any(
            'pale' in flag.get('condition', '').lower() or
            'yellow' in flag.get('condition', '').lower() or
            'blue' in flag.get('condition', '').lower() or
            'redness' in flag.get('condition', '').lower()
            for flag in screening_flags
        )
        if not color_abnormalities_found:
            screening_flags.append({
                "condition": "Color: Normal",
                "severity": "none",
                "note": "Within expected range"
            })
        
    else:
        # ===== FALLBACK: ABSOLUTE THRESHOLDS =====
        # Less reliable - used only when skin reference unavailable
        confidence_score = 40.0  # Lower confidence without skin reference
        
        # Absolute yellow detection (very conservative)
        if real_b > 25:
            screening_flags.append({
                "condition": "Yellowish tone detected",
                "severity": "uncertain",
                "note": "No skin reference available",
                "requires_persistence": True
            })
        
        # Absolute blue detection (very conservative)
        if real_b < -15:
            screening_flags.append({
                "condition": "Bluish tone detected",
                "severity": "uncertain",
                "note": "No skin reference available",
                "requires_persistence": True
            })
        
        
    # Normal range if no concerning indicators
    if not screening_flags:
        screening_flags.append({"condition": "Color: Normal", "severity": "none", "note": "Within expected range"})
        confidence_score = min(confidence_score + 15.0, 100.0)  # Boost for normal finding
    
    # Determine screening summary
    has_concerns = any(flag.get('severity') not in ['none', 'uncertain'] for flag in screening_flags)
    screening_summary = "recommend_health_check" if has_concerns else "normal"

    result = {
        "median_lab": [
            round(median_L, 1),
            round(median_a, 1),
            round(median_b, 1),
        ],
        "std_lab": [
            round(std_L, 1),
            round(std_a, 1),
            round(std_b, 1),
        ],
        # Deviation from reference normal (absolute metrics - for reference only)
        "deviation_from_normal": {
            "L_deviation": round(dev_L, 1),  # >0 = lighter, <0 = darker
            "a_deviation": round(dev_a, 1),  # >0 = redder, <0 = greener
            "b_deviation": round(dev_b, 1),  # >0 = more yellow, <0 = more blue
        },
        # Color saturation (based on corrected real a/b)
        "chroma": round(chroma, 1),
        
        # Texture metric (higher = uneven surface/lighting)
        "texture_std": round(texture_std, 1),
        
        # Medical screening with severity levels (PRIMARY OUTPUT)
        "screening_flags": screening_flags,
        
        # Confidence score (0-100) based on available data
        "screening_confidence": round(confidence_score, 1),
        
        # Data quality indicator
        "has_skin_reference": skin_reference_lab is not None,
        
        # Overall screening summary
        "screening_summary": screening_summary,
        
        # Polish detection
        "is_polished_detected": is_polished_detected,
        
        # Lighting quality
        "lighting_quality": lighting_quality,
    }
    
    # Polish override: if polish detected, color screening is unreliable
    if is_polished_detected:
        result["screening_flags"] = [{
            "condition": "Polish detected \u2014 color screening unreliable",
            "severity": "none",
            "note": "Remove polish for accurate health screening"
        }]
        result["screening_confidence"] = 15.0
        result["screening_summary"] = "insufficient_data"
    
    # Lighting quality penalty
    if lighting_quality != "ok":
        result["screening_confidence"] = round(
            result["screening_confidence"] * 0.5, 1
        )

    # Compute color cluster purity using coefficient of variation
    # Low CV = tight cluster = pure region = high quality sample
    L_cv = float(std_L / max(median_L, 1.0))  # coefficient of variation
    color_purity = max(0.0, min(1.0, 1.0 - (L_cv * 3.0)))

    result["color_sample_purity"] = round(color_purity, 2)
    # Adjust confidence based on purity
    if color_purity < 0.5:
        result["screening_confidence"] = round(
            result["screening_confidence"] * 0.75, 1
        )
        result["screening_flags"].append({
            "condition": "Low color sample purity",
            "severity": "none",
            "note": "Uneven lighting or mixed regions affecting color reading"
        })
    
    # Add relative metrics if skin reference was available (PRIMARY for medical)
    if relative_metrics is not None:
        result["relative_metrics"] = relative_metrics
    
    # ===== SIMPLIFIED TEXTURE SCREENING (OPTIONAL) =====
    # Stable ridge/roughness screening using statistical normalization
    if enable_texture is None:
        enable_texture = ENABLE_TEXTURE_ANALYSIS
    
    if enable_texture and TEXTURE_AVAILABLE and image is not None and mask is not None:
        try:
            texture_results = analyze_nail_texture(image, mask)
            
            if texture_results:
                screening_result = texture_results.get('screening_result', 'unknown')
                
                # Check if screening was successful or skipped
                if screening_result == 'insufficient_quality':
                    result["texture_screening"] = {
                        "status": "skipped",
                        "reason": texture_results.get('message', 'Quality check failed'),
                        "recommendation": texture_results.get('recommendation', 'Improve image quality'),
                        "quality_score": texture_results.get('image_quality_score', 0)
                    }
                else:
                    # Add simplified texture screening results
                    result["texture_screening"] = {
                        "status": "completed",
                        "screening_result": screening_result,
                        "message": texture_results.get('message', ''),
                        "quality_score": texture_results.get('image_quality_score', 0),
                        "ridge_strength": texture_results.get('ridge_strength', 0),
                        "surface_roughness": texture_results.get('surface_roughness', 0),
                        "ridge_z_score": texture_results.get('ridge_deviation_zscore', 0),
                        "roughness_z_score": texture_results.get('roughness_deviation_zscore', 0),
                        "disclaimer": texture_results.get('disclaimer', 'Screening tool only, not diagnostic')
                    }
                    
                    # Add simplified screening flag if abnormal
                    if screening_result not in ['normal', 'insufficient_quality']:
                        # Remove generic normal appearance
                        screening_flags[:] = [
                            f for f in screening_flags 
                            if 'normal' not in f.get('condition', '').lower()
                        ]
                        
                        # Add texture screening flag
                        severity_map = {
                            'elevated_ridges': 'mild',
                            'rough_surface': 'mild',
                            'elevated_ridges_and_roughness': 'moderate'
                        }
                        
                        result["screening_flags"].append({
                            'condition': 'Texture: ' + screening_result.replace('_', ' ').title(),
                            'severity': severity_map.get(screening_result, 'mild'),
                            'note': texture_results.get('message', ''),
                            'requires_persistence': True  # Should persist 3+ frames
                        })
                        
                        # Update screening summary if texture abnormality found
                        result["screening_summary"] = "recommend_health_check"
                
        except Exception as e:
            # Graceful degradation - don't fail if texture analysis has issues
            print(f"Warning: Texture screening failed: {e}")
            result["texture_screening"] = {"status": "error", "message": str(e)}
    
    return result
