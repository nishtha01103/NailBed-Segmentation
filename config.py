"""
Configuration parameters for nail segmentation and analysis pipeline.
"""

# Model configuration
MODEL_PATH = "models/best.pt"

# Mask processing
MASK_THRESHOLD = 0.5
MIN_MASK_PIXELS = 1000

# Color extraction (LAB space)
MIN_PIXELS_FOR_COLOR = 100  # Require more pixels for reliable color
MIN_PIXELS_AFTER_FILTERING = 80  # More strict after filtering
LAB_L_MIN = 40  # Remove very dark pixels (shadows)
LAB_L_MAX = 230  # Remove extreme highlights

  # More aggressive outlier removal (was 10)
# Color extraction robustness
LAB_A_OUTLIER_PERCENTILE = 15  # More aggressive outlier removal (was 10)
LAB_B_OUTLIER_PERCENTILE = 15
# White balance normalization (reduces lighting bias for medical analysis)
NORMALIZE_WHITE_BALANCE = False  # Disable white balance - can overcorrect and desaturate nail colors
WHITE_BALANCE_METHOD = "gray_world"  # Method: 'gray_world' or 'none'

# Single-Scale Retinex illumination normalization (L channel only)
# Retinex improves illumination invariance for nail color diagnostics.
# Must process 720p frame in <10ms — uses GaussianBlur on L channel only.
ENABLE_RETINEX = True          # Enable SSR illumination normalization
RETINEX_SIGMA = 30              # Gaussian blur radius for SSR surround function

# Geometry measurement (rotation-invariant)
MIN_CONTOUR_AREA = 800  # Pixels² (stricter to filter noise)
MIN_NAIL_ASPECT_RATIO = 0.40  # Minimum length/width ratio (allows very wide thumbs)
MAX_NAIL_ASPECT_RATIO = 3.0  # Maximum length/width ratio

# Real-world nail size constraints (to reject non-nails like lips, skin)
MIN_NAIL_LENGTH_MM = 3.0    # Reject detections smaller than 3mm (noise)
MAX_NAIL_LENGTH_MM = 30.0   # Reject detections larger than 30mm (finger-sized = not a nail)
MIN_NAIL_AREA_MM2 = 15.0    # Minimum nail plate area in mm²
MAX_NAIL_WIDTH_MM = 22.0    # Maximum nail width — rejects wide skin patches

# Color validation for nails (to reject lips, skin, etc.)
NAIL_LAB_L_MIN = 130        # Nails are typically darker/more colorful
NAIL_LAB_L_MAX = 210        # Don't detect very pale regions as nails
NAIL_LAB_A_MIN = 100        # Nails have moderate redness
NAIL_LAB_A_MAX = 145        # Avoid very red (lips) or very green regions
NAIL_LAB_B_MIN = 105        # Nails have warmth/yellowness
NAIL_LAB_B_MAX = 145        # Avoid very blue or very yellow

# Real-time detection settings
REALTIME_DETECTION_CONFIDENCE = 0.7  # Higher = more selective (0.5-0.9 range)
REALTIME_MASK_THRESHOLD = 0.6        # Stricter than batch processing
REALTIME_MIN_MASK_PIXELS = 2000      # Require larger masks in real-time
DEFAULT_PIXELS_PER_MM = 5.0  # ~200 DPI (adjust based on camera/image source)
USE_AUTOMATIC_CALIBRATION = False  # Auto-estimate from nail dimensions if True
TYPICAL_NAIL_LENGTH_MM = 15.0  # For automatic calibration estimation
INCLUDE_REAL_WORLD_MEASUREMENTS = True  # Add mm/cm values to output

# Geometry extraction settings
CLEAN_MASK_BEFORE_GEOMETRY = True  # Remove outliers/noise before measuring
MORPHOLOGY_KERNEL_SIZE = 5  # Kernel size for morphological operations (odd number, e.g., 3, 5, 7)

# Nail bed extraction (adaptive LAB gradient method)
NAIL_BED_NUM_SLICES = 60       # Number of projection slices along major axis for boundary detection
NAIL_BED_DELTA_L_SCALE = 1.5   # Multiplier on std(delta_L) to set adaptive L-jump threshold
NAIL_BED_VISUALIZE_BOUNDARY = False  # Draw detected boundary line on image for debugging

# Free-edge detection thresholds (used by adaptive nail bed boundary classifier)
FREE_EDGE_L_THRESHOLD = 5    # Minimum delta_L (distal - proximal) to confirm free edge present
FREE_EDGE_A_THRESHOLD = 3    # Minimum abs(delta_a) to confirm free edge present
FREE_EDGE_L_SMALL = 10        # Maximum abs(delta_L) to classify as no free edge (short nail)
FREE_EDGE_A_SMALL = 6       # Maximum abs(delta_a) to classify as no free edge (short nail)
FREE_EDGE_K_SIGMA = 1.0      # Sigma multiplier for adaptive boundary scan threshold (tighter with 4-method voting)

# Temporal smoothing for real-time
CENTROID_DISTANCE_THRESHOLD = 100  # Max pixel distance for nail matching across frames
ASPECT_RATIO_SIMILARITY_THRESHOLD = 0.3  # Max symmetric-ratio difference for matching

# Performance optimization settings
FRAME_SKIP_EXPENSIVE_OPS = 0  # Skip expensive ops every N frames (0=never skip, recommended for medical)
ENABLE_PERFORMANCE_LOGGING = False  # Log frame processing times for profiling

# Advanced texture analysis for medical screening
# WARNING: Texture analysis is sensitive to image quality and can produce false positives
# with poor lighting, blur, or low resolution images. Use conservative settings for screening.
ENABLE_TEXTURE_ANALYSIS = True  # Enable advanced ridge/roughness/pitting detection
TEXTURE_ANALYSIS_DETAIL_LEVEL = "full"  # Options: "basic", "standard", "full"
TEXTURE_SENSITIVITY = "conservative"  # Options: "conservative", "moderate", "sensitive"
                                       # conservative = fewer false positives (recommended)
                                       # moderate = balanced
                                       # sensitive = more detections (higher false positive rate)
TEXTURE_MIN_IMAGE_QUALITY = 0.5  # Minimum quality score (0-1) to run texture analysis
                                  # Lower = accept lower quality images (more false positives)
                                  # Higher = require better images (fewer false positives)

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
