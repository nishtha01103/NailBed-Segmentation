"""src.geometry — modular nail-geometry and free-edge detection package.

Re-exports every symbol that was previously importable from
``src.geometry_utils`` so that existing consumer code continues to work
unchanged via ``from src.geometry import <name>``.
"""

# --- Mask cleaning ---
from .mask_processing import _clean_mask_morphology

# --- PCA utilities & basic geometry extraction ---
from .pca_utils import (
    _pca_on_mask_pixels,
    extract_geometry_rotation_invariant,
    extract_geometry,
    extract_geometry_with_diagnostics,
)

# --- Anatomical axis orientation ---
from .axis_orientation import _orient_anatomical_axis

# --- Adaptive boundary detection (LAB gradient) ---
from .boundary_detection import (
    _smooth_profile,
)

# --- Alternative boundary estimators ---
from .boundary_estimators import (
    _kmeans_boundary_estimate,
    _otsu_boundary_estimate,
    _canny_boundary_estimate,
)

# --- Boundary validation / cross-checking ---
from .boundary_validation import (
    _otsu_validate_color_boundary,
    _canny_validate_edge_at_boundary,
)

# --- Curved nail-bed mask builder ---
from .bed_mask_builder import _build_curved_bed_mask

# --- End-to-end pipeline & public entry points ---
from .pipeline import (
    _detect_free_edge_by_brightness,
    _boundary_from_free_edge_mask,
    _extract_nail_bed_internal,
    extract_geometry_nail_bed,
    extract_geometry_nail_bed_with_diagnostics,
    extract_nail_bed_overlay_data,
)

__all__ = [
    # Mask processing
    "_clean_mask_morphology",
    # PCA / basic geometry
    "_pca_on_mask_pixels",
    "extract_geometry_rotation_invariant",
    "extract_geometry",
    "extract_geometry_with_diagnostics",
    # Axis orientation
    "_orient_anatomical_axis",
    # Boundary detection
    "_detect_nail_bed_boundary_adaptive",
    "_smooth_profile",
    # Boundary estimators
    "_kmeans_boundary_estimate",
    "_otsu_boundary_estimate",
    "_canny_boundary_estimate",
    # Boundary validation
    "_otsu_validate_color_boundary",
    "_canny_validate_edge_at_boundary",
    # Bed mask builder
    "_build_curved_bed_mask",
    # Pipeline
    "_detect_free_edge_by_brightness",
    "_boundary_from_free_edge_mask",
    "_extract_nail_bed_internal",
    "extract_geometry_nail_bed",
    "extract_geometry_nail_bed_with_diagnostics",
    "extract_nail_bed_overlay_data",
]
