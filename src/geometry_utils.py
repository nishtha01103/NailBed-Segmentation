"""Geometry extraction utilities for nail measurements.

**Backward-compatibility shim** -- the implementation has moved to the
``src.geometry`` package.  This module re-exports every public and private
symbol so that all existing ``from src.geometry_utils import ...`` statements
continue to work without modification.
"""

# Re-export the full public (and private) API from the new package.
from .geometry import (                         # noqa: F401
    # Mask processing
    _clean_mask_morphology,
    # PCA / basic geometry
    _pca_on_mask_pixels,
    extract_geometry_rotation_invariant,
    extract_geometry,
    extract_geometry_with_diagnostics,
    # Axis orientation
    _orient_anatomical_axis,
    # Boundary detection
    _smooth_profile,
    # Boundary estimators
    _kmeans_boundary_estimate,
    _otsu_boundary_estimate,
    _canny_boundary_estimate,
    # Boundary validation
    _otsu_validate_color_boundary,
    _canny_validate_edge_at_boundary,
    # Bed mask builder
    _build_curved_bed_mask,
    # Pipeline
    _detect_free_edge_by_brightness,
    _boundary_from_free_edge_mask,
    _extract_nail_bed_internal,
    extract_geometry_nail_bed,
    extract_geometry_nail_bed_with_diagnostics,
    extract_nail_bed_overlay_data,
)
