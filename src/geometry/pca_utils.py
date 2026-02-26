"""PCA-based axis computation and geometry extraction for nail measurements."""

from typing import Tuple, Optional
import cv2
import numpy as np
from config import MIN_CONTOUR_AREA

from .mask_processing import _clean_mask_morphology
from .axis_orientation import _orient_anatomical_axis

# Nail aspect ratio constraints (rotation-invariant)
MIN_NAIL_ASPECT_RATIO = 0.5  # Allows wide nails (thumbs)
MAX_NAIL_ASPECT_RATIO = 3.0  # Allows long nails


def _pca_on_mask_pixels(
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run PCA on ALL mask pixels (not contour points) for stable axis estimation.

    Using all interior pixels rather than contour boundary points makes the
    axes far more stable when nail boundaries are irregular or slightly
    noisy.  numpy.linalg.eigh is used (symmetric matrix) to guarantee real
    eigenvalues.

    IMPORTANT: The returned axes are ordered by PCA eigenvalue (variance),
    NOT by anatomical meaning.  The caller MUST use ``_orient_anatomical_axis``
    to determine which axis corresponds to anatomical length vs width.

    Args:
        mask: Binary mask (255 = nail)

    Returns:
        centroid   : (2,) float64  [x, y]
        major_axis : (2,) float64  unit vector of highest-variance PCA axis
        minor_axis : (2,) float64  unit vector of lowest-variance PCA axis
    """
    nail_yx = np.argwhere(mask == 255)           # (N,2)  row, col
    nail_xy = nail_yx[:, ::-1].astype(np.float64)  # (N,2)  x, y

    centroid = nail_xy.mean(axis=0)
    centered = nail_xy - centroid

    cov = np.cov(centered.T)                     # 2x2 symmetric
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # real, sorted ascending
    # eigh returns ascending order -> reverse for descending
    major_axis = eigenvectors[:, -1].astype(np.float64)   # largest eigenvalue
    minor_axis = eigenvectors[:,  0].astype(np.float64)   # smallest eigenvalue

    return centroid, major_axis, minor_axis


def extract_geometry_rotation_invariant(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract geometric measurements using PCA on all mask pixels.

    Uses all interior mask pixels (not just contour points) for stable
    major/minor axis estimation.  numpy.linalg.eigh guarantees real eigenvalues
    on the symmetric covariance matrix.

    When ``image`` is provided, ``_orient_anatomical_axis`` is called to ensure
    ``major_axis`` = proximal\u2192distal direction.  For nails wider than they are
    long (very wide thumbnails), the raw PCA major axis would be the anatomical
    width direction; the orientation step swaps the axes so that
    ``length_px`` always reflects the proximal\u2192distal extent.

    Args:
        mask:  Binary mask (255\u202f= nail)
        image: Optional BGR image used for axis orientation.  When None,
               the raw PCA major axis is used (highest variance direction).

    Returns:
        (length_px, width_px, area_px) or (None, None, None)
        length = extent along proximal\u2192distal axis (can be < width for wide nails
                 when no image is supplied and PCA gives landscape orientation)
        width  = extent along anatomical width axis
    """
    mask_cleaned = _clean_mask_morphology(mask)

    contours, _ = cv2.findContours(
        mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None, None

    cnt = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(cnt)
    if area_px < MIN_CONTOUR_AREA:
        return None, None, None

    perimeter = cv2.arcLength(cnt, True)
    if perimeter > 0:
        circularity = 4 * np.pi * area_px / (perimeter ** 2)
        if circularity < 0.25:  # only reject very non-compact shapes (lines/fragments)
            return None, None, None

    if mask_cleaned.sum() == 0:
        return None, None, None

    # PCA on ALL mask pixels for stable axis
    centroid, major_axis, minor_axis = _pca_on_mask_pixels(mask_cleaned)

    # Orient major_axis = proximal\u2192distal when image is available.
    if image is not None:
        major_axis, minor_axis = _orient_anatomical_axis(
            image, mask_cleaned, centroid, major_axis, minor_axis
        )

    nail_yx = np.argwhere(mask_cleaned == 255)
    nail_xy = nail_yx[:, ::-1].astype(np.float64)
    centered_all = nail_xy - centroid

    proj_major = centered_all @ major_axis
    proj_minor = centered_all @ minor_axis

    # Anatomical axis measurements:
    #   length = extent along the anatomical (proximal\u2192distal) axis
    #   width  = extent along the perpendicular (lateral) axis
    # These are determined purely by the anatomical axis scoring in
    # _orient_anatomical_axis \u2014 no span-ratio swap is applied.
    length_px = float(proj_major.max() - proj_major.min())
    width_px  = float(proj_minor.max() - proj_minor.min())

    # Validate shape: reject shapes that are too elongated to be a nail.
    # Uses the larger/smaller ratio regardless of which dimension is
    # anatomical length so that the check fires for both needle-like and
    # very-wide false-positive detections.
    if min(length_px, width_px) > 0:
        sym_ratio = max(length_px, width_px) / min(length_px, width_px)
        if sym_ratio > MAX_NAIL_ASPECT_RATIO:
            return None, None, None

    return length_px, width_px, float(area_px)


def extract_geometry(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Main geometry extraction entry point (rotation-invariant PCA).

    Pass ``image`` to enable anatomical axis orientation so that
    ``length_px`` is always the proximal\u2192distal extent.
    """
    return extract_geometry_rotation_invariant(mask, image=image)


def extract_geometry_with_diagnostics(
    mask: np.ndarray,
    verbose: bool = False,
    image: Optional[np.ndarray] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[dict]]:
    """Extract geometry with detailed diagnostic information.

    Args:
        mask:    Binary mask (255 = nail)
        verbose: Print diagnostics to stdout when True
        image:   Optional BGR image for anatomical axis orientation

    Returns:
        (length_px, width_px, area_px, diagnostics_dict)
    """
    from config import CLEAN_MASK_BEFORE_GEOMETRY

    mask_cleaned = _clean_mask_morphology(mask)

    contours, _ = cv2.findContours(
        mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    diagnostics: dict = {
        "num_contours": len(contours),
        "area": None,
        "aspect_ratio": None,
        "length_px": None,
        "width_px": None,
        "reason_for_rejection": None,
        "mask_cleaning_applied": CLEAN_MASK_BEFORE_GEOMETRY,
        "method": "pca_all_mask_pixels",
    }

    if not contours:
        diagnostics["reason_for_rejection"] = "No contours found"
        return None, None, None, diagnostics

    cnt = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(cnt)
    diagnostics["area"] = area_px

    if area_px < MIN_CONTOUR_AREA:
        diagnostics["reason_for_rejection"] = (
            f"Area too small ({area_px:.0f} < {MIN_CONTOUR_AREA})"
        )
        return None, None, None, diagnostics

    length_px, width_px, _ = extract_geometry_rotation_invariant(mask, image=image)
    if length_px is None:
        diagnostics["reason_for_rejection"] = (
            "PCA extraction failed or aspect ratio out of range"
        )
        return None, None, None, diagnostics

    diagnostics["length_px"]   = length_px
    diagnostics["width_px"]    = width_px
    diagnostics["aspect_ratio"] = round(length_px / width_px, 3) if width_px else None

    if verbose:
        print(f"  Contours: {diagnostics['num_contours']}")
        print(f"  Area: {area_px:.0f} px\u00b2")
        print(f"  Length: {length_px:.1f}px  Width: {width_px:.1f}px")
        print(f"  Ratio: {diagnostics['aspect_ratio']}")
        print(f"  Status: VALID")

    return float(length_px), float(width_px), float(area_px), diagnostics
