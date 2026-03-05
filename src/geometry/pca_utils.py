"""Moment-based axis computation and geometry extraction for nail measurements.

Moment-based axis is faster and more robust than full PCA for real-time mask
analysis.  Falls back to PCA when cv2.moments reports zero area.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from config import MIN_CONTOUR_AREA

from .mask_processing import _clean_mask_morphology
from .axis_orientation import _orient_anatomical_axis

# Optional thin/skeletonize backends (fastest available is used at runtime)
try:
    _cv2_thinning = cv2.ximgproc.thinning          # OpenCV contrib
    _HAS_XIMGPROC = True
except AttributeError:
    _HAS_XIMGPROC = False
try:
    from skimage.morphology import skeletonize as _ski_skeletonize
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

# Nail aspect ratio constraints (rotation-invariant)
MIN_NAIL_ASPECT_RATIO = 0.5  # Allows wide nails (thumbs)
MAX_NAIL_ASPECT_RATIO = 3.0  # Allows long nails


def _pca_on_mask_pixels(
    mask: np.ndarray,
    sample_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute principal axes using second-order central moments from cv2.moments.

    Moment-based axis is faster and more robust than full PCA for real-time
    mask analysis.

    Falls back to PCA when the mask area (m00) is zero.

    IMPORTANT: The returned axes are ordered by variance (moment magnitude),
    NOT by anatomical meaning.  The caller MUST use ``_orient_anatomical_axis``
    to determine which axis corresponds to anatomical length vs width.

    Args:
        mask:        Binary mask (255 = nail)
        sample_rate: Kept for API compatibility (unused by moment method;
                     used only by the PCA fallback path).

    Returns:
        centroid   : (2,) float64  [x, y]
        major_axis : (2,) float64  unit vector of highest-variance axis
        minor_axis : (2,) float64  unit vector of lowest-variance axis
    """
    # ── Moment-based computation ──────────────────────────────────────────────
    moments = cv2.moments(mask.astype(np.uint8))
    m00 = moments["m00"]

    if m00 == 0:
        # Fallback to PCA method when mask has no area
        return _pca_fallback(mask, sample_rate)


    # Centroid
    cx = moments["m10"] / m00
    cy = moments["m01"] / m00
    centroid = np.array([cx, cy], dtype=np.float64)

    # Second-order central moments
    mu20 = moments["mu20"]
    mu02 = moments["mu02"]
    mu11 = moments["mu11"]

    # Axis stability: low value means mu20 ≈ mu02 (near-square mask)
    axis_stability = abs(mu20 - mu02) / (mu20 + mu02 + 1e-6)

    if axis_stability < 0.02:
        # Skeleton-based axis provides robust orientation for near-square nails.
        # Try skeleton PCA first; if it fails, fall back to minAreaRect.
        skel_result = _skeleton_axis(mask, centroid)
        if skel_result is not None:
            return skel_result

        # Secondary fallback: minAreaRect orientation.
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            _, _, angle = cv2.minAreaRect(largest)
            theta = angle * np.pi / 180.0
        else:
            # No contour found; fall back to moment-based angle.
            theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    else:
        # Orientation angle from central moments
        theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)

    # Major axis vector (highest-variance direction)
    major_axis = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    major_axis /= np.linalg.norm(major_axis)  # normalize

    # Minor axis vector (perpendicular to major)
    minor_axis = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    minor_axis /= np.linalg.norm(minor_axis)  # normalize

    return centroid, major_axis, minor_axis


def _skeleton_axis(
    mask: np.ndarray,
    centroid: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Estimate major axis from the mask skeleton via PCA.

    Skeleton-based axis provides robust orientation for near-square nails.

    Uses cv2.ximgproc.thinning when available (fastest); falls back to
    skimage.morphology.skeletonize.  Returns None when neither backend is
    available, when the mask is too small, or when the skeleton has fewer
    than 20 pixels (too sparse for reliable PCA).

    Args:
        mask:     Binary mask (255 = nail).  Not modified.
        centroid: (2,) float64 [x, y] from cv2.moments (kept as-is).

    Returns:
        (centroid, major_axis, minor_axis) on success, or None on failure.
    """
    # ── Step 1: Skeletonize ───────────────────────────────────────────────────
    m = mask.astype(np.uint8)
    if _HAS_XIMGPROC:
        skeleton = _cv2_thinning(m)
    elif _HAS_SKIMAGE:
        # skimage returns a bool array; convert to uint8 0/255
        skeleton = _ski_skeletonize(m > 0).astype(np.uint8) * 255
    else:
        return None  # no thinning backend available

    # ── Step 2: Extract skeleton pixel coordinates ────────────────────────────
    ys, xs = np.where(skeleton > 0)

    # ── Step 2.5: Remove skeleton branches — keep only main skeleton path ─────
    # Keeps only main skeleton path for axis estimation.
    # Small side-branches can skew the PCA eigenvector away from the true
    # nail length direction; retaining only the largest connected component
    # of the skeleton eliminates them before fitting.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton, connectivity=8
    )
    if num_labels > 2:  # label 0 = background; >1 foreground component exists
        # Find the largest foreground component (label 0 is background)
        largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        # Zero out all other components in-place on a local copy
        pruned = np.where(labels == largest_label, np.uint8(255), np.uint8(0))
        ys, xs = np.where(pruned > 0)

    # ── Step 3: Guard — too few skeleton points for reliable PCA ─────────────
    if len(xs) < 20:
        return None

    # ── Step 4: PCA on skeleton coordinates ──────────────────────────────────
    coords = np.column_stack((xs, ys)).astype(np.float64)
    mean_pt = np.mean(coords, axis=0)
    centered = coords - mean_pt
    cov = np.cov(centered, rowvar=False)  # 2×2 symmetric matrix
    eigvals, eigvecs = np.linalg.eigh(cov)  # sorted ascending

    major_axis = eigvecs[:, np.argmax(eigvals)].astype(np.float64)

    # ── Step 5: Normalize ─────────────────────────────────────────────────────
    norm = np.linalg.norm(major_axis)
    if norm < 1e-9:
        return None
    major_axis /= norm

    # ── Step 6: Minor axis (perpendicular) ────────────────────────────────────
    minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float64)

    # ── Step 7: Return centroid from moments and skeleton-derived axes ────────
    return centroid, major_axis, minor_axis


def _pca_fallback(
    mask: np.ndarray,
    sample_rate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA fallback for when moment-based computation fails (m00 == 0).

    Uses numpy.linalg.eigh on the covariance matrix of mask pixels.
    """
    nail_yx = np.argwhere(mask == 255)           # (N,2)  row, col

    MIN_SAMPLE = 200
    if sample_rate < 1.0:
        n_total  = len(nail_yx)
        n_sample = int(n_total * sample_rate)
        if n_sample >= MIN_SAMPLE:
            idx     = np.random.choice(n_total, size=n_sample, replace=False)
            nail_yx = nail_yx[idx]

    nail_xy = nail_yx[:, ::-1].astype(np.float64)  # (N,2)  x, y

    centroid = nail_xy.mean(axis=0)
    centered = nail_xy - centroid

    cov = np.cov(centered.T)                     # 2x2 symmetric
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # real, sorted ascending
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
        major_axis, minor_axis, _ = _orient_anatomical_axis(
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
        "method": "moments_with_pca_fallback",
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