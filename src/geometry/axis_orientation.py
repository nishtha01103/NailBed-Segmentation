"""Anatomical axis orientation using scoring-based selection.

Fully rotation-invariant — does not assume PCA eigenvalue ordering
corresponds to anatomical length vs width.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


def _orient_anatomical_axis(
    image: np.ndarray,
    mask: np.ndarray,
    centroid: np.ndarray,
    major_axis: np.ndarray,  # Unused, kept for API compatibility
    minor_axis: np.ndarray,  # Unused, kept for API compatibility
    lab_frame: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Orient anatomical axes using minAreaRect, not PCA.

    Steps:
    - Extract largest contour from mask.
    - Use cv2.minAreaRect to get rectangle.
    - The longer side is the anatomical axis (proximal→distal).
    - The shorter side is the width axis.
    - Determine distal direction by width at ends (narrower = distal),
      or fallback to brightness if taper is weak.
    Returns:
        anatomical_axis_vector, width_axis_vector, distal_sign (+1 or -1)
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: return x and y axes
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (width, height), angle = rect
    # Determine which is longer
    if width >= height:
        long_len, short_len = width, height
        angle_rad = np.deg2rad(angle)
    else:
        long_len, short_len = height, width
        angle_rad = np.deg2rad(angle + 90)
    # Anatomical axis (proximal→distal) from minAreaRect
    axis_rect = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    # PCA axis
    nail_yx = np.argwhere(mask == 255)
    nail_xy = nail_yx[:, ::-1].astype(np.float64)
    centered = nail_xy - np.array([center_x, center_y])
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis_pca = eigvecs[:, np.argmax(eigvals)]
    axis_pca = axis_pca / np.linalg.norm(axis_pca)
    # Compare axes
    dot = np.clip(np.dot(axis_rect, axis_pca), -1, 1)
    angle_diff = np.degrees(np.arccos(abs(dot)))
    if angle_diff > 40:
        anatomical_axis = axis_pca
        width_axis = np.array([-anatomical_axis[1], anatomical_axis[0]])
    else:
        anatomical_axis = axis_rect
        width_axis = np.array([-anatomical_axis[1], anatomical_axis[0]])
    # Project all mask points onto anatomical axis
    nail_yx = np.argwhere(mask == 255)
    nail_xy = nail_yx[:, ::-1].astype(np.float64)
    proj = (nail_xy - np.array([center_x, center_y])) @ anatomical_axis
    # Split into two ends
    q = max(10, len(proj) // 5)
    sort_idx = np.argsort(proj)
    end1_idx = sort_idx[:q]
    end2_idx = sort_idx[-q:]
    # Width at each end
    perp = width_axis
    p_perp = (nail_xy - np.array([center_x, center_y])) @ perp
    # Robust distal determination
    if lab_frame is not None:
        lab_img = lab_frame
    else:
        lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_all = lab_img[:, :, 0][nail_yx[:, 0], nail_yx[:, 1]]
    a_all = lab_img[:, :, 1][nail_yx[:, 0], nail_yx[:, 1]]
    perp = width_axis
    p_perp = (nail_xy - np.array([center_x, center_y])) @ perp
    w1 = float(np.mean(np.abs(p_perp[end1_idx])))
    w2 = float(np.mean(np.abs(p_perp[end2_idx])))
    norm_L1 = (np.mean(L_all[end1_idx]) - np.mean(L_all)) / (np.std(L_all) + 1e-6)
    norm_L2 = (np.mean(L_all[end2_idx]) - np.mean(L_all)) / (np.std(L_all) + 1e-6)
    norm_a1 = (np.mean(a_all[end1_idx]) - np.mean(a_all)) / (np.std(a_all) + 1e-6)
    norm_a2 = (np.mean(a_all[end2_idx]) - np.mean(a_all)) / (np.std(a_all) + 1e-6)
    norm_w1 = (w1 - np.mean([w1, w2])) / (np.std([w1, w2]) + 1e-6)
    norm_w2 = (w2 - np.mean([w1, w2])) / (np.std([w1, w2]) + 1e-6)
    score1 = +norm_L1 - norm_a1 - norm_w1
    score2 = +norm_L2 - norm_a2 - norm_w2
    distal_sign = 1 if score2 > score1 else -1
    anatomical_axis = anatomical_axis * distal_sign
    width_axis = np.array([-anatomical_axis[1], anatomical_axis[0]])
    return anatomical_axis, width_axis, distal_sign
    w1 = float(np.percentile(p_perp[end1_idx], 95) - np.percentile(p_perp[end1_idx], 5))
    w2 = float(np.percentile(p_perp[end2_idx], 95) - np.percentile(p_perp[end2_idx], 5))
    avg_w = 0.5 * (w1 + w2)
    taper = abs(w1 - w2) / max(avg_w, 1.0)
    # Decide distal direction
    if taper > 0.10:
        distal_sign = 1 if w2 < w1 else -1  # distal = narrower end
    else:
        # Use brightness (L channel) if taper is weak
        if lab_frame is not None:
            lab_img = lab_frame
        else:
            lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_all = lab_img[:, :, 0][nail_yx[:, 0], nail_yx[:, 1]]
        L1 = float(np.mean(L_all[end1_idx]))
        L2 = float(np.mean(L_all[end2_idx]))
        distal_sign = 1 if L2 > L1 else -1  # distal = brighter end
    # Orient anatomical_axis so positive direction is proximal→distal
    anatomical_axis = anatomical_axis * distal_sign
    width_axis = np.array([-anatomical_axis[1], anatomical_axis[0]])
    return anatomical_axis, width_axis, distal_sign