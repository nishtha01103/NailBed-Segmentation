"""Mask cleaning utilities for nail segmentation geometry pipeline."""

import cv2
import numpy as np
from config import CLEAN_MASK_BEFORE_GEOMETRY, MORPHOLOGY_KERNEL_SIZE


def _clean_mask_morphology(mask: np.ndarray) -> np.ndarray:
    """Remove outliers and noise from mask using morphological operations.

    1. Closing: Fill small holes/gaps inside nail
    2. Opening: Remove small protrusions and noise at edges

    Args:
        mask: Binary mask where 255 = nail region, 0 = background

    Returns:
        Cleaned binary mask with outliers removed
    """
    if not CLEAN_MASK_BEFORE_GEOMETRY:
        return mask

    # Morphological closing and opening
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE)
    )
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # --- A. Keep Only Largest Connected Component ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_opened, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    if len(areas) > 1:
        largest_label = np.argmax(areas[1:]) + 1
        clean_mask = np.zeros_like(mask_opened)
        clean_mask[labels == largest_label] = 255
        mask = clean_mask
    else:
        mask = mask_opened.copy()

    # --- B. Remove Small Fragments ---
    min_area = 0.05 * np.sum(mask > 0)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255
    mask = clean_mask


    # Optional: slight erosion to remove distal skin contamination
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_final = cv2.erode(mask, erode_kernel, iterations=1)
    return mask_final
