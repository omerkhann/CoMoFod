"""
Post-Processing Module for Copy-Move Forgery Detection (CMFD).

Implements:
  1. Binary mask creation from RANSAC inlier coordinates.
  2. **Manual Morphological Operations** — Dilation, Erosion, Closing —
     via a shift-and-accumulate technique with NumPy (no cv2.morphologyEx).
  3. IoU (Intersection over Union) metric against ground-truth masks.

All morphological ops loop only over kernel elements (typically 3×3 → 9 passes),
not over pixels — fully vectorised per pass.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────
# 1. BINARY MASK FROM INLIER COORDINATES
# ──────────────────────────────────────────────────────────────────────

def create_binary_mask(
    image_shape: Tuple[int, int],
    keypoints: list,
    inliers: List[Tuple[int, int]],
    radius: int = 4,
) -> np.ndarray:
    """
    Build a binary detection mask by marking inlier keypoint locations.

    For every inlier match (i, j), **both** the source (i) and destination (j)
    keypoint locations are marked, because in a copy-move forgery the copied
    region *and* the pasted region are both part of the tampering.

    Each keypoint is drawn as a filled circle of the given radius to create
    a coarse initial mask that will be refined by morphological closing.

    Parameters
    ----------
    image_shape : (H, W) of the original image.
    keypoints   : list of cv2.KeyPoint.
    inliers     : list of (src_idx, dst_idx) inlier matches.
    radius      : Pixel radius of the circle drawn at each keypoint.

    Returns
    -------
    np.ndarray, shape (H, W), dtype uint8, values in {0, 255}.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for src_i, dst_i in inliers:
        pt1 = tuple(map(int, keypoints[src_i].pt))
        pt2 = tuple(map(int, keypoints[dst_i].pt))
        cv2.circle(mask, pt1, radius, 255, -1)
        cv2.circle(mask, pt2, radius, 255, -1)

    return mask


# ──────────────────────────────────────────────────────────────────────
# 2. MANUAL MORPHOLOGICAL OPERATIONS
# ──────────────────────────────────────────────────────────────────────

def manual_dilate(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Manual morphological **dilation** using the shift-and-max technique.

    Dilation expands bright (foreground) regions.  For every pixel, the
    output is the *maximum* over all neighbours within the structuring
    element (a square of side `kernel_size`).

    Implementation trick — instead of sliding a window over every pixel
    (expensive in pure Python), we **shift** the entire image by each
    kernel offset and take the element-wise maximum.  This gives k²
    vectorised NumPy passes for a k×k kernel — very fast.

        dilated(x, y) = max{ mask(x+i, y+j) }  for (i, j) in kernel

    Parameters
    ----------
    mask        : Binary image, shape (H, W), dtype uint8.
    kernel_size : Side length of the square structuring element (odd).

    Returns
    -------
    Dilated binary mask, same shape and dtype.
    """
    pad = kernel_size // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=0)

    h, w = mask.shape
    result = np.zeros_like(mask)

    for i in range(kernel_size):
        for j in range(kernel_size):
            result = np.maximum(result, padded[i : i + h, j : j + w])

    return result


def manual_erode(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Manual morphological **erosion** using the shift-and-min technique.

    Erosion shrinks bright regions.  The output at each pixel is the
    *minimum* over all neighbours within the structuring element.

    Padding uses constant value 255 so that border pixels are not
    automatically eroded away.

        eroded(x, y) = min{ mask(x+i, y+j) }  for (i, j) in kernel

    Parameters
    ----------
    mask        : Binary image, shape (H, W), dtype uint8.
    kernel_size : Side length of the square structuring element (odd).

    Returns
    -------
    Eroded binary mask.
    """
    pad = kernel_size // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=255)

    h, w = mask.shape
    result = np.full_like(mask, 255)

    for i in range(kernel_size):
        for j in range(kernel_size):
            result = np.minimum(result, padded[i : i + h, j : j + w])

    return result


def manual_closing(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    Manual morphological **closing** = dilation → erosion.

    Closing fills small holes and gaps in the foreground without
    significantly expanding the overall shape.

    A larger kernel bridges wider gaps between nearby detected keypoints,
    producing a more solid, region-level mask.

    Parameters
    ----------
    mask        : Binary mask.
    kernel_size : Structuring element size (larger = more aggressive fill).

    Returns
    -------
    Closed binary mask.
    """
    dilated = manual_dilate(mask, kernel_size)
    closed  = manual_erode(dilated, kernel_size)
    return closed


# ──────────────────────────────────────────────────────────────────────
# 3. METRICS: IoU AND DICE
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Intersection over Union (IoU) and DICE coefficient (F1 Score) 
    between the predicted and ground-truth masks.

        IoU  = |Pred ∩ GT| / |Pred ∪ GT|
        DICE = 2 * |Pred ∩ GT| / (|Pred| + |GT|)

    Parameters
    ----------
    pred_mask : Predicted binary mask, shape (H, W).
    gt_mask   : Ground-truth binary mask, shape (H, W).

    Returns
    -------
    iou, dice : Tuple of floats in [0, 1].
    """
    # Binarise
    p = (pred_mask > 127).astype(np.uint8)
    g = (gt_mask   > 127).astype(np.uint8)

    # Resize predicted mask to match ground truth if shapes differ
    if p.shape != g.shape:
        p = cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection = np.sum(p & g)
    union        = np.sum(p | g)
    total_area   = np.sum(p) + np.sum(g)

    iou = float(intersection) / float(union) if union > 0 else 0.0
    dice = 2.0 * float(intersection) / float(total_area) if total_area > 0 else 0.0

    return iou, dice


# ──────────────────────────────────────────────────────────────────────
# 4. FULL POST-PROCESSING PIPELINE
# ──────────────────────────────────────────────────────────────────────

def postprocess(
    image_shape: Tuple[int, int],
    keypoints: list,
    inliers: List[Tuple[int, int]],
    gt_mask: Optional[np.ndarray] = None,
    circle_radius: int = 4,
    dilate_size: int = 5,
    close_size: int = 9,
) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """
    End-to-end post-processing: mask creation → dilation → closing → Metrics.

    Parameters
    ----------
    image_shape    : (H, W) of the input image.
    keypoints      : list of cv2.KeyPoint.
    inliers        : Inlier matches from RANSAC.
    gt_mask        : Optional ground-truth mask for metrics.
    circle_radius  : Radius for initial keypoint circles.
    dilate_size    : Kernel size for extra dilation pass.
    close_size     : Kernel size for morphological closing.

    Returns
    -------
    final_mask : Refined binary mask (H, W), uint8 {0, 255}.
    iou        : IoU score (None if gt_mask not provided).
    dice       : DICE score (None if gt_mask not provided).
    """
    raw_mask  = create_binary_mask(image_shape, keypoints, inliers, circle_radius)
    dilated   = manual_dilate(raw_mask, dilate_size)
    final_mask = manual_closing(dilated, close_size)

    iou = None
    dice = None
    if gt_mask is not None:
        # Ensure GT is single-channel
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, :, 0]
        iou, dice = compute_metrics(final_mask, gt_mask)

    return final_mask, iou, dice
