"""
Feature Matching Module for Copy-Move Forgery Detection (CMFD).

Implements:
  1. SIFT keypoint / descriptor extraction (cv2 — allowed by spec).
  2. Manual pairwise Euclidean (L₂) distance matrix — fully vectorized.
  3. Manual Lowe's Ratio Test with spatial-distance filtering.

"""

import numpy as np
import cv2
from typing import Tuple, List


# ──────────────────────────────────────────────────────────────────────
# 1. SIFT FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────

def extract_sift_features(
    gray_image: np.ndarray,
    n_features: int = 0,
) -> Tuple[list, np.ndarray]:
    """
    Detect SIFT keypoints and compute 128-D descriptors.

    SIFT builds a scale-space pyramid, detects DoG extrema, refines them
    to sub-pixel accuracy, assigns dominant gradient orientations, and
    finally constructs a 4×4 grid of 8-bin orientation histograms →
    128-dimensional descriptor per keypoint.

    Parameters
    ----------
    gray_image : Grayscale uint8 image.
    n_features : Cap on keypoints (0 = unlimited).

    Returns
    -------
    keypoints   : list of cv2.KeyPoint
    descriptors : np.ndarray, shape (N, 128), dtype float32
    """
    # sift = cv2.SIFT_create(nfeatures=n_features)
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.03, edgeThreshold=10)
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors


# ──────────────────────────────────────────────────────────────────────
# 2. MANUAL EUCLIDEAN DISTANCE MATRIX
# ──────────────────────────────────────────────────────────────────────

def compute_distance_matrix(desc: np.ndarray) -> np.ndarray:
    """
    Compute the full N×N pairwise L₂ distance matrix for self-matching.

    Uses the algebraic identity to avoid a huge (N, N, 128) tensor:

        ‖a − b‖² = ‖a‖² + ‖b‖² − 2·(a · b)

    Memory: O(N²) instead of O(N²·D).  All operations are BLAS-backed
    matrix multiplications — extremely fast even for N > 5 000.

    The diagonal is set to +∞ so a descriptor never matches itself.

    Parameters
    ----------
    desc : np.ndarray, shape (N, 128)
        SIFT descriptors (all from one image for copy-move detection).

    Returns
    -------
    np.ndarray, shape (N, N), dtype float64
        Entry [i, j] = ‖desc[i] − desc[j]‖₂.
    """
    d = desc.astype(np.float64)
    
    # Using the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    # We compute it carefully to avoid any weirdness
    sq_norms = np.einsum('ij,ij->i', d, d)  # (N,)
    dot = np.dot(d, d.T)                    # (N, N)
    
    # Use broadcasting to get (N, N) distance squared matrix
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * dot
    dist_sq = np.maximum(dist_sq, 0.0)      # numerical guard
    
    distances = np.sqrt(dist_sq)
    np.fill_diagonal(distances, np.inf)      # forbid self-match
    return distances


# ──────────────────────────────────────────────────────────────────────
# 3. LOWE'S RATIO TEST  +  SPATIAL FILTERING
# ──────────────────────────────────────────────────────────────────────

def lowes_ratio_test(
    distance_matrix: np.ndarray,
    keypoints: list,
    ratio_threshold: float = 0.7,
    min_spatial_dist: float = 10.0,
) -> List[Tuple[int, int]]:
    """
    Filter matches with Lowe's Ratio Test and a spatial-separation guard.

    ── Lowe's Ratio Test ──
    For each descriptor i, let d₁ and d₂ be the distances to its nearest
    and second-nearest neighbour.  Accept the match only if:

        d₁ / d₂  <  ratio_threshold

    This rejects *ambiguous* matches where several descriptors look alike.

    ── Spatial Filter ──
    In copy-move detection both source and copy live in the **same** image.
    Nearby keypoints often share texture and produce trivial "matches" that
    are not forgery.  We discard any match whose two keypoints are closer
    than `min_spatial_dist` pixels apart.

    Parameters
    ----------
    distance_matrix   : (N, N) self-distance matrix (diagonal = ∞).
    keypoints         : list of cv2.KeyPoint for spatial coordinates.
    ratio_threshold   : Lowe's ratio (default 0.7, slider range 0.4–0.9).
    min_spatial_dist  : Minimum pixel separation to keep a match.

    Returns
    -------
    List of (query_idx, match_idx) index pairs into the keypoints list.
    """
    N = distance_matrix.shape[0]
    if N < 3:
        return []

    # ── 1. Apply Spatial Constraint BEFORE NN Search ───────────────
    # If we don't do this, overlapping SIFT keypoints (spatially close) 
    # will be selected as the 2nd Nearest Neighbor, which makes d1 ≈ d2 
    # and causes valid matches to falsely fail Lowe's Ratio Test!
    kp_xy = np.array([kp.pt for kp in keypoints])  # (N, 2)
    # Compute pairwise spatial distances efficiently using algebraic expansion
    sq_norms_xy = np.sum(kp_xy ** 2, axis=1, keepdims=True)
    spatial_dist_sq = sq_norms_xy + sq_norms_xy.T - 2.0 * (kp_xy @ kp_xy.T)
    spatial_dist_sq = np.maximum(spatial_dist_sq, 0.0) # Guard against floating point drift
    
    # Set descriptor distance to infinity for keypoints that are too close spatially
    # (This also correctly handles the diagonal, which is distance 0)
    min_sq = min_spatial_dist ** 2
    distance_matrix[spatial_dist_sq < min_sq] = np.inf

    # ── Vectorised 2-NN search ──────────────────────────────────────
    # argpartition is O(N) per row (vs O(N log N) for argsort).
    idx_2nn   = np.argpartition(distance_matrix, kth=2, axis=1)[:, :2]
    rows      = np.arange(N)[:, None]
    dists_2nn = distance_matrix[rows, idx_2nn]                   # (N, 2)

    # sort the two so col-0 = nearest, col-1 = second-nearest
    order      = np.argsort(dists_2nn, axis=1)
    dists_sort = np.take_along_axis(dists_2nn, order, axis=1)
    idx_sort   = np.take_along_axis(idx_2nn, order, axis=1)

    d1       = dists_sort[:, 0]
    d2       = dists_sort[:, 1]
    best_idx = idx_sort[:, 0]

    # ── Ratio test (vectorised) ─────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(d2 > 0, d1 / d2, 1.0)
    
    ratio_mask = ratios < ratio_threshold

    # ── Spatial distance filter ─────────────────────────────────────
    kp_xy = np.array([kp.pt for kp in keypoints])               # (N, 2)
    query_ids = np.where(ratio_mask)[0]
    match_ids = best_idx[query_ids]

    # ── Symmetric Cross-Check (Mutual Nearest Neighbors) ────────────
    # A match is only valid if query_i's best match is match_j, 
    # AND match_j's best match is query_i.
    cross_check_mask = (best_idx[match_ids] == query_ids)
    
    query_ids = query_ids[cross_check_mask]
    match_ids = match_ids[cross_check_mask]

    return list(zip(query_ids.tolist(), match_ids.tolist()))


# ──────────────────────────────────────────────────────────────────────
# 4. SPATIAL DENSITY FILTERING
# ──────────────────────────────────────────────────────────────────────

def spatial_density_filter(
    matches: List[Tuple[int, int]],
    keypoints: list,
    radius: float = 50.0,
    min_neighbors: int = 3
) -> List[Tuple[int, int]]:
    """
    Remove isolated "stray" matches that don't belong to a cluster.
    
    In copy-move forgery, a copied region will generate a dense cluster 
    of matches. Stray false matches will be isolated. A match is kept only 
    if it has at least `min_neighbors` other matches nearby in BOTH the 
    source region and the destination region.
    """
    if not matches:
        return []

    kp_xy = np.array([kp.pt for kp in keypoints])
    src_pts = kp_xy[[m[0] for m in matches]]
    dst_pts = kp_xy[[m[1] for m in matches]]
    
    valid_matches = []
    # Vectorized computation of pairwise distances is possible, but 
    # a simple loop is fine since len(matches) is usually < 1000 after ratio test.
    for i in range(len(matches)):
        src_dist = np.linalg.norm(src_pts - src_pts[i], axis=1)
        dst_dist = np.linalg.norm(dst_pts - dst_pts[i], axis=1)
        
        # A match is a neighbor if it's close in BOTH source and dest spaces
        neighbors = np.sum((src_dist < radius) & (dst_dist < radius)) - 1 # subtract self
        
        if neighbors >= min_neighbors:
            valid_matches.append(matches[i])
            
    return valid_matches


# ──────────────────────────────────────────────────────────────────────
# 5. CONVENIENCE WRAPPER
# ──────────────────────────────────────────────────────────────────────

def match_keypoints(
    keypoints: list,
    descriptors: np.ndarray,
    ratio: float = 0.7,
    min_spatial_dist: float = 10.0,
) -> List[Tuple[int, int]]:
    """
    End-to-end matching pipeline: distance matrix → ratio test → spatial filter.

    Parameters
    ----------
    keypoints        : SIFT keypoints.
    descriptors      : SIFT descriptors, (N, 128).
    ratio            : Lowe's ratio threshold.
    min_spatial_dist : Minimum pixel gap between matched keypoints.

    Returns
    -------
    List of (src_idx, dst_idx) accepted match pairs.
    """
    if descriptors is None or len(descriptors) < 3:
        return []

    dist_mat = compute_distance_matrix(descriptors)
    matches = lowes_ratio_test(dist_mat, keypoints, ratio, min_spatial_dist)
    
    # Apply density filter to kill isolated blobs but keep smaller legitimate clusters
    return spatial_density_filter(matches, keypoints, radius=100.0, min_neighbors=1)


def draw_matches(image: np.ndarray,
                 keypoints: list,
                 matches: List[Tuple[int, int]],
                 max_draw: int = 200) -> np.ndarray:
    """
    Visualise matches by drawing circles and connecting lines on the image.

    Parameters
    ----------
    image     : BGR image.
    keypoints : list of cv2.KeyPoint.
    matches   : list of (src_idx, dst_idx).
    max_draw  : cap on drawn matches (avoids visual clutter).

    Returns
    -------
    BGR image with match visualisation.
    """
    vis = image.copy()
    rng = np.random.RandomState(42)

    for src_i, dst_i in matches[:max_draw]:
        colour = tuple(rng.randint(80, 255, 3).tolist())
        pt1 = tuple(map(int, keypoints[src_i].pt))
        pt2 = tuple(map(int, keypoints[dst_i].pt))
        cv2.circle(vis, pt1, 4, colour, 1, cv2.LINE_AA)
        cv2.circle(vis, pt2, 4, colour, 1, cv2.LINE_AA)
        cv2.line(vis, pt1, pt2, colour, 1, cv2.LINE_AA)

    return vis
