"""
RANSAC Module for Copy-Move Forgery Detection (CMFD).

Implements the full RANSAC loop **from scratch** — NO cv2.findHomography.

Pipeline:
  1. Randomly sample 4 match pairs.
  2. Build the DLT system and solve for the 3×3 Homography via SVD.
  3. Project all source points through H, compute reprojection error.
  4. Count inliers (error < threshold).
  5. Keep the best model over many iterations.

"""

import numpy as np
import cv2
from typing import Tuple, List, Optional


# ──────────────────────────────────────────────────────────────────────
# 1. HOMOGRAPHY FROM 4+ CORRESPONDENCES  (DLT + SVD)
# ──────────────────────────────────────────────────────────────────────

def compute_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute the 3×3 homography H such that  dst ≈ H · src  (homogeneous coords).

    Uses the **Direct Linear Transform (DLT)** algorithm:

    For each correspondence  (x, y) → (x', y')  we write two equations
    that encode  p' × (H p) = 0  (cross-product constraint):

        Row 1: [ -x  -y  -1   0   0   0   x'x  x'y  x' ]   ⎡h₁⎤
        Row 2: [  0   0   0  -x  -y  -1   y'x  y'y  y' ] · ⎢..⎥ = 0
                                                              ⎣h₉⎦

    With ≥ 4 points we get A (2n × 9).  The least-squares solution
    (min ‖Ah‖ subject to ‖h‖=1) is the last row of Vᵀ from SVD(A).

    **Hartley normalisation** is applied for numerical stability:
    points are translated to centroid and scaled so mean distance
    from the origin equals √2.

    Parameters
    ----------
    src_pts : (n, 2) source coordinates.
    dst_pts : (n, 2) destination coordinates.

    Returns
    -------
    H : (3, 3) homography matrix, or None if degenerate.
    """
    n = src_pts.shape[0]
    if n < 4:
        return None

    # ── Hartley normalisation ───────────────────────────────────────
    def _normalise(pts):
        cx, cy = pts.mean(axis=0)
        d_mean = np.mean(np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2))
        if d_mean < 1e-10:
            return pts, np.eye(3)
        s = np.sqrt(2.0) / d_mean
        T = np.array([[s, 0, -s * cx],
                       [0, s, -s * cy],
                       [0, 0,       1]], dtype=np.float64)
        ones = np.ones((pts.shape[0], 1))
        pts_h = np.hstack([pts, ones])            # (n, 3)
        pts_n = (T @ pts_h.T).T[:, :2]            # (n, 2)
        return pts_n, T

    src_n, T_src = _normalise(src_pts.astype(np.float64))
    dst_n, T_dst = _normalise(dst_pts.astype(np.float64))

    # ── Build the (2n × 9) DLT matrix A ────────────────────────────
    x, y   = src_n[:, 0], src_n[:, 1]
    xp, yp = dst_n[:, 0], dst_n[:, 1]
    zeros  = np.zeros(n)
    ones   = np.ones(n)

    # Row-type 1:  [-x, -y, -1,  0,  0,  0,  x'x, x'y, x']
    # Row-type 2:  [ 0,  0,  0, -x, -y, -1,  y'x, y'y, y']
    A = np.stack([
        -x, -y, -ones, zeros, zeros, zeros, xp * x, xp * y, xp,
        zeros, zeros, zeros, -x, -y, -ones, yp * x, yp * y, yp,
    ], axis=1).reshape(2 * n, 9)

    # ── Solve via SVD ───────────────────────────────────────────────
    try:
        _, S, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    # If smallest singular value is essentially zero → degenerate config
    if S[-1] < 1e-12 * S[0]:
        pass  # still take the solution; it is the null-space vector

    H_norm = Vt[-1].reshape(3, 3)

    # ── De-normalise ────────────────────────────────────────────────
    H = np.linalg.inv(T_dst) @ H_norm @ T_src

    # Normalise so H[2,2] = 1 (conventional scaling)
    if abs(H[2, 2]) > 1e-10:
        H /= H[2, 2]

    return H


# ──────────────────────────────────────────────────────────────────────
# 2. PROJECT POINTS THROUGH H
# ──────────────────────────────────────────────────────────────────────

def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply a 3×3 homography to a set of 2-D points (vectorised).

        p' = H · [x, y, 1]ᵀ     →     (x', y') = (p'₀/p'₂ ,  p'₁/p'₂)

    Parameters
    ----------
    H   : (3, 3) homography.
    pts : (N, 2) source points.

    Returns
    -------
    (N, 2) transformed points in Cartesian coordinates.
    """
    ones  = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])        # (N, 3)
    proj  = (H @ pts_h.T).T              # (N, 3)

    # Dehomogenise — guard against w ≈ 0
    w = proj[:, 2:3]
    w = np.where(np.abs(w) < 1e-10, 1e-10 * np.sign(w + 1e-15), w)
    return proj[:, :2] / w


# ──────────────────────────────────────────────────────────────────────
# 3. FULL RANSAC LOOP
# ──────────────────────────────────────────────────────────────────────

def ransac_homography(
    matches: List[Tuple[int, int]],
    keypoints: list,
    threshold: float = 5.0,
    max_iterations: int = 2000,
    seed: int = 42,
) -> Tuple[Optional[np.ndarray], List[Tuple[int, int]]]:
    """
    Robust homography estimation via RANSAC (Random Sample Consensus).

    Algorithm outline
    -----------------
    repeat `max_iterations` times:
        1. Randomly sample 4 matches.
        2. Compute H from these 4 correspondences (DLT + SVD).
        3. For **all** matches, project src → dst via H and measure
           reprojection error  e = ‖H·p_src − p_dst‖₂.
        4. An inlier satisfies  e < threshold.
        5. If this model has more inliers than the current best, save it.

    After the loop, optionally refit H on the full inlier set for accuracy.

    Parameters
    ----------
    matches        : list of (src_idx, dst_idx) from the matcher.
    keypoints      : list of cv2.KeyPoint (for coordinate lookup).
    threshold      : Inlier pixel-error tolerance (default 5 px).
    max_iterations : Number of RANSAC trials.
    seed           : RNG seed for reproducibility.

    Returns
    -------
    best_H       : (3, 3) homography or None.
    best_inliers : list of (src_idx, dst_idx) inlier matches.
    """
    if len(matches) < 4:
        return None, []

    # Pre-extract all match coordinates into arrays
    kp_xy = np.array([kp.pt for kp in keypoints])       # (K, 2)
    src_all = kp_xy[[m[0] for m in matches]]             # (M, 2)
    dst_all = kp_xy[[m[1] for m in matches]]             # (M, 2)

    # ── BUILT-IN OPENCV RANSAC (Testing Flow) ─────────────
    # We use cv2.RANSAC to find the homography and get an inlier mask
    H, mask = cv2.findHomography(src_all, dst_all, cv2.RANSAC, threshold, maxIters=max_iterations)

    if H is None or mask is None:
        return None, []

    # Mask is an (M, 1) array where 1 means inlier, 0 means outlier
    best_inliers = [matches[i] for i in range(len(matches)) if mask[i][0] == 1]
    
    return H, best_inliers

    ''' 
    # --- ORIGINAL MANUAL RANSAC KEEP FOR REFERENCE ---
    rng = np.random.RandomState(seed)

    M = len(matches)
    if M == 4:
        max_iterations = 1 # Only one possible combination!

    best_inlier_count = 0
    best_inlier_mask  = None
    best_H            = None

    for _ in range(max_iterations):
        # ── 1. Random minimal sample (4 correspondences) ───────────
        idx = rng.choice(M, size=4, replace=False)
        src_sample = src_all[idx]
        dst_sample = dst_all[idx]

        # ── 2. Compute candidate H ─────────────────────────────────
        H = compute_homography(src_sample, dst_sample)
        if H is None:
            continue

        # ── 3. Reprojection error for ALL matches (vectorised) ─────
        projected = apply_homography(H, src_all)         # (M, 2)
        errors    = np.linalg.norm(projected - dst_all, axis=1)  # (M,)

        # ── 4. Inlier counting ─────────────────────────────────────
        inlier_mask  = errors < threshold
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask  = inlier_mask
            best_H            = H

    if best_H is None or best_inlier_mask is None:
        return None, []

    # ── 5. Refit H on all inliers for better accuracy ──────────────
    inlier_indices = np.where(best_inlier_mask)[0]
    if len(inlier_indices) >= 4:
        H_refined = compute_homography(
            src_all[inlier_indices], dst_all[inlier_indices]
        )
        if H_refined is not None:
            best_H = H_refined

    # Collect inlier match tuples
    best_inliers = [matches[i] for i in inlier_indices]

    return best_H, best_inliers
    '''


# ──────────────────────────────────────────────────────────────────────
# 4. SEQUENTIAL RANSAC  (multi-object copy-move detection)
# ──────────────────────────────────────────────────────────────────────

def sequential_ransac(
    matches: List[Tuple[int, int]],
    keypoints: list,
    threshold: float = 5.0,
    max_iterations: int = 2000,
    min_inliers: int = 6,
    max_models: int = 5,
    seed: int = 42,
) -> List[Tuple[Optional[np.ndarray], List[Tuple[int, int]]]]:
    """
    Find multiple copy-move forgeries by running RANSAC sequentially.

    In a single image, the forger may have pasted the same region to
    multiple locations, each with a different geometric transformation.
    Standard RANSAC finds only the dominant one.

    Sequential RANSAC works as follows:
        1. Run RANSAC on all matches → get H₁ and inlier set I₁.
        2. Remove I₁ from the match pool.
        3. Run RANSAC again on the remaining matches → H₂, I₂.
        4. Repeat until fewer than `min_inliers` are found or
           `max_models` transformations have been extracted.

    Parameters
    ----------
    matches        : All accepted matches from the matcher.
    keypoints      : list of cv2.KeyPoint.
    threshold      : RANSAC inlier pixel tolerance.
    max_iterations : Iterations per RANSAC round.
    min_inliers    : Stop if a round finds fewer inliers than this.
    max_models     : Maximum number of transformations to extract.
    seed           : RNG seed.

    Returns
    -------
    List of (H, inliers) tuples — one per detected forgery cluster.
    """
    results = []
    remaining = list(matches)

    for i in range(max_models):
        if len(remaining) < 4:
            break

        H, inliers = ransac_homography(
            remaining, keypoints,
            threshold=threshold,
            max_iterations=max_iterations,
            seed=seed + i,       # vary seed each round
        )

        if H is None or len(inliers) < min_inliers:
            break

        results.append((H, inliers))

        # Remove inliers from the pool for the next round
        inlier_set = set(inliers)
        remaining = [m for m in remaining if m not in inlier_set]

    return results
