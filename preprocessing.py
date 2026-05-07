"""
Preprocessing Module for Copy-Move Forgery Detection (CMFD).

This module handles:
  1. Manual grayscale conversion using the ITU-R BT.601 luminance formula.
  2. Manual Gaussian smoothing via a hand-built kernel and 2D convolution.

All operations are implemented with NumPy vectorization — no per-pixel loops.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. GRAYSCALE CONVERSION
# ──────────────────────────────────────────────────────────────────────

def manual_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale using the ITU-R BT.601 luminance formula.

    The human visual system is most sensitive to green, then red, then blue.
    The luminance (perceived brightness) is therefore a *weighted* sum:

        Y = 0.2989·R + 0.5870·G + 0.1140·B

    OpenCV loads images in BGR channel order, so we index:
        B = channel 0,  G = channel 1,  R = channel 2.

    The entire computation is a single vectorized multiply-accumulate
    over (H, W) arrays — no Python loops.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3)
        Input image in BGR colour space (as loaded by cv2.imread).

    Returns
    -------
    np.ndarray, shape (H, W), dtype uint8
        Single-channel grayscale image.
    """
    if image.ndim == 2:
        return image.astype(np.uint8)

    # Separate channels (views, zero-copy) and promote to float for precision
    b = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    r = image[:, :, 2].astype(np.float64)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return np.clip(gray, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────
# 2. GAUSSIAN SMOOTHING
# ──────────────────────────────────────────────────────────────────────

def gaussian_kernel(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Build a normalised 2-D Gaussian kernel.

    The 2-D Gaussian is separable and defined as:

        G(x, y) = ──────────1────────── · exp( -(x² + y²) / (2σ²) )
                    2 π σ²

    We evaluate this on a discrete (size × size) grid centred at (0, 0),
    then normalise so the kernel sums to 1 (energy preservation).

    Parameters
    ----------
    size  : int   – Side length of the square kernel (must be odd).
    sigma : float – Standard deviation of the Gaussian.

    Returns
    -------
    np.ndarray, shape (size, size), dtype float64
        Normalised Gaussian kernel.
    """
    ax = np.arange(size) - size // 2          # e.g. [-1, 0, 1] for size=3
    xx, yy = np.meshgrid(ax, ax)              # coordinate grids
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def manual_convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2-D convolution via the sliding-window / shift-and-multiply approach.

    Instead of looping over every pixel, we loop only over the (small)
    kernel elements and perform full-image array operations at each step.
    For a k×k kernel this is k² vectorised passes — negligible for k ≤ 7.

    Boundary handling: reflect-padding preserves edge detail.

    Parameters
    ----------
    image  : np.ndarray, shape (H, W) – Grayscale image.
    kernel : np.ndarray, shape (kH, kW) – Convolution kernel.

    Returns
    -------
    np.ndarray, shape (H, W), dtype uint8
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    padded = np.pad(
        image.astype(np.float64),
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="reflect",
    )

    h, w = image.shape
    result = np.zeros((h, w), dtype=np.float64)

    for i in range(kh):
        for j in range(kw):
            result += kernel[i, j] * padded[i : i + h, j : j + w]

    return np.clip(result, 0, 255).astype(np.uint8)


def preprocess(image: np.ndarray,
               smooth: bool = True,
               kernel_size: int = 3,
               sigma: float = 1.0) -> np.ndarray:
    """
    Full preprocessing pipeline: BGR → grayscale → (optional) Gaussian blur.

    Parameters
    ----------
    image       : BGR input image.
    smooth      : Apply Gaussian smoothing (reduces SIFT noise sensitivity).
    kernel_size : Gaussian kernel side length.
    sigma       : Gaussian σ.

    Returns
    -------
    Preprocessed grayscale image ready for SIFT.
    """
    gray = manual_grayscale(image)
    if smooth:
        gray = manual_convolve2d(gray, gaussian_kernel(kernel_size, sigma))
    return gray
