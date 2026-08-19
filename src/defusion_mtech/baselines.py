"""Classical, deterministic image-fusion baselines."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pywt

from .image_io import validate_image


def _validate_pair(image_a: np.ndarray, image_b: np.ndarray) -> None:
    validate_image(image_a, "image_a")
    validate_image(image_b, "image_b")
    if image_a.shape != image_b.shape:
        raise ValueError(f"Input shapes must match: {image_a.shape} != {image_b.shape}")


def average_fusion(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    _validate_pair(image_a, image_b)
    return ((image_a + image_b) * 0.5).astype(np.float32)


def pca_fusion(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Fuse two images using global first-principal-component weights."""
    _validate_pair(image_a, image_b)
    samples = np.stack([image_a.reshape(-1), image_b.reshape(-1)])
    covariance = np.cov(samples)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    weights = np.abs(eigenvectors[:, int(np.argmax(eigenvalues))])
    if float(weights.sum()) <= 1e-12:
        weights = np.array([0.5, 0.5], dtype=np.float64)
    else:
        weights = weights / weights.sum()
    fused = weights[0] * image_a + weights[1] * image_b
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def laplacian_pyramid_fusion(
    image_a: np.ndarray,
    image_b: np.ndarray,
    levels: int = 4,
) -> np.ndarray:
    """Select high-energy Laplacian coefficients and average coarse content."""
    _validate_pair(image_a, image_b)
    if levels < 1:
        raise ValueError("levels must be at least 1")

    smallest_side = min(image_a.shape[:2])
    usable_levels = max(1, min(levels, int(np.floor(np.log2(smallest_side))) - 2))
    ga = [image_a]
    gb = [image_b]
    for _ in range(usable_levels):
        ga.append(cv2.pyrDown(ga[-1]))
        gb.append(cv2.pyrDown(gb[-1]))

    la = _laplacian_levels(ga)
    lb = _laplacian_levels(gb)
    fused_levels: list[np.ndarray] = []
    for index, (level_a, level_b) in enumerate(zip(la, lb, strict=True)):
        if index == len(la) - 1:
            fused_levels.append((level_a + level_b) * 0.5)
            continue
        energy_a = np.mean(np.abs(level_a), axis=2, keepdims=True)
        energy_b = np.mean(np.abs(level_b), axis=2, keepdims=True)
        fused_levels.append(np.where(energy_a >= energy_b, level_a, level_b))

    reconstructed = fused_levels[-1]
    for level in reversed(fused_levels[:-1]):
        reconstructed = cv2.pyrUp(reconstructed, dstsize=(level.shape[1], level.shape[0]))
        reconstructed = reconstructed + level
    return np.clip(reconstructed, 0.0, 1.0).astype(np.float32)


def _laplacian_levels(gaussian_levels: list[np.ndarray]) -> list[np.ndarray]:
    levels: list[np.ndarray] = []
    for index in range(len(gaussian_levels) - 1):
        current = gaussian_levels[index]
        expanded = cv2.pyrUp(
            gaussian_levels[index + 1], dstsize=(current.shape[1], current.shape[0])
        )
        levels.append(current - expanded)
    levels.append(gaussian_levels[-1])
    return levels


def wavelet_fusion(
    image_a: np.ndarray,
    image_b: np.ndarray,
    wavelet: str = "db2",
) -> np.ndarray:
    """Fuse approximation coefficients by mean and detail coefficients by magnitude."""
    _validate_pair(image_a, image_b)
    channels: list[np.ndarray] = []
    height, width = image_a.shape[:2]
    for channel in range(3):
        coeffs_a = pywt.dwt2(image_a[..., channel], wavelet)
        coeffs_b = pywt.dwt2(image_b[..., channel], wavelet)
        approx = (coeffs_a[0] + coeffs_b[0]) * 0.5
        details = tuple(
            np.where(np.abs(a) >= np.abs(b), a, b)
            for a, b in zip(coeffs_a[1], coeffs_b[1], strict=True)
        )
        reconstructed = pywt.idwt2((approx, details), wavelet)
        channels.append(reconstructed[:height, :width])
    return np.clip(np.stack(channels, axis=2), 0.0, 1.0).astype(np.float32)


def local_focus_fusion(
    image_a: np.ndarray,
    image_b: np.ndarray,
    blur_size: int = 9,
) -> np.ndarray:
    """Blend inputs using smoothed local Laplacian energy as a focus cue."""
    _validate_pair(image_a, image_b)
    if blur_size % 2 == 0 or blur_size < 3:
        raise ValueError("blur_size must be an odd integer of at least 3")
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY)
    energy_a = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray_a, cv2.CV_32F)), (blur_size,) * 2, 0)
    energy_b = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray_b, cv2.CV_32F)), (blur_size,) * 2, 0)
    weight_a = energy_a / (energy_a + energy_b + 1e-8)
    weight_a = np.clip(weight_a, 0.05, 0.95)[..., None]
    fused = weight_a * image_a + (1.0 - weight_a) * image_b
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


FUSION_METHODS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "average": average_fusion,
    "pca": pca_fusion,
    "laplacian": laplacian_pyramid_fusion,
    "wavelet": wavelet_fusion,
    "local_focus": local_focus_fusion,
}


def fuse(image_a: np.ndarray, image_b: np.ndarray, method: str) -> np.ndarray:
    try:
        operation = FUSION_METHODS[method]
    except KeyError as error:
        choices = ", ".join(sorted(FUSION_METHODS))
        raise ValueError(f"Unknown fusion method '{method}'. Choose from: {choices}") from error
    return operation(image_a, image_b)
