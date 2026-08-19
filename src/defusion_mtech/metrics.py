"""Clearly named image-fusion quality indicators."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from .image_io import validate_image


def _gray(image: np.ndarray) -> np.ndarray:
    return (0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]).astype(
        np.float32
    )


def entropy(image: np.ndarray, bins: int = 256) -> float:
    gray = _gray(image)
    histogram, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64)
    probabilities /= max(float(probabilities.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def spatial_frequency(image: np.ndarray) -> float:
    gray = _gray(image)
    row_frequency = np.sqrt(np.mean(np.diff(gray, axis=0) ** 2)) if gray.shape[0] > 1 else 0.0
    col_frequency = np.sqrt(np.mean(np.diff(gray, axis=1) ** 2)) if gray.shape[1] > 1 else 0.0
    return float(np.sqrt(row_frequency**2 + col_frequency**2))


def mutual_information(image_a: np.ndarray, image_b: np.ndarray, bins: int = 64) -> float:
    gray_a = _gray(image_a).reshape(-1)
    gray_b = _gray(image_b).reshape(-1)
    joint, _, _ = np.histogram2d(gray_a, gray_b, bins=bins, range=((0, 1), (0, 1)))
    joint = joint / max(float(joint.sum()), 1.0)
    marginal_a = joint.sum(axis=1)
    marginal_b = joint.sum(axis=0)
    independent = marginal_a[:, None] * marginal_b[None, :]
    valid = (joint > 0) & (independent > 0)
    return float((joint[valid] * np.log2(joint[valid] / independent[valid])).sum())


def source_ssim_proxy(source_a: np.ndarray, source_b: np.ndarray, fused: np.ndarray) -> float:
    """Average SSIM to the two sources; this is not ground-truth SSIM."""
    gray_fused = _gray(fused)
    score_a = structural_similarity(_gray(source_a), gray_fused, data_range=1.0)
    score_b = structural_similarity(_gray(source_b), gray_fused, data_range=1.0)
    return float((score_a + score_b) * 0.5)


def fusion_metrics(
    source_a: np.ndarray,
    source_b: np.ndarray,
    fused: np.ndarray,
) -> dict[str, float]:
    for name, array in (("source_a", source_a), ("source_b", source_b), ("fused", fused)):
        validate_image(array, name)
    if source_a.shape != source_b.shape or source_a.shape != fused.shape:
        raise ValueError("All images must share the same shape")
    return {
        "entropy_bits": entropy(fused),
        "spatial_frequency": spatial_frequency(fused),
        "mutual_information_sum": mutual_information(source_a, fused)
        + mutual_information(source_b, fused),
        "source_ssim_proxy": source_ssim_proxy(source_a, source_b, fused),
    }
