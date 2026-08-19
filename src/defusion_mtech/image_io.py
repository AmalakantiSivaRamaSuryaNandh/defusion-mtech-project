"""Image loading, alignment, conversion, and serialization helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

ImageSource = str | Path | BinaryIO


def load_rgb(source: ImageSource) -> np.ndarray:
    """Load an image as an H x W x 3 float32 array in the [0, 1] range."""
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array)


def validate_image(array: np.ndarray, name: str = "image") -> None:
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape H x W x 3; received {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must use a floating-point dtype")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if array.min() < 0.0 or array.max() > 1.0:
        raise ValueError(f"{name} values must be in the [0, 1] range")


def align_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    strategy: str = "resize_b_to_a",
) -> tuple[np.ndarray, np.ndarray]:
    """Make two RGB images share a spatial size using an explicit strategy.

    This function only handles image size. It does not perform geometric or
    cross-sensor registration, which should happen before fusion.
    """
    validate_image(image_a, "image_a")
    validate_image(image_b, "image_b")
    if image_a.shape == image_b.shape:
        return image_a, image_b

    if strategy == "strict":
        raise ValueError(
            f"Source image sizes differ: {image_a.shape[:2]} and {image_b.shape[:2]}"
        )

    if strategy == "resize_b_to_a":
        height, width = image_a.shape[:2]
        resized = Image.fromarray(to_uint8(image_b)).resize(
            (width, height), Image.Resampling.LANCZOS
        )
        return image_a, np.asarray(resized, dtype=np.float32) / 255.0

    if strategy == "center_crop":
        height = min(image_a.shape[0], image_b.shape[0])
        width = min(image_a.shape[1], image_b.shape[1])
        return _center_crop(image_a, height, width), _center_crop(image_b, height, width)

    raise ValueError(f"Unknown alignment strategy: {strategy}")


def _center_crop(array: np.ndarray, height: int, width: int) -> np.ndarray:
    top = (array.shape[0] - height) // 2
    left = (array.shape[1] - width) // 2
    return np.ascontiguousarray(array[top : top + height, left : left + width])


def to_uint8(array: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def encode_png(array: np.ndarray) -> bytes:
    validate_image(np.asarray(array, dtype=np.float32))
    buffer = BytesIO()
    Image.fromarray(to_uint8(array)).save(buffer, format="PNG")
    return buffer.getvalue()


def save_image(array: np.ndarray, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(array)).save(output)
    return output
