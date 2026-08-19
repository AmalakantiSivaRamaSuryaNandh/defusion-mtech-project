"""Minimal unlabeled image dataset for CUD pretraining."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_images(root: str | Path, max_images: int | None = None) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"Training image directory does not exist: {directory}")
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise ValueError(f"No supported images found under {directory}")
    return paths


class UnlabeledImageDataset(Dataset[Tensor]):
    def __init__(self, root: str | Path, crop_size: int = 256, max_images: int | None = None):
        if crop_size < 32 or crop_size % 8:
            raise ValueError("crop_size must be at least 32 and divisible by 8")
        self.paths = discover_images(root, max_images=max_images)
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        with Image.open(self.paths[index]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = self._resize_and_crop(image)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))

    def _resize_and_crop(self, image: Image.Image) -> Image.Image:
        scale = max(self.crop_size / image.width, self.crop_size / image.height)
        if scale > 1.0:
            image = image.resize(
                (int(round(image.width * scale)), int(round(image.height * scale))),
                Image.Resampling.BICUBIC,
            )
        max_left = image.width - self.crop_size
        max_top = image.height - self.crop_size
        left = random.randint(0, max_left) if max_left else 0
        top = random.randint(0, max_top) if max_top else 0
        return image.crop((left, top, left + self.crop_size, top + self.crop_size))
