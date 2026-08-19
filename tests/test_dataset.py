from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from defusion_mtech.dataset import UnlabeledImageDataset, discover_images


def _write_image(path: Path, width: int = 48, height: int = 40) -> None:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[..., 1] = 128
    Image.fromarray(array).save(path)


def test_discover_and_load_training_images(tmp_path: Path) -> None:
    _write_image(tmp_path / "b.png")
    _write_image(tmp_path / "a.jpg")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    assert [path.name for path in discover_images(tmp_path, max_images=1)] == ["a.jpg"]
    dataset = UnlabeledImageDataset(tmp_path, crop_size=32)
    sample = dataset[0]
    assert sample.shape == (3, 32, 32)
    assert sample.dtype.is_floating_point
    assert 0.0 <= float(sample.min()) <= float(sample.max()) <= 1.0


def test_dataset_rejects_missing_or_invalid_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_images(tmp_path / "missing")
    with pytest.raises(ValueError, match="No supported images"):
        discover_images(tmp_path)
    with pytest.raises(ValueError, match="crop_size"):
        UnlabeledImageDataset(tmp_path, crop_size=30)
