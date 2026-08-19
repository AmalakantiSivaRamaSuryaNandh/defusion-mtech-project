from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from defusion_mtech.image_io import align_pair, encode_png, load_rgb


def test_png_round_trip() -> None:
    source = np.zeros((20, 30, 3), dtype=np.float32)
    source[..., 1] = 0.5
    encoded = encode_png(source)
    loaded = load_rgb(BytesIO(encoded))
    assert loaded.shape == source.shape
    np.testing.assert_allclose(loaded, source, atol=1 / 255)
    assert Image.open(BytesIO(encoded)).format == "PNG"


def test_align_pair_strict_rejects_mismatch() -> None:
    image_a = np.zeros((20, 30, 3), dtype=np.float32)
    image_b = np.zeros((10, 15, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="sizes differ"):
        align_pair(image_a, image_b, "strict")


def test_align_pair_resize_and_crop() -> None:
    image_a = np.zeros((20, 30, 3), dtype=np.float32)
    image_b = np.ones((10, 40, 3), dtype=np.float32)
    resized_a, resized_b = align_pair(image_a, image_b, "resize_b_to_a")
    assert resized_a.shape == resized_b.shape == (20, 30, 3)
    cropped_a, cropped_b = align_pair(image_a, image_b, "center_crop")
    assert cropped_a.shape == cropped_b.shape == (10, 30, 3)
