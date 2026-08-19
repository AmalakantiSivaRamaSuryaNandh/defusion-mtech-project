from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from defusion_mtech.inference import fuse_with_model, load_model, save_checkpoint
from defusion_mtech.model import CUDNet, ModelConfig


def test_model_shapes_and_source_order_invariance() -> None:
    torch.manual_seed(2)
    model = CUDNet(ModelConfig(base_channels=8)).eval()
    source_a = torch.rand((1, 3, 32, 40))
    source_b = torch.rand((1, 3, 32, 40))
    with torch.inference_mode():
        forward = model(source_a, source_b)
        reverse = model(source_b, source_a)
    for name in ("common", "unique1", "unique2", "fused"):
        assert forward[name].shape == source_a.shape
        assert torch.all((0 <= forward[name]) & (forward[name] <= 1))
    torch.testing.assert_close(forward["fused"], reverse["fused"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(forward["common"], reverse["common"], atol=1e-6, rtol=1e-6)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = CUDNet(ModelConfig(base_channels=8)).eval()
    checkpoint = save_checkpoint(tmp_path / "model.pt", model, epoch=1)
    restored = load_model(checkpoint)
    for expected, actual in zip(
        model.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(expected, actual)


def test_model_rejects_non_multiple_of_eight() -> None:
    model = CUDNet(ModelConfig(base_channels=8))
    source = torch.rand((1, 3, 30, 32))
    try:
        model(source, source)
    except ValueError as error:
        assert "divisible by 8" in str(error)
    else:
        raise AssertionError("Expected a size validation error")


def test_array_inference_pads_and_crops() -> None:
    model = CUDNet(ModelConfig(base_channels=8)).eval()
    image_a = np.zeros((31, 35, 3), dtype=np.float32)
    image_b = np.ones((31, 35, 3), dtype=np.float32)
    fused, components = fuse_with_model(model, image_a, image_b)
    assert fused.shape == image_a.shape
    assert set(components) == {"common", "unique1", "unique2", "fused"}
