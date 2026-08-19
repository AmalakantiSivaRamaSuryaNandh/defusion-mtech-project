"""Checkpoint loading and array-based inference for CUDNet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .image_io import validate_image
from .model import CUDNet, ModelConfig


def save_checkpoint(
    path: str | Path,
    model: CUDNet,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "format_version": 1,
        "epoch": epoch,
        "model_config": model.config_dict(),
        "model_state": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, output)
    return output


def load_model(path: str | Path, device: str | torch.device = "cpu") -> CUDNet:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError("Unsupported checkpoint format")
    config = ModelConfig(**payload.get("model_config", {}))
    model = CUDNet(config)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model


def _array_to_tensor(array: np.ndarray, device: torch.device) -> Tensor:
    validate_image(array)
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


@torch.inference_mode()
def fuse_with_model(
    model: CUDNet,
    image_a: np.ndarray,
    image_b: np.ndarray,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    validate_image(image_a, "image_a")
    validate_image(image_b, "image_b")
    if image_a.shape != image_b.shape:
        raise ValueError("CUD inference requires aligned images with matching shapes")
    model_device = next(model.parameters()).device
    requested_device = torch.device(device) if device is not None else model_device
    if requested_device != model_device:
        model = model.to(requested_device)

    tensor_a = _array_to_tensor(image_a, requested_device)
    tensor_b = _array_to_tensor(image_b, requested_device)
    height, width = image_a.shape[:2]
    pad_height = (-height) % 8
    pad_width = (-width) % 8
    if pad_height or pad_width:
        tensor_a = F.pad(tensor_a, (0, pad_width, 0, pad_height), mode="reflect")
        tensor_b = F.pad(tensor_b, (0, pad_width, 0, pad_height), mode="reflect")

    predictions = model(tensor_a, tensor_b)
    outputs = {
        name: _tensor_to_array(prediction, height, width)
        for name, prediction in predictions.items()
        if name in {"fused", "common", "unique1", "unique2"}
    }
    return outputs["fused"], outputs


def _tensor_to_array(tensor: Tensor, height: int, width: int) -> np.ndarray:
    array = tensor[0, :, :height, :width].detach().float().cpu().numpy().transpose(1, 2, 0)
    return np.clip(array, 0.0, 1.0).astype(np.float32)
