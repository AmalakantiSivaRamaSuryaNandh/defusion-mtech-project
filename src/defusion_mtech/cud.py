"""Common and Unique Decomposition (CUD) self-supervision utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class CUDTargets:
    view1: Tensor
    view2: Tensor
    common: Tensor
    unique1: Tensor
    unique2: Tensor
    reconstruction: Tensor
    mask1: Tensor
    mask2: Tensor


def make_cud_targets(
    image: Tensor,
    mask_grid: int = 8,
    noise_std: float = 0.25,
    generator: torch.Generator | None = None,
) -> CUDTargets:
    """Create two corrupted views and exact common/unique targets.

    The two binary masks are constrained so their union covers every pixel,
    matching the information-preservation condition in the DeFusion paper.
    """
    if image.ndim != 4:
        raise ValueError(f"image must have shape B x C x H x W; received {image.shape}")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("image contains NaN or infinite values")
    minimum = float(image.min())
    maximum = float(image.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError("image values must be in the [0, 1] range")
    if mask_grid < 2:
        raise ValueError("mask_grid must be at least 2")

    batch, _, height, width = image.shape
    grid_height = min(mask_grid, height)
    grid_width = min(mask_grid, width)
    shape = (batch, 1, grid_height, grid_width)
    mask1_low = torch.rand(shape, device=image.device, generator=generator) > 0.35
    mask2_low = torch.rand(shape, device=image.device, generator=generator) > 0.35

    uncovered = ~(mask1_low | mask2_low)
    assignment = torch.rand(shape, device=image.device, generator=generator) > 0.5
    mask1_low = mask1_low | (uncovered & assignment)
    mask2_low = mask2_low | (uncovered & ~assignment)

    mask1 = F.interpolate(mask1_low.float(), size=(height, width), mode="nearest")
    mask2 = F.interpolate(mask2_low.float(), size=(height, width), mode="nearest")
    if not bool(torch.all((mask1 + mask2) > 0)):
        raise RuntimeError("CUD mask coverage invariant was violated")

    noise1 = torch.randn(image.shape, device=image.device, generator=generator) * noise_std + 0.5
    noise2 = torch.randn(image.shape, device=image.device, generator=generator) * noise_std + 0.5
    noise1 = noise1.clamp(0.0, 1.0)
    noise2 = noise2.clamp(0.0, 1.0)

    view1 = image * mask1 + noise1 * (1.0 - mask1)
    view2 = image * mask2 + noise2 * (1.0 - mask2)
    common_mask = mask1 * mask2
    unique1_mask = mask1 * (1.0 - mask2)
    unique2_mask = (1.0 - mask1) * mask2
    return CUDTargets(
        view1=view1,
        view2=view2,
        common=image * common_mask,
        unique1=image * unique1_mask,
        unique2=image * unique2_mask,
        reconstruction=image,
        mask1=mask1,
        mask2=mask2,
    )


class CUDLoss(nn.Module):
    """Mean absolute error over three decompositions and reconstruction."""

    def __init__(self, decomposition_weight: float = 1.0, reconstruction_weight: float = 1.0):
        super().__init__()
        self.decomposition_weight = decomposition_weight
        self.reconstruction_weight = reconstruction_weight

    def forward(self, predictions: dict[str, Tensor], targets: CUDTargets) -> dict[str, Tensor]:
        common = F.l1_loss(predictions["common"], targets.common)
        unique1 = F.l1_loss(predictions["unique1"], targets.unique1)
        unique2 = F.l1_loss(predictions["unique2"], targets.unique2)
        reconstruction = F.l1_loss(predictions["fused"], targets.reconstruction)
        decomposition = common + unique1 + unique2
        total = (
            self.decomposition_weight * decomposition
            + self.reconstruction_weight * reconstruction
        )
        return {
            "total": total,
            "decomposition": decomposition,
            "reconstruction": reconstruction,
            "common": common,
            "unique1": unique1,
            "unique2": unique2,
        }
