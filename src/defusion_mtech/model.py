"""A compact, order-invariant CUD-inspired neural fusion architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        widths = (base_channels, base_channels * 2, base_channels * 4)
        layers: list[nn.Module] = []
        current = in_channels
        for width in widths:
            layers.extend(
                [
                    nn.Conv2d(current, width, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(_groups(width), width),
                    nn.SiLU(inplace=True),
                    ResidualBlock(width),
                ]
            )
            current = width
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, out_channels: int):
        super().__init__()
        widths = (base_channels * 4, base_channels * 2, base_channels)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, widths[0], kernel_size=3, padding=1, bias=False),
            ResidualBlock(widths[0]),
        ]
        for current, following in zip(widths, widths[1:], strict=False):
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(current, following, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(_groups(following), following),
                    nn.SiLU(inplace=True),
                    ResidualBlock(following),
                ]
            )
        layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(widths[-1], out_channels, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    base_channels: int = 32


class CUDNet(nn.Module):
    """Decompose a source pair into common/unique features and reconstruct a fusion.

    This is an independent educational implementation inspired by DeFusion. It
    intentionally uses order-invariant common and reconstruction inputs so
    swapping the two source images produces the same fused output.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        base = self.config.base_channels
        bottleneck = base * 4
        self.encoder = Encoder(self.config.in_channels, base)
        self.ensembler = nn.Sequential(
            nn.Conv2d(bottleneck * 2, bottleneck, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(bottleneck), bottleneck),
            nn.SiLU(inplace=True),
            ResidualBlock(bottleneck),
        )
        self.common_decoder = UpsampleDecoder(bottleneck, base, base)
        self.unique_decoder = UpsampleDecoder(bottleneck * 2, base, base)
        self.common_projector = nn.Conv2d(base, self.config.in_channels, kernel_size=1)
        self.unique_projector = nn.Conv2d(base, self.config.in_channels, kernel_size=1)
        self.reconstruction_projector = nn.Sequential(
            nn.Conv2d(base * 2, base, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            ResidualBlock(base),
            nn.Conv2d(base, self.config.in_channels, kernel_size=1),
        )

    def forward(self, source1: Tensor, source2: Tensor) -> dict[str, Tensor]:
        self._validate_inputs(source1, source2)
        encoded1 = self.encoder(source1)
        encoded2 = self.encoder(source2)
        mean = (encoded1 + encoded2) * 0.5
        difference = torch.abs(encoded1 - encoded2)
        common_latent = self.ensembler(torch.cat([mean, difference], dim=1))
        common_features = self.common_decoder(common_latent)
        unique1_features = self.unique_decoder(torch.cat([encoded1, common_latent], dim=1))
        unique2_features = self.unique_decoder(torch.cat([encoded2, common_latent], dim=1))
        unique_sum = unique1_features + unique2_features
        return {
            "common": torch.sigmoid(self.common_projector(common_features)),
            "unique1": torch.sigmoid(self.unique_projector(unique1_features)),
            "unique2": torch.sigmoid(self.unique_projector(unique2_features)),
            "fused": torch.sigmoid(
                self.reconstruction_projector(torch.cat([common_features, unique_sum], dim=1))
            ),
            "common_features": common_features,
            "unique1_features": unique1_features,
            "unique2_features": unique2_features,
        }

    def config_dict(self) -> dict[str, int]:
        return asdict(self.config)

    @staticmethod
    def _validate_inputs(source1: Tensor, source2: Tensor) -> None:
        if source1.shape != source2.shape:
            raise ValueError(
                f"Source tensors must have matching shapes: {source1.shape} != {source2.shape}"
            )
        if source1.ndim != 4:
            raise ValueError("Source tensors must have shape B x C x H x W")
        if source1.shape[-2] % 8 or source1.shape[-1] % 8:
            raise ValueError("Image height and width must both be divisible by 8")
