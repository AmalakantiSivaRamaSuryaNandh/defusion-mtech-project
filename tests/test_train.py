from __future__ import annotations

from dataclasses import replace

import pytest

from defusion_mtech.train import TrainConfig, validate_config


def test_default_training_configuration_is_valid() -> None:
    validate_config(TrainConfig())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("crop_size", 30, "crop_size"),
        ("epochs", 0, "epochs"),
        ("batch_size", 0, "batch_size"),
        ("learning_rate", 0.0, "learning_rate"),
        ("lr_step_size", 0, "lr_step_size"),
        ("lr_gamma", 0.0, "lr_gamma"),
        ("base_channels", 0, "base_channels"),
        ("num_workers", -1, "num_workers"),
        ("max_images", 0, "max_images"),
        ("seed", -1, "seed"),
    ],
)
def test_invalid_training_configuration_is_rejected(
    field: str, value: int | float, message: str
) -> None:
    config = replace(TrainConfig(), **{field: value})
    with pytest.raises(ValueError, match=message):
        validate_config(config)
