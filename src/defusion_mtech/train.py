"""Command-line CUD self-supervised pretraining."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cud import CUDLoss, make_cud_targets
from .dataset import UnlabeledImageDataset
from .inference import save_checkpoint
from .model import CUDNet, ModelConfig


@dataclass
class TrainConfig:
    data_dir: str = "data/coco/train2017"
    output_dir: str = "runs/cud"
    crop_size: int = 256
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    base_channels: int = 32
    num_workers: int = 0
    seed: int = 42
    max_images: int | None = 50000
    mixed_precision: bool = False


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--crop-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lr-step-size", type=int)
    parser.add_argument("--lr-gamma", type=float)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def load_config(arguments: argparse.Namespace) -> TrainConfig:
    values: dict[str, object] = {}
    if arguments.config:
        with arguments.config.open("r", encoding="utf-8") as stream:
            values.update(json.load(stream))
    for field in fields(TrainConfig):
        value = getattr(arguments, field.name)
        if value is not None:
            values[field.name] = value

    integer_fields = {
        "crop_size",
        "epochs",
        "batch_size",
        "lr_step_size",
        "base_channels",
        "num_workers",
        "seed",
        "max_images",
    }
    float_fields = {"learning_rate", "lr_gamma"}
    for name in integer_fields & values.keys():
        values[name] = None if values[name] in {None, "none", "None"} else int(values[name])
    for name in float_fields & values.keys():
        values[name] = float(values[name])
    return TrainConfig(**values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config: TrainConfig) -> None:
    """Reject settings that cannot produce a valid trained checkpoint."""
    if config.crop_size < 32 or config.crop_size % 8:
        raise ValueError("crop_size must be at least 32 and divisible by 8")
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1; zero epochs would save untrained weights")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if config.lr_step_size < 1:
        raise ValueError("lr_step_size must be at least 1")
    if config.lr_gamma <= 0:
        raise ValueError("lr_gamma must be greater than 0")
    if config.base_channels < 1:
        raise ValueError("base_channels must be at least 1")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if config.max_images is not None and config.max_images < 1:
        raise ValueError("max_images must be at least 1 or None")
    if config.seed < 0:
        raise ValueError("seed cannot be negative")


def train(config: TrainConfig) -> Path:
    validate_config(config)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as stream:
        json.dump(asdict(config), stream, indent=2)

    dataset = UnlabeledImageDataset(
        config.data_dir,
        crop_size=config.crop_size,
        max_images=config.max_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= config.batch_size,
    )
    model = CUDNet(ModelConfig(base_channels=config.base_channels)).to(device)
    loss_function = CUDLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
    )
    amp_enabled = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history_path = output_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")

    for epoch in range(1, config.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        started = time.perf_counter()
        for images in loader:
            images = images.to(device, non_blocking=True)
            targets = make_cud_targets(images)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predictions = model(targets.view1, targets.view2)
                losses = loss_function(predictions, targets)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1
        scheduler.step()

        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{name: value / max(batches, 1) for name, value in totals.items()},
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        save_checkpoint(
            output_dir / "latest.pt",
            model,
            epoch=epoch,
            optimizer=optimizer,
            metadata={"train_config": asdict(config)},
        )

    final_path = output_dir / "final.pt"
    save_checkpoint(
        final_path,
        model,
        epoch=config.epochs,
        optimizer=optimizer,
        metadata={"train_config": asdict(config)},
    )
    return final_path


def main() -> None:
    config = load_config(_parse_arguments())
    checkpoint = train(config)
    print(f"Saved final checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
