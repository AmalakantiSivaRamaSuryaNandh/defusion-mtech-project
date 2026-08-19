"""Fuse an aligned image pair and print transparent quality indicators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .baselines import FUSION_METHODS, fuse
from .image_io import align_pair, load_rgb, save_image
from .inference import fuse_with_model, load_model
from .metrics import fusion_metrics


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-a", type=Path, required=True)
    parser.add_argument("--image-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=[*sorted(FUSION_METHODS), "cud"], default="laplacian")
    parser.add_argument("--checkpoint", type=Path, help="Required for --method cud")
    parser.add_argument(
        "--alignment",
        choices=["strict", "resize_b_to_a", "center_crop"],
        default="strict",
    )
    parser.add_argument("--metrics-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    image_a, image_b = align_pair(load_rgb(args.image_a), load_rgb(args.image_b), args.alignment)
    if args.method == "cud":
        if args.checkpoint is None:
            raise SystemExit(
                "--checkpoint is required for CUD inference; random weights are not used"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(args.checkpoint, device=device)
        fused, _ = fuse_with_model(model, image_a, image_b)
    else:
        fused = fuse(image_a, image_b, args.method)
    save_image(fused, args.output)
    results = {"method": args.method, **fusion_metrics(image_a, image_b, fused)}
    print(json.dumps(results, indent=2))
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
