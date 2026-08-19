"""Evaluate fusion methods on filename-matched source-pair folders."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import torch

from defusion_mtech.baselines import FUSION_METHODS, fuse
from defusion_mtech.dataset import SUPPORTED_EXTENSIONS
from defusion_mtech.evaluation import sha256_file, summarize_records
from defusion_mtech.image_io import align_pair, load_rgb, save_image
from defusion_mtech.inference import fuse_with_model, load_model
from defusion_mtech.metrics import fusion_metrics
from defusion_mtech.model import CUDNet

ALL_METHODS = [*sorted(FUSION_METHODS), "cud"]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-a", type=Path, required=True)
    parser.add_argument("--source-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=sorted(FUSION_METHODS),
    )
    parser.add_argument("--checkpoint", type=Path, help="Required when 'cud' is evaluated")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--alignment", choices=["strict", "resize_b_to_a", "center_crop"], default="strict"
    )
    parser.add_argument(
        "--task",
        choices=["multi-focus", "multi-exposure", "infrared-visible", "general"],
        default="general",
    )
    parser.add_argument("--dataset-name", default="not-recorded")
    parser.add_argument("--commit", default="not-recorded")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = arguments()
    for directory, label in ((args.source_a, "source A"), (args.source_b, "source B")):
        if not directory.is_dir():
            raise SystemExit(f"The {label} directory does not exist: {directory}")

    device = resolve_device(args.device)
    model: CUDNet | None = None
    checkpoint_hash: str | None = None
    if "cud" in args.methods:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required when evaluating the CUD method")
        if not args.checkpoint.is_file():
            raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")
        model = load_model(args.checkpoint, device=device)
        checkpoint_hash = sha256_file(args.checkpoint)

    files_a = {
        path.name: path
        for path in args.source_a.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    files_b = {
        path.name: path
        for path in args.source_b.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    names = sorted(files_a.keys() & files_b.keys())
    if not names:
        raise SystemExit("No filename-matched source pairs were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | float]] = []
    for name in names:
        image_a, image_b = align_pair(
            load_rgb(files_a[name]), load_rgb(files_b[name]), args.alignment
        )
        for method in args.methods:
            started = time.perf_counter()
            if method == "cud":
                if model is None:
                    raise RuntimeError("CUD model was not initialized")
                fused, _ = fuse_with_model(model, image_a, image_b, device=device)
            else:
                fused = fuse(image_a, image_b, method)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            output = args.output_dir / method / f"{Path(name).stem}.png"
            save_image(fused, output)
            records.append(
                {
                    "pair": name,
                    "method": method,
                    **fusion_metrics(image_a, image_b, fused),
                    "elapsed_ms": elapsed_ms,
                }
            )

    fieldnames = list(records[0].keys())
    metrics_path = args.output_dir / "per_pair_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    document = {
        "metadata": {
            "task": args.task,
            "dataset_name": args.dataset_name,
            "pair_count": len(names),
            "methods": args.methods,
            "alignment": args.alignment,
            "commit": args.commit,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "checkpoint_sha256": checkpoint_hash,
            "device": str(device),
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "source_a": str(args.source_a.resolve()),
            "source_b": str(args.source_b.resolve()),
        },
        "methods": summarize_records(records, args.methods),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
