"""Evaluate methods on filename-matched source-pair folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from defusion_mtech.baselines import FUSION_METHODS, fuse
from defusion_mtech.image_io import align_pair, load_rgb, save_image
from defusion_mtech.metrics import fusion_metrics


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-a", type=Path, required=True)
    parser.add_argument("--source-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(FUSION_METHODS),
        default=sorted(FUSION_METHODS),
    )
    parser.add_argument(
        "--alignment", choices=["strict", "resize_b_to_a", "center_crop"], default="strict"
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    files_a = {path.name: path for path in args.source_a.iterdir() if path.is_file()}
    files_b = {path.name: path for path in args.source_b.iterdir() if path.is_file()}
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
            fused = fuse(image_a, image_b, method)
            output = args.output_dir / method / f"{Path(name).stem}.png"
            save_image(fused, output)
            records.append(
                {"pair": name, "method": method, **fusion_metrics(image_a, image_b, fused)}
            )

    fieldnames = list(records[0].keys())
    metrics_path = args.output_dir / "per_pair_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary: dict[str, dict[str, float]] = {}
    for method in args.methods:
        rows = [record for record in records if record["method"] == method]
        summary[method] = {
            field: float(np.mean([float(row[field]) for row in rows]))
            for field in fieldnames
            if field not in {"pair", "method"}
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
