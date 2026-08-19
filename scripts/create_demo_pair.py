"""Create a synthetic multi-focus-like pair for smoke testing the application."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/demo_pair"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    height, width = 256, 384
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 0] = xx / width
    base[..., 1] = yy / height
    base[..., 2] = 0.3 + 0.2 * np.sin(xx / 15.0)
    cv2.circle(base, (95, 128), 55, (1.0, 0.85, 0.1), thickness=-1)
    cv2.rectangle(base, (240, 72), (340, 185), (0.1, 0.9, 0.8), thickness=-1)
    blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=7.0)
    source_a = np.concatenate([base[:, : width // 2], blurred[:, width // 2 :]], axis=1)
    source_b = np.concatenate([blurred[:, : width // 2], base[:, width // 2 :]], axis=1)
    for name, image in (("source_a.png", source_a), ("source_b.png", source_b)):
        Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(
            args.output_dir / name
        )
    print(f"Created demo pair in {args.output_dir}")


if __name__ == "__main__":
    main()
