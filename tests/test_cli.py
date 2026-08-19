from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from defusion_mtech.cli import main


def _save(path: Path, value: int) -> None:
    Image.fromarray(np.full((24, 32, 3), value, dtype=np.uint8)).save(path)


def test_cli_fuses_pair_and_writes_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_a = tmp_path / "a.png"
    source_b = tmp_path / "b.png"
    output = tmp_path / "result.png"
    metrics = tmp_path / "metrics.json"
    _save(source_a, 40)
    _save(source_b, 200)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "defusion-fuse",
            "--image-a",
            str(source_a),
            "--image-b",
            str(source_b),
            "--output",
            str(output),
            "--method",
            "average",
            "--metrics-json",
            str(metrics),
        ],
    )

    main()

    assert output.is_file()
    result = json.loads(metrics.read_text(encoding="utf-8"))
    assert result["method"] == "average"
    assert "source_ssim_proxy" in result


def test_cli_requires_checkpoint_for_cud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_a = tmp_path / "a.png"
    source_b = tmp_path / "b.png"
    _save(source_a, 40)
    _save(source_b, 200)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "defusion-fuse",
            "--image-a",
            str(source_a),
            "--image-b",
            str(source_b),
            "--output",
            str(tmp_path / "result.png"),
            "--method",
            "cud",
        ],
    )

    with pytest.raises(SystemExit, match="--checkpoint is required"):
        main()
