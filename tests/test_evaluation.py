from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluate_pairs import sha256_file, summarize_records


def test_evaluation_summary_includes_count_mean_and_sample_std() -> None:
    records: list[dict[str, str | float]] = [
        {"pair": "a.png", "method": "average", "score": 1.0},
        {"pair": "b.png", "method": "average", "score": 3.0},
    ]
    summary = summarize_records(records, ["average"])["average"]
    assert summary["pair_count"] == 2
    assert summary["score_mean"] == pytest.approx(2.0)
    assert summary["score_std"] == pytest.approx(2**0.5)


def test_checkpoint_hash_is_reproducible(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"trusted checkpoint test")
    assert sha256_file(checkpoint) == sha256_file(checkpoint)
    assert len(sha256_file(checkpoint)) == 64
