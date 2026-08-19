"""Reusable helpers for reproducible image-fusion evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a checkpoint or other experiment artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_records(
    records: list[dict[str, str | float]], methods: list[str]
) -> dict[str, dict[str, int | float]]:
    """Calculate pair counts, means, and sample standard deviations by method."""
    if not records:
        raise ValueError("At least one evaluation record is required")
    metric_fields = [field for field in records[0] if field not in {"pair", "method"}]
    summary: dict[str, dict[str, int | float]] = {}
    for method in methods:
        rows = [record for record in records if record["method"] == method]
        if not rows:
            raise ValueError(f"No evaluation records found for method: {method}")
        method_summary: dict[str, int | float] = {"pair_count": len(rows)}
        for field in metric_fields:
            values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            method_summary[f"{field}_mean"] = float(np.mean(values))
            method_summary[f"{field}_std"] = float(np.std(values, ddof=1 if len(values) > 1 else 0))
        summary[method] = method_summary
    return summary
