from __future__ import annotations

import numpy as np

from defusion_mtech.metrics import fusion_metrics


def test_metrics_are_finite_and_named_transparently() -> None:
    rng = np.random.default_rng(3)
    source_a = rng.random((32, 32, 3), dtype=np.float32)
    source_b = rng.random((32, 32, 3), dtype=np.float32)
    fused = (source_a + source_b) / 2
    metrics = fusion_metrics(source_a, source_b, fused)
    assert set(metrics) == {
        "entropy_bits",
        "spatial_frequency",
        "mutual_information_sum",
        "source_ssim_proxy",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0 <= metrics["source_ssim_proxy"] <= 1
