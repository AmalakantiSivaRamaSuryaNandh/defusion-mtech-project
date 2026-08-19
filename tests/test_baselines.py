from __future__ import annotations

import numpy as np
import pytest

from defusion_mtech.baselines import FUSION_METHODS, average_fusion, fuse, pca_fusion


@pytest.fixture
def pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    return (
        rng.random((32, 40, 3), dtype=np.float32),
        rng.random((32, 40, 3), dtype=np.float32),
    )


@pytest.mark.parametrize("method", sorted(FUSION_METHODS))
def test_all_baselines_return_valid_rgb(pair: tuple[np.ndarray, np.ndarray], method: str) -> None:
    output = fuse(*pair, method=method)
    assert output.shape == pair[0].shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0


def test_average_is_pixel_mean(pair: tuple[np.ndarray, np.ndarray]) -> None:
    np.testing.assert_allclose(average_fusion(*pair), (pair[0] + pair[1]) / 2, atol=1e-7)


def test_pca_is_invariant_to_source_order(pair: tuple[np.ndarray, np.ndarray]) -> None:
    np.testing.assert_allclose(pca_fusion(*pair), pca_fusion(pair[1], pair[0]), atol=1e-6)


def test_unknown_method_is_rejected(pair: tuple[np.ndarray, np.ndarray]) -> None:
    with pytest.raises(ValueError, match="Unknown fusion method"):
        fuse(*pair, method="not-a-method")
