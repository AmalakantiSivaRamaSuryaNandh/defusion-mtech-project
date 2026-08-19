from __future__ import annotations

import torch

from defusion_mtech.cud import CUDLoss, make_cud_targets


def test_cud_targets_cover_and_partition_scene() -> None:
    generator = torch.Generator().manual_seed(4)
    image = torch.rand((2, 3, 32, 40), generator=generator)
    targets = make_cud_targets(image, mask_grid=4, generator=generator)
    assert torch.all((targets.mask1 + targets.mask2) > 0)
    reconstructed_partition = targets.common + targets.unique1 + targets.unique2
    torch.testing.assert_close(reconstructed_partition, image)
    assert targets.view1.shape == image.shape
    assert targets.view2.shape == image.shape


def test_cud_loss_is_zero_for_exact_predictions() -> None:
    image = torch.rand((1, 3, 32, 32))
    targets = make_cud_targets(image, mask_grid=4)
    predictions = {
        "common": targets.common,
        "unique1": targets.unique1,
        "unique2": targets.unique2,
        "fused": targets.reconstruction,
    }
    losses = CUDLoss()(predictions, targets)
    assert float(losses["total"]) == 0.0
