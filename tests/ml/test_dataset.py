"""Tests for ValueDataset and PolicyDataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from uttt_ml.dataset import PolicyDataset, ValueDataset

COMBINED_VALUE = Path(
    "/home/calebh/claude-willow/uttt/data/series1_combined_value.npz"
)
COMBINED_POLICY = Path(
    "/home/calebh/claude-willow/uttt/data/series1_combined_policy.npz"
)

pytestmark = pytest.mark.skipif(
    not (COMBINED_VALUE.exists() and COMBINED_POLICY.exists()),
    reason="converted .npz fixtures not available; run convert_bin_to_npz first",
)


def test_value_dataset_yields_correct_shapes() -> None:
    ds = ValueDataset(COMBINED_VALUE)
    assert len(ds) > 0
    features, label = ds[0]
    assert isinstance(features, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert features.shape == (4, 9, 3, 3), features.shape
    assert features.dtype == torch.float32
    assert label.shape == (1,)
    assert label.dtype == torch.float32


def test_value_dataset_channel_semantics() -> None:
    """Exactly one of (X, O, empty) should be hot per cell."""
    ds = ValueDataset(COMBINED_VALUE)
    features, _ = ds[0]
    one_hot_sum = features[:3].sum(dim=0)  # sum across X, O, empty channels
    assert torch.allclose(one_hot_sum, torch.ones_like(one_hot_sum)), (
        f"expected exactly one hot of {{X, O, empty}} per cell, got "
        f"sum={one_hot_sum}"
    )


def test_policy_dataset_yields_correct_shapes() -> None:
    ds = PolicyDataset(COMBINED_POLICY)
    assert len(ds) > 0
    image, label = ds[0]
    assert image.shape == (3, 76, 76), image.shape
    assert image.dtype == torch.float32
    assert label.shape == (81,)
    assert label.dtype == torch.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0


def test_policy_dataset_label_sums_match_visit_ratios() -> None:
    """A random sample of policy labels should mostly sum to ~1."""
    ds = PolicyDataset(COMBINED_POLICY)
    rng = np.random.default_rng(42)
    sample = rng.choice(len(ds), size=min(50, len(ds)), replace=False)
    sums = []
    for i in sample:
        _, label = ds[int(i)]
        sums.append(float(label.sum().item()))
    sums_arr = np.array(sums)
    close = np.sum(np.isclose(sums_arr, 1.0, atol=1e-3))
    assert close > len(sums) * 0.5, (
        f"expected most policy labels to sum to ~1, got close={close}/{len(sums)}"
    )
