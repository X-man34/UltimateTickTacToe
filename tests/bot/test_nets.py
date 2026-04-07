"""Shape-sanity tests for the PyTorch value and policy networks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from uttt_bot.encoders import board_to_policy_image, board_to_value_tensor
from uttt_bot.nets import PolicyNet, ValueNet
from uttt_engine import Board


def test_value_net_forward_pass_shape() -> None:
    net = ValueNet()
    net.eval()
    batch = torch.zeros(2, 4, 9, 3, 3)
    with torch.no_grad():
        out = net(batch)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()


def test_value_net_encodes_board_correctly() -> None:
    net = ValueNet()
    net.eval()
    board = Board()
    sample = board_to_value_tensor(board)  # (9, 3, 3, 4) NDHWC
    batch = ValueNet.encode_batch([sample])
    assert batch.shape == (1, 4, 9, 3, 3)
    with torch.no_grad():
        out = net(batch)
    assert out.shape == (1, 1)


def test_policy_net_forward_pass_shape() -> None:
    net = PolicyNet()
    net.eval()
    batch = torch.zeros(2, 3, 76, 76)
    with torch.no_grad():
        out = net(batch)
    assert out.shape == (2, 81)
    assert torch.isfinite(out).all()


def test_policy_net_encodes_board_correctly() -> None:
    net = PolicyNet()
    net.eval()
    board = Board()
    sample = board_to_policy_image(board)  # (3, 76, 76)
    batch = PolicyNet.encode_batch([sample])
    assert batch.shape == (1, 3, 76, 76)
    with torch.no_grad():
        out = net(batch)
    assert out.shape == (1, 81)


def test_value_net_save_and_load_round_trip(tmp_path: Path) -> None:
    net = ValueNet()
    # Put some non-trivial weights so "loaded == original" is meaningful.
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1 * torch.randn_like(p))
    save_path = tmp_path / "value.pt"
    net.save(save_path)

    loaded = ValueNet.load(save_path, device="cpu")
    # Weights must match.
    for orig, copy in zip(net.parameters(), loaded.parameters(), strict=True):
        assert torch.allclose(orig.cpu(), copy.cpu())
    # Output must match on the same input.
    batch = torch.zeros(1, 4, 9, 3, 3)
    with torch.no_grad():
        assert torch.allclose(net(batch), loaded(batch))


def test_policy_net_save_and_load_round_trip(tmp_path: Path) -> None:
    net = PolicyNet()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1 * torch.randn_like(p))
    save_path = tmp_path / "policy.pt"
    net.save(save_path)

    loaded = PolicyNet.load(save_path, device="cpu")
    batch = torch.zeros(1, 3, 76, 76)
    with torch.no_grad():
        assert torch.allclose(net(batch), loaded(batch))


def test_value_net_output_finite_on_numpy_encoded_board() -> None:
    net = ValueNet()
    net.eval()
    board = Board()
    encoded = board_to_value_tensor(board)
    batch = ValueNet.encode_batch([encoded])
    with torch.no_grad():
        out = net(batch)
    assert np.isfinite(out.numpy()).all()
