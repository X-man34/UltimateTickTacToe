"""Tests for the value-tensor and policy-image encoders."""

from __future__ import annotations

import numpy as np

from uttt_bot.encoders import (
    CHANNEL_ACTIVE,
    CHANNEL_EMPTY,
    CHANNEL_O,
    CHANNEL_X,
    POLICY_IMAGE_SIZE,
    VALUE_TENSOR_SHAPE,
    board_to_policy_image,
    board_to_value_tensor,
    value_vector_to_tensor,
)
from uttt_engine import Board, Marker, Move


def test_value_tensor_shape() -> None:
    board = Board()
    tensor = board_to_value_tensor(board)
    assert tensor.shape == VALUE_TENSOR_SHAPE
    assert tensor.dtype == np.float32


def test_empty_board_value_tensor_channels() -> None:
    """An empty board: empty channel hot everywhere, activity hot, X/O cold."""
    board = Board()
    tensor = board_to_value_tensor(board)
    # Empty channel: everywhere 1.
    assert np.all(tensor[..., CHANNEL_EMPTY] == 1.0)
    # Active channel: everywhere 1 on a fresh board.
    assert np.all(tensor[..., CHANNEL_ACTIVE] == 1.0)
    # X and O channels: everywhere 0.
    assert np.all(tensor[..., CHANNEL_X] == 0.0)
    assert np.all(tensor[..., CHANNEL_O] == 0.0)


def test_value_tensor_reflects_marker_placement() -> None:
    board = Board()
    # P1 plays at global (1, 1), local (2, 2). After the move, sub-board
    # (2, 2) is the only active one due to the send-to rule.
    board.apply_move(Move(1, 1, 2, 2, Marker.P1))

    tensor = board_to_value_tensor(board)
    sub_idx_1_1 = 1 * 3 + 1
    sub_idx_2_2 = 2 * 3 + 2

    # The P1 marker at global (1,1) local (2,2) is X and no longer empty.
    assert tensor[sub_idx_1_1, 2, 2, CHANNEL_X] == 1.0
    assert tensor[sub_idx_1_1, 2, 2, CHANNEL_EMPTY] == 0.0
    # Activity: only sub-board (2, 2) is active.
    assert np.all(tensor[sub_idx_2_2, :, :, CHANNEL_ACTIVE] == 1.0)
    assert np.all(tensor[sub_idx_1_1, :, :, CHANNEL_ACTIVE] == 0.0)


def test_value_vector_to_tensor_matches_board_to_value_tensor() -> None:
    """The fast reshape path and the Board path must agree on the same position."""
    board = Board()
    board.apply_move(Move(0, 0, 1, 1, Marker.P1))
    board.apply_move(Move(1, 1, 2, 2, Marker.P2))

    direct = board_to_value_tensor(board)
    via_vec = value_vector_to_tensor(board.flat_vector())

    # The flat vector normalizes so player-to-move == P1 (inverts if
    # it's P2's turn). After the two moves above it's P1's turn again,
    # so the vector is not inverted and the two paths must match.
    assert np.array_equal(direct, via_vec)


def test_policy_image_shape_and_range() -> None:
    board = Board()
    image = board_to_policy_image(board)
    assert image.shape == (3, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)
    assert image.dtype == np.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_policy_image_empty_board_has_blue_active_backgrounds() -> None:
    """On a fresh board every sub-board is active, so blue pixels should appear."""
    board = Board()
    image = board_to_policy_image(board)
    # Blue channel should be 1.0 somewhere.
    assert (image[2] == 1.0).any()
    # Red pixels exist only for P1 markers or P1-won boards. An empty
    # fresh board has no P1 markers.
    p1_marker_pixels = (
        (image[0] == 1.0) & (image[1] == 0.0) & (image[2] == 0.0)
    )
    assert not p1_marker_pixels.any()


def test_policy_image_has_p1_marker_as_red() -> None:
    board = Board()
    board.apply_move(Move(0, 0, 0, 0, Marker.P1))
    image = board_to_policy_image(board)
    # At least one pure-red pixel exists now (a marker).
    p1_marker_pixels = (
        (image[0] == 1.0) & (image[1] == 0.0) & (image[2] == 0.0)
    )
    assert p1_marker_pixels.any()


def test_policy_image_has_p2_marker_as_green() -> None:
    board = Board()
    board.apply_move(Move(0, 0, 0, 0, Marker.P1))
    board.apply_move(Move(0, 0, 1, 1, Marker.P2))
    image = board_to_policy_image(board)
    p2_marker_pixels = (
        (image[0] == 0.0) & (image[1] == 1.0) & (image[2] == 0.0)
    )
    assert p2_marker_pixels.any()
