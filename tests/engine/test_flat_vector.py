"""Round-trip and layout tests for the 90-element flat vector serialization.

These tests are critical for old-data compatibility: the legacy .bin
training files store boards in this exact format, so changing the
indexing would break legacy data loading.
"""

from __future__ import annotations

import numpy as np

from uttt_engine import Board, Marker, Move
from uttt_engine.board import (
    FLAT_CELL_COUNT,
    FLAT_VECTOR_SIZE,
    NUM_CELLS,
    activity_index,
    flat_index,
)


def test_vector_size_is_90() -> None:
    assert FLAT_VECTOR_SIZE == 90
    assert FLAT_CELL_COUNT == 81
    assert NUM_CELLS == 9


def test_flat_index_formula() -> None:
    # Matches Resources.getIndex in the Java code: row-major on majors
    # then on minors, total range 0..80 inclusive.
    assert flat_index(0, 0, 0, 0) == 0
    assert flat_index(0, 0, 0, 1) == 1
    assert flat_index(0, 0, 2, 2) == 8
    assert flat_index(0, 1, 0, 0) == 9
    assert flat_index(2, 2, 2, 2) == 80


def test_activity_index_formula() -> None:
    assert activity_index(0, 0) == 81
    assert activity_index(1, 1) == 85
    assert activity_index(2, 2) == 89


def test_empty_board_vector_is_all_zeros_for_cells_and_all_plus_one_for_activity() -> None:
    board = Board()
    vec = board.flat_vector()
    assert vec.shape == (90,)
    # Every cell is empty.
    assert np.all(vec[:81] == 0)
    # Every sub-board is active on a fresh board, encoded as +1.
    assert np.all(vec[81:] == 1)


def test_vector_reflects_player_perspective_inversion() -> None:
    """On P2's turn, every cell is negated so the player-to-move is always +1."""
    board = Board()
    board.apply_move(Move(0, 0, 0, 0, Marker.P1))
    # It's now P2's turn. The single marker we placed was P1 (value 1),
    # so from P2's perspective it must read as -1.
    vec = board.flat_vector()
    assert vec[flat_index(0, 0, 0, 0)] == -1
    # And the sub-board (0, 0) should be inactive because the send-to rule
    # sent P2 to sub-board (0, 0) itself — which is now active.
    assert vec[activity_index(0, 0)] == 1


def test_round_trip_empty() -> None:
    board = Board()
    vec = board.flat_vector()
    reconstructed = Board.from_flat_vector(vec)
    assert np.array_equal(reconstructed.flat_vector(), vec)


def test_round_trip_after_several_moves() -> None:
    board = Board()
    moves = [
        Move(1, 1, 0, 0, Marker.P1),
        Move(0, 0, 2, 2, Marker.P2),
        Move(2, 2, 1, 1, Marker.P1),
    ]
    for move in moves:
        board.apply_move(move)
    vec = board.flat_vector()
    reconstructed = Board.from_flat_vector(vec)
    # The reconstructed board is always in the player-one-to-move
    # perspective, so we can't compare player_one_turn directly. We
    # compare the serialized form, which is canonical.
    assert np.array_equal(reconstructed.flat_vector(), vec)
