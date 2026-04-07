"""Tests that Board.invert() round-trips and preserves game semantics."""

from __future__ import annotations

from uttt_engine import Board, Evaluation, Marker, Move


def test_double_invert_is_identity() -> None:
    board = Board()
    board.apply_move(Move(0, 0, 1, 2, Marker.P1))
    board.apply_move(Move(1, 2, 0, 0, Marker.P2))
    before = board.flat_vector().copy()
    turn_before = board.player_one_turn

    board.invert()
    board.invert()

    assert board.player_one_turn == turn_before
    assert (board.flat_vector() == before).all()


def test_invert_swaps_winner() -> None:
    """If P1 has won the game, inverting flips it to a P2 win."""
    board = Board()
    for c in range(3):
        sub = board.sub_board_at(0, c)
        sub.cells[0, 0] = 1
        sub.cells[0, 1] = 1
        sub.cells[0, 2] = 1
        sub._cached_eval = None
    assert board.evaluation() == Evaluation.P1_WIN.value

    board.invert()
    assert board.evaluation() == Evaluation.P2_WIN.value


def test_invert_toggles_turn() -> None:
    board = Board()
    assert board.player_one_turn is True
    board.invert()
    assert board.player_one_turn is False
