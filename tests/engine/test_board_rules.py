"""Tests for Board legality, move generation, the send-to rule, and overall wins."""

from __future__ import annotations

import pytest

from uttt_engine import Board, Evaluation, Marker, Move


def test_initial_position_has_81_legal_moves() -> None:
    """A fresh board has every cell available."""
    board = Board()
    assert len(board.legal_moves()) == 81
    assert board.player_one_turn is True
    assert board.current_player() == Marker.P1


def test_illegal_move_raises() -> None:
    board = Board()
    # Wrong marker (P2 moving first).
    with pytest.raises(ValueError):
        board.apply_move(Move(0, 0, 0, 0, Marker.P2))


def test_send_to_rule_single_target() -> None:
    """After P1 plays at local (1, 2), only sub-board (1, 2) stays active."""
    board = Board()
    board.apply_move(Move(0, 0, 1, 2, Marker.P1))
    # Exactly one sub-board active, and it's at (1, 2).
    active = [
        (r, c)
        for r in range(3)
        for c in range(3)
        if board.sub_board_at(r, c).active
    ]
    assert active == [(1, 2)]
    # All legal moves must be within that sub-board.
    legal = board.legal_moves()
    assert len(legal) == 9  # the sub-board is empty, 9 cells free
    for move in legal:
        assert (move.major_row, move.major_col) == (1, 2)
        assert move.marker == Marker.P2


def test_send_to_rule_across_multiple_moves() -> None:
    """Follow the send-to chain for three moves and verify it stays consistent."""
    board = Board()
    board.apply_move(Move(0, 0, 1, 2, Marker.P1))  # -> sends to (1, 2)
    board.apply_move(Move(1, 2, 0, 1, Marker.P2))  # -> sends to (0, 1)
    active = [
        (r, c)
        for r in range(3)
        for c in range(3)
        if board.sub_board_at(r, c).active
    ]
    assert active == [(0, 1)]
    assert board.current_player() == Marker.P1


def test_send_to_terminal_board_activates_all_non_terminal() -> None:
    """If the target sub-board is already won, every non-terminal board becomes active."""
    board = Board()

    # Force sub-board (1, 1) to be a P1 win by hand.
    sub = board.sub_board_at(1, 1)
    sub.cells[0, 0] = 1
    sub.cells[0, 1] = 1
    sub.cells[0, 2] = 1
    sub._cached_eval = None
    assert sub.evaluation() == Evaluation.P1_WIN.value

    # P1 plays at local (1, 1) in some other sub-board.
    # Normally this would send P2 to the (1, 1) sub-board, but since
    # (1, 1) is already terminal the rule says every non-terminal
    # sub-board activates.
    board.apply_move(Move(0, 0, 1, 1, Marker.P1))

    active_non_terminal = [
        (r, c)
        for r in range(3)
        for c in range(3)
        if board.sub_board_at(r, c).active
    ]
    # All eight non-won sub-boards should be active.
    assert len(active_non_terminal) == 8
    assert (1, 1) not in active_non_terminal
    # (1, 1) is terminal and must stay inactive.
    assert board.sub_board_at(1, 1).active is False


def test_meta_win_detection_row() -> None:
    """If P1 wins three sub-boards in the top row, the overall game is a P1 win."""
    board = Board()
    for c in range(3):
        sub = board.sub_board_at(0, c)
        sub.cells[0, 0] = 1
        sub.cells[0, 1] = 1
        sub.cells[0, 2] = 1
        sub._cached_eval = None
    assert board.evaluation() == Evaluation.P1_WIN.value
    assert board.is_terminal()


def test_meta_draw_when_all_sub_boards_terminal_no_line() -> None:
    """A position where every sub-board is drawn counts as a meta-draw."""
    draw_cells = [[1, -1, 1], [1, -1, -1], [-1, 1, 1]]
    board = Board()
    for r in range(3):
        for c in range(3):
            sub = board.sub_board_at(r, c)
            for i in range(3):
                for j in range(3):
                    sub.cells[i, j] = draw_cells[i][j]
            sub._cached_eval = None
    assert board.evaluation() == Evaluation.DRAW.value
    assert board.is_terminal()


def test_in_progress_mid_game() -> None:
    board = Board()
    board.apply_move(Move(0, 0, 1, 1, Marker.P1))
    assert board.evaluation() == Evaluation.IN_PROGRESS.value
    assert not board.is_terminal()


def test_simulate_does_not_mutate_original() -> None:
    board = Board()
    move = Move(0, 0, 1, 1, Marker.P1)
    new_board = board.simulate(move)
    assert board.sub_board_at(0, 0).item_at(1, 1) == 0
    assert new_board.sub_board_at(0, 0).item_at(1, 1) == 1
    assert board.player_one_turn is True
    assert new_board.player_one_turn is False


def test_clone_is_fully_independent() -> None:
    board = Board()
    board.apply_move(Move(0, 0, 1, 1, Marker.P1))
    clone = board.clone()
    clone.apply_move(Move(1, 1, 0, 0, Marker.P2))
    # The original should be unchanged.
    assert board.sub_board_at(1, 1).item_at(0, 0) == 0
    assert clone.sub_board_at(1, 1).item_at(0, 0) == -1
