"""MCTS behavioral tests — forced-win-in-1 and threading sanity."""

from __future__ import annotations

import time

import pytest

from uttt_bot.config import MCTSConfig
from uttt_bot.mcts import MCTSEvaluator
from uttt_engine import Board, Marker, Move


def _forced_win_in_one_board() -> tuple[Board, Move]:
    """Construct a position where P1 has a forced one-move win.

    Setup: P1 has already won sub-boards (0,0) and (0,1). P1 is about
    to play in sub-board (0,2). If P1 plays a move there that both (a)
    is legal and (b) completes a 3-in-a-row on the meta-board (row 0),
    P1 wins the game on the spot.

    We engineer the state so P1 is forced to play in sub-board (0,2)
    (via the send-to rule) and any move that wins (0,2) wins the game.
    To make the test deterministic we set (0,2) up so that P1 plays
    at (0,2)-local-(1,1) which directly completes a pre-placed
    horizontal two-in-a-row.
    """
    board = Board()

    # Mark sub-boards (0, 0) and (0, 1) as already won by P1.
    for mc in (0, 1):
        sub = board.sub_board_at(0, mc)
        sub.cells[0, 0] = 1
        sub.cells[0, 1] = 1
        sub.cells[0, 2] = 1
        sub._cached_eval = None

    # Sub-board (0, 2): two P1 markers already in the middle row, one
    # cell away from a sub-board win.
    sub = board.sub_board_at(0, 2)
    sub.cells[1, 0] = 1
    sub.cells[1, 2] = 1
    # Fill a few P2 cells so the sub-board isn't trivially empty
    # and the middle row completion is the only winning move in 1
    # that finishes both (0, 2) and therefore the meta-board.
    sub.cells[2, 0] = -1
    sub.cells[2, 2] = -1
    sub._cached_eval = None

    # Deactivate every other sub-board so MCTS has no choice but to
    # play at (0, 2), making the test deterministic regardless of
    # which specific move it considers first.
    for r in range(3):
        for c in range(3):
            board.sub_board_at(r, c).active = False
    board.sub_board_at(0, 2).active = True

    board.player_one_turn = True

    winning_move = Move(
        major_row=0, major_col=2, minor_row=1, minor_col=1, marker=Marker.P1
    )
    # Sanity-check the setup before running MCTS.
    assert board.is_legal(winning_move)
    preview = board.simulate(winning_move)
    assert preview.evaluation() == 1.0, (
        f"setup broken: winning move gives eval {preview.evaluation()}"
    )
    return board, winning_move


@pytest.mark.parametrize("num_threads", [1, 4])
def test_mcts_finds_forced_win_in_one(num_threads: int) -> None:
    """With plenty of time, MCTS must pick the move that wins now."""
    board, winning_move = _forced_win_in_one_board()
    config = MCTSConfig(
        compute_time_seconds=0.5,
        num_threads=num_threads,
        task_queue_depth=8,
        seed=123,
    )
    evaluator = MCTSEvaluator(config=config)
    result = evaluator.search(board)

    assert result.best_move.major_row == winning_move.major_row
    assert result.best_move.major_col == winning_move.major_col
    assert result.best_move.minor_row == winning_move.minor_row
    assert result.best_move.minor_col == winning_move.minor_col
    assert result.best_move.marker == Marker.P1


def test_mcts_respects_compute_time() -> None:
    """Search should spend close to the configured compute time."""
    board = Board()
    config = MCTSConfig(
        compute_time_seconds=0.3, num_threads=1, seed=0
    )
    evaluator = MCTSEvaluator(config=config)
    start = time.monotonic()
    evaluator.search(board)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.25, f"search ended too early: {elapsed:.3f}s"
    assert elapsed < 1.5, f"search ran too long: {elapsed:.3f}s"


def test_mcts_plays_legal_move_from_fresh_board() -> None:
    """Basic smoke test: MCTS returns a legal move from the starting position."""
    board = Board()
    config = MCTSConfig(compute_time_seconds=0.2, num_threads=1, seed=0)
    evaluator = MCTSEvaluator(config=config)
    result = evaluator.search(board)
    assert board.is_legal(result.best_move)
    assert result.num_iterations > 0


def test_mcts_handles_p2_turn() -> None:
    """MCTS must handle positions where it's P2's turn by inverting internally."""
    board = Board()
    board.apply_move(Move(1, 1, 1, 1, Marker.P1))
    # Now it's P2's turn. MCTS should still produce a legal P2 move.
    assert board.current_player() == Marker.P2
    config = MCTSConfig(compute_time_seconds=0.2, num_threads=1, seed=0)
    evaluator = MCTSEvaluator(config=config)
    result = evaluator.search(board)
    assert result.best_move.marker == Marker.P2
    assert board.is_legal(result.best_move)
