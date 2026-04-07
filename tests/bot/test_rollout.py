"""Tests for the random rollout helper."""

from __future__ import annotations

import random

from uttt_bot.rollout import random_rollout
from uttt_engine import Board, Evaluation


def test_rollout_terminates_on_fresh_board() -> None:
    """A rollout on a fresh board must reach a terminal state within depth budget."""
    board = Board()
    rng = random.Random(42)
    score = random_rollout(board, max_depth=200, rng=rng)
    # Score should be one of the terminal evaluations.
    valid = {
        Evaluation.P1_WIN.value,
        Evaluation.P2_WIN.value,
        Evaluation.DRAW.value,
    }
    assert score in valid, f"rollout returned {score}, expected one of {valid}"


def test_rollout_does_not_mutate_input_board() -> None:
    board = Board()
    snapshot = board.flat_vector().copy()
    random_rollout(board, max_depth=100, rng=random.Random(0))
    assert (board.flat_vector() == snapshot).all()


def test_rollout_with_max_depth_zero_returns_in_progress() -> None:
    """A zero-depth cap returns the board's current (in-progress) evaluation."""
    board = Board()
    score = random_rollout(board, max_depth=0, rng=random.Random(0))
    assert score == Evaluation.IN_PROGRESS.value
