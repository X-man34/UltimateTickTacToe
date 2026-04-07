"""
random_rollout — play a game to completion (or a depth cap) with random moves.

This is the default leaf-evaluation strategy when no value network is
loaded. Mirrors `MCTSEvaluator.preformRollout()` in the Java code
(MCTSEvaluator.java lines 429-448). Each move is drawn uniformly from
the legal moves of the current position; the sequence alternates
players because `apply_move` toggles whose turn it is.

The rollout does not mutate the input board — it clones first so the
MCTS tree's game states stay untouched.
"""

from __future__ import annotations

import random

from uttt_engine import Board


def random_rollout(
    board: Board,
    max_depth: int = 1000,
    rng: random.Random | None = None,
) -> float:
    """Return the terminal (or depth-capped) evaluation of a random playout.

    Args:
        board: Starting position. Cloned internally so the caller's
            state is preserved.
        max_depth: Safety cap on the number of moves to play. The Java
            default is 1000, which is much larger than any real UTTT
            game will ever take.
        rng: Optional pre-seeded `random.Random` instance for
            reproducibility. Defaults to the global random module, which
            is seeded by `random.seed` if the caller has configured it.

    Returns:
        The board's static evaluation once the playout ends: 1.0 for a
        P1 win, -1.0 for a P2 win, -0.25 for a draw, 0.0 if it bailed
        at max_depth with the game still in progress.
    """
    state = board.clone()
    turns = 0
    rng = rng or random
    while turns < max_depth:
        if state.is_terminal():
            break
        legal = state.legal_moves()
        if not legal:
            break  # shouldn't happen unless the game is terminal
        move = legal[rng.randrange(len(legal))]
        state.apply_move(move)
        turns += 1
    return state.evaluation()


__all__ = ["random_rollout"]
