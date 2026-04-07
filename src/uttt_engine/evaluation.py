"""
Evaluation codes — the four possible results of a static position evaluation
for a 3x3 tic tac toe sub-board or the 3x3 meta-board of an Ultimate Tic Tac
Toe game.

Values match the Java `BoardState.Evaluation` enum exactly (BoardState.java
lines 131-161). The unusual `-0.25` value for DRAW is preserved on purpose —
the Java MCTS bot treats draws as slightly worse than indeterminate states
and we are committed to matching base behavior. See the plan's Known Risks
section for discussion.
"""

from __future__ import annotations

from enum import Enum


class Evaluation(float, Enum):
    """Terminal-state codes for a 3x3 board (sub-board or meta-board)."""

    IN_PROGRESS = 0.0
    DRAW = -0.25
    P1_WIN = 1.0
    P2_WIN = -1.0

    @classmethod
    def is_terminal(cls, value: float) -> bool:
        """True if the evaluation represents a finished sub-board."""
        return value != cls.IN_PROGRESS.value
