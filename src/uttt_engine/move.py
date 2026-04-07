"""
Move — an immutable action on an Ultimate Tic Tac Toe board.

A move is fully specified by the major (outer) board coordinates, the minor
(inner) cell coordinates within that board, and the marker being placed.
Mirrors `UltimateTickTacToeGameAction.java`, but uses explicit row/col
fields throughout instead of the Java base class's ambiguous `x`/`y` names.

Moves are frozen dataclasses so they can be used as dictionary keys and as
elements of sets. MCTS uses move equality to match parent-child transitions.
"""

from __future__ import annotations

from dataclasses import dataclass

from uttt_engine.marker import Marker


@dataclass(frozen=True, slots=True)
class Move:
    """A single placement on the 9x9 Ultimate Tic Tac Toe board.

    Attributes:
        major_row: Row index (0-2) of the sub-board being played on.
        major_col: Column index (0-2) of the sub-board being played on.
        minor_row: Row index (0-2) of the cell within that sub-board.
        minor_col: Column index (0-2) of the cell within that sub-board.
        marker: Which player is making the move. Must be P1 or P2, never EMPTY.
    """

    major_row: int
    major_col: int
    minor_row: int
    minor_col: int
    marker: Marker

    def __post_init__(self) -> None:
        if self.marker == Marker.EMPTY:
            raise ValueError("A Move must place a non-empty marker.")

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        return (
            f"Move(major=({self.major_row},{self.major_col}), "
            f"minor=({self.minor_row},{self.minor_col}), "
            f"marker={self.marker.name})"
        )
