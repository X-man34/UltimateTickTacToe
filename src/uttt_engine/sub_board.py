"""
SubBoard — a single 3x3 tic tac toe board inside Ultimate Tic Tac Toe.

Mirrors `SubBoardState.java`. The evaluation logic is ported directly from
`BoardState.getTicTacToeEvaluationBruteForce()` (BoardState.java lines
131-161) so sub-board wins and the meta-board win check share the exact
same 3x3 line-scanning algorithm.

An `active` flag controls whether moves may currently be played on this
sub-board. The send-to rule in `Board.apply_move()` is what flips this flag
on and off during a game.
"""

from __future__ import annotations

import numpy as np

from uttt_engine.evaluation import Evaluation
from uttt_engine.marker import Marker


class SubBoard:
    """A 3x3 board with cells, an active flag, and a cached evaluation.

    The cell grid is stored as a NumPy int8 array of shape (3, 3) with
    values in {-1, 0, 1}. Using NumPy rather than a Python list gives us
    cheap deep-copy via `.copy()` and efficient flat-vector serialization.
    """

    SIZE = 3

    __slots__ = ("cells", "active", "_cached_eval")

    def __init__(
        self,
        cells: np.ndarray | None = None,
        active: bool = True,
    ) -> None:
        """Create a sub-board.

        Args:
            cells: Initial cell contents as an int8 (3, 3) array, or None to
                start empty. If provided, the caller's array is copied so
                the new SubBoard owns its own data.
            active: Whether the sub-board is currently playable. Defaults
                to True; the Board class manages activation via the
                send-to rule.
        """
        if cells is None:
            self.cells = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)
        else:
            if cells.shape != (self.SIZE, self.SIZE):
                raise ValueError(f"cells must be shape ({self.SIZE}, {self.SIZE})")
            self.cells = cells.astype(np.int8, copy=True)
        self.active = active
        self._cached_eval: float | None = None

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    def item_at(self, row: int, col: int) -> int:
        """Return the value (as an int) at a cell."""
        return int(self.cells[row, col])

    def is_empty(self, row: int, col: int) -> bool:
        """True iff the cell is unoccupied."""
        return self.cells[row, col] == Marker.EMPTY

    def apply(self, row: int, col: int, marker: Marker) -> None:
        """Place a marker at (row, col), invalidating the cached evaluation.

        Does not enforce legality — the calling Board has already checked
        that the cell is empty and that the sub-board is active. Mirrors
        the trust-the-caller pattern from the Java side.
        """
        self.cells[row, col] = int(marker)
        self._cached_eval = None

    def invert(self) -> None:
        """Flip every non-empty marker to the opposing player.

        Mirrors `SubBoardState.invertState()`. Used during MCTS rollouts
        and when building training data from the perspective of the
        player about to move. Invalidates the cached evaluation because
        a win for P1 becomes a win for P2 (and vice versa).
        """
        np.negative(self.cells, out=self.cells)
        self._cached_eval = None

    def clone(self) -> "SubBoard":
        """Return a deep copy with its own cell array.

        Copies the cached evaluation — safe because it's a float, not a
        reference into the cells array.
        """
        new = SubBoard(cells=self.cells, active=self.active)
        new._cached_eval = self._cached_eval
        return new

    # ------------------------------------------------------------------
    # Evaluation — ported from BoardState.getTicTacToeEvaluationBruteForce
    # ------------------------------------------------------------------

    def evaluation(self) -> float:
        """Return the terminal code for this sub-board.

        Returns one of the Evaluation values: IN_PROGRESS (0.0), DRAW
        (-0.25), P1_WIN (1.0), or P2_WIN (-1.0). The result is cached
        across calls until the sub-board is mutated.

        Ported from BoardState.java lines 131-161 — same loop order and
        same three-step check: rows, columns, both diagonals, then fill
        check for draw vs. in-progress.
        """
        if self._cached_eval is not None:
            return self._cached_eval

        s = self.cells  # shorthand — avoids retyping self.cells below
        result: float

        # Rows and columns: in the Java loop, state[i][0] == state[i][1] ==
        # state[i][2] checks row i. We do the same with NumPy indexing.
        for i in range(self.SIZE):
            if s[i, 0] != 0 and s[i, 0] == s[i, 1] == s[i, 2]:
                result = (
                    Evaluation.P1_WIN.value if s[i, 0] == 1 else Evaluation.P2_WIN.value
                )
                self._cached_eval = result
                return result
            if s[0, i] != 0 and s[0, i] == s[1, i] == s[2, i]:
                result = (
                    Evaluation.P1_WIN.value if s[0, i] == 1 else Evaluation.P2_WIN.value
                )
                self._cached_eval = result
                return result

        # Diagonals.
        if s[0, 0] != 0 and s[0, 0] == s[1, 1] == s[2, 2]:
            result = Evaluation.P1_WIN.value if s[0, 0] == 1 else Evaluation.P2_WIN.value
            self._cached_eval = result
            return result
        if s[0, 2] != 0 and s[0, 2] == s[1, 1] == s[2, 0]:
            result = Evaluation.P1_WIN.value if s[0, 2] == 1 else Evaluation.P2_WIN.value
            self._cached_eval = result
            return result

        # No winner — either a draw (full board) or still in progress.
        if np.count_nonzero(s) == self.SIZE * self.SIZE:
            result = Evaluation.DRAW.value
        else:
            result = Evaluation.IN_PROGRESS.value
        self._cached_eval = result
        return result

    def is_terminal(self) -> bool:
        """True if the sub-board is won or drawn (no more legal moves here)."""
        return Evaluation.is_terminal(self.evaluation())

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        flag = "*" if self.active else " "
        return f"SubBoard({flag}, {self.cells.tolist()})"
