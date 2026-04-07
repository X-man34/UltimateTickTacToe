"""
Board — the top-level 9x9 Ultimate Tic Tac Toe game state.

A Board is a 3x3 grid of SubBoards (so 81 cells total), plus whose turn it
is and which sub-boards are currently active. Mirrors `BoardState.java`.

The central rule of Ultimate Tic Tac Toe is the "send to" rule implemented
in `apply_move()`: after a player moves at local cell (minor_row, minor_col),
the opponent must play in the sub-board at global coordinates
(minor_row, minor_col). If that target sub-board is already terminal, every
non-terminal sub-board is activated instead, giving the opponent a free
choice.

The 90-element flat vector format exposed by `flat_vector()` /
`from_flat_vector()` matches the legacy ND4J .bin training data format
exactly, so old models can be retrained on the converted data without any
semantic drift. The indexing matches `Resources.getIndex()` in the Java
side: position = major_row*27 + major_col*9 + minor_row*3 + minor_col, with
activity flags appended as the last 9 values.
"""

from __future__ import annotations

import numpy as np

from uttt_engine.evaluation import Evaluation
from uttt_engine.marker import Marker
from uttt_engine.move import Move
from uttt_engine.sub_board import SubBoard

BOARD_SIZE = 3  # Each level (major and minor) is 3x3.
NUM_CELLS = BOARD_SIZE * BOARD_SIZE  # 9 sub-boards, each with 9 cells.
FLAT_CELL_COUNT = NUM_CELLS * NUM_CELLS  # 81 cells total.
FLAT_VECTOR_SIZE = FLAT_CELL_COUNT + NUM_CELLS  # 81 cells + 9 activity flags = 90.


def flat_index(major_row: int, major_col: int, minor_row: int, minor_col: int) -> int:
    """Row-major flat index for a cell in the 90-element vector.

    Mirrors `Resources.getIndex(majRow, majCol, minRow, minCol, 3)`
    in the Java code (Resources.java lines 41-45). The formula
    `major_row*27 + major_col*9 + minor_row*3 + minor_col` is what the
    legacy .bin training data uses, so we must match it byte-for-byte
    for old data to load into the new trainers.
    """
    major_idx = major_row * BOARD_SIZE + major_col
    minor_idx = minor_row * BOARD_SIZE + minor_col
    return major_idx * NUM_CELLS + minor_idx


def activity_index(major_row: int, major_col: int) -> int:
    """Index of a sub-board's activity flag inside the 90-element vector.

    The 9 activity flags live after the 81 cell values. Mirrors
    `Resources.getActivityIndex()` in the Java code. Indexing:
    81 + major_row*3 + major_col.
    """
    return FLAT_CELL_COUNT + major_row * BOARD_SIZE + major_col


class Board:
    """The full 9x9 Ultimate Tic Tac Toe game state.

    Attributes:
        sub_boards: 3x3 NumPy object array of SubBoard instances.
        player_one_turn: True if it is P1's (X) turn to move.
    """

    SIZE = BOARD_SIZE

    __slots__ = ("sub_boards", "player_one_turn")

    def __init__(
        self,
        sub_boards: np.ndarray | None = None,
        player_one_turn: bool = True,
    ) -> None:
        """Create a board.

        Args:
            sub_boards: 3x3 object array of SubBoards, or None to start
                from an empty board with all nine sub-boards active.
            player_one_turn: Whose turn it is. Default True (P1 moves first).
        """
        if sub_boards is None:
            grid = np.empty((self.SIZE, self.SIZE), dtype=object)
            for r in range(self.SIZE):
                for c in range(self.SIZE):
                    grid[r, c] = SubBoard(active=True)
            self.sub_boards = grid
        else:
            if sub_boards.shape != (self.SIZE, self.SIZE):
                raise ValueError(f"sub_boards must be shape ({self.SIZE}, {self.SIZE})")
            self.sub_boards = sub_boards
        self.player_one_turn = player_one_turn

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    def current_player(self) -> Marker:
        """Return the marker of the player whose turn it is."""
        return Marker.P1 if self.player_one_turn else Marker.P2

    def sub_board_at(self, major_row: int, major_col: int) -> SubBoard:
        """Return the sub-board at the given major coordinates."""
        return self.sub_boards[major_row, major_col]

    # ------------------------------------------------------------------
    # Legality and move generation
    # ------------------------------------------------------------------

    def is_legal(self, move: Move) -> bool:
        """Return True iff `move` is legal in the current state.

        Mirrors `BoardState.isLegal()` (BoardState.java lines 328-341).
        Checks:
          1. The marker matches whose turn it is.
          2. The target sub-board is active.
          3. The target cell is empty.
        """
        # Correct player's marker.
        if self.player_one_turn and move.marker != Marker.P1:
            return False
        if not self.player_one_turn and move.marker != Marker.P2:
            return False

        sub = self.sub_boards[move.major_row, move.major_col]
        if not sub.active:
            return False
        return sub.is_empty(move.minor_row, move.minor_col)

    def legal_moves(self) -> list[Move]:
        """Return every legal move for the current player.

        Mirrors `BoardState.getActions()` (BoardState.java lines 307-319):
        iterate over every sub-board and, if it is active, enumerate every
        empty cell as a candidate move.
        """
        marker = self.current_player()
        moves: list[Move] = []
        for major_row in range(self.SIZE):
            for major_col in range(self.SIZE):
                sub = self.sub_boards[major_row, major_col]
                if not sub.active:
                    continue
                for minor_row in range(self.SIZE):
                    for minor_col in range(self.SIZE):
                        if sub.is_empty(minor_row, minor_col):
                            moves.append(
                                Move(
                                    major_row=major_row,
                                    major_col=major_col,
                                    minor_row=minor_row,
                                    minor_col=minor_col,
                                    marker=marker,
                                )
                            )
        return moves

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def apply_move(self, move: Move) -> None:
        """Mutate this board by applying `move`.

        Mirrors `BoardState.preformAction()` (BoardState.java lines
        373-401). In order:
          1. Validate legality (raise on illegal).
          2. Place the marker on the target sub-board.
          3. Toggle whose turn it is.
          4. Deactivate every sub-board.
          5. If the sub-board at (minor_row, minor_col) is now terminal,
             reactivate every non-terminal sub-board instead of just one.
          6. Otherwise, activate only the target sub-board at
             (minor_row, minor_col) — the "send to" rule.

        Raises:
            ValueError: if the move is illegal.
        """
        if not self.is_legal(move):
            raise ValueError(f"Illegal move: {move}")

        target_sub = self.sub_boards[move.major_row, move.major_col]
        target_sub.apply(move.minor_row, move.minor_col, move.marker)

        self.player_one_turn = not self.player_one_turn

        # Deactivate everything; send-to or fallback activation follows.
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                self.sub_boards[r, c].active = False

        send_to = self.sub_boards[move.minor_row, move.minor_col]
        if send_to.is_terminal():
            # Target is already decided — opponent gets a free choice
            # among every non-terminal sub-board.
            for r in range(self.SIZE):
                for c in range(self.SIZE):
                    sub = self.sub_boards[r, c]
                    sub.active = not sub.is_terminal()
        else:
            send_to.active = True

    def simulate(self, move: Move) -> "Board":
        """Return a new Board with `move` applied; this board is unchanged.

        Mirrors `BoardState.simulateAction()`. Used by MCTS to create
        child node states without mutating the parent.
        """
        new_board = self.clone()
        new_board.apply_move(move)
        return new_board

    def clone(self) -> "Board":
        """Return a full deep copy.

        Every sub-board is cloned so the new Board owns its own cell
        arrays. This is the hot path for MCTS tree expansion — we rely
        on NumPy's cheap array copy for speed.
        """
        new_grid = np.empty((self.SIZE, self.SIZE), dtype=object)
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                new_grid[r, c] = self.sub_boards[r, c].clone()
        return Board(sub_boards=new_grid, player_one_turn=self.player_one_turn)

    def invert(self) -> None:
        """Flip every marker and toggle whose turn it is.

        Mirrors `BoardState.invertState()` (BoardState.java lines
        347-360). Used to normalize the board so the player-to-move is
        always represented as P1 — this is how MCTS and the training
        pipeline ensure a consistent perspective for the value network.
        """
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                self.sub_boards[r, c].invert()
        self.player_one_turn = not self.player_one_turn

    # ------------------------------------------------------------------
    # Terminal check and overall evaluation
    # ------------------------------------------------------------------

    def _meta_state(self) -> np.ndarray:
        """Return the 3x3 meta-board of sub-board evaluations.

        Used by `evaluation()` and `is_terminal()`. Each cell holds the
        current evaluation of the corresponding sub-board, so the meta
        board can be checked with the same 3-in-a-row logic that the
        sub-boards use. Draws (-0.25) are treated as blocking cells
        (non-1 and non-(-1)), which matches the Java convention.
        """
        meta = np.zeros((self.SIZE, self.SIZE), dtype=np.float64)
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                meta[r, c] = self.sub_boards[r, c].evaluation()
        return meta

    def evaluation(self) -> float:
        """Return the static evaluation of the whole game.

        Mirrors `BoardState.getEvaluation()` (BoardState.java lines
        294-298) — run the tic tac toe check on the 3x3 meta-board of
        sub-board evaluations.

        Returns:
            1.0 if P1 has won the overall game, -1.0 if P2 has won,
            -0.25 if the game is a draw (every sub-board terminal with
            no winning line), and 0.0 if still in progress.
        """
        meta = self._meta_state()

        # Rows and columns on the meta-board. A sub-board with eval 1
        # counts as P1's, eval -1 is P2's; any other value (0 or -0.25)
        # is not claimed by either player and cannot contribute to a line.
        for i in range(self.SIZE):
            # Row i: three sub-boards across.
            if meta[i, 0] != 0 and meta[i, 0] == meta[i, 1] == meta[i, 2]:
                if meta[i, 0] == 1:
                    return Evaluation.P1_WIN.value
                if meta[i, 0] == -1:
                    return Evaluation.P2_WIN.value
            # Column i.
            if meta[0, i] != 0 and meta[0, i] == meta[1, i] == meta[2, i]:
                if meta[0, i] == 1:
                    return Evaluation.P1_WIN.value
                if meta[0, i] == -1:
                    return Evaluation.P2_WIN.value

        # Diagonals.
        if meta[0, 0] != 0 and meta[0, 0] == meta[1, 1] == meta[2, 2]:
            if meta[0, 0] == 1:
                return Evaluation.P1_WIN.value
            if meta[0, 0] == -1:
                return Evaluation.P2_WIN.value
        if meta[0, 2] != 0 and meta[0, 2] == meta[1, 1] == meta[2, 0]:
            if meta[0, 2] == 1:
                return Evaluation.P1_WIN.value
            if meta[0, 2] == -1:
                return Evaluation.P2_WIN.value

        # No winning line found. If every sub-board is terminal this is a
        # draw; otherwise the game is still in progress.
        all_terminal = all(
            self.sub_boards[r, c].is_terminal()
            for r in range(self.SIZE)
            for c in range(self.SIZE)
        )
        return Evaluation.DRAW.value if all_terminal else Evaluation.IN_PROGRESS.value

    def is_terminal(self) -> bool:
        """True iff the game is over (someone has won or it is drawn)."""
        return Evaluation.is_terminal(self.evaluation())

    # ------------------------------------------------------------------
    # Flat vector serialization — 90-element format for ML pipelines
    # ------------------------------------------------------------------

    def flat_vector(self) -> np.ndarray:
        """Return the 90-element canonical serialization.

        Mirrors `BoardState.getBoardStateFlatVector()` (BoardState.java
        lines 241-255). The format is:

          - Indices 0..80: cell contents in row-major
            (major_row, major_col, minor_row, minor_col) order, with
            values {-1, 0, 1} from the perspective of the player to
            move. If it is P2's turn, every cell is negated so that
            the player-to-move is always represented by 1.
          - Indices 81..89: sub-board activity flags, encoded as
            +1 if active and -1 if inactive (NOT 0/1 — the Java
            comment notes this is for ML normalization, keeping all
            values in the [-1, 1] range).

        This is the exact layout used by the legacy .bin training
        data, so do not reorder without also updating the converter
        and all downstream dataset loaders.
        """
        vec = np.zeros(FLAT_VECTOR_SIZE, dtype=np.float64)
        perspective_sign = 1 if self.player_one_turn else -1
        for major_row in range(self.SIZE):
            for major_col in range(self.SIZE):
                sub = self.sub_boards[major_row, major_col]
                vec[activity_index(major_row, major_col)] = 1.0 if sub.active else -1.0
                for minor_row in range(self.SIZE):
                    for minor_col in range(self.SIZE):
                        idx = flat_index(major_row, major_col, minor_row, minor_col)
                        vec[idx] = perspective_sign * sub.item_at(minor_row, minor_col)
        return vec

    @classmethod
    def from_flat_vector(cls, vec: np.ndarray) -> "Board":
        """Reconstruct a Board from its 90-element flat vector.

        Inverse of `flat_vector()`. The reconstructed board is always
        returned in the player-to-move == P1 perspective because the
        flat vector itself is already normalized that way. Activity
        flags are decoded from the {+1, -1} encoding.

        Used by the training pipeline (uttt_ml.dataset.PolicyDataset)
        to rebuild a Board from legacy .bin data so the policy image
        encoder can be applied on the fly.
        """
        if vec.shape != (FLAT_VECTOR_SIZE,):
            raise ValueError(
                f"flat vector must have shape ({FLAT_VECTOR_SIZE},), got {vec.shape}"
            )

        board = cls()
        # The default constructor leaves everything active with empty cells.
        # Reset activity to False and fill cells/activity from the vector.
        for major_row in range(cls.SIZE):
            for major_col in range(cls.SIZE):
                sub = board.sub_boards[major_row, major_col]
                sub.active = vec[activity_index(major_row, major_col)] > 0
                for minor_row in range(cls.SIZE):
                    for minor_col in range(cls.SIZE):
                        idx = flat_index(major_row, major_col, minor_row, minor_col)
                        value = int(round(float(vec[idx])))
                        sub.cells[minor_row, minor_col] = value
                # Cells were mutated directly, so the cached eval is stale.
                sub._cached_eval = None

        board.player_one_turn = True
        return board

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        turn = "P1" if self.player_one_turn else "P2"
        active_flags = [
            (r, c)
            for r in range(self.SIZE)
            for c in range(self.SIZE)
            if self.sub_boards[r, c].active
        ]
        return f"Board(turn={turn}, active={active_flags})"
