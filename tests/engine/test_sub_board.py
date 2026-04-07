"""Tests for SubBoard evaluation and basic state transitions."""

from __future__ import annotations

import numpy as np

from uttt_engine import Evaluation, Marker, SubBoard


def _set(sub: SubBoard, cells: list[list[int]]) -> None:
    """Helper: overwrite a sub-board's cells from a 3x3 python list."""
    sub.cells = np.array(cells, dtype=np.int8)
    sub._cached_eval = None


def test_empty_board_is_in_progress() -> None:
    sub = SubBoard()
    assert sub.evaluation() == Evaluation.IN_PROGRESS.value
    assert not sub.is_terminal()


def test_row_wins() -> None:
    # Every row as a P1 win, then as a P2 win.
    for row in range(3):
        sub = SubBoard()
        cells = [[0] * 3 for _ in range(3)]
        cells[row] = [1, 1, 1]
        _set(sub, cells)
        assert sub.evaluation() == Evaluation.P1_WIN.value, f"P1 row {row}"

    for row in range(3):
        sub = SubBoard()
        cells = [[0] * 3 for _ in range(3)]
        cells[row] = [-1, -1, -1]
        _set(sub, cells)
        assert sub.evaluation() == Evaluation.P2_WIN.value, f"P2 row {row}"


def test_column_wins() -> None:
    for col in range(3):
        sub = SubBoard()
        cells = [[0] * 3 for _ in range(3)]
        for r in range(3):
            cells[r][col] = 1
        _set(sub, cells)
        assert sub.evaluation() == Evaluation.P1_WIN.value, f"P1 col {col}"


def test_diagonal_wins() -> None:
    # Main diagonal.
    sub = SubBoard()
    _set(sub, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert sub.evaluation() == Evaluation.P1_WIN.value

    # Anti-diagonal.
    sub = SubBoard()
    _set(sub, [[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
    assert sub.evaluation() == Evaluation.P2_WIN.value


def test_full_board_draw() -> None:
    sub = SubBoard()
    # Classic drawn position: no row/col/diagonal completes.
    _set(sub, [[1, -1, 1], [1, -1, -1], [-1, 1, 1]])
    assert sub.evaluation() == Evaluation.DRAW.value
    assert sub.is_terminal()


def test_apply_invalidates_cached_eval() -> None:
    sub = SubBoard()
    assert sub.evaluation() == Evaluation.IN_PROGRESS.value
    sub.apply(0, 0, Marker.P1)
    sub.apply(0, 1, Marker.P1)
    sub.apply(0, 2, Marker.P1)
    assert sub.evaluation() == Evaluation.P1_WIN.value


def test_invert_flips_winner() -> None:
    sub = SubBoard()
    _set(sub, [[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    assert sub.evaluation() == Evaluation.P1_WIN.value
    sub.invert()
    assert sub.evaluation() == Evaluation.P2_WIN.value


def test_clone_is_independent() -> None:
    sub = SubBoard()
    sub.apply(1, 1, Marker.P1)
    twin = sub.clone()
    twin.apply(0, 0, Marker.P2)
    assert sub.item_at(0, 0) == 0
    assert twin.item_at(0, 0) == -1
