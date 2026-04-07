"""
uttt_engine — pure Python game rules for Ultimate Tic Tac Toe.

This package has zero dependencies on PyTorch, PySide6, or any ML tooling.
It can be extracted standalone for use in future projects. Only the
standard library and NumPy are required.

The core abstractions are:
    - Marker: enum for P1, P2, and EMPTY cells (numeric codes 1, -1, 0)
    - Move: immutable action (major/minor row/col + marker)
    - SubBoard: a single 3x3 tic tac toe board with an active flag
    - Board: the 3x3 grid of SubBoards that make up Ultimate Tic Tac Toe,
      including the "send to" rule and meta-board win detection
    - Evaluation: terminal state codes (in_progress, draw, p1_win, p2_win)
"""

from uttt_engine.board import Board, activity_index, flat_index
from uttt_engine.evaluation import Evaluation
from uttt_engine.marker import Marker
from uttt_engine.move import Move
from uttt_engine.sub_board import SubBoard

__all__ = [
    "Board",
    "Evaluation",
    "Marker",
    "Move",
    "SubBoard",
    "activity_index",
    "flat_index",
]
