"""
BoardScene — QGraphicsScene rendering an Ultimate Tic Tac Toe board.

The scene draws a 9x9 board at any resolution (as opposed to the Java
side's fixed-pixel BufferedImage rendering). Cells, sub-boards, and
meta-board overlays are all drawn with QPainter primitives, so zooming
the view stays crisp.

Click handling: mouse presses on a cell emit `move_selected(Move)`
with the clicked cell's coordinates. The game view wires this signal
to its game loop.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene

from uttt_engine import Board, Marker, Move

# --- Color palette. Tweak here if you want to re-skin the board. ---
COLOR_BG = QColor("#1e1f24")
COLOR_GRID = QColor("#e8e8ef")
COLOR_GRID_MAJOR = QColor("#f5f5fa")
COLOR_ACTIVE_TINT = QColor(84, 101, 255, 45)  # translucent indigo
COLOR_P1 = QColor("#ff4c4c")
COLOR_P2 = QColor("#3fd07d")
COLOR_WON_P1 = QColor(255, 76, 76, 60)
COLOR_WON_P2 = QColor(63, 208, 125, 60)
COLOR_DRAW = QColor(245, 200, 114, 60)

# --- Layout constants in scene units (pixels at 1x zoom). ---
BOARD_SIZE = 600.0         # total scene width/height
MAJOR_LINE_WIDTH = 6.0
MINOR_LINE_WIDTH = 2.0


class BoardScene(QGraphicsScene):
    """A QGraphicsScene that renders a live `Board` and accepts clicks."""

    #: Emitted when the user clicks a legal-looking cell. The game
    #: view will re-check legality and apply the move if valid.
    move_selected = Signal(object)  # Move

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(COLOR_BG))
        self.setSceneRect(0, 0, BOARD_SIZE, BOARD_SIZE)
        self._board: Board | None = None
        self._accept_clicks = False

    # ------------------------------------------------------------------
    # Board plumbing
    # ------------------------------------------------------------------

    def set_board(self, board: Board) -> None:
        """Swap in a new board to render, and force a redraw."""
        self._board = board
        self.update()

    def set_accept_clicks(self, accept: bool) -> None:
        """Turn user input on or off (e.g. off while the bot is thinking)."""
        self._accept_clicks = accept

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        """Paint the entire board. Called by Qt every time the scene needs redraw."""
        super().drawBackground(painter, rect)
        painter.fillRect(rect, COLOR_BG)

        if self._board is None:
            return

        painter.setRenderHint(QPainter.Antialiasing)

        sub_size = BOARD_SIZE / 3.0
        cell_size = sub_size / 3.0

        # Draw sub-board backgrounds: tint active ones, color terminal ones.
        for major_row in range(3):
            for major_col in range(3):
                sub_rect = QRectF(
                    major_col * sub_size,
                    major_row * sub_size,
                    sub_size,
                    sub_size,
                )
                sub = self._board.sub_board_at(major_row, major_col)
                evaluation = sub.evaluation()
                if evaluation == 1.0:
                    painter.fillRect(sub_rect, COLOR_WON_P1)
                    self._draw_big_marker(
                        painter, sub_rect, COLOR_P1, "X"
                    )
                elif evaluation == -1.0:
                    painter.fillRect(sub_rect, COLOR_WON_P2)
                    self._draw_big_marker(
                        painter, sub_rect, COLOR_P2, "O"
                    )
                elif evaluation == -0.25:
                    painter.fillRect(sub_rect, COLOR_DRAW)
                elif sub.active:
                    painter.fillRect(sub_rect, COLOR_ACTIVE_TINT)

                # Draw the individual cell markers for in-progress sub-boards.
                if evaluation == 0.0:
                    self._draw_cells(
                        painter, sub, sub_rect, cell_size
                    )

        # Draw the minor grid lines (3x3 per sub-board).
        minor_pen = QPen(COLOR_GRID, MINOR_LINE_WIDTH)
        painter.setPen(minor_pen)
        for major_row in range(3):
            for major_col in range(3):
                sub = self._board.sub_board_at(major_row, major_col)
                # Skip grid lines on fully-colored terminal sub-boards.
                if sub.evaluation() != 0.0:
                    continue
                for i in range(1, 3):
                    # Vertical
                    x = major_col * sub_size + i * cell_size
                    painter.drawLine(
                        x, major_row * sub_size,
                        x, major_row * sub_size + sub_size,
                    )
                    # Horizontal
                    y = major_row * sub_size + i * cell_size
                    painter.drawLine(
                        major_col * sub_size, y,
                        major_col * sub_size + sub_size, y,
                    )

        # Draw the major grid lines on top.
        major_pen = QPen(COLOR_GRID_MAJOR, MAJOR_LINE_WIDTH)
        painter.setPen(major_pen)
        for i in range(1, 3):
            painter.drawLine(i * sub_size, 0, i * sub_size, BOARD_SIZE)
            painter.drawLine(0, i * sub_size, BOARD_SIZE, i * sub_size)
        # Outer border.
        painter.drawRect(0, 0, BOARD_SIZE, BOARD_SIZE)

    def _draw_cells(
        self,
        painter: QPainter,
        sub,
        sub_rect: QRectF,
        cell_size: float,
    ) -> None:
        """Draw X and O markers inside a sub-board."""
        pen = QPen(COLOR_P1, 6)
        pen.setCapStyle(Qt.RoundCap)
        for r in range(3):
            for c in range(3):
                value = sub.item_at(r, c)
                if value == 0:
                    continue
                cell_rect = QRectF(
                    sub_rect.x() + c * cell_size,
                    sub_rect.y() + r * cell_size,
                    cell_size,
                    cell_size,
                )
                margin = cell_size * 0.2
                inner = cell_rect.adjusted(margin, margin, -margin, -margin)
                if value == 1:
                    painter.setPen(QPen(COLOR_P1, 4, Qt.SolidLine, Qt.RoundCap))
                    painter.drawLine(inner.topLeft(), inner.bottomRight())
                    painter.drawLine(inner.topRight(), inner.bottomLeft())
                else:
                    painter.setPen(QPen(COLOR_P2, 4))
                    painter.drawEllipse(inner)

    def _draw_big_marker(
        self,
        painter: QPainter,
        sub_rect: QRectF,
        color: QColor,
        letter: str,
    ) -> None:
        """Draw a large X or O overlay on a terminal sub-board."""
        painter.save()
        pen = QPen(color, 10, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        margin = sub_rect.width() * 0.22
        inner = sub_rect.adjusted(margin, margin, -margin, -margin)
        if letter == "X":
            painter.drawLine(inner.topLeft(), inner.bottomRight())
            painter.drawLine(inner.topRight(), inner.bottomLeft())
        else:
            painter.drawEllipse(inner)
        painter.restore()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Translate a click into a Move and emit `move_selected`."""
        if not self._accept_clicks or self._board is None:
            return
        pos = event.scenePos()
        x, y = pos.x(), pos.y()
        if x < 0 or y < 0 or x >= BOARD_SIZE or y >= BOARD_SIZE:
            return

        sub_size = BOARD_SIZE / 3.0
        cell_size = sub_size / 3.0
        major_col = int(x // sub_size)
        major_row = int(y // sub_size)
        minor_col = int((x - major_col * sub_size) // cell_size)
        minor_row = int((y - major_row * sub_size) // cell_size)

        # Clamp in case of floating point edge cases.
        major_row = max(0, min(2, major_row))
        major_col = max(0, min(2, major_col))
        minor_row = max(0, min(2, minor_row))
        minor_col = max(0, min(2, minor_col))

        marker = self._board.current_player()
        if marker == Marker.EMPTY:
            return
        move = Move(
            major_row=major_row,
            major_col=major_col,
            minor_row=minor_row,
            minor_col=minor_col,
            marker=marker,
        )
        if self._board.is_legal(move):
            self.move_selected.emit(move)


__all__ = ["BoardScene", "BOARD_SIZE"]
