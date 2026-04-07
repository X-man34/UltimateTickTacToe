"""
PlayerBox — small panel showing a player's name, role, and current status.

Mirrors `PlayerBox.java` from the original project but trimmed down to
what the new UI actually uses: a name label, a swatch of the player's
marker color, and a dynamic status line ("waiting", "thinking",
"winner", etc).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from uttt_engine import Marker


class _MarkerSwatch(QWidget):
    """A small colored square showing a player's marker color."""

    def __init__(self, color: QColor, size: int = 28, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = color
        self.setFixedSize(size, size)

    def paintEvent(self, event) -> None:  # noqa: D401, N802
        """Paint the swatch as a filled rounded square."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self._color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 6, 6)


class PlayerBox(QFrame):
    """A panel displaying one player's identity and state.

    Attributes:
        marker: Which marker (P1 or P2) this box represents.
        name_label: The player's display name.
        status_label: A short dynamic string ("Your move", "Thinking…",
            "Winner!", etc). Styled via the `state` dynamic property
            picked up from the theme QSS.
    """

    def __init__(self, marker: Marker, display_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.marker = marker

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        header = QLabel(f"Player {1 if marker == Marker.P1 else 2}")
        header.setObjectName("SectionLabel")
        layout.addWidget(header)

        swatch_and_name = QLabel(display_name)
        swatch_and_name.setObjectName("PlayerNameLabel")
        self.name_label = swatch_and_name

        # Red for P1, Green for P2 — matches the board renderer's scheme.
        color = QColor("#ff4c4c") if marker == Marker.P1 else QColor("#3fd07d")
        self._swatch = _MarkerSwatch(color)

        row = QFrame()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(self._swatch, alignment=Qt.AlignLeft)
        row_layout.addWidget(self.name_label)
        layout.addWidget(row)

        self.status_label = QLabel("Waiting…")
        self.status_label.setObjectName("PlayerStatusLabel")
        self.status_label.setProperty("state", "idle")
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def set_status(self, text: str, state: str = "idle") -> None:
        """Update the status line. `state` picks the QSS highlight color.

        Known states: "idle", "active" (your move), "thinking", "winner".
        """
        self.status_label.setText(text)
        self.status_label.setProperty("state", state)
        # Force a style re-apply so the dynamic property takes effect.
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)


__all__ = ["PlayerBox"]
