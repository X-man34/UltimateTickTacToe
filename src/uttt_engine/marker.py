"""
Marker enum — the three possible values of a cell on an Ultimate Tic Tac Toe board.

Numeric codes match the Java convention (BoardState.java's `itemAt` returns 1
for player one, -1 for player two, 0 for empty). Those same codes are used in
the 90-element flat vector serialization that the legacy .bin training data
relies on, so changing them would break old data compatibility.
"""

from __future__ import annotations

from enum import IntEnum


class Marker(IntEnum):
    """Contents of a single cell."""

    EMPTY = 0
    P1 = 1
    P2 = -1

    @classmethod
    def opponent(cls, marker: "Marker") -> "Marker":
        """Return the opposing marker. EMPTY returns EMPTY."""
        if marker == cls.P1:
            return cls.P2
        if marker == cls.P2:
            return cls.P1
        return cls.EMPTY
