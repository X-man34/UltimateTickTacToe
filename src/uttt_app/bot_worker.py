"""
BotWorker — runs `UTTTBot.choose_move` off the main Qt thread.

Qt forbids long-running work on the GUI thread because the event loop
blocks until `choose_move` returns. MCTS can take several seconds, so
we move the bot into a worker `QObject` living on its own `QThread`
and communicate via signals.

Flow:
    1. GameView creates a UTTTBot and a BotWorker(QObject).
    2. GameView moves the worker onto a QThread and starts the thread.
    3. When it's the bot's turn, GameView calls `worker.request_move(board)`
       (queued into the thread).
    4. The worker runs the bot, then emits `move_chosen(Move)`.
    5. GameView's slot applies the move and updates the scene.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from uttt_bot.bot import UTTTBot
from uttt_engine import Board, Move


class BotWorker(QObject):
    """QObject wrapper around a UTTTBot. Moved to its own QThread."""

    #: Emitted when the bot has picked a move.
    move_chosen = Signal(object)  # Move

    #: Emitted if the bot fails (e.g. all moves illegal — shouldn't happen).
    error = Signal(str)

    def __init__(self, bot: UTTTBot, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._bot = bot

    @Slot(object)
    def request_move(self, board: Board) -> None:
        """Slot: called from the GUI thread to kick off a bot search."""
        try:
            # Work on a clone so even if the GUI mutates the original
            # mid-search we don't corrupt it. (MCTS search already
            # doesn't mutate, but clones are cheap.)
            move: Move = self._bot.choose_move(board.clone())
        except Exception as exc:  # pragma: no cover
            self.error.emit(str(exc))
            return
        self.move_chosen.emit(move)


__all__ = ["BotWorker"]
