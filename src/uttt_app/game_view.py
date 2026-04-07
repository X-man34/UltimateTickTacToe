"""
GameView — the main gameplay widget.

Holds a live `Board`, a `BoardScene` for rendering, two `PlayerBox`
widgets, and a `BotWorker` on its own thread for computer moves.
The view handles the alternation between human clicks and bot
searches, updates the scene after every move, and announces the
winner when the game is over.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import torch
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from uttt_app.board_scene import BOARD_SIZE, BoardScene
from uttt_app.bot_worker import BotWorker
from uttt_app.player_box import PlayerBox
from uttt_bot.bot import UTTTBot
from uttt_bot.config import MCTSConfig
from uttt_engine import Board, Evaluation, Marker, Move


class PlayerKind(Enum):
    """Whether a seat is occupied by a human or the bot."""

    HUMAN = auto()
    BOT = auto()


@dataclass
class GameSetup:
    """Everything the GameView needs to start a new game.

    Passed in from the MenuScreen.
    """

    player_one_kind: PlayerKind
    player_two_kind: PlayerKind
    player_one_name: str
    player_two_name: str
    mcts_config: MCTSConfig
    device: torch.device


class GameView(QWidget):
    """The main in-game widget — board + two player panels + back button."""

    #: Emitted when the user clicks "Back to Menu".
    return_to_menu = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ----- Layout -----
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Board display.
        self._scene = BoardScene()
        self._scene.move_selected.connect(self._on_human_move)
        from PySide6.QtGui import QPainter  # local import so the UI module
        # file doesn't carry the symbol when it isn't used elsewhere.
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.Antialiasing, True)
        self._view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setMinimumSize(int(BOARD_SIZE), int(BOARD_SIZE))

        layout.addWidget(self._view, stretch=3)

        # Side panel: player boxes + back button.
        side = QVBoxLayout()
        side.setSpacing(16)
        self._player_one_box = PlayerBox(Marker.P1, "Player One")
        self._player_two_box = PlayerBox(Marker.P2, "Player Two")
        side.addWidget(self._player_one_box)
        side.addWidget(self._player_two_box)
        self._status_label = QLabel("")
        self._status_label.setObjectName("SectionLabel")
        self._status_label.setAlignment(Qt.AlignCenter)
        side.addWidget(self._status_label)
        side.addStretch(1)
        self._back_button = QPushButton("Back to Menu")
        self._back_button.clicked.connect(self.return_to_menu.emit)
        side.addWidget(self._back_button)
        layout.addLayout(side, stretch=1)

        # ----- State -----
        self._board: Board | None = None
        self._setup: GameSetup | None = None
        self._bot: UTTTBot | None = None
        self._bot_thread: QThread | None = None
        self._bot_worker: BotWorker | None = None
        self._is_bot_thinking = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_game(self, setup: GameSetup) -> None:
        """Start (or restart) a game using the provided setup."""
        # Clean up any previous bot thread.
        self._teardown_bot_thread()

        self._setup = setup
        self._board = Board()
        self._player_one_box.name_label.setText(setup.player_one_name)
        self._player_two_box.name_label.setText(setup.player_two_name)
        self._scene.set_board(self._board)

        needs_bot = PlayerKind.BOT in (setup.player_one_kind, setup.player_two_kind)
        if needs_bot:
            self._bot = UTTTBot(setup.mcts_config, device=setup.device)
            self._bot_thread = QThread(self)
            self._bot_worker = BotWorker(self._bot)
            self._bot_worker.moveToThread(self._bot_thread)
            self._bot_worker.move_chosen.connect(self._on_bot_move)
            self._bot_worker.error.connect(self._on_bot_error)
            self._bot_thread.start()
        else:
            self._bot = None

        self._status_label.setText("")
        self._advance_turn()

    def shutdown(self) -> None:
        """Tear down the bot thread. Call before closing the window."""
        self._teardown_bot_thread()

    def _teardown_bot_thread(self) -> None:
        if self._bot_thread is not None:
            self._bot_thread.quit()
            self._bot_thread.wait(2000)
            self._bot_thread = None
            self._bot_worker = None

    # ------------------------------------------------------------------
    # Turn handling
    # ------------------------------------------------------------------

    def _advance_turn(self) -> None:
        """Evaluate the current position and route control to human or bot."""
        assert self._board is not None and self._setup is not None

        if self._board.is_terminal():
            self._announce_winner()
            self._scene.set_accept_clicks(False)
            return

        current_marker = self._board.current_player()
        current_kind = (
            self._setup.player_one_kind
            if current_marker == Marker.P1
            else self._setup.player_two_kind
        )

        self._update_player_statuses()

        if current_kind == PlayerKind.HUMAN:
            self._scene.set_accept_clicks(True)
        else:
            self._scene.set_accept_clicks(False)
            self._request_bot_move()

    def _request_bot_move(self) -> None:
        """Ask the BotWorker to search the current position."""
        assert self._bot_worker is not None and self._board is not None
        self._is_bot_thinking = True
        # Use a direct call that Qt will dispatch into the worker's
        # thread because the worker lives there.
        self._bot_worker.request_move(self._board)

    @Slot(object)
    def _on_human_move(self, move: Move) -> None:
        """Handle a click on the board."""
        if self._board is None:
            return
        if not self._board.is_legal(move):
            return
        self._board.apply_move(move)
        self._scene.update()
        self._advance_turn()

    @Slot(object)
    def _on_bot_move(self, move: Move) -> None:
        """Handle a move returned by the BotWorker."""
        self._is_bot_thinking = False
        if self._board is None:
            return
        if not self._board.is_legal(move):
            self._status_label.setText(f"Bot returned illegal move: {move}")
            return
        self._board.apply_move(move)
        self._scene.update()
        self._advance_turn()

    @Slot(str)
    def _on_bot_error(self, message: str) -> None:  # pragma: no cover
        """Surface a bot failure on the status label."""
        self._is_bot_thinking = False
        self._status_label.setText(f"Bot error: {message}")

    # ------------------------------------------------------------------
    # Status labels
    # ------------------------------------------------------------------

    def _update_player_statuses(self) -> None:
        assert self._board is not None and self._setup is not None
        current = self._board.current_player()
        if current == Marker.P1:
            p1_state = (
                "thinking" if self._setup.player_one_kind == PlayerKind.BOT else "active"
            )
            p1_text = "Thinking…" if p1_state == "thinking" else "Your move"
            self._player_one_box.set_status(p1_text, p1_state)
            self._player_two_box.set_status("Waiting…", "idle")
        else:
            p2_state = (
                "thinking" if self._setup.player_two_kind == PlayerKind.BOT else "active"
            )
            p2_text = "Thinking…" if p2_state == "thinking" else "Your move"
            self._player_two_box.set_status(p2_text, p2_state)
            self._player_one_box.set_status("Waiting…", "idle")

    def _announce_winner(self) -> None:
        assert self._board is not None
        evaluation = self._board.evaluation()
        if evaluation == Evaluation.P1_WIN.value:
            self._player_one_box.set_status("Winner!", "winner")
            self._player_two_box.set_status("Defeated", "idle")
            self._status_label.setText("Player One wins.")
        elif evaluation == Evaluation.P2_WIN.value:
            self._player_two_box.set_status("Winner!", "winner")
            self._player_one_box.set_status("Defeated", "idle")
            self._status_label.setText("Player Two wins.")
        else:
            self._player_one_box.set_status("Draw", "idle")
            self._player_two_box.set_status("Draw", "idle")
            self._status_label.setText("Draw.")

    # ------------------------------------------------------------------
    # Auto-fit the view on resize
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: D401, N802
        """Keep the board scene fitted inside the view as the window resizes."""
        super().resizeEvent(event)
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)


__all__ = ["GameSetup", "GameView", "PlayerKind"]
