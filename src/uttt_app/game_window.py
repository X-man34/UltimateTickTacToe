"""
GameWindow — top-level QMainWindow hosting the menu and game views.

A `QStackedWidget` flips between `MenuScreen` and `GameView`. The menu
emits `start_game_requested(GameSetup)`, which the window forwards to
`GameView.start_game(setup)` and then switches the stack. The game
view emits `return_to_menu` when the user clicks the back button, and
the window flips back to the menu.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QWidget

from uttt_app.game_view import GameSetup, GameView
from uttt_app.menu_screen import MenuScreen


class GameWindow(QMainWindow):
    """Main application window."""

    def __init__(
        self,
        default_compute_time: float,
        default_num_threads: int,
        default_value_net: Path | None,
        default_policy_net: Path | None,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Ultimate Tic Tac Toe")
        self.resize(1100, 800)

        self._stack = QStackedWidget(self)
        self.setCentralWidget(self._stack)

        self._menu = MenuScreen(
            default_compute_time=default_compute_time,
            default_num_threads=default_num_threads,
            default_value_net=default_value_net,
            default_policy_net=default_policy_net,
            device=device,
        )
        self._menu.start_game_requested.connect(self._on_start_game)
        self._stack.addWidget(self._menu)

        self._game = GameView()
        self._game.return_to_menu.connect(self._on_return_to_menu)
        self._stack.addWidget(self._game)

        self._stack.setCurrentWidget(self._menu)

    @Slot(object)
    def _on_start_game(self, setup: GameSetup) -> None:
        self._game.start_game(setup)
        self._stack.setCurrentWidget(self._game)

    @Slot()
    def _on_return_to_menu(self) -> None:
        self._game.shutdown()
        self._stack.setCurrentWidget(self._menu)

    def closeEvent(self, event) -> None:  # noqa: D401, N802
        """Shut the bot thread down cleanly on app exit."""
        self._game.shutdown()
        super().closeEvent(event)


__all__ = ["GameWindow"]
