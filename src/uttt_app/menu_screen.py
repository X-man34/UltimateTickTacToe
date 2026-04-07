"""
MenuScreen — main menu for configuring a new game.

Lets the user pick:
    - Each player's name
    - Whether each player is a human or the bot
    - Per-bot MCTS compute time
    - Per-bot number of MCTS threads
    - Per-bot optional value and policy network checkpoint paths

Mirrors `MenuScreen.java` from the original project but with a more
modern flat layout. Defaults are read from the loaded `AppConfig` so
per-machine `config.yaml` overrides show up here automatically.

When the user clicks "Start Game", the menu builds a `GameSetup` and
emits the `start_game_requested` signal. The `GameWindow` wires that
into `GameView.start_game`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from uttt_app.game_view import GameSetup, PlayerKind
from uttt_bot.config import MCTSConfig


class _PlayerConfigPanel(QFrame):
    """Form for configuring one player — name, type, optional bot knobs."""

    def __init__(
        self,
        default_name: str,
        default_compute_time: float,
        default_num_threads: int,
        default_value_net: Path | None,
        default_policy_net: Path | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        layout = QFormLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.name_edit = QLineEdit(default_name)
        layout.addRow("Name", self.name_edit)

        self.kind_combo = QComboBox()
        self.kind_combo.addItems(["Human", "Computer (MCTS)"])
        layout.addRow("Type", self.kind_combo)

        self.compute_time_spin = QDoubleSpinBox()
        self.compute_time_spin.setRange(0.1, 600.0)
        self.compute_time_spin.setSingleStep(0.5)
        self.compute_time_spin.setValue(default_compute_time)
        self.compute_time_spin.setSuffix(" s")
        layout.addRow("MCTS time", self.compute_time_spin)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(default_num_threads)
        layout.addRow("Threads", self.threads_spin)

        self.value_net_edit = QLineEdit(
            str(default_value_net) if default_value_net else ""
        )
        value_browse = QPushButton("…")
        value_browse.setFixedWidth(40)
        value_browse.clicked.connect(
            lambda: self._pick_file(self.value_net_edit, "Value network checkpoint")
        )
        row = QHBoxLayout()
        row.addWidget(self.value_net_edit)
        row.addWidget(value_browse)
        layout.addRow("Value net", row)

        self.policy_net_edit = QLineEdit(
            str(default_policy_net) if default_policy_net else ""
        )
        policy_browse = QPushButton("…")
        policy_browse.setFixedWidth(40)
        policy_browse.clicked.connect(
            lambda: self._pick_file(self.policy_net_edit, "Policy network checkpoint")
        )
        row = QHBoxLayout()
        row.addWidget(self.policy_net_edit)
        row.addWidget(policy_browse)
        layout.addRow("Policy net", row)

        self.kind_combo.currentIndexChanged.connect(self._update_bot_fields_enabled)
        self._update_bot_fields_enabled()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_file(self, edit: QLineEdit, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            "PyTorch checkpoints (*.pt);;All files (*)",
        )
        if path:
            edit.setText(path)

    def _update_bot_fields_enabled(self) -> None:
        is_bot = self.kind_combo.currentIndex() == 1
        self.compute_time_spin.setEnabled(is_bot)
        self.threads_spin.setEnabled(is_bot)
        self.value_net_edit.setEnabled(is_bot)
        self.policy_net_edit.setEnabled(is_bot)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def player_kind(self) -> PlayerKind:
        return PlayerKind.HUMAN if self.kind_combo.currentIndex() == 0 else PlayerKind.BOT

    def player_name(self) -> str:
        name = self.name_edit.text().strip()
        return name or "Player"

    def build_mcts_config(self) -> MCTSConfig:
        """Collect the form fields into a MCTSConfig."""
        value_text = self.value_net_edit.text().strip()
        policy_text = self.policy_net_edit.text().strip()
        return MCTSConfig(
            compute_time_seconds=float(self.compute_time_spin.value()),
            num_threads=int(self.threads_spin.value()),
            value_net_path=Path(value_text) if value_text else None,
            policy_net_path=Path(policy_text) if policy_text else None,
        )


class MenuScreen(QWidget):
    """Top-level menu widget. Emits `start_game_requested` on button click."""

    start_game_requested = Signal(object)  # GameSetup

    def __init__(
        self,
        default_compute_time: float,
        default_num_threads: int,
        default_value_net: Path | None,
        default_policy_net: Path | None,
        device: torch.device,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._device = device

        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 40, 40, 40)
        outer.setSpacing(16)

        title = QLabel("Ultimate Tic Tac Toe")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        outer.addWidget(title)

        subtitle = QLabel("Local hotseat and MCTS bot")
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        outer.addWidget(subtitle)

        panels = QHBoxLayout()
        panels.setSpacing(16)
        self._player_one_panel = _PlayerConfigPanel(
            "Player One",
            default_compute_time,
            default_num_threads,
            default_value_net,
            default_policy_net,
        )
        self._player_two_panel = _PlayerConfigPanel(
            "Computer",
            default_compute_time,
            default_num_threads,
            default_value_net,
            default_policy_net,
        )
        # Default player two to the bot, matching a common use case.
        self._player_two_panel.kind_combo.setCurrentIndex(1)

        panels.addWidget(self._player_one_panel)
        panels.addWidget(self._player_two_panel)
        outer.addLayout(panels)

        outer.addStretch(1)
        start_button = QPushButton("Start Game")
        start_button.setObjectName("PrimaryButton")
        start_button.clicked.connect(self._on_start_clicked)
        outer.addWidget(start_button, alignment=Qt.AlignCenter)

    def _on_start_clicked(self) -> None:
        p1_kind = self._player_one_panel.player_kind()
        p2_kind = self._player_two_panel.player_kind()
        # Pick whichever bot's config to use — if both are bots, P1's config
        # drives MCTS (we assume the user sets them symmetrically for bot-vs-bot
        # matches, which is how the Java side behaved).
        if p1_kind == PlayerKind.BOT:
            mcts_config = self._player_one_panel.build_mcts_config()
        elif p2_kind == PlayerKind.BOT:
            mcts_config = self._player_two_panel.build_mcts_config()
        else:
            # Two humans: build a default config; it'll never be used.
            mcts_config = MCTSConfig()

        setup = GameSetup(
            player_one_kind=p1_kind,
            player_two_kind=p2_kind,
            player_one_name=self._player_one_panel.player_name(),
            player_two_name=self._player_two_panel.player_name(),
            mcts_config=mcts_config,
            device=self._device,
        )
        self.start_game_requested.emit(setup)


__all__ = ["MenuScreen"]
