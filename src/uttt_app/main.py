"""
Ultimate Tic Tac Toe — PySide6 desktop app entry point.

Loads `config.yaml` (if present) to resolve per-machine defaults,
applies the theme stylesheet, and launches `GameWindow`.

Config keys (all optional; sensible defaults are baked in):

    device: auto | cpu | cuda
    num_threads: int — default MCTS thread count shown in the menu
    mcts.compute_time_seconds: float
    paths.value_net: path
    paths.policy_net: path

Run via the console script:
    uttt

Or directly:
    python -m uttt_app.main
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from PySide6.QtWidgets import QApplication

from uttt_app.game_window import GameWindow


@dataclass
class AppConfig:
    """Snapshot of the fields from `config.yaml` that the UI cares about."""

    device: torch.device
    default_num_threads: int
    default_compute_time: float
    default_value_net: Path | None
    default_policy_net: Path | None


def _load_config(project_root: Path) -> AppConfig:
    """Load `config.yaml` from `project_root` if it exists, else use defaults.

    Defaults:
        device = auto (cuda if available, else cpu)
        num_threads = 4
        compute_time = 2.0 seconds
        value_net / policy_net = None (random rollouts / UCB1)
    """
    config_path = project_root / "config.yaml"
    data: dict = {}
    if config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text()) or {}
        if isinstance(loaded, dict):
            data = loaded

    # --- device ---
    device_str = str(data.get("device", "auto"))
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    num_threads = int(data.get("num_threads", 4))
    mcts_block = data.get("mcts") or {}
    compute_time = float(mcts_block.get("compute_time_seconds", 2.0))

    paths_block = data.get("paths") or {}
    def _resolve_path(key: str) -> Path | None:
        value = paths_block.get(key)
        if value is None or value == "":
            return None
        p = Path(str(value))
        if not p.is_absolute():
            p = (project_root / p).resolve()
        return p if p.is_file() else None

    value_net = _resolve_path("value_net")
    policy_net = _resolve_path("policy_net")

    return AppConfig(
        device=device,
        default_num_threads=num_threads,
        default_compute_time=compute_time,
        default_value_net=value_net,
        default_policy_net=policy_net,
    )


def _project_root() -> Path:
    """Return the project root — walks up from this file to find pyproject.toml."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parent


def _load_theme(app: QApplication, project_root: Path) -> None:
    """Apply `theme.qss` as the application-wide stylesheet if present."""
    qss_path = Path(__file__).parent / "theme.qss"
    if qss_path.is_file():
        app.setStyleSheet(qss_path.read_text())


def main(argv: list[str] | None = None) -> int:
    """CLI entry point registered as `uttt`."""
    app = QApplication(argv or sys.argv)
    app.setApplicationName("Ultimate Tic Tac Toe")

    project_root = _project_root()
    _load_theme(app, project_root)
    config = _load_config(project_root)

    window = GameWindow(
        default_compute_time=config.default_compute_time,
        default_num_threads=config.default_num_threads,
        default_value_net=config.default_value_net,
        default_policy_net=config.default_policy_net,
        device=config.device,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
