"""
UTTTBot — high-level bot wrapper that the UI consumes.

The UI does not need to know about MCTSEvaluator, value nets, policy
nets, or devices. It just wants a single function: "given a board,
give me a move." `UTTTBot` is that function.

Internally it builds an MCTSEvaluator once in the constructor and
reuses it across `choose_move` calls so the neural network weights
aren't loaded from disk on every turn.
"""

from __future__ import annotations

import torch

from uttt_bot.config import MCTSConfig
from uttt_bot.mcts import MCTSEvaluator, build_evaluator
from uttt_engine import Board, Move


class UTTTBot:
    """A stateful bot that picks moves via MCTS.

    Typical usage from the UI:

        cfg = MCTSConfig(
            compute_time_seconds=2.0,
            num_threads=4,
            value_net_path=Path("./models/value/.../value_final.pt"),
            policy_net_path=None,
        )
        bot = UTTTBot(cfg, device=torch.device("cpu"))
        move = bot.choose_move(board)
        board.apply_move(move)
    """

    def __init__(
        self,
        config: MCTSConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        """Create a bot.

        Args:
            config: MCTSConfig with compute time, number of threads,
                and optional paths to saved value/policy nets.
            device: PyTorch device for network inference. For the
                dual-machine story, typical values are `"cpu"` on the
                Linux dev box and `"cuda"` on the Windows training
                machine.
        """
        self.config = config
        self.device = torch.device(device)
        self._evaluator: MCTSEvaluator = build_evaluator(config, device=self.device)

    def choose_move(self, board: Board) -> Move:
        """Return the bot's recommended move for the current position.

        The bot respects `compute_time_seconds` from its config. The
        returned move is in the caller's perspective — if it's P2's
        turn, the move's marker is P2 — so the caller can pass it
        straight into `Board.apply_move`.
        """
        result = self._evaluator.search(board)
        return result.best_move

    def rebuild(self, config: MCTSConfig) -> None:
        """Swap in a new config (and reload nets if their paths changed)."""
        self.config = config
        self._evaluator = build_evaluator(config, device=self.device)


__all__ = ["UTTTBot"]
