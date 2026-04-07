"""
MCTSConfig — bundle of knobs that configure an `MCTSEvaluator`.

Mirrors `EvaluatorConfiguration.java` from the original project. Serialized
to / from JSON as a plain dataclass. The original Java side used a JSON+ZIP
combo; we drop the ZIP layer because plain JSON is already tiny.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class MCTSConfig:
    """All tunable parameters for a single MCTS search run.

    Attributes:
        compute_time_seconds: Wall-clock time budget per `search()` call.
            Mirrors `EvaluatorConfiguration.computeTime`.
        c_ucb1: Exploration constant in the UCB1 formula. Java default
            is 2.0.
        max_rollout_depth: Safety cap on random playouts to keep runaway
            rollouts from dominating the time budget.
        num_threads: Size of the worker pool. Set to 1 for the
            single-threaded path; any higher value engages the
            virtual-loss multithreading described in mcts.py.
        task_queue_depth: Approximate number of MCTS iterations kept in
            flight in the worker pool at once. Mirrors the ~25-task
            buffering pattern in the Java main loop.
        value_net_path: Optional path to a saved `ValueNet` checkpoint.
            If provided, leaf evaluation uses the network; if not, MCTS
            falls back to random rollouts.
        policy_net_path: Optional path to a saved `PolicyNet` checkpoint.
            If provided, child selection samples from the network's
            probability distribution.
        use_rollout_if_no_value_net: When no value net is supplied,
            whether to fall back to random rollouts (True) or error out
            (False). Always True in the original Java code.
        seed: Optional RNG seed for reproducible searches (used for
            random rollouts and policy-net sampling).
    """

    compute_time_seconds: float = 1.0
    c_ucb1: float = 2.0
    max_rollout_depth: int = 1000
    num_threads: int = 4
    task_queue_depth: int = 25
    value_net_path: Path | None = None
    policy_net_path: Path | None = None
    use_rollout_if_no_value_net: bool = True
    seed: int | None = None

    def to_json(self, path: str | Path) -> None:
        """Serialize this config to a JSON file."""
        data = asdict(self)
        # Paths are not JSON-native; convert to strings.
        data["value_net_path"] = (
            str(self.value_net_path) if self.value_net_path is not None else None
        )
        data["policy_net_path"] = (
            str(self.policy_net_path) if self.policy_net_path is not None else None
        )
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "MCTSConfig":
        """Load a config from a JSON file."""
        data = json.loads(Path(path).read_text())
        if data.get("value_net_path") is not None:
            data["value_net_path"] = Path(data["value_net_path"])
        if data.get("policy_net_path") is not None:
            data["policy_net_path"] = Path(data["policy_net_path"])
        return cls(**data)


__all__ = ["MCTSConfig"]
