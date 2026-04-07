"""
Shared training helpers — checkpointing and early stopping.

Both the value and policy trainers need a wall-clock checkpointer (the
Java side saves every 30 seconds) and an early-stopping policy. The
policies differ:

    - Value net: stop after N epochs with no validation-MSE improvement.
      Mirrors `ValueNetworkTrainer.java` which uses a patience counter.

    - Policy net: stop when the relative train/val divergence exceeds a
      threshold. Mirrors `PolicyNetworkTrainer.java` lines 139-145, which
      computes `|train_mse - val_mse| / val_mse * 100` and halts above
      2.5%.

These helpers live in a shared module so trainers aren't tangled
with each other and the policies can be unit-tested independently.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn


class TimedCheckpointer:
    """Write model checkpoints to disk on a wall-clock interval.

    Mirrors DL4J's `CheckpointListener.Builder().saveEvery(30, SECONDS)`.
    The trainer calls `maybe_save(model, epoch)` at least once per
    epoch; the checkpointer only actually writes to disk if the
    elapsed-time budget has passed since the last write.
    """

    def __init__(
        self,
        out_dir: Path,
        interval_seconds: float = 30.0,
        keep_last: int = 2,
    ) -> None:
        """Create a checkpointer.

        Args:
            out_dir: Directory where checkpoint files are written.
                Created if it does not exist.
            interval_seconds: Minimum elapsed wall-clock time between
                successive checkpoints. Defaults to 30s to match the Java.
            keep_last: Number of most-recent checkpoints to retain.
                Older ones are deleted. Mirrors
                `CheckpointListener.keepLast(2)`.
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval_seconds
        self.keep_last = keep_last
        self._last_save_time = 0.0
        self._checkpoints: list[Path] = []

    def maybe_save(self, model: nn.Module, epoch: int) -> Path | None:
        """Save a checkpoint if enough wall-clock time has elapsed.

        Returns the path written (or None if the interval hadn't
        elapsed yet and nothing was written).
        """
        now = time.monotonic()
        if now - self._last_save_time < self.interval:
            return None
        self._last_save_time = now

        ckpt_path = self.out_dir / f"checkpoint_epoch{epoch:04d}.pt"
        torch.save({"state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)
        self._checkpoints.append(ckpt_path)

        # Enforce the retention policy by deleting older checkpoints.
        while len(self._checkpoints) > self.keep_last:
            old = self._checkpoints.pop(0)
            try:
                old.unlink()
            except OSError:
                pass
        return ckpt_path


class PatienceEarlyStopper:
    """Stop training after `patience` epochs without validation improvement.

    Value-network policy — mirrors ValueNetworkTrainer.java lines 85-111.
    """

    def __init__(self, patience: int = 20) -> None:
        self.patience = patience
        self.best = float("inf")
        self.no_improvement = 0

    def update(self, val_metric: float) -> bool:
        """Record a new validation metric; return True if training should stop."""
        if val_metric < self.best:
            self.best = val_metric
            self.no_improvement = 0
        else:
            self.no_improvement += 1
        return self.no_improvement >= self.patience


class DivergenceEarlyStopper:
    """Stop when train/val MSE diverge by more than `threshold_percent`.

    Policy-network policy — mirrors PolicyNetworkTrainer.java lines 139-145.
    `threshold_percent` is a percentage, not a fraction (2.5 means 2.5%).
    """

    def __init__(self, threshold_percent: float = 2.5) -> None:
        self.threshold = threshold_percent

    def update(self, train_metric: float, val_metric: float) -> bool:
        """Return True if the relative divergence exceeds the threshold."""
        if val_metric <= 0.0:
            return False
        percent = abs(train_metric - val_metric) / val_metric * 100.0
        return percent > self.threshold


__all__ = [
    "DivergenceEarlyStopper",
    "PatienceEarlyStopper",
    "TimedCheckpointer",
]
