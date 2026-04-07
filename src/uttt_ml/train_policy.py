"""
train_policy — fit the policy network on a combined .npz training archive.

Mirrors `PolicyNetworkTrainer.java` as closely as PyTorch allows:

    - Loads a single combined `.npz` and shuffles deterministically
    - 80/20 train/validation split
    - Adam optimizer, learning rate 0.006
    - Fixed seed 2039402
    - MSE loss, identity activation on the output
    - Batch size 1000 (Java default)
    - Up to 500 epochs with divergence-based early stopping: halt when
      `|train_mse - val_mse| / val_mse * 100 > 2.5%`
    - Checkpoints written every 30 seconds
    - Final model saved to `policy_final_mse<mse>.pt`

Note: the policy dataset renders a 76x76 image per sample on the fly
from the 90-element flat vector. On a CPU-only machine this is
noticeably slower than loading pre-rendered images, but it keeps the
data archive compact and makes encoder changes trivial. Use
`--num-workers 4` or higher on beefy machines.

CLI:

    python -m uttt_ml.train_policy \
        --data ./data/series1_combined_policy.npz \
        --out ./models/policy \
        --epochs 500 --seed 2039402 \
        --lr 0.006 --batch 1000 --device auto
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from uttt_bot.nets import PolicyNet
from uttt_ml.checkpoint import DivergenceEarlyStopper, TimedCheckpointer
from uttt_ml.dataset import PolicyDataset


def _resolve_device(device_arg: str) -> torch.device:
    """Map --device {auto,cpu,cuda} to a torch.device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)


def _shuffled_split(
    dataset_size: int, train_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Return shuffled train/test index lists with a fixed seed."""
    rng = np.random.default_rng(seed)
    indices = np.arange(dataset_size)
    rng.shuffle(indices)
    split = int(math.ceil(dataset_size * train_fraction))
    return indices[:split].tolist(), indices[split:].tolist()


def _evaluate_mse(
    model: PolicyNet, loader: DataLoader, device: torch.device
) -> float:
    """Return the pooled MSE of `model` over `loader`."""
    model.eval()
    squared_error_sum = 0.0
    count = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(features)
            diff = preds - labels
            squared_error_sum += float((diff * diff).sum().item())
            count += labels.numel()
    return squared_error_sum / max(count, 1)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point registered as `uttt-train-policy`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--divergence-threshold", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=2039402)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        default=30.0,
        help="Seconds between disk checkpoints",
    )
    args = parser.parse_args(argv)

    if not args.data.is_file():
        print(f"error: data file not found: {args.data}", file=sys.stderr)
        return 2

    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    model_run_dir = args.out / f"policy_{int(time.time())}"
    model_run_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_run_dir / "trainingLog.log"
    log_file = log_path.open("w")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"device={device}")
    log(f"seed={args.seed}")
    log(f"data={args.data}")
    log(f"output_dir={model_run_dir}")

    # -------- dataset + split --------
    dataset = PolicyDataset(args.data)
    train_idx, val_idx = _shuffled_split(
        len(dataset), args.train_fraction, seed=42
    )
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    log(f"train_size={len(train_set)}, val_size={len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # -------- model --------
    model = PolicyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    checkpointer = TimedCheckpointer(
        out_dir=model_run_dir, interval_seconds=args.checkpoint_interval
    )
    stopper = DivergenceEarlyStopper(threshold_percent=args.divergence_threshold)

    # -------- training loop --------
    log("Beginning training.")
    last_val_mse = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            running_count += labels.size(0)
        train_mse = running_loss / max(running_count, 1)
        val_mse = _evaluate_mse(model, val_loader, device)
        last_val_mse = val_mse
        log(f"Epoch {epoch:4d} | train MSE {train_mse:.6f} | val MSE {val_mse:.6f}")

        ckpt = checkpointer.maybe_save(model, epoch)
        if ckpt is not None:
            log(f"  checkpoint -> {ckpt.name}")

        if stopper.update(train_mse, val_mse):
            log(
                f"Early stopping: train/val divergence exceeded "
                f"{args.divergence_threshold}% at epoch {epoch}."
            )
            break

    final_name = f"policy_final_mse{last_val_mse:.6f}.pt"
    final_path = model_run_dir / final_name
    model.save(final_path)
    log(f"Model saved to {final_path}")
    log_file.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
