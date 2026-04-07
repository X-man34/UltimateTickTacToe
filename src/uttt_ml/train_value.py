"""
train_value — fit the value network on a combined .npz training archive.

Mirrors `ValueNetworkTrainer.java` as closely as PyTorch allows:

    - Loads a single combined `.npz` and shuffles deterministically
    - 80/20 train/test split
    - Adam optimizer, learning rate 0.001
    - Fixed seed 2039402 for reproducibility
    - MSE loss, identity activation on the output (matches Java)
    - Up to 500 epochs with early stopping after 20 epochs of no
      validation-loss improvement (PatienceEarlyStopper)
    - Checkpoints written every 30 seconds, keeping the two most recent
    - Regression metrics (MSE, MAE, RMSE) logged per epoch
    - Final model saved to `value_final_mse<mse>.pt`

CLI:

    python -m uttt_ml.train_value \
        --data ./data/series1_combined_value.npz \
        --out ./models/value \
        --epochs 500 --patience 20 --seed 2039402 \
        --lr 0.001 --batch 64 --device auto
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

from uttt_bot.nets import ValueNet
from uttt_ml.checkpoint import PatienceEarlyStopper, TimedCheckpointer
from uttt_ml.dataset import ValueDataset


def _resolve_device(device_arg: str) -> torch.device:
    """Map --device {auto,cpu,cuda} to a torch.device, falling back on CPU."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover — CPU-only test machine
        torch.cuda.manual_seed_all(seed)


def _shuffled_split(
    dataset_size: int, train_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Return shuffled train/test index lists with a fixed seed.

    Mirrors `DataSet.shuffle(42)` + `splitTestAndTrain(0.8)` from the
    Java side but under our control.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(dataset_size)
    rng.shuffle(indices)
    split = int(math.ceil(dataset_size * train_fraction))
    return indices[:split].tolist(), indices[split:].tolist()


def _regression_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float, float]:
    """Compute MSE, MAE, RMSE over a batch of 1-d predictions."""
    diff = predictions - targets
    mse = float((diff * diff).mean().item())
    mae = float(diff.abs().mean().item())
    rmse = float(math.sqrt(mse))
    return mse, mae, rmse


def _evaluate(
    model: ValueNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run the model over `loader` and return pooled regression metrics."""
    model.eval()
    squared_error_sum = 0.0
    abs_error_sum = 0.0
    count = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(features)
            diff = preds - labels
            squared_error_sum += float((diff * diff).sum().item())
            abs_error_sum += float(diff.abs().sum().item())
            count += labels.numel()
    if count == 0:
        return float("inf"), float("inf"), float("inf")
    mse = squared_error_sum / count
    mae = abs_error_sum / count
    rmse = math.sqrt(mse)
    return mse, mae, rmse


def main(argv: list[str] | None = None) -> int:
    """CLI entry point registered as `uttt-train-value`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Combined .npz file")
    parser.add_argument("--out", type=Path, required=True, help="Models output dir")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2039402)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--num-workers", type=int, default=2, help="DataLoader worker processes"
    )
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

    model_run_dir = args.out / f"value_{int(time.time())}"
    model_run_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_run_dir / "trainingLog.log"
    log_file = log_path.open("w")

    def log(msg: str) -> None:
        """Write `msg` both to stdout and the training log file, like the Java `log()` helper."""
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"device={device}")
    log(f"seed={args.seed}")
    log(f"data={args.data}")
    log(f"output_dir={model_run_dir}")

    # -------- dataset + split --------
    dataset = ValueDataset(args.data)
    train_idx, test_idx = _shuffled_split(
        len(dataset), args.train_fraction, seed=42
    )
    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)
    log(f"train_size={len(train_set)}, test_size={len(test_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # -------- model --------
    model = ValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    checkpointer = TimedCheckpointer(
        out_dir=model_run_dir, interval_seconds=args.checkpoint_interval
    )
    stopper = PatienceEarlyStopper(patience=args.patience)

    # -------- training loop --------
    log("Beginning training.")
    best_test_mse = float("inf")
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

        mse, mae, rmse = _evaluate(model, test_loader, device)
        if mse < best_test_mse:
            best_test_mse = mse
        log(
            f"Epoch {epoch:4d} | train MSE {train_mse:.6f} | "
            f"test MSE {mse:.6f} | MAE {mae:.6f} | RMSE {rmse:.6f}"
        )

        ckpt = checkpointer.maybe_save(model, epoch)
        if ckpt is not None:
            log(f"  checkpoint -> {ckpt.name}")

        if stopper.update(mse):
            log(
                f"Early stopping: no improvement for {args.patience} epochs. "
                f"Best test MSE: {stopper.best:.6f}"
            )
            break

    # -------- final save --------
    final_name = f"value_final_mse{best_test_mse:.6f}.pt"
    final_path = model_run_dir / final_name
    model.save(final_path)
    log(f"Model saved to {final_path}")
    log_file.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
