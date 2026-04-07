"""
PyTorch Dataset classes over the `.npz` training format.

The same on-disk schema is used for both legacy converted data and
newly generated self-play data:

    features    (N, 90)    float32   — board flat vectors
    labels      (N, K)     float32   — K=1 for value, K=81 for policy

This module provides two Datasets, one per network:

  - `ValueDataset` reshapes each 90-element flat vector into the
    (4, 9, 3, 3) NCDHW tensor that the value net consumes. Labels are
    scalar game outcomes in roughly [-1, 1].

  - `PolicyDataset` reconstructs a Board from each flat vector and
    renders the 76x76 policy image on the fly. Labels are 81-element
    visit-ratio vectors.

Both classes use `np.load(..., mmap_mode="r")` so training runs with
large combined archives don't blow up RAM — rows are pulled off disk
as the DataLoader iterates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from uttt_bot.encoders import board_to_policy_image, value_vector_to_tensor
from uttt_engine import Board


# ----------------------------------------------------------------------
# Value dataset
# ----------------------------------------------------------------------

class ValueDataset(Dataset):
    """Training dataset for the value network.

    Each item is `(tensor, label)`:
      - `tensor` is shape `(4, 9, 3, 3)` float32 — the NCDHW value input.
      - `label` is shape `(1,)` float32 — the game outcome target.
    """

    def __init__(self, npz_path: str | Path) -> None:
        """Open the .npz archive and validate shapes.

        Args:
            npz_path: Path to a `.npz` with `features` (N, 90) and
                `labels` (N, 1) arrays. Both legacy converted data
                and new self-play data follow this schema.
        """
        self.path = Path(npz_path)
        data = np.load(self.path, mmap_mode="r")
        self.features = data["features"]
        self.labels = data["labels"]
        if self.features.ndim != 2 or self.features.shape[1] != 90:
            raise ValueError(
                f"features must have shape (N, 90), got {self.features.shape}"
            )
        if self.labels.ndim != 2 or self.labels.shape[1] != 1:
            raise ValueError(
                f"labels must have shape (N, 1), got {self.labels.shape}"
            )
        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"row-count mismatch between features {self.features.shape} "
                f"and labels {self.labels.shape}"
            )

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat = np.asarray(self.features[index], dtype=np.float64)
        # Build the (9, 3, 3, 4) NDHWC tensor first, then permute to
        # PyTorch's (4, 9, 3, 3) NCDHW layout at the end. Doing the
        # permute per-item is fine because the array is tiny (324
        # floats) — DataLoader workers will parallelize it if needed.
        ndhwc = value_vector_to_tensor(flat)  # (9, 3, 3, 4) float32
        ncdhw = np.transpose(ndhwc, (3, 0, 1, 2))  # (4, 9, 3, 3)
        feature_tensor = torch.from_numpy(ncdhw.copy())
        label_tensor = torch.from_numpy(
            np.asarray(self.labels[index], dtype=np.float32).copy()
        )
        return feature_tensor, label_tensor


# ----------------------------------------------------------------------
# Policy dataset
# ----------------------------------------------------------------------

class PolicyDataset(Dataset):
    """Training dataset for the policy network.

    Each item is `(image, label)`:
      - `image` is shape `(3, 76, 76)` float32 — the policy-net image.
      - `label` is shape `(81,)` float32 — MCTS visit ratios.

    The image is re-rendered on every access via `board_to_policy_image`,
    which is slower than a pre-rendered cache but keeps the dataset file
    compact and lets us iterate on the encoder without regenerating
    training data.
    """

    def __init__(self, npz_path: str | Path) -> None:
        """Open the .npz archive and validate shapes.

        Args:
            npz_path: Path to a `.npz` with `features` (N, 90) and
                `labels` (N, 81) arrays.
        """
        self.path = Path(npz_path)
        data = np.load(self.path, mmap_mode="r")
        self.features = data["features"]
        self.labels = data["labels"]
        if self.features.ndim != 2 or self.features.shape[1] != 90:
            raise ValueError(
                f"features must have shape (N, 90), got {self.features.shape}"
            )
        if self.labels.ndim != 2 or self.labels.shape[1] != 81:
            raise ValueError(
                f"labels must have shape (N, 81), got {self.labels.shape}"
            )
        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"row-count mismatch between features {self.features.shape} "
                f"and labels {self.labels.shape}"
            )

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat = np.asarray(self.features[index], dtype=np.float64)
        board = Board.from_flat_vector(flat)
        image = board_to_policy_image(board)  # (3, 76, 76) float32
        image_tensor = torch.from_numpy(image.copy())
        label_tensor = torch.from_numpy(
            np.asarray(self.labels[index], dtype=np.float32).copy()
        )
        return image_tensor, label_tensor


__all__ = ["PolicyDataset", "ValueDataset"]
