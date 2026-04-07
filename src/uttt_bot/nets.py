"""
Neural network modules — PyTorch ports of the DL4J value and policy nets.

Both networks take a board encoding as input and output unnormalized
regression targets. Loss is MSE during training. Identity activation on
the output layer is preserved from the Java side (there is no sigmoid or
tanh), so values are not bounded to a specific range.

Architectures below mirror `ValueNetworkTrainer.java` (lines 51-65) and
`PolicyNetworkTrainer.java` (lines 70-94). Because DL4J's
`setInputType(...)` auto-computes `nIn` on the first dense layer after a
Flatten, the explicit `nIn` values in the Java code are effectively
ignored — we derive the real dense-layer input size by chasing shapes
through the convolution stack.

Save/load helpers use `torch.save` / `torch.load` with `map_location`
so a model trained on a Windows GPU loads cleanly on a Linux CPU and
vice versa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


# ----------------------------------------------------------------------
# Value network — Conv3D over a (4, 9, 3, 3) input
# ----------------------------------------------------------------------

class ValueNet(nn.Module):
    """Predicts a scalar value for a board position.

    Architecture:
        - Conv3d(4 -> 4, kernel=3, padding=1), identity activation
        - Conv3d(4 -> 4, kernel=3, padding=1), ReLU
        - Flatten
        - Linear(4*9*3*3 = 324 -> 100), ReLU
        - Linear(100 -> 100), ReLU
        - Linear(100 -> 1), identity

    Input shape: (N, 4, 9, 3, 3) — PyTorch Conv3d expects (N, C, D, H, W).
    The NumPy encoder produces (9, 3, 3, 4) NDHWC per-sample, so callers
    need to transpose via `encode_batch(...)` below before invoking
    `forward`.
    """

    HIDDEN = 100

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1, stride=1
        )
        self.conv2 = nn.Conv3d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1, stride=1
        )
        # After two padding=1, stride=1 convs the spatial shape is unchanged:
        # (4, 9, 3, 3) -> flatten to 4 * 9 * 3 * 3 = 324.
        self.flatten_size = 4 * 9 * 3 * 3
        self.fc1 = nn.Linear(self.flatten_size, self.HIDDEN)
        self.fc2 = nn.Linear(self.HIDDEN, self.HIDDEN)
        self.out = nn.Linear(self.HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Run a batched forward pass.

        Args:
            x: Tensor of shape (N, 4, 9, 3, 3).

        Returns:
            Tensor of shape (N, 1) with the raw scalar predictions.
        """
        # Layer 0: Conv3D with identity activation (DL4J override, see
        # ValueNetworkTrainer.java line 57).
        x = self.conv1(x)
        # Layer 1: Conv3D inherits the top-level RELU activation.
        x = torch.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

    # ------------------------------------------------------------------
    # Batch encoding helper — bridges NumPy encoder output to the net
    # ------------------------------------------------------------------

    @staticmethod
    def encode_batch(
        tensors_ndhwc: np.ndarray | list[np.ndarray],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Stack NDHWC tensors into an NCDHW torch batch on `device`.

        The NumPy encoder returns each sample as (9, 3, 3, 4). This
        helper stacks several such samples and permutes to the
        (N, 4, 9, 3, 3) layout that PyTorch Conv3d expects.
        """
        if isinstance(tensors_ndhwc, list):
            stacked = np.stack(tensors_ndhwc, axis=0)
        else:
            stacked = (
                tensors_ndhwc
                if tensors_ndhwc.ndim == 5
                else tensors_ndhwc[np.newaxis, ...]
            )
        # NDHWC (N, 9, 3, 3, 4) -> NCDHW (N, 4, 9, 3, 3).
        nchwd = np.transpose(stacked, (0, 4, 1, 2, 3)).astype(np.float32)
        return torch.from_numpy(nchwd).to(device)

    # ------------------------------------------------------------------
    # Save / load — portable across CPU and CUDA machines
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the module's state dict to `path`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "value_net", "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> "ValueNet":
        """Load a ValueNet from `path`, mapping weights to `device`."""
        data = torch.load(Path(path), map_location=device, weights_only=True)
        if isinstance(data, dict) and "state_dict" in data:
            state_dict = data["state_dict"]
        else:
            state_dict = data
        net = cls()
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
        return net


# ----------------------------------------------------------------------
# Policy network — Conv2D over a (3, 76, 76) image
# ----------------------------------------------------------------------

class PolicyNet(nn.Module):
    """Predicts MCTS visit ratios for every cell on the 9x9 board.

    Architecture (mirrors PolicyNetworkTrainer.java lines 70-94):
        - Conv2d(3 -> 3, kernel=5, padding=1), identity
        - MaxPool2d(2, 2)
        - Conv2d(3 -> 3, kernel=5, padding=1), identity
        - MaxPool2d(2, 2)
        - Conv2d(3 -> 3, kernel=4, padding=1), identity
        - MaxPool2d(2, 2)
        - Conv2d(3 -> 3, kernel=4, padding=1), identity
        - MaxPool2d(2, 2)
        - Flatten (final spatial dims computed in __init__)
        - Linear(-> 300), ReLU  x4
        - Linear(-> 81), identity

    Input shape: (N, 3, 76, 76). Output shape: (N, 81) — one entry per
    move on the 9x9 board. MCTS masks illegal moves and renormalizes
    before sampling.
    """

    HIDDEN = 300
    OUTPUT_DIM = 81

    def __init__(self) -> None:
        super().__init__()

        # All DL4J conv layers in the Java trainer explicitly set
        # activation=IDENTITY. The top-level RELU does NOT apply to
        # them because the per-layer activation overrides it. We follow
        # the same convention here.
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=5, padding=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=4, padding=1, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=4, padding=1, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Compute the flattened size by running a dummy tensor through
        # the convolution stack. This avoids hard-coding a number that
        # depends on the exact conv/pool arithmetic — if the upstream
        # shape ever changes, we will pick it up automatically rather
        # than silently breaking with a wrong linear layer.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 76, 76)
            flat = self._conv_stack(dummy).flatten(start_dim=1)
            self.flatten_size = int(flat.shape[1])

        self.fc1 = nn.Linear(self.flatten_size, self.HIDDEN)
        self.fc2 = nn.Linear(self.HIDDEN, self.HIDDEN)
        self.fc3 = nn.Linear(self.HIDDEN, self.HIDDEN)
        self.fc4 = nn.Linear(self.HIDDEN, self.HIDDEN)
        self.out = nn.Linear(self.HIDDEN, self.OUTPUT_DIM)

    def _conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        """Shared conv+pool ladder used by `forward` and the init shape probe."""
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Run a batched forward pass.

        Args:
            x: Tensor of shape (N, 3, 76, 76) with values in [0, 1].

        Returns:
            Tensor of shape (N, 81). Raw (unnormalized) scores.
        """
        x = self._conv_stack(x)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.out(x)

    # ------------------------------------------------------------------
    # Batch encoding helper
    # ------------------------------------------------------------------

    @staticmethod
    def encode_batch(
        images_chw: np.ndarray | list[np.ndarray],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Stack (3, 76, 76) images into an (N, 3, 76, 76) batch on `device`."""
        if isinstance(images_chw, list):
            stacked = np.stack(images_chw, axis=0)
        else:
            stacked = (
                images_chw
                if images_chw.ndim == 4
                else images_chw[np.newaxis, ...]
            )
        return torch.from_numpy(stacked.astype(np.float32)).to(device)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the module's state dict to `path`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "policy_net", "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> "PolicyNet":
        """Load a PolicyNet from `path`, mapping weights to `device`."""
        data = torch.load(Path(path), map_location=device, weights_only=True)
        if isinstance(data, dict) and "state_dict" in data:
            state_dict = data["state_dict"]
        else:
            state_dict = data
        net = cls()
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
        return net


__all__ = ["PolicyNet", "ValueNet"]
