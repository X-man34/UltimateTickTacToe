"""
Board encoders — convert a Board into tensors consumable by the neural networks.

There are two encodings, one per network:

  - `board_to_value_tensor(board)` returns a dense 5D tensor in NDHWC order
    `(9, 3, 3, 4)`. This matches `Resources.getValueNetworkInputV1()` in the
    Java code (Resources.java lines 84-126). The four channels are X presence,
    O presence, empty presence, and sub-board activity.

  - `board_to_policy_image(board)` returns a 3D image tensor in channels-first
    `(3, 76, 76)` order, normalized to [0, 1]. This reproduces
    `Resources.getImageForAI(state, scaleFactor=1)` (Resources.java lines
    151-252), pixel-for-pixel up to a deliberate BGR->RGB swap. Because the
    old DL4J model weights are discarded and we retrain from scratch, the
    color order only needs to be self-consistent.

Both encoders assume the caller has already oriented the board so the
player to move is represented as P1. This matches the Java convention and
keeps the networks seeing a consistent perspective.

The encoders intentionally return NumPy arrays rather than torch tensors.
This lets the training data pipeline use them without importing torch, and
lets `nets.py` own the batch-dim / device-transfer responsibility.
"""

from __future__ import annotations

import numpy as np

from uttt_engine import Board, Marker

# ----------------------------------------------------------------------
# Value net encoder
# ----------------------------------------------------------------------

#: Shape of the value-net input tensor in NDHWC order (as in the Java code).
VALUE_TENSOR_SHAPE = (9, 3, 3, 4)

# Channel layout (matches Resources.java lines 73-77).
CHANNEL_X = 0      # 1 where the cell contains player one ("X")
CHANNEL_O = 1      # 1 where the cell contains player two ("O")
CHANNEL_EMPTY = 2  # 1 where the cell is empty
CHANNEL_ACTIVE = 3 # 1 across every cell of an active sub-board


def board_to_value_tensor(board: Board) -> np.ndarray:
    """Encode a Board as a dense (9, 3, 3, 4) float32 tensor.

    The first axis walks the nine sub-boards in row-major order (sub-board
    index = major_row * 3 + major_col). The next two axes are the minor
    row and column within that sub-board. The last axis holds the four
    channels described at module top.

    The returned array is in NDHWC order to match the Java code and the
    layout encoded in the legacy .bin training data. The net wrapper in
    `nets.py` is responsible for transposing to PyTorch's NCDHW order and
    adding a batch dimension before calling `Conv3d`.

    Args:
        board: A Board, typically already inverted so the player-to-move
            is P1 (the caller in MCTS and in the training pipeline handles
            the orientation flip).

    Returns:
        A float32 array of shape (9, 3, 3, 4).
    """
    tensor = np.zeros(VALUE_TENSOR_SHAPE, dtype=np.float32)
    for major_row in range(Board.SIZE):
        for major_col in range(Board.SIZE):
            sub = board.sub_board_at(major_row, major_col)
            sub_idx = major_row * Board.SIZE + major_col
            active_val = 1.0 if sub.active else 0.0
            for minor_row in range(Board.SIZE):
                for minor_col in range(Board.SIZE):
                    # Every cell of an active sub-board is hot in the
                    # activity channel, per Resources.java line 77.
                    tensor[sub_idx, minor_row, minor_col, CHANNEL_ACTIVE] = active_val

                    value = sub.item_at(minor_row, minor_col)
                    if value == 0:
                        tensor[sub_idx, minor_row, minor_col, CHANNEL_EMPTY] = 1.0
                    elif value == 1:
                        tensor[sub_idx, minor_row, minor_col, CHANNEL_X] = 1.0
                    else:  # value == -1
                        tensor[sub_idx, minor_row, minor_col, CHANNEL_O] = 1.0
    return tensor


def value_vector_to_tensor(flat_vector: np.ndarray) -> np.ndarray:
    """Reshape a 90-element flat vector into the (9, 3, 3, 4) value tensor.

    Used by `uttt_ml.dataset.ValueDataset` to convert legacy .bin-derived
    data (which is stored as 90-element flat vectors) into the shape the
    value net expects, without having to reconstruct a full Board first.

    The conversion is equivalent to calling
    `Board.from_flat_vector(flat_vector)` followed by
    `board_to_value_tensor(board)`, but faster because it operates
    directly on the input array.
    """
    from uttt_engine.board import activity_index, flat_index

    if flat_vector.shape != (90,):
        raise ValueError(f"expected shape (90,), got {flat_vector.shape}")
    tensor = np.zeros(VALUE_TENSOR_SHAPE, dtype=np.float32)
    for major_row in range(Board.SIZE):
        for major_col in range(Board.SIZE):
            sub_idx = major_row * Board.SIZE + major_col
            active = 1.0 if flat_vector[activity_index(major_row, major_col)] > 0 else 0.0
            for minor_row in range(Board.SIZE):
                for minor_col in range(Board.SIZE):
                    tensor[sub_idx, minor_row, minor_col, CHANNEL_ACTIVE] = active
                    value = int(round(float(
                        flat_vector[flat_index(major_row, major_col, minor_row, minor_col)]
                    )))
                    if value == 0:
                        tensor[sub_idx, minor_row, minor_col, CHANNEL_EMPTY] = 1.0
                    elif value == 1:
                        tensor[sub_idx, minor_row, minor_col, CHANNEL_X] = 1.0
                    else:
                        tensor[sub_idx, minor_row, minor_col, CHANNEL_O] = 1.0
    return tensor


# ----------------------------------------------------------------------
# Policy net encoder — the 76x76 "image of the board"
# ----------------------------------------------------------------------

# Layout constants lifted from Resources.getImageForAI() with scaleFactor = 1.
_SCALE_FACTOR = 1
_MARKER_SIZE = 4 * _SCALE_FACTOR         # 4
_MAIN_BORDER_WIDTH = int(3 / 4 * _MARKER_SIZE)  # 3
_SUB_BOARD_LINE_WIDTH = int(0.25 * _MARKER_SIZE)  # 1
_MAJOR_BOARD_LINE_WIDTH = int(0.5 * _MARKER_SIZE)  # 2
_SUB_BOARD_SIZE = 3 * _MARKER_SIZE + 10 * _SUB_BOARD_LINE_WIDTH  # 22
_IMAGE_SIZE = 3 * _SUB_BOARD_SIZE + 2 * _MAIN_BORDER_WIDTH + 2 * _MAJOR_BOARD_LINE_WIDTH  # 76

#: Side length (pixels) of the rendered policy-net image.
POLICY_IMAGE_SIZE = _IMAGE_SIZE
assert _IMAGE_SIZE == 76, f"expected 76x76 policy image, got {_IMAGE_SIZE}"

# Colors as (R, G, B) triples, 0..255. Note that the Java code uses
# TYPE_3BYTE_BGR — the raw byte order on disk is different — but the
# semantic colors (red for P1, green for P2, etc.) are identical, so we
# write them as RGB here. Since we retrain all models from scratch the
# absolute channel order only has to be self-consistent.
_COLOR_WHITE = (255, 255, 255)
_COLOR_BLACK = (0, 0, 0)
_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)
_COLOR_YELLOW = (255, 255, 0)

#: Evaluation values that signal a finished sub-board — used to decide
#: whether to paint a terminal-state overlay instead of the grid.
_P1_WIN_VAL = 1.0
_P2_WIN_VAL = -1.0


def _fill_rect(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    """Paint a filled rectangle into an HWC image array.

    Mirrors `Graphics2D.fillRect(x, y, w, h)`. Clips to image bounds so
    minor arithmetic rounding in the layout code cannot crash the encoder.
    """
    h, w = image.shape[0], image.shape[1]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + width)
    y1 = min(h, y + height)
    if x1 <= x0 or y1 <= y0:
        return
    image[y0:y1, x0:x1, 0] = color[0]
    image[y0:y1, x0:x1, 1] = color[1]
    image[y0:y1, x0:x1, 2] = color[2]


def board_to_policy_image(board: Board) -> np.ndarray:
    """Render a Board as a (3, 76, 76) float32 image tensor in [0, 1].

    This is a line-for-line port of `Resources.getImageForAI()` in the
    Java code (Resources.java lines 151-252), with the BGR output of the
    Java BufferedImage swapped to RGB for NumPy / PyTorch convention.

    The image is first built in HWC order then transposed to CHW at the
    end, and pixel values are normalized to [0, 1] by dividing by 255.
    Everything outside the grid is white; the outer border, major grid
    lines, and sub-board grid lines are black; active in-progress
    sub-boards get a blue background; sub-boards won by P1 or P2 get a
    solid red or green overlay; drawn sub-boards get yellow.

    Args:
        board: A Board oriented so the player to move is P1. (The policy
            network takes perspective into account via this same
            convention that the value net uses.)

    Returns:
        A float32 array of shape (3, 76, 76) with values in [0, 1].
    """
    size = _IMAGE_SIZE
    img = np.full((size, size, 3), 255, dtype=np.uint8)  # white background

    # Outer frame: four black rectangles on the edges, mirroring
    # Resources.java lines 164-167.
    _fill_rect(img, 0, 0, _MAIN_BORDER_WIDTH, size, _COLOR_BLACK)  # left
    _fill_rect(img, 0, 0, size, _MAIN_BORDER_WIDTH, _COLOR_BLACK)  # top
    _fill_rect(
        img, 0, size - _MAIN_BORDER_WIDTH, size, _MAIN_BORDER_WIDTH, _COLOR_BLACK
    )  # bottom
    _fill_rect(
        img, size - _MAIN_BORDER_WIDTH, 0, _MAIN_BORDER_WIDTH, size, _COLOR_BLACK
    )  # right

    # Draw every sub-board. The Java code draws major-board grid lines
    # inside this same loop, once per row iteration. We mirror that.
    for row in range(3):
        # Major-board grid lines between rows (and between columns by
        # reusing the same index). Java draws these first, before the
        # sub-board contents, so they sit under the fills that follow.
        _fill_rect(
            img,
            0,
            _MAIN_BORDER_WIDTH
            + (row + 1) * _SUB_BOARD_SIZE
            + row * _MAJOR_BOARD_LINE_WIDTH,
            size,
            _MAJOR_BOARD_LINE_WIDTH,
            _COLOR_BLACK,
        )
        _fill_rect(
            img,
            _MAIN_BORDER_WIDTH
            + (row + 1) * _SUB_BOARD_SIZE
            + row * _MAJOR_BOARD_LINE_WIDTH,
            0,
            _MAJOR_BOARD_LINE_WIDTH,
            size,
            _COLOR_BLACK,
        )

        for col in range(3):
            column_offset = (
                _MAIN_BORDER_WIDTH
                + col * _SUB_BOARD_SIZE
                + col * _MAJOR_BOARD_LINE_WIDTH
            )
            row_offset = (
                _MAIN_BORDER_WIDTH
                + row * _SUB_BOARD_SIZE
                + row * _MAJOR_BOARD_LINE_WIDTH
            )

            sub = board.sub_board_at(row, col)
            evaluation = sub.evaluation()

            if evaluation != 0.0:
                # Terminal sub-board: paint a solid overlay (red, green,
                # or yellow) and skip the grid + marker rendering.
                if evaluation == _P1_WIN_VAL:
                    overlay = _COLOR_RED
                elif evaluation == _P2_WIN_VAL:
                    overlay = _COLOR_GREEN
                else:
                    overlay = _COLOR_YELLOW
                _fill_rect(
                    img,
                    column_offset + _SUB_BOARD_LINE_WIDTH,
                    row_offset + _SUB_BOARD_LINE_WIDTH,
                    _SUB_BOARD_SIZE - 2 * _SUB_BOARD_LINE_WIDTH,
                    _SUB_BOARD_SIZE - 2 * _SUB_BOARD_LINE_WIDTH,
                    overlay,
                )
                continue

            # In-progress sub-board. If it's active, lay down a blue
            # background; then draw grid lines and markers on top.
            if sub.active:
                _fill_rect(
                    img,
                    column_offset,
                    row_offset,
                    _SUB_BOARD_SIZE,
                    _SUB_BOARD_SIZE,
                    _COLOR_BLUE,
                )

            # Sub-board internal grid lines (between the nine cells) plus
            # the markers. The Java code draws both in one loop.
            for min_row in range(3):
                if min_row < 2:
                    # Horizontal grid line after rows 0 and 1.
                    _fill_rect(
                        img,
                        column_offset + _SUB_BOARD_LINE_WIDTH,
                        row_offset
                        + _SUB_BOARD_LINE_WIDTH
                        + (min_row + 1) * (2 * _SUB_BOARD_LINE_WIDTH + _MARKER_SIZE)
                        + min_row * _SUB_BOARD_LINE_WIDTH,
                        _SUB_BOARD_SIZE - 2 * _SUB_BOARD_LINE_WIDTH,
                        _SUB_BOARD_LINE_WIDTH,
                        _COLOR_BLACK,
                    )
                    # Vertical grid line after cols 0 and 1.
                    _fill_rect(
                        img,
                        column_offset
                        + (min_row + 1) * (_MARKER_SIZE + 3 * _SUB_BOARD_LINE_WIDTH),
                        row_offset + _SUB_BOARD_LINE_WIDTH,
                        _SUB_BOARD_LINE_WIDTH,
                        _SUB_BOARD_SIZE - 2 * _SUB_BOARD_LINE_WIDTH,
                        _COLOR_BLACK,
                    )
                for min_col in range(3):
                    marker_col_offset = (
                        column_offset
                        + 2 * _SUB_BOARD_LINE_WIDTH
                        + min_col * _MARKER_SIZE
                        + min_col * 3 * _SUB_BOARD_LINE_WIDTH
                    )
                    marker_row_offset = (
                        row_offset
                        + 2 * _SUB_BOARD_LINE_WIDTH
                        + min_row * _MARKER_SIZE
                        + min_row * 3 * _SUB_BOARD_LINE_WIDTH
                    )
                    value = sub.item_at(min_row, min_col)
                    if value == 1:
                        _fill_rect(
                            img,
                            marker_col_offset,
                            marker_row_offset,
                            _MARKER_SIZE,
                            _MARKER_SIZE,
                            _COLOR_RED,
                        )
                    elif value == -1:
                        _fill_rect(
                            img,
                            marker_col_offset,
                            marker_row_offset,
                            _MARKER_SIZE,
                            _MARKER_SIZE,
                            _COLOR_GREEN,
                        )

    # HWC uint8 -> CHW float32 in [0, 1]. PyTorch conv layers expect
    # (C, H, W) per-sample and a leading batch dim added by the caller.
    chw = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
    return chw


# Alias for parity with the Java API, which has both getValueNetworkInputV1
# and getPolicyNetworkInputV1 (the latter currently delegates to the former
# — but the image-based input is the one PolicyNetworkTrainer actually
# trains on, so it is what we expose here).
def board_to_policy_tensor(board: Board) -> np.ndarray:
    """Alias for `board_to_policy_image`, matching the policy-net naming."""
    return board_to_policy_image(board)


__all__ = [
    "POLICY_IMAGE_SIZE",
    "VALUE_TENSOR_SHAPE",
    "board_to_policy_image",
    "board_to_policy_tensor",
    "board_to_value_tensor",
    "value_vector_to_tensor",
]
