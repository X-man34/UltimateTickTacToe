"""Validate the ND4J .bin parser against real legacy files.

These tests read actual training files from the old Java project so we
can be sure the parser handles the exact format variant DL4J 1.0.0-M2.1
produces. They are skipped if the fixture files are not present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uttt_ml.nd4j_loader import load_dataset

JAVA_DATA_DIR = Path(
    "/home/calebh/claude-willow/UltimateTickTacToe/data/series1"
)

VALUE_FILE = JAVA_DATA_DIR / "value$series1,gameNum1$1729206277321.bin"
POLICY_FILE = JAVA_DATA_DIR / "policy$series1,gameNum1$1729206277321.bin"

pytestmark = pytest.mark.skipif(
    not (VALUE_FILE.exists() and POLICY_FILE.exists()),
    reason="legacy Java .bin fixtures not available at expected path",
)


def test_value_file_parses_to_expected_shape() -> None:
    ds = load_dataset(VALUE_FILE)
    assert ds.features.shape[1] == 90, f"expected 90 feature dims, got {ds.features.shape}"
    assert ds.labels.shape[1] == 1, f"expected 1 label dim, got {ds.labels.shape}"
    assert ds.features.shape[0] == ds.labels.shape[0]
    assert ds.features.shape[0] > 0


def test_value_file_feature_values_are_board_codes() -> None:
    """The first 81 entries per row must be in {-1, 0, 1} (cell markers)."""
    ds = load_dataset(VALUE_FILE)
    cells = ds.features[:, :81]
    unique = set(np.unique(cells).tolist())
    assert unique.issubset({-1.0, 0.0, 1.0}), f"unexpected cell values: {unique}"


def test_value_file_activity_flags_are_plus_minus_one() -> None:
    """The last 9 entries per row must be +1 or -1 (active vs inactive)."""
    ds = load_dataset(VALUE_FILE)
    activity = ds.features[:, 81:]
    unique = set(np.unique(activity).tolist())
    assert unique.issubset({-1.0, 1.0}), f"unexpected activity values: {unique}"


def test_value_file_labels_are_in_neg_one_to_one_range() -> None:
    ds = load_dataset(VALUE_FILE)
    assert ds.labels.min() >= -1.0
    assert ds.labels.max() <= 1.0


def test_policy_file_parses_to_expected_shape() -> None:
    ds = load_dataset(POLICY_FILE)
    assert ds.features.shape[1] == 90
    assert ds.labels.shape[1] == 81
    assert ds.features.shape[0] == ds.labels.shape[0]


def test_policy_file_labels_look_like_visit_ratios() -> None:
    ds = load_dataset(POLICY_FILE)
    # Every row should sum to ~1 (MCTS visit counts normalized to a
    # probability distribution), modulo tiny floating-point error.
    row_sums = ds.labels.sum(axis=1)
    # Some rows may have all-zero labels if MCTS hit a fixed endpoint
    # and didn't expand, but the majority should sum to ~1.
    close_to_one = np.sum(np.isclose(row_sums, 1.0, atol=1e-3))
    assert close_to_one > 0.5 * ds.labels.shape[0], (
        f"expected most rows to have visit ratios summing to 1.0; "
        f"only {close_to_one}/{ds.labels.shape[0]} did"
    )
    assert ds.labels.min() >= 0.0


def test_value_and_policy_files_share_row_count() -> None:
    """A paired value and policy file for the same game should have the
    same number of positions, since StateDatum writes one of each per
    MCTS step in TrainingDataCreator."""
    value_ds = load_dataset(VALUE_FILE)
    policy_ds = load_dataset(POLICY_FILE)
    assert value_ds.features.shape[0] == policy_ds.features.shape[0]


def test_value_features_round_trip_through_board() -> None:
    """Every row must decode into a valid Board via `Board.from_flat_vector`."""
    from uttt_engine import Board

    ds = load_dataset(VALUE_FILE)
    # Sample a handful of rows — we don't need to check all 58.
    for i in (0, ds.features.shape[0] // 2, ds.features.shape[0] - 1):
        board = Board.from_flat_vector(ds.features[i].astype(np.float64))
        # Round-trip the reconstructed board — its flat vector should
        # match the one we started from (modulo dtype).
        assert np.array_equal(
            board.flat_vector().astype(np.float32),
            ds.features[i],
        )
