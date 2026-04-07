"""
StateDatum — a single per-position training sample collected during self-play.

Mirrors `StateDatum.java` (machinelearning/simulation/StateDatum.java).
Each move of a self-play game produces one StateDatum carrying:

    - The 90-element flat board vector (from the perspective of the
      player to move), which is what the value network is trained on.
    - An 81-element vector of MCTS visit ratios, one per cell, which is
      what the policy network is trained on. Illegal moves are 0.
    - The final game result (set after the game ends), sign-flipped so
      it is always from the perspective of the player who was to move
      at the moment this state was recorded.
    - A `is_player_one_turn_originally` flag tracking whose turn it was
      before the normalization, used at end-of-game to decide whether
      to flip the game outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uttt_bot.mcts_node import MCTSNode
from uttt_engine import Board
from uttt_engine.board import FLAT_VECTOR_SIZE, flat_index


@dataclass
class StateDatum:
    """One training sample — a board + its MCTS-derived policy target."""

    policy_output: np.ndarray  # shape (81,), MCTS visit ratios
    value_input: np.ndarray  # shape (90,), canonical flat vector
    is_player_one_turn_originally: bool
    eval_for_this_state: float = 0.0
    _board_repr: Board | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_mcts_root(
        cls,
        board: Board,
        root: MCTSNode,
        original_is_player_one_turn: bool,
    ) -> "StateDatum":
        """Extract a StateDatum from a completed MCTS search.

        Mirrors `StateDatum(BoardState, GenericTreeNode, boolean, int)`
        in the Java code (StateDatum.java lines 45-82). The board is
        assumed to already be oriented so the player to move is P1
        (MCTS handles the inversion). We read the visit counts of the
        root's children and normalize by the root's total visits to
        build an 81-element policy vector.

        Args:
            board: Normalized board (player-to-move == P1).
            root: The root node returned by the MCTS search.
            original_is_player_one_turn: Whose turn it was in the
                caller's un-normalized board. Stored so the game-
                outcome labeling at the end of the game can invert the
                score appropriately.

        Returns:
            A StateDatum with `value_input`, `policy_output`,
            `is_player_one_turn_originally`, and `eval_for_this_state`
            initialized to zero. The caller is responsible for setting
            the final game evaluation once the game ends.
        """
        # Value input — the 90-element flat vector from the P1-to-move
        # perspective. This is already the canonical format the old
        # .bin files use, so new self-play data is byte-for-byte
        # compatible with the dataset loaders.
        value_input = board.flat_vector().astype(np.float64)

        # Policy output — visit ratios per action. Unvisited actions
        # (illegal or unexpanded) get 0. Matches the Java logic which
        # iterates over every possible (majRow, majCol, minRow, minCol)
        # and writes visits/parent_visits for each child that took the
        # corresponding action.
        policy = np.zeros(FLAT_VECTOR_SIZE - 9, dtype=np.float64)  # length 81
        parent_visits = max(root.visits, 1)
        for child in root.children:
            action = child.action
            if action is None:
                continue
            idx = flat_index(
                action.major_row,
                action.major_col,
                action.minor_row,
                action.minor_col,
            )
            policy[idx] = child.visits / parent_visits

        return cls(
            policy_output=policy,
            value_input=value_input,
            is_player_one_turn_originally=original_is_player_one_turn,
        )


__all__ = ["StateDatum"]
