"""
MCTSNode — a single node in the MCTS search tree.

Mirrors `NodeData.java` and `GenericTreeNode.java` combined into one class.
Each node owns a full `Board` clone (the Java project has a BitSet-compressed
variant too, but the default path stores the full state); a visit count;
a cumulative score; a virtual-loss counter used by threaded MCTS to steer
concurrent workers onto different branches; a parent pointer; and a list
of children.

Thread safety: mutations to `visits`, `total_score`, `virtual_loss`, and
`children` happen under `self.lock`. Readers that only need a snapshot
(e.g. the `best_child_ucb1` scan) also hold the parent's lock so the
children list does not race with `expand`.
"""

from __future__ import annotations

import math
import threading

from uttt_engine import Board, Evaluation, Move


class MCTSNode:
    """A node in the Monte Carlo tree search tree."""

    __slots__ = (
        "action",
        "state",
        "visits",
        "total_score",
        "virtual_loss",
        "parent",
        "children",
        "is_expanded",
        "lock",
    )

    def __init__(
        self,
        state: Board,
        parent: "MCTSNode | None" = None,
        action: Move | None = None,
    ) -> None:
        """Create an unvisited node.

        Args:
            state: The board state at this node. The caller is
                responsible for cloning — MCTSNode does not copy.
            parent: The parent node, or None for the root.
            action: The move that led from `parent` to `state`, or None
                at the root. Used for parent-child action matching when
                building training data.
        """
        self.action = action
        self.state = state
        self.visits = 0
        self.total_score = 0.0
        self.virtual_loss = 0
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.is_expanded = False
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def expand(self) -> None:
        """Materialize children for every legal move.

        Mirrors `MCTSEvaluator.addChildren()` (MCTSEvaluator.java lines
        379-393). Never creates children for terminal states — the game
        is already decided there. Callers check `is_expanded` to avoid
        re-expanding the same node when multiple MCTS iterations race.
        """
        with self.lock:
            if self.is_expanded:
                return
            # Don't expand terminal states — there are no more legal moves.
            if self.state.is_terminal():
                self.is_expanded = True
                return
            for move in self.state.legal_moves():
                child_state = self.state.simulate(move)
                self.children.append(
                    MCTSNode(state=child_state, parent=self, action=move)
                )
            self.is_expanded = True

    # ------------------------------------------------------------------
    # UCB1 selection
    # ------------------------------------------------------------------

    def ucb1(self, c: float, parent_visits: int) -> float:
        """Return this node's UCB1 score relative to the given `parent_visits`.

        Mirrors `MCTSEvaluator.getUCB1()` (MCTSEvaluator.java lines 363-369):
        an unvisited node scores +infinity (guaranteed to be picked first),
        otherwise it's the mean score plus an exploration bonus. The
        virtual-loss term is subtracted from the numerator to temporarily
        penalize branches that another thread is currently exploring.
        """
        effective_visits = self.visits + self.virtual_loss
        if effective_visits <= 0:
            return math.inf
        exploitation = (self.total_score - self.virtual_loss) / effective_visits
        exploration = c * math.sqrt(
            math.log(max(parent_visits, 1)) / effective_visits
        )
        return exploitation + exploration

    def best_child_ucb1(self, c: float) -> "MCTSNode":
        """Return the child with the highest UCB1 score.

        Caller must already hold `self.lock` — the loop reads each
        child's mutable state via its own lock as well.
        """
        parent_visits = self.visits
        best: MCTSNode | None = None
        best_score = -math.inf
        for child in self.children:
            with child.lock:
                score = child.ucb1(c, parent_visits)
            if score > best_score:
                best = child
                best_score = score
                if math.isinf(best_score) and best_score > 0:
                    break  # unvisited child — no point looking further
        assert best is not None, "best_child_ucb1 called on a node with no children"
        return best

    # ------------------------------------------------------------------
    # Terminal handling
    # ------------------------------------------------------------------

    def terminal_evaluation(self) -> float | None:
        """Return the terminal value of this node, or None if in-progress."""
        eval_value = self.state.evaluation()
        if Evaluation.is_terminal(eval_value):
            return eval_value
        return None


__all__ = ["MCTSNode"]
