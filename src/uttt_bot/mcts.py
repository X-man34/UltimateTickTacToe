"""
MCTSEvaluator — Monte Carlo Tree Search over Ultimate Tic Tac Toe.

Ports `MCTSEvaluator.java` from the original Java project. The algorithm
is the standard MCTS four-phase loop (select, expand, evaluate,
backpropagate), with two optional neural-network hooks:

    - The value network replaces random rollouts as the leaf-evaluation
      strategy. Mirrors `MCTSEvaluator.evaluateEndPoint()`.
    - The policy network replaces UCB1 as the child-selection strategy.
      Mirrors `MCTSEvaluator.getBestChild()`.

Either or both can be omitted, in which case the evaluator falls back
to random rollouts and UCB1.

## Parallelism

The Java side uses a `ThreadPoolExecutor` and keeps roughly 25 tasks in
flight at once. We mirror that design with `concurrent.futures.ThreadPoolExecutor`
and a virtual-loss scheme so threads do not all descend the same branch.
Virtual loss works like this:

    1. During selection, every node on the path from root to leaf gets
       its `virtual_loss` counter incremented by 1.
    2. UCB1 treats a node with virtual loss as if it has more visits and
       a worse score, so other threads are nudged toward different
       branches.
    3. During backpropagation, the virtual loss on every node along the
       path is decremented by the same amount.

Node mutations happen under per-node locks. NN inference happens on the
caller's thread but releases the GIL inside PyTorch, so multi-threaded
MCTS with a value net scales roughly linearly up to the GPU/CPU bottleneck.

## Perspective handling

Every search runs with the board oriented so the player to move is P1.
The caller's board is inverted once before the search if necessary;
the returned move's marker is flipped back to the caller's perspective
before returning.
"""

from __future__ import annotations

import math
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from uttt_bot.config import MCTSConfig
from uttt_bot.encoders import board_to_policy_image, board_to_value_tensor
from uttt_bot.mcts_node import MCTSNode
from uttt_bot.nets import PolicyNet, ValueNet
from uttt_bot.rollout import random_rollout
from uttt_engine import Board, Marker, Move
from uttt_engine.board import flat_index


# ----------------------------------------------------------------------
# Search result wrapper
# ----------------------------------------------------------------------

@dataclass
class SearchResult:
    """What a completed MCTS search returns."""

    best_move: Move
    root: MCTSNode
    num_iterations: int


# ----------------------------------------------------------------------
# Main evaluator
# ----------------------------------------------------------------------

class MCTSEvaluator:
    """Monte Carlo Tree Search with optional value/policy network guidance."""

    def __init__(
        self,
        config: MCTSConfig | None = None,
        value_net: ValueNet | None = None,
        policy_net: PolicyNet | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Create an evaluator.

        Args:
            config: MCTSConfig. If None, a default-initialized config is
                used (1 second compute time, UCB1 C=2.0, 4 threads).
            value_net: Pre-loaded ValueNet, or None to use random rollouts.
            policy_net: Pre-loaded PolicyNet, or None to use UCB1.
            device: Device to run NN inference on.
        """
        self.config = config or MCTSConfig()
        self.device = torch.device(device)
        self.value_net = value_net
        self.policy_net = policy_net
        if self.value_net is not None:
            self.value_net.to(self.device).eval()
        if self.policy_net is not None:
            self.policy_net.to(self.device).eval()

        # Master RNG used for thread-local RNG seeding. Each call to
        # `_thread_rng()` returns a fresh `random.Random` derived from
        # this one so threads don't share mutable RNG state. Seeded
        # from config.seed if provided.
        self._master_rng = random.Random(self.config.seed)
        self._master_rng_lock = threading.Lock()
        self._thread_local = threading.local()

        # NN inference must be serialized across threads — PyTorch
        # models are not re-entrant. The lock is released around CPU
        # work so selection/backprop can still run in parallel.
        self._nn_lock = threading.Lock()

    def _thread_rng(self) -> random.Random:
        """Return a `random.Random` private to the current thread.

        The per-thread RNG is seeded once from the master RNG on first
        access. This keeps multi-threaded rollouts from stomping on
        each other's RNG state while still letting the user get
        reproducible behavior by passing `seed` to the config.
        """
        rng = getattr(self._thread_local, "rng", None)
        if rng is None:
            with self._master_rng_lock:
                seed = self._master_rng.randrange(2**63)
            rng = random.Random(seed)
            self._thread_local.rng = rng
        return rng

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def search(self, board: Board) -> SearchResult:
        """Run MCTS on `board` for the configured compute time.

        The board is never mutated. The returned `best_move` is in the
        caller's perspective — if the caller's turn is P2, the move's
        marker is P2 even though MCTS internally ran on an inverted
        board.
        """
        # Orient the search so the player to move is always P1.
        inverted = not board.player_one_turn
        working_board = board.clone()
        if inverted:
            working_board.invert()

        root = MCTSNode(state=working_board)
        deadline = time.monotonic() + self.config.compute_time_seconds

        if self.config.num_threads <= 1:
            num_iterations = self._search_single_threaded(root, deadline)
        else:
            num_iterations = self._search_multi_threaded(root, deadline)

        # Find the child with the most visits, matching the Java
        # getCurrentBestMove (MCTSEvaluator.java lines 455-467).
        best_move = self._most_visited_action(root)
        if best_move is None:
            # Should only happen if the position is already terminal,
            # in which case the caller shouldn't have asked.
            raise RuntimeError("No legal moves available from the root position.")

        # Flip the marker back to the caller's perspective.
        if inverted:
            best_move = Move(
                major_row=best_move.major_row,
                major_col=best_move.major_col,
                minor_row=best_move.minor_row,
                minor_col=best_move.minor_col,
                marker=Marker.opponent(best_move.marker),
            )

        return SearchResult(
            best_move=best_move, root=root, num_iterations=num_iterations
        )

    # ------------------------------------------------------------------
    # Single-threaded and multi-threaded loops
    # ------------------------------------------------------------------

    def _search_single_threaded(self, root: MCTSNode, deadline: float) -> int:
        """Run iterations one at a time until `deadline` passes."""
        iterations = 0
        while time.monotonic() < deadline:
            self._perform_iteration(root)
            iterations += 1
        return iterations

    def _search_multi_threaded(self, root: MCTSNode, deadline: float) -> int:
        """Run iterations in parallel using a ThreadPoolExecutor.

        Mirrors `MCTSEvaluator.java` lines 86-185: keep roughly
        `task_queue_depth` iterations in flight, and stop submitting
        new ones once `deadline` has elapsed.
        """
        iterations = 0
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as pool:
            in_flight: set[Future] = set()

            def submit() -> None:
                nonlocal iterations
                in_flight.add(pool.submit(self._perform_iteration, root))
                iterations += 1

            # Prime the queue.
            for _ in range(self.config.task_queue_depth):
                if time.monotonic() >= deadline:
                    break
                submit()

            # Drain completed tasks and refill until the deadline.
            while in_flight:
                done: set[Future] = set()
                for fut in list(in_flight):
                    if fut.done():
                        done.add(fut)
                for fut in done:
                    in_flight.remove(fut)
                    # Re-raise exceptions so they don't get swallowed.
                    fut.result()
                    if time.monotonic() < deadline:
                        submit()
                if not done:
                    # Short sleep to avoid a busy wait while tasks run.
                    time.sleep(0.0005)

        return iterations

    # ------------------------------------------------------------------
    # One MCTS iteration
    # ------------------------------------------------------------------

    def _perform_iteration(self, root: MCTSNode) -> None:
        """Run one selection/expansion/evaluation/backprop cycle.

        Mirrors `MCTSEvaluator.preformIteration()` (MCTSEvaluator.java
        lines 193-239). The path and virtual-loss bookkeeping are
        needed so the multi-threaded version can undo the loss during
        backprop.

        Flow:
            1. Walk from root using UCB1/policy selection until we hit
               a node that either has never been visited or has no
               expanded children yet.
            2. If unvisited, evaluate it directly.
            3. If visited but unexpanded and not terminal, expand,
               descend to one of its children, and evaluate that.
            4. Backpropagate the score up `path`, flipping sign each
               level and undoing the virtual loss.
        """
        # --- Selection ---
        path: list[MCTSNode] = [root]
        node = root
        self._apply_virtual_loss(node)
        while True:
            with node.lock:
                has_children = bool(node.children)
                visits_snapshot = node.visits
            # Leaf if either unvisited or expanded-but-childless (terminal).
            if visits_snapshot == 0 or not has_children:
                break
            with node.lock:
                node = node.best_child_ucb1(self.config.c_ucb1)
            path.append(node)
            self._apply_virtual_loss(node)

        # --- Expansion + evaluation ---
        # At this point `node` is either unvisited (visits==0) or
        # visited-and-unexpanded-and-terminal (visits>0, no children).
        score_to_add: float
        if visits_snapshot == 0:
            # First visit — evaluate the leaf in place.
            score_to_add = self._evaluate_leaf(node.state)
        else:
            # Visited at least once before. If it isn't terminal,
            # expand it now and descend to one of its children.
            if not node.state.is_terminal():
                node.expand()
                with node.lock:
                    has_children_now = bool(node.children)
                if has_children_now:
                    child = self._select_child_for_expansion(node)
                    path.append(child)
                    self._apply_virtual_loss(child)
                    node = child
                    score_to_add = self._evaluate_leaf(node.state)
                else:
                    # Expansion produced no children — treat as terminal.
                    score_to_add = node.state.evaluation()
            else:
                # Terminal state: use the static evaluation directly.
                score_to_add = node.state.evaluation()

        # --- Backpropagation ---
        # Start from the leaf and walk back up. Each step flips the
        # score sign to reflect the alternating perspective and undoes
        # the virtual loss applied during selection.
        sign = 1.0
        for n in reversed(path):
            with n.lock:
                n.visits += 1
                n.total_score += sign * score_to_add
                if self.config.num_threads > 1:
                    n.virtual_loss = max(0, n.virtual_loss - 1)
            sign = -sign

    def _apply_virtual_loss(self, node: MCTSNode) -> None:
        """Add a virtual-loss pending-penalty to a node during selection."""
        if self.config.num_threads > 1:
            with node.lock:
                node.virtual_loss += 1

    def _select_child_for_expansion(self, parent: MCTSNode) -> MCTSNode:
        """Pick a child of a freshly expanded node.

        If a policy net is configured, sample from its distribution;
        otherwise return the first child, matching the Java default
        in `preformIteration()` line 219.
        """
        if self.policy_net is not None:
            return self._policy_sample_child(parent)
        return parent.children[0]

    # ------------------------------------------------------------------
    # Leaf evaluation
    # ------------------------------------------------------------------

    def _evaluate_leaf(self, state: Board) -> float:
        """Return a score for `state` — via value net if available, else rollout.

        Mirrors `MCTSEvaluator.evaluateEndPoint()` (MCTSEvaluator.java
        lines 404-418). The value net sees a board normalized to
        player-one-to-move, so we flip the sign based on whose turn it
        really is if the caller's board is inverted. (Inside this
        method, `state.player_one_turn` is always True by construction
        because the root was normalized.)
        """
        if state.is_terminal():
            return state.evaluation()

        if self.value_net is not None:
            tensor = board_to_value_tensor(state)
            with self._nn_lock:
                batch = ValueNet.encode_batch([tensor], device=self.device)
                with torch.no_grad():
                    output = self.value_net(batch)
                value = float(output[0, 0].item())
            # The board is already oriented player-one-to-move, so no
            # extra sign flip is needed — that is all handled at the
            # backprop level by alternating `sign` each depth.
            return value

        return random_rollout(
            state, max_depth=self.config.max_rollout_depth, rng=self._thread_rng()
        )

    # ------------------------------------------------------------------
    # Policy-net child sampling
    # ------------------------------------------------------------------

    def _policy_sample_child(self, parent: MCTSNode) -> MCTSNode:
        """Sample one of `parent.children` using the policy network.

        Mirrors `MCTSEvaluator.getChildToExploreFromPolicyNetworkOutput()`:
        mask illegal moves to 0, renormalize to a probability mass
        function, sample via the cumulative distribution, and fall back
        to the first legal move if the sampled action is illegal.
        """
        assert self.policy_net is not None
        state = parent.state
        image = board_to_policy_image(state)
        with self._nn_lock:
            batch = PolicyNet.encode_batch([image], device=self.device)
            with torch.no_grad():
                output = self.policy_net(batch)
            scores = output[0].detach().cpu().numpy().astype(np.float64)

        # Mask illegal moves to zero. The legal-moves list gives us
        # exactly the legal indices, so set everything else to 0 and
        # clamp negatives (the identity-activated net can produce them).
        mask = np.zeros(81, dtype=np.float64)
        for move in state.legal_moves():
            idx = flat_index(move.major_row, move.major_col, move.minor_row, move.minor_col)
            mask[idx] = 1.0
        scores = np.maximum(scores, 0.0) * mask
        total = scores.sum()
        if total <= 0.0:
            # Degenerate distribution — fall back to the first child.
            return parent.children[0]
        scores /= total

        sampled_idx = self._sample_from_pmf(scores)
        # Find the child whose action matches the sampled flat index.
        for child in parent.children:
            assert child.action is not None
            child_idx = flat_index(
                child.action.major_row,
                child.action.major_col,
                child.action.minor_row,
                child.action.minor_col,
            )
            if child_idx == sampled_idx:
                return child
        # Shouldn't happen after masking, but fall back defensively.
        return parent.children[0]

    def _sample_from_pmf(self, pmf: np.ndarray) -> int:
        """Sample a single index from a probability mass function.

        Uses a cumulative-sum + binary-search pattern, matching
        `MCTSEvaluator.getActionFromPMF()` (MCTSEvaluator.java lines
        300-320).
        """
        cumulative = np.cumsum(pmf)
        # Subtract a tiny epsilon so r==1.0 can't sample past the end.
        r = max(self._thread_rng().random() - 1e-9, 0.0)
        # Binary search — numpy provides this directly.
        return int(np.searchsorted(cumulative, r, side="right"))

    # ------------------------------------------------------------------
    # Best-move selection
    # ------------------------------------------------------------------

    def _most_visited_action(self, root: MCTSNode) -> Move | None:
        """Return the action of the most-visited root child, or None."""
        best: MCTSNode | None = None
        best_visits = -math.inf
        with root.lock:
            for child in root.children:
                with child.lock:
                    visits = child.visits
                if visits > best_visits:
                    best = child
                    best_visits = visits
        if best is None or best.action is None:
            return None
        return best.action


# ----------------------------------------------------------------------
# Convenience factory — load nets from paths and build an evaluator
# ----------------------------------------------------------------------

def build_evaluator(
    config: MCTSConfig,
    device: torch.device | str = "cpu",
) -> MCTSEvaluator:
    """Build an `MCTSEvaluator` from paths in the config.

    Skips network loading if the corresponding path is None — MCTS
    falls back to random rollouts / UCB1 in that case.
    """
    value_net: ValueNet | None = None
    policy_net: PolicyNet | None = None
    if config.value_net_path is not None:
        value_net = ValueNet.load(config.value_net_path, device=device)
    if config.policy_net_path is not None:
        policy_net = PolicyNet.load(config.policy_net_path, device=device)
    return MCTSEvaluator(
        config=config,
        value_net=value_net,
        policy_net=policy_net,
        device=device,
    )


__all__ = ["MCTSEvaluator", "SearchResult", "build_evaluator"]
