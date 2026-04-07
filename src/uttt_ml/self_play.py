"""
self_play — generate training data by running MCTS-vs-MCTS games.

Mirrors `TrainingDataCreator.java` and the supporting
`AdversarialGameSimulation` class in the Java project. Each game:

    1. Start with an empty board.
    2. Loop until the board is terminal:
       a. Invert the board so the player to move is P1.
       b. Run MCTS with the configured compute time.
       c. Capture a `StateDatum` from the search root (board +
          visit ratios).
       d. Apply the best move to the true (un-inverted) board.
    3. After the game ends, set `eval_for_this_state` on every datum,
       flipping the sign for positions where it was P2's turn so
       every label is in the perspective of the player who was about
       to move.
    4. Write two `.npz` files — one for the value net (labels = game
       outcome scalars) and one for the policy net (labels = 81-elt
       visit ratio vectors). Both files use the same row order, so
       the N-th row of each file describes the same game state.

The CLI mirrors the interactive prompts from `TrainingDataCreator.java`:
data folder, series name, starting game number, number of games, compute
time per move, and number of MCTS threads. All prompts can also be
passed as flags for non-interactive use.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from uttt_bot.config import MCTSConfig
from uttt_bot.mcts import MCTSEvaluator, build_evaluator
from uttt_engine import Board
from uttt_ml.state_datum import StateDatum


def _prompt(msg: str, default: str | None = None) -> str:
    """Ask the user for input via stdin, with an optional default."""
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{msg}{suffix}: ").strip()
    if not raw and default is not None:
        return default
    return raw


def _run_one_game(
    evaluator: MCTSEvaluator,
    max_plies: int = 200,
) -> list[StateDatum]:
    """Play a full self-play game and return the per-position training data.

    Args:
        evaluator: A configured MCTSEvaluator.
        max_plies: Safety cap on game length. Ultimate Tic Tac Toe
            cannot exceed 81 plies in practice, so 200 is loose.

    Returns:
        A list of StateDatum with `eval_for_this_state` already set
        based on the game outcome.
    """
    board = Board()
    collected: list[StateDatum] = []

    ply = 0
    while ply < max_plies and not board.is_terminal():
        original_p1_turn = board.player_one_turn

        # MCTSEvaluator.search() handles perspective internally — we
        # pass the real board and it normalizes to P1-to-move for the
        # search, then flips the returned move's marker back. The
        # StateDatum, however, needs the normalized form so the flat
        # vector matches the training convention (player-to-move == P1).
        result = evaluator.search(board)
        normalized = board.clone()
        if not original_p1_turn:
            normalized.invert()
        datum = StateDatum.from_mcts_root(
            board=normalized,
            root=result.root,
            original_is_player_one_turn=original_p1_turn,
        )
        collected.append(datum)

        # The returned move already has the correct marker for the
        # caller's un-normalized board (see MCTSEvaluator.search).
        board.apply_move(result.best_move)
        ply += 1

    # Label every position with the final game outcome, flipping the
    # sign for states that were collected during P2's turn so every
    # label is from the perspective of "the player about to move at
    # that moment".
    final_eval = board.evaluation()
    for datum in collected:
        if datum.is_player_one_turn_originally:
            datum.eval_for_this_state = final_eval
        else:
            datum.eval_for_this_state = -final_eval

    return collected


def _save_game(
    data: Iterable[StateDatum],
    out_dir: Path,
    series_name: str,
    game_num: int,
    timestamp: int,
) -> tuple[Path, Path]:
    """Write one game's data to matching value/policy .npz files.

    Filenames mirror the Java naming convention:
        value$<series>,gameNum<N>$<timestamp>.npz
        policy$<series>,gameNum<N>$<timestamp>.npz
    """
    data_list = list(data)
    if not data_list:
        raise ValueError("Cannot save an empty game — no StateDatum collected.")

    value_features = np.stack([d.value_input.astype(np.float32) for d in data_list])
    value_labels = np.array(
        [[d.eval_for_this_state] for d in data_list], dtype=np.float32
    )
    policy_features = value_features.copy()  # same flat vectors
    policy_labels = np.stack([d.policy_output.astype(np.float32) for d in data_list])

    out_dir.mkdir(parents=True, exist_ok=True)
    value_path = out_dir / f"value${series_name},gameNum{game_num}${timestamp}.npz"
    policy_path = out_dir / f"policy${series_name},gameNum{game_num}${timestamp}.npz"
    np.savez_compressed(value_path, features=value_features, labels=value_labels)
    np.savez_compressed(policy_path, features=policy_features, labels=policy_labels)
    return value_path, policy_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point registered as `uttt-self-play`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--series", type=str, default=None)
    parser.add_argument("--start-game", type=int, default=None)
    parser.add_argument("--num-games", type=int, default=None)
    parser.add_argument("--compute-time", type=float, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--value-net", type=Path, default=None)
    parser.add_argument("--policy-net", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use provided flags only — do not prompt for missing values.",
    )
    args = parser.parse_args(argv)

    # Interactive prompts mirror TrainingDataCreator.java lines 37-117.
    def resolve(value, name, default, caster=str):
        if value is not None:
            return value
        if args.non_interactive:
            print(f"error: --{name.replace('_', '-')} is required", file=sys.stderr)
            raise SystemExit(2)
        return caster(_prompt(name, str(default)))

    data_dir: Path = Path(resolve(args.data_dir, "data directory", "./data"))
    series: str = str(resolve(args.series, "series name", "seriesNew"))
    start_game: int = int(resolve(args.start_game, "starting game number", 1, int))
    num_games: int = int(resolve(args.num_games, "number of games", 1, int))
    compute_time: float = float(
        resolve(args.compute_time, "compute time per move (seconds)", 1.0, float)
    )
    num_threads: int = int(resolve(args.num_threads, "number of MCTS threads", 4, int))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    config = MCTSConfig(
        compute_time_seconds=compute_time,
        num_threads=num_threads,
        value_net_path=args.value_net,
        policy_net_path=args.policy_net,
    )
    evaluator = build_evaluator(config, device=device)

    out_dir = data_dir / series
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = data_dir / "logs" / series
    log_dir.mkdir(parents=True, exist_ok=True)

    for g in range(num_games):
        game_num = start_game + g
        print(f"=== game {game_num} ({g + 1}/{num_games}) ===")
        game_start = time.time()
        data = _run_one_game(evaluator)
        elapsed = time.time() - game_start
        timestamp = int(time.time() * 1000)
        value_path, policy_path = _save_game(
            data, out_dir, series, game_num, timestamp
        )
        # Also write a minimal log file so bulk runs leave a paper
        # trail, mirroring the Java side's per-game log files.
        log_file = log_dir / f"game{game_num}${timestamp}.log"
        log_file.write_text(
            f"game_num={game_num}\n"
            f"positions={len(data)}\n"
            f"elapsed_seconds={elapsed:.2f}\n"
            f"value_path={value_path.name}\n"
            f"policy_path={policy_path.name}\n"
        )
        print(f"  {len(data)} positions, {elapsed:.1f}s -> {value_path.name}")

    print(f"done: wrote {num_games} game(s) under {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
