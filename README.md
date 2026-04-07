# Ultimate Tic Tac Toe (Python)

A Python port of the original Java Ultimate Tic Tac Toe project. Ships:

- A PySide6 desktop UI for local hotseat or human-vs-bot games
- A Monte Carlo Tree Search bot with optional value and policy neural
  networks guiding leaf evaluation and child selection
- A self-play training data pipeline
- Training scripts for the value and policy networks in PyTorch
- A one-time converter that reads the legacy Java project's `.bin`
  training data so old games can be reused to train the new networks

No accounts, no cloud, no network calls. Runs as a single desktop app.

## Project layout

Four independent Python packages under `src/`. Each depends only on the
ones above it — the bot and game-solver can be extracted and reused in
other projects without dragging the UI along.

```
uttt/
├── src/
│   ├── uttt_engine/   # pure game rules — numpy + stdlib only
│   ├── uttt_bot/      # MCTS, value/policy nets, encoders, bot wrapper
│   ├── uttt_ml/       # training, self-play, legacy .bin migration
│   └── uttt_app/      # PySide6 desktop UI
├── tests/             # pytest suite (63 tests and counting)
├── data/              # gitignored — converted .npz + self-play output
├── models/            # gitignored — trained checkpoints
├── pyproject.toml
├── config.example.yaml
└── README.md
```

## Install

Create a virtualenv and install the package in editable mode. The PyPI
`torch` wheel defaults to CUDA on Linux (a multi-gigabyte download),
which you don't need on a CPU-only machine, so the install commands
below explicitly opt into the right wheel index.

### CPU (Linux with integrated graphics, or any machine without an NVIDIA GPU)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

### CUDA (Windows with a discrete NVIDIA GPU)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,gpu]" --extra-index-url https://download.pytorch.org/whl/cu121
```

The CUDA install pulls the full `torch` + `nvidia-*` bundle, which is
what you want for training. The same code runs identically on both
boxes — `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
is the pattern used throughout.

### Per-machine config

Copy `config.example.yaml` to `config.yaml` and tweak. The file is
gitignored, so each machine keeps its own settings without branching:

```bash
cp config.example.yaml config.yaml
```

Typical knobs:

- `device: auto | cpu | cuda` — force a particular PyTorch device
- `num_threads` — default MCTS worker pool size
- `mcts.compute_time_seconds` — bot time budget per move
- `paths.value_net` / `paths.policy_net` — optional trained checkpoint
  paths. If null, MCTS falls back to random rollouts + UCB1.

## Playing a game

Launch the desktop app:

```bash
uttt
```

Or directly:

```bash
python -m uttt_app.main
```

The menu lets you pick each player's name, whether they are a human or
the MCTS bot, and — for bot players — the compute time, thread count,
and optional value/policy checkpoint paths. Click *Start Game* to
enter the board view. Humans click cells to move; bots run MCTS on a
background thread so the UI stays responsive.

## Training pipeline

The training data format is a `.npz` archive with two keys:

- `features` — shape `(N, 90)` float32, the board flat vector
- `labels` — shape `(N, 1)` for value data or `(N, 81)` for policy data

Both the legacy converter and the self-play generator produce archives
in this format, so the dataset loaders in `uttt_ml.dataset` consume old
and new data interchangeably.

### 1. Convert the legacy Java `.bin` training data (one-time)

```bash
uttt-convert-bin \
    --src /path/to/UltimateTickTacToe/data/series1 \
    --dst ./data/series1_npz \
    --combine
```

`--combine` concatenates every same-prefix file into a single
`series1_combined_value.npz` / `series1_combined_policy.npz` archive
under `data/`, mirroring `NetworkTrainer.concatenateDataFiles()` on the
Java side. On the bundled sample data this produces ~28k training
positions.

### 2. Train the value network

Mirrors `ValueNetworkTrainer.java`: Adam lr=0.001, seed 2039402, 80/20
split, up to 500 epochs, early stop after 20 epochs of no validation
improvement, checkpoints every 30 seconds.

```bash
uttt-train-value \
    --data ./data/series1_combined_value.npz \
    --out ./models/value \
    --epochs 500 \
    --device auto
```

Final model is written to
`models/value/value_<timestamp>/value_final_mse<mse>.pt`. The
checkpoint format is a plain PyTorch `state_dict`, so a model trained
on a Windows GPU loads cleanly on a Linux CPU via `map_location`.

### 3. Train the policy network

Mirrors `PolicyNetworkTrainer.java`: Adam lr=0.006, batch 1000, early
stop when `|train_mse - val_mse| / val_mse > 2.5%`.

```bash
uttt-train-policy \
    --data ./data/series1_combined_policy.npz \
    --out ./models/policy \
    --device auto
```

The policy dataset renders the 76×76 image on the fly from the flat
vector. On a CPU-only box that's the slowest part of training — pass
`--num-workers 4` or higher to parallelize the rendering.

### 4. Generate fresh self-play data

Mirrors `TrainingDataCreator.java` with the same interactive prompts:

```bash
uttt-self-play
```

Or non-interactively:

```bash
uttt-self-play \
    --data-dir ./data \
    --series seriesNew \
    --start-game 1 --num-games 50 \
    --compute-time 2.0 --num-threads 4 \
    --value-net ./models/value/.../value_final.pt \
    --non-interactive
```

Each game writes paired `value$<series>,gameNum<N>$<ts>.npz` and
`policy$<series>,gameNum<N>$<ts>.npz` files, schema identical to
converted legacy data. Run `uttt-convert-bin --combine` with the
series directory (or just concatenate manually) to produce a combined
archive for retraining.

## Tests

The full suite runs in well under five seconds:

```bash
pytest
```

Covers the engine rules (legal moves, send-to, win detection, invert,
flat-vector round-trip), the encoders (shape + channel semantics), the
nets (forward-pass shapes, save/load round-trip), the legacy `.bin`
parser against real fixtures, the dataset loaders, MCTS (forced-win-in-1
in both single-threaded and 4-threaded modes), rollouts, and the
compute-time budget.

## Base functionality and design notes

- **Game rules** match the Java side exactly — including the unusual
  `-0.25` draw evaluation preserved from the original project, so bot
  behavior stays comparable.
- **MCTS** mirrors `MCTSEvaluator.java`: UCB1 exploration constant
  `C=2.0`, most-visits-wins best-move selection, random rollout
  fallback, and a threaded worker pool with virtual loss.
- **Value network** — Conv3D(4→4)x2 + Dense(324→100)x2 + Dense(100→1),
  identity output activation, MSE loss. Same layer sizes as DL4J.
- **Policy network** — Conv2D(3→3)x4 + MaxPool, Dense(→300)x4 +
  Dense(→81), identity output, MSE loss. Input is the 76×76 board
  image; callers mask illegal moves before sampling.
- **Dual machine** — the same code runs on CPU and CUDA; pick the
  right install extras and the `torch.device` switch does the rest.
- **No DL4J model weights are migrated.** DL4J and PyTorch state
  formats are incompatible — the right move is to retrain from the
  converted `.bin` data, which works out of the box with the new
  training scripts.

## Origin

Ported from `/home/calebh/claude-willow/UltimateTickTacToe` (the
original Java project). The Java sources are still present for
reference; the migration plan lives at
`~/.claude/plans/noble-wandering-otter.md`.
