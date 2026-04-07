"""
convert_bin_to_npz — one-time migration of legacy ND4J .bin training data.

The original Java project wrote DL4J-flavored binary files; PyTorch has
no native reader for them. Rather than pay the format-parsing cost on
every training run, this script parses every `.bin` in a source
directory once and writes a matching `.npz` file with two keys:

    features    (N, 90)   float32   — board flat vectors
    labels      (N, 1)    float32   — value targets
                  or (N, 81) for policy files

Usage:

    python -m uttt_ml.convert_bin_to_npz \
        --src /path/to/UltimateTickTacToe/data/series1 \
        --dst ./data/series1_npz \
        --combine

Passing `--combine` also writes concatenated files
`series1_combined_value.npz` and `series1_combined_policy.npz` at
`--dst/..`, mirroring `NetworkTrainer.concatenateDataFiles()` in the
Java side (NetworkTrainer.java lines 256-272).

Every `.bin` in the source directory is categorized by its filename
prefix: files starting with `value$` go to the value-net pool, files
starting with `policy$` go to the policy-net pool. Anything else is
skipped with a warning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from uttt_ml.nd4j_loader import load_dataset


def _iter_bin_files(src: Path) -> list[Path]:
    """Return every `.bin` file in `src`, sorted for deterministic output."""
    return sorted(p for p in src.iterdir() if p.is_file() and p.suffix == ".bin")


def _convert_one(bin_path: Path, dst_dir: Path) -> Path:
    """Parse `bin_path` and write a matching `.npz` into `dst_dir`."""
    ds = load_dataset(bin_path)
    out_path = dst_dir / (bin_path.stem + ".npz")
    np.savez_compressed(
        out_path,
        features=ds.features.astype(np.float32),
        labels=ds.labels.astype(np.float32),
    )
    return out_path


def _concatenate(npz_paths: list[Path], out_path: Path) -> None:
    """Vstack features and labels from every npz file into a combined archive.

    Mirrors `NetworkTrainer.concatenateDataSets()` in the Java side,
    which uses `Nd4j.vstack` to merge every file's features and labels.
    """
    if not npz_paths:
        return
    feats = []
    labels = []
    for path in npz_paths:
        with np.load(path) as data:
            feats.append(data["features"])
            labels.append(data["labels"])
    combined_features = np.concatenate(feats, axis=0).astype(np.float32)
    combined_labels = np.concatenate(labels, axis=0).astype(np.float32)
    np.savez_compressed(
        out_path,
        features=combined_features,
        labels=combined_labels,
    )
    print(
        f"  combined -> {out_path.name}: "
        f"features {combined_features.shape}, labels {combined_labels.shape}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point registered as `uttt-convert-bin`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source directory containing .bin files (e.g. data/series1/)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Destination directory for per-file .npz output",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Also write concatenated value/policy .npz archives under --dst/..",
    )
    args = parser.parse_args(argv)

    src: Path = args.src
    dst: Path = args.dst
    if not src.is_dir():
        print(f"error: source directory not found: {src}", file=sys.stderr)
        return 2
    dst.mkdir(parents=True, exist_ok=True)

    bin_files = _iter_bin_files(src)
    if not bin_files:
        print(f"warning: no .bin files under {src}", file=sys.stderr)
        return 1

    value_outputs: list[Path] = []
    policy_outputs: list[Path] = []
    skipped: list[Path] = []

    print(f"converting {len(bin_files)} file(s) from {src} -> {dst}")
    for bin_path in bin_files:
        try:
            out_path = _convert_one(bin_path, dst)
        except Exception as exc:  # pragma: no cover — robust CLI
            print(f"  FAILED: {bin_path.name}: {exc}", file=sys.stderr)
            continue
        name = bin_path.name
        if name.startswith("value$"):
            value_outputs.append(out_path)
        elif name.startswith("policy$"):
            policy_outputs.append(out_path)
        else:
            skipped.append(out_path)
        print(f"  {bin_path.name} -> {out_path.name}")

    print(
        f"converted {len(value_outputs)} value, {len(policy_outputs)} policy, "
        f"{len(skipped)} unclassified file(s)"
    )

    if args.combine:
        combined_dir = dst.parent
        combined_dir.mkdir(parents=True, exist_ok=True)
        series = src.name
        if value_outputs:
            _concatenate(
                sorted(value_outputs),
                combined_dir / f"{series}_combined_value.npz",
            )
        if policy_outputs:
            _concatenate(
                sorted(policy_outputs),
                combined_dir / f"{series}_combined_policy.npz",
            )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
