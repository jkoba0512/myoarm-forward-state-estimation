"""Assemble a TransitionDataset (.npz) from one or more episode run dirs.

Each input directory is one ``runs/episodes/<UTC>/`` produced by
``scripts/collect_episodes.py``: it contains an ``index.json`` plus one
``NNNN.npz`` per episode (an ``EpisodeLog``).

For each input dir we load its episodes, build a ``TransitionDataset``
(via ``build_transitions``), then concatenate everything with
``concat_datasets`` so that ``source_dataset_index`` / ``source_episode_id``
provenance is preserved across batches.

This is the scripted replacement for the previously ad-hoc dataset
assembly used to produce ``runs/datasets/expanded.npz``; R3 reachable
re-collections (Stage A.3) go through this path.

Usage::

    uv run python scripts/build_dataset.py \\
        --episodes runs/episodes/<batch1> runs/episodes/<batch2> ... \\
        --output runs/datasets/expanded_reachable.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from myoarm_fse.data.schema import EpisodeLog
from myoarm_fse.models.datasets import (
    TransitionDataset,
    build_transitions,
    concat_datasets,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--episodes",
        nargs="+",
        type=Path,
        required=True,
        help="One or more episode run directories (each contains index.json + NNNN.npz files).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz path for the merged TransitionDataset.",
    )
    return p.parse_args(argv)


def _load_batch(run_dir: Path) -> TransitionDataset:
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")
    index_path = run_dir / "index.json"
    if not index_path.exists():
        raise SystemExit(f"missing index.json in {run_dir}")
    with open(index_path) as f:
        index = json.load(f)
    episode_files = sorted(p for p in run_dir.glob("*.npz") if p.name != "index.json")
    if not episode_files:
        raise SystemExit(f"no episode .npz files in {run_dir}")
    logs = [EpisodeLog.load(p) for p in episode_files]
    ds = build_transitions(logs)
    print(f"  {run_dir.name}: {len(episode_files)} episodes "
          f"-> {ds.n} transitions  (controller={index['config']['controller']['name']})")
    return ds


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"output: {args.output}")
    print(f"input batches ({len(args.episodes)}):")
    datasets = [_load_batch(d) for d in args.episodes]
    merged = concat_datasets(datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.save(args.output)
    print(f"\nsaved: {args.output}")
    print(f"  total transitions: {merged.n}")
    print(f"  total episodes:    {merged.n_episodes}")
    print(f"  state_dim={merged.state_dim} action_dim={merged.action_dim}")


if __name__ == "__main__":
    main()
