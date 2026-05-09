"""Generate myoArm target set splits from a YAML config.

Usage::

    uv run python scripts/generate_targets.py --config configs/targets/default.yaml

Each split is written to ``{output_dir}/{split}.npz``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from myoarm_fse.envs.targets import (
    TargetGenerationConfig,
    generate_all_target_sets,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config (e.g. configs/targets/default.yaml).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = TargetGenerationConfig.from_yaml(args.config)
    print(f"Loaded config: {args.config}")
    print(f"  env_id: {config.env_id}")
    print(f"  generator_seed: {config.generator_seed}")
    print(f"  output_dir: {config.output_dir}")
    print(f"  splits: {[(s.name, s.n) for s in config.splits]}")

    target_sets = generate_all_target_sets(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, ts in target_sets.items():
        path = output_dir / f"{name}.npz"
        ts.save(path)
        d = ts.tip_to_target_init_distance
        print(
            f"  saved {path}  (n={ts.n}, "
            f"init_distance min={d.min():.3f} mean={d.mean():.3f} max={d.max():.3f})"
        )


if __name__ == "__main__":
    main()
