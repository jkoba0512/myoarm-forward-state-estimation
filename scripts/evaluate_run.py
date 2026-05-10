"""Evaluate a saved ``ForwardMLP`` against a recorded episode run.

Usage::

    uv run python scripts/evaluate_run.py \\
        --model runs/models/{model_id} \\
        --run runs/episodes/{run_id} \\
        [--horizons 1,10,50]

Outputs ``runs/models/{model_id}/eval_{run_id}.json`` containing both
the reaching summary (Step 7) and the prediction metrics per horizon
(Step 8). Multiple eval runs of the same model produce sibling
``eval_*.json`` files in the same directory.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myoarm_fse.data import EpisodeLog
from myoarm_fse.data.logger import RunIndex
from myoarm_fse.metrics import aggregate_reaching
from myoarm_fse.models import (
    build_transitions,
    load_model,
    rollout_predictions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved forward model.")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Comma-separated rollout horizons. Defaults to model config.eval.rollout_horizons.",
    )
    return p.parse_args(argv)


def _resolve_horizons(arg: str | None, config: dict[str, Any]) -> tuple[int, ...]:
    if arg is not None:
        return tuple(int(s) for s in arg.split(",") if s.strip())
    raw = config.get("eval", {}).get("rollout_horizons")
    if raw is None:
        return (1, 10, 50)
    return tuple(int(h) for h in raw)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)

    model, model_config, _model_metrics = load_model(args.model)
    print(
        f"Loaded model: {args.model} "
        f"(state_dim={model.state_dim}, action_dim={model.action_dim}, "
        f"hidden_dims={model.hidden_dims})"
    )

    index = RunIndex.load(args.run / "index.json")
    logs = [EpisodeLog.load(args.run / e.file) for e in index.episodes]
    print(f"Loaded run {args.run.name}: {len(logs)} episodes")

    reaching_summary = aggregate_reaching(logs)
    for k in (
        "n",
        "minimum_tip_error_mean",
        "final_tip_error_mean",
        "success_rate",
        "effort_mean",
    ):
        v = reaching_summary.get(k)
        if isinstance(v, float):
            print(f"  reaching.{k:30s} {v:.4f}")
        else:
            print(f"  reaching.{k:30s} {v}")

    horizons = _resolve_horizons(args.horizons, model_config)
    dataset = build_transitions(logs)
    print(
        f"Built transitions: N={dataset.n}, n_episodes={dataset.n_episodes}, "
        f"state_dim={dataset.state_dim}"
    )

    if dataset.state_dim != model.state_dim or dataset.action_dim != model.action_dim:
        raise SystemExit(
            f"dim mismatch: dataset has ({dataset.state_dim}, {dataset.action_dim}), "
            f"model has ({model.state_dim}, {model.action_dim})"
        )

    rollout = rollout_predictions(model, dataset, horizons=horizons)
    rollout_serializable = {str(h): r for h, r in rollout.items()}
    for h, r in rollout.items():
        print(
            f"  rollout h={h:3d}: mse={r['rollout_mse']:.6f}, "
            f"tip_err={r['tip_prediction_error']:.6f}"
        )

    payload = {
        "model_path": str(args.model),
        "run_path": str(args.run),
        "run_id": index.run_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "reaching": reaching_summary,
        "prediction": rollout_serializable,
        "horizons": list(horizons),
    }
    out_path = args.model / f"eval_{index.run_id}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    main()
