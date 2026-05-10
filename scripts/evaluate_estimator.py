"""Grid-sweep a fixed-gain Kalman-like estimator over baseline runs.

Usage::

    uv run python scripts/evaluate_estimator.py --config configs/estimators/fixed_kalman_default.yaml

For each ``(gain, delay)`` combination in the config grid, the
estimator is built fresh and run over every episode of every listed
run. Per-run / per-setting summaries land in
``runs/estimators/{eval_id}/per_setting/gain_*_delay_*/<run_label>.json``;
a flat ``summary.json`` collects the same data with grid coordinates
for easy plotting.

Limited CLI overrides for ad-hoc experiments::

    --master-seed     override config.seed
    --output-root     override config.output_root
    --forward-model   override config.forward_model
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from myoarm_fse.data import EpisodeLog
from myoarm_fse.data.logger import RunIndex
from myoarm_fse.envs.state import StateSpec
from myoarm_fse.estimators import (
    FixedGainKalmanEstimator,
    aggregate_estimation_metrics,
    evaluate_estimator_on_log,
)
from myoarm_fse.models import load_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grid-sweep fixed-gain Kalman-like estimator across runs."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--master-seed", type=int, default=None)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--forward-model", type=str, default=None)
    return p.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.master_seed is not None:
        cfg["seed"] = int(args.master_seed)
    if args.output_root is not None:
        cfg["output_root"] = str(args.output_root)
    if args.forward_model is not None:
        cfg["forward_model"] = str(args.forward_model)
    return cfg


def _make_eval_id(now: datetime | None = None) -> str:
    t = now if now is not None else datetime.now(timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _hash_config(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _state_spec_from_model_config(model_config: dict[str, Any]) -> StateSpec:
    """Recover StateSpec from a saved model's architecture entry.

    Phase 0 hard-codes myoArm reach dims (qpos=20, qvel=20, act=34) when
    state_dim == 83. Other dims are unsupported here and would need a
    real layout stored alongside the model.
    """
    state_dim = int(model_config["architecture"]["state_dim"])
    if state_dim == 83:
        return StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    raise NotImplementedError(
        f"state_dim={state_dim} layout is not auto-derivable; "
        "extend _state_spec_from_model_config when adding a new schema."
    )


def _short_run_label(run_path: Path) -> str:
    """Return a filesystem-friendly label for a run directory."""
    return run_path.name


def _load_run_logs(run_path: Path) -> tuple[RunIndex, list[EpisodeLog]]:
    index = RunIndex.load(run_path / "index.json")
    logs = [EpisodeLog.load(run_path / e.file) for e in index.episodes]
    return index, logs


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    cfg = _apply_overrides(_load_config(args.config), args)

    print(f"Loaded config: {args.config}")
    for key in ("seed", "forward_model", "obs_compose", "skip_cold_start"):
        print(f"  {key}: {cfg.get(key)}")
    print(f"  gain_grid: {cfg['gain_grid']}")
    print(f"  delay_grid: {cfg['delay_grid']}")
    print(f"  runs: {cfg['runs']}")

    model, model_config, _model_metrics = load_model(cfg["forward_model"])
    state_spec = _state_spec_from_model_config(model_config)
    print(
        f"Loaded model: {cfg['forward_model']} "
        f"(state_dim={model.state_dim}, action_dim={model.action_dim})"
    )

    # Pre-load all runs so we don't re-read npz on every grid setting.
    run_logs: list[tuple[Path, RunIndex, list[EpisodeLog]]] = []
    for run_str in cfg["runs"]:
        run_path = Path(run_str)
        index, logs = _load_run_logs(run_path)
        print(f"Loaded run {index.run_id}: {len(logs)} episodes")
        run_logs.append((run_path, index, logs))

    eval_id = _make_eval_id()
    config_hash = _hash_config(cfg)
    out_dir = Path(cfg.get("output_root", "runs/estimators")) / eval_id
    per_setting_dir = out_dir / "per_setting"
    per_setting_dir.mkdir(parents=True, exist_ok=True)

    # Persist the resolved config for traceability.
    (out_dir / "config.json").write_text(
        json.dumps({"eval_id": eval_id, "config_hash": config_hash, "config": cfg}, indent=2)
    )

    summary: list[dict[str, Any]] = []
    obs_noise_sigma = dict(cfg.get("obs_noise_sigma", {}))
    obs_compose = str(cfg.get("obs_compose", "noisy_then_delayed"))
    skip_cold_start = bool(cfg.get("skip_cold_start", True))

    # Derive a per-setting child seed via SeedSequence so re-runs are
    # reproducible and disjoint across settings.
    rng_seq = np.random.SeedSequence(int(cfg.get("seed", 0)))
    n_grid = len(cfg["gain_grid"]) * len(cfg["delay_grid"]) * len(run_logs)
    child_seeds = rng_seq.spawn(n_grid)
    seed_iter = iter(child_seeds)

    print(f"\nGrid sweep: {len(cfg['gain_grid'])} gains x "
          f"{len(cfg['delay_grid'])} delays x {len(run_logs)} runs = "
          f"{n_grid} settings")

    for gain in cfg["gain_grid"]:
        for delay in cfg["delay_grid"]:
            estimator = FixedGainKalmanEstimator(
                model, gain=float(gain), state_spec=state_spec,
                delay_steps=int(delay),
            )
            setting_dir = per_setting_dir / f"gain_{gain}_delay_{delay}"
            setting_dir.mkdir(parents=True, exist_ok=True)
            for (run_path, index, logs) in run_logs:
                child = next(seed_iter)
                # Use first 32 bits of the spawned seed deterministically.
                child_seed = int(child.generate_state(1)[0])
                results = []
                for log in logs:
                    res = evaluate_estimator_on_log(
                        estimator, log,
                        state_spec=state_spec,
                        obs_noise_sigma=obs_noise_sigma,
                        obs_delay_steps=int(delay),
                        obs_noise_seed=child_seed,
                        obs_compose=obs_compose,
                    )
                    results.append(res)
                metrics = aggregate_estimation_metrics(
                    results, skip_cold_start=skip_cold_start,
                )
                run_label = _short_run_label(run_path)
                (setting_dir / f"{run_label}.json").write_text(
                    json.dumps({
                        "gain": gain,
                        "delay": delay,
                        "run_label": run_label,
                        "run_id": index.run_id,
                        "obs_noise_sigma": obs_noise_sigma,
                        "obs_compose": obs_compose,
                        "obs_noise_seed": child_seed,
                        "skip_cold_start": skip_cold_start,
                        **metrics,
                    }, indent=2)
                )
                row = {
                    "gain": gain,
                    "delay": delay,
                    "run_label": run_label,
                    "run_id": index.run_id,
                    **metrics,
                }
                summary.append(row)
                print(
                    f"  gain={gain:>4}  delay={delay:>2}  run={run_label:30s}  "
                    f"tip_err_mean={metrics.get('tip_estimation_error_mean', float('nan')):.4f}  "
                    f"mse_qpos_mean={metrics.get('mse_qpos_mean', float('nan')):.4f}"
                )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "eval_id": eval_id,
        "config_hash": config_hash,
        "gain_grid": cfg["gain_grid"],
        "delay_grid": cfg["delay_grid"],
        "runs": [str(r) for r in cfg["runs"]],
        "obs_noise_sigma": obs_noise_sigma,
        "obs_compose": obs_compose,
        "skip_cold_start": skip_cold_start,
        "results": summary,
    }, indent=2))
    print(f"\nSaved summary: {summary_path}")
    return out_dir


if __name__ == "__main__":
    main()
