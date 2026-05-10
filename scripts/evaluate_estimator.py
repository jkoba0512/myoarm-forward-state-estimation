"""Grid-sweep a fixed-gain Kalman-like estimator over baseline runs.

Usage::

    uv run python scripts/evaluate_estimator.py --config configs/estimators/fixed_kalman_default.yaml

For each ``(noise_condition, gain, delay)`` combination in the config
grid, the estimator is built fresh and run over every episode of every
listed run. Per-run / per-setting summaries land in
``runs/estimators/{eval_id}/per_setting/[noise_<name>/]gain_*_delay_*/<run_label>.json``;
a flat ``summary.json`` collects the same data with grid coordinates
for plotting, and (when ``noise_conditions`` is provided)
``metrics.csv`` plus ``best_by_condition.csv`` give tabular views over
``controller × noise_condition × delay × gain``.

Two config schemas are supported:

- **Single-noise mode** (Phase 3.1 default): provide ``obs_noise_sigma``
  as a single dict. The sweep is ``gain × delay × runs`` and only
  ``summary.json`` / per-setting jsons are emitted (no CSVs).
- **Robustness mode** (Phase 3.3-min): provide ``noise_conditions:
  {name: sigma_dict}``. The sweep adds an outer loop over named
  conditions and emits the CSV table outputs alongside the summary.

Limited CLI overrides for ad-hoc experiments::

    --master-seed     override config.seed
    --output-root     override config.output_root
    --forward-model   override config.forward_model
"""

from __future__ import annotations

import argparse
import csv
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
    """Recover StateSpec from a saved model's architecture entry."""
    state_dim = int(model_config["architecture"]["state_dim"])
    if state_dim == 83:
        return StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    raise NotImplementedError(
        f"state_dim={state_dim} layout is not auto-derivable; "
        "extend _state_spec_from_model_config when adding a new schema."
    )


def _short_run_label(run_path: Path) -> str:
    return run_path.name


def _load_run_logs(run_path: Path) -> tuple[RunIndex, list[EpisodeLog]]:
    index = RunIndex.load(run_path / "index.json")
    logs = [EpisodeLog.load(run_path / e.file) for e in index.episodes]
    return index, logs


def _resolve_noise_conditions(
    cfg: dict[str, Any],
) -> list[tuple[str, dict[str, float]]]:
    """Return ``[(name, sigma_dict), ...]`` from either schema.

    Robustness schema (``noise_conditions: {name: dict}``) takes
    precedence; if absent, falls back to a single ``"default"`` entry
    using ``obs_noise_sigma`` (or empty dict).
    """
    if "noise_conditions" in cfg and cfg["noise_conditions"] is not None:
        nc = cfg["noise_conditions"]
        if not isinstance(nc, dict):
            raise ValueError(
                f"noise_conditions must be a mapping, got {type(nc).__name__}"
            )
        if not nc:
            raise ValueError("noise_conditions is empty")
        out: list[tuple[str, dict[str, float]]] = []
        for name, sigma in nc.items():
            if not isinstance(sigma, dict):
                raise ValueError(
                    f"noise_conditions[{name!r}] must be a mapping, "
                    f"got {type(sigma).__name__}"
                )
            out.append((str(name), {str(k): float(v) for k, v in sigma.items()}))
        return out
    return [("default", dict(cfg.get("obs_noise_sigma", {})))]


# CSV column order for metrics.csv. Keep stable so downstream tools
# (notebooks, plot scripts) can rely on it.
_METRICS_CSV_COLUMNS: tuple[str, ...] = (
    "controller",
    "noise_condition",
    "delay_steps",
    "gain",
    "tip_estimation_error_mean",
    "tip_estimation_error_final",
    "tip_estimation_error_std",
    "state_mse_mean",
    "mse_qpos_mean",
    "mse_qvel_mean",
    "mse_act_mean",
    "mse_tip_pos_mean",
    "mse_target_pos_mean",
    "mse_reach_err_mean",
    "n_episodes",
    "model_run_id",
)


def _row_for_csv(
    *,
    controller: str,
    noise_condition: str,
    delay_steps: int,
    gain: float,
    metrics: dict[str, Any],
    model_run_id: str,
) -> dict[str, Any]:
    return {
        "controller": controller,
        "noise_condition": noise_condition,
        "delay_steps": delay_steps,
        "gain": gain,
        "tip_estimation_error_mean": metrics.get(
            "tip_estimation_error_mean", float("nan")
        ),
        "tip_estimation_error_final": metrics.get(
            "tip_estimation_error_final", float("nan")
        ),
        "tip_estimation_error_std": metrics.get(
            "tip_estimation_error_std", float("nan")
        ),
        "state_mse_mean": metrics.get("state_mse_mean", float("nan")),
        "mse_qpos_mean": metrics.get("mse_qpos_mean", float("nan")),
        "mse_qvel_mean": metrics.get("mse_qvel_mean", float("nan")),
        "mse_act_mean": metrics.get("mse_act_mean", float("nan")),
        "mse_tip_pos_mean": metrics.get("mse_tip_pos_mean", float("nan")),
        "mse_target_pos_mean": metrics.get("mse_target_pos_mean", float("nan")),
        "mse_reach_err_mean": metrics.get("mse_reach_err_mean", float("nan")),
        "n_episodes": metrics.get("n", 0),
        "model_run_id": model_run_id,
    }


def _write_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_METRICS_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _METRICS_CSV_COLUMNS})


def _write_best_by_condition_csv(
    rows: list[dict[str, Any]], path: Path
) -> None:
    """Group by (controller, noise_condition, delay_steps), keep best gain.

    Best = smallest ``tip_estimation_error_mean``.
    """
    best: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["controller"]),
            str(row["noise_condition"]),
            int(row["delay_steps"]),
        )
        prev = best.get(key)
        cur_err = row.get("tip_estimation_error_mean", float("inf"))
        if prev is None or cur_err < prev.get(
            "tip_estimation_error_mean", float("inf")
        ):
            best[key] = row
    sorted_keys = sorted(best.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_METRICS_CSV_COLUMNS))
        writer.writeheader()
        for key in sorted_keys:
            row = best[key]
            writer.writerow({k: row.get(k, "") for k in _METRICS_CSV_COLUMNS})


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    cfg = _apply_overrides(_load_config(args.config), args)

    print(f"Loaded config: {args.config}")
    for key in ("seed", "forward_model", "obs_compose", "skip_cold_start"):
        print(f"  {key}: {cfg.get(key)}")
    print(f"  gain_grid: {cfg['gain_grid']}")
    print(f"  delay_grid: {cfg['delay_grid']}")
    print(f"  runs: {cfg['runs']}")

    noise_conditions = _resolve_noise_conditions(cfg)
    multi_noise = "noise_conditions" in cfg and cfg["noise_conditions"] is not None
    print(
        f"  noise_conditions: "
        f"{[name for name, _ in noise_conditions]}"
        + ("" if multi_noise else " (single-noise legacy mode)")
    )

    model, model_config, _model_metrics = load_model(cfg["forward_model"])
    state_spec = _state_spec_from_model_config(model_config)
    model_run_id = Path(cfg["forward_model"]).name
    print(
        f"Loaded model: {cfg['forward_model']} "
        f"(state_dim={model.state_dim}, action_dim={model.action_dim})"
    )

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

    (out_dir / "config.json").write_text(
        json.dumps(
            {"eval_id": eval_id, "config_hash": config_hash, "config": cfg},
            indent=2,
        )
    )

    summary: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    obs_compose = str(cfg.get("obs_compose", "noisy_then_delayed"))
    skip_cold_start = bool(cfg.get("skip_cold_start", True))

    rng_seq = np.random.SeedSequence(int(cfg.get("seed", 0)))
    n_grid = (
        len(noise_conditions)
        * len(cfg["gain_grid"])
        * len(cfg["delay_grid"])
        * len(run_logs)
    )
    child_seeds = rng_seq.spawn(n_grid)
    seed_iter = iter(child_seeds)

    print(
        f"\nGrid sweep: {len(noise_conditions)} noise x "
        f"{len(cfg['gain_grid'])} gains x "
        f"{len(cfg['delay_grid'])} delays x {len(run_logs)} runs = "
        f"{n_grid} settings"
    )

    for noise_name, noise_sigma in noise_conditions:
        for gain in cfg["gain_grid"]:
            for delay in cfg["delay_grid"]:
                estimator = FixedGainKalmanEstimator(
                    model, gain=float(gain), state_spec=state_spec,
                    delay_steps=int(delay),
                )
                if multi_noise:
                    setting_dir = (
                        per_setting_dir
                        / f"noise_{noise_name}"
                        / f"gain_{gain}_delay_{delay}"
                    )
                else:
                    setting_dir = per_setting_dir / f"gain_{gain}_delay_{delay}"
                setting_dir.mkdir(parents=True, exist_ok=True)
                for run_path, index, logs in run_logs:
                    child = next(seed_iter)
                    child_seed = int(child.generate_state(1)[0])
                    results = []
                    for log in logs:
                        res = evaluate_estimator_on_log(
                            estimator, log,
                            state_spec=state_spec,
                            obs_noise_sigma=noise_sigma,
                            obs_delay_steps=int(delay),
                            obs_noise_seed=child_seed,
                            obs_compose=obs_compose,
                        )
                        results.append(res)
                    metrics = aggregate_estimation_metrics(
                        results, skip_cold_start=skip_cold_start,
                    )
                    run_label = _short_run_label(run_path)
                    setting_payload: dict[str, Any] = {
                        "noise_condition": noise_name,
                        "gain": gain,
                        "delay": delay,
                        "run_label": run_label,
                        "run_id": index.run_id,
                        "obs_noise_sigma": noise_sigma,
                        "obs_compose": obs_compose,
                        "obs_noise_seed": child_seed,
                        "skip_cold_start": skip_cold_start,
                        **metrics,
                    }
                    (setting_dir / f"{run_label}.json").write_text(
                        json.dumps(setting_payload, indent=2)
                    )
                    summary.append({
                        "noise_condition": noise_name,
                        "gain": gain,
                        "delay": delay,
                        "run_label": run_label,
                        "run_id": index.run_id,
                        **metrics,
                    })
                    csv_rows.append(_row_for_csv(
                        controller=run_label,
                        noise_condition=noise_name,
                        delay_steps=int(delay),
                        gain=float(gain),
                        metrics=metrics,
                        model_run_id=model_run_id,
                    ))
                    print(
                        f"  noise={noise_name:>6}  gain={gain:>4}  "
                        f"delay={delay:>2}  run={run_label:30s}  "
                        f"tip_err_mean="
                        f"{metrics.get('tip_estimation_error_mean', float('nan')):.4f}  "
                        f"mse_qpos_mean="
                        f"{metrics.get('mse_qpos_mean', float('nan')):.4f}"
                    )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "eval_id": eval_id,
        "config_hash": config_hash,
        "model_run_id": model_run_id,
        "gain_grid": cfg["gain_grid"],
        "delay_grid": cfg["delay_grid"],
        "noise_conditions": [
            {"name": name, "sigma": sigma} for name, sigma in noise_conditions
        ],
        "runs": [str(r) for r in cfg["runs"]],
        "obs_compose": obs_compose,
        "skip_cold_start": skip_cold_start,
        "results": summary,
    }, indent=2))
    print(f"\nSaved summary: {summary_path}")

    if multi_noise:
        metrics_csv = out_dir / "metrics.csv"
        best_csv = out_dir / "best_by_condition.csv"
        _write_metrics_csv(csv_rows, metrics_csv)
        _write_best_by_condition_csv(csv_rows, best_csv)
        print(f"Saved metrics csv: {metrics_csv}")
        print(f"Saved best_by_condition csv: {best_csv}")
    return out_dir


if __name__ == "__main__":
    main()
