"""Evaluate the Phase 3.2 Stage A learned gain against 7 strategies.

Usage::

    uv run python scripts/evaluate_learned_gain.py \\
        --config configs/estimators/learned_gain_eval_default.yaml \\
        --learned-gain-model runs/learned_gain_models/{model_id}

Per condition (controller × noise × delay), the script picks the K
each strategy would assign and runs ``evaluate_estimator_on_log`` over
every episode of every listed run. Outputs:

```
runs/learned_gain_evals/{eval_id}/
├── config.json
├── summary.json
├── comparison.csv          # 7 strategies × 36 conditions = 252 rows
├── delta_vs_oracle.csv     # same rows + tip_err - oracle_tip_err
└── per_setting/strategy_*/noise_*/gain_*_delay_*/<run>.json
```

Limited CLI overrides::

    --master-seed         override config.seed
    --output-root         override config.output_root
    --forward-model       override config.forward_model
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
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
    GainPredictor,
    LearnedGainKalmanEstimator,
    StateAwareGainPredictor,
    StateAwareLearnedGainKalmanEstimator,
    aggregate_estimation_metrics,
    evaluate_estimator_on_log,
    load_learned_gain_model,
    load_oracle_table,
)
from myoarm_fse.models import load_model


_VALID_STRATEGIES = (
    "K=0.0",
    "K=1.0",
    "global_best",
    "best_per_delay",
    "best_per_noise",
    "learned",
    "oracle",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Stage A learned gain against fixed-K and oracle baselines."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--learned-gain-model", type=Path, required=True)
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


# --- strategy K resolution ---


def _build_strategy_k_lookup(
    oracle_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int], float]:
    """oracle row index keyed by (controller, noise_condition, delay_steps)."""
    out: dict[tuple[str, str, int], float] = {}
    for row in oracle_rows:
        key = (
            str(row["controller"]),
            str(row["noise_condition"]),
            int(row["delay_steps"]),
        )
        out[key] = float(row["gain"])
    return out


def _build_global_best_k(oracle_rows: list[dict[str, Any]]) -> float:
    """Pick the K (from oracle K column) that minimises mean tip error.

    Aggregates oracle rows by gain value and picks the gain whose row
    set has the smallest mean tip_estimation_error_mean. Falls back to
    the gain that appears most often among the oracle rows if the
    error column is missing.
    """
    by_gain: dict[float, list[float]] = defaultdict(list)
    for row in oracle_rows:
        g = float(row["gain"])
        err = row.get("tip_estimation_error_mean")
        if err is None:
            by_gain[g].append(0.0)
        else:
            by_gain[g].append(float(err))
    best_gain = min(by_gain, key=lambda g: float(np.mean(by_gain[g])))
    return float(best_gain)


def _build_per_axis_best_k(
    oracle_rows: list[dict[str, Any]],
    *,
    axis: str,
) -> dict[Any, float]:
    """Pick the K minimising mean tip_err within each value of ``axis``."""
    by_value: dict[Any, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in oracle_rows:
        value = row[axis] if axis != "delay_steps" else int(row["delay_steps"])
        g = float(row["gain"])
        err = float(row.get("tip_estimation_error_mean", 0.0))
        by_value[value][g].append(err)
    out: dict[Any, float] = {}
    for value, gains_to_errors in by_value.items():
        best_gain = min(
            gains_to_errors,
            key=lambda g: float(np.mean(gains_to_errors[g])),
        )
        out[value] = float(best_gain)
    return out


def _resolve_strategy_k(
    *,
    strategy: str,
    controller: str,
    noise_condition: str,
    delay_steps: int,
    oracle_lookup: dict[tuple[str, str, int], float],
    global_best_k: float,
    best_per_delay: dict[int, float],
    best_per_noise: dict[str, float],
) -> float | None:
    """Return a scalar K for non-learned strategies, None for 'learned'.

    Caller handles ``learned`` separately because it requires the
    GainPredictor + LearnedGainKalmanEstimator path.
    """
    if strategy == "K=0.0":
        return 0.0
    if strategy == "K=1.0":
        return 1.0
    if strategy == "global_best":
        return global_best_k
    if strategy == "best_per_delay":
        return best_per_delay[int(delay_steps)]
    if strategy == "best_per_noise":
        return best_per_noise[str(noise_condition)]
    if strategy == "oracle":
        key = (controller, noise_condition, int(delay_steps))
        if key not in oracle_lookup:
            raise KeyError(
                f"oracle table missing condition {key!r}; cannot evaluate oracle strategy"
            )
        return oracle_lookup[key]
    if strategy == "learned":
        return None
    raise ValueError(
        f"unknown strategy {strategy!r}; valid: {_VALID_STRATEGIES}"
    )


# --- CSV writers ---


_COMPARISON_COLUMNS: tuple[str, ...] = (
    "strategy",
    "controller",
    "noise_condition",
    "delay_steps",
    "gain_used",
    "learned_k_std",
    "learned_k_min",
    "learned_k_max",
    "tip_estimation_error_mean",
    "tip_estimation_error_final",
    "state_mse_mean",
    "n_episodes",
    "model_run_id",
    "learned_gain_model_id",
)


def _write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_COMPARISON_COLUMNS))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in _COMPARISON_COLUMNS})


def _write_delta_vs_oracle_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Same shape as comparison.csv plus a delta_vs_oracle column."""
    # Build a (controller, noise, delay) → tip_err lookup for the oracle strategy.
    oracle_lookup: dict[tuple[str, str, int], float] = {}
    for row in rows:
        if row["strategy"] == "oracle":
            oracle_lookup[(
                str(row["controller"]),
                str(row["noise_condition"]),
                int(row["delay_steps"]),
            )] = float(row["tip_estimation_error_mean"])
    fieldnames = list(_COMPARISON_COLUMNS) + ["delta_vs_oracle"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            key = (
                str(row["controller"]),
                str(row["noise_condition"]),
                int(row["delay_steps"]),
            )
            oracle_err = oracle_lookup.get(key)
            if oracle_err is None:
                delta = ""
            else:
                delta = float(row["tip_estimation_error_mean"]) - oracle_err
            payload = {k: row.get(k, "") for k in _COMPARISON_COLUMNS}
            payload["delta_vs_oracle"] = delta
            w.writerow(payload)


# --- main ---


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    cfg = _apply_overrides(_load_config(args.config), args)

    strategies = list(cfg.get("strategies", _VALID_STRATEGIES))
    for s in strategies:
        if s not in _VALID_STRATEGIES:
            raise SystemExit(f"unknown strategy {s!r}; valid: {_VALID_STRATEGIES}")

    print(f"Loaded config: {args.config}")
    print(f"  forward_model: {cfg.get('forward_model')}")
    print(f"  oracle_table: {cfg.get('oracle_table')}")
    print(f"  strategies: {strategies}")
    print(f"  delay_grid: {cfg['delay_grid']}")
    print(f"  noise_conditions: {sorted(cfg['noise_conditions'].keys())}")

    # Load forward model.
    forward_model, model_config, _model_metrics = load_model(cfg["forward_model"])
    state_spec = _state_spec_from_model_config(model_config)
    forward_model_run_id = Path(cfg["forward_model"]).name

    # Load learned gain model.
    learned_predictor, learned_config, _learned_metrics = load_learned_gain_model(
        args.learned_gain_model
    )
    learned_model_id = Path(args.learned_gain_model).name
    enc = learned_config.get("input_encoding", {})
    controller_names = tuple(
        enc.get("controller_names", ("random", "lowamp", "hold"))
    )
    sigma_field_order = tuple(
        enc.get("sigma_field_order", ("qpos", "qvel", "tip_pos", "reach_err"))
    )
    delay_max = int(enc.get("delay_max", 6))

    # Load oracle table for oracle / global_best / per-axis-best strategies.
    oracle_rows = load_oracle_table(cfg["oracle_table"])

    # Canonicalize controller column so it matches the per-run controller
    # label used below. best_by_condition.csv stores run_id as
    # `controller`; we want canonical names everywhere downstream.
    run_to_controller = cfg.get("run_to_controller") or {}
    if run_to_controller:
        canonical_set = set(run_to_controller.values())
        for row in oracle_rows:
            ctrl = row["controller"]
            if ctrl in run_to_controller:
                row["controller"] = run_to_controller[ctrl]
            elif ctrl not in canonical_set:
                raise SystemExit(
                    f"oracle row controller {ctrl!r} not in run_to_controller "
                    f"and not a canonical name. Update config.run_to_controller."
                )

    oracle_lookup = _build_strategy_k_lookup(oracle_rows)
    global_best_k = _build_global_best_k(oracle_rows)
    best_per_delay = _build_per_axis_best_k(oracle_rows, axis="delay_steps")
    best_per_noise = _build_per_axis_best_k(oracle_rows, axis="noise_condition")
    print(f"  global_best K = {global_best_k}")
    print(f"  best_per_delay = {best_per_delay}")
    print(f"  best_per_noise = {best_per_noise}")

    # Load runs.
    run_logs: list[tuple[Path, RunIndex, list[EpisodeLog]]] = []
    for run_str in cfg["runs"]:
        run_path = Path(run_str)
        index, logs = _load_run_logs(run_path)
        print(f"Loaded run {index.run_id}: {len(logs)} episodes")
        run_logs.append((run_path, index, logs))

    eval_id = _make_eval_id()
    config_hash = _hash_config(cfg)
    out_dir = Path(cfg.get("output_root", "runs/learned_gain_evals")) / eval_id
    per_setting_dir = out_dir / "per_setting"
    per_setting_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "eval_id": eval_id,
                "config_hash": config_hash,
                "config": cfg,
                "forward_model_run_id": forward_model_run_id,
                "learned_gain_model_id": learned_model_id,
            },
            indent=2,
        )
    )

    obs_compose = str(cfg.get("obs_compose", "noisy_then_delayed"))
    skip_cold_start = bool(cfg.get("skip_cold_start", True))
    delay_grid = [int(d) for d in cfg["delay_grid"]]
    noise_conditions = cfg["noise_conditions"]

    # Per-setting child seeds (one per strategy × noise × delay × run cell).
    rng_seq = np.random.SeedSequence(int(cfg.get("seed", 0)))
    n_grid = (
        len(strategies)
        * len(noise_conditions)
        * len(delay_grid)
        * len(run_logs)
    )
    child_seeds = rng_seq.spawn(n_grid)
    seed_iter = iter(child_seeds)

    print(
        f"\nGrid sweep: {len(strategies)} strategies x "
        f"{len(noise_conditions)} noise x "
        f"{len(delay_grid)} delays x {len(run_logs)} runs = {n_grid} settings"
    )

    summary: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for strategy in strategies:
        for noise_name, noise_sigma in noise_conditions.items():
            for delay in delay_grid:
                for run_path, index, logs in run_logs:
                    child = next(seed_iter)
                    child_seed = int(child.generate_state(1)[0])
                    run_label = _short_run_label(run_path)
                    controller = run_to_controller.get(run_label, run_label)
                    if strategy == "learned":
                        if isinstance(learned_predictor, StateAwareGainPredictor):
                            estimator = StateAwareLearnedGainKalmanEstimator(
                                forward_model=forward_model,
                                gain_predictor=learned_predictor,
                                state_spec=state_spec,
                                delay_steps=int(delay),
                                controller_name=controller,
                                noise_sigma=noise_sigma,
                                controller_names=controller_names,
                                sigma_field_order=sigma_field_order,
                                delay_max=delay_max,
                            )
                            gain_used = float("nan")  # filled after eval
                        else:
                            estimator = LearnedGainKalmanEstimator(
                                forward_model=forward_model,
                                gain_predictor=learned_predictor,
                                state_spec=state_spec,
                                delay_steps=int(delay),
                                controller_name=controller,
                                noise_sigma=noise_sigma,
                                controller_names=controller_names,
                                sigma_field_order=sigma_field_order,
                                delay_max=delay_max,
                            )
                            gain_used = float(estimator.learned_k)
                    else:
                        gain_value = _resolve_strategy_k(
                            strategy=strategy,
                            controller=controller,
                            noise_condition=noise_name,
                            delay_steps=int(delay),
                            oracle_lookup=oracle_lookup,
                            global_best_k=global_best_k,
                            best_per_delay=best_per_delay,
                            best_per_noise=best_per_noise,
                        )
                        assert gain_value is not None
                        estimator = FixedGainKalmanEstimator(
                            forward_model=forward_model,
                            gain=float(gain_value),
                            state_spec=state_spec,
                            delay_steps=int(delay),
                        )
                        gain_used = float(gain_value)

                    results = []
                    state_aware_k_log: list[float] = []
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
                        if isinstance(estimator, StateAwareLearnedGainKalmanEstimator):
                            state_aware_k_log.extend(estimator.k_history)
                    metrics = aggregate_estimation_metrics(
                        results, skip_cold_start=skip_cold_start,
                    )
                    if state_aware_k_log:
                        k_arr = np.asarray(state_aware_k_log, dtype=np.float64)
                        gain_used = float(k_arr.mean())
                        metrics["learned_k_mean"] = float(k_arr.mean())
                        metrics["learned_k_std"] = float(k_arr.std())
                        metrics["learned_k_min"] = float(k_arr.min())
                        metrics["learned_k_max"] = float(k_arr.max())

                    setting_dir = (
                        per_setting_dir
                        / f"strategy_{strategy}"
                        / f"noise_{noise_name}"
                        / f"gain_{gain_used:.4f}_delay_{delay}"
                    )
                    setting_dir.mkdir(parents=True, exist_ok=True)
                    payload: dict[str, Any] = {
                        "strategy": strategy,
                        "controller": controller,
                        "noise_condition": noise_name,
                        "delay_steps": int(delay),
                        "gain_used": gain_used,
                        "noise_sigma": noise_sigma,
                        "obs_compose": obs_compose,
                        "obs_noise_seed": child_seed,
                        "skip_cold_start": skip_cold_start,
                        "model_run_id": forward_model_run_id,
                        "learned_gain_model_id": learned_model_id
                        if strategy == "learned"
                        else "",
                        **metrics,
                    }
                    (setting_dir / f"{controller}.json").write_text(
                        json.dumps(payload, indent=2)
                    )
                    summary.append(payload)
                    csv_rows.append({
                        "strategy": strategy,
                        "controller": controller,
                        "noise_condition": noise_name,
                        "delay_steps": int(delay),
                        "gain_used": gain_used,
                        "learned_k_std": metrics.get("learned_k_std", ""),
                        "learned_k_min": metrics.get("learned_k_min", ""),
                        "learned_k_max": metrics.get("learned_k_max", ""),
                        "tip_estimation_error_mean": metrics.get(
                            "tip_estimation_error_mean", float("nan")
                        ),
                        "tip_estimation_error_final": metrics.get(
                            "tip_estimation_error_final", float("nan")
                        ),
                        "state_mse_mean": metrics.get(
                            "state_mse_mean", float("nan")
                        ),
                        "n_episodes": metrics.get("n", 0),
                        "model_run_id": forward_model_run_id,
                        "learned_gain_model_id": learned_model_id
                        if strategy == "learned"
                        else "",
                    })
                    print(
                        f"  {strategy:<14s} noise={noise_name:>6}  "
                        f"delay={delay:>2}  ctrl={controller:30s}  "
                        f"K={gain_used:.4f}  "
                        f"tip_err={metrics.get('tip_estimation_error_mean', float('nan')):.4f}"
                    )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "eval_id": eval_id,
        "config_hash": config_hash,
        "forward_model_run_id": forward_model_run_id,
        "learned_gain_model_id": learned_model_id,
        "strategies": strategies,
        "delay_grid": delay_grid,
        "noise_conditions": noise_conditions,
        "runs": [str(r) for r in cfg["runs"]],
        "obs_compose": obs_compose,
        "skip_cold_start": skip_cold_start,
        "global_best_k": global_best_k,
        "best_per_delay": best_per_delay,
        "best_per_noise": best_per_noise,
        "results": summary,
    }, indent=2))
    print(f"\nSaved summary: {summary_path}")

    comparison_path = out_dir / "comparison.csv"
    delta_path = out_dir / "delta_vs_oracle.csv"
    _write_comparison_csv(csv_rows, comparison_path)
    _write_delta_vs_oracle_csv(csv_rows, delta_path)
    print(f"Saved comparison csv: {comparison_path}")
    print(f"Saved delta_vs_oracle csv: {delta_path}")
    return out_dir


if __name__ == "__main__":
    main()
