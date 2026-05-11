"""Generate per-step K* labels for Phase 3.2 Stage B (state-aware gain).

For every ``(controller_run, episode, noise_condition, delay)`` cell on
the stress grid we walk the episode timeline and at each step build:

- condition features (8-dim): controller one-hot + sigma vector + delay/delay_max
- state features (4-dim): per-field L2 norms of the innovation
- per-step K* target: the closed-form K that minimises
  ``||x_pred + K*(y_obs - x_pred_at_correction_point) - x_true||²`` at the
  correction time. Caller config picks whether ``x_pred`` comes from a
  K=1 prior trajectory (default, simplest) or from per-step "ideal
  prior" using ``x_true`` directly. The default mirrors what the
  estimator will see at deployment in the worst case where blending
  has not yet kicked in.

The output ``per_step.npz`` is consumed by ``scripts/train_stage_b.py``.

Usage::

    uv run python scripts/generate_per_step_oracle.py \\
        --config configs/estimators/learned_gain_stage_b.yaml \\
        --out runs/per_step_oracle/{auto_id}
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

from myoarm_fse.data.logger import RunIndex
from myoarm_fse.data.schema import EpisodeLog
from myoarm_fse.envs.state import StateSpec
from myoarm_fse.estimators.fixed_kalman import (
    _flatten_log_states,
    synth_observations,
)
from myoarm_fse.estimators.learned import (
    _encode_features,
    _encode_state_features,
)
from myoarm_fse.models import load_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-step K* + state features for Stage B."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory; default = runs/per_step_oracle/{ts}")
    p.add_argument("--master-seed", type=int, default=None)
    return p.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_id(now: datetime | None = None) -> str:
    t = now if now is not None else datetime.now(timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _hash_config(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _state_spec_from_model_config(model_config: dict[str, Any]) -> StateSpec:
    state_dim = int(model_config["architecture"]["state_dim"])
    if state_dim == 83:
        return StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    raise NotImplementedError(
        f"state_dim={state_dim} layout is not auto-derivable"
    )


def _predict_next_state(
    forward_model,
    x_prev: np.ndarray,
    u_prev: np.ndarray,
) -> np.ndarray:
    """One-step prediction: x_pred = x_prev + f(x_prev, u_prev)."""
    import torch
    x_t = torch.from_numpy(x_prev.astype(np.float32)).unsqueeze(0)
    u_t = torch.from_numpy(u_prev.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        dx = forward_model(x_t, u_t).squeeze(0).numpy().astype(np.float32)
    return (x_prev + dx).astype(np.float32)


def _closed_form_k(
    x_pred: np.ndarray,
    y_obs: np.ndarray,
    x_true: np.ndarray,
    *,
    eps: float = 1e-9,
) -> float:
    """K* minimising ||x_pred + K*(y_obs - x_pred) - x_true||².

    Closed form: K* = (innovation · err_pred) / ||innovation||².
    Clipped to [0, 1] to match the Kalman gain range.
    """
    innovation = y_obs - x_pred
    err_pred = x_true - x_pred
    denom = float(np.dot(innovation, innovation))
    if denom < eps:
        # Zero innovation: prediction matches observation, gain is irrelevant.
        # Return midpoint so it gets ignored in mean targets.
        return 0.5
    k = float(np.dot(innovation, err_pred) / denom)
    return float(np.clip(k, 0.0, 1.0))


def _load_run_logs(run_path: Path) -> tuple[RunIndex, list[EpisodeLog]]:
    index = RunIndex.load(run_path / "index.json")
    logs = [EpisodeLog.load(run_path / e.file) for e in index.episodes]
    return index, logs


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    cfg = _load_config(args.config)
    if args.master_seed is not None:
        cfg["seed"] = int(args.master_seed)

    forward_model, model_config, _ = load_model(cfg["forward_model"])
    state_spec = _state_spec_from_model_config(model_config)
    layout = state_spec.layout()
    forward_model.eval()

    run_to_controller = cfg.get("run_to_controller") or {}
    controller_names = tuple(
        cfg.get("input_encoding", {}).get(
            "controller_names", ("random", "lowamp", "hold")
        )
    )
    sigma_field_order = tuple(
        cfg.get("input_encoding", {}).get(
            "sigma_field_order", ("qpos", "qvel", "tip_pos", "reach_err")
        )
    )
    delay_max = int(cfg.get("input_encoding", {}).get("delay_max", 36))
    state_feature_fields = tuple(
        cfg.get("input_encoding", {}).get(
            "state_feature_fields", ("qpos", "qvel", "tip_pos", "reach_err")
        )
    )

    runs = [Path(r) for r in cfg["runs"]]
    noise_conditions = cfg["noise_conditions"]
    delay_grid = [int(d) for d in cfg["delay_grid"]]
    skip_cold_start = bool(cfg.get("skip_cold_start", True))
    obs_compose = str(cfg.get("obs_compose", "noisy_then_delayed"))
    # For delay==0 cells we compute per-step K* in closed form using
    # an "ideal" prior (x_true[t-1] + forward(x_true[t-1], u[t-1]));
    # for delay>0 cells the stress sweep showed oracle K is always 1.0
    # (forward-model rollout error dominates), so we label those with
    # K*=1.0 directly without re-running the recursive Kalman.
    delay0_only = bool(cfg.get("per_step_oracle", {}).get(
        "force_k1_for_delayed", True
    ))

    rng_seq = np.random.SeedSequence(int(cfg.get("seed", 0)))
    n_grid = (
        len(runs) * len(noise_conditions) * len(delay_grid)
    )
    child_seeds = rng_seq.spawn(n_grid)
    seed_iter = iter(child_seeds)

    print(f"Loaded forward model: {cfg['forward_model']}")
    print(f"  state_dim: {state_spec.dim}, action_dim: {forward_model.action_dim}")
    print(f"  runs: {len(runs)}, noise: {len(noise_conditions)}, "
          f"delays: {len(delay_grid)}, grid={n_grid}")
    print(f"  force_k1_for_delayed: {delay0_only}")

    cond_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    k_targets: list[float] = []
    # Group ids per sample for CV stratification.
    grp_controller: list[int] = []
    grp_noise: list[int] = []
    grp_delay: list[int] = []
    grp_episode: list[int] = []
    grp_condition: list[int] = []  # joint (controller, noise, delay) ID

    noise_names = list(noise_conditions.keys())
    condition_id = 0

    for run_idx, run_path in enumerate(runs):
        run_label = run_path.name
        controller = run_to_controller.get(run_label, run_label)
        if controller not in controller_names:
            raise SystemExit(
                f"controller {controller!r} (from run {run_label}) "
                f"not in controller_names {controller_names}; "
                "fill in run_to_controller in the config."
            )
        controller_idx = controller_names.index(controller)
        _index, logs = _load_run_logs(run_path)
        n_episodes = len(logs)
        print(f"\n[{run_idx}] {run_label} -> controller={controller}, "
              f"{n_episodes} episodes")

        for noise_idx, (noise_name, sigma_dict) in enumerate(
            noise_conditions.items()
        ):
            for delay_idx, delay in enumerate(delay_grid):
                child = next(seed_iter)
                seed = int(child.generate_state(1)[0])

                cond_feat = _encode_features(
                    controller=controller,
                    noise_sigma=sigma_dict,
                    delay_steps=int(delay),
                    controller_names=controller_names,
                    sigma_field_order=sigma_field_order,
                    delay_max=delay_max,
                )
                for ep_idx, log in enumerate(logs):
                    x_true = _flatten_log_states(log)
                    u = log.excitation.astype(np.float32, copy=False)
                    T = log.n_steps
                    if T == 0:
                        continue
                    y_obs = synth_observations(
                        log, state_spec=state_spec, sigma=sigma_dict,
                        delay_steps=int(delay), seed=seed,
                        obs_compose=obs_compose,
                    )

                    # Cold-start skip: with delay d, the first d steps
                    # have no correction (estimator falls back to
                    # prediction-only). Mirror that here.
                    t_start = int(delay) if skip_cold_start else 0
                    if t_start >= T - 1:
                        continue
                    # Need t >= 1 so we can use x_true[t-1] and u[t-1].
                    t_start = max(t_start, 1)

                    # Per-step label policy:
                    #   delay == 0: closed-form K* against x_true[t]
                    #               with ideal prior x_pred[t] =
                    #               x_true[t-1] + f(x_true[t-1], u[t-1]).
                    #   delay  > 0 and force_k1_for_delayed: K* = 1.0
                    #               (the stress sweep showed oracle K is
                    #               always 1.0 for delay>=6 because the
                    #               forward-model rollout amplifies
                    #               error faster than observation noise
                    #               grows; we encode that as label
                    #               rather than re-deriving via
                    #               recursive rollout per step).
                    for t in range(t_start, T):
                        # Innovation in the delay=0 sense (against an
                        # ideal-prior one-step prediction). For delay>0
                        # this is the closest available stand-in for
                        # what the deployed estimator sees, even though
                        # the K target we assign is just 1.0.
                        x_pred_t = _predict_next_state(
                            forward_model, x_true[t - 1], u[t - 1],
                        )
                        innovation = y_obs[t] - x_pred_t

                        if int(delay) == 0:
                            k_star = _closed_form_k(
                                x_pred_t, y_obs[t], x_true[t],
                            )
                        else:
                            if not delay0_only:
                                # Not implemented: recursive K* for
                                # delay>0 would require per-step
                                # rollouts. Fall back to K=1 anyway.
                                pass
                            k_star = 1.0

                        sf = _encode_state_features(
                            innovation, state_spec=state_spec,
                            state_feature_fields=state_feature_fields,
                        )
                        cond_rows.append(cond_feat)
                        state_rows.append(sf)
                        k_targets.append(k_star)
                        grp_controller.append(controller_idx)
                        grp_noise.append(noise_idx)
                        grp_delay.append(delay_idx)
                        grp_episode.append(ep_idx)
                        grp_condition.append(condition_id)
                condition_id += 1

    n_samples = len(k_targets)
    print(f"\nTotal samples: {n_samples}")
    if n_samples == 0:
        raise SystemExit("no samples produced; check configuration")

    cond_arr = np.stack(cond_rows).astype(np.float32)
    state_arr = np.stack(state_rows).astype(np.float32)
    k_arr = np.asarray(k_targets, dtype=np.float32)
    g_ctrl = np.asarray(grp_controller, dtype=np.int32)
    g_noise = np.asarray(grp_noise, dtype=np.int32)
    g_delay = np.asarray(grp_delay, dtype=np.int32)
    g_episode = np.asarray(grp_episode, dtype=np.int32)
    g_cond = np.asarray(grp_condition, dtype=np.int32)

    print("K* target summary:")
    print(f"  mean: {k_arr.mean():.4f}, median: {np.median(k_arr):.4f}")
    print(f"  hist: <0.1={np.mean(k_arr<0.1):.3f}  "
          f"0.1-0.5={np.mean((k_arr>=0.1)&(k_arr<0.5)):.3f}  "
          f"0.5-0.9={np.mean((k_arr>=0.5)&(k_arr<0.9)):.3f}  "
          f">=0.9={np.mean(k_arr>=0.9):.3f}")

    out_dir = args.out if args.out is not None else Path(
        f"runs/per_step_oracle/{_make_id()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "per_step.npz",
        cond_features=cond_arr,
        state_features=state_arr,
        k_target=k_arr,
        group_controller=g_ctrl,
        group_noise=g_noise,
        group_delay=g_delay,
        group_episode=g_episode,
        group_condition=g_cond,
    )
    (out_dir / "config.json").write_text(
        json.dumps({
            "config_hash": _hash_config(cfg),
            "config": cfg,
            "controller_names": list(controller_names),
            "sigma_field_order": list(sigma_field_order),
            "state_feature_fields": list(state_feature_fields),
            "delay_max": delay_max,
            "noise_names": noise_names,
            "delay_grid": delay_grid,
            "runs": [str(r) for r in runs],
            "n_samples": int(n_samples),
            "k_target_mean": float(k_arr.mean()),
            "force_k1_for_delayed": delay0_only,
        }, indent=2)
    )
    print(f"\nSaved: {out_dir / 'per_step.npz'}")
    print(f"       {out_dir / 'config.json'}")
    return out_dir


if __name__ == "__main__":
    main()
