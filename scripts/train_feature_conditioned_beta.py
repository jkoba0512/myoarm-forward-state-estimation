"""Train a feature-conditioned β adapter via SPSA.

Diagonal field-wise linear model (codex recommended Model B, RQ3b):

  per field f (qpos, qvel, act, tip_pos, reach_err):
    Δβ₀_f = w0_mean_f · z_mean_f + w0_var_f · z_var_f + b0_f
    Δβ₁_f = w1_mean_f · z_mean_f + w1_var_f · z_var_f + b1_f

  effective β_f = β_base_f + Δβ_f

Total adapter weights: 5 fields × (2 outputs × 3 params) = 30.

`β_base` is taken as the **global SPSA β** from the multi-cell fullgrid
training (codex primary recommendation).

Input features come from `cell_features.json` (pre-computed by
`scripts/compute_per_cell_features.py`): per-cell z-scored log innovation
statistics.

SPSA optimises the 30 adapter weights to minimise 6-cell mean min-tip.

Outputs (under `runs/feature_conditioned_beta/<eval_id>/`):
  history.json
  final_W.json    (the 30 trained weights)
  config.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
import yaml

from myoarm_fse.controllers import JointSpacePDController
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim
from myoarm_fse.envs.factory import make_env
from myoarm_fse.envs.ik import actuator_moment_dense, solve_ik
from myoarm_fse.envs.state import StateSpec
from myoarm_fse.envs.targets import TargetSet
from myoarm_fse.envs.wrappers import (
    DelayedObservationWrapper,
    NoisyObservationWrapper,
)
from myoarm_fse.estimators import (
    ReliabilityAdaptiveConfig,
    ReliabilityAdaptiveObserver,
)
from myoarm_fse.evaluation.closed_loop import run_closed_loop_episode
from myoarm_fse.metrics.reaching import minimum_tip_error
from myoarm_fse.models import load_model


NOISE_PRESETS = {
    "none":  {"qpos": 0.0,  "qvel": 0.0,  "tip_pos": 0.0,  "reach_err": 0.0},
    "high":  {"qpos": 0.02, "qvel": 0.02, "tip_pos": 0.01, "reach_err": 0.01},
    "xhigh": {"qpos": 0.08, "qvel": 0.08, "tip_pos": 0.04, "reach_err": 0.04},
}
FIELDS = ("qpos", "qvel", "act", "tip_pos", "reach_err")
N_FIELDS = len(FIELDS)
CELLS = [(n, d) for d in (0, 18) for n in ("none", "high", "xhigh")]


def W_flat_to_dict(W_flat: np.ndarray) -> np.ndarray:
    """Reshape 30-dim vector to (5 fields, 2 outputs, 3 params [w_mean, w_var, b])."""
    return W_flat.reshape(N_FIELDS, 2, 3)


def compute_beta(
    W: np.ndarray,  # (5, 2, 3)
    base_beta0: dict[str, float],
    base_beta1: dict[str, float],
    cell_features: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute (β₀, β₁) per field from adapter weights and z-scored features."""
    beta0 = dict(base_beta0)
    beta1 = dict(base_beta1)
    for i, f in enumerate(FIELDS):
        x_mean = cell_features[f"mean_{f}"]
        x_var = cell_features[f"var_{f}"]
        db0 = W[i, 0, 0] * x_mean + W[i, 0, 1] * x_var + W[i, 0, 2]
        db1 = W[i, 1, 0] * x_mean + W[i, 1, 1] * x_var + W[i, 1, 2]
        beta0[f] = base_beta0[f] + db0
        beta1[f] = base_beta1[f] + db1
    return beta0, beta1


def episode_outcome(
    env_id, fm, state_spec, target_pos, noise, delay, beta0, beta1, seed,
) -> float:
    env = make_env(env_id, horizon=600)
    try:
        action_dim = detect_action_dim(env)
        action_adapter = ActionAdapter(action_dim=action_dim)
        env.reset(seed=seed)
        mujoco.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
        target_qpos, _ = solve_ik(env, target_pos)
        moment_arm = actuator_moment_dense(env)
        controller = JointSpacePDController(
            action_dim=action_dim, target_qpos=target_qpos,
            moment_arm=moment_arm, Kp=30, Kd=3, action_scale=5,
        )
        controller.reset(seed=seed)
        sigma = NOISE_PRESETS[noise]
        obs_noise = (
            NoisyObservationWrapper(spec=state_spec, sigma=sigma, rng=seed)
            if any(v != 0 for v in sigma.values()) else None
        )
        obs_delay = (
            DelayedObservationWrapper(spec=state_spec, delay_steps=delay)
            if delay > 0 else None
        )
        cfg = ReliabilityAdaptiveConfig(beta0=beta0, beta1=beta1)
        estimator = ReliabilityAdaptiveObserver(
            forward_model=fm, state_spec=state_spec,
            delay_steps=delay, config=cfg,
        )
        result = run_closed_loop_episode(
            env, controller, estimator, target_pos,
            state_spec=state_spec, action_adapter=action_adapter,
            obs_noise=obs_noise, obs_delay=obs_delay,
            obs_compose="noisy_then_delayed", max_steps=600,
        )
        return float(minimum_tip_error(result.log))
    finally:
        env.close()


def aggregate_outcome(
    W: np.ndarray, base_beta0, base_beta1,
    cell_features_map: dict, fm, state_spec, target_set,
    env_id: str, samples_per_cell: int, rng, seed_base: int,
) -> float:
    """Average min-tip across 6 cells × samples_per_cell episodes."""
    outcomes = []
    for noise, delay in CELLS:
        cell_label = f"{noise}_d{delay}"
        features = cell_features_map[cell_label]
        beta0, beta1 = compute_beta(W, base_beta0, base_beta1, features)
        for s in range(samples_per_cell):
            target_idx = int(rng.integers(0, target_set.n))
            target_pos = target_set.target_pos[target_idx].astype(np.float32)
            seed = seed_base + s * 1000 + hash(cell_label) % 1000
            mt = episode_outcome(
                env_id, fm, state_spec, target_pos,
                noise, delay, beta0, beta1, seed,
            )
            outcomes.append(mt)
    return float(np.mean(outcomes))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forward-model", type=str, default="2026-05-11T11-47-34Z")
    p.add_argument("--target-set", type=Path, default=Path("runs/targets/train.npz"))
    p.add_argument("--cell-features", type=Path,
                   default=Path("runs/diagnostics/feature_conditioned/cell_features.json"))
    p.add_argument("--base", choices=["default", "global_spsa"], default="global_spsa")
    p.add_argument("--global-spsa-beta", type=Path,
                   default=Path("runs/reliability_adaptive_v2/2026-05-13T23-40-35Z/final_beta.json"))
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--samples-per-cell", type=int, default=2)
    p.add_argument("--a", type=float, default=0.5)
    p.add_argument("--c", type=float, default=0.3)
    p.add_argument("--A", type=float, default=10.0)
    p.add_argument("--alpha-spall", type=float, default=0.602)
    p.add_argument("--gamma-spall", type=float, default=0.101)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", type=Path, default=Path("runs/feature_conditioned_beta"))
    args = p.parse_args()

    eval_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = args.output_root / eval_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Save config (convert PosixPath to str for yaml)
    cfg = {k: (str(v) if hasattr(v, "__fspath__") else v) for k, v in vars(args).items()}
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    target_set = TargetSet.load(args.target_set)
    fm, _, _ = load_model(f"runs/models/{args.forward_model}")
    fm.eval()

    # Load per-cell features
    feat_data = json.load(open(args.cell_features))
    cell_features_map = feat_data["features_zscored"]

    # Load base β
    if args.base == "default":
        base_beta0 = {f: 0.0 for f in FIELDS}
        base_beta1 = {f: 0.5 for f in FIELDS}
        print(f"Base: default reliability (β₀=0, β₁=0.5)")
    else:
        gs = json.load(open(args.global_spsa_beta))
        base_beta0 = {f: float(gs["beta0"][f]) for f in FIELDS}
        base_beta1 = {f: float(gs["beta1"][f]) for f in FIELDS}
        print(f"Base: global SPSA β:")
        print(f"  β₀ = {base_beta0}")
        print(f"  β₁ = {base_beta1}")

    rng = np.random.default_rng(args.seed)
    W_dim = N_FIELDS * 2 * 3  # 30
    W_flat = np.zeros(W_dim, dtype=np.float32)  # init at zero → adapter is identity to base

    print(f"\nSPSA: W_dim={W_dim}, max_iter={args.max_iter}, "
          f"S={args.samples_per_cell} per cell × 6 cells = "
          f"{args.samples_per_cell * 6} episodes per side per iter")

    history = []
    for n in range(args.max_iter):
        t0 = time.time()
        a_n = args.a / (n + 1 + args.A) ** args.alpha_spall
        c_n = args.c / (n + 1) ** args.gamma_spall

        delta = (rng.integers(0, 2, size=W_dim) * 2 - 1).astype(np.float32)
        W_plus = W_flat + c_n * delta
        W_minus = W_flat - c_n * delta

        o_plus = aggregate_outcome(
            W_flat_to_dict(W_plus), base_beta0, base_beta1,
            cell_features_map, fm, state_spec, target_set,
            "myoArmReachFixed-v0", args.samples_per_cell, rng,
            seed_base=10000 + n * 100 + 0,
        )
        o_minus = aggregate_outcome(
            W_flat_to_dict(W_minus), base_beta0, base_beta1,
            cell_features_map, fm, state_spec, target_set,
            "myoArmReachFixed-v0", args.samples_per_cell, rng,
            seed_base=10000 + n * 100 + 50,
        )
        g_hat = (o_plus - o_minus) / (2 * c_n) * (1.0 / delta)
        W_flat = W_flat - a_n * g_hat
        elapsed = time.time() - t0
        history.append({
            "iter": n,
            "a_n": a_n,
            "c_n": c_n,
            "outcome_plus": o_plus,
            "outcome_minus": o_minus,
            "outcome_mean": (o_plus + o_minus) / 2,
            "elapsed_sec": elapsed,
        })
        print(f"  iter {n:3d}: a={a_n:.3f} c={c_n:.3f} "
              f"o_plus={o_plus:.3f} o_minus={o_minus:.3f} "
              f"mean={(o_plus + o_minus)/2:.3f} {elapsed:.0f}s")
        if (n + 1) % 10 == 0:
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            with open(out_dir / "final_W.json", "w") as f:
                json.dump({
                    "W_flat": W_flat.tolist(),
                    "base_beta0": base_beta0,
                    "base_beta1": base_beta1,
                    "iter": n + 1,
                }, f, indent=2)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "final_W.json", "w") as f:
        json.dump({
            "W_flat": W_flat.tolist(),
            "base_beta0": base_beta0,
            "base_beta1": base_beta1,
        }, f, indent=2)
    print(f"\nFinal W saved.")


if __name__ == "__main__":
    main()
