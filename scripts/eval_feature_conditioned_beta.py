"""Deploy the trained feature-conditioned β adapter on 10 fresh episodes
per cell to compare against global SPSA β and per-cell β baselines.

Loads:
  runs/feature_conditioned_beta/<id>/final_W.json
  runs/diagnostics/feature_conditioned/cell_features.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np

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
CELLS = [(n, d) for d in (0, 18) for n in ("none", "high", "xhigh")]


def compute_beta(W, base_beta0, base_beta1, cell_features):
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


def run_episode(env, fm, state_spec, target_pos, noise, delay,
                beta0, beta1, seed):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-conditioned-run", type=Path,
                   default=Path("runs/feature_conditioned_beta/2026-05-17T01-20-04Z"))
    p.add_argument("--cell-features", type=Path,
                   default=Path("runs/diagnostics/feature_conditioned/cell_features.json"))
    p.add_argument("--forward-model", type=str, default="2026-05-11T11-47-34Z")
    p.add_argument("--target-set", type=Path, default=Path("runs/targets/train.npz"))
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()

    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    target_set = TargetSet.load(args.target_set)
    fm, _, _ = load_model(f"runs/models/{args.forward_model}")
    fm.eval()

    final = json.load(open(args.feature_conditioned_run / "final_W.json"))
    W = np.array(final["W_flat"]).reshape(5, 2, 3)
    base_beta0 = final["base_beta0"]
    base_beta1 = final["base_beta1"]

    feat_data = json.load(open(args.cell_features))
    cell_features_map = feat_data["features_zscored"]

    rows = []
    for noise, delay in CELLS:
        cell_label = f"{noise}_d{delay}"
        feats = cell_features_map[cell_label]
        beta0, beta1 = compute_beta(W, base_beta0, base_beta1, feats)
        print(f"\n== {cell_label} ==")
        print(f"  β₀ = {{ {', '.join(f'{f}: {beta0[f]:+.3f}' for f in FIELDS)} }}")
        print(f"  β₁ = {{ {', '.join(f'{f}: {beta1[f]:+.3f}' for f in FIELDS)} }}")
        t0 = time.time()
        mts = []
        for ep in range(args.episodes):
            env = make_env("myoArmReachFixed-v0", horizon=600)
            try:
                target_pos = target_set.target_pos[ep % target_set.n].astype(np.float32)
                mt = run_episode(env, fm, state_spec, target_pos,
                                 noise, delay, beta0, beta1, seed=ep + 5000)
                mts.append(mt)
            finally:
                env.close()
        elapsed = time.time() - t0
        mt_arr = np.array(mts)
        rows.append({
            "noise": noise,
            "delay": delay,
            "cell": cell_label,
            "feat_cond_deployed_mean": float(mt_arr.mean()),
            "feat_cond_deployed_std": float(mt_arr.std()),
            "n_episodes": len(mt_arr),
            "elapsed_sec": elapsed,
        })
        print(f"  → mean = {mt_arr.mean():.3f} m, std = {mt_arr.std():.3f} m"
              f" ({elapsed:.0f}s)")

    out_csv = args.feature_conditioned_run / "deployed_eval.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    grand = float(np.mean([r["feat_cond_deployed_mean"] for r in rows]))
    print(f"\n6-cell mean: {grand:.3f} m")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
