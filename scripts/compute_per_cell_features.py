"""Pre-compute per-cell innovation feature vectors for feature-conditioned β.

For each of the 6 focused-grid cells, run N default-reliability episodes
on the H=8 forward model, collect innovation statistics per field, and
output a normalised (log + z-score) feature vector that serves as the
"context" for the feature-conditioned β adapter.

Output: runs/diagnostics/feature_conditioned/cell_features.json
  per cell: {
    "innov_mean_f": log mean(|e_f|) over second half of T,
    "innov_var_f":  log var(e_f²) ...
  }
  + global normalization stats (mean, std per feature across cells)
"""

from __future__ import annotations

import argparse
import json
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
from myoarm_fse.models import load_model


NOISE_PRESETS = {
    "none":  {"qpos": 0.0,  "qvel": 0.0,  "tip_pos": 0.0,  "reach_err": 0.0},
    "high":  {"qpos": 0.02, "qvel": 0.02, "tip_pos": 0.01, "reach_err": 0.01},
    "xhigh": {"qpos": 0.08, "qvel": 0.08, "tip_pos": 0.04, "reach_err": 0.04},
}
CELLS = [(n, d) for d in (0, 18) for n in ("none", "high", "xhigh")]
FIELDS = ("qpos", "qvel", "act", "tip_pos", "reach_err")


def run_episode_capture_innovation(
    env, fm, state_spec, target_pos, noise, delay, seed,
):
    """Run one default-reliability episode and return per-field innovation
    second-half statistics (mean abs and variance)."""
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
    cfg = ReliabilityAdaptiveConfig()  # default reliability
    estimator = ReliabilityAdaptiveObserver(
        forward_model=fm, state_spec=state_spec,
        delay_steps=delay, config=cfg,
    )
    _ = run_closed_loop_episode(
        env, controller, estimator, target_pos,
        state_spec=state_spec, action_adapter=action_adapter,
        obs_noise=obs_noise, obs_delay=obs_delay,
        obs_compose="noisy_then_delayed", max_steps=600,
    )
    innov = estimator.innovation_history()  # (T, 83)
    T = innov.shape[0]
    half = T // 2

    # Field index ranges (must match ReliabilityAdaptiveObserver internal)
    idx = {
        "qpos":      (0, 20),
        "qvel":      (20, 40),
        "act":       (40, 74),
        "tip_pos":   (74, 77),
        "reach_err": (80, 83),
    }
    out = {}
    for f, (lo, hi) in idx.items():
        slc = innov[half:, lo:hi]
        out[f"mean_{f}"] = float(np.mean(np.abs(slc)))
        out[f"var_{f}"] = float(np.mean(slc ** 2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forward-model", type=str, default="2026-05-11T11-47-34Z")
    p.add_argument("--target-set", type=Path, default=Path("runs/targets/train.npz"))
    p.add_argument("--n-warmup-episodes", type=int, default=5)
    p.add_argument("--output", type=Path,
                   default=Path("runs/diagnostics/feature_conditioned/cell_features.json"))
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    target_set = TargetSet.load(args.target_set)
    fm, _, _ = load_model(f"runs/models/{args.forward_model}")
    fm.eval()

    per_cell_raw = {}
    for noise, delay in CELLS:
        cell = f"{noise}_d{delay}"
        print(f"== {cell} ==")
        # average over n_warmup_episodes
        stat_lists = {f"mean_{f}": [] for f in FIELDS}
        stat_lists.update({f"var_{f}": [] for f in FIELDS})
        for ep in range(args.n_warmup_episodes):
            env = make_env("myoArmReachFixed-v0", horizon=600)
            try:
                target_pos = target_set.target_pos[ep].astype(np.float32)
                stats = run_episode_capture_innovation(
                    env, fm, state_spec, target_pos, noise, delay, seed=2000 + ep,
                )
                for k, v in stats.items():
                    stat_lists[k].append(v)
            finally:
                env.close()
        per_cell_raw[cell] = {k: float(np.mean(v)) for k, v in stat_lists.items()}
        print(f"  {per_cell_raw[cell]}")

    # log transform + z-score across cells
    feat_names = [f"mean_{f}" for f in FIELDS] + [f"var_{f}" for f in FIELDS]
    n_cells = len(per_cell_raw)
    n_feats = len(feat_names)
    raw_matrix = np.zeros((n_cells, n_feats))
    cell_order = list(per_cell_raw.keys())
    for i, c in enumerate(cell_order):
        for j, fn in enumerate(feat_names):
            raw_matrix[i, j] = per_cell_raw[c][fn]

    log_matrix = np.log1p(raw_matrix)
    feat_mean = log_matrix.mean(axis=0)
    feat_std = log_matrix.std(axis=0) + 1e-12
    z_matrix = (log_matrix - feat_mean) / feat_std

    per_cell_features = {}
    for i, c in enumerate(cell_order):
        per_cell_features[c] = {
            feat_names[j]: float(z_matrix[i, j]) for j in range(n_feats)
        }

    out_dict = {
        "raw": per_cell_raw,
        "features_zscored": per_cell_features,
        "norm_log_mean": dict(zip(feat_names, feat_mean.tolist())),
        "norm_log_std": dict(zip(feat_names, feat_std.tolist())),
        "feat_names": feat_names,
        "cell_order": cell_order,
    }
    with open(args.output, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
