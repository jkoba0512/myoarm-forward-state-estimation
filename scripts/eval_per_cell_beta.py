"""Fresh deployment evaluation of the 6 per-cell SPSA-trained betas.

Each cell c trained its own beta from per-cell SPSA (RQ3a). This script
loads each final_beta.json, configures a reliability-adaptive observer
with that beta, and runs 10 fresh episodes on the *training cell only*.
The deployed eval min-tip is compared against the SPSA training-time
outcome (last-20-iter mean) to check for training-vs-deployed drift.

Output: runs/per_cell_beta_diagnostic/<eval_id>/deployed_eval.csv
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

CELLS = [(n, d) for d in (0, 18) for n in ("none", "high", "xhigh")]


def run_episode(env, fm, state_spec, target_pos, noise, delay, beta_cfg, seed):
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

    estimator = ReliabilityAdaptiveObserver(
        forward_model=fm, state_spec=state_spec,
        delay_steps=delay, config=beta_cfg,
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
    p.add_argument("--per-cell-dir", type=Path,
                   default=Path("runs/per_cell_beta_diagnostic/2026-05-15T12-59-01Z"))
    p.add_argument("--forward-model", type=str,
                   default="2026-05-11T11-47-34Z")
    p.add_argument("--target-set", type=Path,
                   default=Path("runs/targets/train.npz"))
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--env-id", type=str, default="myoArmReachFixed-v0")
    args = p.parse_args()

    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    target_set = TargetSet.load(args.target_set)

    fm, _, _ = load_model(f"runs/models/{args.forward_model}")
    fm.eval()

    rows = []
    for noise, delay in CELLS:
        cell_label = f"{noise}_d{delay}"
        cell_dir = args.per_cell_dir / f"cell_{cell_label}"
        beta_dirs = sorted(d for d in cell_dir.iterdir()
                           if d.is_dir() and d.name.startswith("2026-"))
        if not beta_dirs:
            print(f"  {cell_label}: SPSA dir not found, skip")
            continue
        beta_path = beta_dirs[-1] / "final_beta.json"
        if not beta_path.exists():
            print(f"  {cell_label}: final_beta.json not found, skip")
            continue
        beta_dict = json.load(open(beta_path))
        beta_cfg = ReliabilityAdaptiveConfig(
            alpha=float(beta_dict.get("alpha", 0.05)),
            epsilon=float(beta_dict.get("epsilon", 1e-6)),
            var_init=float(beta_dict.get("var_init", 1.0)),
            beta0={k: float(v) for k, v in beta_dict["beta0"].items()},
            beta1={k: float(v) for k, v in beta_dict["beta1"].items()},
            target_pos_gain=float(beta_dict.get("target_pos_gain", 1.0)),
        )

        print(f"\n== cell ({noise}, d={delay}) — deploy beta on {args.episodes} fresh episodes ==")
        t0 = time.time()
        min_tips = []
        for ep in range(args.episodes):
            env = make_env(args.env_id, horizon=600)
            try:
                target_pos = target_set.target_pos[ep % target_set.n].astype(np.float32)
                mt = run_episode(env, fm, state_spec, target_pos,
                                 noise, delay, beta_cfg, seed=ep + 1000)
                min_tips.append(mt)
                print(f"  ep {ep}: min-tip = {mt:.3f} m")
            finally:
                env.close()
        elapsed = time.time() - t0
        mt_arr = np.array(min_tips)
        rows.append({
            "noise": noise,
            "delay": delay,
            "cell": cell_label,
            "deployed_min_tip_mean": float(mt_arr.mean()),
            "deployed_min_tip_std": float(mt_arr.std()),
            "n_episodes": len(mt_arr),
            "elapsed_sec": elapsed,
        })
        print(f"  → mean = {mt_arr.mean():.3f} m, std = {mt_arr.std():.3f} m"
              f", elapsed = {elapsed:.0f}s")

    out_csv = args.per_cell_dir / "deployed_eval.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
