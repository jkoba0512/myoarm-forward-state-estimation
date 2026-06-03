"""Verification 1 + 3: lower-Kp sweep on both controllers, with muscle
activation saturation diagnostics.

Hypothesis: current PD demands an instantaneous force proportional to
the (large) tip-to-target error. Muscle activations clip at [0, 1], so
the demand can exceed the model's force capability the moment the
target is set. Lowering Kp should relieve the saturation and let the
arm actually move.

Outputs for each (controller, ep, Kp):
  - min_tip, t_min, final_tip
  - mean muscle activation
  - fraction of (muscle, step) pairs at >= 0.95 / >= 0.99 (saturation)
  - max activation reached
  - time-step at which saturation first crosses 50% of muscles
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco as mj  # noqa: E402

from myoarm_fse.controllers import (  # noqa: E402
    EndpointErrorFeedbackController,
    JointSpacePDController,
)
from myoarm_fse.data.rollout import EpisodeSpec  # noqa: E402
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim  # noqa: E402
from myoarm_fse.envs.factory import make_env  # noqa: E402
from myoarm_fse.envs.ik import (  # noqa: E402
    actuator_moment_dense, solve_ik, tip_jacobian_dense,
)
from myoarm_fse.envs.state import StateSpec  # noqa: E402
from myoarm_fse.envs.targets import TargetSet  # noqa: E402
from myoarm_fse.estimators import FixedGainKalmanEstimator  # noqa: E402
from myoarm_fse.estimators.fixed_kalman import _flatten_log_states  # noqa: E402
from myoarm_fse.evaluation import run_closed_loop_episode  # noqa: E402
from myoarm_fse.models import load_model  # noqa: E402

CONFIG = REPO / "configs/closed_loop/oracle_k_sweep_r3.yaml"
EP_IDS = [0, 1, 10]
KP_GRID = [0.3, 1.0, 3.0, 10.0, 30.0]
KD_RATIO = 0.1  # Kd = Kp * KD_RATIO  (preserves PD damping ratio)
ACTION_SCALE = 5.0  # R3 baseline


def run_one(env, fm, state_spec, target_set, ep_idx, controller_name,
            Kp, Kd, action_scale, controller_spec_base, action_adapter,
            action_dim):
    target_pos = target_set.target_pos[ep_idx]
    env.reset()
    mj.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)

    if controller_name == "joint_pd":
        target_qpos, _ = solve_ik(
            env, target_pos,
            max_iter=int(controller_spec_base["ik_max_iter"]),
            tol=float(controller_spec_base["ik_tol"]),
            damping=float(controller_spec_base["ik_damping"]),
        )
        moment_arm = actuator_moment_dense(env)
        controller = JointSpacePDController(
            action_dim=action_dim,
            target_qpos=target_qpos, moment_arm=moment_arm,
            Kp=Kp, Kd=Kd, action_scale=action_scale,
        )
    elif controller_name == "endpoint_feedback":
        jacobian = tip_jacobian_dense(env)
        moment_arm = actuator_moment_dense(env)
        controller = EndpointErrorFeedbackController(
            action_dim=action_dim, target_pos=target_pos,
            jacobian=jacobian, moment_arm=moment_arm,
            Kp=Kp, Kd=Kd, action_scale=action_scale,
        )
    else:
        raise ValueError(controller_name)
    controller.reset(seed=0)

    estimator = FixedGainKalmanEstimator(
        forward_model=fm, gain=1.0,
        state_spec=state_spec, delay_steps=0,
    )
    spec = EpisodeSpec(
        episode_id=ep_idx,
        target_id=str(int(target_set.seeds[ep_idx])),
        target_split=target_set.split,
        target_seed=int(target_set.seeds[ep_idx]),
        controller_name=controller_name,
        controller_seed=0, sdn_seed=0, obs_noise_seed=0,
        config_hash="diag",
        meta={"estimator_name": "K=1.0"},
    )
    result = run_closed_loop_episode(
        env, controller, estimator, target_pos,
        state_spec=state_spec, action_adapter=action_adapter,
        sdn=None, obs_noise=None, obs_delay=None,
        obs_compose="noisy_then_delayed", max_steps=600, spec=spec,
    )
    x_true = _flatten_log_states(result.log).astype(np.float64)
    layout = state_spec.layout()
    true_tip = x_true[:, layout["tip_pos"]]
    err = np.linalg.norm(true_tip - np.asarray(target_pos), axis=1)

    # muscle activation (T, nu); env logs MuJoCo's internal activation
    act = np.asarray(result.log.true_act, dtype=np.float64)  # (T, nu)
    nu = act.shape[1]
    sat95 = float((act >= 0.95).mean())
    sat99 = float((act >= 0.99).mean())
    max_act = float(act.max())
    mean_act = float(act.mean())
    # first step at which >= 50% of muscles are saturated (>= 0.95)
    sat_per_step = (act >= 0.95).sum(axis=1)
    half_sat_idx = np.argmax(sat_per_step >= nu / 2) if np.any(
        sat_per_step >= nu / 2,
    ) else -1
    return {
        "min_tip": float(err.min()),
        "t_min": int(err.argmin()),
        "final_tip": float(err[-1]),
        "init_tip_to_target": float(err[0]),
        "mean_act": mean_act,
        "max_act": max_act,
        "sat95_frac": sat95,
        "sat99_frac": sat99,
        "half_sat_t": int(half_sat_idx),
        "nu": int(nu),
    }


def main() -> None:
    cfg = yaml.safe_load(open(CONFIG))
    fm, model_cfg, _ = load_model(cfg["forward_model"])
    fm.eval()
    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    env = make_env(cfg["env_id"], horizon=int(cfg["horizon"]))
    try:
        target_set = TargetSet.load(cfg["target_set"])
        action_dim = detect_action_dim(env)
        action_adapter = ActionAdapter(action_dim=action_dim)
        controller_spec_base = cfg["controller"]

        results: dict = {}
        for ctrl in ("joint_pd", "endpoint_feedback"):
            results[ctrl] = {}
            for ep in EP_IDS:
                results[ctrl][ep] = {}
                for Kp in KP_GRID:
                    Kd = Kp * KD_RATIO
                    r = run_one(
                        env, fm, state_spec, target_set, ep, ctrl,
                        Kp, Kd, ACTION_SCALE, controller_spec_base,
                        action_adapter, action_dim,
                    )
                    results[ctrl][ep][Kp] = r

        # ========== reporting ==========
        for ctrl in ("joint_pd", "endpoint_feedback"):
            print(f"\n\n========== {ctrl}  (Kd = Kp * {KD_RATIO}, "
                  f"action_scale = {ACTION_SCALE}, K=1 oracle) ==========")
            for ep in EP_IDS:
                init_d = results[ctrl][ep][KP_GRID[0]]["init_tip_to_target"]
                print(f"\n  ep={ep}  init→target = {init_d:.4f} m")
                hdr = (f"  {'Kp':>6} | {'min_tip':>8} {'t_min':>5} "
                       f"{'final_tip':>9} | {'mean_act':>8} {'max_act':>7} "
                       f"{'sat95%':>7} {'sat99%':>7} {'half_sat_t':>10}")
                print(hdr)
                for Kp in KP_GRID:
                    r = results[ctrl][ep][Kp]
                    print(
                        f"  {Kp:>6.2f} | "
                        f"{r['min_tip']:>8.4f} {r['t_min']:>5} "
                        f"{r['final_tip']:>9.4f} | "
                        f"{r['mean_act']:>8.3f} {r['max_act']:>7.3f} "
                        f"{100*r['sat95_frac']:>6.2f}% "
                        f"{100*r['sat99_frac']:>6.2f}% "
                        f"{r['half_sat_t']:>10}"
                    )

        # ========== best Kp per controller × ep ==========
        print("\n\n=== best Kp per (controller, ep) ===")
        print(f"{'ctrl':<20} {'ep':>3} {'best_Kp':>9} {'min_tip@best':>13} "
              f"{'min_tip@Kp=30':>14}")
        for ctrl in ("joint_pd", "endpoint_feedback"):
            for ep in EP_IDS:
                kps = list(results[ctrl][ep].keys())
                mins = [results[ctrl][ep][k]["min_tip"] for k in kps]
                idx = int(np.argmin(mins))
                best_k = kps[idx]
                print(
                    f"{ctrl:<20} {ep:>3} {best_k:>9.2f} "
                    f"{mins[idx]:>13.4f} "
                    f"{results[ctrl][ep][30.0]['min_tip']:>14.4f}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
