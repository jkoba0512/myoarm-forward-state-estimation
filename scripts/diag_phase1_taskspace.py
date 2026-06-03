"""Phase 1 sanity check: drop-in compare joint-PD vs endpoint_feedback
on the same 5 diagnostic episodes, with K=1 oracle / noise=none / d=0.

Both controllers share Kp=30, Kd=3, action_scale=5 (R3 baseline). The
endpoint_feedback controller uses the Jacobian and moment-arm matrix
captured at episode start (neutral pose) — i.e. fixed-J task-space PD.

Goal: how much of the joint-PD bottleneck disappears just by removing
the IK pre-step and pushing in task-space?
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
EP_IDS = [0, 1, 2, 5, 10]


def run_one(env, fm, state_spec, target_set, ep_idx, controller_name,
            controller_spec, action_adapter, action_dim):
    target_pos = target_set.target_pos[ep_idx]
    env.reset()
    mj.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)

    if controller_name == "joint_pd":
        target_qpos, ik_info = solve_ik(
            env, target_pos,
            max_iter=int(controller_spec["ik_max_iter"]),
            tol=float(controller_spec["ik_tol"]),
            damping=float(controller_spec["ik_damping"]),
        )
        moment_arm = actuator_moment_dense(env)
        controller = JointSpacePDController(
            action_dim=action_dim,
            target_qpos=target_qpos, moment_arm=moment_arm,
            Kp=float(controller_spec["Kp"]),
            Kd=float(controller_spec["Kd"]),
            action_scale=float(controller_spec["action_scale"]),
        )
        ik_err = ik_info["final_error"]
    elif controller_name == "endpoint_feedback":
        jacobian = tip_jacobian_dense(env)
        moment_arm = actuator_moment_dense(env)
        controller = EndpointErrorFeedbackController(
            action_dim=action_dim,
            target_pos=target_pos,
            jacobian=jacobian, moment_arm=moment_arm,
            Kp=float(controller_spec["Kp"]),
            Kd=float(controller_spec["Kd"]),
            action_scale=float(controller_spec["action_scale"]),
        )
        ik_err = None
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

    return {
        "ik_err": ik_err,
        "min_tip": float(err.min()),
        "t_min": int(err.argmin()),
        "final_tip": float(err[-1]),
        "init_tip_to_target": float(err[0]),
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
        # Use the R3 joint-PD hyperparams for both controllers
        controller_spec = cfg["controller"]

        results = {ctrl: {} for ctrl in ("joint_pd", "endpoint_feedback")}
        for ctrl in results:
            for ep in EP_IDS:
                results[ctrl][ep] = run_one(
                    env, fm, state_spec, target_set, ep, ctrl,
                    controller_spec, action_adapter, action_dim,
                )

        print("\n=== Phase 1: joint-PD vs endpoint_feedback  "
              "(K=1 oracle, noise=none, d=0, Kp=30/Kd=3/as=5) ===\n")
        print(f"{'ep':>3} | {'init_dist':>10} | "
              f"{'jPD min_tip':>12} {'jPD final':>10} {'jPD t_min':>10} | "
              f"{'ef  min_tip':>12} {'ef  final':>10} {'ef  t_min':>10} | "
              f"{'Δ min_tip':>12} {'% gain':>8}")
        for ep in EP_IDS:
            jr = results["joint_pd"][ep]
            er = results["endpoint_feedback"][ep]
            delta = er["min_tip"] - jr["min_tip"]
            gain = 100.0 * (jr["min_tip"] - er["min_tip"]) / jr["min_tip"]
            print(
                f"{ep:>3} | {jr['init_tip_to_target']:>10.4f} | "
                f"{jr['min_tip']:>12.4f} {jr['final_tip']:>10.4f} "
                f"{jr['t_min']:>10} | "
                f"{er['min_tip']:>12.4f} {er['final_tip']:>10.4f} "
                f"{er['t_min']:>10} | "
                f"{delta:>+12.4f} {gain:>+7.1f}%"
            )

        # Group stats
        jpd_mins = np.array([results["joint_pd"][ep]["min_tip"] for ep in EP_IDS])
        efb_mins = np.array([results["endpoint_feedback"][ep]["min_tip"] for ep in EP_IDS])
        init_dist = np.array([results["joint_pd"][ep]["init_tip_to_target"] for ep in EP_IDS])
        print("\nmean over 5 eps:")
        print(f"  joint_pd          min_tip = {jpd_mins.mean():.4f} m")
        print(f"  endpoint_feedback min_tip = {efb_mins.mean():.4f} m")
        print(f"  init→target dist  mean    = {init_dist.mean():.4f} m")
        print(f"  joint_pd          moved   = {(init_dist - jpd_mins).mean():.4f} m "
              f"({100*(init_dist - jpd_mins).mean()/init_dist.mean():.1f}% of init)")
        print(f"  endpoint_feedback moved   = {(init_dist - efb_mins).mean():.4f} m "
              f"({100*(init_dist - efb_mins).mean()/init_dist.mean():.1f}% of init)")
        print(f"  endpoint_feedback gain    = "
              f"{100*(jpd_mins.mean()-efb_mins.mean())/jpd_mins.mean():.1f}% "
              f"vs joint_pd")
    finally:
        env.close()


if __name__ == "__main__":
    main()
