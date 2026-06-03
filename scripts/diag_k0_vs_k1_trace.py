"""One-shot diagnostic: trace x_est vs x_true for K=0 vs K=1 on a single
clean episode (noise=none, delay=0) to verify whether the 4 m
tip_estimation_error at K=0 reflects (a) forward-model autoregressive
drift, (b) a normalization/space mismatch, or (c) state-layout
misalignment.

Outputs:
  runs/diag/k0_vs_k1_trace/K-0.00.npz   raw (x_est, x_true, target_pos, target_qpos)
  runs/diag/k0_vs_k1_trace/K-1.00.npz
  stdout: spot-checks on tip_pos / qpos magnitudes over time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from myoarm_fse.controllers import JointSpacePDController  # noqa: E402
from myoarm_fse.data.rollout import EpisodeSpec  # noqa: E402
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim  # noqa: E402
from myoarm_fse.envs.factory import make_env  # noqa: E402
from myoarm_fse.envs.ik import actuator_moment_dense, solve_ik  # noqa: E402
from myoarm_fse.envs.state import StateSpec  # noqa: E402
from myoarm_fse.envs.targets import TargetSet  # noqa: E402
from myoarm_fse.estimators import FixedGainKalmanEstimator  # noqa: E402
from myoarm_fse.estimators.fixed_kalman import _flatten_log_states  # noqa: E402
from myoarm_fse.evaluation import run_closed_loop_episode  # noqa: E402
from myoarm_fse.models import load_model  # noqa: E402

CONFIG = REPO / "configs/closed_loop/oracle_k_sweep_r3.yaml"
EP_IDX = 0
OUT = REPO / "runs/diag/k0_vs_k1_trace"


def main() -> None:
    cfg = yaml.safe_load(open(CONFIG))
    fm, model_cfg, _ = load_model(cfg["forward_model"])
    fm.eval()
    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    layout = state_spec.layout()
    env = make_env(cfg["env_id"], horizon=int(cfg["horizon"]))
    try:
        target_set = TargetSet.load(cfg["target_set"])
        target_pos = target_set.target_pos[EP_IDX]
        action_dim = detect_action_dim(env)
        action_adapter = ActionAdapter(action_dim=action_dim)
        controller_spec = cfg["controller"]

        OUT.mkdir(parents=True, exist_ok=True)

        for K in (0.0, 1.0):
            import mujoco as mj
            env.reset()
            mj.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
            target_qpos, _ik = solve_ik(
                env, target_pos,
                max_iter=int(controller_spec["ik_max_iter"]),
                tol=float(controller_spec["ik_tol"]),
                damping=float(controller_spec["ik_damping"]),
            )
            moment_arm = actuator_moment_dense(env)
            controller = JointSpacePDController(
                action_dim=action_dim,
                target_qpos=target_qpos,
                moment_arm=moment_arm,
                Kp=float(controller_spec["Kp"]),
                Kd=float(controller_spec["Kd"]),
                action_scale=float(controller_spec["action_scale"]),
            )
            controller.reset(seed=0)
            estimator = FixedGainKalmanEstimator(
                forward_model=fm, gain=K,
                state_spec=state_spec, delay_steps=0,
            )
            spec = EpisodeSpec(
                episode_id=EP_IDX,
                target_id=str(int(target_set.seeds[EP_IDX])),
                target_split=target_set.split,
                target_seed=int(target_set.seeds[EP_IDX]),
                controller_name="joint_pd",
                controller_seed=0,
                sdn_seed=0,
                obs_noise_seed=0,
                config_hash="diag",
                meta={"estimator_name": f"K={K}"},
            )

            result = run_closed_loop_episode(
                env, controller, estimator, target_pos,
                state_spec=state_spec, action_adapter=action_adapter,
                sdn=None, obs_noise=None, obs_delay=None,
                obs_compose="noisy_then_delayed",
                max_steps=int(cfg["horizon"]), spec=spec,
            )

            x_est = np.asarray(result.x_est, dtype=np.float64)
            x_true = _flatten_log_states(result.log).astype(np.float64)

            np.savez(
                OUT / f"K-{K:.2f}.npz",
                x_est=x_est, x_true=x_true,
                target_pos=np.asarray(target_pos),
                target_qpos=np.asarray(target_qpos),
            )

            tip_s = layout["tip_pos"]
            qpos_s = layout["qpos"]
            qvel_s = layout["qvel"]
            tgt_s = layout["target_pos"]

            est_tip = x_est[:, tip_s]
            true_tip = x_true[:, tip_s]
            est_qpos = x_est[:, qpos_s]
            true_qpos = x_true[:, qpos_s]
            est_qvel = x_est[:, qvel_s]
            true_qvel = x_true[:, qvel_s]
            est_tgt = x_est[:, tgt_s]
            true_tgt = x_true[:, tgt_s]

            tip_d = np.linalg.norm(est_tip - true_tip, axis=1)
            qpos_d = np.linalg.norm(est_qpos - true_qpos, axis=1)
            qvel_d = np.linalg.norm(est_qvel - true_qvel, axis=1)
            tgt_d = np.linalg.norm(est_tgt - true_tgt, axis=1)

            T = x_est.shape[0]
            ix = [0, 1, 5, 25, 50, 100, 200, 400, T - 1]

            print(f"\n========== K = {K} ==========")
            print(f"x_est shape={x_est.shape}  x_true shape={x_true.shape}")
            print(f"target_pos (env): {np.asarray(target_pos)}")
            print(f"target_qpos[:5]: {np.asarray(target_qpos)[:5]}")
            print()
            print("--- tip_pos ---")
            print(f"  est_tip range: [{est_tip.min():.3f}, {est_tip.max():.3f}] m")
            print(f"  true_tip range: [{true_tip.min():.3f}, {true_tip.max():.3f}] m")
            print(f"  est_tip[0]  = {est_tip[0]}")
            print(f"  true_tip[0] = {true_tip[0]}")
            print(f"  est_tip[-1]  = {est_tip[-1]}")
            print(f"  true_tip[-1] = {true_tip[-1]}")
            print(f"  tip_dist mean = {tip_d.mean():.4f} m")
            print("  tip_dist trace t = " + ", ".join(
                f"{i}:{tip_d[i]:.3f}" for i in ix
            ))
            print()
            print("--- qpos ---")
            print(f"  est_qpos range: [{est_qpos.min():.3f}, {est_qpos.max():.3f}]")
            print(f"  true_qpos range: [{true_qpos.min():.3f}, {true_qpos.max():.3f}]")
            print(f"  qpos_dist (L2 over 20 dims) trace t = " + ", ".join(
                f"{i}:{qpos_d[i]:.3f}" for i in ix
            ))
            print()
            print("--- qvel ---")
            print(f"  est_qvel range: [{est_qvel.min():.3f}, {est_qvel.max():.3f}]")
            print(f"  true_qvel range: [{true_qvel.min():.3f}, {true_qvel.max():.3f}]")
            print(f"  qvel_dist trace t = " + ", ".join(
                f"{i}:{qvel_d[i]:.3f}" for i in ix
            ))
            print()
            print("--- target_pos (held in state) ---")
            print(f"  est_target[0] = {est_tgt[0]}")
            print(f"  true_target[0]= {true_tgt[0]}")
            print(f"  target_dist trace t = " + ", ".join(
                f"{i}:{tgt_d[i]:.3f}" for i in ix
            ))

    finally:
        env.close()

    print(f"\nSaved npz traces to: {OUT}")


if __name__ == "__main__":
    main()
