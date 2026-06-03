"""Diagnose the joint-PD + IK controller's reachability with oracle
observation (K=1.0, noise=none, delay=0). If the controller can't hit
the IK target even when the estimator is perfect, the Stage B "best K=0"
result is a controller artefact, not an estimator finding.

Pivots:
  (a) IK info        : did IK converge? what is the residual ‖tip(IK(target_pos)) − target_pos‖?
  (b) PD hyperparams : sweep (Kp, Kd, action_scale)
  (c) target id      : a few episodes to rule out ep=0 being pathological

For each (ep, hyperparams) we run a single K=1 / noise=none / d=0 rollout
and print:
  - IK final_error / converged
  - final_tip_error, min_tip_error  (env's true tip vs target)
  - achieved qpos vs target_qpos at end of episode (joint-space residual)
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

EP_IDS = [0, 1, 2, 5, 10]
HYPERS = [
    # (Kp, Kd, action_scale)   -- (30, 3, 5) is the R3 baseline
    (30.0, 3.0, 5.0),
    (60.0, 6.0, 5.0),
    (30.0, 3.0, 10.0),
    (100.0, 10.0, 10.0),
    (200.0, 20.0, 20.0),
]


def run_one(env, fm, state_spec, target_set, ep_idx, K, hp, controller_spec_base, action_adapter, action_dim):
    import mujoco as mj
    target_pos = target_set.target_pos[ep_idx]
    env.reset()
    mj.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
    target_qpos, ik_info = solve_ik(
        env, target_pos,
        max_iter=int(controller_spec_base["ik_max_iter"]),
        tol=float(controller_spec_base["ik_tol"]),
        damping=float(controller_spec_base["ik_damping"]),
    )
    moment_arm = actuator_moment_dense(env)
    Kp, Kd, action_scale = hp
    controller = JointSpacePDController(
        action_dim=action_dim,
        target_qpos=target_qpos, moment_arm=moment_arm,
        Kp=Kp, Kd=Kd, action_scale=action_scale,
    )
    controller.reset(seed=0)
    estimator = FixedGainKalmanEstimator(
        forward_model=fm, gain=float(K),
        state_spec=state_spec, delay_steps=0,
    )
    spec = EpisodeSpec(
        episode_id=ep_idx,
        target_id=str(int(target_set.seeds[ep_idx])),
        target_split=target_set.split,
        target_seed=int(target_set.seeds[ep_idx]),
        controller_name="joint_pd",
        controller_seed=0, sdn_seed=0, obs_noise_seed=0,
        config_hash="diag",
        meta={"estimator_name": f"K={K}"},
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
    true_qpos = x_true[:, layout["qpos"]]

    err_to_target = np.linalg.norm(true_tip - np.asarray(target_pos), axis=1)
    final_tip = float(err_to_target[-1])
    min_tip = float(err_to_target.min())
    t_min = int(err_to_target.argmin())

    qpos_resid_final = float(np.linalg.norm(true_qpos[-1] - np.asarray(target_qpos)))
    qpos_resid_min = float(np.linalg.norm(true_qpos - np.asarray(target_qpos), axis=1).min())

    return {
        "ik_converged": ik_info["converged"],
        "ik_final_error": ik_info["final_error"],
        "ik_n_iter": ik_info["n_iter"],
        "final_tip": final_tip,
        "min_tip": min_tip,
        "t_min": t_min,
        "qpos_resid_final": qpos_resid_final,
        "qpos_resid_min": qpos_resid_min,
        "target_qpos_norm": float(np.linalg.norm(target_qpos)),
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

        # Per-episode IK reachability (a): how close can IK get to target_pos in tip space?
        print("\n=== (a) IK reachability per episode (K=1 oracle, baseline PD) ===")
        print(f"{'ep':>3} | {'ik_conv':>7} | {'ik_err':>8} | {'ik_iter':>7} | {'min_tip':>8} | {'final_tip':>9} | {'t_min':>5} | {'qpos_resid_min':>14} | {'tgt_qpos_norm':>13}")
        baseline_hp = HYPERS[0]
        for ep in EP_IDS:
            r = run_one(env, fm, state_spec, target_set, ep, 1.0, baseline_hp,
                        controller_spec_base, action_adapter, action_dim)
            print(f"{ep:>3} | {str(r['ik_converged']):>7} | {r['ik_final_error']:>8.4f} | "
                  f"{r['ik_n_iter']:>7} | {r['min_tip']:>8.4f} | {r['final_tip']:>9.4f} | "
                  f"{r['t_min']:>5} | {r['qpos_resid_min']:>14.4f} | {r['target_qpos_norm']:>13.4f}")

        # PD hyperparam sweep on ep=0 (b)
        print("\n=== (b) PD hyperparam sweep on ep=0 (K=1 oracle) ===")
        print(f"{'Kp':>5} {'Kd':>5} {'act_sc':>7} | {'min_tip':>8} | {'final_tip':>9} | {'t_min':>5} | {'qpos_resid_min':>14}")
        for hp in HYPERS:
            r = run_one(env, fm, state_spec, target_set, 0, 1.0, hp,
                        controller_spec_base, action_adapter, action_dim)
            print(f"{hp[0]:>5.1f} {hp[1]:>5.1f} {hp[2]:>7.1f} | "
                  f"{r['min_tip']:>8.4f} | {r['final_tip']:>9.4f} | "
                  f"{r['t_min']:>5} | {r['qpos_resid_min']:>14.4f}")

        # PD sweep × episodes (b × c)
        print("\n=== (b×c) PD sweep × episodes (K=1 oracle, min_tip only) ===")
        header = "ep |" + "".join(f" Kp={hp[0]:>5.0f}/Kd={hp[1]:>4.0f}/as={hp[2]:>4.0f} " for hp in HYPERS)
        print(header)
        for ep in EP_IDS:
            row = f"{ep:>2} |"
            for hp in HYPERS:
                r = run_one(env, fm, state_spec, target_set, ep, 1.0, hp,
                            controller_spec_base, action_adapter, action_dim)
                row += f"     min={r['min_tip']:>6.3f}      "
            print(row)

    finally:
        env.close()


if __name__ == "__main__":
    main()
