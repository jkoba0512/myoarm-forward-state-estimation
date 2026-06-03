"""Stage A h=600 stability under closed-loop action (companion to
``diag_stage_a_h600_stability.py``).

The zero-action diagnostic showed that under no actuation the four R3
models behave very differently (H=1/H=4 diverge wildly; H=8 main and
undertrained-H=8 stay close to physical state). But the original
artefact (qpos ±12 rad, tip drift 8 m on the H=8 main model) was
observed under a *closed-loop action sequence* — the controller was
feeding actuation in every step, which the forward model then
incorporated into its autoregressive prediction.

This script measures field-wise stability under two closed-loop
controllers (one ep per controller per model):

  - joint-PD K=0          the original artefact-producing controller
  - stabilized_endpoint K=0  the post-pivot controller

Both are run with K=0 (forward-model only) at noise=none / delay=0, on
the same R3 reference target. We use the original (un-filtered) target
set for joint-PD (to reproduce the artefact exactly) and the new
below-shoulder target set for stabilized_endpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco as mj  # noqa: E402

from myoarm_fse.controllers import (  # noqa: E402
    JointSpacePDController,
    StabilizedEndpointController,
)
from myoarm_fse.data.rollout import EpisodeSpec  # noqa: E402
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim  # noqa: E402
from myoarm_fse.envs.extractors import extract_state  # noqa: E402
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

OUT = REPO / "runs/diag/stage_a_h600_closed_loop"

MODELS = {
    "H=1":               "runs/models/2026-05-21T09-25-37Z",
    "H=4":               "runs/models/2026-05-21T09-33-13Z",
    "H=8 (main)":        "runs/models/2026-05-21T09-46-39Z",
    "undertrained-H=8":  "runs/models/2026-05-21T09-54-14Z",
}

TARGET_ORIG = REPO / "runs/targets_reachable/2026-05-21T09-13-52Z/reachable_train.npz"
TARGET_NEW = REPO / "runs/targets_reachable/2026-05-27T07-37-54Z/reachable_train.npz"

EP_IDX = 0
ENV_ID = "myoArmReachFixed-v0"
HORIZON = 600

QVEL_MAX = 10.0
TIP_WORKSPACE = 1.0
TARGET_DRIFT_TOL = 0.05


def _stats(x_est: np.ndarray, state_spec: StateSpec) -> dict:
    layout = state_spec.layout()
    qpos = x_est[:, layout["qpos"]]
    qvel = x_est[:, layout["qvel"]]
    tip = x_est[:, layout["tip_pos"]]
    tgt = x_est[:, layout["target_pos"]]
    nan_step = -1
    for t in range(x_est.shape[0]):
        if not np.isfinite(x_est[t]).all():
            nan_step = t
            break
    over_qvel = np.where(np.abs(qvel).max(axis=1) > QVEL_MAX)[0]
    tip_dist = np.linalg.norm(tip - tip[0], axis=1)
    over_tip = np.where(tip_dist > TIP_WORKSPACE)[0]
    tgt_drift = np.linalg.norm(tgt - tgt[0], axis=1)
    over_tgt = np.where(tgt_drift > TARGET_DRIFT_TOL)[0]
    return {
        "qpos_max_abs": float(np.max(np.abs(qpos))),
        "qvel_max_abs": float(np.max(np.abs(qvel))),
        "tip_dist_max": float(tip_dist.max()),
        "target_drift_max": float(tgt_drift.max()),
        "qvel_first_violation": int(over_qvel[0]) if over_qvel.size else -1,
        "tip_first_violation": int(over_tip[0]) if over_tip.size else -1,
        "target_first_violation": int(over_tgt[0]) if over_tgt.size else -1,
        "first_nan_step": int(nan_step),
        "final_qvel_max_abs": float(np.max(np.abs(qvel[-1]))),
        "final_qpos_max_abs": float(np.max(np.abs(qpos[-1]))),
    }


def run_episode(env, fm, state_spec, target_set, ep_idx, mode,
                action_adapter, action_dim):
    target_pos = target_set.target_pos[ep_idx]
    env.reset()
    mj.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)

    if mode == "joint_pd":
        target_qpos, _ = solve_ik(
            env, target_pos, max_iter=200, tol=0.01, damping=0.1,
        )
        moment_arm = actuator_moment_dense(env)
        controller = JointSpacePDController(
            action_dim=action_dim,
            target_qpos=target_qpos, moment_arm=moment_arm,
            Kp=30.0, Kd=3.0, action_scale=5.0,
        )
    elif mode == "stabilized_endpoint":
        init_tip = np.asarray(extract_state(env).tip_pos, dtype=np.float32)
        jacobian = tip_jacobian_dense(env)
        moment_arm = actuator_moment_dense(env)
        controller = StabilizedEndpointController(
            action_dim=action_dim,
            init_tip=init_tip, target_pos=target_pos,
            jacobian=jacobian, moment_arm=moment_arm,
            Kp=30.0, Kd=3.0, action_scale=5.0, T_ramp=300,
        )
    else:
        raise ValueError(mode)
    controller.reset(seed=0)

    estimator = FixedGainKalmanEstimator(
        forward_model=fm, gain=0.0,
        state_spec=state_spec, delay_steps=0,
    )
    spec = EpisodeSpec(
        episode_id=ep_idx,
        target_id=str(int(target_set.seeds[ep_idx])),
        target_split=target_set.split,
        target_seed=int(target_set.seeds[ep_idx]),
        controller_name=mode, controller_seed=0,
        sdn_seed=0, obs_noise_seed=0, config_hash="stage_a_h600_cl",
        meta={"mode": mode, "K": "0.0"},
    )
    result = run_closed_loop_episode(
        env, controller, estimator, target_pos,
        state_spec=state_spec, action_adapter=action_adapter,
        sdn=None, obs_noise=None, obs_delay=None,
        obs_compose="noisy_then_delayed", max_steps=HORIZON, spec=spec,
    )
    x_est = np.asarray(result.x_est, dtype=np.float64)
    return x_est


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    env = make_env(ENV_ID, horizon=HORIZON)
    try:
        action_dim = detect_action_dim(env)
        action_adapter = ActionAdapter(action_dim=action_dim)

        target_orig = TargetSet.load(str(TARGET_ORIG))
        target_new = TargetSet.load(str(TARGET_NEW))
        print(f"target_orig n={target_orig.n}  z={target_orig.target_pos[EP_IDX][2]:.3f}")
        print(f"target_new  n={target_new.n}   z={target_new.target_pos[EP_IDX][2]:.3f}")

        results: dict = {}
        for name, path in MODELS.items():
            print(f"\n========== {name}  ({path}) ==========")
            fm, _, _ = load_model(path)
            fm.eval()

            # joint-PD on original target_set (artefact reproduction)
            x_est_jpd = run_episode(env, fm, state_spec, target_orig, EP_IDX,
                                     "joint_pd", action_adapter, action_dim)
            s_jpd = _stats(x_est_jpd, state_spec)
            print(f"  joint-PD K=0 (orig target_set):  "
                  f"qpos_max={s_jpd['qpos_max_abs']:.2f}  "
                  f"qvel_max={s_jpd['qvel_max_abs']:.2f}  "
                  f"tip_drift={s_jpd['tip_dist_max']:.2f}  "
                  f"tgt_drift={s_jpd['target_drift_max']:.2f}")

            # stabilized_endpoint on new (below-shoulder) target_set
            x_est_se = run_episode(env, fm, state_spec, target_new, EP_IDX,
                                    "stabilized_endpoint", action_adapter,
                                    action_dim)
            s_se = _stats(x_est_se, state_spec)
            print(f"  stabilized_endpoint K=0 (new tgt): "
                  f"qpos_max={s_se['qpos_max_abs']:.2f}  "
                  f"qvel_max={s_se['qvel_max_abs']:.2f}  "
                  f"tip_drift={s_se['tip_dist_max']:.2f}  "
                  f"tgt_drift={s_se['target_drift_max']:.2f}")

            results[name] = {
                "path": path,
                "joint_pd_K0": s_jpd,
                "stabilized_endpoint_K0": s_se,
            }

            safe_name = (name.replace(' ', '_').replace('(', '')
                              .replace(')', '').replace('=', ''))
            np.savez(
                OUT / f"trace_{safe_name}.npz",
                x_est_joint_pd=x_est_jpd,
                x_est_stabilized=x_est_se,
            )

        # ----- summary -----
        print("\n\n=== closed-loop K=0 / forward model divergence under action ===")
        print(f"qvel threshold = {QVEL_MAX} rad/s, "
              f"tip workspace = {TIP_WORKSPACE} m, "
              f"target drift tol = {TARGET_DRIFT_TOL} m\n")

        for mode_key, mode_label in [
            ("joint_pd_K0", "joint-PD K=0 (artefact reproduction, orig target)"),
            ("stabilized_endpoint_K0", "stabilized_endpoint K=0 (new tgt)"),
        ]:
            print(f"--- {mode_label} ---")
            hdr = (f"{'model':<20} | "
                   f"{'qpos|max':>9} {'qvel|max':>9} {'tip drift':>10} "
                   f"{'tgt drift':>10} | "
                   f"{'qvel viol':>10} {'tip viol':>10} {'tgt viol':>10} "
                   f"{'NaN@':>5}")
            print(hdr); print("-" * len(hdr))
            for name, r in results.items():
                s = r[mode_key]
                print(
                    f"{name:<20} | "
                    f"{s['qpos_max_abs']:>9.2f} {s['qvel_max_abs']:>9.2f} "
                    f"{s['tip_dist_max']:>10.2f} "
                    f"{s['target_drift_max']:>10.2f} | "
                    f"{s['qvel_first_violation']:>10} "
                    f"{s['tip_first_violation']:>10} "
                    f"{s['target_first_violation']:>10} "
                    f"{s['first_nan_step']:>5}"
                )
            print()

        with open(OUT / "results.json", "w") as f:
            json.dump({
                "gates": {"QVEL_MAX": QVEL_MAX,
                          "TIP_WORKSPACE": TIP_WORKSPACE,
                          "TARGET_DRIFT_TOL": TARGET_DRIFT_TOL},
                "ep_idx": EP_IDX,
                "models": results,
            }, f, indent=2)
        print(f"saved: {OUT / 'results.json'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
