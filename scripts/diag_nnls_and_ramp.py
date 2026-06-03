"""Diagnose two controller modifications, individually and combined:

A) NNLS muscle routing
   Current `u_muscle = clip(action_scale * relu(moment_arm @ u_joint), 0, 1)`
   uses ReLU which throws away "antagonist" drive components and creates
   inconsistent muscle patterns when a single muscle crosses multiple
   joints. Replace with non-negative least squares:
       activation = argmin ||moment_arm.T @ activation - u_joint||²
                    s.t. activation >= 0

B) Virtual target ramping
   Linearly ramp the controller's target from the initial tip position
   to the actual target over T_ramp steps. This caps the instantaneous
   error magnitude and gives the muscle pattern time to settle —
   essentially a minimum-time-like trajectory plan.

Variants are evaluated on endpoint_feedback with the same Kp/Kd/action_scale
and K=1 oracle / noise=none / d=0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import nnls

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco as mj  # noqa: E402

from myoarm_fse.data.rollout import EpisodeSpec  # noqa: E402
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim  # noqa: E402
from myoarm_fse.envs.extractors import extract_state  # noqa: E402
from myoarm_fse.envs.factory import make_env  # noqa: E402
from myoarm_fse.envs.ik import (  # noqa: E402
    actuator_moment_dense, tip_jacobian_dense,
)
from myoarm_fse.envs.state import MyoArmState, StateSpec  # noqa: E402
from myoarm_fse.envs.targets import TargetSet  # noqa: E402
from myoarm_fse.estimators import FixedGainKalmanEstimator  # noqa: E402
from myoarm_fse.estimators.fixed_kalman import _flatten_log_states  # noqa: E402
from myoarm_fse.evaluation import run_closed_loop_episode  # noqa: E402
from myoarm_fse.models import load_model  # noqa: E402

CONFIG = REPO / "configs/closed_loop/oracle_k_sweep_r3.yaml"
EP_IDS = [0, 1, 10]
KP_GRID = [10.0, 30.0]
KD_RATIO = 0.1
ACTION_SCALE = 5.0
T_RAMP = 300


class EndpointFB_Variant:
    """endpoint_feedback with optional NNLS muscle routing and / or
    target ramping. action_dim and shapes are validated at __init__."""

    def __init__(self, *, action_dim, init_tip, target_pos, jacobian,
                 moment_arm, Kp, Kd, action_scale,
                 use_nnls: bool, use_ramp: bool, T_ramp: int):
        self._action_dim = int(action_dim)
        self._init_tip = np.asarray(init_tip, dtype=np.float32)
        self._final_target = np.asarray(target_pos, dtype=np.float32)
        self._jacobian = np.asarray(jacobian, dtype=np.float32)
        self._moment_arm = np.asarray(moment_arm, dtype=np.float32)
        self._mt = self._moment_arm.T.astype(np.float64)  # (nv, nu)
        self._Kp = float(Kp); self._Kd = float(Kd)
        self._action_scale = float(action_scale)
        self._use_nnls = bool(use_nnls)
        self._use_ramp = bool(use_ramp)
        self._T_ramp = int(T_ramp)
        self._step = 0

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, *, seed=None) -> None:
        del seed
        self._step = 0

    def act(self, observation: MyoArmState) -> np.ndarray:
        if self._use_ramp:
            s = min(self._step / max(self._T_ramp, 1), 1.0)
            cur_target = (
                self._init_tip + s * (self._final_target - self._init_tip)
            )
        else:
            cur_target = self._final_target
        self._step += 1

        tip_est = np.asarray(observation.tip_pos, dtype=np.float32)
        qvel = np.asarray(observation.qvel, dtype=np.float32)

        e_tip = cur_target - tip_est
        v_tip = self._jacobian @ qvel
        u_tip = self._Kp * e_tip - self._Kd * v_tip          # (3,)
        u_joint = self._jacobian.T @ u_tip                   # (nv,)

        if self._use_nnls:
            # Solve activation >= 0 minimizing ||M.T @ a − u_joint||².
            # NNLS scales linearly with nv*nu and is cheap per step.
            activation, _ = nnls(self._mt, u_joint.astype(np.float64))
            u_muscle = np.clip(
                self._action_scale * activation, 0.0, 1.0,
            )
        else:
            drive = self._moment_arm @ u_joint               # (nu,)
            u_muscle = np.clip(
                self._action_scale * np.maximum(drive, 0.0), 0.0, 1.0,
            )
        return u_muscle.astype(np.float32, copy=False)


def run_one(env, fm, state_spec, target_set, ep_idx, Kp, Kd, use_nnls,
            use_ramp, action_adapter, action_dim):
    target_pos = target_set.target_pos[ep_idx]
    env.reset()
    mj.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)

    # capture initial tip + Jacobian + moment_arm at neutral pose
    init_tip = np.asarray(extract_state(env).tip_pos)
    jacobian = tip_jacobian_dense(env)
    moment_arm = actuator_moment_dense(env)

    controller = EndpointFB_Variant(
        action_dim=action_dim, init_tip=init_tip, target_pos=target_pos,
        jacobian=jacobian, moment_arm=moment_arm,
        Kp=Kp, Kd=Kd, action_scale=ACTION_SCALE,
        use_nnls=use_nnls, use_ramp=use_ramp, T_ramp=T_RAMP,
    )
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
        controller_name="endpoint_feedback_variant",
        controller_seed=0, sdn_seed=0, obs_noise_seed=0,
        config_hash="diag",
        meta={"nnls": str(use_nnls), "ramp": str(use_ramp)},
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
    act = np.asarray(result.log.true_act, dtype=np.float64)
    return {
        "min_tip": float(err.min()),
        "t_min": int(err.argmin()),
        "final_tip": float(err[-1]),
        "init_d": float(err[0]),
        "mean_act": float(act.mean()),
        "max_act": float(act.max()),
        "sat95_frac": float((act >= 0.95).mean()),
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

        CONDITIONS = [
            ("baseline",     False, False),
            ("B (ramp)",     False, True),
            ("A (nnls)",     True,  False),
            ("A+B",          True,  True),
        ]

        results: dict = {}
        for Kp in KP_GRID:
            Kd = Kp * KD_RATIO
            for cond_name, use_nnls, use_ramp in CONDITIONS:
                key = (Kp, cond_name)
                results[key] = {}
                for ep in EP_IDS:
                    r = run_one(
                        env, fm, state_spec, target_set, ep, Kp, Kd,
                        use_nnls, use_ramp, action_adapter, action_dim,
                    )
                    results[key][ep] = r

        # ---- print table ----
        for Kp in KP_GRID:
            print(f"\n\n========== Kp={Kp}, Kd={Kp*KD_RATIO}, "
                  f"action_scale={ACTION_SCALE}, T_ramp={T_RAMP} ==========")
            for ep in EP_IDS:
                init_d = results[(Kp, "baseline")][ep]["init_d"]
                print(f"\n  ep={ep}  init→target = {init_d:.4f} m")
                hdr = (f"  {'cond':<12} | {'min_tip':>8} {'t_min':>5} "
                       f"{'final':>8} | {'mean_act':>9} {'max_act':>8} "
                       f"{'sat95%':>7} | {'gain vs baseline':>17}")
                print(hdr)
                base = results[(Kp, "baseline")][ep]["min_tip"]
                for cond_name, _, _ in CONDITIONS:
                    r = results[(Kp, cond_name)][ep]
                    gain = 100.0 * (base - r["min_tip"]) / max(base, 1e-9)
                    print(
                        f"  {cond_name:<12} | "
                        f"{r['min_tip']:>8.4f} {r['t_min']:>5} "
                        f"{r['final_tip']:>8.4f} | "
                        f"{r['mean_act']:>9.3f} {r['max_act']:>8.3f} "
                        f"{100*r['sat95_frac']:>6.2f}% | "
                        f"{gain:>+16.1f}%"
                    )

        # summary: best variant per (Kp, ep)
        print("\n\n=== best variant per (Kp, ep), and overall ===")
        print(f"{'Kp':>5} {'ep':>3} {'best':<14} {'min_tip':>10} "
              f"{'baseline':>10} {'gain':>9}")
        all_results = []
        for Kp in KP_GRID:
            for ep in EP_IDS:
                base = results[(Kp, "baseline")][ep]["min_tip"]
                mins = [(c, results[(Kp, c)][ep]["min_tip"])
                        for c, _, _ in CONDITIONS]
                bn, bv = min(mins, key=lambda x: x[1])
                gain = 100.0 * (base - bv) / max(base, 1e-9)
                all_results.append((Kp, ep, bn, bv, base, gain))
                print(f"{Kp:>5.1f} {ep:>3} {bn:<14} {bv:>10.4f} "
                      f"{base:>10.4f} {gain:>+8.1f}%")
    finally:
        env.close()


if __name__ == "__main__":
    main()
