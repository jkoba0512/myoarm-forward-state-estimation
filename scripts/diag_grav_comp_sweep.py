"""Diagnose: does joint-PD reach IK target_qpos once we add a baseline /
gravity-compensation activation? K=1 oracle, noise=none, delay=0.

Controllers compared (all wrap the same PD law + ReLU/clip, only the
baseline term differs):
  - none           : current production behaviour
  - flat(b)        : add scalar `b` to every muscle activation, then clip
  - qfrc_bias(g)   : lstsq(moment_arm.T, qfrc_bias) → ReLU → scale by g
                     (true gravity + Coriolis compensation, projected
                      into the non-negative orthant)

Compares against the K=1 baseline (min_tip 0.46 – 0.82 m). If any comp
brings min_tip down meaningfully on the same eps, the controller is
salvageable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from myoarm_fse.controllers.base import Controller  # noqa: E402
from myoarm_fse.data.rollout import EpisodeSpec  # noqa: E402
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim  # noqa: E402
from myoarm_fse.envs.factory import make_env  # noqa: E402
from myoarm_fse.envs.ik import actuator_moment_dense, solve_ik  # noqa: E402
from myoarm_fse.envs.state import MyoArmState, StateSpec  # noqa: E402
from myoarm_fse.envs.targets import TargetSet  # noqa: E402
from myoarm_fse.estimators import FixedGainKalmanEstimator  # noqa: E402
from myoarm_fse.estimators.fixed_kalman import _flatten_log_states  # noqa: E402
from myoarm_fse.evaluation import run_closed_loop_episode  # noqa: E402
from myoarm_fse.models import load_model  # noqa: E402

CONFIG = REPO / "configs/closed_loop/oracle_k_sweep_r3.yaml"
EP_IDS = [0, 1, 2, 5, 10]

# (mode, param)
COMPS = [
    ("none",       0.0),
    ("flat",       0.05),
    ("flat",       0.10),
    ("flat",       0.20),
    ("qfrc_bias",  1.0),
    ("qfrc_bias",  2.0),
]


class JointPDPlus:
    """Joint-PD with optional baseline / gravity-compensation activation.

    The env reference is used only for the qfrc_bias readout each step;
    PD math is identical to JointSpacePDController.
    """

    def __init__(self, *, env, action_dim, target_qpos, moment_arm,
                 Kp, Kd, action_scale, comp_mode, comp_param):
        self._env = env
        self._action_dim = int(action_dim)
        self._target_qpos = np.asarray(target_qpos, dtype=np.float32)
        self._moment_arm = np.asarray(moment_arm, dtype=np.float32)
        self._Kp = float(Kp); self._Kd = float(Kd)
        self._action_scale = float(action_scale)
        self._comp_mode = comp_mode
        self._comp_param = float(comp_param)
        self._mt = self._moment_arm.T.astype(np.float64)  # (nv, nu)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, *, seed=None) -> None:
        pass

    def act(self, observation: MyoArmState) -> np.ndarray:
        qpos = np.asarray(observation.qpos, dtype=np.float32)
        qvel = np.asarray(observation.qvel, dtype=np.float32)
        u_joint = self._Kp * (self._target_qpos - qpos) - self._Kd * qvel
        drive = self._moment_arm @ u_joint
        pd_act = np.clip(
            self._action_scale * np.maximum(drive, 0.0), 0.0, 1.0,
        )

        if self._comp_mode == "none":
            base = 0.0
        elif self._comp_mode == "flat":
            base = self._comp_param
        elif self._comp_mode == "qfrc_bias":
            mj_data = self._env.unwrapped.mj_data
            qfrc_bias = np.asarray(mj_data.qfrc_bias, dtype=np.float64).copy()
            # Want moment_arm.T @ act_baseline = qfrc_bias
            #   (joint torque produced by muscle activation = passive bias)
            sol, *_ = np.linalg.lstsq(self._mt, qfrc_bias, rcond=None)
            base = np.maximum(self._comp_param * sol, 0.0)
        else:
            raise ValueError(f"unknown comp_mode: {self._comp_mode}")

        out = np.clip(pd_act + base, 0.0, 1.0).astype(np.float32)
        if out.shape[0] != self._action_dim:
            out = np.resize(out, self._action_dim).astype(np.float32)
        return out


def run_one(env, fm, state_spec, target_set, ep_idx, K, comp, controller_spec_base, action_adapter, action_dim):
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
    controller = JointPDPlus(
        env=env, action_dim=action_dim,
        target_qpos=target_qpos, moment_arm=moment_arm,
        Kp=float(controller_spec_base["Kp"]),
        Kd=float(controller_spec_base["Kd"]),
        action_scale=float(controller_spec_base["action_scale"]),
        comp_mode=comp[0], comp_param=comp[1],
    )
    estimator = FixedGainKalmanEstimator(
        forward_model=fm, gain=float(K),
        state_spec=state_spec, delay_steps=0,
    )
    spec = EpisodeSpec(
        episode_id=ep_idx,
        target_id=str(int(target_set.seeds[ep_idx])),
        target_split=target_set.split,
        target_seed=int(target_set.seeds[ep_idx]),
        controller_name="joint_pd_plus",
        controller_seed=0, sdn_seed=0, obs_noise_seed=0,
        config_hash="diag",
        meta={"estimator_name": f"K={K}", "comp": f"{comp[0]}/{comp[1]}"},
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
        "ik_err": ik_info["final_error"],
        "min_tip": min_tip,
        "final_tip": final_tip,
        "t_min": t_min,
        "qpos_resid_min": qpos_resid_min,
        "qpos_resid_final": qpos_resid_final,
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

        # comps × ep grid, K=1 oracle
        print("\n=== min_tip [m]  (K=1 oracle, noise=none, d=0) ===")
        header = "ep |" + "".join(f" {c[0]:>9}({c[1]:>4.2f}) " for c in COMPS)
        print(header)
        for ep in EP_IDS:
            row = f"{ep:>2} |"
            for comp in COMPS:
                r = run_one(env, fm, state_spec, target_set, ep, 1.0, comp,
                            controller_spec_base, action_adapter, action_dim)
                row += f"     {r['min_tip']:>6.3f}      "
            print(row)

        print("\n=== final_tip [m] (same grid) ===")
        print(header)
        for ep in EP_IDS:
            row = f"{ep:>2} |"
            for comp in COMPS:
                r = run_one(env, fm, state_spec, target_set, ep, 1.0, comp,
                            controller_spec_base, action_adapter, action_dim)
                row += f"     {r['final_tip']:>6.3f}      "
            print(row)

        print("\n=== qpos_resid_min [rad-L2] (joint-space follow quality) ===")
        print(header)
        for ep in EP_IDS:
            row = f"{ep:>2} |"
            for comp in COMPS:
                r = run_one(env, fm, state_spec, target_set, ep, 1.0, comp,
                            controller_spec_base, action_adapter, action_dim)
                row += f"     {r['qpos_resid_min']:>6.3f}      "
            print(row)

    finally:
        env.close()


if __name__ == "__main__":
    main()
