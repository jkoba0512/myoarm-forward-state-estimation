"""Step 1: T_ramp × Kp grid sweep for A+B controller (NNLS + ramp).

Sweep over T_ramp ∈ {100, 200, 300, 500, 800} × Kp ∈ {5, 10, 20, 30, 50}
on 3 representative episodes [0, 1, 10]. K=1 oracle, noise=none, d=0.
Selection criterion: minimise mean(min_tip) over the 3 eps.

Output:
  runs/diag/step1_sweep/sweep_results.json
  stdout: ranked table.
"""

from __future__ import annotations

import json
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
OUT = REPO / "runs/diag/step1_sweep"

EP_IDS = [0, 1, 10]
T_RAMP_GRID = [100, 200, 300, 500, 800]
KP_GRID = [5.0, 10.0, 20.0, 30.0, 50.0]
KD_RATIO = 0.1
ACTION_SCALE = 5.0


class APlusB:
    def __init__(self, *, action_dim, init_tip, target_pos, jacobian,
                 moment_arm, Kp, Kd, action_scale, T_ramp):
        self._action_dim = int(action_dim)
        self._init_tip = np.asarray(init_tip, dtype=np.float32)
        self._final_target = np.asarray(target_pos, dtype=np.float32)
        self._jacobian = np.asarray(jacobian, dtype=np.float32)
        self._mt = np.asarray(moment_arm, dtype=np.float32).T.astype(
            np.float64,
        )
        self._Kp = float(Kp); self._Kd = float(Kd)
        self._action_scale = float(action_scale)
        self._T_ramp = int(T_ramp)
        self._step = 0

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, *, seed=None) -> None:
        del seed
        self._step = 0

    def act(self, observation: MyoArmState) -> np.ndarray:
        s = min(self._step / max(self._T_ramp, 1), 1.0)
        cur_target = self._init_tip + s * (self._final_target - self._init_tip)
        self._step += 1
        tip_est = np.asarray(observation.tip_pos, dtype=np.float32)
        qvel = np.asarray(observation.qvel, dtype=np.float32)
        e_tip = cur_target - tip_est
        v_tip = self._jacobian @ qvel
        u_tip = self._Kp * e_tip - self._Kd * v_tip
        u_joint = self._jacobian.T @ u_tip
        activation, _ = nnls(self._mt, u_joint.astype(np.float64))
        return np.clip(
            self._action_scale * activation, 0.0, 1.0,
        ).astype(np.float32, copy=False)


def run_one(env, fm, state_spec, target_set, ep_idx, T_ramp, Kp, Kd,
            action_adapter, action_dim):
    target_pos = target_set.target_pos[ep_idx]
    env.reset()
    mj.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
    init_tip = np.asarray(extract_state(env).tip_pos)
    jacobian = tip_jacobian_dense(env)
    moment_arm = actuator_moment_dense(env)

    controller = APlusB(
        action_dim=action_dim, init_tip=init_tip,
        target_pos=target_pos, jacobian=jacobian, moment_arm=moment_arm,
        Kp=Kp, Kd=Kd, action_scale=ACTION_SCALE, T_ramp=T_ramp,
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
        controller_name="a_plus_b",
        controller_seed=0, sdn_seed=0, obs_noise_seed=0,
        config_hash="step1",
        meta={"T_ramp": str(T_ramp), "Kp": str(Kp)},
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
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(CONFIG))
    fm, model_cfg, _ = load_model(cfg["forward_model"])
    fm.eval()
    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    env = make_env(cfg["env_id"], horizon=int(cfg["horizon"]))
    try:
        target_set = TargetSet.load(cfg["target_set"])
        action_dim = detect_action_dim(env)
        action_adapter = ActionAdapter(action_dim=action_dim)

        all_results: list[dict] = []
        n_total = len(T_RAMP_GRID) * len(KP_GRID) * len(EP_IDS)
        i = 0
        for T_ramp in T_RAMP_GRID:
            for Kp in KP_GRID:
                Kd = Kp * KD_RATIO
                per_ep: dict[int, dict] = {}
                for ep in EP_IDS:
                    i += 1
                    r = run_one(env, fm, state_spec, target_set, ep,
                                T_ramp, Kp, Kd, action_adapter, action_dim)
                    per_ep[ep] = r
                    print(f"  [{i}/{n_total}] T_ramp={T_ramp} Kp={Kp} "
                          f"ep={ep}: min_tip={r['min_tip']:.4f} "
                          f"final_tip={r['final_tip']:.4f}")
                mins = [per_ep[ep]["min_tip"] for ep in EP_IDS]
                all_results.append({
                    "T_ramp": int(T_ramp),
                    "Kp": float(Kp),
                    "Kd": float(Kd),
                    "per_ep": per_ep,
                    "mean_min_tip": float(np.mean(mins)),
                    "max_min_tip": float(np.max(mins)),
                    "min_min_tip": float(np.min(mins)),
                })

        out_path = OUT / "sweep_results.json"
        with open(out_path, "w") as f:
            json.dump({
                "grid": {"T_ramp": T_RAMP_GRID, "Kp": KP_GRID,
                         "EP_IDS": EP_IDS, "Kd_ratio": KD_RATIO,
                         "action_scale": ACTION_SCALE},
                "results": all_results,
            }, f, indent=2)

        print(f"\nsaved: {out_path}")

        # ---- ranking ----
        ranked = sorted(all_results, key=lambda r: r["mean_min_tip"])
        print("\n=== top 10 configurations by mean_min_tip ===")
        print(f"{'rank':>4} {'T_ramp':>7} {'Kp':>6} | "
              f"{'mean':>8} {'max':>8} {'min':>8} | "
              f"{'ep0':>8} {'ep1':>8} {'ep10':>8}")
        for k, r in enumerate(ranked[:10], 1):
            ep_mins = [r["per_ep"][ep]["min_tip"] for ep in EP_IDS]
            print(
                f"{k:>4} {r['T_ramp']:>7} {r['Kp']:>6.1f} | "
                f"{r['mean_min_tip']:>8.4f} {r['max_min_tip']:>8.4f} "
                f"{r['min_min_tip']:>8.4f} | "
                f"{ep_mins[0]:>8.4f} {ep_mins[1]:>8.4f} {ep_mins[2]:>8.4f}"
            )
        print("\n=== bottom 5 ===")
        for k, r in enumerate(ranked[-5:], len(ranked) - 4):
            ep_mins = [r["per_ep"][ep]["min_tip"] for ep in EP_IDS]
            print(
                f"{k:>4} {r['T_ramp']:>7} {r['Kp']:>6.1f} | "
                f"{r['mean_min_tip']:>8.4f} {r['max_min_tip']:>8.4f} "
                f"{r['min_min_tip']:>8.4f} | "
                f"{ep_mins[0]:>8.4f} {ep_mins[1]:>8.4f} {ep_mins[2]:>8.4f}"
            )

        best = ranked[0]
        print(f"\n→ best: T_ramp={best['T_ramp']}, Kp={best['Kp']}, "
              f"mean_min_tip={best['mean_min_tip']:.4f}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
