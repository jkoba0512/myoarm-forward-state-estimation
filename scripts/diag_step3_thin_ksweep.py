"""Step 3: thin K-sweep (K ∈ {0.0, 0.5, 1.0}) × 200 ep × 1 cell
with the A+B controller (T_ramp=300, Kp=30). noise=none, delay=0.

Question: did the joint-PD-era 'best K=0' result reverse after the
controller pivot? If K=0 still wins, the K-sweep finding was
controller-invariant (estimator-level phenomenon). If K=1 / K=0.5
wins, it was a joint-PD artefact and the new K-sweep is the
correct measurement.
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
OUT = REPO / "runs/diag/step3_thin_ksweep"

T_RAMP = 300
KP = 30.0
KD = KP * 0.1
ACTION_SCALE = 5.0
N_EP = 200
K_GRID = [0.0, 0.5, 1.0]


class APlusB:
    def __init__(self, *, action_dim, init_tip, target_pos, jacobian,
                 moment_arm, Kp, Kd, action_scale, T_ramp):
        self._action_dim = int(action_dim)
        self._init_tip = np.asarray(init_tip, dtype=np.float32)
        self._final_target = np.asarray(target_pos, dtype=np.float32)
        self._jacobian = np.asarray(jacobian, dtype=np.float32)
        self._mt = np.asarray(moment_arm, dtype=np.float32).T.astype(np.float64)
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


def run_episode(env, fm, state_spec, target_set, ep, K, action_adapter,
                action_dim):
    target_pos = target_set.target_pos[ep]
    env.reset()
    mj.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
    init_tip = np.asarray(extract_state(env).tip_pos)
    jacobian = tip_jacobian_dense(env)
    moment_arm = actuator_moment_dense(env)
    controller = APlusB(
        action_dim=action_dim, init_tip=init_tip,
        target_pos=target_pos, jacobian=jacobian, moment_arm=moment_arm,
        Kp=KP, Kd=KD, action_scale=ACTION_SCALE, T_ramp=T_RAMP,
    )
    controller.reset(seed=0)
    estimator = FixedGainKalmanEstimator(
        forward_model=fm, gain=float(K),
        state_spec=state_spec, delay_steps=0,
    )
    spec = EpisodeSpec(
        episode_id=ep,
        target_id=str(int(target_set.seeds[ep])),
        target_split=target_set.split,
        target_seed=int(target_set.seeds[ep]),
        controller_name="a_plus_b",
        controller_seed=0, sdn_seed=0, obs_noise_seed=0,
        config_hash="step3",
        meta={"K": str(K)},
    )
    result = run_closed_loop_episode(
        env, controller, estimator, target_pos,
        state_spec=state_spec, action_adapter=action_adapter,
        sdn=None, obs_noise=None, obs_delay=None,
        obs_compose="noisy_then_delayed",
        max_steps=600, spec=spec,
    )
    x_true = _flatten_log_states(result.log).astype(np.float64)
    layout = state_spec.layout()
    true_tip = x_true[:, layout["tip_pos"]]
    err = np.linalg.norm(true_tip - np.asarray(target_pos), axis=1)
    return {
        "ep": ep,
        "min_tip": float(err.min()),
        "final_tip": float(err[-1]),
        "init_d": float(err[0]),
        "target_z": float(target_pos[2]),
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

        per_K: dict[float, list[dict]] = {}
        for K in K_GRID:
            print(f"\n--- K = {K} ---")
            rows: list[dict] = []
            for ep in range(N_EP):
                r = run_episode(env, fm, state_spec, target_set, ep, K,
                                action_adapter, action_dim)
                rows.append(r)
                if (ep + 1) % 40 == 0:
                    mt = np.array([x["min_tip"] for x in rows])
                    print(f"  ep={ep+1:>3}/{N_EP}  "
                          f"min_tip mean={mt.mean():.4f}  "
                          f"S005={100*np.mean(mt < 0.05):.1f}%  "
                          f"S010={100*np.mean(mt < 0.10):.1f}%")
            per_K[K] = rows
            mt = np.array([x["min_tip"] for x in rows])
            print(f"  K={K} done: "
                  f"mean={mt.mean():.4f} median={np.median(mt):.4f} "
                  f"S005={100*np.mean(mt < 0.05):.1f}% "
                  f"S010={100*np.mean(mt < 0.10):.1f}% "
                  f"S015={100*np.mean(mt < 0.15):.1f}%")

        # ---- comparison table ----
        print("\n\n=== K-sweep comparison (200 ep, none/d=0, A+B controller) ===")
        print(f"{'K':>5} | {'min_tip mean':>13} {'median':>8} "
              f"{'S005%':>7} {'S010%':>7} {'S015%':>7} {'final mean':>11}")
        for K in K_GRID:
            mt = np.array([x["min_tip"] for x in per_K[K]])
            ft = np.array([x["final_tip"] for x in per_K[K]])
            print(
                f"{K:>5.2f} | {mt.mean():>13.4f} "
                f"{np.median(mt):>8.4f} "
                f"{100*np.mean(mt < 0.05):>6.2f}% "
                f"{100*np.mean(mt < 0.10):>6.2f}% "
                f"{100*np.mean(mt < 0.15):>6.2f}% "
                f"{ft.mean():>11.4f}"
            )

        # which K wins on what metric?
        print("\n=== best K per metric ===")
        for metric in ("min_tip_mean", "S005", "S010", "S015"):
            vals = []
            for K in K_GRID:
                mt = np.array([x["min_tip"] for x in per_K[K]])
                if metric == "min_tip_mean":
                    v = mt.mean(); better = "lower"
                elif metric == "S005":
                    v = np.mean(mt < 0.05); better = "higher"
                elif metric == "S010":
                    v = np.mean(mt < 0.10); better = "higher"
                elif metric == "S015":
                    v = np.mean(mt < 0.15); better = "higher"
                vals.append((K, float(v)))
            if better == "lower":
                bk, bv = min(vals, key=lambda x: x[1])
            else:
                bk, bv = max(vals, key=lambda x: x[1])
            print(f"  {metric:<16}: best K = {bk:.2f}  value = {bv:.4f}")

        # save
        out_path = OUT / "results.json"
        with open(out_path, "w") as f:
            json.dump({
                "settings": {
                    "T_ramp": T_RAMP, "Kp": KP, "Kd": KD,
                    "action_scale": ACTION_SCALE,
                    "N_EP": N_EP, "K_GRID": K_GRID,
                },
                "per_K": {str(K): per_K[K] for K in K_GRID},
            }, f, indent=2)
        print(f"\nsaved: {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
