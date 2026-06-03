"""γ: distance-adaptive T_ramp via Fitts' law.

T_ramp = T_min + α · log₂(2 · init_d / W)

Stage 1: small sweep over (T_min, α) on 3 representative eps to pick a
         setting that doesn't over-/under-shoot short / long targets.
Stage 2: 200 ep × K ∈ {0.0, 0.5, 1.0} K-sweep with the best (T_min, α).

W is fixed at 0.05 m (the 5-cm success threshold).
"""

from __future__ import annotations

import json
import math
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
OUT = REPO / "runs/diag/step1b_fitts"

KP = 30.0
KD = KP * 0.1
ACTION_SCALE = 5.0
W = 0.05  # success threshold for Fitts

# Stage 1 sweep
SWEEP_EP_IDS = [0, 1, 10]
TMIN_GRID = [50, 100, 150, 200]
ALPHA_GRID = [50, 100, 150, 200]

# Stage 2 K-sweep
N_EP = 200
K_GRID = [0.0, 0.5, 1.0]


def fitts_T_ramp(init_d: float, T_min: float, alpha: float, W: float = W) -> int:
    """Return T_ramp >= 1 from Fitts' law (clamped)."""
    arg = max(2.0 * init_d / max(W, 1e-9), 1.0)  # log₂ floor at 0
    T = T_min + alpha * math.log2(arg)
    return int(max(1, round(T)))


class APlusBAdaptive:
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


def run_one(env, fm, state_spec, target_set, ep, K, T_min, alpha,
            action_adapter, action_dim):
    target_pos = target_set.target_pos[ep]
    env.reset()
    mj.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
    init_tip = np.asarray(extract_state(env).tip_pos)
    init_d = float(np.linalg.norm(init_tip - np.asarray(target_pos)))
    T_ramp = fitts_T_ramp(init_d, T_min, alpha)
    jacobian = tip_jacobian_dense(env)
    moment_arm = actuator_moment_dense(env)
    controller = APlusBAdaptive(
        action_dim=action_dim, init_tip=init_tip,
        target_pos=target_pos, jacobian=jacobian, moment_arm=moment_arm,
        Kp=KP, Kd=KD, action_scale=ACTION_SCALE, T_ramp=T_ramp,
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
        controller_name="a_plus_b_fitts",
        controller_seed=0, sdn_seed=0, obs_noise_seed=0,
        config_hash="step1b",
        meta={"T_min": str(T_min), "alpha": str(alpha), "K": str(K)},
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
        "ep": ep, "T_ramp_used": T_ramp,
        "init_d": init_d,
        "target_z": float(target_pos[2]),
        "min_tip": float(err.min()),
        "final_tip": float(err[-1]),
        "t_min": int(err.argmin()),
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

        # ===== Stage 1: small sweep on 3 eps with K=1 oracle =====
        print("\n========== Stage 1: (T_min, alpha) sweep on 3 eps "
              "(K=1 oracle) ==========")
        stage1: list[dict] = []
        for T_min in TMIN_GRID:
            for alpha in ALPHA_GRID:
                per_ep = {}
                for ep in SWEEP_EP_IDS:
                    r = run_one(env, fm, state_spec, target_set, ep, 1.0,
                                T_min, alpha, action_adapter, action_dim)
                    per_ep[ep] = r
                mins = [per_ep[ep]["min_tip"] for ep in SWEEP_EP_IDS]
                T_used = [per_ep[ep]["T_ramp_used"] for ep in SWEEP_EP_IDS]
                stage1.append({
                    "T_min": T_min, "alpha": alpha,
                    "mean_min_tip": float(np.mean(mins)),
                    "max_min_tip": float(np.max(mins)),
                    "per_ep": per_ep,
                    "T_ramp_used": T_used,
                })
                print(f"  T_min={T_min:>4} α={alpha:>4} | "
                      f"T_ramp used = {T_used} | "
                      f"min_tip = {mins[0]:.4f} {mins[1]:.4f} {mins[2]:.4f} | "
                      f"mean = {np.mean(mins):.4f}")

        stage1_ranked = sorted(stage1, key=lambda r: r["mean_min_tip"])
        best = stage1_ranked[0]
        print("\n=== top 5 ===")
        for r in stage1_ranked[:5]:
            print(f"  T_min={r['T_min']:>4} α={r['alpha']:>4}  "
                  f"mean={r['mean_min_tip']:.4f}  max={r['max_min_tip']:.4f} "
                  f"T_ramp={r['T_ramp_used']}")
        print(f"\n→ Stage 1 best: T_min={best['T_min']}, alpha={best['alpha']}, "
              f"mean_min_tip={best['mean_min_tip']:.4f}")

        # ===== Stage 2: 200-ep K-sweep at best (T_min, alpha) =====
        T_min_best = best["T_min"]; alpha_best = best["alpha"]
        print(f"\n========== Stage 2: 200-ep K-sweep at "
              f"(T_min={T_min_best}, alpha={alpha_best}) ==========")
        per_K: dict[float, list[dict]] = {}
        for K in K_GRID:
            print(f"\n--- K = {K} ---")
            rows = []
            for ep in range(N_EP):
                r = run_one(env, fm, state_spec, target_set, ep, K,
                            T_min_best, alpha_best,
                            action_adapter, action_dim)
                rows.append(r)
                if (ep + 1) % 50 == 0:
                    mt = np.array([x["min_tip"] for x in rows])
                    print(f"  ep={ep+1:>3}/{N_EP}  "
                          f"min_tip mean={mt.mean():.4f}  "
                          f"S005={100*np.mean(mt < 0.05):.1f}%  "
                          f"S010={100*np.mean(mt < 0.10):.1f}%")
            per_K[K] = rows
            mt = np.array([x["min_tip"] for x in rows])
            print(f"  K={K} done: mean={mt.mean():.4f} "
                  f"S005={100*np.mean(mt < 0.05):.1f}% "
                  f"S010={100*np.mean(mt < 0.10):.1f}% "
                  f"S015={100*np.mean(mt < 0.15):.1f}%")

        # ===== summary =====
        print("\n\n=== K-sweep (200 ep, A+B+Fitts) ===")
        print(f"{'K':>5} | {'min_tip mean':>13} {'median':>8} "
              f"{'S005%':>7} {'S010%':>7} {'S015%':>7} {'final mean':>11}")
        for K in K_GRID:
            mt = np.array([x["min_tip"] for x in per_K[K]])
            ft = np.array([x["final_tip"] for x in per_K[K]])
            print(
                f"{K:>5.2f} | {mt.mean():>13.4f} {np.median(mt):>8.4f} "
                f"{100*np.mean(mt < 0.05):>6.2f}% "
                f"{100*np.mean(mt < 0.10):>6.2f}% "
                f"{100*np.mean(mt < 0.15):>6.2f}% "
                f"{ft.mean():>11.4f}"
            )

        # below-shoulder subset
        SHOULDER_Z = 1.393
        print(f"\n=== K-sweep restricted to z < {SHOULDER_Z} ===")
        print(f"{'K':>5} | {'n':>4} | {'min_tip mean':>13} {'median':>8} "
              f"{'S005%':>7} {'S010%':>7} {'S015%':>7}")
        for K in K_GRID:
            sub = [r for r in per_K[K] if r["target_z"] < SHOULDER_Z]
            mt = np.array([r["min_tip"] for r in sub])
            print(
                f"{K:>5.2f} | {len(sub):>4d} | {mt.mean():>13.4f} "
                f"{np.median(mt):>8.4f} "
                f"{100*np.mean(mt < 0.05):>6.2f}% "
                f"{100*np.mean(mt < 0.10):>6.2f}% "
                f"{100*np.mean(mt < 0.15):>6.2f}%"
            )

        with open(OUT / "results.json", "w") as f:
            json.dump({
                "stage1_sweep": stage1,
                "stage1_best": {"T_min": T_min_best, "alpha": alpha_best},
                "stage2_per_K": {str(K): per_K[K] for K in K_GRID},
            }, f, indent=2)
        print(f"\nsaved: {OUT / 'results.json'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
