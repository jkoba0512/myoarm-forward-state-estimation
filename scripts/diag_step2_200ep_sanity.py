"""Step 2: 200-ep sanity check at the best (T_ramp*, Kp*) selected in
Step 1. Noise=none, delay=0, K=1.0 oracle. Reports success rate
@ {5, 10, 15} cm and the min_tip distribution.
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
OUT = REPO / "runs/diag/step2_200ep"

# best from Step 1
T_RAMP = 300
KP = 30.0
KD = KP * 0.1
ACTION_SCALE = 5.0
N_EP = 200


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

        print(f"settings: T_ramp={T_RAMP} Kp={KP} Kd={KD} "
              f"action_scale={ACTION_SCALE} N_EP={N_EP}")
        print(f"target_set: n={target_set.n} (using ep 0..{N_EP-1})")

        rows: list[dict] = []
        for ep in range(N_EP):
            target_pos = target_set.target_pos[ep]
            env.reset()
            mj.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
            init_tip = np.asarray(extract_state(env).tip_pos)
            jacobian = tip_jacobian_dense(env)
            moment_arm = actuator_moment_dense(env)
            controller = APlusB(
                action_dim=action_dim, init_tip=init_tip,
                target_pos=target_pos, jacobian=jacobian,
                moment_arm=moment_arm,
                Kp=KP, Kd=KD, action_scale=ACTION_SCALE, T_ramp=T_RAMP,
            )
            controller.reset(seed=0)
            estimator = FixedGainKalmanEstimator(
                forward_model=fm, gain=1.0,
                state_spec=state_spec, delay_steps=0,
            )
            spec = EpisodeSpec(
                episode_id=ep,
                target_id=str(int(target_set.seeds[ep])),
                target_split=target_set.split,
                target_seed=int(target_set.seeds[ep]),
                controller_name="a_plus_b",
                controller_seed=0, sdn_seed=0, obs_noise_seed=0,
                config_hash="step2",
                meta={"T_ramp": str(T_RAMP), "Kp": str(KP)},
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
            rows.append({
                "ep": ep,
                "min_tip": float(err.min()),
                "final_tip": float(err[-1]),
                "t_min": int(err.argmin()),
                "init_d": float(err[0]),
                "target_z": float(target_pos[2]),
            })
            if (ep + 1) % 20 == 0:
                mt = np.array([r["min_tip"] for r in rows])
                print(f"  ep={ep+1:>3}/{N_EP}  "
                      f"running mean min_tip={mt.mean():.4f}  "
                      f"S005={100*np.mean(mt < 0.05):.1f}%  "
                      f"S010={100*np.mean(mt < 0.10):.1f}%  "
                      f"S015={100*np.mean(mt < 0.15):.1f}%")

        # ---- summary stats ----
        mt = np.array([r["min_tip"] for r in rows])
        ft = np.array([r["final_tip"] for r in rows])
        init_d = np.array([r["init_d"] for r in rows])
        tz = np.array([r["target_z"] for r in rows])

        print("\n=== 200-ep summary ===")
        print(f"min_tip:   mean={mt.mean():.4f}  median={np.median(mt):.4f}  "
              f"p25={np.percentile(mt, 25):.4f}  "
              f"p75={np.percentile(mt, 75):.4f}  "
              f"max={mt.max():.4f}  min={mt.min():.4f}")
        print(f"final_tip: mean={ft.mean():.4f}  median={np.median(ft):.4f}")
        print(f"init_d:    mean={init_d.mean():.4f}  "
              f"max={init_d.max():.4f}  min={init_d.min():.4f}")
        print(f"\nsuccess rate @ 5 cm:  {100*(mt < 0.05).mean():.1f}%  "
              f"({int((mt < 0.05).sum())}/{N_EP})")
        print(f"success rate @ 10 cm: {100*(mt < 0.10).mean():.1f}%  "
              f"({int((mt < 0.10).sum())}/{N_EP})")
        print(f"success rate @ 15 cm: {100*(mt < 0.15).mean():.1f}%  "
              f"({int((mt < 0.15).sum())}/{N_EP})")
        print(f"success rate @ 20 cm: {100*(mt < 0.20).mean():.1f}%  "
              f"({int((mt < 0.20).sum())}/{N_EP})")

        # correlation with target z (= gravity load)
        corr = float(np.corrcoef(tz, mt)[0, 1])
        print(f"\ntarget_z vs min_tip correlation: r = {corr:+.3f}")

        # save
        out_path = OUT / "results.json"
        with open(out_path, "w") as f:
            json.dump({
                "settings": {
                    "T_ramp": T_RAMP, "Kp": KP, "Kd": KD,
                    "action_scale": ACTION_SCALE, "N_EP": N_EP,
                },
                "rows": rows,
                "summary": {
                    "min_tip_mean": float(mt.mean()),
                    "min_tip_median": float(np.median(mt)),
                    "success_005": float((mt < 0.05).mean()),
                    "success_010": float((mt < 0.10).mean()),
                    "success_015": float((mt < 0.15).mean()),
                    "success_020": float((mt < 0.20).mean()),
                    "z_vs_min_tip_corr": corr,
                },
            }, f, indent=2)
        print(f"\nsaved: {out_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
