"""For each diagnostic episode, check whether the IK-solved target_qpos
sits AT the joint range boundary (= the controller will push that joint
against the wall and the muscle activations needed will saturate).

For each joint we report:
  qpos_init       neutral pose
  qpos_target     IK solution
  jnt_range       [lo, hi] from the model
  ratio           (qpos_target - lo) / (hi - lo)   ∈ [0, 1]
  margin_lo       qpos_target - lo
  margin_hi       hi - qpos_target
  saturated       True iff min(margin_lo, margin_hi) / (hi-lo) < 5%
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco as mj  # noqa: E402

from myoarm_fse.envs.factory import make_env  # noqa: E402
from myoarm_fse.envs.ik import solve_ik  # noqa: E402
from myoarm_fse.envs.targets import TargetSet  # noqa: E402

TARGET_NPZ = REPO / "runs/targets_reachable/2026-05-21T09-13-52Z/reachable_train.npz"
EPS = [0, 1, 2, 5, 10]
SATURATION_FRAC = 0.05  # within 5% of either boundary
RAD2DEG = 180.0 / np.pi


def main() -> None:
    ts = TargetSet.load(str(TARGET_NPZ))
    targets = ts.target_pos
    env = make_env("myoArmReachFixed-v0")
    try:
        uw = env.unwrapped
        m = uw.mj_model
        env.reset()
        mj.mj_forward(m, uw.mj_data)
        qpos_init = uw.mj_data.qpos.copy()
        jnt_range = m.jnt_range.copy()  # (njnt, 2)
        jnt_limited = m.jnt_limited.astype(bool)
        njnt = m.njnt
        jnt_names = [m.joint(j).name for j in range(njnt)]
        jnt_qposadr = m.jnt_qposadr.copy()

        print(f"njnt={njnt}  nv={m.nv}  nq={m.nq}")
        print(f"jnt_limited count: {int(jnt_limited.sum())}/{njnt}")

        # for each diag ep, solve IK and inspect target_qpos
        for ep in EPS:
            env.reset()
            mj.mj_forward(m, uw.mj_data)
            qpos_sol, info = solve_ik(
                env, targets[ep], max_iter=200, tol=0.01, damping=0.1,
            )
            print(f"\n========== ep={ep}  target={targets[ep]}  "
                  f"IK_err={info['final_error']:.4f}  iter={info['n_iter']} ==========")
            print(f"{'jnt_name':<22} {'lim':>4} {'init°':>8} {'target°':>9} "
                  f"{'lo°':>8} {'hi°':>8} {'ratio':>6} {'margin_lo°':>10} "
                  f"{'margin_hi°':>10} {'sat'}")

            n_sat = 0
            saturated_joints: list[str] = []
            for j in range(njnt):
                if not jnt_limited[j]:
                    continue
                adr = int(jnt_qposadr[j])
                lo, hi = jnt_range[j]
                qi = qpos_init[adr]
                qt = qpos_sol[adr]
                rng = hi - lo
                ratio = (qt - lo) / rng if rng > 0 else 0.5
                margin_lo = qt - lo
                margin_hi = hi - qt
                sat = (
                    (min(margin_lo, margin_hi) / rng) < SATURATION_FRAC
                    if rng > 0 else False
                )
                if sat:
                    n_sat += 1
                    saturated_joints.append(jnt_names[j])
                print(
                    f"{jnt_names[j]:<22} {'lim':>4} "
                    f"{qi * RAD2DEG:>8.2f} {qt * RAD2DEG:>9.2f} "
                    f"{lo * RAD2DEG:>8.2f} {hi * RAD2DEG:>8.2f} "
                    f"{ratio:>6.3f} {margin_lo * RAD2DEG:>10.2f} "
                    f"{margin_hi * RAD2DEG:>10.2f} "
                    f"{'*** SAT' if sat else ''}"
                )
            print(f"  -> {n_sat} joint(s) within {SATURATION_FRAC*100:.0f}% "
                  f"of a limit: {saturated_joints}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
