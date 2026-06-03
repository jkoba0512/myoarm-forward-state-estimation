"""Quantify the physical reach radius of myoArm and compare to target_set.

For each target in the set we compute:
  - shoulder z (most-proximal body in the arm chain)
  - shoulder-to-target distance
  - shoulder-to-tip-after-IK distance (= actual reach the IK solution
    realises in kinematic space)
  - max possible reach = max over the target set of ‖tip_after_IK − shoulder‖
    after solving IK against the most-distant feasible point
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco as mj  # noqa: E402

from myoarm_fse.envs.factory import make_env  # noqa: E402
from myoarm_fse.envs.extractors import extract_state  # noqa: E402
from myoarm_fse.envs.ik import solve_ik  # noqa: E402
from myoarm_fse.envs.targets import TargetSet  # noqa: E402

TARGET_NPZ = REPO / "runs/targets_reachable/2026-05-21T09-13-52Z/reachable_train.npz"


def main() -> None:
    ts = TargetSet.load(str(TARGET_NPZ))
    targets = ts.target_pos
    print(f"target_set: n={ts.n}")

    env = make_env("myoArmReachFixed-v0")
    try:
        uw = env.unwrapped
        env.reset()
        mj.mj_forward(uw.mj_model, uw.mj_data)

        # list all body names + positions so we can spot the shoulder
        print("\n=== body chain (neutral pose) ===")
        for i in range(uw.mj_model.nbody):
            name = uw.mj_model.body(i).name
            pos = uw.mj_data.xpos[i]
            if any(k in name.lower() for k in ("clav", "scap", "humer",
                                                "rad", "uln", "hand",
                                                "thorax", "torso",
                                                "world", "root")):
                print(f"  body[{i:>2}] {name:<25s} xpos={pos}")

        # heuristically pick the shoulder as humerus parent root, or fall
        # back to the body whose name contains 'humerus'
        shoulder_idx = None
        for i in range(uw.mj_model.nbody):
            n = uw.mj_model.body(i).name.lower()
            if "humerus" in n or "humer" in n:
                shoulder_idx = i; break
        if shoulder_idx is None:
            # fall back to second body
            shoulder_idx = 1
        shoulder = uw.mj_data.xpos[shoulder_idx].copy()
        print(f"\nshoulder body: idx={shoulder_idx} "
              f"name={uw.mj_model.body(shoulder_idx).name} pos={shoulder}")

        # init tip
        state = extract_state(env)
        init_tip = np.asarray(state.tip_pos)
        arm_length_extended = float(np.linalg.norm(init_tip - shoulder))
        print(f"init tip: {init_tip}")
        print(f"  ‖init_tip − shoulder‖ = {arm_length_extended:.3f} m  "
              f"(arm length in neutral pose, used as a lower bound on "
              f"max reach)")

        # max reach (assume rigid bodies, so the actual maximum reach is
        # ‖humerus−shoulder‖ + ‖radius−humerus‖ + ‖hand−radius‖ + ...
        # we approximate it by following the kinematic chain to the tip
        # body. Find tip body by name pattern.
        tip_body_idx = None
        for i in range(uw.mj_model.nbody):
            n = uw.mj_model.body(i).name.lower()
            if "iftip" in n or "tip" in n or "distal" in n:
                tip_body_idx = i
        print(f"tip body idx={tip_body_idx} "
              f"name={uw.mj_model.body(tip_body_idx).name if tip_body_idx else 'N/A'}")

        # walk from tip up to shoulder summing segment lengths
        max_reach = None
        if tip_body_idx is not None:
            chain = []
            cur = tip_body_idx
            while cur != shoulder_idx and cur > 0:
                chain.append(cur); cur = uw.mj_model.body_parentid[cur]
            chain.append(shoulder_idx)
            chain.reverse()
            seg_total = 0.0
            for a, b in zip(chain[:-1], chain[1:]):
                d = float(np.linalg.norm(
                    uw.mj_data.xpos[b] - uw.mj_data.xpos[a]
                ))
                seg_total += d
                print(f"    segment {uw.mj_model.body(a).name} → "
                      f"{uw.mj_model.body(b).name}: {d:.3f} m")
            max_reach = seg_total
            print(f"  Σ segment lengths (shoulder → tip in neutral pose): "
                  f"{seg_total:.3f} m")

        # distance from shoulder to each target
        d_to_target = np.linalg.norm(targets - shoulder, axis=1)
        print(f"\n=== shoulder-to-target distance over {ts.n} targets ===")
        print(f"  min  = {d_to_target.min():.3f} m")
        print(f"  p25  = {np.percentile(d_to_target, 25):.3f} m")
        print(f"  med  = {np.median(d_to_target):.3f} m")
        print(f"  p75  = {np.percentile(d_to_target, 75):.3f} m")
        print(f"  p95  = {np.percentile(d_to_target, 95):.3f} m")
        print(f"  max  = {d_to_target.max():.3f} m  (ep="
              f"{int(np.argmax(d_to_target))})")

        # how many targets are beyond the rigid max-reach?
        if max_reach is not None:
            n_beyond = int(np.sum(d_to_target > max_reach))
            print(f"  targets beyond rigid max reach "
                  f"({max_reach:.3f} m): {n_beyond}/{ts.n} "
                  f"({100*n_beyond/ts.n:.1f}%)")

        # for diagnostic eps, IK-tip vs shoulder distance: this is the
        # "kinematic reach the IK solution actually realises"
        print("\n=== diagnostic eps: target dist vs IK-realised reach ===")
        diag_eps = [0, 1, 2, 5, 10]
        for ep in diag_eps:
            env.reset()
            mj.mj_forward(uw.mj_model, uw.mj_data)
            qpos_sol, info = solve_ik(
                env, targets[ep], max_iter=200, tol=0.01, damping=0.1,
            )
            uw.mj_data.qpos[:qpos_sol.shape[0]] = qpos_sol
            mj.mj_forward(uw.mj_model, uw.mj_data)
            tip = extract_state(env).tip_pos
            dt = float(np.linalg.norm(np.asarray(targets[ep]) - shoulder))
            dik = float(np.linalg.norm(np.asarray(tip) - shoulder))
            ik_err = float(info["final_error"])
            print(f"  ep={ep:>2}  target_z={targets[ep][2]:.3f}  "
                  f"‖tgt-shoulder‖={dt:.3f}  ‖ik_tip-shoulder‖={dik:.3f}  "
                  f"IK_err={ik_err:.4f}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
