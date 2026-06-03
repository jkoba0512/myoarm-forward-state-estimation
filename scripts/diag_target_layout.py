"""Visualize target_set vs myoArm body skeleton + target_reach_range bbox.

Outputs:
  runs/diag/target_layout/skeleton_targets.png   matplotlib 3D + 3 ortho views
  runs/diag/target_layout/mujoco_render.png      MuJoCo offscreen render of
                                                 neutral pose with target
                                                 markers overlaid in matplotlib

Highlights the diagnostic eps (0, 1, 2, 5, 10) so we can see whether the
"hard" eps (10) is geometrically out of reach.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

import mujoco as mj  # noqa: E402

from myoarm_fse.envs.factory import make_env  # noqa: E402
from myoarm_fse.envs.extractors import extract_state  # noqa: E402
from myoarm_fse.envs.ik import solve_ik  # noqa: E402
from myoarm_fse.envs.targets import TargetSet  # noqa: E402

TARGET_NPZ = REPO / "runs/targets_reachable/2026-05-21T09-13-52Z/reachable_train.npz"
ENV_ID = "myoArmReachFixed-v0"
HIGHLIGHT_EPS = [0, 1, 2, 5, 10]
OUT = REPO / "runs/diag/target_layout"


def _bbox_edges(low: np.ndarray, high: np.ndarray) -> list[np.ndarray]:
    """12 edge segments of an axis-aligned bbox for plotting."""
    corners = np.array(np.meshgrid([low[0], high[0]],
                                    [low[1], high[1]],
                                    [low[2], high[2]])).T.reshape(-1, 3)
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            diff = np.abs(corners[i] - corners[j])
            if np.count_nonzero(diff > 1e-9) == 1:
                edges.append(np.stack([corners[i], corners[j]]))
    return edges


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    ts = TargetSet.load(str(TARGET_NPZ))
    targets = ts.target_pos  # (N, 3)
    print(f"loaded target_set: n={ts.n}  split={ts.split}")

    env = make_env(ENV_ID)
    try:
        uw = env.unwrapped
        env.reset()
        mj.mj_forward(uw.mj_model, uw.mj_data)
        state = extract_state(env)
        init_tip = np.asarray(state.tip_pos)

        # bbox of generator
        site_name, (low_t, high_t) = next(iter(uw.target_reach_range.items()))
        low = np.asarray(low_t)
        high = np.asarray(high_t)
        print(f"target_reach_range[{site_name}]: low={low}, high={high}")
        print(f"init tip: {init_tip}")

        # all body positions (skeleton)
        n_body = uw.mj_model.nbody
        body_positions = uw.mj_data.xpos.copy()
        body_names = [uw.mj_model.body(i).name for i in range(n_body)]

        # connect bodies through their parent chain to draw the skeleton
        parents = uw.mj_model.body_parentid.copy()
        bones = []
        for b in range(1, n_body):
            p = parents[b]
            if p >= 0:
                bones.append(np.stack([body_positions[p], body_positions[b]]))

        # diagnostic eps
        hl_pts = targets[HIGHLIGHT_EPS]  # (5, 3)

        # for each diagnostic ep, also run IK and compute the tip-after-IK
        # (= geometric reach achievable by the IK solution)
        ik_tips = []
        for ep in HIGHLIGHT_EPS:
            env.reset()
            mj.mj_forward(uw.mj_model, uw.mj_data)
            qpos_sol, info = solve_ik(
                env, targets[ep], max_iter=200, tol=0.01, damping=0.1,
            )
            # apply qpos_sol and read tip
            uw.mj_data.qpos[:qpos_sol.shape[0]] = qpos_sol
            mj.mj_forward(uw.mj_model, uw.mj_data)
            tip_after_ik = extract_state(env).tip_pos
            ik_tips.append(np.asarray(tip_after_ik))
            print(f"  ep={ep:>2}  target={targets[ep]}  "
                  f"ik_err={info['final_error']:.4f}  tip_after_ik={tip_after_ik}")
        ik_tips = np.stack(ik_tips)

        edges = _bbox_edges(low, high)

        # ====== matplotlib figure ======
        fig = plt.figure(figsize=(16, 12))
        views = [
            ((25, -60), "3D perspective"),
            ((90, -90), "Top view (X–Y, looking down Z)"),
            ((0, -90),  "Front view (X–Z, looking along Y)"),
            ((0, 0),    "Side view (Y–Z, looking along X)"),
        ]

        for i, ((elev, azim), title) in enumerate(views):
            ax = fig.add_subplot(2, 2, i + 1, projection="3d")
            ax.view_init(elev=elev, azim=azim)

            # bbox edges
            for e in edges:
                ax.plot(e[:, 0], e[:, 1], e[:, 2], color="black",
                        alpha=0.4, linewidth=1.0, linestyle="--")

            # bones
            for b in bones:
                ax.plot(b[:, 0], b[:, 1], b[:, 2], color="gray",
                        alpha=0.35, linewidth=1.0)

            # body joints
            ax.scatter(body_positions[:, 0], body_positions[:, 1],
                       body_positions[:, 2], c="lightgray", s=8, alpha=0.5)

            # all targets
            ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2],
                       c="red", s=4, alpha=0.25,
                       label=f"targets (n={ts.n})")

            # diagnostic eps
            ax.scatter(hl_pts[:, 0], hl_pts[:, 1], hl_pts[:, 2],
                       c="red", s=80, marker="o", edgecolor="black",
                       linewidth=1.5, label="ep 0,1,2,5,10")
            for k, ep in enumerate(HIGHLIGHT_EPS):
                ax.text(hl_pts[k, 0], hl_pts[k, 1], hl_pts[k, 2] + 0.02,
                        f"ep{ep}", fontsize=9)

            # IK-achievable tip (per diagnostic ep)
            ax.scatter(ik_tips[:, 0], ik_tips[:, 1], ik_tips[:, 2],
                       c="green", s=60, marker="x", linewidth=2,
                       label="IK-achievable tip")

            # init tip
            ax.scatter(*init_tip, c="blue", s=120, marker="*",
                       label="init tip (neutral pose)")

            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
            ax.set_title(title, fontsize=11)
            if i == 0:
                ax.legend(loc="upper left", fontsize=8)

        fig.suptitle(
            f"myoArm: target_set layout vs skeleton  "
            f"(target_set: reachable_train n={ts.n}, IK tol=0.01 m)",
            fontsize=12,
        )
        fig.tight_layout()
        skel_png = OUT / "skeleton_targets.png"
        fig.savefig(skel_png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\nsaved: {skel_png}")

        # ====== MuJoCo offscreen render of neutral pose ======
        try:
            env.reset()
            mj.mj_forward(uw.mj_model, uw.mj_data)
            # MuJoCo's default offscreen framebuffer is 640x480.
            renderer = mj.Renderer(uw.mj_model, height=480, width=640)
            views_mj = [
                ("default", -1),  # default camera
            ]
            imgs = []
            # neutral pose
            env.reset()
            mj.mj_forward(uw.mj_model, uw.mj_data)
            renderer.update_scene(uw.mj_data, camera=-1)
            imgs.append(("neutral pose", renderer.render().copy()))
            # IK-solved poses for diagnostic eps (show how the arm would
            # look if it could actually hold target_qpos)
            for ep in (0, 10):
                env.reset()
                mj.mj_forward(uw.mj_model, uw.mj_data)
                qpos_sol, _ = solve_ik(
                    env, targets[ep], max_iter=200, tol=0.01, damping=0.1,
                )
                uw.mj_data.qpos[:qpos_sol.shape[0]] = qpos_sol
                mj.mj_forward(uw.mj_model, uw.mj_data)
                renderer.update_scene(uw.mj_data, camera=-1)
                imgs.append((f"IK pose for ep={ep}", renderer.render().copy()))
            fig2, axs = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 5))
            if len(imgs) == 1:
                axs = [axs]
            for ax, (title, img) in zip(axs, imgs):
                ax.imshow(img); ax.set_axis_off(); ax.set_title(title)
            mj_png = OUT / "mujoco_render.png"
            fig2.savefig(mj_png, dpi=120, bbox_inches="tight")
            plt.close(fig2)
            print(f"saved: {mj_png}")
        except Exception as e:
            print(f"[warn] MuJoCo offscreen render failed: {e}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
