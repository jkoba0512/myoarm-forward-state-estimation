"""Collect BC demonstrations with ScriptedReachController + true_state.

For each target from a target set, runs the env with a scripted
IK-interpolation reach controller using ``true_state`` as the
observation (no noise/delay, no estimator) and saves the resulting
``(state, action)`` pairs plus per-episode summary statistics.

Usage::

    uv run python scripts/collect_bc_demos.py \\
        --config configs/closed_loop/bc_demo_collect.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from myoarm_fse.controllers import ScriptedReachController
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim
from myoarm_fse.envs.extractors import extract_ctrl, extract_state
from myoarm_fse.envs.factory import make_env
from myoarm_fse.envs.ik import actuator_moment_dense, solve_ik
from myoarm_fse.envs.targets import TargetSet


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect BC demonstrations.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args(argv)


def _make_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_id = cfg["env_id"]
    horizon = int(cfg["horizon"])
    n_episodes = int(cfg["n_episodes"])
    target_set_path = cfg["target_set"]
    ctrl_cfg = cfg["controller"]
    out_dir = args.output or Path(cfg.get("output_root", "runs/bc_demos")) / _make_id()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"env_id: {env_id}  horizon: {horizon}  n_episodes: {n_episodes}")
    print(f"controller: {ctrl_cfg}")
    print(f"output: {out_dir}")

    env = make_env(env_id, horizon=horizon)
    action_dim = detect_action_dim(env)
    adapter = ActionAdapter(action_dim=action_dim)
    target_set = TargetSet.load(target_set_path)
    if n_episodes > target_set.n:
        raise SystemExit(
            f"n_episodes={n_episodes} > target_set.n={target_set.n}"
        )

    uw = env.unwrapped
    target_sid = int(uw.target_sids[0])

    state_dim = None
    states_all: list[np.ndarray] = []
    actions_all: list[np.ndarray] = []
    summary: list[dict[str, Any]] = []

    try:
        for ep in range(n_episodes):
            env.reset()
            target = target_set.target_pos[ep].astype(np.float64)
            uw.mj_model.site_pos[target_sid] = target
            mujoco.mj_forward(uw.mj_model, uw.mj_data)

            init_state = extract_state(env)
            init_qpos = init_state.qpos.copy()
            init_reach = float(np.linalg.norm(init_state.reach_err))

            target_qpos, ik_info = solve_ik(
                env, target,
                max_iter=int(ctrl_cfg.get("ik_max_iter", 200)),
                tol=float(ctrl_cfg.get("ik_tol", 0.01)),
                damping=float(ctrl_cfg.get("ik_damping", 0.1)),
            )
            M = actuator_moment_dense(env)
            controller = ScriptedReachController(
                action_dim=action_dim,
                init_qpos=init_qpos,
                target_qpos=target_qpos,
                moment_arm=M,
                ramp_steps=int(ctrl_cfg.get("ramp_steps", 200)),
                Kp=float(ctrl_cfg.get("Kp", 30.0)),
                Kd=float(ctrl_cfg.get("Kd", 3.0)),
                action_scale=float(ctrl_cfg.get("action_scale", 5.0)),
            )
            controller.reset()

            ep_states: list[np.ndarray] = []
            ep_actions: list[np.ndarray] = []
            min_err = float("inf")
            for t in range(horizon):
                st = extract_state(env)
                state_flat = st.flatten().astype(np.float32)
                if state_dim is None:
                    state_dim = state_flat.shape[0]
                u = controller.act(st)
                u_excit = adapter.clip_excitation(u)
                api = adapter.excitation_to_api_action(u_excit)
                ep_states.append(state_flat)
                ep_actions.append(u_excit.astype(np.float32))
                env.step(api)
                err = float(np.linalg.norm(st.reach_err))
                if err < min_err:
                    min_err = err

            final_state = extract_state(env)
            final_err = float(np.linalg.norm(final_state.reach_err))
            states_arr = np.stack(ep_states)
            actions_arr = np.stack(ep_actions)
            states_all.append(states_arr)
            actions_all.append(actions_arr)
            summary.append({
                "episode": ep,
                "target_index": ep,
                "target_seed": int(target_set.seeds[ep]),
                "init_reach_err": init_reach,
                "min_tip_error": min_err,
                "final_tip_error": final_err,
                "ik_converged": bool(ik_info["converged"]),
                "ik_final_err": float(ik_info["final_error"]),
                "n_steps": int(actions_arr.shape[0]),
            })
            print(f"  ep {ep:>3d}: init={init_reach:.3f}  min={min_err:.3f}  "
                  f"final={final_err:.3f}  ik={'Y' if ik_info['converged'] else 'N'}")
    finally:
        env.close()

    states_concat = np.concatenate(states_all, axis=0)
    actions_concat = np.concatenate(actions_all, axis=0)
    episode_lengths = np.array([s.shape[0] for s in states_all], dtype=np.int64)

    np.savez_compressed(
        out_dir / "demos.npz",
        states=states_concat,
        actions=actions_concat,
        episode_lengths=episode_lengths,
    )
    (out_dir / "summary.json").write_text(json.dumps({
        "config": cfg,
        "state_dim": int(state_dim or 0),
        "action_dim": int(action_dim),
        "n_episodes": int(n_episodes),
        "total_steps": int(states_concat.shape[0]),
        "per_episode": summary,
        "aggregate": {
            "init_reach_mean": float(np.mean([s["init_reach_err"] for s in summary])),
            "min_tip_mean": float(np.mean([s["min_tip_error"] for s in summary])),
            "min_tip_median": float(np.median([s["min_tip_error"] for s in summary])),
            "final_tip_mean": float(np.mean([s["final_tip_error"] for s in summary])),
            "success_005_min": float(np.mean(
                [s["min_tip_error"] < 0.05 for s in summary]
            )),
            "success_010_min": float(np.mean(
                [s["min_tip_error"] < 0.10 for s in summary]
            )),
        },
    }, indent=2))
    print(f"\nSaved: {out_dir / 'demos.npz'}  "
          f"({states_concat.shape[0]} samples)")
    print(f"       {out_dir / 'summary.json'}")
    return out_dir


if __name__ == "__main__":
    main()
