"""Stage A h=600 autoregressive stability diagnostic.

Per the Codex response
``2026-05-27_stage-b-controller-pivot-proposal_response_to-claude-code``
(Q11), Stage A passed h=50 MSE but the K=0 closed-loop run autoregressed
the forward model for h=600 steps and observed non-physical states
(qpos ±12 rad, qvel ±30 rad/s, target_pos drift of 9.5 m). This
diagnostic measures field-wise stability for the four R3 models under
**zero action** (free response from env.reset).

For each model we report:

  - qpos max, min                                   vs env jnt_range
  - |qvel|_max                                      vs conservative threshold
  - target_pos drift max ‖target_pos[t]-target[0]‖  ideally 0 (invariant)
  - tip_pos max distance from init_tip              physical workspace
  - first violation step per field
  - first NaN/Inf step

The forward model is run via ``FixedGainKalmanEstimator(gain=0.0)``,
which is exactly the K=0 forward-model-only path. Zero action is
emitted by ``HoldController(value=0.0)``. Observations are ignored
because gain=0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco as mj  # noqa: E402

from myoarm_fse.controllers import HoldController  # noqa: E402
from myoarm_fse.data.rollout import EpisodeSpec  # noqa: E402
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim  # noqa: E402
from myoarm_fse.envs.factory import make_env  # noqa: E402
from myoarm_fse.envs.state import StateSpec  # noqa: E402
from myoarm_fse.estimators import FixedGainKalmanEstimator  # noqa: E402
from myoarm_fse.estimators.fixed_kalman import _flatten_log_states  # noqa: E402
from myoarm_fse.evaluation import run_closed_loop_episode  # noqa: E402
from myoarm_fse.models import load_model  # noqa: E402

OUT = REPO / "runs/diag/stage_a_h600_stability"

MODELS = {
    "H=1":               "runs/models/2026-05-21T09-25-37Z",
    "H=4":               "runs/models/2026-05-21T09-33-13Z",
    "H=8 (main)":        "runs/models/2026-05-21T09-46-39Z",
    "undertrained-H=8":  "runs/models/2026-05-21T09-54-14Z",
}

ENV_ID = "myoArmReachFixed-v0"
HORIZON = 600

# Conservative field-wise gates (per Codex Q11 guidance: field-wise,
# not a single ||x||).
QVEL_MAX = 10.0          # rad/s; any beyond this is implausibly fast
TIP_WORKSPACE = 1.0      # m from init tip; arm length is ~0.7 m
TARGET_DRIFT_TOL = 0.05  # m; target should not move at all (invariant)


def _stats_per_field(x_est: np.ndarray, state_spec: StateSpec) -> dict:
    """Compute field-wise stability metrics on the predicted trajectory.

    ``x_est`` has shape ``(T, state_dim)``; uses state_spec.layout()
    to slice the fields.
    """
    layout = state_spec.layout()
    qpos = x_est[:, layout["qpos"]]
    qvel = x_est[:, layout["qvel"]]
    act = x_est[:, layout["act"]]
    tip = x_est[:, layout["tip_pos"]]
    tgt = x_est[:, layout["target_pos"]]

    T = x_est.shape[0]

    out: dict = {"T": int(T)}

    # NaN/Inf detection
    nan_step = -1
    for t in range(T):
        if not np.isfinite(x_est[t]).all():
            nan_step = t
            break
    out["first_nan_step"] = int(nan_step)

    # qpos magnitude
    out["qpos_max_abs"] = float(np.max(np.abs(qpos))) if T else 0.0
    out["qpos_max_abs_step"] = int(np.unravel_index(
        np.argmax(np.abs(qpos)), qpos.shape,
    )[0]) if T else -1

    # qvel magnitude
    out["qvel_max_abs"] = float(np.max(np.abs(qvel))) if T else 0.0
    out["qvel_max_abs_step"] = int(np.unravel_index(
        np.argmax(np.abs(qvel)), qvel.shape,
    )[0]) if T else -1
    over_qvel = np.where(np.abs(qvel).max(axis=1) > QVEL_MAX)[0]
    out["qvel_first_violation"] = int(over_qvel[0]) if over_qvel.size else -1

    # activation range
    out["act_max"] = float(act.max())
    out["act_min"] = float(act.min())

    # tip workspace
    tip_init = tip[0]
    tip_dist = np.linalg.norm(tip - tip_init, axis=1)
    out["tip_dist_from_init_max"] = float(tip_dist.max())
    out["tip_dist_from_init_max_step"] = int(np.argmax(tip_dist))
    over_tip = np.where(tip_dist > TIP_WORKSPACE)[0]
    out["tip_first_violation"] = int(over_tip[0]) if over_tip.size else -1

    # target drift (this should be ~0; the target is task-invariant)
    tgt_drift = np.linalg.norm(tgt - tgt[0], axis=1)
    out["target_drift_max"] = float(tgt_drift.max())
    out["target_drift_max_step"] = int(np.argmax(tgt_drift))
    over_tgt = np.where(tgt_drift > TARGET_DRIFT_TOL)[0]
    out["target_first_violation"] = (
        int(over_tgt[0]) if over_tgt.size else -1
    )

    # snapshot end-of-horizon for context
    out["qpos_final_range"] = [float(qpos[-1].min()), float(qpos[-1].max())]
    out["qvel_final_range"] = [float(qvel[-1].min()), float(qvel[-1].max())]
    out["tip_dist_final"] = float(tip_dist[-1])
    out["target_drift_final"] = float(tgt_drift[-1])

    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    env = make_env(ENV_ID, horizon=HORIZON)

    try:
        action_dim = detect_action_dim(env)
        action_adapter = ActionAdapter(action_dim=action_dim)

        # arbitrary target (K=0 ignores observation; target_pos is
        # written into the env state but the forward model has its own
        # state copy). Use the env's default reach target.
        env.reset()
        mj.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
        from myoarm_fse.envs.extractors import extract_state
        target_pos = np.asarray(extract_state(env).target_pos)
        print(f"reference env target_pos: {target_pos}")

        results: dict[str, dict] = {}
        for name, path in MODELS.items():
            print(f"\n========== {name}  ({path}) ==========")
            fm, _, _ = load_model(path)
            fm.eval()

            controller = HoldController(action_dim=action_dim, value=0.0)
            controller.reset(seed=0)
            estimator = FixedGainKalmanEstimator(
                forward_model=fm, gain=0.0,
                state_spec=state_spec, delay_steps=0,
            )
            spec = EpisodeSpec(
                episode_id=0,
                target_id="diag",
                target_split="diag",
                target_seed=0,
                controller_name="hold",
                controller_seed=0, sdn_seed=0, obs_noise_seed=0,
                config_hash="stage_a_h600",
                meta={"model": name, "path": path},
            )
            result = run_closed_loop_episode(
                env, controller, estimator, target_pos,
                state_spec=state_spec, action_adapter=action_adapter,
                sdn=None, obs_noise=None, obs_delay=None,
                obs_compose="noisy_then_delayed",
                max_steps=HORIZON, spec=spec,
            )
            x_est = np.asarray(result.x_est, dtype=np.float64)
            x_true = _flatten_log_states(result.log).astype(np.float64)

            stats = _stats_per_field(x_est, state_spec)
            # also compute env-side stats as a sanity comparison: env
            # rolled out under zero action should be physical.
            stats_env = _stats_per_field(x_true, state_spec)
            results[name] = {
                "path": path,
                "fm_autoregressive": stats,
                "env_zero_action": stats_env,
            }
            # save trace too for any deeper post-hoc inspection
            safe_name = (name.replace(' ', '_').replace('(', '')
                              .replace(')', '').replace('=', ''))
            np.savez(
                OUT / f"trace_{safe_name}.npz",
                x_est=x_est, x_true=x_true,
            )

            print(
                f"  fm:  qpos_max_abs={stats['qpos_max_abs']:.2f}  "
                f"qvel_max_abs={stats['qvel_max_abs']:.2f}  "
                f"tip_dist_max={stats['tip_dist_from_init_max']:.2f}  "
                f"target_drift_max={stats['target_drift_max']:.2f}  "
                f"NaN@{stats['first_nan_step']}"
            )
            print(
                f"  env: qpos_max_abs={stats_env['qpos_max_abs']:.2f}  "
                f"qvel_max_abs={stats_env['qvel_max_abs']:.2f}  "
                f"tip_dist_max={stats_env['tip_dist_from_init_max']:.2f}  "
                f"target_drift_max={stats_env['target_drift_max']:.4f}"
            )

        # ----- short summary table -----
        print("\n\n=== Stage A h=600 stability (zero action, autoregressive) ===")
        print(f"qvel threshold = {QVEL_MAX} rad/s, "
              f"tip workspace = {TIP_WORKSPACE} m, "
              f"target drift tol = {TARGET_DRIFT_TOL} m\n")
        hdr = (f"{'model':<20} | "
               f"{'qpos|max':>9} {'qvel|max':>9} {'tip drift':>10} "
               f"{'tgt drift':>10} | "
               f"{'qvel viol':>10} {'tip viol':>10} {'tgt viol':>10} "
               f"{'NaN@':>5}")
        print(hdr)
        print("-" * len(hdr))
        for name, r in results.items():
            s = r["fm_autoregressive"]
            print(
                f"{name:<20} | "
                f"{s['qpos_max_abs']:>9.2f} {s['qvel_max_abs']:>9.2f} "
                f"{s['tip_dist_from_init_max']:>10.2f} "
                f"{s['target_drift_max']:>10.2f} | "
                f"{s['qvel_first_violation']:>10} "
                f"{s['tip_first_violation']:>10} "
                f"{s['target_first_violation']:>10} "
                f"{s['first_nan_step']:>5}"
            )

        print("\n=== Env-side (zero action ground truth) for reference ===")
        print(hdr); print("-" * len(hdr))
        for name, r in results.items():
            s = r["env_zero_action"]
            print(
                f"{name:<20} | "
                f"{s['qpos_max_abs']:>9.2f} {s['qvel_max_abs']:>9.2f} "
                f"{s['tip_dist_from_init_max']:>10.2f} "
                f"{s['target_drift_max']:>10.4f} | "
                f"{s['qvel_first_violation']:>10} "
                f"{s['tip_first_violation']:>10} "
                f"{s['target_first_violation']:>10} "
                f"{s['first_nan_step']:>5}"
            )

        with open(OUT / "results.json", "w") as f:
            json.dump({
                "gates": {
                    "QVEL_MAX": QVEL_MAX,
                    "TIP_WORKSPACE": TIP_WORKSPACE,
                    "TARGET_DRIFT_TOL": TARGET_DRIFT_TOL,
                },
                "models": results,
                "ref_target_pos": [float(x) for x in target_pos],
            }, f, indent=2)
        print(f"\nsaved: {OUT / 'results.json'}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
