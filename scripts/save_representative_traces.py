"""SB-6: save representative traces for K=0 and K=0.5 from a B.6 K-sweep.

Re-runs the episodes whose min_tip is closest to the per-cell median,
and dumps the full per-step trace:
  x_est, x_true, err_to_target, target_pos,
  activation (T, nu), nnls_residual (T,), ramp_progress (T,)

These complement the aggregate Stage B metrics so reviewers (and we)
can audit whether the controller is actually working or the aggregate
metric is hiding a failure mode — the lesson from the old joint-PD
artefact (Codex 2026-05-27 response, Stage B gate guidance).

Default reads a B.6 v2 eval dir and saves K=0 / K=0.5 × 2 ep each
under the clean cell (noise=none, delay=0).

Usage:
  uv run python scripts/save_representative_traces.py \\
      --eval-dir runs/closed_loop/<eval_id> \\
      --k-list K=0.00 K=0.50 --n-per-k 2 \\
      --cell-noise none --cell-delay 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco as mj  # noqa: E402

from myoarm_fse.controllers import StabilizedEndpointController  # noqa: E402
from myoarm_fse.data.rollout import EpisodeSpec  # noqa: E402
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim  # noqa: E402
from myoarm_fse.envs.extractors import extract_state  # noqa: E402
from myoarm_fse.envs.factory import make_env  # noqa: E402
from myoarm_fse.envs.ik import (  # noqa: E402
    actuator_moment_dense, tip_jacobian_dense,
)
from myoarm_fse.envs.state import StateSpec  # noqa: E402
from myoarm_fse.envs.targets import TargetSet  # noqa: E402
from myoarm_fse.estimators import FixedGainKalmanEstimator  # noqa: E402
from myoarm_fse.estimators.fixed_kalman import _flatten_log_states  # noqa: E402
from myoarm_fse.evaluation import run_closed_loop_episode  # noqa: E402
from myoarm_fse.models import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eval-dir", type=Path, required=True,
                   help="Path to runs/closed_loop/<eval_id>/ from B.6.")
    p.add_argument("--cell-noise", default="none",
                   help="Which noise condition to sample from.")
    p.add_argument("--cell-delay", type=int, default=0,
                   help="Which delay (steps) to sample from.")
    p.add_argument("--k-list", nargs="+",
                   default=["K=0.00", "K=0.50"],
                   help="Estimator names to save traces for.")
    p.add_argument("--n-per-k", type=int, default=2,
                   help="How many representative episodes per K.")
    p.add_argument("--output", type=Path,
                   default=REPO / "runs/diag/r3_stage_b_traces",
                   help="Output directory for npz traces.")
    return p.parse_args()


def pick_representative_eps(
    metrics_csv: Path, k_name: str, noise: str, delay: int, n: int,
) -> list[int]:
    """Return the n episode indices whose min_tip is closest to the
    per-cell median for (k_name, noise, delay)."""
    rows: list[dict] = []
    with open(metrics_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (
                r["estimator"] == k_name
                and r["noise_condition"] == noise
                and int(r["delay_steps"]) == delay
            ):
                rows.append({
                    "ep": int(r["episode_id"]),
                    "min_tip": float(r["min_tip_error"]),
                })
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda x: x["min_tip"])
    median_min_tip = sorted_rows[len(sorted_rows) // 2]["min_tip"]
    rows_with_dist = [
        (r, abs(r["min_tip"] - median_min_tip)) for r in rows
    ]
    rows_with_dist.sort(key=lambda x: x[1])
    return [r["ep"] for r, _ in rows_with_dist[:n]]


def re_run_episode(env, fm, state_spec, target_set, ep_idx, K,
                   controller_spec, action_adapter, action_dim,
                   horizon: int):
    target_pos = target_set.target_pos[ep_idx]
    env.reset()
    mj.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
    init_tip = np.asarray(extract_state(env).tip_pos, dtype=np.float32)
    jacobian = tip_jacobian_dense(env)
    moment_arm = actuator_moment_dense(env)
    T_ramp_raw = controller_spec.get("T_ramp", 300)
    T_ramp_arg = None if T_ramp_raw is None else int(T_ramp_raw)
    controller = StabilizedEndpointController(
        action_dim=action_dim,
        init_tip=init_tip,
        target_pos=target_pos,
        jacobian=jacobian,
        moment_arm=moment_arm,
        Kp=float(controller_spec.get("Kp", 30.0)),
        Kd=float(controller_spec.get("Kd", 3.0)),
        action_scale=float(controller_spec.get("action_scale", 5.0)),
        T_ramp=T_ramp_arg,
        record_history=True,
    )
    controller.reset(seed=0)
    estimator = FixedGainKalmanEstimator(
        forward_model=fm, gain=float(K),
        state_spec=state_spec, delay_steps=0,
    )
    spec = EpisodeSpec(
        episode_id=ep_idx,
        target_id=str(int(target_set.seeds[ep_idx])),
        target_split=target_set.split,
        target_seed=int(target_set.seeds[ep_idx]),
        controller_name="stabilized_endpoint",
        controller_seed=0, sdn_seed=0, obs_noise_seed=0,
        config_hash="sb6_trace",
        meta={"K": str(K)},
    )
    result = run_closed_loop_episode(
        env, controller, estimator, target_pos,
        state_spec=state_spec, action_adapter=action_adapter,
        sdn=None, obs_noise=None, obs_delay=None,
        obs_compose="noisy_then_delayed",
        max_steps=horizon, spec=spec,
    )
    x_est = np.asarray(result.x_est, dtype=np.float64)
    x_true = _flatten_log_states(result.log).astype(np.float64)
    layout = state_spec.layout()
    true_tip = x_true[:, layout["tip_pos"]]
    err_to_target = np.linalg.norm(
        true_tip - np.asarray(target_pos), axis=1,
    )
    return {
        "x_est": x_est,
        "x_true": x_true,
        "err_to_target": err_to_target,
        "target_pos": np.asarray(target_pos),
        "activation": np.asarray(controller.activation_history),
        "nnls_residual": np.asarray(controller.nnls_residual_history),
        "ramp_progress": np.asarray(controller.ramp_progress_history),
    }


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir
    metrics_csv = eval_dir / "metrics.csv"
    config_json = eval_dir / "config.json"
    if not metrics_csv.exists():
        sys.exit(f"metrics.csv not found at {metrics_csv}")
    if not config_json.exists():
        sys.exit(f"config.json not found at {config_json}")

    meta = json.loads(config_json.read_text())
    cfg = meta["config"]
    forward_model = cfg["forward_model"]
    target_set_path = cfg["target_set"]
    controller_spec = cfg["controller"]
    env_id = cfg["env_id"]
    horizon = int(cfg["horizon"])

    args.output.mkdir(parents=True, exist_ok=True)

    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    fm, _, _ = load_model(forward_model)
    fm.eval()
    env = make_env(env_id, horizon=horizon)
    try:
        target_set = TargetSet.load(target_set_path)
        action_dim = detect_action_dim(env)
        action_adapter = ActionAdapter(action_dim=action_dim)

        for k_name in args.k_list:
            K = float(k_name.replace("K=", ""))
            eps = pick_representative_eps(
                metrics_csv, k_name,
                args.cell_noise, args.cell_delay, args.n_per_k,
            )
            print(f"[{k_name}] representative eps for cell "
                  f"(noise={args.cell_noise}, d={args.cell_delay}): "
                  f"{eps}")
            for ep in eps:
                trace = re_run_episode(
                    env, fm, state_spec, target_set, ep, K,
                    controller_spec, action_adapter, action_dim,
                    horizon,
                )
                safe = k_name.replace("=", "-")
                out_path = args.output / (
                    f"{safe}_n-{args.cell_noise}_d-{args.cell_delay}"
                    f"_ep-{ep}.npz"
                )
                np.savez(out_path, **trace)
                print(
                    f"  saved: {out_path}  "
                    f"(min_tip={trace['err_to_target'].min():.4f}, "
                    f"final_tip={trace['err_to_target'][-1]:.4f}, "
                    f"max_nnls_res={trace['nnls_residual'].max():.3f}, "
                    f"max_act={trace['activation'].max():.3f})"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
