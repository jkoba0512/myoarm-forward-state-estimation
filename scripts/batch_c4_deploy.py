"""Batch driver for Stage C C4 feature-conditioned β deploy.

Reads the precomputed per-cell β JSONs (from
``precompute_feature_conditioned_betas.py``) and invokes
``evaluate_closed_loop.py`` once per cell so the metrics.csv schema
matches C1-C3 / per-cell deploy (controller-health + field-wise K +
git/source metadata).

Usage:
    uv run python scripts/batch_c4_deploy.py \\
        --per-cell-beta-dir runs/feature_conditioned_beta/<eval>/per_cell \\
        --output-root runs/closed_loop \\
        --episodes 200
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

NOISE_PRESETS = {
    "none":  {"qpos": 0.0,  "qvel": 0.0,  "tip_pos": 0.0,  "reach_err": 0.0},
    "high":  {"qpos": 0.02, "qvel": 0.02, "tip_pos": 0.01, "reach_err": 0.01},
    "xhigh": {"qpos": 0.08, "qvel": 0.08, "tip_pos": 0.04, "reach_err": 0.04},
}
CELLS = [
    ("none", 0), ("none", 18),
    ("high", 0), ("high", 18),
    ("xhigh", 0), ("xhigh", 18),
]

FORWARD_MODEL = "runs/models/2026-05-21T09-46-39Z"
TARGET_SET = "runs/targets_reachable/2026-05-27T07-37-54Z/reachable_train.npz"


def build_config(noise: str, delay: int, beta_source: Path,
                 episodes: int, output_root: Path,
                 estimator_name: str) -> dict:
    return {
        "seed": 0,
        "env_id": "myoArmReachFixed-v0",
        "horizon": 600,
        "forward_model": FORWARD_MODEL,
        "target_set": TARGET_SET,
        "episodes_per_cell": int(episodes),
        "obs_compose": "noisy_then_delayed",
        "noise_conditions": {noise: NOISE_PRESETS[noise]},
        "delay_grid": [int(delay)],
        "estimators": [{
            "name": estimator_name,
            "kind": "reliability_adaptive",
            "beta_source": str(beta_source),
            "config": {
                "alpha": 0.05,
                "epsilon": 1.0e-6,
                "var_init": 1.0,
                "target_pos_gain": 1.0,
            },
        }],
        "controller": {
            "name": "stabilized_endpoint",
            "Kp": 30.0, "Kd": 3.0, "action_scale": 5.0,
            "T_ramp": 300, "record_history": True,
        },
        "sdn": {"sigma": 0.0},
        "success_thresholds": [0.05, 0.10, 0.15],
        "success_duration": 10,
        "skip_cold_start_steps_use_delay": True,
        "save_per_episode": True,
        "output_root": str(output_root),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--per-cell-beta-dir", type=Path, required=True,
                   help="dir produced by precompute_feature_conditioned_betas.py")
    p.add_argument("--output-root", type=Path,
                   default=REPO / "runs/closed_loop",
                   help="evaluate_closed_loop output_root")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    eval_script = REPO / "scripts/evaluate_closed_loop.py"
    t_start = time.time()

    for noise, delay in CELLS:
        cell_label = f"{noise}_d{delay}"
        beta = args.per_cell_beta_dir / f"cell_{cell_label}.json"
        if not beta.exists():
            raise SystemExit(f"missing precomputed β: {beta}")
        print(f"\n[C4 deploy] cell {cell_label}  β={beta}")

        cfg = build_config(
            noise, delay, beta,
            episodes=1 if args.smoke else args.episodes,
            output_root=args.output_root,
            estimator_name=f"c4_feature_conditioned_{cell_label}",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_c4_deploy_{cell_label}.yaml",
            delete=False,
        ) as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False)
            tmp_path = Path(tmp.name)

        cmd = ["uv", "run", "--no-sync", "python", str(eval_script),
               "--config", str(tmp_path), "--rss-log-every", "200"]
        if args.smoke:
            cmd.append("--smoke")
        ret = subprocess.run(cmd, cwd=REPO)
        if ret.returncode != 0:
            raise SystemExit(
                f"evaluate_closed_loop failed for C4 {cell_label} "
                f"(exit code {ret.returncode})"
            )

    print(f"\n[C4 deploy] all 6 cells done in "
          f"{time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
