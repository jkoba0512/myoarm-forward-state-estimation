"""Batch driver for Stage C per-cell β diagnostic deploy.

For each of the 6 focused-grid cells (3 noise × 2 delay), build a
1-cell evaluate_closed_loop config that loads the cell's own B.5
SPSA-trained β via beta_source, then invoke evaluate_closed_loop.py
sequentially. Output schema matches C1-C3 (controller-health +
field-wise K + git/source metadata).

Usage:
    uv run python scripts/batch_per_cell_deploy.py \\
        --per-cell-dir runs/per_cell_beta_diagnostic/2026-05-29T02-41-42Z \\
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


def find_beta_path(per_cell_dir: Path, noise: str, delay: int) -> Path:
    """Locate the cell_{label}/<spsa_eval>/final_beta.json."""
    cell_label = f"{noise}_d{delay}"
    cell_dir = per_cell_dir / f"cell_{cell_label}"
    spsa_dirs = sorted(
        d for d in cell_dir.iterdir()
        if d.is_dir() and d.name.startswith("2026-")
    )
    if not spsa_dirs:
        raise SystemExit(f"no SPSA run under {cell_dir}")
    beta = spsa_dirs[-1] / "final_beta.json"
    if not beta.exists():
        raise SystemExit(f"missing final_beta.json: {beta}")
    return beta


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
    p.add_argument("--per-cell-dir", type=Path, required=True,
                   help="B.5 per-cell SPSA root dir")
    p.add_argument("--output-root", type=Path,
                   default=REPO / "runs/closed_loop",
                   help="evaluate_closed_loop output_root")
    p.add_argument("--episodes", type=int, default=200,
                   help="episodes per cell (Codex Q22: 200)")
    p.add_argument("--smoke", action="store_true",
                   help="single-cell single-episode smoke per cell")
    args = p.parse_args()

    eval_script = REPO / "scripts/evaluate_closed_loop.py"
    results: list[tuple[str, Path]] = []
    t_start = time.time()

    for noise, delay in CELLS:
        cell_label = f"{noise}_d{delay}"
        beta = find_beta_path(args.per_cell_dir, noise, delay)
        print(f"\n[per-cell deploy] cell {cell_label}  β={beta}")

        cfg = build_config(
            noise, delay, beta,
            episodes=1 if args.smoke else args.episodes,
            output_root=args.output_root,
            estimator_name=f"per_cell_beta_{cell_label}",
        )

        # write the per-cell config to a temp file so evaluate_closed_loop
        # can record it as the official config_path.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_per_cell_deploy_{cell_label}.yaml",
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
                f"evaluate_closed_loop failed for {cell_label} "
                f"(exit code {ret.returncode})"
            )
        results.append((cell_label, tmp_path))

    print(f"\n[per-cell deploy] all 6 cells done in "
          f"{time.time() - t_start:.0f} s")
    print("Configs (for provenance):")
    for label, p in results:
        print(f"  {label}: {p}")


if __name__ == "__main__":
    main()
