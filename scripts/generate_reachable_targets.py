"""Generate an IK-filtered (reachable) target set for myoArm reaching.

Mirrors ``generate_targets.py`` but rejects candidates whose damped-Jacobian
IK solution leaves a final tip-to-target error above ``--threshold``.

Per Codex's 2026-05-21 spec (exchange
``2026-05-21_c2-single-cell-200ep-result_response_to-claude-code.md``):

* Uniform sample from the env's intrinsic ``target_reach_range``.
* For each candidate seed, call ``solve_ik``; accept iff
  ``final_error <= threshold``.
* Advance the seed counter on rejection; keep going until ``n`` accepted.
* Save with extra ``meta``: ``ik_threshold``, attempted / rejected counts,
  acceptance rate, IK error summary, full rejected-seed list.

Output dir defaults to ``runs/targets_reachable/<UTC timestamp>/``.

Usage::

    uv run python scripts/generate_reachable_targets.py \\
        --env-id myoArmReachRandom-v0 \\
        --split smoke --n 20 --threshold 0.01 \\
        --generator-seed 0 --seed-offset 10000
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from myoarm_fse.envs.targets import (
    GENERATOR_VERSION,
    TargetSet,
    _DTYPE_DIST,
    _DTYPE_SEED,
    _DTYPE_TARGET,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-id", default="myoArmReachRandom-v0")
    p.add_argument("--split", default="reachable", help="Name embedded in meta and the output file.")
    p.add_argument("--n", type=int, required=True, help="Target count after IK acceptance.")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Accept candidate iff solve_ik final_error <= threshold (metres).",
    )
    p.add_argument("--generator-seed", type=int, default=0)
    p.add_argument("--seed-offset", type=int, default=10000)
    p.add_argument(
        "--max-attempts-mult",
        type=int,
        default=20,
        help="Hard cap: max_attempts = max_attempts_mult * n. Aborts if exceeded.",
    )
    p.add_argument(
        "--ik-max-iter",
        type=int,
        default=200,
    )
    p.add_argument(
        "--ik-damping",
        type=float,
        default=0.1,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to runs/targets_reachable/<UTC>/<split>.npz",
    )
    return p.parse_args(argv)


def _summary(errs: list[float]) -> dict[str, float]:
    arr = np.asarray(errs, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p95": float(np.quantile(arr, 0.95)),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n <= 0:
        raise SystemExit(f"--n must be > 0, got {args.n}")
    if args.threshold <= 0:
        raise SystemExit(f"--threshold must be > 0, got {args.threshold}")

    import mujoco

    from myoarm_fse.envs.extractors import extract_state
    from myoarm_fse.envs.factory import make_env
    from myoarm_fse.envs.ik import solve_ik

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    if args.output is None:
        out_dir = Path("runs") / "targets_reachable" / timestamp
        out_path = out_dir / f"{args.split}.npz"
    else:
        out_path = args.output
        out_dir = out_path.parent

    print(f"env_id={args.env_id} split={args.split} n={args.n} "
          f"threshold={args.threshold:.4f} m")
    print(f"generator_seed={args.generator_seed} seed_offset={args.seed_offset}")
    print(f"output={out_path}")

    env = make_env(args.env_id)
    try:
        uw = env.unwrapped
        bbox = uw.target_reach_range
        if len(bbox) != 1:
            raise SystemExit(
                f"expected exactly one target_reach_range entry, got {bbox!r}"
            )
        site_name, (low_t, high_t) = next(iter(bbox.items()))
        low = np.asarray(low_t, dtype=np.float64)
        high = np.asarray(high_t, dtype=np.float64)
        target_sid = uw.mj_model.site(site_name + "_target").id

        accepted_seeds: list[int] = []
        accepted_target_pos: list[np.ndarray] = []
        accepted_init_dist: list[float] = []
        accepted_ik_err: list[float] = []
        rejected_seeds: list[int] = []
        rejected_ik_err: list[float] = []

        max_attempts = args.max_attempts_mult * args.n
        base = args.generator_seed + args.seed_offset
        attempt_idx = 0

        while len(accepted_seeds) < args.n:
            if attempt_idx >= max_attempts:
                raise SystemExit(
                    f"hit max_attempts={max_attempts} with only "
                    f"{len(accepted_seeds)}/{args.n} accepted; raise "
                    f"--max-attempts-mult or relax --threshold"
                )
            seed = base + attempt_idx
            attempt_idx += 1

            rng = np.random.default_rng(int(seed))
            t = rng.uniform(low=low, high=high)
            env.reset()
            uw.mj_model.site_pos[target_sid] = t
            mujoco.mj_forward(uw.mj_model, uw.mj_data)
            state = extract_state(env)

            _, info = solve_ik(
                env,
                state.target_pos,
                max_iter=args.ik_max_iter,
                tol=args.threshold,
                damping=args.ik_damping,
                preserve_env_state=True,
            )
            final_err = float(info["final_error"])

            if final_err <= args.threshold:
                accepted_seeds.append(int(seed))
                accepted_target_pos.append(state.target_pos.astype(_DTYPE_TARGET))
                accepted_init_dist.append(
                    float(np.linalg.norm(state.tip_pos - state.target_pos))
                )
                accepted_ik_err.append(final_err)
            else:
                rejected_seeds.append(int(seed))
                rejected_ik_err.append(final_err)

        attempted = attempt_idx
        acceptance_rate = len(accepted_seeds) / attempted if attempted else 0.0

        print()
        print(f"  attempted: {attempted}")
        print(f"  accepted:  {len(accepted_seeds)}")
        print(f"  rejected:  {len(rejected_seeds)}")
        print(f"  acceptance rate: {acceptance_rate:.3f}")
        if accepted_ik_err:
            s = _summary(accepted_ik_err)
            print(f"  accepted IK final_error (m): "
                  f"mean={s['mean']:.5f} median={s['median']:.5f} "
                  f"max={s['max']:.5f} p95={s['p95']:.5f}")
        if rejected_ik_err:
            s = _summary(rejected_ik_err)
            print(f"  rejected IK final_error (m): "
                  f"mean={s['mean']:.4f} median={s['median']:.4f} "
                  f"max={s['max']:.4f} p95={s['p95']:.4f}")

        meta: dict[str, Any] = {
            "env_id": args.env_id,
            "split": args.split,
            "generator_seed": args.generator_seed,
            "seed_offset": args.seed_offset,
            "n": args.n,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "generator_version": GENERATOR_VERSION,
            "source": "ik_filtered_uniform_in_target_reach_range",
            "site_name": site_name,
            "site_low": list(map(float, low_t)),
            "site_high": list(map(float, high_t)),
            "ik_threshold": float(args.threshold),
            "ik_max_iter": int(args.ik_max_iter),
            "ik_damping": float(args.ik_damping),
            "attempted": int(attempted),
            "accepted": int(len(accepted_seeds)),
            "rejected": int(len(rejected_seeds)),
            "acceptance_rate": float(acceptance_rate),
            "accepted_ik_error_summary": _summary(accepted_ik_err),
            "rejected_ik_error_summary": _summary(rejected_ik_err),
            "rejected_seeds": rejected_seeds,
            "notes": (
                "target_pos sampled uniformly in target_reach_range via "
                "np.random.default_rng(seed_i).uniform(low, high). For each "
                "candidate, solve_ik is called with tol=ik_threshold and "
                "max_iter=ik_max_iter; the seed is accepted iff "
                "info['final_error'] <= ik_threshold. Rejected seeds and "
                "their IK final errors are retained in meta for provenance."
            ),
        }

        ts = TargetSet(
            split=args.split,
            seeds=np.asarray(accepted_seeds, dtype=_DTYPE_SEED),
            target_pos=np.stack(accepted_target_pos, axis=0).astype(_DTYPE_TARGET),
            tip_to_target_init_distance=np.asarray(
                accepted_init_dist, dtype=_DTYPE_DIST
            ),
            meta=meta,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        ts.save(out_path)
        print(f"\nsaved: {out_path}")

        meta_path = out_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"saved: {meta_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
