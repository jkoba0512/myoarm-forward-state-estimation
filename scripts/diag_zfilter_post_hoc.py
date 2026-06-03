"""Post-hoc z-filter analysis: from Step 2 and Step 3 results, restrict
to targets below the shoulder (z < shoulder_z) and recompute success
rates. Sanity-check whether a 'below-shoulder' target_set would close
the oracle gap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# shoulder z from diag_arm_reach_radius.py
SHOULDER_Z = 1.393  # humerus body z (neutral pose)

STEP2 = REPO / "runs/diag/step2_200ep/results.json"
STEP3 = REPO / "runs/diag/step3_thin_ksweep/results.json"


def summarize(rows: list[dict], label: str, n_total: int) -> None:
    mt = np.array([r["min_tip"] for r in rows])
    ft = np.array([r["final_tip"] for r in rows])
    n = len(rows)
    print(f"\n  [{label}]  n={n} / {n_total} ({100*n/n_total:.1f}% retained)")
    print(
        f"    min_tip mean={mt.mean():.4f}  median={np.median(mt):.4f}  "
        f"p25={np.percentile(mt, 25):.4f}  p75={np.percentile(mt, 75):.4f}  "
        f"max={mt.max():.4f}"
    )
    print(
        f"    final_tip mean={ft.mean():.4f}  "
        f"S005={100*(mt < 0.05).mean():.1f}% "
        f"S010={100*(mt < 0.10).mean():.1f}% "
        f"S015={100*(mt < 0.15).mean():.1f}% "
        f"S020={100*(mt < 0.20).mean():.1f}%"
    )


def filter_z(rows: list[dict], z_thresh: float) -> list[dict]:
    return [r for r in rows if r["target_z"] < z_thresh]


def main() -> None:
    # ===== Step 2: 200 ep, K=1.0 =====
    print(f"\n========== Step 2 (K=1.0 oracle, 200 ep) "
          f"vs shoulder_z = {SHOULDER_Z} ==========")
    with open(STEP2) as f:
        step2 = json.load(f)
    rows = step2["rows"]
    n_total = len(rows)
    target_z = np.array([r["target_z"] for r in rows])
    print(f"target_z range: [{target_z.min():.3f}, {target_z.max():.3f}]  "
          f"median={np.median(target_z):.3f}")
    # quantile counts vs shoulder
    print(f"  below shoulder   (z < {SHOULDER_Z}):  "
          f"{int((target_z < SHOULDER_Z).sum())}/{n_total}")
    print(f"  at/above shoulder (z >= {SHOULDER_Z}): "
          f"{int((target_z >= SHOULDER_Z).sum())}/{n_total}")

    summarize(rows, "ALL", n_total)
    summarize(filter_z(rows, SHOULDER_Z), f"z < {SHOULDER_Z}", n_total)
    summarize(filter_z(rows, 1.20), "z < 1.20 (clearly below)", n_total)
    # Above shoulder
    above = [r for r in rows if r["target_z"] >= SHOULDER_Z]
    if above:
        summarize(above, f"z >= {SHOULDER_Z} (above shoulder)", n_total)

    # ===== Step 3: K-sweep, 200 ep × 3 K =====
    print(f"\n\n========== Step 3 (K-sweep, 200 ep × 3 K) "
          f"vs shoulder_z = {SHOULDER_Z} ==========")
    with open(STEP3) as f:
        step3 = json.load(f)
    per_K = step3["per_K"]

    for K_str in ("0.0", "0.5", "1.0"):
        rows_K = per_K[K_str]
        n_total = len(rows_K)
        print(f"\n  --- K = {K_str} ---")
        summarize(rows_K, "ALL", n_total)
        summarize(filter_z(rows_K, SHOULDER_Z),
                  f"z < {SHOULDER_Z}", n_total)
        summarize(filter_z(rows_K, 1.20),
                  "z < 1.20", n_total)

    # ===== K-sweep comparison restricted to below-shoulder =====
    print(f"\n\n=== K-sweep (200 ep, A+B), restricted to z < {SHOULDER_Z} ===")
    print(f"{'K':>5} | {'n':>4} | {'min_tip mean':>13} {'median':>8} "
          f"{'S005%':>7} {'S010%':>7} {'S015%':>7} {'final mean':>11}")
    for K_str in ("0.0", "0.5", "1.0"):
        rows_K = per_K[K_str]
        sub = filter_z(rows_K, SHOULDER_Z)
        mt = np.array([r["min_tip"] for r in sub])
        ft = np.array([r["final_tip"] for r in sub])
        print(
            f"{float(K_str):>5.2f} | {len(sub):>4d} | "
            f"{mt.mean():>13.4f} {np.median(mt):>8.4f} "
            f"{100*np.mean(mt < 0.05):>6.2f}% "
            f"{100*np.mean(mt < 0.10):>6.2f}% "
            f"{100*np.mean(mt < 0.15):>6.2f}% "
            f"{ft.mean():>11.4f}"
        )

    print(f"\n=== K-sweep (200 ep, A+B), restricted to z < 1.20 ===")
    print(f"{'K':>5} | {'n':>4} | {'min_tip mean':>13} {'median':>8} "
          f"{'S005%':>7} {'S010%':>7} {'S015%':>7} {'final mean':>11}")
    for K_str in ("0.0", "0.5", "1.0"):
        rows_K = per_K[K_str]
        sub = filter_z(rows_K, 1.20)
        if len(sub) == 0:
            continue
        mt = np.array([r["min_tip"] for r in sub])
        ft = np.array([r["final_tip"] for r in sub])
        print(
            f"{float(K_str):>5.2f} | {len(sub):>4d} | "
            f"{mt.mean():>13.4f} {np.median(mt):>8.4f} "
            f"{100*np.mean(mt < 0.05):>6.2f}% "
            f"{100*np.mean(mt < 0.10):>6.2f}% "
            f"{100*np.mean(mt < 0.15):>6.2f}% "
            f"{ft.mean():>11.4f}"
        )


if __name__ == "__main__":
    main()
