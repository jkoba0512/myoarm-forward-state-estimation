#!/usr/bin/env bash
# Reproduce the R3 Stage C paired-bootstrap CI table
# (Sec. Results/C3-C4, Table tab:r3_paired_ci) from the frozen Stage B
# K-sweep and Stage C deploy run identifiers fixed for the R3 paper.
#
# Usage:
#   bash scripts/r3_reproduce_stage_c_ci.sh
#
# Outputs:
#   runs/diag/r3_stage_c_paired_ci.csv

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$REPO_ROOT"

# Frozen run identifiers for the R3 paper (Codex 2026-06-01).
B6="runs/closed_loop/2026-05-27T11-09-25Z"           # Stage B K-sweep (200 ep × 6 cells × 5 K)
C1="runs/closed_loop/2026-06-01T00-43-36Z"           # default reliability
C2="runs/closed_loop/2026-06-01T02-00-24Z"           # single-cell SPSA at (none, d=18)
C3="runs/closed_loop/2026-06-01T02-22-56Z"           # global SPSA across 6 cells
C4_DIR="runs/closed_loop"                            # C4 outputs live as estimator c4_feature_conditioned_*
PER_CELL_DIR="runs/closed_loop"                      # per-cell deploys as per_cell_beta_*
OUT="runs/diag/r3_stage_c_paired_ci.csv"

mkdir -p "$(dirname "$OUT")"

uv run --no-sync python scripts/stage_c_paired_ci.py \
  --c1 "$C1/metrics.csv" \
  --c2 "$C2/metrics.csv" \
  --c3 "$C3/metrics.csv" \
  --c4-dir "$C4_DIR" --c4-prefix c4_feature_conditioned \
  --per-cell-dir "$PER_CELL_DIR" --per-cell-prefix per_cell_beta \
  --b6 "$B6/metrics.csv" \
  --output "$OUT"

echo "Wrote: $OUT"
