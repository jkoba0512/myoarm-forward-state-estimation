#!/usr/bin/env bash
# Resume the B.6 oracle K-sweep (R3 Stage B) with memory safety net.
#
# Wraps `evaluate_closed_loop.py --resume-dir <existing>` in
# `systemd-run --user --scope -p MemoryMax=...` so an in-process memory
# blow-up (RSS > cap) kills only this process — not the whole machine.
# If the process exits non-zero (OOM / SIGKILL / SIGTERM / crash) the
# wrapper auto-retries with --resume-dir; already-done episodes are
# skipped from disk, so seeds stay reproducible across restarts.
#
# Usage:
#   scripts/resume_b6_oracle_k_sweep.sh \
#       runs/closed_loop/2026-05-22T02-57-29Z \
#       configs/closed_loop/oracle_k_sweep_r3.yaml
#
# Env knobs:
#   MEM_CAP   memory cap passed to systemd-run (default 40G)
#   MAX_TRY   max wrapper retries on non-zero exit (default 3)

set -euo pipefail

RESUME_DIR="${1:?usage: $0 <resume-dir> <config-yaml>}"
CONFIG="${2:?usage: $0 <resume-dir> <config-yaml>}"
MEM_CAP="${MEM_CAP:-40G}"
MAX_TRY="${MAX_TRY:-3}"

if [[ ! -d "$RESUME_DIR" ]]; then
  echo "[resume] resume-dir does not exist: $RESUME_DIR" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[resume] config does not exist: $CONFIG" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$REPO_ROOT"

# Pre-check host disk/IO health (anomalies in the last hour). The
# script returns 0 / 1 / 2 (2 = host not covered, harmless).
if [[ -x scripts/check_host_health.sh ]]; then
  scripts/check_host_health.sh "1 hour" || {
    rc=$?
    if [[ "$rc" -eq 1 ]]; then
      echo "[resume] host health anomalies detected; abort." >&2
      exit 1
    fi
  }
fi

TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG="logs/r3_stage_b_b6_resume_${TS}.log"
mkdir -p "$(dirname "$LOG")"

echo "[resume] resume_dir=$RESUME_DIR" | tee -a "$LOG"
echo "[resume] config=$CONFIG"       | tee -a "$LOG"
echo "[resume] mem_cap=$MEM_CAP"     | tee -a "$LOG"
echo "[resume] max_try=$MAX_TRY"     | tee -a "$LOG"
echo "[resume] log=$LOG"             | tee -a "$LOG"

# systemd-run --user --scope keeps the process in our session but
# applies MemoryMax via the user.slice's transient scope. If the
# python RSS exceeds MEM_CAP the kernel OOM-kills it (anon kill, not
# system-wide), which we catch and retry below.
RUN_CMD=(
  systemd-run --user --scope --quiet
  -p "MemoryMax=$MEM_CAP"
  -p "MemorySwapMax=0"
  -- uv run --no-sync python scripts/evaluate_closed_loop.py
  --config "$CONFIG"
  --resume-dir "$RESUME_DIR"
  --rss-log-every 50
)

attempt=0
while (( attempt < MAX_TRY )); do
  attempt=$((attempt + 1))
  echo "[resume] === attempt $attempt / $MAX_TRY at $(date -Is) ===" | tee -a "$LOG"
  set +e
  "${RUN_CMD[@]}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  echo "[resume] attempt $attempt exit_code=$rc" | tee -a "$LOG"
  if [[ "$rc" -eq 0 ]]; then
    echo "[resume] success on attempt $attempt" | tee -a "$LOG"
    exit 0
  fi
  # Retryable: OOM (137 = 128+9 SIGKILL), SIGTERM (143), SIGINT (130
  # we treat as fatal — user asked us to stop). Any non-zero exit
  # other than 130 we retry, since resume is idempotent.
  if [[ "$rc" -eq 130 ]]; then
    echo "[resume] SIGINT — abort retries." | tee -a "$LOG"
    exit "$rc"
  fi
  echo "[resume] non-zero exit ($rc); sleeping 10s then retrying with --resume-dir" | tee -a "$LOG"
  sleep 10
done

echo "[resume] all $MAX_TRY attempts exhausted; aborting." | tee -a "$LOG"
exit 1
