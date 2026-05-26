#!/usr/bin/env bash
# Pre-compute host disk/IO health check (host-agnostic dispatcher).
#
# Each host has a different boot-storage failure mode worth guarding
# against before launching multi-hour evaluations or training jobs.
# This script branches on hostname and applies the relevant
# journalctl filter for that host.
#
# Hosts currently covered:
#   mercury  -- root FS lives on external USB SSD (Samsung T7 Shield).
#               On 2026-05-19 a transient USB host-controller fault
#               froze the OS for ~41h (see
#               Logs/2026-05-19-mercury-usb-root-freeze.md). Watches
#               for xhci_hcd / uas_eh / sda failures and generic I/O
#               errors.
#   joker    -- root FS on internal NVMe. Watches for nvme-driver
#               level errors and generic I/O errors.
#   other    -- no-op (exit 2). Harmless to call from any host.
#
# Exit codes:
#   0 -- no recent anomalies in the lookback window; safe to launch.
#   1 -- one or more anomaly events; investigate before launching.
#        Suggested follow-up: `journalctl -k --since "<window> ago"`.
#   2 -- host not covered by this script (no-op pass).
#
# Usage:
#   scripts/check_host_health.sh             # default: last 1 hour
#   scripts/check_host_health.sh "6 hours"   # custom lookback window

set -euo pipefail

WINDOW="${1:-1 hour}"
HOST="$(hostname)"

case "$HOST" in
  mercury)
    PATTERN='xhci_hcd.*ERROR|uas_eh|sda.*FAILED|I/O error'
    ;;
  joker)
    PATTERN='nvme.*(error|EIO|Cannot|failed)|I/O error'
    ;;
  *)
    echo "[check_host_health] host=$HOST not covered; skipping." >&2
    exit 2
    ;;
esac

COUNT="$(journalctl -k --since "$WINDOW ago" 2>/dev/null \
  | grep -cE "$PATTERN" || true)"

if [[ "$COUNT" -eq 0 ]]; then
  echo "[check_host_health] host=$HOST window='$WINDOW ago' anomalies=0; OK"
  exit 0
fi

echo "[check_host_health] host=$HOST window='$WINDOW ago' anomalies=$COUNT; INVESTIGATE" >&2
echo "  journalctl -k --since '$WINDOW ago' | grep -E '$PATTERN'" >&2
exit 1
