#!/usr/bin/env bash
# Pre-compute mercury USB / I/O health check.
#
# Background: mercury's root filesystem lives on an external USB SSD
# (Samsung T7 Shield). On 2026-05-19 a transient USB host-controller
# fault froze the OS for ~41h (see
# Logs/2026-05-19-mercury-usb-root-freeze.md). Until the root is
# migrated to the internal NVMe, run this before launching multi-hour
# evaluations or training jobs.
#
# Exit codes:
#   0 -- no recent USB/sda anomalies; safe to launch long compute.
#   1 -- one or more anomaly events in the lookback window; investigate
#        with `journalctl -k --since "<window> ago"` before launching.
#   2 -- not on mercury (no-op check passes; harmless to call from other
#        hosts).
#
# Usage:
#   scripts/check_mercury_health.sh             # default: last 1 hour
#   scripts/check_mercury_health.sh "6 hours"   # custom lookback window

set -euo pipefail

WINDOW="${1:-1 hour}"
HOST="$(hostname)"

if [[ "$HOST" != "mercury" ]]; then
  echo "[check_mercury_health] host=$HOST not mercury; skipping." >&2
  exit 2
fi

COUNT="$(journalctl -k --since "$WINDOW ago" 2>/dev/null \
  | grep -cE 'xhci_hcd.*ERROR|uas_eh|sda.*FAILED|I/O error' || true)"

if [[ "$COUNT" -eq 0 ]]; then
  echo "[check_mercury_health] window='$WINDOW ago' anomalies=0; OK"
  exit 0
fi

echo "[check_mercury_health] window='$WINDOW ago' anomalies=$COUNT; INVESTIGATE" >&2
echo "  journalctl -k --since '$WINDOW ago' | grep -E 'xhci_hcd.*ERROR|uas_eh|sda.*FAILED|I/O error'" >&2
exit 1
