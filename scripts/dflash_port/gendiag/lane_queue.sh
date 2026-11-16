#!/usr/bin/env bash
# GENDIAG lane reading its run names from an editable queue file (one NAME per
# line, popped from the top), so runs can be inserted while the lane runs.
#
#   scripts/dflash_port/gendiag/lane_queue.sh GPU LOG_DIR QUEUE_FILE [WAIT_PATTERN]
#
# WAIT_PATTERN: first wait until no process matches it (a run started by an
# earlier lane). Each NAME runs through lane.sh (diagnostics on, then publish).
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
gpu="$1"; logs="$2"; queue="$3"; wait_for="${4:-}"
if [ -n "$wait_for" ]; then
  while pgrep -f "$wait_for" > /dev/null; do sleep 30; done
fi
while true; do
  name="$(head -n 1 "$queue" 2>/dev/null || true)"
  [ -z "$name" ] && break
  sed -i '1d' "$queue"
  scripts/dflash_port/gendiag/lane.sh "$gpu" "$logs" "$name"
done
echo "[$(date -u +%H:%M:%S)] gendiag queue lane gpu=$gpu: queue empty"
