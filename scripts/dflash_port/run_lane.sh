#!/usr/bin/env bash
# One GPU lane: run tier matrices in order, publish each to HF when it is done.
#
#   scripts/dflash_port/run_lane.sh GPU LOG_DIR MATRIX [MATRIX ...]
#
# - Each matrix runs with `run_matrix.py --no-hub` on CUDA device GPU (W&B on);
#   completed entries are skipped, so a rerun resumes where the lane stopped.
# - The lane stops at the first failure (nothing after it runs).
# - After each matrix, its completed training runs are published in the
#   background (`push_matrix.py --visibility public`), so uploads never hold
#   the GPU; push logs go to LOG_DIR/push_<matrix>.log.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
gpu="${1:?usage: run_lane.sh GPU LOG_DIR MATRIX [MATRIX ...]}"
logs="${2:?usage: run_lane.sh GPU LOG_DIR MATRIX [MATRIX ...]}"
shift 2
mkdir -p "$logs"
for matrix in "$@"; do
  name="$(basename "$matrix" .yaml)"
  echo "[$(date -u +%H:%M:%S)] lane gpu=$gpu: matrix $matrix"
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/dflash_port/run_matrix.py "$matrix" --no-hub
  echo "[$(date -u +%H:%M:%S)] lane gpu=$gpu: matrix $matrix done; publishing in the background"
  nohup nice -n 10 .venv/bin/python scripts/dflash_port/push_matrix.py "$matrix" \
    --visibility public > "$logs/push_${name}.log" 2>&1 &
done
echo "[$(date -u +%H:%M:%S)] lane gpu=$gpu: all matrices done"
