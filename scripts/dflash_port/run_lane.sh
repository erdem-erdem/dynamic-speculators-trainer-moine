#!/usr/bin/env bash
# One GPU lane: run tier matrices in order, publish each to HF when it is done.
#
#   scripts/dflash_port/run_lane.sh GPU LOG_DIR MATRIX [MATRIX ...]
#
# - Each matrix runs with `run_matrix.py --no-hub` on CUDA device GPU (W&B on);
#   completed entries are skipped, so a rerun resumes where the lane stopped.
# - A failed matrix is retried once (completed entries are skipped and training
#   resumes from its last checkpoint), for transient failures such as the
#   Triton cache race on the shared BeeGFS compiler cache (OSError EBUSY,
#   2026-11-09). A second failure stops the lane (nothing after it runs).
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
# The Triton cache may be a symlink to local disk (BeeGFS gives EBUSY on
# Triton's temp-dir cleanup); recreate its target after a container restart.
triton="output/dflash-lora/compiler-cache/triton"
if [ -L "$triton" ]; then mkdir -p "$(readlink "$triton")"; fi
for matrix in "$@"; do
  name="$(basename "$matrix" .yaml)"
  echo "[$(date -u +%H:%M:%S)] lane gpu=$gpu: matrix $matrix"
  if ! CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/dflash_port/run_matrix.py "$matrix" --no-hub; then
    echo "[$(date -u +%H:%M:%S)] lane gpu=$gpu: matrix $matrix FAILED; retrying once"
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/dflash_port/run_matrix.py "$matrix" --no-hub
  fi
  echo "[$(date -u +%H:%M:%S)] lane gpu=$gpu: matrix $matrix done; publishing in the background"
  nohup nice -n 10 .venv/bin/python scripts/dflash_port/push_matrix.py "$matrix" \
    --visibility public > "$logs/push_${name}.log" 2>&1 &
done
echo "[$(date -u +%H:%M:%S)] lane gpu=$gpu: all matrices done"
