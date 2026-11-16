#!/usr/bin/env bash
# GENDIAG lane: run gendiag matrices in order on one GPU with training-time
# generator diagnostics on (GENDIAG_DIR per run), then publish each.
#
#   scripts/dflash_port/gendiag/lane.sh GPU LOG_DIR NAME [NAME ...]
#
# NAME = suffix of examples/dflash-lora/matrices/gendiag_<NAME>.yaml.
# Diagnostics: /workspace/yusuf/runs/dflash/gendiag/diag/<NAME>/ (diag.jsonl, snapshots/).
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
gpu="${1:?usage}"; logs="${2:?usage}"; shift 2
mkdir -p "$logs"
triton="output/dflash-lora/compiler-cache/triton"
if [ -L "$triton" ]; then mkdir -p "$(readlink "$triton")"; fi
export GENDIAG_INTERVAL="${GENDIAG_INTERVAL:-20}"
export GENDIAG_SNAPSHOTS="${GENDIAG_SNAPSHOTS:-10,25,50,100,200,400,700,1000,1333,1700,2000,2333,2667}"
export GENDIAG_PROBE_PER_CATEGORY="${GENDIAG_PROBE_PER_CATEGORY:-16}"
for name in "$@"; do
  matrix="examples/dflash-lora/matrices/gendiag_${name}.yaml"
  export GENDIAG_DIR="/workspace/yusuf/runs/dflash/gendiag/diag/${name}"
  echo "[$(date -u +%H:%M:%S)] gendiag lane gpu=$gpu: $matrix (GENDIAG_DIR=$GENDIAG_DIR)"
  if ! CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/dflash_port/run_matrix.py "$matrix" --no-hub; then
    echo "[$(date -u +%H:%M:%S)] gendiag lane gpu=$gpu: $matrix FAILED; retrying once"
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/dflash_port/run_matrix.py "$matrix" --no-hub
  fi
  echo "[$(date -u +%H:%M:%S)] gendiag lane gpu=$gpu: $matrix done; publishing in the background"
  nohup nice -n 10 .venv/bin/python scripts/dflash_port/push_matrix.py "$matrix" \
    --visibility public > "$logs/push_gendiag_${name}.log" 2>&1 &
done
echo "[$(date -u +%H:%M:%S)] gendiag lane gpu=$gpu: all matrices done"
