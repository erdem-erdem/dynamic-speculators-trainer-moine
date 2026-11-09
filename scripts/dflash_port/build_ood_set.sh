#!/usr/bin/env bash
# Offline out-of-distribution set, end to end (user 2026-11-09):
#   scripts/dflash_port/build_ood_set.sh [--gpu 1] [--port 8001] [--output DIR]
# 1. Starts the target with `launch_vllm.py responses --strict-provenance`
#    (provenance in <output>.provenance/vllm).
# 2. Builds the set: RedHatAI/speculator_benchmarks HumanEval (raw prompt),
#    math_reasoning and summarization (chat template, thinking off), 50 each,
#    r0001 sampling settings (build_ood_set.py, speculators.generator.ood_set).
# 3. Always stops the server.
# Then extract its activations: fill_activation_store.py RECIPE --ood DIR.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
gpu=1; port=8001; output=output/dflash-lora/ood-v1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift ;;
    --port) port="$2"; shift ;;
    --output) output="$2"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ ! -e "$output" ]] || { echo "$output exists; choose a new --output" >&2; exit 1; }
target=models/qwen3-8b
provenance="$output.provenance/vllm"
mkdir -p "$provenance" output/dflash-lora/logs
log="output/dflash-lora/logs/responses-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "responses server: gpu=$gpu port=$port log=$log"
CUDA_VISIBLE_DEVICES="$gpu" .venv-vllm/bin/python scripts/launch_vllm.py responses "$target" \
  --strict-provenance --provenance-dir "$provenance" \
  -- --served-model-name Qwen/Qwen3-8B --host 127.0.0.1 --port "$port" \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.40 \
  --generation-config vllm \
  > "$log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
deadline=$((SECONDS + 1800))
until curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; do
  kill -0 "$server" 2>/dev/null || { tail -40 "$log" >&2; echo "server exited" >&2; exit 1; }
  (( SECONDS < deadline )) || { tail -40 "$log" >&2; echo "server not ready in 30 min" >&2; exit 1; }
  sleep 5
done
echo "server ready; building $output"
.venv/bin/python scripts/dflash_port/build_ood_set.py --output "$output" \
  --endpoint "http://127.0.0.1:${port}/v1" --tokenizer "$target" \
  --provenance-dir "$provenance"
