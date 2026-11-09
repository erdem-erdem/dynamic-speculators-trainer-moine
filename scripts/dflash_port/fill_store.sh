#!/usr/bin/env bash
# P0: fill an offline activation store for a DFlash recipe, end to end.
#
#   scripts/dflash_port/fill_store.sh examples/dflash-lora/m0/generator.yaml \
#       [--gpu 0] [--compare 32] [--port 8000] [--ood output/dflash-lora/ood-v2]
#
# 1. Reads the recipe (resolved with this box's machine config): target model,
#    extraction layers, store path, shared dir, endpoint.
# 2. Starts the vLLM extractor (.venv-vllm) with --strict-provenance writing to
#    <store>/provenance/vllm (same settings as the EAGLE-3 reference recipe).
# 3. Waits until it serves, fills every train/validation row of the recipe's
#    job (fill_activation_store.py), then re-extracts --compare rows and
#    reports bit-identity / max |diff| (online == offline check).
# 4. Always stops the server. The fill record is <store>/fills/<time>.json.
# Rerunning only extracts missing rows (existing entries are read back).
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
recipe="${1:?usage: fill_store.sh RECIPE [--gpu N] [--compare N] [--port N]}"
shift
gpu=0; compare=32; port=8000; ood=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift ;;
    --compare) compare="$2"; shift ;;
    --port) port="$2"; shift ;;
    --ood) ood="$2"; shift ;;  # fill only the out-of-distribution set's rows
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# Resolve what the server must serve from the recipe itself.
eval "$(.venv/bin/python - "$recipe" "$port" <<'PY'
import shlex, sys
from pathlib import Path
sys.path.insert(0, "scripts/dflash_port")
import run_arm
recipe, port = sys.argv[1], sys.argv[2]
resolved = run_arm.resolve(
    Path(recipe).resolve(), tier="store-fill", output_root=Path("output/dflash-lora/store-fill"),
    wandb=False, store_mode="write", hub=False,
)
if resolved.unresolved:
    raise SystemExit(f"Unresolved {resolved.unresolved}: run setup.sh --prepare-drafter")
cfg, bank = resolved.cfg, resolved.bank
endpoint = cfg.online.endpoint
if f":{port}/" not in endpoint:
    raise SystemExit(f"recipe endpoint {endpoint} does not use port {port}")
values = {
    "TARGET": str(cfg.online.target_model_path),
    "SERVED": bank.models.target,
    "LAYERS": " ".join(str(i) for i in bank.hidden_states.layer_ids[:-1]),
    "SHARED": str(cfg.online.shared_dir),
    "STORE": str(cfg.online.activation_store.path),
}
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"
provenance="$STORE/provenance/vllm"
mkdir -p "$SHARED" "$provenance" output/dflash-lora/logs
log="output/dflash-lora/logs/extractor-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "extractor: target=$TARGET layers=[$LAYERS]+last gpu=$gpu port=$port log=$log"
echo "store: $STORE"

CUDA_VISIBLE_DEVICES="$gpu" .venv-vllm/bin/python scripts/launch_vllm.py train "$TARGET" \
  --target-layer-ids $LAYERS --include-last-layer \
  --hidden-states-backend file --hidden-states-path "$SHARED" \
  --strict-provenance --provenance-dir "$provenance" \
  -- --served-model-name "$SERVED" \
  --host 127.0.0.1 --port "$port" --dtype bfloat16 \
  --max-model-len 8193 --max-num-batched-tokens 8193 --max-num-seqs 128 \
  --gpu-memory-utilization 0.40 --enforce-eager \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  > "$log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT

deadline=$((SECONDS + 1800))
until curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; do
  kill -0 "$server" 2>/dev/null || { tail -40 "$log" >&2; echo "extractor exited" >&2; exit 1; }
  (( SECONDS < deadline )) || { tail -40 "$log" >&2; echo "extractor not ready in 30 min" >&2; exit 1; }
  sleep 5
done
echo "extractor ready; filling"
.venv/bin/python scripts/dflash_port/fill_activation_store.py "$recipe" --compare "$compare" \
  ${ood:+--ood "$ood"}
