#!/usr/bin/env bash
# GENDIAG analyses of existing checkpoints, GPU 1 lane (reading controls).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
export CUDA_VISIBLE_DEVICES=1
O=/workspace/yusuf/runs/dflash/gendiag; E=examples/dflash-lora
run() { echo "[$(date -u +%H:%M:%S)] $*"; .venv/bin/python "$@" || echo "[$(date -u +%H:%M:%S)] FAILED: $*"; }
while pgrep -f "gendiag/controls.py.*abl24k-pt256_val" > /dev/null; do sleep 30; done
for spec in "abl24k-pm256 abl24k $E/abl24k/generator_pm256_kl_r32.yaml" \
            "mix100k-r64 mix100k $E/mix100k/generator_pt256_kl_r64.yaml" \
            "m0-4k tier1 $E/m0/generator.yaml" \
            "abl24k-pt256-r64 abl24k $E/abl24k/generator_pt256_kl_r64.yaml"; do
  set -- $spec
  [ -f $O/controls/$1_val.json ] || run scripts/dflash_port/gendiag/controls.py $3 --tier $2 --out $O/controls/$1_val.json
done
for spec in "abl24k-pt32-r512 abl24k $E/abl24k/generator_pt32_kl_r512.yaml" \
            "abl24k-hybrid-r512 abl24k $E/abl24k/generator_hybrid32_kl_r512.yaml"; do
  set -- $spec
  [ -f $O/controls/$1_val.json ] || run scripts/dflash_port/gendiag/controls.py $3 --tier $2 --per-subset 20 --out $O/controls/$1_val.json
done
# Memorization (E5): the same controls on training rows.
for spec in "abl24k-pt256 abl24k $E/abl24k/generator_pt256_kl_r32.yaml" \
            "abl24k-pm256 abl24k $E/abl24k/generator_pm256_kl_r32.yaml" \
            "m0-4k tier1 $E/m0/generator.yaml"; do
  set -- $spec
  [ -f $O/controls/$1_train.json ] || run scripts/dflash_port/gendiag/controls.py $3 --tier $2 --split train --per-category 50 --quick --out $O/controls/$1_train.json
done
echo "[$(date -u +%H:%M:%S)] QUEUE GPU1 DONE"
