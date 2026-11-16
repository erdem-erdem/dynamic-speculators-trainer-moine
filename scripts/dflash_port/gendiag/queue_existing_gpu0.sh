#!/usr/bin/env bash
# GENDIAG analyses of existing checkpoints, GPU 0 lane (direction + internals).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
export GENDIAG_MEM_FRACTION=0.3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
O=/workspace/yusuf/runs/dflash/gendiag; L=/workspace/yusuf/runs/s0/gendiag
E=examples/dflash-lora
run() { echo "[$(date -u +%H:%M:%S)] $*"; .venv/bin/python "$@" || echo "[$(date -u +%H:%M:%S)] FAILED: $*"; }
for spec in "abl24k-pt256 abl24k $E/abl24k/generator_pt256_kl_r32.yaml" \
            "abl24k-pm256 abl24k $E/abl24k/generator_pm256_kl_r32.yaml" \
            "mix100k-r64 mix100k $E/mix100k/generator_pt256_kl_r64.yaml" \
            "abl24k-pt256-r64 abl24k $E/abl24k/generator_pt256_kl_r64.yaml" \
            "abl24k-pt128-r128 abl24k $E/abl24k/generator_pt128_kl_r128.yaml" \
            "abl24k-pt32-r512 abl24k $E/abl24k/generator_pt32_kl_r512.yaml" \
            "abl24k-hybrid-r512 abl24k $E/abl24k/generator_hybrid32_kl_r512.yaml" \
            "mix24k-pm256 mix24k $E/mix24k/generator.yaml" \
            "m0-4k tier1 $E/m0/generator.yaml"; do
  set -- $spec
  [ -f $O/internals/$1.json ] || run scripts/dflash_port/gendiag/internals.py $3 --tier $2 --per-category 32 --grad-rows 32 --out $O/internals/$1.json
  [ -f $O/direction/$1_val.json ] || run scripts/dflash_port/gendiag/direction.py $3 --tier $2 --per-subset 20 --out $O/direction/$1_val.json
done
# Point zero (base) for the reference generator; training rows (memorization).
run scripts/dflash_port/gendiag/direction.py $E/abl24k/generator_pt256_kl_r32.yaml --tier abl24k --per-subset 20 --point zero --out $O/direction/abl24k-pt256_val_zero.json
run scripts/dflash_port/gendiag/direction.py $E/abl24k/generator_pt256_kl_r32.yaml --tier abl24k --split train --per-category 50 --out $O/direction/abl24k-pt256_train.json
echo "[$(date -u +%H:%M:%S)] QUEUE GPU0 DONE"
