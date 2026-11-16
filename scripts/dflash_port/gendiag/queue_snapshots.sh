#!/usr/bin/env bash
# GENDIAG: analyses of the instrumented runs' snapshots, one GPU, after the
# existing-checkpoint queue of that GPU is done and as each run finishes.
#
#   scripts/dflash_port/gendiag/queue_snapshots.sh GPU NAME [NAME ...]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
gpu="$1"; shift
export CUDA_VISIBLE_DEVICES="$gpu" GENDIAG_MEM_FRACTION="${GENDIAG_MEM_FRACTION:-0.3}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/workspace/yusuf/runs/dflash/gendiag; L=/workspace/yusuf/runs/s0/gendiag; E=examples/dflash-lora
run() { echo "[$(date -u +%H:%M:%S)] $*"; .venv/bin/python "$@" || echo "[$(date -u +%H:%M:%S)] FAILED: $*"; }
until grep -q "QUEUE GPU${gpu} DONE" $L/queue_existing_gpu${gpu}.log; do sleep 60; done
for name in "$@"; do
  run_dir=$O/gendiag-generator-attn-r32-2ep-kl-$name
  until [ -f $run_dir/pretraining/$(basename $run_dir)/result.json ]; do sleep 60; done
  recipe=$E/gendiag/generator_$name.yaml
  snaps=$O/diag/$name/snapshots
  mkdir -p $O/snapshots/$name
  [ -f $O/controls/gd-${name}_val.json ] || run scripts/dflash_port/gendiag/controls.py $recipe --tier gendiag --out $O/controls/gd-${name}_val.json
  [ -f $O/direction/gd-${name}_val.json ] || run scripts/dflash_port/gendiag/direction.py $recipe --tier gendiag --per-subset 20 --out $O/direction/gd-${name}_val.json
  for u in 00050 00200 00700 01333 02000; do
    [ -f $snaps/u$u.pt ] || continue
    [ -f $O/snapshots/$name/internals_u$u.json ] || run scripts/dflash_port/gendiag/internals.py $recipe --tier gendiag --snapshot $snaps/u$u.pt --per-category 32 --grad-rows 32 --out $O/snapshots/$name/internals_u$u.json
    [ -f $O/snapshots/$name/controls_u$u.json ] || run scripts/dflash_port/gendiag/controls.py $recipe --tier gendiag --snapshot $snaps/u$u.pt --per-subset 20 --quick --out $O/snapshots/$name/controls_u$u.json
  done
  [ -f $O/internals/gd-$name.json ] || run scripts/dflash_port/gendiag/internals.py $recipe --tier gendiag --per-category 32 --grad-rows 32 --out $O/internals/gd-$name.json
done
echo "[$(date -u +%H:%M:%S)] SNAPSHOT QUEUE GPU${gpu} DONE"
