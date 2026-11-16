#!/usr/bin/env bash
# GENDIAG: resume everything after a container restart (or check what is left).
#
#   scripts/dflash_port/gendiag/resume.sh          # dry run: print the plan
#   scripts/dflash_port/gendiag/resume.sh --go     # relaunch lanes + analysis queues
#
# Training runs: a run is done when its run_record_test.json says "completed";
# otherwise run_matrix.py resumes it (training from its last checkpoint.pt,
# saved every 500 updates; completed matrix entries are skipped). Analyses skip
# outputs that exist. Nothing is deleted.
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
R=/workspace/yusuf/runs/dflash/gendiag; L=/workspace/yusuf/runs/s0/gendiag; mkdir -p $L
GPU0=(pm256 pm256-twin pm256-fp32 pm64 pt64)
GPU1=(pt256 pt256-twin pt256-fp32 pt256-fp32-centred pt1024)
done_run() { grep -q '"status": "completed"' "$R/gendiag-generator-attn-r32-2ep-kl-$1/run_record_test.json" 2>/dev/null; }
pending() { for n in "$@"; do done_run "$n" || echo "$n"; done; }
echo "GPU0 pending: $(pending "${GPU0[@]}" | tr '\n' ' ')"
echo "GPU1 pending: $(pending "${GPU1[@]}" | tr '\n' ' ')"
for q in 0 1; do grep -q "QUEUE GPU$q DONE" $L/queue_existing_gpu$q.log 2>/dev/null && echo "existing-checkpoint queue GPU$q: done" || echo "existing-checkpoint queue GPU$q: pending"; done
running=$(pgrep -af "run_matri[x].py|gendiag/lane_queu[e]|queue_existin[g]|queue_snapshot[s]" | wc -l)
echo "processes already running: $running"
[ "${1:-}" = "--go" ] || { echo "(dry run; pass --go to relaunch)"; exit 0; }
[ "$running" -gt 0 ] && { echo "refusing: GENDIAG processes are still running"; exit 1; }
# Credentials live in /root and are lost on restart: W&B / HF must be re-logged first.
[ -f ~/.netrc ] || echo "WARNING: ~/.netrc missing (W&B login); runs log to W&B"
pending "${GPU0[@]}" > $L/queue_gpu0.txt
pending "${GPU1[@]}" > $L/queue_gpu1.txt
setsid nohup scripts/dflash_port/gendiag/lane_queue.sh 0 $L $L/queue_gpu0.txt >> $L/lane0.log 2>&1 < /dev/null &
setsid nohup scripts/dflash_port/gendiag/lane_queue.sh 1 $L $L/queue_gpu1.txt >> $L/lane1.log 2>&1 < /dev/null &
for q in 0 1; do
  grep -q "QUEUE GPU$q DONE" $L/queue_existing_gpu$q.log 2>/dev/null || \
    setsid nohup scripts/dflash_port/gendiag/queue_existing_gpu$q.sh >> $L/queue_existing_gpu$q.log 2>&1 < /dev/null &
done
setsid nohup scripts/dflash_port/gendiag/queue_snapshots.sh 0 "${GPU0[@]}" >> $L/queue_snapshots_gpu0.log 2>&1 < /dev/null &
setsid nohup scripts/dflash_port/gendiag/queue_snapshots.sh 1 "${GPU1[@]}" >> $L/queue_snapshots_gpu1.log 2>&1 < /dev/null &
echo "relaunched; logs in $L"
