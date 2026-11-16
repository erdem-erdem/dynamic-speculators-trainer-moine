# GENDIAG: status and resume checkpoint

> Living file. If the box was reset, start here. Last update: 2026-11-16 13:00 UTC.

## What this is

User mandate 2026-11-16: run the collapse-diagnostics plan
(`reporting/dflash/experiment_suggestions/collapse_diagnostics_plan_2026-11-16.md`) end to end, unattended, on 2 H100s.
Priorities:
- what happens inside the generator during training: does it learn per-prompt adaptation, do gradients die, does it collapse;
- shared vs per-module heads;
- generator size;
- whether the head-dominated parameter budget hurts.

Deliverables:
- report `reporting/dflash/GENDIAG.md`;
- a visual results page (Artifact), built by `scripts/dflash_port/gendiag/build_page.py`;
- living docs, commits, pushes.

## Resume after a reset

0. Restore Claude Code state first: `bash /workspace/yusuf/dyn-spec/claude_restore.sh` (backups by `claude_backup.sh`, no secrets; a loop re-ran it every 30 min since 2026-11-16 13:10, log `/workspace/yusuf/runs/s0/claude_backup_loop.log`; restart the loop after a reset).
1. Re-login W&B and HF (`/root` is wiped on restart: `~/.netrc`, `~/.cache/huggingface`).
2. `scripts/dflash_port/gendiag/resume.sh` (dry run: shows the pending runs and queues), then `resume.sh --go`.
   - Training resumes from each run's `checkpoint.pt`, saved every 500 updates; completed matrix entries are skipped.
   - Analyses skip outputs that exist.
   - Caveat: after a resume, `diag.jsonl` repeats updates (the summarizer keeps the last record per update). The movement-from-init metric is measured from the resumed parameters, so it restarts at 0.
3. After the queues finish: rerun the existing-checkpoint queues once more (`queue_existing_gpu{0,1}.sh`), because some GPU-0 jobs OOMed next to training and are skipped only if their output exists. Then build the page and write the report.

## Code (all on main)

| Piece | Path |
|---|---|
| Training-time diagnostics (env `GENDIAG_DIR`, observation only; reproduces ABL24K τ exactly) | `src/speculators/generator/diagnostics.py`, hooks in `campaign_train.py` |
| Context-free twin `architecture.input_mode: constant`; centred heads `architecture.head_centering: true` | `generator/config.py`, `generator/model.py` |
| Tests | `tests/unit/generator/test_gendiag.py` |
| Reading controls (own vs swap / mean / centroid / z̄ / bias-only / coefficient / PCA-k) | `scripts/dflash_port/gendiag/controls.py` |
| Gradient budget + direction test | `scripts/dflash_port/gendiag/direction.py` |
| Internals (stage spreads, gradient by component, head spectra) | `scripts/dflash_port/gendiag/internals.py` |
| Diag summarizer, page builder, page sources | `summarize_diag.py`, `build_page.py`, `page/` |
| Lanes and queues | `lane.sh`, `lane_queue.sh`, `queue_existing_gpu{0,1}.sh`, `queue_snapshots.sh`, `resume.sh` |

## Runs (tier `gendiag`, 24K mixed, 2 epochs = 2,667 updates, KL, r32; ABL24K settings)

Outputs:
- runs: `/workspace/yusuf/runs/dflash/gendiag/<run>/`;
- diagnostics: `…/gendiag/diag/<name>/` (`diag.jsonl`, `snapshots/u*.pt`);
- logs: `/workspace/yusuf/runs/s0/gendiag/`.

| GPU | Name | What | Status |
|---|---|---|---|
| 0 | pm256 | per-module heads, decoder 256, 1.1B | done: val 7.4233 / test 7.4716 (= ABL24K pm256) |
| 0 | pm256-twin | no-input twin of pm256 | running (started 12:41) |
| 0 | pm256-fp32 | pm256, fp32 generator | queued |
| 0 | pm64 | per-module, decoder 64 (≈0.28B) | queued |
| 0 | pt64 | shared, decoder 64 (≈0.06B) | queued |
| 1 | pt256 | shared heads, decoder 256, 0.23B | done: val 7.4221 / test 7.4619 (= ABL24K pt256) |
| 1 | pt256-twin | no-input twin of pt256 | running (started 12:38) |
| 1 | pt256-fp32 | pt256, fp32 generator | queued |
| 1 | pt256-fp32-centred | fp32 + centred heads | queued |
| 1 | pt1024 | shared, decoder 1024 (≈0.9B) | queued |

Each run takes ≈2.7 h with evaluation. Expected end of lanes ≈ 23:30 UTC; analyses ≈ 00:30; report and page ≈ 01:30.

## Analyses

Results go to `/workspace/yusuf/runs/dflash/gendiag/{controls,direction,internals,snapshots}/`.
- **Existing checkpoints** (ABL24K pt256 / pm256 / r64 / r128 / r512 / hybrid, MIX24K, MIX100K r64, M0 4K):
  - controls, GPU 1: pt256, pm256, mix100k done; m0, r64, r512, hybrid and the training-row (memorization) controls are pending;
  - direction, GPU 0: done for 8 checkpoints;
  - internals: only pt256, pm256 and mix24k so far (others OOMed; rerun later).
- **Snapshot queue** (`queue_snapshots.sh`), per run:
  - final controls and direction;
  - internals and quick controls at updates 50 / 200 / 700 / 1333 / 2000.

## Findings so far (measured; details in RUNLOG 2026-11-16)

1. **No per-prompt reading at 24K.** The ABL24K pt256 and pm256 generators score the same with:
   - their own LoRA;
   - a same-category row's LoRA (−0.002 / +0.001 τ);
   - one average LoRA for every row (+0.003 / −0.001);
   - the decoder fed the average input.

   Their gain over static r32 is therefore a fixed LoRA. Halving the per-row part helps pt256 (+0.008, CI excludes 0).
2. **The shared part lives in the head weights, not the biases.** Biases alone give base τ (−0.19). The head weights times the mean decoder output carry ≈ 94% of B's shared part.
3. **100K shows a first, small row-level effect.** Own beats a same-category swap by 0.012 [0.001, 0.023], but the per-row part is too large (× 0.5 gains +0.009).
4. **Per-row signal is tiny.** Split-half reproducible row-specific gradient is 0.6–1.4% of per-row gradient energy, almost all within category. The generator's deviation follows it only weakly (first order: own − same-category donor = +0.0012 for pt256, +0.003 for 100K, ≈0 for r128–r512).
5. **Training dynamics (pt256, pm256):**
   - the trunk gets 0 gradient at update 1 (zero-initialised head weights);
   - the trunk's share of the gradient rises to 54% / 76% by the end;
   - 73–92% of trunk updates are swallowed by bf16 rounding throughout, and ≈36–40% of head-weight updates late in training;
   - dynamism grows steadily (dyn_ratio → 0.11 / 0.09) while validation is flat after update 1,000;
   - train and validation geometry are equal.
