# DFlash port — living progress & TODO

> Read this first in every new session. Keep it brief; update as you go.
> Spec: `starting_docs/task_description.md`. Secondary guide (original, unverified):
> `../../Porting the dynamic-speculators-trainer LoRA Generator from EAGLE-3 to a DFlash Drafter.md`.

## Ground rules (from task spec)
- The repo code is the primary source of truth. The original guide is only a list of hypotheses; `GUIDE_CORRECTED.md` replaces it.
- Commit locally at checkpoints as `erdem-erdem <yusuf.erd.erdem@gmail.com>`, using `git -c user.name=... -c user.email=...`. Never push or add remotes; give the user the command instead.
- Stop and report after deliverables 1–2 and before any expensive training run.
- No silent fallbacks; fail loudly. Log git SHA, config, seed, env and data hashes for every run.
- Compute:
  - Local RTX 4060 8 GB (Windows, Python 3.11, torch 2.5.1) is for tiny tests only. `speculators` is NOT importable there: `fcntl`, torch < 2.9, not installed.
  - Full runs **and smoke tests** go on remote H100 80 GB / 96 GB Linux boxes. Keep the code portable.
  - Follow the "Remote protocol" in `PLAN.md`: log every action in `RUNLOG.md`, so another agent on any machine can continue.
- Never commit `datasets/` (≈1 GB local trajectories; excluded via `.git/info/exclude`).

## Key facts (details in FINDINGS.md)
- Fork point: upstream `e3041dd`. Fork author Ahmed Zeer (GitHub `AhmedZeer`, git author `AhmetZeer`). HEAD at the start of the port: `b17276c`.
- The generator is trained "direct" through the frozen drafter loss (`generator/campaign_train.py`).
  - One LoRA per cohort, applied via `functional_call`. Targets are q/k/v/o, rank 32, α 64. Loss `lk_hybrid`.
- DFlash files were never modified by the fork. The same `k_proj`/`v_proj` serve both the context and the block path (`models/dflash/model_definitions.py:116-119`).
- The fork's DeepSeek recipe already extracts layers `[2,10,18,26,34]`+36, which equals the z-lab DFlash aux ids. Reuse it via `trajectory_recipe`.
- The fork's generator margin over a matched static LoRA is small: math acceptance +4–6 pp, chat none. All single seed, ~5 % run-to-run drift. The presentation adds no matched evidence (`notes/07` §A).
- Data: reuse the r0001 trajectories (`datasets/eagle3-qwen3-8b-trajectories-v1`, Qwen3-8B@b968826, non-thinking).
  - Extract activations online at `[2,10,18,26,34]`+36.
  - The EAGLE-3 activation dump (`[2,18,33,36]`) is unusable and not needed.

## Status
- [x] 2026-11-05 Read task + guide; 6 exploration agents → `notes/01..06_*.md`
- [x] 2026-11-05 `FINDINGS.md` (deliverable 1), `GUIDE_CORRECTED.md` (deliverable 2), `PLAN.md`
- [x] 2026-11-05 Report → user decisions:
  - D1 z-lab; D3 remote smoke; D4 upstream added (no merge); D5 reuse trajectories, online activations;
  - vLLM DFlash supported. Presentation, dataset and upstream reviewed → `notes/07`.
- [x] 2026-11-05 `IMPLEMENTATION_PLAN.md` drafted (drafter-family layer, test tiers T0–T3, steps S0–S8). User wants paired EAGLE-3 ↔ DFlash comparisons (plan §5)
- [x] 2026-11-06 Plan approved ("start S0"). Run layer added (§6a: `remote_setup.sh`, per-arm `run.sh`, `run_matrix.py` + matrices). `SESSION_PROMPT.md` written
- [x] 2026-11-06 S0 done on the remote (env from offline wheels; T1 baseline 588 passed, 10 skipped, 0 failed; tiny DFlash forward+backward OK on CPU and GPU, eager + flex attention, CE+D-PACE fused + eager). Gotchas: on CPU set `CUDA_VISIBLE_DEVICES=` (forward is torch.compiled whenever CUDA exists) and use `--loss-impl eager` (fused losses are Triton). `lk_hybrid` fused smoke did not finish in 10 min (ran alongside the suite): recheck at S3. `.venv-vllm` pending wheels (vLLM 0.31.0, DFlash supported per user)
- [x] 2026-11-06 S1 code: `generator/drafters/{base,eagle3}.py`; engine/data/campaign_train call the family; `test_drafter_families.py` (9 pass). T1 after S1: 43 failures from `Runtime.__new__` test doubles, fixed via `Runtime.family` cached_property (affected files 155 passed)
- [x] 2026-11-06 `EXPERIMENT_MATRIX.md` agreed. User decisions (supersede the "paired EAGLE-3 ↔ DFlash, multiple seeds" rule in PLAN/SESSION_PROMPT and IMPLEMENTATION_PLAN §5):
  - D7 **DFlash mirrors exactly** the EAGLE-3 method and the experiments already run for EAGLE-3 (M0–M11); no new EAGLE-3 axes. DFlash-only ablations D1–D6 are extras. No norm-gain adapters.
  - D8 **No EAGLE-3 runs in this project**; compare to existing records (GEN-010/012 for DeepSeek base/static/generator).
  - D9 **One seed (42)**, as EAGLE-3.
  - D10 **Serving port (P5) later**, after method + training are established; until then offline metrics only.
  - Flagged as **important later addition**: D7 training block-size override (`EXPERIMENT_MATRIX.md` §2).
  - D11 (2026-11-07) **Experiment list is discussed at experiment preparation.** `EXPERIMENT_MATRIX.md` is the agreed *scope*, not a run plan. When we reach experiment preparation (S6 recipes/matrices, before P2), the agent first lists every experiment and how it will run (arms, configs, controls, budget, order, metrics) and discusses it with the user; matrices are written only after that.
  - Flagged for later: bank warmup, feedback continuation, prefix refresh/replay, same-category random donor, lr/dropout sweep, concurrency.
- [x] 2026-11-06 S2: generic `install_adapters(targets=)` + `leaf_targets`; `LoRAGenerator(layout, head_sharing)`; family hooks `adapter_targets`/`generator_layout`; legacy-equivalence tests vs `_legacy_model.py`. T1: 606 passed / 10 skipped / 0 failed
- [x] 2026-11-06 S3 code: `drafters/dflash.py` (targets `l{i}_{type}` + `fc`, per-type heads, CE+D-PACE call kwargs, anchor density per microbatch, content-seeded deterministic validation anchors, `full_acc` score); config `bank.training.block_drafter` (all fields required; its presence selects DFlash) and `ExperimentConfig.adapter`; campaign loss contract per family (`campaign_inputs.check_campaign_family`); DFlash metrics routed in tracking; serving export raises for DFlash until P5. `test_dflash_generator.py` (11) incl. zero-LoRA ≡ base bit-identical. T1 regression: green after the fix above
- [x] 2026-11-07 S4: `_project_kv` seam in `models/dflash/model_definitions.py` (behaviour-identical, the only upstream edit); `drafters/dflash_attention.py` (`KVPathDFlashAttention`, `install_kv_path`, installed by `DFlashFamily.prepare_drafter` when `adapter.kv_lora_path != both`); `fc` adapter via `adapter.fc`. `test_dflash_kv_path.py` (11): seamed forward bit-identical to the frozen pre-seam forward (`_legacy_dflash_attention.py`), per-path K/V routing, zero LoRA ≡ base for every path, distinct losses per path, `fc` forward/gradient, install errors; a mutation of the path switch fails 2 tests. Commit 18da5a4; T1 at 18da5a4: **628 passed / 10 skipped / 0 failed** (also confirms the S3 fix)
- [x] 2026-11-06 S3 regression: 5 failures from test doubles/legacy identity shape, fixed (affected files pass; full T1 expected 617 passed / 10 skipped — re-confirm with the S4 regression). `lk_hybrid` fused DFlash smoke OK (loss 2.9782).
- [ ] 2026-11-07 **S5 rigor suite (in progress)**:
  - `Runtime.attach_drafter(drafter)` split out of `Runtime.__init__` (pure refactor) so tests attach an in-memory drafter through the production setup path.
  - `engine.check_trainable` (rigor items 2–3) runs before the optimizer is built, in `engine._run_job` and `campaign_train.run_training`. Generator arm: the optimizer holds exactly the generator params and no drafter param requires grad. Static arm: exactly the adapter factors. Raises otherwise.
  - **Finding + fix: DFlash microbatch padding.** Upstream `select_anchors` never anchors in the last `block_size` packed positions. Upstream packs to a fixed 8192, so that tail is normally padding, but generator microbatches pack to the sum of row lengths. Without a fix, the tail of the last row in each microbatch could never be an anchor, and a pack of ≤ `block_size` tokens crashed with `IndexError` (found by the tiny DFlash campaign run). `DFlashFamily.microbatch` now right-pads every packed tensor by `block_size` (loss mask 0, document −1, as `CollateFn`). The anchor count still uses the unpadded length. `check_drafter` records `block_size` and logs the geometry (block, mask, sample_from_anchor, attention, anchor density, validation seed, kv path, loss); `microbatch` raises if `check_drafter` has not run.
  - Note (upstream behaviour, unchanged): with `sample_from_anchor=False`, an anchor on a row's last supervised token supervises nothing.
  - Tests: `test_dflash_rigor.py` covers items 2–4, 6–8, plus a tiny `campaign.run_one` end to end (generator + static LoRA) for both families, via `build_tiny_campaign(..., family)` in `test_campaign.py` and `dflash_fixtures.py`.
- [ ] **Then: S6–S8** (checklist at the bottom of `IMPLEMENTATION_PLAN.md`), then P0 + P2 on the remote
- [ ] P0 env + baseline reproduction (remote)
- [ ] P1 implementation + rigor tests
- [ ] P2 smoke runs → STOP before P3 (budget approval)
- [ ] P3 main matched comparison · P4 ablations · P5 serving
- [ ] Deliverable 4: exact configs/commands (after P1)

## Doc map
| File | Purpose |
|---|---|
| `PROGRESS.md` | this file: status, todo, rules |
| `FINDINGS.md` | deliverable 1: what the fork adds, synthesized |
| `GUIDE_CORRECTED.md` | deliverable 2: corrected guide + rewritten port & ablation plans |
| `PLAN.md` | phased execution plan, gates, decisions, rigor rules, remote protocol |
| `EXPERIMENT_MATRIX.md` | agreed experiment scope: DFlash mirror of EAGLE-3 experiments (M0–M11) + DFlash-only ablations (D1–D6) |
| `RUNLOG.md` | append-only log of every run/action on any machine |
| `IMPLEMENTATION_PLAN.md` | P1 design: family interface, file layout, test tiers, steps + checklist |
| `SESSION_PROMPT.md` | paste-in prompt to start a new session (after compaction, or on the remote) |
| `notes/01_generator_core.md` | generator model, LoRA injection, loss, frozen params, port seams |
| `notes/02_data_and_campaigns.md` | data pipeline, campaign stages, LoRA bank, CLI, trajectory reuse |
| `notes/03_serving_and_eval.md` | vLLM patch, bundle, metrics, benchmark suite, silent fallbacks |
| `notes/04_experiments_and_results.md` | every recorded experiment and number, with sources |
| `notes/05_upstream_dflash.md` | upstream DFlash model/training/mask/loss, paper facts, seams |
| `notes/06_history_tests_env.md` | commit phases, upstream-file mods, tests, env, hard-coded paths |
| `notes/07_presentation_data_upstream.md` | presentation results, the local trajectory dataset and activations, upstream sync |
