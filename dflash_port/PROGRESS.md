# DFlash port — living progress & TODO

> Read this first in every new session. Keep it brief; update as you go.
> Spec: `../../task_description.md` (outside repo). Secondary guide (original, unverified):
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
- [ ] **Now: S0–S8** (checklist at the bottom of `IMPLEMENTATION_PLAN.md`), then P0 + P2 on the remote
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
