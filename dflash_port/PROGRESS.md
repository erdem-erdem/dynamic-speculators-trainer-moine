# DFlash port — living progress & TODO

> Read this first in every new session. Keep it brief; update as you go.
> Spec: `starting_docs/task_description.md`. Secondary guide (original, unverified):
> `../../Porting the dynamic-speculators-trainer LoRA Generator from EAGLE-3 to a DFlash Drafter.md`.

## Current state (2026-11-09): start here
- **Code is complete through S8, plus round 6** (2026-11-09; `EXPERIMENT_MATRIX.md` §4):
  - 50 fixed rows per category (200) for every evaluation;
  - an untouched test set (`*-00001`) and an unseen donor pool (`*-00002`);
  - **one LoRA per row** in validation and training (training: groups of 1 row, 18 per update);
  - per-row results in every offline eval, and the M4 per-row coefficient table;
  - store push/fetch (`push_store.py`, `setup.sh --fetch-store`);
  - one brief report per experiment in `reporting/dflash/`.
- **P0 done** (2026-11-09): z-lab drafter `z-lab/Qwen3-8B-DFlash-b16@9b41424b` prepared; 4K store filled (5,200 rows, 333 GB; online ≡ offline bit-identical for full states). OOD set v2 built (`output/dflash-lora/ood-v2`), store extraction pending.
- **2026-11-09 11:22 container restart** interrupted the P2 smoke (base eval done; generator resumable from update 50) and the OOD v2 extraction (nothing written). Details in RUNLOG. Container limit 512 GB: budget host RAM before running jobs in parallel.
- **To run on any GPU box:** `examples/dflash-lora/QUICKSTART.md` (setup → P0 store fill → P2 smoke → STOP → Tier 1).
- **Planned experiments, order and status:** `EXPERIMENT_MATRIX.md` (status board at the top; open items there).
- **Regression:** green at aed10b9 (1014 passed; only the 5 environmental `modal` failures). How to rerun: `REGRESSION.md`.
- **P2 smoke done** (2026-11-09, `reporting/dflash/P2.md`); two offline-eval bugs fixed (5833910, 4b3b110).
- **Tier 1 (M0) done** (`reporting/dflash/M0.md`, plus M4, M7, D13, D14): static LoRA best (val 0.5304, test 0.5317, OOD 0.4579); the generator overfits the 4K pool (peak 0.5289 at update 250, 0.5220 at 1,500; base 0.5246); its LoRAs are dynamic and category-aware (D13) but ≈ 7× larger than the static ΔW, and best at half scale (M4); wrong-donor conditioning scores higher than its own (M7). Published: M0 runs, 4K store (`…-store-qwen3-8b-4k@63cdc4df`), OOD set.
- **User 2026-11-09 (delegation):** run every planned 4K tier in order without per-tier STOPs until the 24K data arrives; HF public; W&B; reports; git push to main at milestones (**blocked: 403**, the box's GitHub user `yusuf-gule` has no write access; see RUNLOG 15:18).
- **Tier 2 done** (21:42; `reporting/dflash/M1.md`, `M2.md`, `M5.md`): no generator arm beats its static LoRA; all generators peak ≈ 0.529 at update 250 and then overfit. Least overfitting: KL and lk_hybrid (final 0.5247 = base) and rank 8 (0.5239); more modules or rank end lower. CE + exp-decay γ7 is worst (0.5165). Triton compile cache moved to local disk after BeeGFS EBUSY crashes (RUNLOG 19:25).
- **Tier 3 so far:** D1 (K/V path: no measurable difference) and D2 (layers: static best on all layers; one-layer generators overfit less but stay below their statics) done. D5 anchors256 crashed on a FlexAttention limit (Q_LEN < 128), fixed by an 8-anchor floor on CUDA flex.
- **Tiers 3–4 done** (2026-11-10 06:08): D3 head sharing no loss at 1/5 size; D4 fc no gain; D5 keep α 0.5 / 512 anchors; M6 K-trained generators best in the K=1 view (0.530 at s=1) but still ≤ static; M3 decoder width does not matter (256 ≈ 512). Every planned 4K run is done; nothing is running.
- **Next (user 2026-11-10, awaiting confirmation):** 10K-math static vs small generator comparison (rank 32, coefficient 2, 3 epochs; evals 200 math + 100 each code/stem/chat, OOD doubled).
- **(Done) lanes:** two GPU lanes (`scripts/dflash_port/run_lane.sh`), each publishing its matrices to HF when done:
  - GPU 0: Tier 2 (`tier2_gpu0`) → Tier 3 (`tier3_gpu0`) → Tier 4 (`tier4_gpu0`); log `/workspace/yusuf/runs/s0/lane0.log`;
  - GPU 1: Tier 2 (`tier2_gpu1`) → Tier 3 D2 (`tier3_gpu1`, done) → M6 (`tier3_m6`) → Tier 4 (`tier4_gpu1`) → D5 density arms (`tier3_gpu1` remainder, after the anchor-floor fix 8b954dd; RUNLOG 2026-11-10 00:06); log `/workspace/yusuf/runs/s0/lane1.log`.
  - A rerun of a lane skips completed entries. Memory: `memwatch.log`; container limit 512 GB.
- **ABL24K done (2026-11-14, user mandate 2026-11-13):** `reporting/dflash/ABL24K.md`, page https://claude.ai/artifact/GJDcjp7Wd4K2Egvb22mHxQ. 24K, 2 epochs, 13 arms. Static rank is the biggest lever (r512 +0.046, r1024 +0.050 τ vs r32 on test); every generator is ≈ +0.03 whatever its size (shared heads 0.23B ≈ 1.1B); CE, KL and the new label-gated KL (`dpace-label`) tie in distribution. Next: the user's 100–150K run.
- **24K plan (2026-11-13, proposed, not started; awaiting user go):** `reporting/dflash/experiment_suggestions/plan_24k_2026-11-13.md`. Phases: diagnostics + SVD compression of the 1.1B generator → gated losses (D-PACE gate from the drafter's label probability; today's KL runs gate on exp(−KL)) → generator architecture at ≈ 280M and sizes 30M–440M (cap 500M) → LoRA rank (static, generator at fixed size) → controls + seeds → one 100–150K run.
- **Roadmap (2026-11-12, suggestions only, awaiting user decisions):** `reporting/dflash/experiment_suggestions/roadmap_2026-11-12.md`: objective fix first (PERTURB24K: held-out loss rises while τ rises), G0 diagnostics on the 24K generator, parameter-matched small generators (5–50M) vs static LoRAs, positive controls. Correction: EAGLE-3 generators were 293M–1.45B, not small (the ≈ 3M figure was the static LoRA). PERTURB24K lives only in worktree `.claude/worktrees/agent-aff1c02dc527887a0` (2fc73aa), not yet on main.
- **Next:**
  1. Tier 2–4 reports as each experiment completes (`reporting/dflash/`); Tier 3 M8 is not run (needs code, EXPERIMENT_MATRIX round 7);
  2. git push to main once access is fixed;
  3. Tier 5 (24K / 100K) when the 24K data arrives;
  4. serving check on stock vLLM and P5 (guidellm suite research) later.

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
- [x] 2026-11-07 **S5 rigor suite** (7d566d2; T1 637 passed / 10 skipped / 0 failed):
  - `Runtime.attach_drafter(drafter)` split out of `Runtime.__init__` (pure refactor) so tests attach an in-memory drafter through the production setup path.
  - `engine.check_trainable` (rigor items 2–3) runs before the optimizer is built, in `engine._run_job` and `campaign_train.run_training`. Generator arm: the optimizer holds exactly the generator params and no drafter param requires grad. Static arm: exactly the adapter factors. Raises otherwise.
  - **Finding + fix: DFlash microbatch padding.** Upstream `select_anchors` never anchors in the last `block_size` packed positions. Upstream packs to a fixed 8192, so that tail is normally padding, but generator microbatches pack to the sum of row lengths. Without a fix, the tail of the last row in each microbatch could never be an anchor, and a pack of ≤ `block_size` tokens crashed with `IndexError` (found by the tiny DFlash campaign run). `DFlashFamily.microbatch` now right-pads every packed tensor by `block_size` (loss mask 0, document −1, as `CollateFn`). The anchor count still uses the unpadded length. `check_drafter` records `block_size` and logs the geometry (block, mask, sample_from_anchor, attention, anchor density, validation seed, kv path, loss); `microbatch` raises if `check_drafter` has not run.
  - Note (upstream behaviour, unchanged): with `sample_from_anchor=False`, an anchor on a row's last supervised token supervises nothing.
  - Tests: `test_dflash_rigor.py` covers items 2–4, 6–8, plus a tiny `campaign.run_one` end to end (generator + static LoRA) for both families, via `build_tiny_campaign(..., family)` in `test_campaign.py` and `dflash_fixtures.py`.
- [x] 2026-11-07 **S7 guards** (first part done ahead of S6, rest after S6d):
  - Done:
    - `provenance.require_git_sha`, called by `engine.begin_run` before anything is written, so generator runs (engine and campaign) refuse a git SHA of "unknown". `save_train_command` itself is unchanged (shared with upstream training).
    - `Runtime.attach_drafter` raises on any non-finite drafter tensor (the NaN verifier weights left uninitialised by a checkpoint or conversion).
    - Tests: `test_generator_guards.py` (6).
  - Open:
    - (a) **Done after S6d:** `launch_vllm.py --strict-provenance` (requires `--provenance-dir`; a provenance dir that cannot be created, or any writer failure, raises before vLLM starts; default unchanged, best-effort). Our instructions always pass it, and `fill_activation_store.py` refuses to start without the store's vLLM provenance. Tests: `test_vllm_provenance.py` +4 (28 passed), fill-script test extended. Original note: `scripts/launch_vllm.py::_save_vllm_provenance` is deliberately best-effort (warns, never blocks), and `tests/unit/scripts/test_vllm_provenance.py` pins that. Proposal: add `--strict-provenance` (raise on any provenance failure) and always pass it from our run layer; upstream default unchanged.
    - (b) **Done in S6a** (generic, not z-lab-only). Original note — Local model dirs vs pinned SHAs: the online campaign already requires the drafter's `config.json`/`original_config.json`/`preparation.json`. Proposal: the DFlash prepare step (P0, analogous to `prepare_deepspec_eagle3.py`) writes the z-lab revision + weight SHA256 into `preparation.json`, and the runtime verifies them on load.
    - (c) `output_mismatch: warn` is serving-only → P5.
- [x] 2026-11-07 D11 discussion rounds 1–3 → `EXPERIMENT_MATRIX.md` §4 (decisions D12 drafter-agnostic, D13 LoRA similarity analysis, scale strategy, D2 statics, D14 rank/architecture cost: report both generator strength (acceptance, D13) and end-to-end cost (generator latency, params, merge; tok/s at P5)) and §5 (execution tiers). The D11 discussion is closed; S6 starts.
- [x] 2026-11-07 **S6a drafter-agnostic (D12) + S7(b) drafter pinning**:
  - `generator/drafters/prepare_dflash.py` + CLI `scripts/dflash_port/prepare_dflash_drafter.py`: any DFlash repository at an exact commit SHA → `<output-root>/<sha>/`.
    - External (publisher) format: upstream `DFlashConverter`.
    - Native speculators format (e.g. a RedHat `.dflash` release): loaded as is; verifier-owned tensors filled; every stored tensor must survive loading unchanged, else raise. Finding: upstream `from_pretrained` always reloads a full-vocab DFlash `lm_head` from the verifier, so a stored head that differs (wrong verifier, or a trained head) would be silently discarded; preparation now refuses it.
    - `preparation.json`: source repo/SHA/format, SHA256 of source config + every weight file, converted config/weight SHA256, verifier, filled tensors, geometry (aux layers, block, mask, draft layers, hidden, vocab).
  - `DrafterFamily.verify_local_drafter` hook (EAGLE-3: no-op, unchanged). DFlash: `verify_prepared_drafter` re-hashes weights and checks record ↔ bank (`models.drafter`, `drafter_revision`, `hidden_states.layer_ids[:-1]`) before a local drafter dir (`model_paths`) is loaded. Hub path is unchanged (already pinned by the manifest SHA).
  - Stage label no longer says EAGLE3.
  - Tests `test_dflash_drafter_agnostic.py` (12): external and native prepare; record contents; SHA required; existing output refused; every mismatch (repo, revision, aux layers, weight bytes, missing record) raises; a second, differently shaped drafter (block 4, 2 layers, 2 aux layers) attaches and installs adapters through the production path; Runtime verifies before loading. Mutations (drop the Runtime call; drop the weight hash) each fail a test.
  - Real second checkpoint: P0 (candidate: the RedHat `.dflash` fallback from PLAN D1; to confirm the repo exists and pin its SHA).
- [x] 2026-11-07 **S6b offline activation store**. Design (changed from the earlier "bank-format extraction + local CampaignCorpus" idea, because this reuses the exact online code path and row membership):
  - `generator/activation_store.py`: a persistent, content-addressed store under the online campaign path. Key = digest(target, pinned target revision, layers, dtype, kind full|summary, token ids). It is independent of the drafter, so every arm, ablation and pool (4K, 24K, 100K overlap) reading the same rows shares entries.
  - Config `online.activation_store: {path, mode}`; None keeps today's transient extraction (unchanged).
    - `write`: extract through vLLM when missing, save, and read existing entries.
    - `read`: offline; vLLM is never contacted (`check_server` skipped); a missing entry raises `MissingActivationError`.
  - Stored and fresh payloads pass the same checks (`OnlineActivations._accept`). Publishing uses an exclusive `os.link`, so concurrent writers of the same content (threads or processes) never overwrite.
  - The store's identity is set at its root (`identity.json`); a mismatch raises. It is recorded in the run identity (`campaign.activation_store`); the mode is not, so runs resume across modes.
  - CLI `scripts/dflash_port/fill_activation_store.py RECIPE [--subsets] [--compare N]`: fills every train/validation row (full + prompt summary), then optionally re-extracts N rows bypassing the store and reports bit-identity / max |diff| (the P0 online ≡ offline check). Writes `<store>/fills/<time>.json` with the git SHA.
  - Tests `test_activation_store.py` (6), including the key one: **a full 4-job campaign run offline (vLLM patched to fail) gives the same result metrics and bit-identical checkpoint tensors as the online run that filled the store.** Also: fill once then all hits, content dedup, read-mode errors, identity mismatch / unpinned target, drift detection, CLI. A mutation (read mode does not raise) fails a test.
  - Storage: about 48 KiB/token for full entries (estimate) plus 80 KB per prompt summary; to be measured on the 4K fill at P0.
- [x] 2026-11-07 **S6c offline evaluation tools** (decisions: `EXPERIMENT_MATRIX.md` §4 round 4):
  - `generator/offline_eval.py`:
    - `Variant`: coefficient override (M4) and donor conditioning (M7), applied inside the ordinary validation loop through a `variant=` hook in `campaign_train.evaluate` (None = unchanged training validation). The ablation row set restricts the evaluated rows; subsets with no selected rows are skipped.
    - `similarity` (D13): exact Frobenius Gram of ΔW per module and pooled.
    - `generator_cost` (D14).
    - `evaluate_job(cfg, job, name=...)`: full-size reproduction check, M4 × M7 on the 24-rows-per-category set, D13, D14, written to `<output_root>/offline_eval/<name>/<stage>/<label>.json`.
  - `campaign.run_one(..., evaluate=)` evaluates a completed job without training or writing into its run dir; `require_trained` refuses a missing or differently configured run (the activation-store key is ignored).
  - Tests `test_offline_eval.py` (7): Gram equals materialised products; metrics equal explicit-vector formulas; static LoRA has zero dynamism; clustered categories are separated; coefficient semantics; donor rows; end-to-end on the tiny online campaign (exact reproduction; coefficient 0 identical across donors; run dir byte-identical; refusals). Mutation (ignore coefficient) fails 2 tests.
  - Still to do in S6d: a CLI for `evaluate_job`, wired to the structure-study job used by our recipes.
- [x] 2026-11-07 **S6d recipes + run layer** (decisions: `EXPERIMENT_MATRIX.md` §4 round 5):
  - `examples/dflash-lora/`: `training_recipe.yaml`, `common.yaml`, `m0/{base,generator,static_lora}.yaml` (structure-study jobs, matched arms), `matrices/tier1.yaml`, README. Two `PIN_AT_P0` placeholders (drafter SHA, prepared drafter dir) fail loudly until P0.
  - Base-only evaluation: `Runtime(adapters=False)` / `attach_drafter(adapters=False)` (no adapter modules, no K/V switch); `campaign.run_one(evaluate=, base_only=True)`; `offline_eval.evaluate_base`.
  - **Bug found by the equivalence test and fixed:** with `base_only`, `run_one` fell through to `begin_run`, which returned the trained run's result (and would have written into its run dir). Evaluation now never starts or resumes a run.
  - Run layer, in `scripts/dflash_port/`:
    - `run_arm.py` (modes train/base/eval; `--output-root`, `--no-wandb`, `--store-mode`, `--dry-run`; writes `run_record_<mode>.json`). `--mode eval` is the offline-eval CLI.
    - `run_matrix.py` (in order, one process per arm; skips completed entries; stops on the first failure).
    - `remote_setup.sh` (check by default; `--install-vllm`, `--download-target`, `--prepare-drafter SHA`; writes an env record).
  - Tests:
    - `test_offline_eval.py` +2 (base ≡ initial validation, eagle3 + dflash; run dir untouched);
    - `tests/unit/scripts/test_dflash_run_layer.py` (5): every committed matrix/recipe resolves; M0 arms are matched; overrides; bad matrices refused; completed entries skipped.
  - Found during the setup check: Qwen3-8B (target) and the z-lab drafter are **not** on this box (no `models/`, not in the HF cache). P0 downloads them (`remote_setup.sh --download-target`, `--prepare-drafter`).
- [x] 2026-11-07 **S8 review** (`/code-review high 86cd846~1..HEAD`): 10 findings, each verified; fixes approved by the user. Each fix has a test pinning the requirement it protects.
  1. Offline evaluation never writes into a run dir. It no longer calls `resume_identity`; `require_trained` compares read-only, and only the output root and store key may differ.
  2. and 3. The feedback paths (saved-response, online feedback) are not ported to DFlash. They mis-scaled the loss and fixed the anchors. `require_family` now makes them raise for DFlash. M0 was unaffected: `train_update` puts the drafter in train mode, so training anchors are random.
  4. The static arm is evaluated in its trained adapter dtype (`job.adapter_dtype`).
  5. M7 `fixed` donor = the first 4 **training** rows of the first subset: never evaluated rows, one LoRA for every cohort including short ones. The user chose (a) "a subset that isn't evaluated". The 4K corpus loads only `*-00000`, so this uses its training split. **Confirm with the user.**
  6. `run_matrix` refuses a "completed" record whose configuration digest no longer matches. The digest covers the resolved config + training recipe and excludes W&B, HF push and store mode.
  7. `--strict-provenance` refuses `--no-hash-checkpoints` and unhashable checkpoints. It accepts local checkpoints and **HF repos pinned to a commit SHA**: new `--model-revision`/`--spec-revision` for responses/train/eval resolve the exact snapshot, vLLM serves that local copy, `checkpoint_source.txt` records `repo@sha`, and the served name stays the repo id (user 2026-11-07: models/drafters pushed to HF after training must be usable). Wheel installs get `vllm_install.txt` (version + SHA256 of the dist-info RECORD).
  8. The store records its extractor (vLLM version, build, target-weight hashes) from its strict provenance. Write mode never mixes extractors; the run identity includes it.
  9. Not a bug: on GPU, a padded DFlash microbatch (128k + 16) under `simple_flex_attention` matches eager to 3e-6. 10. Not a bug: an inefficiency only; left as is.
  - Also: the full T1 at 903220a showed two test-double gaps (Namespace without `strict_provenance`; the EAGLE-3 runtime double compiled its forward with CUDA visible, unlike production's uncompiled forward). Both fixed; the affected suites pass with the GPU visible (115 passed).
- [x] 2026-11-07 Run layer, machine config: `examples/dflash-lora/machine.local.yaml` (git-ignored; target path, drafter revision, drafter dir) replaces hand-editing `PIN_AT_P0`. `run_arm` writes `<run>/inputs/training_recipe.resolved.yaml` and refuses unresolved pins on real runs (a dry-run reports them). Also `--machine`, `--no-hub` (StructureStudy `hub_push`, default on). `fill_activation_store.py` resolves structure-study recipes the same way (default subsets = the job's).
- [x] 2026-11-07 **Portability: clone → setup → P0 → train/eval** (user: any GPU workspace). **Start at `examples/dflash-lora/QUICKSTART.md`.**
  - `scripts/dflash_port/setup.sh` (replaces remote_setup.sh): `--fetch-wheels` (HF `erdem-erdem/dyn-spec-train-vllm-wheels-fixed-vllm`@44f65461), `--create-envs` (offline; `.venv` from wheels/train; `.venv-vllm` **guidellm 0.7.1 first, then vllm 0.31.0**, user caveat), `--fetch-data r0001|r0002-100k` (HF `azeeer/eagle3-qwen3-8b-trajectories-v1`@93750ccc; data.json SHA256s pinned), `--target DIR` (existing copy; HF metadata must show revision b968826d) or `--download-target`, `--prepare-drafter SHA` (writes `machine.local.yaml`). Check mode by default.
  - `scripts/dflash_port/fill_store.sh RECIPE --gpu N --compare N`: the whole P0 fill (extractor with strict provenance → fill → online ≡ offline compare → stop).
  - P2 smoke recipes `examples/dflash-lora/p2/` + `matrices/p2_smoke.yaml` (M0 arms at 100 updates, per PLAN.md P2; run with `--no-hub`).
  - `.gitignore`: `.venv-vllm/`, `/models/`, `/output/`, `machine.local.yaml` (checkpoints, stores and models are never committed).
  - Checked here: setup check mode is all `[ok]` except the drafter. `--target /workspace/models/Qwen3-8B` verified revision b968826d and linked `models/qwen3-8b`. **Correction:** the target *was* on this box (S0 survey); my earlier "target missing" only checked `models/qwen3-8b`. `.venv-vllm` has no real `hs_connectors` (an empty namespace package from the repo dir); extraction uses `launch_vllm.py`'s vendored File backend, reported as such.
  - Not tested here (would replace the working envs): `--create-envs`, `--fetch-wheels`, `--fetch-data`. First run on a new box = their test.
- [ ] Later, tier by tier: Tier 2–6 recipes and matrices, written after each tier's settings are confirmed with the user (EXPERIMENT_MATRIX §5).
- [ ] **Then P0** (any GPU box, QUICKSTART §1–2): `setup.sh --prepare-drafter <z-lab SHA>` (only missing item on this box), then `fill_store.sh examples/dflash-lora/m0/generator.yaml --compare 32` → P2 smoke (`matrices/p2_smoke.yaml --no-hub`) → **STOP and report** → Tier 1 (`matrices/tier1.yaml`).
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
