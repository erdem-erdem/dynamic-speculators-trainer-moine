# Run log (append-only)

Every action that runs code, uses a GPU, or produces artifacts gets an entry, on any machine. Newest entries go at the bottom.
The template is below. Record failures as well as successes. Never edit past entries; add a correction entry instead.

```
## <YYYY-MM-DD HH:MM> <host> — <short title>
- GPU / driver / CUDA:            - git SHA + `git status --short`:
- env: python, torch, vllm, speculators (pip freeze path):
- command(s):
- config + seed:                  - data: release + data.json sha256
- output dir:
- outcome: exit status, key metrics, artifact paths + sha256, anomalies
- next:
```

## 2026-11-05 local (Windows, RTX 4060 8 GB) — docs only
- No code run. Reviewed the presentation, the local trajectory dataset and upstream (`notes/07`).
- Excluded `datasets/` from git via `.git/info/exclude`.
- next: P1 implementation.

## 2026-11-06 remote (workspace-gom-v2-55dfd64787-9zsbx) — session start, environment survey (no code run)
- GPU / driver / CUDA: 2× H100 80GB HBM3, driver 580.173.02, CUDA 13.0 (driver); nvcc at /usr/local/cuda. Host: 224 CPUs, 2 TB RAM, /workspace beegfs 37 TB free, / overlay 1.4 TB free.
- git SHA + `git status --short`: 4bb7d22; ` M dflash_port/SESSION_PROMPT.md` (user edit: MACHINE line). Remotes: only `origin` (erdem-erdem fork); **no `upstream` remote on this box** (D4: no merge, so not needed now).
- env: system python 3.12.3, NGC torch 2.6.0a0 (cu12.6) — **below the repo pin torch>=2.9**; transformers/vllm/speculators not installed. No uv/conda.
- network: huggingface.co, github.com, download.pytorch.org reachable. **pypi.org index times out (IPv4 and IPv6); pypi.ngc.nvidia.com unreachable** → `pip install` from PyPI blocked. This blocks S0/T1 env build.
- data: `datasets/eagle3-qwen3-8b-trajectories-v1` (4.6 GB, git-excluded) has r0001 (558 MB) **and r0002-100k for all 4 categories** (3.8 GB; notes/07 said chat-only on the Windows copy). data.json sha256 (first 16 hex): r0001 math 59edf56d93d16c7b, code af3b2d1df46786f8, stem 84303548406fafe6, chat da2f80c43b4a4b8b; r0002-100k math e84ce9d0c67b1bc7, code eaa8450575061466, stem c96d13528dfe57ab, chat f72a3db6987d1109.
- models: `/workspace/models/Qwen3-8B` present; HF download metadata records revision b968826d9c46dd6066d109eabc6255188de91218 (= pinned target). Weights not yet hash-checked. z-lab DFlash-b16 not downloaded.
- outcome: survey only. Blocker: Python package index access.
- next: resolve package access → S0 on remote (venv torch≥2.9 + `.[dev,generator,lora]` + hs_connectors; separate vLLM venv), T1 baseline unit suite, tiny DFlash forward CPU+GPU.

## 2026-11-06 remote (workspace-gom-v2-55dfd64787-9zsbx) — S0/T1 baseline unit suite (pre-run)
- GPU / driver / CUDA: 2× H100 80GB, driver 580.173.02, torch cu130.   - git SHA + status: 4bb7d22; M RUNLOG.md, M SESSION_PROMPT.md
- env: .venv python 3.12.3, torch 2.13.0+cu130, transformers 5.16.1, peft 0.21.2, datasets 5.0.1, numpy 2.4.6, speculators 0.8.0.dev0 (-e), hs-connectors 0.7.1a0 (-e). Wheels built offline on the Windows box (PyPI blocked here), in ./wheels (git-ignored). pip freeze: /workspace/yusuf/runs/s0/pip_freeze_venv.txt (sha256 1328f3636261970b…). vllm: not yet (.venv-vllm pending wheels).
- command(s): `.venv/bin/pytest tests/unit/generator tests/unit/models tests/unit/convert -q` (rerun: first attempt exit 4, `-p no:cacheprovider` conflicts with addopts `--cache-clear`) → /workspace/yusuf/runs/s0/t1_baseline.log
- config + seed: test defaults.   - data: none
- output dir: /workspace/yusuf/runs/s0
- outcome: exit 0 — **588 passed, 10 skipped, 14 warnings, 0 failed** in 22:14 (pre-existing failures: none). Log sha256 a7467ecbbf06f688….
- next: S0 tiny DFlash smoke.

## 2026-11-06 remote — S0 tiny DFlash forward+backward smoke
- GPU / env / SHA: as above (4bb7d22 + new `scripts/dflash_port/smoke_tiny_dflash.py`).
- command(s): `.venv/bin/python scripts/dflash_port/smoke_tiny_dflash.py --device {cpu,cuda} --attn {eager,simple_flex_attention} --loss-impl {eager,fused} [--weight fixed-exp-decay] [--loss lk_hybrid]` (CPU with `CUDA_VISIBLE_DEVICES=`).
- config + seed: hidden 64, 2 layers, vocab 128, 5 aux layers, block 16, seq 96, max_anchors 4, seed 0.   - data: random tensors
- outcome:
  - FAIL (first attempt, both devices): loss NaN. Cause: `lm_head`/`embed_tokens`/`verifier_lm_head` are verifier-owned and left uninitialised (NaN) without `load_verifier_weights`; script now inits them. Not a model bug.
  - FAIL (CPU, first attempt): `forward` is `conditional_torch_compile`d whenever CUDA is available, and inductor fails for CPU tensors ("0 compatible backends for target (cpu)"); default/fused losses are Triton-only. Fix: `CUDA_VISIBLE_DEVICES=` + `--loss-impl eager`.
  - OK: cpu/eager/ce-eager-dpace loss 3.3719; cuda/eager & cuda/flex × ce-{eager,fused}-dpace loss 3.0441/3.0440 (eager≡fused); cuda/flex/ce-fused/fixed-exp-decay 6.6703. 25 trainable params all receive grads.
  - INCOMPLETE: cuda/flex/lk_hybrid-fused did not finish within 10 min (concurrent with the unit suite); stopped. Recheck at S3.
- next: S1 (family layer + EAGLE-3 extraction). .venv-vllm when the vLLM wheels arrive.

## 2026-11-06 remote — S1 family layer (code only)
- change: new `src/speculators/generator/drafters/{__init__,base,eagle3}.py`; `engine.py`/`data.py`/`campaign_train.py` call the family (check_drafter, call_kwargs, preprocess, trained_length, supervised_tokens, training_noise, score_key, metric_label, identity). Frozen pre-port model copy: `tests/unit/generator/_legacy_model.py` (for S2).
- command: `.venv/bin/pytest tests/unit/generator/test_drafter_families.py -q` → 9 passed.
- outcome: committed with full T1 regression pending (next entry).

## 2026-11-06 remote — S1 T1 regression
- SHA: 0064dec (run in worktree /workspace/yusuf/runs/s1_worktree with PYTHONPATH=<worktree>/src, same .venv).
- command: `pytest tests/unit/generator tests/unit/models tests/unit/convert -q` → /workspace/yusuf/runs/s0/t1_after_s1.log
- outcome: FAIL — 35 failed, 8 errors, 554 passed, 10 skipped (12:06). All 43 = `AttributeError: 'Runtime' object has no attribute 'family'`: test doubles build `Runtime.__new__(Runtime)` without `__init__`.
- fix: `Runtime.family` is now a `cached_property` (pure function of cfg+bank). Rerun of the 9 affected files + family tests: **155 passed, 0 failed** (7:41). Remaining suite was already green at 0064dec → full T1 = 597 passed / 10 skipped expected; full rerun deferred to the S2 regression.

## 2026-11-06 remote — S2 T1 regression (pre-run)
- SHA: c6fa565 (worktree /workspace/yusuf/runs/s2_worktree, PYTHONPATH=<worktree>/src, .venv as S0). New tests: test_generator_layout.py + test_drafter_families.py = 18 passed (1.1 s).
- command: `pytest tests/unit/generator tests/unit/models tests/unit/convert -q` → /workspace/yusuf/runs/s0/t1_after_s2.log
- outcome: exit 0 — **606 passed, 10 skipped, 0 failed** (11:23) = 588 baseline + 18 new. S2 green.

## 2026-11-06 remote — S3 T1 regression + lk_hybrid DFlash smoke (pre-run)
- SHA: e7fd5f4 (worktree /workspace/yusuf/runs/s3_worktree). New S3 tests: 29 passed with S1/S2 family tests (3.5 s).
- commands: (a) `pytest tests/unit/generator tests/unit/models tests/unit/convert -q` → /workspace/yusuf/runs/s0/t1_after_s3.log; (b) after (a): `smoke_tiny_dflash.py --device cuda --attn simple_flex_attention --loss-impl fused --loss lk_hybrid` alone on GPU → /workspace/yusuf/runs/s0/lk_hybrid_smoke.log (S0 attempt did not finish in 10 min while the suite ran)

## 2026-11-06 remote — handoff (session paused by user)
- State: S0–S3 committed; S4 partially implemented and committed as WIP (see PROGRESS "S4 in progress"). Quick check on the WIP tree: family/layout/DFlash/target tests 33 passed.
- Not finished at handoff: S3 full T1 regression and lk_hybrid smoke (background on this box, results only in /workspace/yusuf/runs/s0/ here). Re-run on the next machine.
- Local-only artifacts on this box (never committed): `.venv` (offline wheels in `./wheels`), `/workspace/yusuf/runs/` logs + worktree `/workspace/yusuf/runs/s3_worktree`.

## 2026-11-06 remote — S3 T1 regression + lk_hybrid smoke (results)
- (a) T1 at e7fd5f4: exit 1 — 5 failed, 612 passed, 10 skipped (10:12). Causes (both mine, test-double/legacy shape): `test_original_continuation` ×3 use `streaming_runtime()` (SimpleNamespace runtime without `family`, needed by `Runtime.forward`'s microbatch hook); `test_serving` export ×2 write a legacy identity without `drafter_training` (new DFlash export guard KeyError'd).
- fix: `streaming_runtime` double gets an EAGLE-3 `family`; export guard treats a missing `drafter_training`/`family` as EAGLE-3 (legacy identities). Rerun of the 3 affected files: 61 passed + 9 passed, 0 failed. Full T1 now expected 617 passed / 10 skipped.
- (b) lk_hybrid fused DFlash smoke (cuda, flex, alone on GPU): **OK**, loss 2.9782, all 25 trainable params get grads. S0 "did not finish" was contention with the suite; lk_hybrid works on the DFlash forward.

## 2026-11-07 remote — S4 T1 regression (pre-run)
- host: workspace-gom-v2-55dfd64787-9zsbx, 2× H100 (GPUs idle; suite runs CPU + CUDA tests). SHA: 18da5a4 (worktree /workspace/yusuf/runs/s4_worktree, PYTHONPATH=<worktree>/src, .venv as S0). Seed: test-defined.
- New: test_dflash_kv_path.py = 11 passed (2.7 s, CPU); mutation (block path forced on) → 2 failed, as intended.
- command: `pytest tests/unit/generator tests/unit/models tests/unit/convert -q` → /workspace/yusuf/runs/s0/t1_after_s4.log
- expected: 628 passed / 10 skipped (617 + 11).
- outcome: exit 0 — **628 passed, 10 skipped, 0 failed** (13:16) = 617 (S3, confirms the S3 fix) + 11 new. S4 green.

## 2026-11-07 remote — S5 T1 regression (pre-run)
- host: workspace-gom-v2-55dfd64787-9zsbx, 2× H100 (idle). SHA: 7d566d2 (worktree /workspace/yusuf/runs/s5_worktree, PYTHONPATH=<worktree>/src, .venv as S0). Seed: test-defined.
- New/changed: test_dflash_rigor.py = 9 passed (45 s, CPU); test_dflash_generator.py anchor test extended. Mutations: block-path switch forced (S4) → 2 failed; DFlash padding removed → 3 failed (incl. the original IndexError).
- Probes (scratch, not committed): tiny DFlash campaign.run_one first raised `IndexError: index 6 is out of bounds for dimension 0 with size 6` in DFlashDraftModel._backbone_forward (pack of 6 tokens < block 8) → padding fix. Overfit probe: lr 3e-3 oscillates; lr 1e-3 loss 2.73 → 0.016, full_acc 1.0 by step 120. Note: a CPU run with default threads (32) was ~100× slower on these tiny ops (oversubscription); the overfit test pins 4 threads.
- command: `pytest tests/unit/generator tests/unit/models tests/unit/convert -q` → /workspace/yusuf/runs/s0/t1_after_s5.log
- expected: 637 passed / 10 skipped (628 + 9).
- outcome: exit 0 — **637 passed, 10 skipped, 0 failed** (14:43) = 628 + 9. S5 green. (S7-part tests, test_generator_guards.py = 6 passed locally; full T1 for 8b18e1d deferred to the next regression.)

## 2026-11-07 remote — S6a–S6c regression (pre-run + outcome)
- host: workspace-gom-v2-55dfd64787-9zsbx, 2× H100 (idle; CPU run, `CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=4`). Tree: b5c14b8 (S6a) + uncommitted S6b/S6c (committed next). Seed: test-defined.
- New tests: test_dflash_drafter_agnostic.py 12, test_activation_store.py 6, test_offline_eval.py 7, all passed. Mutations: drop the Runtime verify / drop the weight hash / read mode not raising / ignore the coefficient: each fails ≥1 test.
- command: `pytest -q -p no:randomly tests/unit/generator/ tests/unit/scripts/` (scripts added to the usual T1 set this time).
- outcome: **670 passed, 12 skipped, 6 failed** (17:13). All 6 are in tests/unit/scripts and environmental, not from this work:
  - test_modal_prepare.py ×5: `modal` is not installed in `.venv` (a namespace stub has no `App`); the same 5 fail with S6b/S6c stashed.
  - test_train_attention_100k.py ×1: rejects the empty `CUDA_VISIBLE_DEVICES=` used for this CPU run; the file passes (24) without it.
- tests/unit/generator: 0 failures. Full T1 incl. models/convert at the S6 commits is still to do (planned with S8).

## 2026-11-07 remote — S6d checks
- host: workspace-gom-v2-55dfd64787-9zsbx, 2× H100. Tree: 8575e91 + S6d (committed next).
- Tests: test_offline_eval.py 9 passed (incl. base ≡ initial validation, eagle3 + dflash); test_dflash_run_layer.py 5 passed. `run_matrix.py examples/dflash-lora/matrices/tier1.yaml --dry-run`: 5 entries planned, as intended.
- `scripts/dflash_port/remote_setup.sh` (check mode): .venv speculators OK; .venv-vllm vllm 0.31.0 + hs_connectors OK; r0001 data.json SHA256 match (chat da2f80c4…, code af3b2d1d…, math 59edf56d…, stem 84303548…); **missing: models/qwen3-8b, prepared z-lab drafter**. Env record: output/dflash-lora/env/env-20261107T093046Z.json (git-ignored). The version line was fixed after this run.

## 2026-11-07 remote — full T1 regression before S8 (pre-run)
- host: workspace-gom-v2-55dfd64787-9zsbx, 2× H100. SHA: 903220a (worktree /workspace/yusuf/runs/s8_worktree, PYTHONPATH=<worktree>/src, .venv as S0; `datasets` symlinked from the main checkout). Seed: test-defined. Covers S6a–S7 (b5c14b8, 8575e91, 43e1778, 903220a) and 8b18e1d.
- command: `pytest -q tests/unit/generator tests/unit/models tests/unit/convert tests/unit/scripts` → /workspace/yusuf/runs/s0/t1_after_s7.log (GPUs visible, as earlier T1 runs).
- expected: generator/models/convert all pass; the scripts suite has the known environmental modal ×5 failures (the train_attention_100k failure only appeared with `CUDA_VISIBLE_DEVICES=`).
- outcome (903220a): exit 1 — **974 passed, 10 skipped, 7 failed** (18:26). Log sha256 fab8f334d801687a….
  - test_modal_prepare ×5: known environmental (`modal` not installed).
  - test_launch_vllm::test_launch_inherits_output_and_preserves_statistics_logging: a hand-built `argparse.Namespace` lacks the new `strict_provenance` field (test double; fix pending).
  - test_offline_eval::test_base_only_equals_initial_validation_of_both_arms[eagle3]: ~1e-9 loss differences with CUDA visible. Passes with `TORCH_COMPILE_DISABLE=1` and on CPU. Cause: the EAGLE-3 runtime double in `build_tiny_campaign` skips the uncompiled forward that production `attach_drafter` applies, so adapted and plain graphs are compiled differently. The DFlash variant (production path) is exact. Fix pending (test double mirrors production).
  - generator/models/convert otherwise all pass (incl. 8b18e1d's guards).
- S8 `/code-review high 86cd846~1..HEAD`: 10 findings, verified and reported to the user. Fixes await the user's decisions (see PROGRESS).

## 2026-11-07 remote — review fixes + portability: regression (pre-run)
- host: workspace-gom-v2-55dfd64787-9zsbx, 2× H100 (GPUs visible). Tree: ce07459 + uncommitted review fixes and portability work (committed after this run). Seed: test-defined.
- command: `pytest -q -p no:randomly tests/unit/generator tests/unit/models tests/unit/convert tests/unit/scripts` → /workspace/yusuf/runs/s0/t1_after_review.log
- expected: all pass except the known environmental test_modal_prepare ×5.
- setup check on this box (`setup.sh --target /workspace/models/Qwen3-8B`): wheels, .venv, .venv-vllm (vllm 0.31.0, vendored File backend), r0001 + r0002-100k data.json SHA256 all OK; target revision b968826d verified, linked as models/qwen3-8b; missing only the prepared drafter / machine config. Env record output/dflash-lora/env/env-20261107T105818Z.json.
- outcome: **990 passed, 10 skipped, 5 failed** (16:15). The 5 failures are the known environmental test_modal_prepare (`modal` not installed); there are no other failures. Log sha256 bfefe6cdf7d1d426…. The code under test equals f8a92f6 (committed before the run finished, at the user's request; later edits were docs only).

## 2026-11-09 09:21 remote (workspace-gom-v2-55dfd64787-9zsbx) — P0 step 1: prepare the z-lab DFlash drafter (pre-run)
- GPU: 2× H100 80GB (idle). git SHA: 4188cd5 + uncommitted round-6 work (evaluation rows, one LoRA per row, per-row results, store push/fetch; committed after tests). The preparation code is unchanged since f8a92f6.
- user: the venvs were fixed on 2026-11-09 (before this session's runs).
- command: `scripts/dflash_port/setup.sh --prepare-drafter 9b41424b7109f9c5413454f481b09a82b85333f4`. That SHA is `git ls-remote https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16 HEAD` on 2026-11-09.
- expected: `output/dflash-drafters/z-lab-qwen3-8b-dflash-b16/9b41424b…/` holding preparation.json (source SHA, weight SHA256s, geometry), plus examples/dflash-lora/machine.local.yaml.
- outcome: exit 0, in about 4 minutes. Converted and validated by DFlashConverter ("Validation succeeded", 58 weights). Prepared at `output/dflash-drafters/z-lab-qwen3-8b-dflash-b16/9b41424b7109f9c5413454f481b09a82b85333f4/`.
  - preparation.json: sha256 2e35068a9629d210…; aux layers [2,10,18,26,34]; block 16; mask id 151669; 5 draft layers.
  - Wrote `examples/dflash-lora/machine.local.yaml`.
  - setup check: everything [ok]; the store is [later] (P0). Env record: output/dflash-lora/env/env-20261109T092601Z.json. Log: /workspace/yusuf/runs/s0/p0_prepare_drafter.log.
  - Tests while it ran: test_offline_eval.py 11 passed (including the new round-6 end-to-end test); test_push_store.py 3 passed.
- next: commit round 6, then P0 fill (`fill_store.sh examples/dflash-lora/m0/generator.yaml --gpu 0 --compare 32`) and the full regression on GPU 1.

## 2026-11-09 09:27 remote (workspace-gom-v2-55dfd64787-9zsbx) — P0 step 2: fill the 4K activation store + online ≡ offline check (pre-run)
- GPU: H100 #0 (vLLM extractor, gpu-memory-utilization 0.40). git SHA: 6da2d3c (clean except this RUNLOG entry).
- command: `scripts/dflash_port/fill_store.sh examples/dflash-lora/m0/generator.yaml --gpu 0 --compare 32` → /workspace/yusuf/runs/s0/p0_fill.log
- config:
  - target Qwen/Qwen3-8B@b968826d (models/qwen3-8b);
  - layers [2,10,18,26,34]+36, bf16, seq 8192;
  - drafter pin 9b41424b (used only for the recipe's identity; the extractor serves the target);
  - seed 42.
- data: r0001 (DATA_REVISION 93750ccc…; data.json sha256 as in setup.sh).
  - rows: train + validation of  (4×1000 + 4×100);
  - validation only of  (test) and  (donors): 4×100 each.
  - About 5,200 rows, each with a summary entry and a full entry.
- output: output/dflash-lora/activation-store-4k (provenance/vllm, fills/<time>.json).

## 2026-11-09 09:28 remote — full T1 regression at 6da2d3c (round 6) (pre-run)
- Worktree /workspace/yusuf/runs/r6_worktree (PYTHONPATH=/workspace/yusuf/runs/r6_worktree/src, .venv), `CUDA_VISIBLE_DEVICES=1`, because GPU 0 runs the P0 extractor. Seed: test-defined.
- command: `pytest -q -p no:randomly tests/unit/generator tests/unit/models tests/unit/convert tests/unit/scripts` → /workspace/yusuf/runs/s0/t1_round6.log
- expected: all pass except the 5 environmental test_modal_prepare failures.

## 2026-11-09 remote — affected test suites for round 6 (run on the uncommitted tree; code = 6da2d3c)
- command: `pytest -q -p no:randomly tests/unit/scripts/test_dflash_run_layer.py tests/unit/generator/test_activation_store.py tests/unit/generator/test_campaign.py <files using structure_study/evaluation_sizes>`
- outcome: **93 passed, 1 failed** (7:45). The failure: `test_offline_campaign_equals_online_campaign`, `adaptation/fresh_lora/result.json/checkpoint_bytes` 27386 vs 27450. **Passes alone** (50 s).
  - Cause: the adaptation checkpoint pickles its source path, and the two runs' output roots differ in length (`output` vs `offline`). That can shift the zip padding by 64 bytes, depending on the temp path length.
  - Contributing: a brief `git stash` / `git stash pop` (a lint comparison) ran during the suite and could have exposed old sources for a moment. Don't do that while tests run.
  - Fix: the test ignores `checkpoint_bytes` (tensors are still compared exactly). Committed with this entry.
- **outcome (P0 step 2, first attempt): FAILED, exit 1.** The extractor started and served. Then `load_local_inputs` raised `ValueError: The adapter section is for block drafters only`.
  - Cause: the trajectory-source banks (the EAGLE-3 `trajectory_recipe`, read only for its saved responses) were family-checked with the DFlash run's config (adapter section, block_drafter loss). The tests never combined `trajectory_recipe` with a DFlash adapter.
  - Fix, commit 1abecc0 plus a lint follow-up: `campaign_banks(cfg, check_family=False)` for the source banks only; the training banks are still checked.
  - New test `test_dflash_run_reads_eagle3_trajectory_source`; test_deepspec_training.py 6 passed.
  - Nothing was written to the store except provenance/. Rerun below.
- user 2026-11-09: store visibility **public**. Out-of-distribution responses: same settings as the r0001 responses (temperature 0.7, top-p 0.8, top-k 20, seed 0, max 4096 tokens), thinking off, HumanEval raw prompt.

## 2026-11-09 09:42 remote — P0 step 2, second attempt (pre-run)
- Same command and config as the first attempt. git SHA f1cb69d. Log /workspace/yusuf/runs/s0/p0_fill2.log.

## 2026-11-09 09:49 remote — offline out-of-distribution set (pre-run)
- GPU: H100 #1 (the responses server at 0.40 memory; the regression shares this GPU). git SHA 05d5ad7.
- command: `scripts/dflash_port/build_ood_set.sh --gpu 1 --port 8001 --output output/dflash-lora/ood-v1` → /workspace/yusuf/runs/s0/ood_build.log
- config (user 2026-11-09):
  - prompts: RedHatAI/speculator_benchmarks@2ae86aff, HumanEval (raw prompt), math_reasoning and summarization (chat template, thinking off); 50 each, seed 42, the fork's seeded round-robin over source categories;
  - sampling: the r0001 settings (temperature 0.7, top-p 0.8, top-k 20, seed 0, up to 4096 tokens, context 8192), `--generation-config vllm`;
  - target served by `launch_vllm.py responses --strict-provenance` (provenance in output/dflash-lora/ood-v1.provenance/vllm).
- next: `fill_activation_store.py examples/dflash-lora/m0/generator.yaml --ood output/dflash-lora/ood-v1` (needs the extractor; after P0).
- outcome (full T1 at 6da2d3c): **995 passed, 10 skipped, 5 failed** (20:46). The failures are the 5 known environmental test_modal_prepare ones (`modal` not installed). Log /workspace/yusuf/runs/s0/t1_round6.log (sha256 f67fa6eefc2ce6f0…).
  - Later commits are covered by their own test files: 1abecc0 (test_deepspec_training 6 passed), 05d5ad7 (test_ood_set 4 passed), b1315d8 (test-only).
- outcome (out-of-distribution build): exit 0. output/dflash-lora/ood-v1, manifest + 3 parquet files; humaneval sha256 53ad5751…, math-reasoning 0c662759…, summarization 6ba42351….
  - **Finding:** HumanEval with the raw prompt: **48/50 responses hit the 4096-token cap** (median 4096). Qwen3-8B completes the function, then writes repetitive commentary until the limit.
  - math_reasoning: response length median 224 (108–498), 0 truncated. summarization: median 184 (80–639), 0 truncated.
  - Not yet extracted into the store; waiting for the user's decision on HumanEval.
- **outcome (P0 step 2, second attempt): exit 0**, 670 s.
  - 5,200 rows, 10,400 entries (5,200 summary + 5,200 full), 8.9M tokens, 437.5 GB extracted; store 333 GB on disk. Fill record output/dflash-lora/activation-store-4k/fills/<time>.json (git_sha 87e6b4b).
  - **Online ≡ offline check (32 rows):** full hidden states and token ids are **bit-identical** (max |diff| 0.0). Prompt summaries differ by **max |diff| 4.0**: one bf16 step at the largest summary values (|x| up to 604; above 512 the bf16 spacing is 4). That is batch-dependent rounding in vLLM. Every run reads the same stored summaries, so training and eval are self-consistent.
- user 2026-11-09: out-of-distribution set: **all three benchmarks use the training chat template (thinking off)**, **responses capped at 256 tokens**. HumanEval stays. The first build (ood-v1, raw HumanEval, 4096 cap) is superseded.
- user 2026-11-09: W&B login on this box (key in ~/.netrc, never in the repo). Entity yusuf-erd-erdem-y-ld-z-technical-university, project **dyn-spec-dflash-v1** (replaces dflash-generator-v1; the W&B settings are not part of the configuration digest).

## 2026-11-09 11:09 remote — P2 smoke matrix (pre-run)
- GPU: H100 #0. git SHA: the commit below. W&B on (dyn-spec-dflash-v1); `--no-hub`.
- command: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/dflash_port/run_matrix.py examples/dflash-lora/matrices/p2_smoke.yaml --no-hub` → /workspace/yusuf/runs/s0/p2_smoke.log
- config: p2 recipes (100 updates, validation and checkpoints every 50), round-6 protocol (one LoRA per row, 18 groups per update, 200 fixed validation rows); store read mode; seed 42.

## 2026-11-09 11:11 remote — out-of-distribution set v2 + its store extraction (pre-run)
- GPU: H100 #1. git SHA: the commit below.
- config (user 2026-11-09): every benchmark uses the training chat template (thinking off); responses capped at 256 tokens; otherwise as v1 (r0001 sampling, 50 each, seed 42). Progress lines print an ETA.
- commands → /workspace/yusuf/runs/s0/ood_v2.log:
  - `scripts/dflash_port/build_ood_set.sh --gpu 1 --port 8001 --output output/dflash-lora/ood-v2`
  - `scripts/dflash_port/fill_store.sh examples/dflash-lora/m0/generator.yaml --gpu 1 --compare 0 --ood output/dflash-lora/ood-v2`
- **outcome (P2 smoke and OOD v2, both): interrupted at 11:22 by a container restart** (container back at 13:12; GPUs empty, no processes). Cause unknown: the cgroup counters (oom/oom_kill) reset on restart. Container limit 512 GB; at the cut the training process held ~114 GiB RSS (activation RAM cache, `ram_cache_gib: 120`) and the OOD extractor was starting on GPU 1.
  - P2 smoke: `m0-base` base eval **completed** (score 0.5246). Generator training reached update 100 and finished its validation (scores at 0/50/100: 0.5246 / 0.5222 / 0.5265; update-0 equals the base eval exactly), then died while saving the update-100 checkpoint. The update-50 checkpoint is intact: loads, progress 50, 2.197B bf16 params all finite, optimizer + RNG state present; no `.pending` left (saves are atomic). `run_record_train.json` still says running → a matrix rerun skips the base entry and resumes from update 50. Static LoRA and both offline evals had not started.
  - OOD v2: build **completed** (manifest above). The store extraction had not written any entry (no store file newer than the extractor start).
  - Provenance finding: every extractor launch rewrites `<store>/provenance/vllm`. The OOD launch (11:20) replaced the 4K fill's `vllm_command.txt` and so on; the 4K fill's own launch record is lost. The extractor identity (`extractor.json`: vLLM build, target-weight hashes) is enforced and unchanged, so store contents are unaffected. Fixed next: each fill copies the launch provenance to `fills/<time>.provenance/`.
  - Not a bug: `# Git SHA: unknown` in `vllm_command.txt` is the **vLLM** source checkout SHA (`launch_vllm._git_sha(vllm_root)`); a wheel install has none, so the build is in `vllm_install.txt`. Our repo SHA is in the fill record.

## 2026-11-09 13:40 remote — fix: per-fill copy of the extractor provenance
- `fill_activation_store.py` copies `<store>/provenance/vllm` to `<store>/fills/<time>.provenance/` before filling; the fill record names it (`extraction_provenance`). Paths read by the store and recipes are unchanged.
- test_activation_store.py: 7 passed (the fill-script test checks the copy). Committed with this entry.

## 2026-11-09 13:45 remote — OOD v2 store extraction, rerun alone (pre-run)
- GPU: H100 #1 (extractor, gpu-memory-utilization 0.40). git SHA 857dcf6 (+ this entry). Run alone first to measure its host-RAM footprint before the P2 smoke runs next to it (container limit 512 GB).
- Memory monitor: `/workspace/yusuf/runs/s0/memwatch.sh` (not committed) → /workspace/yusuf/runs/s0/memwatch.log every 30 s (cgroup current/anon/file/shmem, oom_kill, top RSS processes).
- command: `scripts/dflash_port/fill_store.sh examples/dflash-lora/m0/generator.yaml --gpu 1 --compare 0 --ood output/dflash-lora/ood-v2` → /workspace/yusuf/runs/s0/ood_v2_fill.log
- expected: 150 OOD rows (3 × 50, validation only), summary + full entries; fill record + `fills/<time>.provenance/`.
- **outcome: exit 0**, 92 s. 150 rows (HumanEval, math_reasoning, summarization × 50; validation only), 300 store writes (150 summary + 150 full), 127,552 tokens, 6.27 GB extracted. Fill record `fills/20261109T135207Z.json` (git_sha 8ad9b1f; ood-v2 manifest sha256 4407257f…); launch provenance copied to `fills/20261109T135207Z.provenance/`.
- Memory: container peak **34.7 GB** (extractor + fill), far below the 512 GB limit.

## 2026-11-09 13:55 remote — P2 smoke matrix, resumed (pre-run)
- GPU: H100 #0. git SHA: the commit with this entry. W&B on (dyn-spec-dflash-v1); `--no-hub`. Memory monitor running (memwatch.log); alarm above 400 GB.
- command: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/dflash_port/run_matrix.py examples/dflash-lora/matrices/p2_smoke.yaml --no-hub` → /workspace/yusuf/runs/s0/p2_smoke_resume.log
- expected: base entry skipped (completed); generator resumes from the update-50 checkpoint (updates 51–100 rerun; the earlier attempt's train_command is archived under provenance/attempt-*); then static LoRA (100 updates) and both offline evaluations.
- **outcome (resume attempt 1): FAILED before training**, exit 1, 13:55. Base entry skipped as intended; the generator arm loaded drafter and generator, then `wandb.init` raised `UsageError: No API key configured`. The container restart reset `/root`, so `~/.netrc` (the W&B key) is gone. Nothing trained; the update-50 checkpoint is unchanged; this attempt's train_command is archived under `provenance/attempt-1794232519989358421/`.
  - Action: the user logs in again (`wandb login`); then rerun the same command. Note for every restart: `/root` (W&B key, HF token) does not persist; `/workspace` does.

## 2026-11-09 14:05 remote — P2 smoke matrix, resume attempt 2 (pre-run)
- Credentials restored by the user (W&B in `~/.netrc`, HF token in `~/.cache/huggingface/token`; never in the repo). Checked: W&B user yusuf-erd-erdem, HF user erdem-erdem.
- Same command and expectations as attempt 1 (13:55) → /workspace/yusuf/runs/s0/p2_smoke_resume2.log. GPU H100 #0. git SHA: the commit with this entry.
- **outcome (attempt 2): trainings completed; the generator offline eval FAILED** (matrix exit 1, 14:14). Memory peak 111 GiB RSS (static arm, RAM cache at its cap); container well below 512 GB.
  - Base skipped. Generator resumed from update 50 (W&B resumed the same run). **Its update-100 validation equals the interrupted attempt's exactly** (score 0.5265, eal 7.187): resume is deterministic. Update-100 checkpoint 13.18 GB.
  - Static LoRA: 100 updates; validation 0.5246 → 0.5249 → 0.5284 (eal 7.108 → 7.224). Checkpoint 25.8 MB.
  - Generator eval: reproduction (200 rows) ran, then the M4 K=4 continuity variant raised `ValueError: Expected 1–1 conditioning examples`. Cause: round 6 sets `context.max_examples: 1`, and the K=4 view feeds 4 rows. The round-6 unit test set `evaluation_sizes: [1]` but kept `max_examples: 8`, so it missed this. Nothing was written to offline_eval/.
  - Fix (committed with this entry): the K=4 continuity evaluation lifts the generator's input bound in a scoped context and records `beyond_trained_examples: true`. The example encoder is a set encoder (mean/std/log-count pooling, no per-slot parameters), so the bound is only a check; K=4 rows are still outside the trained K (std pooling is always 0 at K=1), which the report must state. The round-6 test now uses `max_examples: 1` (failed before the fix with the production error); test_offline_eval.py 11 passed.

## 2026-11-09 14:30 remote — P2 smoke matrix, attempt 3: the two offline evaluations (pre-run)
- GPU H100 #0; git SHA: the commit with this entry (fix 5833910). Same command → /workspace/yusuf/runs/s0/p2_smoke_eval.log. Expected: base and both trainings skipped (completed); generator and static offline evals run.

## 2026-11-09 14:32 remote — full T1 regression at 74889d9 (pre-run)
- Worktree /workspace/yusuf/runs/p2_worktree (PYTHONPATH=<worktree>/src, .venv; `datasets` symlinked), `CUDA_VISIBLE_DEVICES=1` (GPU 0 runs the P2 evals). Covers code commits since 6da2d3c: 1abecc0, 05d5ad7, 37f681e, 584ed9c, 857dcf6, 5833910.
- command: `pytest -q -p no:randomly tests/unit/generator tests/unit/models tests/unit/convert tests/unit/scripts` → /workspace/yusuf/runs/s0/t1_74889d9.log
- expected: all pass except the 5 environmental test_modal_prepare failures.
- **outcome (attempt 3): the generator offline eval FAILED at D13** (14:26). Reproduction, M4 and M7 all ran; then `similarity` → `geometry` raised a device mismatch: helper matrices (centring, masks) were made on the CPU, the Gram matrix of the generated factors on cuda:0. The unit tests ran on the CPU only. Nothing was written.
  - Fix (user 2026-11-09: keep it on the GPU): `geometry` makes its helpers on the Gram matrix's device. New test `test_similarity_of_cuda_factors_equals_cpu` (fails without the fix with the production error); test_offline_eval.py 12 passed on GPU.
  - Also noted: these eval failures are all in `offline_eval.py` (S6c, new for this port: M4/M7/D13/D14), not in the EAGLE-3-derived training path, which ran first time.

## 2026-11-09 14:50 remote — P2 smoke matrix, attempt 4: the two offline evaluations (pre-run)
- GPU H100 #0; git SHA: the commit with this entry. Same command → /workspace/yusuf/runs/s0/p2_smoke_eval2.log.
- **outcome (full T1 at 74889d9): 1000 passed, 10 skipped, 5 failed** (23:44). The 5 are the known environmental test_modal_prepare (`modal` not installed). Log sha256 1978764cdff0d561…. 4b3b110 (D13 device fix, after this SHA) is covered by test_offline_eval.py, 12 passed on GPU.
- **outcome (attempt 4): P2 smoke COMPLETE.** All 5 entries completed, no errors (static eval finished 15:05). Results in `reporting/dflash/P2.md`.
  - Generator eval: reproduction 0.52647 vs training 0.52648 (equal to 1e-5, not bitwise; GPU kernels, reported not enforced). Static eval: reproduction **bit-exact**.
  - Finding: the static LoRA (same LoRA for every row) also scores lower at K=4 than at K=1 (0.5189 vs 0.5284), so the K=4 gap is the validation anchor draw over packed rows, not the generator. K=4 numbers compare only with K=4 numbers.

## 2026-11-09 15:10 remote — Tier 1 (M0) in two parallel lanes (pre-run)
- User 2026-11-09: Tier 1 starts after the smoke; results fast, both GPUs. git SHA: the commit with this entry. W&B on (dyn-spec-dflash-v1); `--no-hub` (checkpoints stay local; HF push later with the user's OK; the configuration digest excludes HF push).
- Memory budget: two training processes ≈ 2 × 130 GiB (RAM cache 120 GiB each) ≈ 260 GiB < 512 GB; memwatch + alarm at 400 GB.
- Lane A, GPU 0 → /workspace/yusuf/runs/s0/tier1_laneA.log: `run_arm.py m0/base.yaml --mode base`, then `m0/generator.yaml --mode train`, then `m0/generator.yaml --mode eval`.
- Lane B, GPU 1 → /workspace/yusuf/runs/s0/tier1_laneB.log: `run_arm.py m0/static_lora.yaml --mode train`, then `--mode eval`.
- All with `--tier tier1 --output-root /workspace/yusuf/runs/dflash --machine examples/dflash-lora/machine.local.yaml --no-hub`; the same commands as `run_matrix.py matrices/tier1.yaml`, which afterwards sees every entry completed.
- Config: M0 recipes (4K pool, 1,500 updates, validation every 250, checkpoint every 500; round-6 protocol), seed 42, store read mode.
