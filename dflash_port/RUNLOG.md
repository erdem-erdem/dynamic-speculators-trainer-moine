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

## 2026-11-09 14:00 remote — P2 smoke matrix, resume attempt 2 (pre-run)
- Credentials restored by the user (W&B in `~/.netrc`, HF token in `~/.cache/huggingface/token`; never in the repo). Checked: W&B user yusuf-erd-erdem, HF user erdem-erdem.
- Same command and expectations as attempt 1 (13:55) → /workspace/yusuf/runs/s0/p2_smoke_resume2.log. GPU H100 #0. git SHA: the commit with this entry.
- **outcome (attempt 2): trainings completed; the generator offline eval FAILED** (matrix exit 1, 14:14). Memory peak 111 GiB RSS (static arm, RAM cache at its cap); container well below 512 GB.
  - Base skipped. Generator resumed from update 50 (W&B resumed the same run). **Its update-100 validation equals the interrupted attempt's exactly** (score 0.5265, eal 7.187): resume is deterministic. Update-100 checkpoint 13.18 GB.
  - Static LoRA: 100 updates; validation 0.5246 → 0.5249 → 0.5284 (eal 7.108 → 7.224). Checkpoint 25.8 MB.
  - Generator eval: reproduction (200 rows) ran, then the M4 K=4 continuity variant raised `ValueError: Expected 1–1 conditioning examples`. Cause: round 6 sets `context.max_examples: 1`, and the K=4 view feeds 4 rows. The round-6 unit test set `evaluation_sizes: [1]` but kept `max_examples: 8`, so it missed this. Nothing was written to offline_eval/.
  - Fix (committed with this entry): the K=4 continuity evaluation lifts the generator's input bound in a scoped context and records `beyond_trained_examples: true`. The example encoder is a set encoder (mean/std/log-count pooling, no per-slot parameters), so the bound is only a check; K=4 rows are still outside the trained K (std pooling is always 0 at K=1), which the report must state. The round-6 test now uses `max_examples: 1` (failed before the fix with the production error); test_offline_eval.py 11 passed.

## 2026-11-09 14:18 remote — P2 smoke matrix, attempt 3: the two offline evaluations (pre-run)
- GPU H100 #0; git SHA: the commit with this entry (fix 5833910). Same command → /workspace/yusuf/runs/s0/p2_smoke_eval.log. Expected: base and both trainings skipped (completed); generator and static offline evals run.

## 2026-11-09 14:19 remote — full T1 regression at 74889d9 (pre-run)
- Worktree /workspace/yusuf/runs/p2_worktree (PYTHONPATH=<worktree>/src, .venv; `datasets` symlinked), `CUDA_VISIBLE_DEVICES=1` (GPU 0 runs the P2 evals). Covers code commits since 6da2d3c: 1abecc0, 05d5ad7, 37f681e, 584ed9c, 857dcf6, 5833910.
- command: `pytest -q -p no:randomly tests/unit/generator tests/unit/models tests/unit/convert tests/unit/scripts` → /workspace/yusuf/runs/s0/t1_74889d9.log
- expected: all pass except the 5 environmental test_modal_prepare failures.
- **outcome (attempt 3): the generator offline eval FAILED at D13** (14:26). Reproduction, M4 and M7 all ran; then `similarity` → `geometry` raised a device mismatch: helper matrices (centring, masks) were made on the CPU, the Gram matrix of the generated factors on cuda:0. The unit tests ran on the CPU only. Nothing was written.
  - Fix (user 2026-11-09: keep it on the GPU): `geometry` makes its helpers on the Gram matrix's device. New test `test_similarity_of_cuda_factors_equals_cpu` (fails without the fix with the production error); test_offline_eval.py 12 passed on GPU.
  - Also noted: these eval failures are all in `offline_eval.py` (S6c, new for this port: M4/M7/D13/D14), not in the EAGLE-3-derived training path, which ran first time.

## 2026-11-09 14:36 remote — P2 smoke matrix, attempt 4: the two offline evaluations (pre-run)
- GPU H100 #0; git SHA: the commit with this entry. Same command → /workspace/yusuf/runs/s0/p2_smoke_eval2.log.
- **outcome (full T1 at 74889d9): 1000 passed, 10 skipped, 5 failed** (23:44). The 5 are the known environmental test_modal_prepare (`modal` not installed). Log sha256 1978764cdff0d561…. 4b3b110 (D13 device fix, after this SHA) is covered by test_offline_eval.py, 12 passed on GPU.
- **outcome (attempt 4): P2 smoke COMPLETE.** All 5 entries completed, no errors (static eval finished 14:45). Results in `reporting/dflash/P2.md`.
  - Generator eval: reproduction 0.52647 vs training 0.52648 (equal to 1e-5, not bitwise; GPU kernels, reported not enforced). Static eval: reproduction **bit-exact**.
  - Finding: the static LoRA (same LoRA for every row) also scores lower at K=4 than at K=1 (0.5189 vs 0.5284), so the K=4 gap is the validation anchor draw over packed rows, not the generator. K=4 numbers compare only with K=4 numbers.

## 2026-11-09 14:47 remote — Tier 1 (M0) in two parallel lanes (pre-run)
- User 2026-11-09: Tier 1 starts after the smoke; results fast, both GPUs. git SHA: the commit with this entry. W&B on (dyn-spec-dflash-v1); `--no-hub` (checkpoints stay local; HF push later with the user's OK; the configuration digest excludes HF push).
- Memory budget: two training processes ≈ 2 × 130 GiB (RAM cache 120 GiB each) ≈ 260 GiB < 512 GB; memwatch + alarm at 400 GB.
- Lane A, GPU 0 → /workspace/yusuf/runs/s0/tier1_laneA.log: `run_arm.py m0/base.yaml --mode base`, then `m0/generator.yaml --mode train`, then `m0/generator.yaml --mode eval`.
- Lane B, GPU 1 → /workspace/yusuf/runs/s0/tier1_laneB.log: `run_arm.py m0/static_lora.yaml --mode train`, then `--mode eval`.
- All with `--tier tier1 --output-root /workspace/yusuf/runs/dflash --machine examples/dflash-lora/machine.local.yaml --no-hub`; the same commands as `run_matrix.py matrices/tier1.yaml`, which afterwards sees every entry completed.
- Config: M0 recipes (4K pool, 1,500 updates, validation every 250, checkpoint every 500; round-6 protocol), seed 42, store read mode.

## 2026-11-09 14:51 remote — HF publishing: naming scheme; 4K store push (pre-run)
- User 2026-11-09: push to HF, everything public, names that scale to many experiments. Scheme in `EXPERIMENT_MATRIX.md` (prefix `erdem-erdem/dyn-spec-dflash-…`). Store repo renamed in setup.sh / QUICKSTART / push_store.py to `erdem-erdem/dyn-spec-dflash-store-qwen3-8b-4k`.
- Model repos: the existing `push_run` exports a serving bundle, which raises for DFlash until P5; a run-directory publisher is needed (to write next). M0 is pushed after its trainings.
- command (CPU only, `nice -n 10 ionice -c3`, alongside Tier 1): `push_store.py push output/dflash-lora/activation-store-4k --repo erdem-erdem/dyn-spec-dflash-store-qwen3-8b-4k --visibility public` → /workspace/yusuf/runs/s0/push_store.log. The store holds the 4K fill and the OOD v2 rows (two fill records).
- Added `scripts/dflash_port/push_run.py` (run-directory publisher for DFlash runs; QUICKSTART §2). tests/unit/scripts/test_push_run.py 3 passed; dry run on the P2 static run: 48 files, 35 MB.
- Published the OOD v2 set: dataset `erdem-erdem/dyn-spec-dflash-ood-v2` (public) @ **c08eadeb95e31a21a6bb8e3489c1ca62020802ae**: manifest + 3 parquet files, the response server's vLLM provenance under `provenance/vllm/`, README. Local source output/dflash-lora/ood-v2 (manifest sha256 4407257f…).

## 2026-11-09 15:18 remote — git push to main (milestone: P0 + P2 done, Tier 2 recipes): BLOCKED
- User 2026-11-09: push the repo to main at milestones. `git push origin main` → **403**: the box's credential (environment `GIT_ASKPASS`) is GitHub user `yusuf-gule`, which has no write access to `erdem-erdem/dynamic-speculators-trainer-moine`. No credential helper, no `gh`.
- Checked before pushing: the 31 unpushed commits contain no W&B or HF credential.
- Needs the user: add `yusuf-gule` as a collaborator with write access, or provide a token for `erdem-erdem`. Commits continue locally; the push is retried at each milestone.
- **outcome (Tier 1 lane B): completed** 15:40. M0 static LoRA trained (1,500 updates) and evaluated offline; no errors. Results in `reporting/dflash/M0.md` when lane A is done.

## 2026-11-09 15:41 remote — GPU 1 lane: Tier 2 → Tier 3 → M6 → Tier 4 (pre-run)
- User 2026-11-09: run the planned 4K tiers in order without per-tier STOPs (EXPERIMENT_MATRIX round 7). git SHA: the commit with this entry.
- command: `scripts/dflash_port/run_lane.sh 1 /workspace/yusuf/runs/s0/lane1 examples/dflash-lora/matrices/{tier2_gpu1,tier3_gpu1,tier3_m6,tier4_gpu1}.yaml` → /workspace/yusuf/runs/s0/lane1.log (matrix logs inline; HF push logs in /workspace/yusuf/runs/s0/lane1/).
- Each matrix: `run_matrix.py --no-hub` (W&B on); afterwards its completed runs are published (public) in the background. The lane stops at the first failure; a rerun skips completed entries.
- Memory: this lane (≤ ~170 GiB at a checkpoint save of the largest arm) + GPU 0's lane; memwatch + alarm at 400 GB.

## 2026-11-09 15:42 remote — M0 training result: the generator overfits the 4K pool
- Validation (200 fixed rows, score = full_acc / τ): generator 0.5246 → **0.5289 @250** → 0.5275 → 0.5259 → 0.5254 → 0.5238 → **0.5220 @1500** (τ 7.108 → 7.238 → 7.129), ending below base (0.5246). Static LoRA 0.5246 → 0.5300 @250, then flat at 0.530 (0.5304 @1500, τ 7.255).
- Training loss (mean per 250 updates): generator 0.4735 → 0.4667 → 0.4620 → 0.4600 → 0.4521 → 0.4501 (still falling); static 0.4629 → 0.4710 → 0.4691 → 0.4714 → 0.4694 → 0.4706 (flat). Falling training loss with falling validation = overfitting: 1,500 × 18 ≈ 27K row visits ≈ 6.75 epochs of the 4K rows, with a 2.2B-parameter generator (one LoRA per row). Validation rows are disjoint from training (round 6 check).
- Decision (agent, under the user's delegation): keep the agreed protocol (1,500 updates) for every 4K ablation, so arms stay matched with M0 and the EAGLE-3 recipe; validation every 250 updates records each arm's peak and decline, and reports show both. Checkpoints exist at 500 / 1000 / 1500 only (offline evals use the final one). The 24K pool (6× the rows) is the natural test of this; capacity ablations (M2 rank, M3 width) show how it depends on generator size.
- **outcome (store push): exit 0.** Dataset `erdem-erdem/dyn-spec-dflash-store-qwen3-8b-4k` (public) @ **63cdc4df4f26233cecfbd986351368e77a7bffb5**: 10,700 entries (4K fill + OOD v2 rows), 361.2 GB, MANIFEST.json sha256 d26e8c83…. Fetch on another box: `setup.sh --fetch-store 63cdc4df4f26233cecfbd986351368e77a7bffb5` (re-hashes against the manifest).
- **outcome (Tier 1 lane A): completed** 15:50. Base eval, M0 generator trained (1,500 updates) and evaluated offline; no errors. All five Tier 1 records completed.

## 2026-11-09 15:51 remote — GPU 0 lane: M0 publish + final evals → Tier 2 → Tier 3 → Tier 4 (pre-run)
- git SHA: the commit with this entry. command: `scripts/dflash_port/run_lane.sh 0 /workspace/yusuf/runs/s0/lane0 examples/dflash-lora/matrices/{tier1,tier1_final,tier1_static_kgroups,tier2_gpu0,tier3_gpu0,tier4_gpu0}.yaml` → /workspace/yusuf/runs/s0/lane0.log.
- `tier1`: every entry completed (skipped); its push publishes the two M0 runs (`erdem-erdem/dyn-spec-dflash-m0-{generator,static-lora}-attn-r32-4k-1500`, public). `tier1_final`: base / generator / static on the untouched test split and the OOD set. `tier1_static_kgroups`: the M0 static at K = 2 and 8 groupings (M6 references).
- **outcome (GPU 0 lane, first matrices) 16:10:** M0 published: `erdem-erdem/dyn-spec-dflash-m0-generator-attn-r32-4k-1500` @ f6bdfb11 (13.2 GB), `…-m0-static-lora-attn-r32-4k-1500` @ f8739c68. `tier1_final` completed: test base 0.5258 / generator 0.5259 / static 0.5317; OOD 0.4494 / 0.4481 / 0.4579. `tier1_static_kgroups` (K = 2, 8) completed. Tier 2 started on GPU 0 (M1 MLP generator, 4.05B parameters).

## 2026-11-09 16:14 remote — full T1 regression at aed10b9 (pre-run)
- Covers code since 74889d9: 5833910 / 4b3b110 (offline eval), 857dcf6 (fill provenance), push_run / push_matrix / summarize_runs / run_lane, run_arm + run_matrix group_size. Worktree /workspace/yusuf/runs/t1_wt, `CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 nice -n 19` alongside the lanes (tiny test models).
- command: `pytest -q -p no:randomly tests/unit/generator tests/unit/models tests/unit/convert tests/unit/scripts` → /workspace/yusuf/runs/s0/t1_aed10b9.log. Expected: only the 5 environmental test_modal_prepare failures.
- **outcome (full T1 at aed10b9): 1014 passed, 10 skipped, 5 failed** (19:10), alongside both training lanes. The 5 are the known environmental test_modal_prepare. Log sha256 3ebb504229914681….

## 2026-11-09 19:25 remote — M5 kl / lk_hybrid: Triton cache EBUSY on BeeGFS; both lanes stopped
- Lane progress before the stop: GPU 0 finished M1 generator MLP + all and M2 r64; GPU 1 finished M1 static MLP + all and M2 r8 / r16. All are trained and evaluated offline with no errors.
- Both M5 trainings failed at their first loss call, at 19:25 (kl, GPU 0) and 19:26 (lk_hybrid, GPU 1): `OSError: [Errno 16] Device or resource busy: '…/compiler-cache/triton/<key>/tmp.pid_<pid>_…'`. The fused loss's new Triton kernels were being autotuned and compiled, and Triton's `cache.put` calls `os.removedirs(temp_dir)` after the rename. On BeeGFS that rmdir can return EBUSY. The two failures were in different processes with different cache keys, so this is a filesystem behaviour, not a code bug. The records are `failed`; no checkpoint had been written.
- Fix (no configuration change, so digests and completed records stay valid): `output/dflash-lora/compiler-cache/triton` is now a symlink to local overlay disk `/var/cache/dflash-triton`, with the old cache copied there. The BeeGFS copy is kept as `triton.beegfs-old`. `run_lane.sh` recreates the symlink target after a container restart (b9f99a9) and retries a failed matrix once (1271655). The TorchInductor cache stays on BeeGFS; it has not failed.
- A first manual resume of lane 0 (nohup from a tool call) died with its parent shell before training. Both lanes were restarted as tracked background tasks at 19:28 with the remaining matrices (GPU 0: tier2_gpu0, tier3_gpu0, tier4_gpu0; GPU 1: tier2_gpu1, tier3_gpu1, tier3_m6, tier4_gpu1). Completed entries are skipped.
- M1 is complete (all four arms): `reporting/dflash/M1.md`.
- Correction 19:33: the nohup resume of lane 0 had **not** died. It was training kl (pid 326236) when the tracked lane 0 started, and the tracked lane's kl got `filelock Timeout` on the run's `.structure.lock`. The lock did its job: no two processes trained into one run directory. The untracked copy had also been started before b9f99a9 edited `run_lane.sh` (bash reads a script incrementally), so it was stopped at 19:31 at update 0, before any checkpoint. The tracked lane's retry then started kl at 19:32. Lesson: start lanes only as tracked background tasks, and do not edit `run_lane.sh` while a lane runs it.

## 2026-11-09 21:42 remote — Tier 2 done (milestone)
- **outcome:** `tier2_gpu1` completed 20:36 and `tier2_gpu0` completed 21:42; every M1 / M2 / M5 run trained (1,500 updates) and evaluated offline with no further errors. After the Triton cache move, the reruns of M5 KL (GPU 0, 19:32) and lk_hybrid (GPU 1) passed the fused-loss compile.
- Published (public, `push_matrix.py`): from tier2_gpu1, `erdem-erdem/dyn-spec-dflash-m1-static-lora-mlp-r32-4k-1500` @ ae031993, `…-m1-static-lora-all-r32-4k-1500` @ 47d7257e, `…-m2-generator-attn-r8-4k-1500` @ fa05f82b, `…-m2-generator-attn-r16-4k-1500` @ d00540f8, `…-m5-generator-attn-r32-4k-1500-lkhybrid` @ e40fa50f. The tier2_gpu0 push is running (log /workspace/yusuf/runs/s0/lane0/push_tier2_gpu0.log).
- Reports: `reporting/dflash/M1.md`, `M2.md`, `M5.md`. Headline: no generator beats its static LoRA on the 4K pool; KL / lk_hybrid overfit least (final 0.5247 vs CE 0.5220). Tier 3 is running on both lanes.
- **outcome (tier2_gpu0 push), done by 22:13:** `erdem-erdem/dyn-spec-dflash-m1-generator-mlp-r32-4k-1500` @ 527fe989, `…-m1-generator-all-r32-4k-1500` @ 5e85c7a7, `…-m2-generator-attn-r64-4k-1500` @ 90db615e, `…-m5-generator-attn-r32-4k-1500-kl` @ a917b089, `…-m5-generator-attn-r32-4k-1500-ce-expdecay7` @ fc88aab9. Every Tier 2 run is now public. Tier 3 at 22:13: D1 kv-context (GPU 0) and D2 static layer0 (GPU 1) are training; memory 282 GB of 512, oom_kill 0.

## 2026-11-10 00:06 remote — D5 anchors256: FlexAttention NoValidChoicesError; anchor floor (8b954dd); lane 1 reordered
- Tier 3 progress before this: D1 (both K/V-path arms) and D2 (all four arms) completed; reports `reporting/dflash/D1.md`, `D2.md`.
- `d5/generator_anchors256.yaml` (anchor density 1/32) failed at about update 20, in the training forward: `torch._inductor.exc.InductorError: LoweringException: NoValidChoicesError: No choices to select` for `flex_attention` (`src/speculators/models/attention.py:91`, the standalone `torch.compile(flex_attention, dynamic=True)`). The lane's one retry was stopped at 00:09: it would resume from the update-0 checkpoint with the same data order and fail again. The record is `failed`; no checkpoint past update 0.
- Cause, reproduced standalone (DFlash anchor mask, Qwen3 drafter head shapes, forward + backward, fresh compile cache): torch 2.13's compiled FlexAttention has no valid kernel config for **Q_LEN < 128**. Every q_len in {64, 96} fails, including as the first compile; q_len in {128, 192, 352, …, 3408} works. Smaller sparse Q block sizes (16 / 32 / 64) also fail. Q_LEN = anchors × block 16 and anchors = round(density × row length), so at density 1/32 any microbatch shorter than about 248 tokens fails; at 1/16 (M0 and every completed run) only under about 120 tokens. Completed runs never drew such a microbatch, or they would have crashed. About 3% of store rows (159 of 5,350 entries with token ids) are shorter than 248 tokens.
- Fix 8b954dd (+ 4fcdf2c lint): on CUDA with `simple_flex_attention`, the DFlash family draws at least ceil(128 / block_size) = 8 anchors per microbatch (`min_anchors`, printed in the "DFlash drafter:" line). CPU and non-flex paths are unchanged. No-op for every completed run and for density ≥ 1/16 except microbatches under 120 tokens, which none of those runs drew. For D5 anchors256 it raises the anchor count for microbatches under 248 tokens, to be noted in the D5 report. New test `test_cuda_flex_attention_anchor_floor`; a GPU sweep of 120 random D5 shapes with the floor passes. Lanes started after 8b954dd import it, including the running lanes' new runs (D3 onward on GPU 0, M6 on GPU 1); for their recipes (density 1/16) it is a no-op.
- Lane 1 reordered: restarted at 00:10 with `tier3_m6` then `tier4_gpu1`. The D5 density arms (`tier3_gpu1` remainder: anchors256 resumes from its update-0 checkpoint, then anchors1024) are queued to run on GPU 1 when that lane exits (background waiter on its PID). The D5 α arms run on GPU 0 within `tier3_gpu0`.
- Regression: `tests/unit/generator` at 4fcdf2c (CPU, nice 19): **386 passed, 13 skipped, 0 failed** (10 min). Log /workspace/yusuf/runs/s0/t_generator_4fcdf2c.log, sha256 d913609a57fdfb32….

## 2026-11-10 02:41 remote — M6 done; HF push fix (ae71cd0)
- **outcome:** `tier3_m6` completed 02:41 (K = 2 / 4 / 8 generators trained; offline eval in the K = 1 view). Report `reporting/dflash/M6.md`. Also done since 00:31: D3 (`reporting/dflash/D3.md`), D4 (`D4.md`). Lane 1 continues with `tier4_gpu1` (M3 decoder 1024), then the queued D5 density arms.
- The automatic push of `tier3_m6` failed: `KeyError: '1'` in `push_run.model_card`, which read validation context "1"; M6 validates at its own K. Its offline eval record is also `run_record_eval-k1.json`, which `push_run` did not count. Fix ae71cd0: every `run_record_eval*.json` is checked and published, and the card names the grouping. New test; dry run clean. Republished (public): `…-m6-generator-attn-r32-4k-1500-k2` @ bb21a924, `…-m6-generator-attn-r32-4k-1500-k4` @ fc080e50, `…-m6-generator-attn-r32-4k-1500-k8` @ 861fb55d.
- Note: the M0 static repo (`…-m0-static-lora-attn-r32-4k-1500` @ f8739c68) was published before its K = 2 / 8 grouping evals (`tier1_static_kgroups`) existed, so those two eval files are only local, under `/workspace/yusuf/runs/dflash/tier1/m0-static-lora-attn-r32-4k-1500/offline_eval/tier1-k{2,8}`.

## 2026-11-10 04:08 remote — tier3_gpu0 and tier4_gpu1 done and published
- **outcome:** `tier4_gpu1` (M3 decoder 1024) completed 03:49 and `tier3_gpu0` (D1 ×2, D3, D4, D5 α ×2) completed 04:07. Published (public): `erdem-erdem/dyn-spec-dflash-d1-generator-attn-r32-4k-1500-kvctx` @ 88165f77, `…-d1-…-kvblock` @ 75aecaec, `…-d3-…-pertype` @ 4fcf7962, `…-d4-…-fc` @ 05fa0397, `…-d5-…-alpha025` @ eadb7659, `…-d5-…-alpha1` @ 160019d9, `…-m3-generator-attn-r32-4k-1500-decoder1024` @ 7dd14c46.
- The D5 anchors256 rerun (GPU 1, from 03:50, with the 8b954dd anchor floor) passed update 100 at 03:57 with no FlexAttention error; it crashed at about update 20 without the floor. GPU 0 started M3 decoder 256 (`tier4_gpu0`) at 04:08.

## 2026-11-10 06:08 remote — every planned 4K run done (milestone)
- **outcome:** `tier4_gpu0` (M3 decoder 256) completed 05:11 and `tier3_gpu1` (D5 anchors256 rerun with the anchor floor, then anchors1024) completed 06:08; all exit 0. Reports: `reporting/dflash/D5.md`, `M3.md`. All of Tiers 1–4 are done except M8 (needs code). Nothing is running.
- Wording fix: "per-position acceptance" → "per-position accuracy" (`position_k_acc` is judged per position, not conditioned on earlier positions) in README, M5, D6.

## 2026-11-10 06:52 remote — 10K-math comparison: data preparation (pre-run)
- User 2026-11-10: train static LoRA and a small generator on 10K math rows for 3 epochs and evaluate on enlarged sets: validation and test each 200 math + 100 code / stem / chat, OOD 3 × 80 = 240. The generator uses decoder width 256 and condition dim 32 (1.1B parameters at rank 32); attention targets, rank 32, coefficient 2. No donors. Eval rows are stored, so later runs cannot train on them.
- Data: 10,000 math rows = train splits of r0002-100k math-00000..00009 (r0001 holds only 6,000 math train rows).
- **Leak found and avoided:** the trajectory data repeats prompts across rows. 221 r0002-100k train rows share the prompt of a validation row; 6 are identical sequences; some sit in math-00006..00009. Primary ids do not overlap. So eval rows are chosen by content: `scripts/dflash_port/make_eval_splits.py` admits only validation rows whose primary id, prompt tokens and full tokens occur in no train split of any local release (124,000 train rows, r0001 + r0002-100k), one row per prompt, seeded 50 per subset. 305 candidates were excluded. Stored file: `examples/dflash-lora/math10k/eval_splits.json` (validation: math-00010..00013, code-00012/-00016, stem-00010/-00011, chat-00010/-00011; test: math-00014..00017, code-00017/-00018, stem-00012/-00013, chat-00012/-00013; code subsets were chosen with ≥ 50 eligible rows; code repeats prompts most). Verify it any time with `make_eval_splits.py verify`.
- Code 80f9ad2: `structure_study.Pool` (explicit, disjoint subset lists plus the stored rows; pool-less recipes keep identical configuration digests, checked for M0 generator / static, M6 K4, D5 anchors256); `apply_row_selection` selects exactly the stored rows; the fill extracts only the pool's train rows and stored eval rows; `run_arm` evaluates pool jobs with the correct donor only. Tests added (43 passed). 09759bd: `build_ood_set.sh --max-samples`.
- Recipes: `examples/dflash-lora/math10k/{generator,static_lora}.yaml` (1,667 updates = 3 epochs at 18 rows / update; validation every 250 updates; own store `output/dflash-lora/activation-store-math10k`).
- Now: (a) GPU 1: `build_ood_set.sh --gpu 1 --port 8001 --output output/dflash-lora/ood-v3 --max-samples 80` (HumanEval, math_reasoning, summarization × 80 = 240); (b) GPU 0: `fill_store.sh examples/dflash-lora/math10k/generator.yaml --gpu 0 --compare 32` (11,000 rows: 10,000 train + 1,000 eval; online ≡ offline check on 32 rows). Then the OOD rows go into the same store (`--ood output/dflash-lora/ood-v3`). Expected disk ≈ 0.7 TB of 36 TB free. Memory: a fill peaked at 34.7 GB before; memwatch alarm at 400 GB stays on.
- **outcome (data preparation) 07:35:** OOD v3 built (`output/dflash-lora/ood-v3`: HumanEval, math_reasoning, summarization × 80 = 240; responses capped at 256 tokens as in v2); 0 of 240 rows share a prompt or sequence with any of the 124,000 train rows. Store `output/dflash-lora/activation-store-math10k` filled in 22 min: 11,000 rows (10,000 train + 1,000 stored eval rows), 22,000 entries, 20.8M tokens, 895 GB on disk. Online ≡ offline on 32 rows: full hidden states max |diff| 0.0, summary 4.0 (as P0). Then OOD v3 added (480 entries, 10.2 GB). Container memory stayed low (fill peak far below 100 GB). git 97be43d.

## 2026-11-10 07:35 remote — 10K-math trainings + evals (pre-run)
- Matrices `math10k_gpu0.yaml` (base on the new validation / test / OOD v3; then the generator: train, eval, test, OOD) and `math10k_gpu1.yaml` (static LoRA: train, eval, test, OOD), via `run_lane.sh` (W&B on; HF publish in the background after each matrix). git: the commit with this entry.
- Memory: two processes, each with a 120 GiB activation RAM cache (≈ 130 GB per process; ≈ 260 GB total of 512); memwatch alarm stays on.
- Then: publish the store (`erdem-erdem/dyn-spec-dflash-store-qwen3-8b-math10k`) and OOD v3 (`erdem-erdem/dyn-spec-dflash-ood-v3`) to HF.
- Published OOD v3: dataset `erdem-erdem/dyn-spec-dflash-ood-v3` (public) @ **bcba573ce7a2db137b0c860847287417f97a9f40** (manifest + 3 parquet files, vLLM provenance under `provenance/vllm/`, README; manifest sha256 9b311d0b…). Store push started in the background: `push_store.py push output/dflash-lora/activation-store-math10k --repo erdem-erdem/dyn-spec-dflash-store-qwen3-8b-math10k --visibility public --workers 8` (nice 10; log /workspace/yusuf/runs/s0/push_store_math10k.log). Lanes started: GPU 0 `math10k_gpu0`, GPU 1 `math10k_gpu1`. HF token replaced by the user's new fine-grained token (repo.write on erdem-erdem; stored only in ~/.cache/huggingface/token).
- **07:45 generator start failed:** `MissingActivationError` in `persistent_summaries.prepare` (no summary entry for a 745-token row). The generator's prompt summaries covered every train + validation row of each job subset, but a pool's store holds only its training rows and stored eval rows. The read-only store refused correctly; no training happened. The lane's retry started 2 s before the fix landed (it might have imported mixed code), so it was stopped before training. Fix 08935cb: pool jobs summarize only the rows the run uses (training subsets: train rows; eval subsets: stored rows), in shards named by those rows; pool-less jobs are unchanged; new test. Lane 0 restarted 07:50 (base evals skipped as completed). Generator updates started 07:54; summaries were 50 per eval subset as intended. Static LoRA (GPU 1, unaffected) reached its update-250 validation at 07:51.
- Base drafter on the new sets (math10k tier): validation 500 rows full_acc 0.5265 / τ 7.703; test 500 rows 0.5315 / 7.803; OOD v3 240 rows 0.4511 / 5.936.

## 2026-11-10 09:34 remote — 10K-math comparison done
- **outcome:** both lanes exit 0. Static LoRA: trained 84 min, eval / test / OOD done 09:14. Generator: trained 88 min, evals done 09:33. Static reproduces its training validation bit-exactly; the generator to 2.5e-6.
- Test (500 rows; τ category mean; accepted draft tokens per 100 blocks, pooled): base 7.251 / 721; static 7.391 / 737; generator 7.389 / 738. Generator − static: Δτ −0.002, CI [−0.017, +0.012]. OOD v3: base 5.936, static 6.141, generator 6.091; generator − static −0.051, CI [−0.092, −0.010]. Both arms beat base in every category on test (CIs exclude zero); math gains most (+0.20 / +0.22 τ).
- The generator does not overfit (validation flat after update 500), but its generated ΔW are nearly constant: cross-row cosine 0.965, dyn_ratio 0.032, effective rank 1.04 (4K: 0.667 / 0.268 / 3.60).
- Report `reporting/dflash/MATH10K.md`; TLDR updated. HF: static @ 7ab9ffc1; the generator publishes from lane 0; the store upload is still running.
- **outcome (HF publication):** store `erdem-erdem/dyn-spec-dflash-store-qwen3-8b-math10k` (public dataset) @ **db676e0be49df1a13d3029072a66f9a4665efe5b**: 22,397 entries (11,000 rows + OOD v3, summary + full), 966.4 GB, MANIFEST.json sha256 6efbd341…. Generator `erdem-erdem/dyn-spec-dflash-math10k-generator-attn-r32-3ep` @ 57512211; static `…-math10k-static-lora-attn-r32-3ep` @ 7ab9ffc1. Every 10K-math artifact is published.

## 2026-11-10 10:40 remote — 24K mixed comparison (pre-run)
- User 2026-11-10: train static LoRA and the generator on 24K rows, 6K from each of the 4 domains. Settings as the 10K-math run (decoder width 256, condition dim 32, attention targets, rank 32, LoRA coefficient 2, 3 epochs, no donors).
- Data: r0002-100k train splits of subsets 00000..00005 in math / code / stem / chat (24,000 rows); 3 epochs = 24,000 × 3 / 18 = 4,000 updates. Evaluation: the 10K-math stored sets (`math10k/eval_splits.json`: validation and test 200 math + 100 code / stem / chat each; OOD v3). Their rows occur in no train split of any local release, so none is a training row; eval subsets are also disjoint from the train subsets. The base drafter on these rows is the math10k tier's base run.
- Recipes `examples/dflash-lora/mix24k/{generator,static_lora}.yaml`; matrices `mix24k_gpu{0,1}.yaml`; own store `output/dflash-lora/activation-store-mix24k` (≈ 25,000 rows ≈ 2 TB of 35 TB free). Dry runs: 24,000 training examples, 4,000 updates for both arms.
- Now (GPU 0): `fill_store.sh examples/dflash-lora/mix24k/generator.yaml --gpu 0 --compare 32`, then `--ood output/dflash-lora/ood-v3` → /workspace/yusuf/runs/s0/fill_mix24k.log. Then both lanes via `run_lane.sh`.
- In parallel: a read-only audit (subagent) of the EAGLE-3 vs DFlash generator training paths (loss, conditioning, LoRA application, optimization, data alignment), prompted by the 10K-math generator's near-constant ΔW.

## 2026-11-10 12:20 remote — 10K-math generator loss variants (pre-run); 24K deferred
- User 2026-11-10: "24K training will come later" — stopped the mix24k store fill (extractor had just started filling; partial store kept, a rerun fills only missing rows) and the two waiting lanes. Both GPUs idle before this entry.
- Audit (subagent, read-only) of EAGLE-3 vs DFlash generator training: no bug found (per-episode factors, per-row summaries with index check, same optimizer, no regularizers in either). Documented differences: EAGLE-3 loss lk_hybrid (soft, full verifier distribution) + K 1..8 groups vs DFlash CE argmax + K=1. Collapse hypotheses: small headroom; unnormalized last-token conditioning (per-layer summary norms 12/56/85/279/805, the deepest layer dominates); static part absorbed into the dynamic heads; single domain.
- User 2026-11-10: two more generators on the same 10K math data: (1) lk_hybrid (soft KL + TV vs the verifier's full distribution, as in the EAGLE-3 trainings), (2) KL (kl_div). Recipes `math10k/generator_{lkhybrid,kl}.yaml` = `math10k/generator.yaml` with only `training.block_drafter.loss_fn` changed (`math10k/training_recipe_{lkhybrid,kl}.yaml`; same `speculators.losses` functions the EAGLE-3 path uses; D-PACE weights unchanged, as M5). Same store, eval rows, 1,667 updates. Matrices `math10k_loss_gpu0.yaml` (lk_hybrid) and `math10k_loss_gpu1.yaml` (KL) via `run_lane.sh`.
- **outcome 14:05:** both lanes exit 0 (train, eval, test, OOD). lk_hybrid trained 90.7 min, KL 89.2 min; offline reproduction matches training validation to ≈ 3e-5 (not bitwise, as the CE generator).
  - Test (500 rows), Δτ vs the CE static LoRA (category mean, 95% paired bootstrap CI): lk_hybrid +0.016 [+0.001, +0.032]; KL +0.013 [−0.003, +0.028]; CE generator −0.002 [−0.017, +0.012]. Accepted draft tokens per 100 blocks: static 737, lk 739, KL 739.
  - Validation Δτ vs static: lk +0.006 [−0.010, +0.021], KL +0.003 [−0.013, +0.019]. OOD v3: lk −0.039 [−0.078, +0.001], KL −0.027 [−0.066, +0.012], CE −0.051 [−0.092, −0.010].
  - D13: generated ΔW even more constant than CE (cross-row cosine 0.976 lk / 0.989 KL vs 0.965; effective rank 1.02). Best coefficient 2 for both.
  - HF (public): `…-math10k-generator-attn-r32-3ep-lkhybrid` @ 152a8e40, `…-kl` @ fdf8bec6. W&B runs `dflash-math10k-generator-attn-r32-3ep-{lkhybrid,kl}-gseed42`.
  - Report `reporting/dflash/MATH10K_LOSSES.md`; TLDR / README / MATH10K linked. Critique and suggestions: `reporting/dflash/experiment_suggestions/future_suggestions.md`; W&B figure guide `reporting/dflash/dflash-first-eval-graphs/README.md`.

## 2026-11-10 14:35 remote — 24K mixed comparison with KL loss (pre-run)
- User 2026-11-10: static LoRA vs generator on 24K data (6K rows from each of math / code / stem / chat), KL loss; full pipeline (store fill, HF push of the store, trainings, evals, W&B, report).
- The mix24k recipes (8171e95, never run) now use the KL loss for both arms: `mix24k/training_recipe.yaml` = `math10k/training_recipe_kl.yaml` (identical body). Names / repos gain `-kl`: `mix24k-{generator,static-lora}-attn-r32-3ep-kl`. Everything else as 8171e95: r0002-100k subsets 00000..00005 per category, 4,000 updates (3 epochs), decoder width 256, condition dim 32, attention targets, rank 32, coefficient 2; eval on the math10k stored sets and OOD v3 (base = math10k tier base run).
- Data prep: `fill_store.sh examples/dflash-lora/mix24k/generator.yaml --gpu 0 --compare 32`, then `--ood output/dflash-lora/ood-v3`, into `output/dflash-lora/activation-store-mix24k` (resumes the 12:06 partial fill; only missing rows are extracted) → /workspace/yusuf/runs/s0/fill_mix24k.log. Then lanes `mix24k_gpu0` (generator) and `mix24k_gpu1` (static) via `run_lane.sh`, and the store upload to `erdem-erdem/dyn-spec-dflash-store-qwen3-8b-mix24k` (public).
- **Correction 14:40 (user 2026-11-10):** the static LoRA keeps **CE** ("the same static lora config we ran", confirmed when asked); only the generator uses KL. So: `mix24k-static-lora-attn-r32-3ep` (CE + D-PACE, new `mix24k/training_recipe_ce.yaml` = `math10k/training_recipe.yaml` body) vs `mix24k-generator-attn-r32-3ep-kl` (`mix24k/training_recipe.yaml`, kl_div). The comparison mixes loss and adapter type; a KL static (D6 control) would separate them.
- The 14:28 session was killed before its fill resumed (only its "resume" header reached the log). Store state at 14:31: 55 complete entries from the 12:06 partial fill (entries publish atomically via `.pending` + hard link); one stale `.tmp*` file (205 MB, 12:13) removed. Fill restarted 14:32 on GPU 0 (`--compare 32`, then OOD v3) → /workspace/yusuf/runs/s0/fill_mix24k.log. Dry runs: both arms 24,000 rows, 4,000 updates.
- **Decision 14:50 (user 2026-11-10, after discussing CE vs KL):** **KL for both arms**, so the 24K comparison isolates the adapter type (CE static vs KL generator would mix loss and adapter). Runs: `mix24k-static-lora-attn-r32-3ep-kl` and `mix24k-generator-attn-r32-3ep-kl`, both `mix24k/training_recipe.yaml` (kl_div + D-PACE); `mix24k/training_recipe_ce.yaml` removed (never run). The two lane waiters queued at 14:45 were stopped before launching anything; the store fill kept running.

## 2026-11-10 15:40 remote — 24K mixed comparison (KL): data preparation done, trainings started
- **outcome (store fill):** `output/dflash-lora/activation-store-mix24k` filled 14:39–15:28 (48 min): 25,000 rows (24,000 train + 1,000 stored eval rows; 49,947 new entries + 53 from the 12:06 partial fill), 42.7M tokens, 2.1 TB extracted. Online ≡ offline on 32 rows: full hidden states max |diff| 0.0, summary 4.0 (as P0 / math10k). OOD v3 added (480 entries, 10.2 GB). Records `fills/20261110T143910Z.json`, `fills/20261110T153416Z.json`. Transient extractor "Connection error" retries (attempt 1/4) only, as in the math10k fill.
- The first store push refused (`push_store.py`: 2 `*.pending` files). Both came from the fill stopped at 12:13 (pid 953773, gone): one had been hard-linked to its final entry before the kill (link count 2), the other's final entry was rewritten by the 14:39 fill. Both pending names removed; the entries are intact. Push restarted: `nice -n 10 ionice -c3 push_store.py push output/dflash-lora/activation-store-mix24k --repo erdem-erdem/dyn-spec-dflash-store-qwen3-8b-mix24k --visibility public --workers 8` → /workspace/yusuf/runs/s0/push_store_mix24k.log.
- Lanes started 15:35 via `run_lane.sh`: GPU 0 `mix24k_gpu0` (`mix24k-generator-attn-r32-3ep-kl`), GPU 1 `mix24k_gpu1` (`mix24k-static-lora-attn-r32-3ep-kl`); both log `loss=kl_div/dpace`. W&B on; HF publish after each lane. Logs /workspace/yusuf/runs/s0/lane{0,1}/mix24k.log.
- **outcome (HF store) 17:58:** `erdem-erdem/dyn-spec-dflash-store-qwen3-8b-mix24k` (public dataset) @ **d638d564f894ef729ae43201333881dd6290f1de**: 50,480 entries (25,000 rows + OOD v3, summary + full), 1.73 TB, MANIFEST.json sha256 cf4f7876…. Fetch with `setup.sh --fetch-store`, which re-hashes against the manifest.

## 2026-11-10 19:05 remote — 24K mixed comparison (KL) done
- **outcome:** both lanes exit 0 (train, eval, test, OOD). Static trained 172.4 min (18:41 lane end), generator 182.8 min (18:52). Static reproduces its training validation bit-exactly; the generator to ≈ 1e-5 (not bitwise, as before). Container memory peaked ≈ 330 GB of 550 GB (two 141 GB training processes + store push).
- Test (500 rows; τ category mean; accepted draft tokens per 100 blocks, pooled): base 7.251 / 721; static (KL) 7.447 / 741; generator (KL) 7.469 / 743. Generator − static: Δτ +0.022, CI [+0.004, +0.040]. Validation: +0.021, CI [+0.006, +0.037]. OOD v3: +0.003, CI [−0.034, +0.041] (both 407 per 100 blocks; base 390).
- Generator ΔW dynamic again: cross-row cosine 0.795, dyn_ratio 0.155, effective rank 2.48, category centroid accuracy 0.972 (10K-math KL: 0.989 / 0.014 / 1.02 / 0.574). Both arms best at coefficient 2; neither overfits (validation rises to update 4,000).
- vs 10K-math arms on the same test rows: 24K static (KL) +0.055 τ over 10K static (CE), CI [+0.037, +0.076]; gains in code / stem / chat, math unchanged.
- HF (public): static `erdem-erdem/dyn-spec-dflash-mix24k-static-lora-attn-r32-3ep-kl` @ 36c816e5; generator `…-mix24k-generator-attn-r32-3ep-kl` @ 62d897d4. W&B `dflash-mix24k-{static-lora,generator}-attn-r32-3ep-kl-gseed42` (823435ba41b9fecf, e98e30a73e80293c). Store @ d638d564 (above).
- Report `reporting/dflash/MIX24K.md`; TLDR and README updated. Every planned step of this experiment is done; nothing is running.
