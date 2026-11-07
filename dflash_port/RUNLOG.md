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
- outcome: **pending**. At the user's request the work was committed before this run finished. It was still running at commit time; its log is above. How to check or rerun, and the expected result: `dflash_port/REGRESSION.md`.
