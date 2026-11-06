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
