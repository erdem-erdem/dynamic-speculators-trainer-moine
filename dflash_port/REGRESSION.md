# Running the regression suite (T1)

Run this after any code change and before GPU runs. Log the result in `RUNLOG.md` before and after.

```bash
cd <repository root>
OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q -p no:randomly \
  tests/unit/generator tests/unit/models tests/unit/convert tests/unit/scripts \
  > /path/to/logs/t1_<sha>.log 2>&1
grep -E "^FAILED|passed|failed" /path/to/logs/t1_<sha>.log | tail -12
sha256sum /path/to/logs/t1_<sha>.log
```

- **Time:** about 20 minutes on the remote box (CPU-bound tests).
- **Don't add `-p no:cacheprovider`:** it conflicts with the project's `--cache-clear` (pytest exits with an argument error, before running anything).
- **Run with the GPUs visible, as above.** Setting `CUDA_VISIBLE_DEVICES=` (CPU only) makes `test_train_attention_100k::test_prepare_only_does_not_launch_gpu_processes` fail, because it rejects an empty device list.
- **Isolating the run:** to test a specific commit while you keep working, run it in a worktree:
  - `git worktree add /tmp/t1 <sha>`
  - `cd /tmp/t1 && ln -s <repo>/datasets datasets`
  - prefix the pytest command with `PYTHONPATH=/tmp/t1/src`.

## Expected result

- **Overall:** all of `tests/unit/generator`, `tests/unit/models` and `tests/unit/convert` pass.
- **Known failures:** 5 in `tests/unit/scripts/test_modal_prepare.py`, all environmental. The `modal` package is not installed in `.venv`, and these tests fail the same way without any port changes.
- **Last full run**, at 903220a, before the review fixes: 974 passed / 10 skipped / 7 failed. That was the 5 environmental failures plus 2 test-double gaps, since fixed. The affected suites then passed with the GPUs visible (115 passed).
- **Any other failure is real.** Investigate it, fix it, and rerun the failing files first:

```bash
.venv/bin/python -m pytest -q <failing test files>
```

## Last result

- **At f8a92f6:** 990 passed / 10 skipped / 5 failed, in 16 minutes. All 5 failures are the environmental `test_modal_prepare` ones. Log: `/workspace/yusuf/runs/s0/t1_after_review.log` on the remote box (sha256 `bfefe6cdf7d1d426…`).
