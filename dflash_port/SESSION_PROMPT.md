# Session-start prompt

Paste the block below as the first message of a new Claude Code session: after compaction, on a new machine, or on the remote GPU box. Run the session from the repo root. Change only the `MACHINE` line.

---

```text
You are continuing the DFlash port of the dynamic LoRA generator in this repo (a fork of vllm-project/speculators).
MACHINE: <local-windows | remote-H100 (meaning here, 2 GPUs)>

Before doing anything:
1. Read dflash_port/PROGRESS.md, which has the rules, status and doc map.
2. Read the tail (last ~5 entries) of dflash_port/RUNLOG.md.
3. Read dflash_port/IMPLEMENTATION_PLAN.md: the checklist at the bottom is the source of truth for the next step, and §6a describes the run layer.
4. Run `git log --oneline -10` and `git status --short`, and reconcile them with PROGRESS/RUNLOG. If they disagree, tell me before acting.
Then tell me in ≤10 lines: where we are, the next step, and what you will run.

Non-negotiable rules:
- Commits are local only, at each checkpoint, with identity
  `git -c user.name=erdem-erdem -c user.email=yusuf.erd.erdem@gmail.com commit ...`
  Never push, never add or change remotes. If pushing is needed, give me the command only.
- Never commit datasets/, .venv-cpu/, checkpoints, caches or activations. Record their paths and SHA256 in RUNLOG.md instead.
- Before and after every run, add an entry to RUNLOG.md using its template: host, GPU, SHA, status, command, config, seed, data hash, outcome. Record failures as well as successes.
- No silent fallbacks: no bare try/except and no default substitutions that hide failures. Fail loudly.
- Rigor: matched arms (base / static LoRA / generator), paired EAGLE-3 ↔ DFlash on the same data, multiple seeds, per-position acceptance and τ per domain, net tok/s at C1 and higher concurrency, and full provenance.
- STOP and report to me before any expensive training run (anything beyond the P2 smoke matrix), and at S8.
- Keep PROGRESS.md, the checklist and RUNLOG.md current, so the next session (possibly on another machine) can continue.

Machine notes:
- local-windows: CPU-only T0 tests in .venv-cpu (`.venv-cpu/Scripts/python.exe -m pytest ...`). Modules that import fcntl (vllm_client, hs_connectors transfer) don't import on Windows, and CUDA/flex tests are skipped.
- remote: run scripts/dflash_port/remote_setup.sh once (after S6 exists; before that, follow PLAN.md P0 by hand). The T1 baseline is the full unit suite before any port code lands. The T2 and T3 tiers are in IMPLEMENTATION_PLAN.md §7.
```
