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
