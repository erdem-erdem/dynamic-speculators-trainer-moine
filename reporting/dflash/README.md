# DFlash generator: experiment reports

One brief file per experiment: what is tested, methodology, measurements, runs, results and notes. Fill each one after its runs, with numbers taken from the run outputs (each linked to its `dflash_port/RUNLOG.md` entry).

- **Plan, order and status:** `dflash_port/EXPERIMENT_MATRIX.md`.
- **These files are Markdown records.** The LaTeX report in `reporting/` is updated only on explicit request (`AGENTS.md`).

## Evaluation protocol

<!-- Single source for every report; update here when the protocol changes. -->

- **Status: under discussion (2026-11-09).**
- **Rows:** 50 per category (math, code, stem, chat), 200 in total, as requested on 2026-11-09. How this applies to training validation and offline evaluation is being confirmed.
- **Source:** the held-out validation split of r0001 (`azeeer/eagle3-qwen3-8b-trajectories-v1`, Qwen3-8B@b968826d responses, non-thinking). In-distribution only; out-of-distribution sets exist only in the serving suite (P5).
- **Metrics (offline, teacher-forced):** `full_acc`, per-position acceptance, accept length τ and `eal`, per category; fixed validation anchors (seed 0, density 1/16); cohorts of 4 conditioning rows.
- **Seed:** 42, one seed (EAGLE-3 practice). Run-to-run drift in the EAGLE-3 records is about 5 %.

## Experiments

| ID | Experiment | Tier | Status |
|---|---|---|---|
| [P0](P0.md) | Activation store fill and online-offline check | T-setup | planned |
| [P2](P2.md) | Smoke matrix on the real models | T-setup | planned |
| [M0](M0.md) | Headline: base vs static LoRA vs generator | 1 | planned |
| [M1](M1.md) | Target modules | 2 | planned |
| [M2](M2.md) | LoRA rank | 2 | planned |
| [M3](M3.md) | Generator architecture | 4 | planned |
| [M4](M4.md) | LoRA coefficient at inference | 1 (offline, on the M0 checkpoints) | planned |
| [M5](M5.md) | Training loss | 2 | planned |
| [M6](M6.md) | Conditioning size K | 3 | planned |
| [M7](M7.md) | Conditioning controls (donors) | 1 (offline, on the M0 checkpoint) | planned |
| [M8](M8.md) | Conditioning representation | 3 | planned |
| [M9](M9.md) | Data scale at 1,500 updates | 5 | planned |
| [M10](M10.md) | Updates at 24K | 5 | planned |
| [M11](M11.md) | 100K pool | 5 | planned |
| [D1](D1.md) | K/V LoRA path | 3 | planned |
| [D2](D2.md) | Adapted layers | 3 | planned |
| [D3](D3.md) | Generator head sharing | 3 | planned |
| [D4](D4.md) | fc context-projection adapter | 3 | planned |
| [D5](D5.md) | Anchor density and D-PACE α | 3 | planned |
| [D6](D6.md) | Draft length at inference | 6 | planned |
| [D7](D7.md) | Training block size (flagged) | later | planned |
| [D13](D13.md) | LoRA similarity and dynamism | 1 (with M0), 2 (with M2) | planned |
| [D14](D14.md) | Generator cost | 1, 2, 4 | planned |
| [P5](P5.md) | Serving throughput | 6 | planned |
