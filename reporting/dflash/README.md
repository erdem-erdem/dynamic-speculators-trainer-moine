# DFlash generator: experiment reports

One brief file per experiment: what is tested, methodology, measurements, runs, results and notes. Fill each one after its runs, with numbers taken from the run outputs (each linked to its `dflash_port/RUNLOG.md` entry).

- **Plan, order and status:** `dflash_port/EXPERIMENT_MATRIX.md`.
- **These files are Markdown records.** The LaTeX report in `reporting/` is updated only on explicit request (`AGENTS.md`).

## Evaluation protocol

<!-- Single source for every report; update here when the protocol changes. -->

Settled with the user on 2026-11-09 (`dflash_port/EXPERIMENT_MATRIX.md` §4, round 6). Configured in `examples/dflash-lora/common.yaml`.

| Set | Source (r0001, Qwen3-8B@b968826d responses, non-thinking) | Rows | Used for |
|---|---|---|---|
| Validation | `*-00000` validation splits, seeded selection (seed 20261109) | 50 per category, 200 total | training validation, base, every offline eval (reproduction, M4, M7, D13) |
| Test (untouched) | `*-00001` validation splits, same selection rule | 50 per category, 200 total | final numbers only (`run_arm.py --mode test` / `base_test`) |
| Donors (unseen) | `*-00002` validation splits | 100 per category | M7 donor rows only; never trained on or evaluated |
| Out-of-distribution | `RedHatAI/speculator_benchmarks@2ae86af`: HumanEval, math_reasoning, summarization; training chat template (thinking off); Qwen3-8B responses with the r0001 sampling settings, capped at 256 tokens | 50 each | offline evals `--mode ood` / `base_ood` (`output/dflash-lora/ood-v2`; HF `erdem-erdem/dyn-spec-dflash-ood-v2@c08eadeb`) |

- **Disjoint:** no r0001 validation row appears in any r0001 (24K) or r0002-100k (100K) training row (checked by prompt hash and primary id).
- **One LoRA per row:** the generator makes one LoRA from each row's own prompt, in validation and in training (training groups of 1 row, 18 per update). M4 also reports K=4 category means.
- **Metrics:** offline and teacher-forced: accept length τ (`eal`), `full_acc`, per-position acceptance, per category. Fixed validation anchors (seed 0, density 1/16). Every offline eval saves **per-row** metrics, which pool exactly to the category numbers.
- **M4 per-row table:** each row's τ for every coefficient, its best coefficient, and per category the best fixed coefficient and the per-row oracle.
- **Seed:** 42, one seed (EAGLE-3 practice). EAGLE-3 records show about 5 % run-to-run drift.
- **Serving (tok/s):** P5, not yet.

## Experiments

| ID | Experiment | Tier | Status |
|---|---|---|---|
| [P0](P0.md) | Activation store fill and online-offline check | T-setup | done |
| [P2](P2.md) | Smoke matrix on the real models | T-setup | done |
| [M0](M0.md) | Headline: base vs static LoRA vs generator | 1 | running |
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
