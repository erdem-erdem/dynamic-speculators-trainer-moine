"""Check: the study's forward reproduces the recorded validation numbers.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/dflash_port/perturbation/verify.py

On all 500 validation rows, base (zero factors) and the 24K static LoRA
(coefficient 2) through the study harness (capture forward + inactive
perturbation hooks): category-mean full_acc / τ against the recorded
0.5267 / 0.5338, per-row τ against the recorded per-row records, and the
study's own accepted-length computation (metrics.row_metrics) against the
upstream metric on every row. Writes perturb24k/verify/verify.json.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import metrics as MT  # noqa: E402


def main():
    def body(runtime, job, _output):
        H.install_capture(runtime)
        hooks = H.DeltaHooks(runtime.drafter)
        rows = H.split_rows(runtime, job, "validation")
        recorded = H.recorded_scores()
        rec_rows = {a: H.recorded_row_metrics(recorded[a]) for a in ("base", "static")}
        factors = {"base": H.zero_factors(runtime), "static": H.static_factors(runtime)}
        counts = {a: {} for a in factors}
        row_diff = {a: 0.0 for a in factors}
        own_diff = 0.0
        started = time.time()
        timing = []
        for n, (subset, row) in enumerate(rows):
            batch = H.load_batch(runtime, row)
            pid = runtime.corpus.rows[row]["primary_id"]
            logits = {}
            for arm, f in factors.items():
                torch.cuda.synchronize()
                t = time.time()
                with torch.no_grad():
                    _, _, metrics = runtime.forward(batch, f)
                torch.cuda.synchronize()
                timing.append(time.time() - t)
                m = {k: float(v) for k, v in metrics.items()}
                target = counts[arm].setdefault(subset, {})
                for k, v in m.items():
                    target[k] = target.get(k, 0.0) + v
                rec = rec_rows[arm][pid]
                row_diff[arm] = max(
                    row_diff[arm],
                    abs(m["eal_sum"] / m["eal_total"] - rec["eal"]),
                    abs(m["full_acc_sum"] / m["full_acc_total"] - rec["full_acc"]),
                )
                logits[arm] = (H.CAPTURE["logits"], m)
            sums, block = MT.row_metrics(
                logits["base"][0], logits["static"][0], H.CAPTURE["targets"],
                H.CAPTURE["mask"],
            )
            mb, ms = logits["base"][1], logits["static"][1]
            own_diff = max(
                own_diff,
                abs(float(block[MT.BK["eal1"]]) - mb["eal_sum"]),
                abs(float(block[MT.BK["eal2"]]) - ms["eal_sum"]),
                abs(float(sums[:, MT.SK["c1"]].sum()) - mb["full_acc_sum"]),
                abs(float(sums[:, MT.SK["c2"]].sum()) - ms["full_acc_sum"]),
                abs(float(sums[:, MT.SK["n"]].sum()) - mb["full_acc_total"]),
            )
            if n % 50 == 0:
                print(f"{n + 1}/{len(rows)} {time.time() - started:.0f}s "
                      f"fwd={sum(timing[-2:]) / 2 * 1000:.0f}ms {H.memory_report()}",
                      flush=True)
        if hooks.active:
            raise RuntimeError("hooks should be inactive")
        out = {
            "rows": len(rows),
            "scores": {a: H.score(c) for a, c in counts.items()},
            "recorded": {
                a: {"full_acc": recorded[a]["score"]} for a in ("base", "static")
            },
            "max_row_abs_diff_vs_recorded": row_diff,
            "max_own_vs_upstream_count_diff": own_diff,
            "mean_forward_seconds": sum(timing) / len(timing),
            "seconds": time.time() - started,
        }
        H.write_json(H.OUT / "verify" / "verify.json", out)
        print(out)

    H.with_runtime(body)


if __name__ == "__main__":
    main()
