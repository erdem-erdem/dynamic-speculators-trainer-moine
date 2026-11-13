"""Training loss vs plain KL vs accuracy on the 100 sweep rows for the trained points.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/dflash_port/perturbation/loss_check.py

For base, the 24K static, and the final T2 deltas: the training objective
(KL + D-PACE weights, as returned by the forward), the plain per-position
KL(p_T||p), full_acc and τ, pooled over the 100 sweep rows (validation
anchors). Writes perturb24k/results/loss_check.json.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402


def main():
    def body(runtime, job, _output):
        H.install_capture(runtime)
        runtime.drafter.requires_grad_(False)
        hooks = H.DeltaHooks(runtime.drafter)
        rows = H.sweep_rows(runtime, job)
        zero = H.zero_factors(runtime)
        points = {"base": (zero, {}), "static24k_kl": (H.static_factors(runtime), {})}
        for run in ("full", "lora_attn_r512", "lora_ffn_r512"):
            path = H.OUT / "t2" / run / "final_delta.pt"
            if not path.exists():
                continue
            saved = torch.load(path, map_location="cpu", weights_only=False)
            active = {}
            for site, v in saved.items():
                if isinstance(v, dict):
                    w = v["scale"] * v["b"].float() @ v["a"].float()
                else:
                    w = v.float()
                active[site] = H.Delta(w=w.to(runtime.device, runtime.dtype))
            points[f"t2_{run}"] = (zero, active)
        out = {}
        for name, (factors, active) in points.items():
            tot = {"loss": 0.0, "kl_sum": 0.0, "positions": 0.0, "acc_sum": 0.0, "eal_sum": 0.0,
                   "blocks": 0.0}
            hooks.set(active)
            for _, row in rows:
                batch = H.load_batch(runtime, row)
                with torch.no_grad():
                    _, loss, m = runtime.forward(batch, factors)
                lg, tg, mk = H.CAPTURE["logits"], H.CAPTURE["targets"], H.CAPTURE["mask"]
                lp = torch.log_softmax(lg.float(), -1)
                lpt = torch.log_softmax(tg.float(), -1)
                kl = (lpt.exp() * (lpt - lp)).sum(-1)
                tot["loss"] += float(loss)
                tot["kl_sum"] += float((kl * mk).sum())
                tot["positions"] += float(mk.sum())
                tot["acc_sum"] += float(m["full_acc_sum"])
                tot["eal_sum"] += float(m["eal_sum"])
                tot["blocks"] += float(m["eal_total"])
                H.CAPTURE.clear()
            hooks.clear()
            out[name] = {"training_loss_mean_per_row": tot["loss"] / len(rows),
                         "kl_per_position": tot["kl_sum"] / tot["positions"],
                         "full_acc_pooled": tot["acc_sum"] / tot["positions"],
                         "tau_pooled": tot["eal_sum"] / tot["blocks"]}
            print(name, out[name], flush=True)
        H.write_json(H.OUT / "results" / "loss_check.json", {"rows": len(rows), "points": out})

    H.with_runtime(body)


if __name__ == "__main__":
    main()
