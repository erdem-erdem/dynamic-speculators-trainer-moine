"""GENDIAG phases C / D on a trained (or snapshot) generator.

    CUDA_VISIBLE_DEVICES=0 python scripts/dflash_port/gendiag/internals.py RECIPE \
        --tier T --out OUT.json [--snapshot u01000.pt]

- probe: `diagnostics.GeneratorDiagnostics.probe_metrics` on `--per-category`
  training and validation rows per category (spread at every stage, head
  static / dynamic split, dead units, D13 geometry);
- gradient by component: the training loss of `--grad-rows` training rows
  (each row its own LoRA, validation anchors, eval mode) back-propagated into
  the generator; RMS per element and share of the squared norm per group;
  the share of the gradient reaching z that all rows share;
- head weight spectra: singular values of every head weight matrix
  (participation ratio, top-1 / top-8 energy share).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def analyse(runtime, job, output, generator, args):
    from speculators.generator.diagnostics import GeneratorDiagnostics, group_of  # noqa: PLC0415

    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        diag = GeneratorDiagnostics(
            runtime, generator, job, tmp, interval=1, snapshots=[], per_category=args.per_category
        )
        probe = diag.probe_metrics()
        diag.out.close()
    # Gradient by component.
    rows = C.eval_rows(runtime, job, "train", per_category=args.grad_rows // 4, seed=5)
    params = dict(generator.named_parameters())
    for p in params.values():
        p.grad = None
    zgrads = []
    for _, row in rows:
        ctx = runtime.corpus.context([row], "last", runtime.device)
        with torch.enable_grad():
            pooled = generator.pool_examples(generator.encode_examples(*ctx))
            z = generator.condition(pooled)
            z.retain_grad()
            factors = generator.decode_factors(z, None)
            loss = 0.0
            for batch, _ in runtime.corpus.microbatches([row], runtime.device, runtime.dtype):
                _, l, _ = runtime.forward(batch, factors)
                loss = loss + l
            (loss / len(rows)).backward()
        zgrads.append(z.grad.detach().float().reshape(-1))
    groups: dict[str, list[float]] = {}
    for name, p in params.items():
        g = 0.0 if p.grad is None else float(p.grad.float().pow(2).sum())
        e = groups.setdefault(group_of(name), [0.0, 0, 0.0])
        e[0] += g
        e[1] += p.numel()
        e[2] += float(p.detach().float().pow(2).sum())
    total = sum(v[0] for v in groups.values())
    grad = {
        k: {"rms": (v[0] / v[1]) ** 0.5, "share": v[0] / total, "numel": v[1],
            "param_rms": (v[2] / v[1]) ** 0.5}
        for k, v in groups.items()
    }
    zg = torch.stack(zgrads)
    zmean = zg.mean(0)
    zgrad = {"shared_share": float(zmean.pow(2).sum() / zg.pow(2).sum(1).mean()),
             "norm_mean": float(zg.norm(dim=1).mean())}
    for p in params.values():
        p.grad = None
    # Head weight spectra.
    spectra = {}
    for hname, head in generator.heads.items():
        for part in ("A", "B"):
            w = head[part].weight.detach().float()
            s = torch.linalg.svdvals(w)
            e = s.pow(2)
            spectra[f"{hname}.{part}"] = {
                "fro": float(e.sum().sqrt()),
                "participation": float(e.sum() ** 2 / e.pow(2).sum().clamp_min(1e-300)),
                "top1": float(e[0] / e.sum()), "top8": float(e[:8].sum() / e.sum()),
                "bias_norm": float(head[part].bias.detach().float().norm()) if head[part].bias is not None else None,
            }
    return {
        "run": str(output), "snapshot": str(args.snapshot) if args.snapshot else None,
        "params": {k: v["numel"] for k, v in grad.items()},
        "probe": probe, "grad": grad, "zgrad": zgrad, "spectra": spectra,
        "seconds": time.time() - started,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recipe")
    p.add_argument("--tier", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--per-category", type=int, default=64)
    p.add_argument("--grad-rows", type=int, default=64)
    args = p.parse_args(argv)
    result = C.with_trained(args.recipe, args.tier, lambda *a: analyse(*a, args), snapshot=args.snapshot)
    C.write_json(args.out, result)
    pv = result["probe"]["val"]
    print("[internals] spread", {k: round(pv[k]["spread"], 6) for k in ("summary", "encoder", "z", "A", "B")},
          "dW", {k: round(v, 4) for k, v in pv["dW"].items() if isinstance(v, float)})
    print("[internals] grad share", {k: f"{v['share']:.2e}" for k, v in result["grad"].items()})


if __name__ == "__main__":
    main()
