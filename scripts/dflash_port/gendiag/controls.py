"""GENDIAG phase A: does the generator's gain depend on the row's own prompt?

    CUDA_VISIBLE_DEVICES=0 python scripts/dflash_port/gendiag/controls.py RECIPE \
        --tier T --out OUT.json [--snapshot u01000.pt] [--split val|train]

Per evaluated row (validation rows, or a seeded training sample) and variant,
the production forward with that row's batch (validation anchors, eval mode):

low-rank variants (compared with `own`, the production LoRA):
  own          the row's own generated LoRA (reproduces training validation)
  base         no LoRA
  swap_same    LoRA generated from another evaluated row of the same category
  swap_cross   LoRA generated from a row of another category
  zbar         decoder fed the mean condition z̄ over the evaluated rows
  dbar         heads fed the mean decoder output (the head's static part)
  bias_only    head biases only (head weights zeroed): the prompt-free part

dense variants (ΔW applied exactly as A = I, B = ΔW / s; compared with
`own_dense`, the own ΔW through the same dense path):
  mean         the mean ΔW over the evaluated rows (the "average patch")
  centroid     the mean ΔW of the row's category
  coef_c       ΔW̄ + c·(ΔW_i − ΔW̄) for c in COEFS (c = 0 is `mean`)
  pca_k        ΔW̄ + projection of (ΔW_i − ΔW̄) on the top-k principal
               directions of the evaluated set (pooled over modules)

Writes per-row counts, headline τ / full_acc per variant and paired bootstrap
Δ vs the reference, and the PCA spectrum.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

COEFS = (-1.0, 0.5, 1.5, 2.0)
PCA_K = (1, 2, 4, 8, 16)
LOW_RANK = ("own", "base", "swap_same", "swap_cross", "zbar", "dbar", "bias_only")


def donors(rows, seed=0):
    """Same-category derangement and a cross-category partner per row."""
    rng = random.Random(seed)
    by_cat: dict[str, list[int]] = {}
    for idx, (subset, _) in enumerate(rows):
        by_cat.setdefault(C.category(subset), []).append(idx)
    same, cross = {}, {}
    cats = sorted(by_cat)
    for cat, members in by_cat.items():
        shuffled = members[:]
        while True:
            rng.shuffle(shuffled)
            if all(a != b for a, b in zip(members, shuffled, strict=True)):
                break
        same.update(zip(members, shuffled, strict=True))
        other = by_cat[cats[(cats.index(cat) + 1) % len(cats)]]
        for m in members:
            cross[m] = rng.choice(other)
    return same, cross


@torch.no_grad()
def generate(generator, runtime, rows):
    """Per-row factors (CPU, generator dtype), conditions z and decoder outputs."""
    factors, zs = [], []
    for _, row in rows:
        ctx = runtime.corpus.context([row], "last", runtime.device)
        pooled = generator.pool_examples(generator.encode_examples(*ctx))
        z = generator.condition(pooled)
        zs.append(z.float())
        out = generator.decode_factors(z, None)
        factors.append({k: (a.cpu(), b.cpu()) for k, (a, b) in out.items()})
    return factors, torch.stack(zs)


@torch.no_grad()
def special_factors(generator, zbar):
    """Factors decoded from z̄, and the head biases alone (weights zeroed)."""
    zf = generator.decode_factors(zbar.to(next(generator.parameters()).dtype), None)
    bias_out = {}
    for name, (in_f, out_f) in generator.shapes.items():
        module_type, _ = generator.layout[name]
        head = generator.heads[name if generator.head_sharing == "per_module" else module_type]
        parts = []
        for part in ("A", "B"):
            lin = head[part]
            parts.append(lin.bias if lin.bias is not None else generator.biases[f"{name}__{part}"])
        bias_out[name] = (parts[0].view(generator.rank, in_f), parts[1].view(out_f, generator.rank))
    return zf, bias_out


@torch.no_grad()
def mean_decoder_factors(generator, runtime, rows):
    """Heads applied to the mean (over rows) decoder output of each module."""
    sums = None
    for _, row in rows:
        ctx = runtime.corpus.context([row], "last", runtime.device)
        z = generator.condition(generator.pool_examples(generator.encode_examples(*ctx)))
        decoded = {}
        for name in generator.shapes:
            module_type, layer = generator.layout[name]
            descriptor = torch.cat([
                z,
                generator.module_embedding.weight[generator.module_types.index(module_type)],
                generator.layer_embedding.weight[layer],
            ], -1)
            out_d = generator.decoder(descriptor)
            if generator.head_centering:
                out_d = generator._centre(name, out_d)
            decoded[name] = out_d.float()
        if sums is None:
            sums = decoded
        else:
            for k in sums:
                sums[k] += decoded[k]
    out = {}
    for name, (in_f, out_f) in generator.shapes.items():
        module_type, _ = generator.layout[name]
        head = generator.heads[name if generator.head_sharing == "per_module" else module_type]
        d = (sums[name] / len(rows)).to(head["A"].weight.dtype)
        a, b = head["A"](d), head["B"](d)
        if generator.head_sharing == "per_type_bias":
            a = a + generator.biases[f"{name}__A"]
            b = b + generator.biases[f"{name}__B"]
        out[name] = (a.view(generator.rank, in_f), b.view(out_f, generator.rank))
    return out


def dense_dw(factors, scale, device):
    return {k: scale[k] * (b.to(device).float() @ a.to(device).float()) for k, (a, b) in factors.items()}


def as_dense_factors(dw, scale, eye):
    return {k: (eye[v.shape[1]], v / scale[k]) for k, v in dw.items()}


def analyse(runtime, job, output, generator, args):
    started = time.time()
    rows = C.eval_rows(runtime, job, args.split, per_category=args.per_category)
    if args.per_subset:
        seen: dict[str, int] = {}
        kept = []
        for subset, row in rows:
            if seen.get(subset, 0) < args.per_subset:
                kept.append((subset, row))
                seen[subset] = seen.get(subset, 0) + 1
        rows = kept
    quick = args.quick
    dev = runtime.device
    scale = C.scales(runtime)
    print(f"[controls] {len(rows)} rows ({args.split}); generating factors", flush=True)
    factors, zs = generate(generator, runtime, rows)
    zbar = zs.mean(0)
    z_factors, bias_factors = special_factors(generator, zbar)
    dbar_factors = mean_decoder_factors(generator, runtime, rows)
    zero = {k: (a.detach().clone(), torch.zeros_like(b)) for k, (a, b) in runtime.adapter_state().items()}
    same, cross = donors(rows)
    labels = [C.category(s) for s, _ in rows]
    n = len(rows)
    # Mean and category centroids of ΔW (dense, fp32).
    keys = list(factors[0])
    mean = {k: torch.zeros(generator.shapes[k][1], generator.shapes[k][0], device=dev) for k in keys}
    cent = {c: {k: torch.zeros_like(mean[k]) for k in keys} for c in set(labels)}
    for i, f in enumerate(factors):
        dw = dense_dw(f, scale, dev)
        for k in keys:
            mean[k] += dw[k] / n
            cent[labels[i]][k] += dw[k] / labels.count(labels[i])
    eye = {d: torch.eye(d, device=dev) for d in {generator.shapes[k][0] for k in keys}}
    # PCA of the deviations, pooled over modules (exact via the Gram).
    from speculators.generator.offline_eval import gram  # noqa: PLC0415

    g = sum(gram([f[k] for f in factors], scale[k], dev) for k in keys)
    h = torch.eye(n, device=dev, dtype=g.dtype) - 1.0 / n
    kc = h @ g @ h
    evals, evecs = torch.linalg.eigh(kc)
    evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)
    total_var = float(evals.sum())
    kmax = max(PCA_K)
    v = evecs[:, :kmax].float()  # n × kmax
    phi = {}
    for k in ([] if quick else keys):
        a_cat = torch.cat([f[k][0].to(dev).float() for f in factors])  # (n r) × in
        bs = torch.stack([f[k][1].to(dev).float() for f in factors])  # n × out × r
        mats = []
        for l in range(kmax):
            b_cat = (bs * v[:, l, None, None]).permute(1, 0, 2).reshape(bs.shape[1], -1)
            mats.append((scale[k] * (b_cat @ a_cat)).to(torch.bfloat16))
        phi[k] = torch.stack(mats)
        del a_cat, bs
    print(f"[controls] PCA ready ({time.time() - started:.0f}s); evaluating", flush=True)

    variants = ["own", "base", "swap_same", "swap_cross", "zbar", "dbar", "bias_only",
                "own_dense", "mean", "centroid", *[f"coef_{c:g}" for c in COEFS], *[f"pca_{k}" for k in PCA_K]]
    if quick:
        variants = ["own", "swap_same", "swap_cross", "zbar", "bias_only", "own_dense", "mean", "coef_-1", "coef_2"]
    records = {name: [] for name in variants}
    for i, (subset, row) in enumerate(rows):
        own = factors[i]
        low = {
            "own": own,
            "base": zero,
            "swap_same": factors[same[i]],
            "swap_cross": factors[cross[i]],
            "zbar": {k: (a[0] if a.ndim == 3 else a, b[0] if b.ndim == 3 else b) for k, (a, b) in z_factors.items()},
            "dbar": dbar_factors,
            "bias_only": bias_factors,
        }
        for name, f in low.items():
            if name not in variants:
                continue
            f = {k: (a.to(dev), b.to(dev)) for k, (a, b) in f.items()}
            records[name].append(C.row_counts(runtime, row, f))
        dw = dense_dw(own, scale, dev)
        dense = {"own_dense": dw, "mean": mean, "centroid": cent[labels[i]]}
        for c in COEFS:
            dense[f"coef_{c:g}"] = {k: mean[k] + c * (dw[k] - mean[k]) for k in keys}
        coeff = v[i]  # this row's coordinates (times sqrt λ normalisation absorbed in Φ)
        for kk in ([] if quick else PCA_K):
            dense[f"pca_{kk}"] = {
                k: mean[k] + torch.einsum("l,loi->oi", coeff[:kk], phi[k][:kk].float()) for k in keys
            }
        for name, d in dense.items():
            if name not in variants:
                continue
            records[name].append(C.row_counts(runtime, row, as_dense_factors(d, scale, eye)))
        if (i + 1) % 50 == 0:
            print(f"[controls] {i + 1}/{n} rows ({time.time() - started:.0f}s)", flush=True)

    summary = {}
    for name in variants:
        tau, tau_cats = C.headline(records[name], rows, "eal")
        acc, _ = C.headline(records[name], rows, "full_acc")
        ref = "own_dense" if name in ("mean", "centroid") or name.startswith(("coef_", "pca_")) else "own"
        entry = {"tau": tau, "full_acc": acc, "tau_categories": tau_cats, "reference": ref}
        if name not in ("own", "own_dense"):
            entry["vs_reference"] = C.paired_delta(records[ref], records[name], rows)
        summary[name] = entry
    # Share of rows where own beats each control (per-row τ).
    def row_tau(c):
        return c["eal_sum"] / max(c["eal_total"], 1)

    wins = {
        name: sum(row_tau(o) > row_tau(x) for o, x in zip(records["own"], records[name], strict=True)) / n
        for name in ("swap_same", "swap_cross", "zbar", "bias_only") if name in variants
    }
    wins.update({
        name: sum(row_tau(o) > row_tau(x) for o, x in zip(records["own_dense"], records[name], strict=True)) / n
        for name in ("mean", "centroid") if name in variants
    })
    return {
        "rows": [[s, r] for s, r in rows],
        "split": args.split,
        "snapshot": str(args.snapshot) if args.snapshot else None,
        "run": str(output),
        "summary": summary,
        "own_win_share": wins,
        "pca": {
            "explained": [float(evals[: k].sum()) / max(total_var, 1e-300) for k in range(1, kmax + 1)],
            "total_var": total_var,
            "mean_norm_sq": float(sum(float(m.pow(2).sum()) for m in mean.values())),
        },
        "records": records,
        "seconds": time.time() - started,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recipe")
    p.add_argument("--tier", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--split", default="val", choices=("val", "train"))
    p.add_argument("--per-category", type=int, default=125)
    p.add_argument("--per-subset", type=int, help="keep the first N rows of each subset")
    p.add_argument("--quick", action="store_true", help="core variants only, no PCA")
    args = p.parse_args(argv)
    result = C.with_trained(
        args.recipe, args.tier, lambda *a: analyse(*a, args), snapshot=args.snapshot
    )
    C.write_json(args.out, result)
    s = result["summary"]
    print(f"[controls] own τ {s['own']['tau']:.4f}; " + "; ".join(
        f"{k} {v['vs_reference']['delta']:+.4f} [{v['vs_reference']['lo']:+.4f},{v['vs_reference']['hi']:+.4f}]"
        for k, v in s.items() if "vs_reference" in v))


if __name__ == "__main__":
    main()
