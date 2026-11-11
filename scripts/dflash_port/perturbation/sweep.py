"""Perturbation sweeps on the fixed 100 validation rows (T0 steps, T1, free points).

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/dflash_port/perturbation/sweep.py \
        --job calibrate|t1|t0steps|free --point base|static [--rows-per-category 25]

Every perturbation theta2 = theta1 + dW is evaluated against its starting
point theta1 on the same anchored blocks (validation anchors, seed 0), row
by row: theta1 is run once per row, then every perturbation of the job.
Per row, per draft slot, the sums of `metrics.SLOT_KEYS` and per-block
accepted lengths are saved, so any category / slot / bootstrap aggregate can
be formed later (analyze.py).

Jobs:
- calibrate: a few locations on a wide eps grid, 1 seed, 20 rows (base);
- t1: dense Gaussian dW at every T1 location x EPS x 2 seeds, plus random
  low-rank dW = B A (ranks RANKS) on 5 locations at RANK_EPS;
- t0steps: one gradient-descent and one gradient-ascent step along the
  mean training gradient (perturb24k/t0/<point>/mean_grads_bf16.pt) of each
  location at STEP_EPS, plus per-category descent steps on group locations;
- free: (base only) the trained adapters as perturbations of the base drafter:
  24K static (KL), 10K-math static (CE), 4K M0 static (CE), 24K generator
  (KL, its own LoRA per row).

Output: perturb24k/<job>/<point>.pt with meta, rows, slot [P, R, 15, K],
block [P, R, 3] (resumable; saved every 10 rows).
"""

from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import metrics as MT  # noqa: E402

# Calibrated on 20 rows (calibrate job): single modules barely move below
# eps 0.1, groups are large at 0.3 and destroyed at 1.0.
EPS = (0.01, 0.03, 0.1, 0.3, 1.0)
# bf16 floor control: any change upstream re-rounds the forward and flips
# near-tied argmaxes; eps 1e-4 on the whole drafter measures that floor.
FLOOR_EPS = (1e-4,)
SEEDS = (0, 1)
RANKS = (32, 64, 128, 512, 1024)
RANK_LOCATIONS = ("all_attn", "all_ffn", "layer1", "layer5", "fc")
RANK_EPS = (0.1, 0.3)
STEP_EPS = (0.003, 0.01, 0.03, 0.1)
CAT_STEP_EPS = 0.01
CAT_STEP_LOCATIONS = ("fc", "all_attn", "all_ffn", "all", "whole",
                      "layer1", "layer2", "layer3", "layer4", "layer5")
CAL_LOCATIONS = ("fc", "l0_q", "l2_v", "l4_down", "all_attn", "all_ffn", "whole")
CAL_EPS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


class Perturbation:
    def __init__(self, name, meta, active=None, factors=None):
        self.name, self.meta = name, meta
        self._active, self._factors = active, factors

    def active(self, row):
        return self._active(row) if callable(self._active) else (self._active or {})

    def factors(self, row):
        return self._factors(row) if self._factors is not None else None


def _seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


class DenseDirections:
    """Seeded Gaussian directions per (seed, site), bfloat16 on the device."""

    def __init__(self, runtime, seeds):
        self.d, self.norm = {}, {}
        for seed in seeds:
            for site in H.all_sites():
                w = H.site_weight(runtime, site)
                g = torch.Generator(device=runtime.device).manual_seed(_seed("dense", seed, site))
                d = torch.randn(w.shape, generator=g, device=runtime.device, dtype=torch.float32)
                self.norm[(seed, site)] = float(d.norm())
                self.d[(seed, site)] = d.to(runtime.dtype)


def dense_perturbations(runtime, wnorm, dirs, locations, eps_grid, seeds, point):
    perts = []
    for loc in locations:
        members = H.location_members(loc)
        for seed in seeds:
            for eps in eps_grid:
                active = {
                    H.hook_key(site, path): H.Delta(
                        w=dirs.d[(seed, site)],
                        coef=eps * wnorm[site] / dirs.norm[(seed, site)],
                    )
                    for site, path in members
                }
                perts.append(Perturbation(
                    f"dense|{loc}|{eps}|s{seed}",
                    {"kind": "dense", "location": loc, "eps": eps, "seed": seed,
                     "rank": "full", "point": point},
                    active,
                ))
    return perts


def lowrank_perturbations(runtime, wnorm, locations, ranks, eps_grid, seeds, point):
    perts = []
    for loc in locations:
        members = H.location_members(loc)
        for rank in ranks:
            for seed in seeds:
                factors = {}
                for site, _ in members:
                    w = H.site_weight(runtime, site)
                    out_f, in_f = w.shape
                    g = torch.Generator(device=runtime.device).manual_seed(
                        _seed("lowrank", seed, rank, site))
                    a = torch.randn(rank, in_f, generator=g, device=runtime.device)
                    b = torch.randn(out_f, rank, generator=g, device=runtime.device)
                    norm = float((b @ a).norm())
                    factors[site] = (a.to(runtime.dtype), b.to(runtime.dtype), norm)
                for eps in eps_grid:
                    active = {
                        H.hook_key(site, path): H.Delta(
                            a=factors[site][0], b=factors[site][1],
                            coef=eps * wnorm[site] / factors[site][2],
                        )
                        for site, path in members
                    }
                    perts.append(Perturbation(
                        f"lowrank|{loc}|{eps}|r{rank}|s{seed}",
                        {"kind": "lowrank", "location": loc, "eps": eps, "seed": seed,
                         "rank": rank, "point": point},
                        active,
                    ))
    return perts


def gradient_directions(runtime, point):
    """{"all" | category: {hook key: gradient (bf16, device)}} with K/V totals."""
    grads = torch.load(H.OUT / "t0" / point / "mean_grads_bf16.pt", map_location="cpu")
    out = {}
    cats = list(H.CATEGORIES)
    for name in cats + ["all"]:
        src = (
            {k: sum(grads[c][k].float() for c in cats) / len(cats) for k in grads[cats[0]]}
            if name == "all" else {k: v.float() for k, v in grads[name].items()}
        )
        d = {}
        for key, g in src.items():
            d[key] = g
        for i in range(H.LAYERS):
            for m in ("k", "v"):
                d[f"l{i}_{m}"] = d[f"l{i}_{m}@ctx"] + d[f"l{i}_{m}@blk"]
        out[name] = {k: v.to(runtime.device, runtime.dtype) for k, v in d.items()}
        out[name + "_norm2"] = {k: float(v.pow(2).sum()) for k, v in d.items()}
    return out


def step_perturbations(gdirs, wnorm, locations, eps_grid, point, source="all",
                       signs=(-1, 1)):
    perts = []
    for loc in locations:
        members = H.location_members(loc)
        keys = [H.hook_key(site, path) for site, path in members]
        gnorm = sum(gdirs[source + "_norm2"][k] for k in keys) ** 0.5
        wn = sum(wnorm[site] ** 2 for site, _ in members) ** 0.5
        for eps in eps_grid:
            for sign in signs:
                coef = sign * eps * wn / gnorm
                active = {k: H.Delta(w=gdirs[source][k], coef=coef) for k in keys}
                kind = "descent" if sign < 0 else "ascent"
                perts.append(Perturbation(
                    f"{kind}|{loc}|{eps}|{source}",
                    {"kind": kind, "location": loc, "eps": eps, "source": source,
                     "point": point, "grad_norm": gnorm, "weight_norm": wn},
                    active,
                ))
    return perts


def run_sweep(runtime, hooks, rows, theta1, perts, out_path, *, save_every=10):
    """Row-major evaluation of `perts` against `theta1` (factors or callable)."""
    out_path = Path(out_path)
    names = [p.name for p in perts]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate perturbation names")
    slot = torch.zeros(len(perts), len(rows), MT.SLOTS, len(MT.SLOT_KEYS), dtype=torch.float64)
    block = torch.zeros(len(perts), len(rows), len(MT.BLOCK_KEYS), dtype=torch.float64)
    done = 0
    if out_path.exists():
        saved = torch.load(out_path, map_location="cpu", weights_only=False)
        if saved["names"] != names or saved["rows"] != [list(r) for r in rows]:
            raise ValueError(f"{out_path} belongs to a different sweep")
        slot, block, done = saved["slot"], saved["block"], saved["done"]
        print(f"resuming {out_path.name} at row {done}", flush=True)

    def save(n):
        tmp = out_path.with_suffix(".tmp")
        torch.save({"names": names, "meta": [p.meta for p in perts],
                    "rows": [list(r) for r in rows], "categories": [H.category(s) for s, _ in rows],
                    "slot_keys": list(MT.SLOT_KEYS), "block_keys": list(MT.BLOCK_KEYS),
                    "slot": slot, "block": block, "done": n}, tmp)
        tmp.replace(out_path)

    started = time.time()
    for r in range(done, len(rows)):
        subset, row = rows[r]
        batch = H.load_batch(runtime, row)
        f1 = theta1(row) if callable(theta1) else theta1
        hooks.clear()
        with torch.no_grad():
            runtime.forward(batch, f1)
        l1 = H.CAPTURE["logits"]
        lt, mask = H.CAPTURE["targets"], H.CAPTURE["mask"]
        pieces = MT.prepare_theta1(l1, lt, mask)
        for p, pert in enumerate(perts):
            hooks.set(pert.active(row))
            f2 = pert.factors(row)
            with torch.no_grad():
                runtime.forward(batch, f1 if f2 is None else f2)
            hooks.clear()
            if not torch.equal(H.CAPTURE["mask"], mask):
                raise RuntimeError("Anchors differ between theta1 and theta2")
            s, b = MT.row_metrics_cached(pieces, H.CAPTURE["logits"])
            slot[p, r] = s
            block[p, r] = b
        del pieces, l1
        H.CAPTURE.clear()
        if (r + 1) % save_every == 0 or r + 1 == len(rows):
            save(r + 1)
        elapsed = time.time() - started
        print(f"{out_path.stem}: row {r + 1}/{len(rows)} {elapsed:.0f}s "
              f"({elapsed / (r + 1 - done):.1f}s/row, {len(perts)} perts) {H.memory_report()}",
              flush=True)
    return out_path


def theta1_factors(runtime, point):
    return H.zero_factors(runtime) if point == "base" else H.static_factors(runtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=("calibrate", "t1", "t0steps", "free"), required=True)
    parser.add_argument("--point", choices=("base", "static"), default="base")
    parser.add_argument("--rows-per-category", type=int, default=25)
    parser.add_argument("--part", default="all", help="t1: dense|lowrank|all")
    args = parser.parse_args()

    def body(runtime, job, _output):
        H.install_capture(runtime)
        runtime.drafter.requires_grad_(False)
        hooks = H.DeltaHooks(runtime.drafter)
        f1 = theta1_factors(runtime, args.point)
        wnorm = {k: float(v.norm()) for k, v in H.effective_weights(runtime, f1).items()}
        per_cat = 5 if args.job == "calibrate" else args.rows_per_category
        rows = H.sweep_rows(runtime, job, per_cat)
        out_dir = H.OUT / args.job
        out_dir.mkdir(parents=True, exist_ok=True)
        H.write_json(out_dir / f"{args.point}_weight_norms.json", wnorm)
        if args.job == "calibrate":
            dirs = DenseDirections(runtime, (0,))
            perts = dense_perturbations(runtime, wnorm, dirs, CAL_LOCATIONS, CAL_EPS, (0,),
                                        args.point)
            run_sweep(runtime, hooks, rows, f1, perts, out_dir / f"{args.point}.pt")
        elif args.job == "t1":
            if args.part in ("dense", "all"):
                dirs = DenseDirections(runtime, SEEDS)
                perts = dense_perturbations(runtime, wnorm, dirs, H.t1_locations(), EPS,
                                            SEEDS, args.point)
                perts += dense_perturbations(runtime, wnorm, dirs, ("whole",), FLOOR_EPS,
                                             SEEDS, args.point)
                run_sweep(runtime, hooks, rows, f1, perts, out_dir / f"{args.point}_dense.pt")
                del dirs, perts
                torch.cuda.empty_cache()
            if args.part in ("lowrank", "all"):
                perts = lowrank_perturbations(runtime, wnorm, RANK_LOCATIONS, RANKS, RANK_EPS,
                                              SEEDS, args.point)
                run_sweep(runtime, hooks, rows, f1, perts, out_dir / f"{args.point}_lowrank.pt")
        elif args.job == "t0steps":
            gdirs = gradient_directions(runtime, args.point)
            perts = step_perturbations(gdirs, wnorm, H.t1_locations(), STEP_EPS, args.point)
            for cat in H.CATEGORIES:
                perts += step_perturbations(gdirs, wnorm, CAT_STEP_LOCATIONS, (CAT_STEP_EPS,),
                                            args.point, source=cat, signs=(-1,))
            run_sweep(runtime, hooks, rows, f1, perts, out_dir / f"{args.point}.pt")
        elif args.job == "free":
            if args.point != "base":
                raise ValueError("free points are measured from the base drafter")
            perts = []
            for name, run_dir in H.FREE_STATICS.items():
                factors = H.static_factors(runtime, run_dir)
                perts.append(Perturbation(name, {"kind": "free", "location": "all_attn",
                                                 "point": "base", "arm": name},
                                          factors=lambda _row, f=factors: f))
            generator = H.load_generator(runtime)
            perts.append(Perturbation(
                "generator24k_kl",
                {"kind": "free", "location": "all_attn", "point": "base",
                 "arm": "generator24k_kl"},
                factors=lambda row: H.generator_factors(runtime, generator, row),
            ))
            run_sweep(runtime, hooks, rows, f1, perts, out_dir / "base.pt")
        print("done", flush=True)

    H.with_runtime(body)


if __name__ == "__main__":
    main()
