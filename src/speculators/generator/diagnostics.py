"""Training-time generator diagnostics (GENDIAG, 2026-11-16).

Observation only: nothing here changes what training computes. Enabled by the
environment (so run identities stay unchanged):

    GENDIAG_DIR=<dir>             where diag.jsonl and snapshots/ go (required)
    GENDIAG_INTERVAL=20           diagnostic step every N updates (and 1..10)
    GENDIAG_SNAPSHOTS=50,100,...  updates at which the generator state is saved
    GENDIAG_PROBE_PER_CATEGORY=16 probe rows per category and split

At a diagnostic update the module records:

- ``grad``: pre-clip gradient per parameter group (sum of squares, element
  count, RMS per element), the clip factor;
- ``step``: per group, the norm of the actual parameter change of this
  optimizer step, the parameter norm, and the share of elements that received
  a nonzero gradient but did not change (bf16 rounding swallowed the update);
- ``init``: per group, movement from the initial parameters;
- ``zgrad``: gradient that reaches the condition vector z (one per cohort):
  per-cohort norms and the share of it shared by all cohorts
  (‖mean g‖² / mean ‖g‖²; 1 = every cohort pushes z the same way);
- ``probe``: a fixed set of training and validation rows through the
  generator (no grad, no randomness): between-row spread at every stage
  (summary → encoder → z → decoder → A, B → ΔW), the head output split into
  its static part (head at the mean decoder input) and dynamic part, dead
  units, and the D13 ΔW geometry; plus how far the mean ΔW and the dynamic
  parts moved since the last probe.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch

GROUP_PATTERNS = (
    ("enc1", r"^encoder\.0\."),
    ("enc2", r"^encoder\.2\."),
    ("enc_ln", r"^encoder\.3\."),
    ("cond1", r"^condition\.0\."),
    ("cond2", r"^condition\.2\."),
    ("emb", r"^(module|layer)_embedding\."),
    ("dec_in", r"^decoder\.0\."),
    ("dec_res", r"^decoder\.[1-9]"),
    ("headA_w", r"^heads\..*\.A\.weight$"),
    ("headA_b", r"^heads\..*\.A\.bias$"),
    ("headB_w", r"^heads\..*\.B\.weight$"),
    ("headB_b", r"^heads\..*\.B\.bias$"),
    ("bias_A", r"^biases\..*__A$"),
    ("bias_B", r"^biases\..*__B$"),
)


def group_of(name: str) -> str:
    for group, pattern in GROUP_PATTERNS:
        if re.search(pattern, name):
            return group
    raise ValueError(f"Generator parameter {name!r} has no diagnostics group")


def _f(x):
    return float(x) if not isinstance(x, float) else x


def spread_stats(x: torch.Tensor, labels: list[str]) -> dict:
    """Between-row statistics of a (n, d) matrix (float64)."""
    x = x.double().reshape(x.shape[0], -1)
    mean = x.mean(0)
    centred = x - mean
    total = centred.pow(2).sum(1).mean()
    norm_mean = mean.pow(2).sum()
    unit = x / x.norm(dim=1, keepdim=True).clamp_min(1e-30)
    cos = unit @ unit.T
    n = x.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=x.device)
    # Participation ratio of the centred row Gram (effective dimensions).
    eig = torch.linalg.eigvalsh(centred @ centred.T).clamp_min(0)
    pr = eig.sum() ** 2 / eig.pow(2).sum().clamp_min(1e-300)
    # Share of between-row variance explained by category (eta^2).
    cats = sorted(set(labels))
    between = 0.0
    for c in cats:
        idx = [i for i, label in enumerate(labels) if label == c]
        between += len(idx) * (x[idx].mean(0) - mean).pow(2).sum()
    eta2 = between / (n * total).clamp_min(1e-300)
    return {
        "spread": _f(total / norm_mean.clamp_min(1e-300)),  # var / ‖mean‖²
        "rms_dev": _f(total.sqrt()),
        "norm_mean": _f(norm_mean.sqrt()),
        "cos_mean": _f(cos[off].mean()),
        "one_minus_cos_median": _f((1 - cos[off]).median()),
        "participation": _f(pr),
        "eta2": _f(eta2),
    }


class GeneratorDiagnostics:
    """See the module docstring."""

    def __init__(self, runtime, generator, job, directory, *, interval, snapshots, per_category):
        self.runtime, self.generator, self.job = runtime, generator, job
        self.dir = Path(directory)
        (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.interval, self.snapshots = interval, set(snapshots)
        self.names = [n for n, _ in generator.named_parameters()]
        self.groups = {n: group_of(n) for n in self.names}
        self.params = dict(generator.named_parameters())
        self.init = {n: p.detach().clone() for n, p in self.params.items()}
        self.active = False
        self.record: dict = {}
        self.before: dict | None = None
        self.previous_probe: dict | None = None
        self.zgrads: list[torch.Tensor] = []
        self._probe_rows(per_category)
        self._hooks()
        self.out = (self.dir / "diag.jsonl").open("a")
        meta = {
            "kind": "meta",
            "label": job["label"],
            "interval": interval,
            "snapshots": sorted(self.snapshots),
            "probe": {k: len(v) for k, v in self.probe.items()},
            "groups": {
                g: sum(p.numel() for n, p in self.params.items() if self.groups[n] == g)
                for g in dict.fromkeys(self.groups.values())
            },
            "head_sharing": generator.head_sharing,
            "constant_input": generator.constant_input,
            "dtype": str(next(generator.parameters()).dtype),
        }
        self._write(meta)

    @classmethod
    def from_env(cls, runtime, generator, job):
        directory = os.environ.get("GENDIAG_DIR")
        if not directory or generator is None:
            return None
        interval = int(os.environ.get("GENDIAG_INTERVAL", "20"))
        snaps = os.environ.get("GENDIAG_SNAPSHOTS", "")
        snapshots = [int(s) for s in snaps.split(",") if s.strip()]
        per_category = int(os.environ.get("GENDIAG_PROBE_PER_CATEGORY", "16"))
        return cls(
            runtime, generator, job, directory,
            interval=interval, snapshots=snapshots, per_category=per_category,
        )

    # ------------------------------------------------------------------ setup
    def _probe_rows(self, per_category):
        corpus = self.runtime.corpus
        gen = torch.Generator().manual_seed(20261116)
        self.probe, self.probe_labels = {}, {}
        for split, subsets, validation in (
            ("train", self.job["subsets"], False),
            ("val", self.job["validation_subsets"], True),
        ):
            by_cat: dict[str, list[int]] = {}
            for subset in subsets:
                by_cat.setdefault(subset.rsplit("-", 1)[0], []).extend(
                    corpus.indices(subset, validation) if validation else corpus.indices(subset)
                )
            rows, labels = [], []
            for cat in sorted(by_cat):
                pool = by_cat[cat]
                pick = torch.randperm(len(pool), generator=gen)[:per_category].tolist()
                rows += [pool[i] for i in pick]
                labels += [cat] * len(pick)
            self.probe[split], self.probe_labels[split] = rows, labels

    def _hooks(self):
        gen = self.generator

        def keep_z_grad(_module, _inputs, output):
            if self.active and output.requires_grad:
                output.register_hook(lambda g: self.zgrads.append(g.detach().float()))

        gen.condition.register_forward_hook(keep_z_grad)

    # ------------------------------------------------------------- schedule
    def due(self, update: int) -> bool:
        return update <= 10 or update % self.interval == 0 or update in self.snapshots

    def begin(self, update):
        self.active = True
        self.zgrads = []
        self.record = {"kind": "step", "update": update, "time": time.time()}

    # --------------------------------------------------------- around step
    @torch.no_grad()
    def before_step(self, optimizer):
        sums: dict[str, list[float]] = {}
        for n, p in self.params.items():
            g = p.grad
            s = 0.0 if g is None else float(g.float().pow(2).sum())
            entry = sums.setdefault(self.groups[n], [0.0, 0])
            entry[0] += s
            entry[1] += p.numel()
        total = sum(v[0] for v in sums.values())
        self.record["grad"] = {
            g: {"sumsq": v[0], "numel": v[1], "rms": (v[0] / v[1]) ** 0.5, "share": v[0] / max(total, 1e-300)}
            for g, v in sums.items()
        }
        self.record["grad_norm"] = total**0.5
        self.before = {n: p.detach().clone() for n, p in self.params.items()}
        self.nonzero = {
            n: (p.grad != 0) if p.grad is not None else torch.zeros_like(p, dtype=torch.bool)
            for n, p in self.params.items()
        }
        self.record["lr"] = optimizer.param_groups[0]["lr"]

    @torch.no_grad()
    def after_step(self, clip_info):
        out: dict[str, dict] = {}
        for n, p in self.params.items():
            delta = (p.detach() - self.before[n]).float()
            changed = delta != 0
            nz = self.nonzero[n]
            e = out.setdefault(
                self.groups[n],
                {"delta_sq": 0.0, "param_sq": 0.0, "init_sq": 0.0, "init0_sq": 0.0,
                 "with_grad": 0, "swallowed": 0, "numel": 0},
            )
            e["delta_sq"] += float(delta.pow(2).sum())
            e["param_sq"] += float(p.detach().float().pow(2).sum())
            e["init_sq"] += float((p.detach().float() - self.init[n].float()).pow(2).sum())
            e["init0_sq"] += float(self.init[n].float().pow(2).sum())
            e["with_grad"] += int(nz.sum())
            e["swallowed"] += int((nz & ~changed).sum())
            e["numel"] += p.numel()
        self.record["step"] = {
            g: {
                "rel_step": (e["delta_sq"] / max(e["param_sq"], 1e-300)) ** 0.5,
                "step_norm": e["delta_sq"] ** 0.5,
                "param_norm": e["param_sq"] ** 0.5,
                "swallowed_share": e["swallowed"] / max(e["with_grad"], 1),
                "with_grad_share": e["with_grad"] / e["numel"],
            }
            for g, e in out.items()
        }
        self.record["init"] = {
            g: {
                "moved": e["init_sq"] ** 0.5,
                "init_norm": e["init0_sq"] ** 0.5,
                "rel_moved": (e["init_sq"] / e["init0_sq"]) ** 0.5 if e["init0_sq"] > 0 else None,
            }
            for g, e in out.items()
        }
        self.record["clip"] = {k: _f(v) for k, v in clip_info.items() if isinstance(v, (int, float))}
        if self.zgrads:
            g = torch.cat([z.reshape(-1, z.shape[-1]) for z in self.zgrads])
            mean = g.mean(0)
            per = g.pow(2).sum(1)
            self.record["zgrad"] = {
                "cohorts": g.shape[0],
                "norm_mean_cohort": _f(per.sqrt().mean()),
                "shared_share": _f(mean.pow(2).sum() / per.mean().clamp_min(1e-300)),
            }
        self.before = None
        self.nonzero = None
        self.active = False

    # ------------------------------------------------------------- probe
    @torch.no_grad()
    def capture(self, rows):
        """Stage tensors of the generator for `rows` (one row per cohort)."""
        gen, corpus, device = self.generator, self.runtime.corpus, self.runtime.device
        stages: dict = {}
        was_training = gen.training
        gen.eval()
        contexts = [corpus.context([r], "last", device) for r in rows]
        summaries = torch.stack([c[0][0] for c in contexts]).float()
        stages["summary"] = summaries
        encoded = torch.stack([gen.encode_examples(*c)[0] for c in contexts])
        stages["encoder"] = encoded.float()
        pooled = torch.stack([gen.pool_examples(gen.encode_examples(*c)) for c in contexts])
        hidden_cond = gen.condition[1](gen.condition[0](pooled))
        z = gen.condition[2](hidden_cond)
        stages["cond_hidden"] = hidden_cond.float()
        stages["z"] = z.float()
        enc_hidden = torch.stack([gen.encoder[1](gen.encoder[0](gen.encoder_input(*c)))[0] for c in contexts])
        stages["enc_hidden"] = enc_hidden.float()
        decoded, heads_out = {}, {}
        n = len(rows)
        for name in gen.shapes:
            module_type, layer = gen.layout[name]
            descriptor = torch.cat(
                [
                    z,
                    gen.module_embedding.weight[gen.module_types.index(module_type)].expand(n, -1),
                    gen.layer_embedding.weight[layer].expand(n, -1),
                ],
                -1,
            )
            first = gen.decoder[1](gen.decoder[0](descriptor))
            d = gen.decoder[2:](first) if len(gen.decoder) > 2 else first
            decoded[name] = (first.float(), d.float())
            head = gen.heads[name if gen.head_sharing == "per_module" else module_type]
            heads_out[name] = {}
            for part in ("A", "B"):
                lin = head[part]
                dbar = d.mean(0, keepdim=True)
                bias = lin.bias if lin.bias is not None else gen.biases[f"{name}__{part}"]
                static = (dbar @ lin.weight.T).float() + bias.float()
                dynamic = ((d - dbar) @ lin.weight.T).float()
                heads_out[name][part] = (static[0], dynamic, bias.float())
        stages["decoded"] = decoded
        stages["heads"] = heads_out
        stages["factors"] = gen.forward_cohorts(contexts)
        if was_training:
            gen.train()
        return stages

    @torch.no_grad()
    def probe_metrics(self):
        from speculators.generator.offline_eval import geometry, gram  # noqa: PLC0415

        scales = {k: self.runtime.drafter.get_submodule(p).scale for k, p in self.runtime.paths.items()}
        result, factors_by_split = {}, {}
        for split, rows in self.probe.items():
            labels = self.probe_labels[split]
            st = self.capture(rows)
            res = {}
            for key in ("summary", "enc_hidden", "encoder", "cond_hidden", "z"):
                res[key] = spread_stats(st[key], labels)
                if key in ("enc_hidden", "cond_hidden"):
                    x = st[key]
                    res[key]["dead_share"] = _f(((x.std(0) < 1e-4 * (x.abs().mean() + 1e-12))).float().mean())
            # summary: per aux-layer norms (5 × 4096 concatenated)
            s = st["summary"]
            width = s.shape[1] // 5 if s.shape[1] % 5 == 0 else None
            if width:
                res["summary_layer_norms"] = [_f(s[:, i * width:(i + 1) * width].norm(dim=1).mean()) for i in range(5)]
            # decoder (mean over modules)
            dec = {"dec_first": [], "dec_out": []}
            for first, d in st["decoded"].values():
                dec["dec_first"].append(spread_stats(first, labels))
                dec["dec_out"].append(spread_stats(d, labels))
                if "dec_dead" not in res:
                    res["dec_dead"] = _f((first.std(0) < 1e-4 * (first.abs().mean() + 1e-12)).float().mean())
            for key, items in dec.items():
                res[key] = {m: sum(i[m] for i in items) / len(items) for m in items[0]}
            # head static / dynamic split, per part, pooled over modules
            for part in ("A", "B"):
                st_sq = dyn_sq = bias_sq = 0.0
                for heads in st["heads"].values():
                    static, dynamic, bias = heads[part]
                    st_sq += float(static.pow(2).sum())
                    dyn_sq += float(dynamic.pow(2).sum(1).mean())
                    bias_sq += float(bias.pow(2).sum())
                res[f"head{part}"] = {
                    "static_norm": st_sq**0.5,
                    "dynamic_rms": dyn_sq**0.5,
                    "dynamic_over_static": (dyn_sq / max(st_sq, 1e-300)) ** 0.5,
                    "bias_norm": bias_sq**0.5,
                }
            # factors and ΔW geometry
            facs = {k: (a.float(), b.float()) for k, (a, b) in st["factors"].items()}
            factors_by_split[split] = facs
            a_all = torch.cat([a.reshape(len(rows), -1) for a, _ in facs.values()], 1)
            b_all = torch.cat([b.reshape(len(rows), -1) for _, b in facs.values()], 1)
            res["A"] = spread_stats(a_all, labels)
            res["B"] = spread_stats(b_all, labels)
            per_row = [{k: (v[0][i], v[1][i]) for k, v in facs.items()} for i in range(len(rows))]
            grams = {k: gram([f[k] for f in per_row], scales[k]) for k in facs}
            total = sum(grams.values())
            geo = geometry(total, labels)
            res["dW"] = {k: _f(v) for k, v in geo.items() if isinstance(v, (int, float)) or torch.is_tensor(v) and v.numel() == 1}
            res["dW"]["mean_norm"] = _f(total.diagonal().mean().sqrt())
            result[split] = res
        # movement of the mean ΔW and of the dynamic parts since the last probe
        val = factors_by_split["val"]
        mean_dw = {k: scales[k] * torch.einsum("nor,nri->oi", b, a) / a.shape[0] for k, (a, b) in val.items()}
        if self.previous_probe is not None:
            prev = self.previous_probe
            num = sum(float((mean_dw[k] - prev[k]).pow(2).sum()) for k in mean_dw)
            den = sum(float(mean_dw[k].pow(2).sum()) for k in mean_dw)
            result["mean_dW_change"] = (num / max(den, 1e-300)) ** 0.5
        self.previous_probe = {k: v.to(torch.bfloat16) for k, v in mean_dw.items()}
        return result

    def after_update(self, update, history):
        if update in self.snapshots:
            state = {k: v.detach().cpu() for k, v in self.generator.state_dict().items()}
            torch.save({"update": update, "model": state}, self.dir / "snapshots" / f"u{update:05d}.pt")
        if not self.record or self.record.get("update") != update:
            return
        started = time.perf_counter()
        self.record["probe"] = self.probe_metrics()
        self.record["probe_seconds"] = time.perf_counter() - started
        if history and history[-1]["update"] == update:
            self.record["validation_score"] = history[-1].get("score")
        self._write(self.record)
        self.record = {}

    def _write(self, value):
        self.out.write(json.dumps(value, default=float) + "\n")
        self.out.flush()
