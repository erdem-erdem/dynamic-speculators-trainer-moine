"""Markdown tables for reporting/dflash/ABL24K.md from abl24k_collect.py output.

    python scripts/dflash_port/abl24k_page/tables.py RESULTS.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import ARMS  # noqa: E402

CATS = ("math", "code", "stem", "chat")


def p(n):
    if not n:
        return "–"
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def d(x):
    if not x:
        return "–"
    t = x["tau"]
    return f"{t['delta']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]"


def main():
    R = json.loads(Path(sys.argv[1]).read_text())
    runs, base = R["runs"], R["base"]
    print("| Arm | trainable | train min | val τ | test τ | OOD τ | test Δτ vs static r32 | val Δτ | OOD Δτ |")
    print("|---|---|---|---|---|---|---|---|---|")
    print(f"| base drafter | 0 | – | {base['validation']['all']['tau']:.3f} | {base['test']['all']['tau']:.3f} | {base['ood']['all']['tau']:.3f} | – | – | – |")
    for a in ARMS:
        r = runs.get(a[0])
        if not r or "test" not in r:
            continue
        ref = a[0] == "static_kl_r32"
        cell = lambda s: "ref" if ref else d(r.get(s, {}).get("vs_reference"))  # noqa: E731
        print(f"| {a[1]} | {p(r.get('trainable_parameters'))} | {r.get('training_seconds', 0) / 60:.0f} | "
              f"{r['validation']['all']['tau']:.3f} | {r['test']['all']['tau']:.3f} | {r['ood']['all']['tau']:.3f} | "
              f"{cell('test')} | {cell('validation')} | {cell('ood')} |")
    print()
    print("Test τ per category (Δ vs static r32 in parentheses):\n")
    print("| Arm | " + " | ".join(CATS) + " |")
    print("|---|" + "---|" * len(CATS))
    ref = runs["static_kl_r32"]["test"]["categories"]
    print("| base drafter | " + " | ".join(f"{base['test']['categories'][c]['tau']:.3f}" for c in CATS) + " |")
    for a in ARMS:
        r = runs.get(a[0])
        if not r or "test" not in r:
            continue
        cc = r["test"]["categories"]
        print(f"| {a[1]} | " + " | ".join(f"{cc[c]['tau']:.3f} ({cc[c]['tau'] - ref[c]['tau']:+.3f})" for c in CATS) + " |")
    print()
    print("Generators (D13 similarity on 500 validation rows, D14 cost):\n")
    print("| Arm | params | cross-row cosine | within-category cosine | dyn_ratio | effective rank | η² | centroid acc | ms / LoRA |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        r = runs.get(a[0])
        if not r or not r.get("similarity"):
            continue
        s, c = r["similarity"]["all"], r.get("cost") or {}
        print(f"| {a[1]} | {p(r.get('trainable_parameters'))} | {s['cos_cross']:.3f} | {s['cos_in']:.3f} | {s['dyn_ratio']:.3f} | "
              f"{s['effective_rank']:.2f} | {s['eta2']:.2f} | {s['centroid_accuracy']:.3f} | {c.get('ms_per_call', 0):.2f} |")
    print()
    print("LoRA coefficient at inference (validation full_acc; training scale 2):\n")
    ks = ["0.5", "1", "2", "4", "8"]
    print("| Arm | " + " | ".join(ks) + " |")
    print("|---|" + "---|" * len(ks))
    for a in ARMS:
        r = runs.get(a[0])
        if not r or not r.get("coefficients"):
            continue
        c = r["coefficients"]
        best = max(c, key=c.get)
        print(f"| {a[1]} | " + " | ".join((f"**{c[k]:.4f}**" if k == best else f"{c[k]:.4f}") if k in c else "–" for k in ks) + " |")


if __name__ == "__main__":
    main()
