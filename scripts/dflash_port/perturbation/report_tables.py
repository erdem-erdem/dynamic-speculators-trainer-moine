"""Print the Markdown tables of PERTURB24K.md from results.json (no hand-copied numbers).

    python scripts/dflash_port/perturbation/report_tables.py > /tmp/tables.md
"""

import json
from pathlib import Path

R = json.loads(Path("/workspace/yusuf/runs/dflash/perturb24k/results/results.json").read_text())
CATS = ("math", "code", "stem", "chat")
FREE = {"static24k_kl": "24K static (KL, r32 attn, 4,000 upd.)", "generator24k_kl": "24K generator (KL)",
        "static10k_math_ce": "10K-math static (CE)", "static4k_m0_ce": "4K M0 static (CE)"}
RUNS = {"full": "full fine-tune (1.05B)", "lora_attn_r512": "LoRA r512 attention (67M)",
        "lora_ffn_r512": "LoRA r512 FFN (126M)"}


def f(v, d=3):
    if v is None:
        return "–"
    a = abs(v)
    if a and (a < 1e-3 or a >= 1e4):
        return f"{v:.1e}"
    return f"{v:.{d}g}"


def s(v, d=3):
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):.{d}f}"


def pct(v, d=2):
    return f"{100 * v:.{d}f}%"


def ci(c):
    return f"[{s(c[0], 3)}, {s(c[1], 3)}]"


def tab_free_t2():
    print("| Run | update | rel. ‖ΔW‖ | KL(p_base‖p) | TV | flips | Δτ [95% CI] | Δ full_acc | Δ KL(p_T‖p) | fixes | breaks | net |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run, d in R["t2"].items():
        rel = d.get("rel_norm") or [None] * len(d["history"])
        for h, r in zip(d["history"], rel):
            if h["update"] == 0:
                continue
            a = h["all"]
            print(f"| {RUNS[run]} | {h['update']} | {f(r)} | {f(a['kl12'])} | {f(a['tv12'])} | {pct(a['flip'], 1)} | "
                  f"{s(a['d_tau'])} {ci(h['ci'])} | {s(a['d_acc'], 4)} | {s(a['d_klT'])} | {pct(a['fix'])} | {pct(a['brk'])} | {s(100 * a['net'], 2)} pp |")
    for k, v in R.get("free", {}).items():
        a = v["all"]
        rel = R.get("free_rel_norm", {}).get(k)
        print(f"| {FREE[k]} | final | {f(rel) if rel else 'per row'} | {f(a['kl12'])} | {f(a['tv12'])} | {pct(a['flip'], 1)} | "
              f"{s(a['d_tau'])} {ci(v['ci'])} | {s(a['d_acc'], 4)} | {s(a['d_klT'])} | {pct(a['fix'])} | {pct(a['brk'])} | {s(100 * a['net'], 2)} pp |")


def tab_val500():
    v = R["verify"]["scores"]
    print("| Model (500 validation rows) | full_acc | τ |")
    print("|---|---|---|")
    print(f"| base (this harness) | {v['base']['full_acc']['all']:.4f} | {v['base']['eal']['all']:.3f} |")
    print(f"| 24K static (this harness) | {v['static']['full_acc']['all']:.4f} | {v['static']['eal']['all']:.3f} |")
    for run, d in R["t2"].items():
        if "validation500" in d:
            x = d["validation500"]
            print(f"| {RUNS[run]}, 600 updates | {x['full_acc']['all']:.4f} | {x['eal']['all']:.3f} |")


def tab1(point):
    P = R["points"][point]
    print(f"| Location | Fisher trace | ½‖W‖²·tr(F)/n | ε: KL=0.01 | ε: flips floor+1pp | KL @ ε=0.3 | ‖g‖·‖W‖ | Δτ step 3e-4 (desc / asc) | best step Δτ (ε) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for loc, d in P["dose"].items():
        fi = P.get("fisher", {}).get(loc, {})
        st = P.get("steps", {}).get(loc)
        kl03 = d["kl12"][d["eps"].index(0.3)]
        if st:
            best = max(st["descent"].items(), key=lambda kv: kv[1]["d_tau"])
            g = f(st["grad_norm"] * st["weight_norm"])
            step = f"{s(st['descent']['0.0003']['d_tau'])} / {s(st['ascent']['0.0003']['d_tau'])}"
            b = f"{s(best[1]['d_tau'])} ({best[0]})"
        else:
            g = step = b = "–"
        print(f"| {loc} | {f(fi.get('trace'))} | {f(fi.get('rand_curv'))} | {f(d['eps_kl001'], 2) if d['eps_kl001'] else '> 1'} | "
              f"{f(d['eps_flip1pp'], 2) if d['eps_flip1pp'] else '> 1'} | {f(kl03)} | {g} | {step} | {b} |")


def tab3():
    print("| θ₁ | location · ε | r32 | r64 | r128 | r512 | r1024 | dense |")
    print("|---|---|---|---|---|---|---|---|")
    for point in ("base", "static"):
        for k, row in sorted(R["points"].get(point, {}).get("rank", {}).items()):
            keys = ["32", "64", "128", "512", "1024", "full"]
            print(f"| {point} | {k.replace('|', ' · ε=')} | " + " | ".join(
                f"{f(row[r]['kl12'])} / {s(row[r]['d_tau'], 3)}" for r in keys) + " |")


def tab4():
    print("| θ₁ | location | mean gradient cosine (raw / ÷ noise ceiling) | own-category step | other-category steps | shared step |")
    print("|---|---|---|---|---|---|")
    for point in ("base", "static"):
        P = R["points"].get(point, {})
        for loc, cross in P.get("cross", {}).items():
            g = P["geometry"][loc]
            raw = sum(g["cos"].values()) / 6
            cor = sum(g["cos"][k] / g["cos_ceiling"][k] for k in g["cos"]) / 6
            own = sum(-cross[c][c]["d_klT"] for c in CATS) / 4
            other = sum(-cross[x][c]["d_klT"] for c in CATS for x in CATS if x != c) / 12
            shared = sum(-cross["all"][c]["d_klT"] for c in CATS) / 4
            print(f"| {point} | {loc} | {raw:.3f} / {cor:.2f} | {1000 * own:+.3f} | {1000 * other:+.3f} | {1000 * shared:+.3f} |")


def tab4_tau():
    print("| θ₁ | location | Δτ own-category step | Δτ other-category steps | Δτ shared step |")
    print("|---|---|---|---|---|")
    for point in ("base", "static"):
        P = R["points"].get(point, {})
        for loc, cross in P.get("cross", {}).items():
            own = sum(cross[c][c]["d_tau"] for c in CATS) / 4
            other = sum(cross[x][c]["d_tau"] for c in CATS for x in CATS if x != c) / 12
            shared = sum(cross["all"][c]["d_tau"] for c in CATS) / 4
            print(f"| {point} | {loc} | {s(own)} | {s(other)} | {s(shared)} |")


def floors():
    for point in ("base", "static"):
        P = R["points"].get(point, {})
        if "floor" in P:
            fl = P["floor"]
            print(f"- floor {point}: KL {f(fl['kl12'])}, TV {f(fl['tv12'])}, flips {pct(fl['flip'])}, Δτ {s(fl['d_tau'])} "
                  f"(CIs {', '.join(ci(c) for c in P['floor_ci'])})")


def pareto():
    for point in ("base", "static"):
        fr = R["pareto"].get(point, [])
        print(f"- pareto {point}: " + "; ".join(
            f"{p['kind']}:{p['location']}@{p.get('eps', p.get('update', ''))} KL {f(p['kl'])} Δτ {s(p['d_tau'])}" for p in fr))


for name, fn in (("free_t2", tab_free_t2), ("val500", tab_val500), ("tab1_base", lambda: tab1("base")),
                 ("tab1_static", lambda: tab1("static")), ("tab3", tab3), ("tab4", tab4), ("tab4_tau", tab4_tau),
                 ("floors", floors), ("pareto", pareto)):
    print(f"\n<!-- {name} -->")
    try:
        fn()
    except (KeyError, ValueError) as e:
        print(f"(unavailable: {e!r})")
