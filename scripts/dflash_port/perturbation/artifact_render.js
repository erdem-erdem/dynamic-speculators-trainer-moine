/* Renders every figure and table of the PERTURB24K page from DATA. */
(function () {
  const V = window.Viz, css = V.css, fmt = V.fmt, signed = V.signed;
  const $ = (s) => document.querySelector(s);
  const CATS = ["math", "code", "stem", "chat"];
  const MODS = ["q", "k", "v", "o", "gate", "up", "down"];
  const S = (i) => css(`--s${i}`);
  const state = { f1: "base", f1m: "kl12", f1v: "groups", f2eps: 0.3, f3: "base", f6: "base", f8: "base", f8loc: "all_attn",
    f9: "base", f9m: "sens", t1: "base" };
  const ptLabel = { base: "base drafter", static: "24K static" };
  const kindLabel = { dense: "random dense ΔW", lowrank: "random low-rank ΔW", descent: "one gradient step (descent)",
    ascent: "one gradient step (ascent)", train: "short training (T2)", free: "existing adapters" };
  const freeLabel = { static24k_kl: "24K static (KL)", static10k_math_ce: "10K-math static (CE)",
    static4k_m0_ce: "4K M0 static (CE)", generator24k_kl: "24K generator (KL)" };
  const runLabel = { full: "full fine-tune", lora_attn_r512: "LoRA r512 attention", lora_ffn_r512: "LoRA r512 FFN" };
  const pct = (v, d = 2) => (v * 100).toFixed(d) + "%";
  const pp = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d) + " pp";

  function seg(id, key, options, onChange) {
    const box = $(id);
    if (!box) return;
    box.innerHTML = "";
    box.className = "seg";
    for (const [value, label] of options) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.setAttribute("aria-pressed", String(state[key] === value));
      b.addEventListener("click", () => {
        state[key] = value;
        for (const c of box.children) c.setAttribute("aria-pressed", String(c === b));
        onChange();
      });
      box.appendChild(b);
    }
  }
  const pts = [["base", "θ₁ = base"], ["static", "θ₁ = 24K static"]];
  const bestDescent = (s) => Object.entries(s.descent).reduce((a, b) => (b[1].d_tau > a[1].d_tau ? b : a));

  /* ---------- F3: movement vs gain ---------- */
  const GROUPS3 = {
    dense: { color: S(1), shape: "circle", size: 2.8, opacity: 0.55 },
    lowrank: { color: S(7), shape: "square", size: 2.6, opacity: 0.6 },
    ascent: { color: S(8), shape: "tridown", size: 3.2, opacity: 0.7 },
    descent: { color: S(3), shape: "triangle", size: 3.6, opacity: 0.85 },
    train: { color: S(2), shape: "diamond", size: 4.5, opacity: 0.95 },
    free: { color: S(4), shape: "star", size: 4.2, opacity: 1 },
  };
  function pointHtml(s) {
    const what = s.kind === "free" ? freeLabel[s.location] : s.kind === "train" ? `${runLabel[s.location]}, update ${s.update}`
      : `${kindLabel[s.kind]} · ${s.location} · ε=${s.eps}${s.rank ? " · rank " + s.rank : ""}${s.seed !== undefined ? " · seed " + s.seed : ""}`;
    const ci = s.ci ? ` <span class="tip-mute">[${signed(s.ci[0], 2)}, ${signed(s.ci[1], 2)}]</span>` : "";
    return `<div class="tip-h">${what}</div>movement KL: <b>${fmt(s.kl)}</b> · flips ${pct(s.flip, 1)}<br>` +
      `Δτ: <b>${signed(s.d_tau, 3)}</b>${ci}<br>fixes ${pct(s.fix)} · breaks ${pct(s.brk)}`;
  }
  function f3() {
    const p = state.f3;
    const points = DATA.scatter.filter((s) => s.point === p && s.kl > 0).map((s) => ({ x: s.kl, y: s.d_tau, group: s.kind, html: pointHtml(s),
      label: s.kind === "free" && s.location === "static24k_kl" ? "24K static" : s.kind === "free" && s.location === "generator24k_kl" ? "generator" : "" }));
    const groups = {};
    for (const k of Object.keys(GROUPS3)) if (points.some((q) => q.group === k)) groups[k] = Object.assign({ label: kindLabel[k] }, GROUPS3[k]);
    V.scatter($("#f3"), { points, groups, logx: true, width: 900, height: 430, ydomain: [-0.45, 0.3],
      xlabel: "movement: KL(p₁ ‖ p₂) per draft position, nats (log)", ylabel: "gain: Δτ vs θ₁ (accepted length per block)",
      lines: [{ name: "Pareto front (best gain at ≤ movement)", color: css("--fg"), points: DATA.pareto[p].map((q) => ({ x: q.kl, y: q.d_tau })), step: true, extend: true, width: 1.4 }],
      label: "Movement versus gain scatter" });
  }

  /* ---------- F1: dose-response ---------- */
  const groupLocs = ["fc", "all_attn", "all_ffn", "whole", "layer1", "layer3", "layer5"];
  function f1() {
    const P = DATA.points[state.f1];
    if (!P || !P.dose) return;
    const m = state.f1m;
    const locs = state.f1v === "groups" ? groupLocs : [0, 1, 2, 3, 4].map((i) => `l${i}_${state.f1v}`);
    const names = state.f1v === "groups" ? locs : locs.map((l, i) => `layer ${i + 1} ${state.f1v}`);
    const logy = m !== "d_tau";
    const series = locs.map((loc, i) => {
      const d = P.dose[loc];
      const fl = (a) => (logy ? a.map((v) => Math.max(v, 1e-5)) : a);
      return { name: names[i], color: S(i + 1), x: d.eps, y: fl(d[m]), lo: fl(d[m + "_lo"]), hi: fl(d[m + "_hi"]) };
    });
    const labels = { kl12: "KL(p₁‖p₂) per position", tv12: "total variation", flip: "top-1 flip rate", d_tau: "Δτ vs θ₁" };
    V.line($("#f1"), { series, logx: true, logy, width: 900, height: 360, xname: "ε", xlabel: "relative size ε = ‖ΔW‖_F / ‖W‖_F (log)",
      ylabel: labels[m] + (logy ? " (log)" : ""), zero: !logy,
      refs: m !== "d_tau" ? [{ y: Math.max(P.floor[m], 1e-5), label: "bf16 floor (ε = 1e-4)" }] : [],
      tipfmt: m === "flip" ? (v) => pct(v) : (v) => fmt(v) });
  }

  /* ---------- F2: heatmaps ---------- */
  function f2() {
    const cols = MODS.concat(["kv@ctx", "kv@blk"]);
    const rows = [1, 2, 3, 4, 5].map((i) => `layer ${i}`);
    const all = [];
    const vals = {};
    for (const p of ["base", "static"]) {
      const P = DATA.points[p];
      if (!P || !P.dose) continue;
      vals[p] = rows.map((_, i) => cols.map((c) => { const d = P.dose[`l${i}_${c}`]; return d.kl12[d.eps.indexOf(state.f2eps)]; }));
      all.push(...vals[p].flat());
    }
    const lo = Math.log10(Math.min(...all)), hi = Math.log10(Math.max(...all));
    const r = V.ramp(["#eef4fc", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]);
    for (const p of ["base", "static"]) {
      if (!vals[p]) continue;
      V.heatmap($(`#f2-${p}`), { rows, cols, values: vals[p], color: (v) => r((Math.log10(v) - lo) / (hi - lo || 1)),
        fmt: (v) => fmt(v, 2), cellW: 52, label: `KL heatmap ${p}`,
        tip: (i, j, v) => `${ptLabel[p]} · layer ${i + 1} · ${cols[j]}<br>KL(p₁‖p₂) at ε=${state.f2eps}: <b>${fmt(v)}</b>` });
      const fc = DATA.points[p].dose.fc;
      $(`#f2-${p}-fc`).textContent = `fc at the same ε: ${fmt(fc.kl12[fc.eps.indexOf(state.f2eps)])}`;
    }
  }

  /* ---------- F4: churn ---------- */
  function churnItems() {
    const items = [];
    const add = (label, s, note) => s && items.push({ label, fix: s.fix, brk: s.brk, kl: s.kl ?? s.kl12, d_tau: s.d_tau, note });
    for (const [k, v] of Object.entries(DATA.free)) add(freeLabel[k], v.all);
    for (const [k, v] of Object.entries(DATA.t2)) add(`${runLabel[k]} (600)`, v.history[v.history.length - 1].all);
    const B = DATA.points.base;
    if (B.steps) for (const loc of ["all_attn", "all_ffn", "whole"]) {
      const e = B.steps[loc] && bestDescent(B.steps[loc]);
      if (e) add(`best step ${loc} ε=${e[0]}`, e[1]);
    }
    if (B.dose) for (const [loc, eps] of [["all_attn", 0.3], ["all_ffn", 0.3], ["whole", 0.1]]) {
      const d = B.dose[loc], i = d.eps.indexOf(eps);
      add(`random ${loc} ε=${eps}`, { fix: d.fix[i], brk: d.brk[i], kl12: d.kl12[i], d_tau: d.d_tau[i] });
    }
    add("bf16 floor (ε=1e-4)", B.floor);
    return items;
  }
  function f4() {
    const items = churnItems();
    V.bars($("#f4"), {
      groups: items.map((it) => ({ label: it.label, values: [it.fix * 100, it.brk * 100], note: `net ${pp(it.fix - it.brk)}`,
        html: `<div class="tip-h">${it.label}</div>fixes ${pct(it.fix)} · breaks ${pct(it.brk)}<br>net ${pp(it.fix - it.brk)} · KL ${fmt(it.kl)} · Δτ ${signed(it.d_tau, 3)}` })),
      series: [{ name: "fixes (wrong → right)", color: S(3) }, { name: "breaks (right → wrong)", color: S(8) }],
      width: 900, height: 360, ylabel: "% of draft positions", yfmt: (v) => fmt(v, 2), label: "Churn bars" });
  }

  /* ---------- F5: per-slot ---------- */
  function f5() {
    const prof = [];
    if (DATA.free.static24k_kl) prof.push(["24K static (KL)", DATA.free.static24k_kl.slot]);
    if (DATA.free.generator24k_kl) prof.push(["24K generator (KL)", DATA.free.generator24k_kl.slot]);
    if (DATA.t2.full) { const h = DATA.t2.full.history; prof.push(["full fine-tune (600)", h[h.length - 1].slot]); }
    const B = DATA.points.base;
    if (B["slot_dense_0.3"]) prof.push(["random all_attn ε=0.3", B["slot_dense_0.3"].all_attn]);
    if (B.slot_descent && B.slot_descent.all_attn) prof.push(["descent all_attn ε=3e-4", B.slot_descent.all_attn]);
    const x = Array.from({ length: 15 }, (_, i) => i + 1);
    const mk = (key, scaleF = 1) => prof.map(([n, s], i) => ({ name: n, color: S(i + 1), x, y: s[key].map((v) => v * scaleF) }));
    V.line($("#f5a"), { series: mk("kl12"), width: 440, height: 280, xname: "slot", xlabel: "draft slot", ylabel: "KL(p_base‖p)", logy: true,
      xticks: [1, 3, 5, 7, 9, 11, 13, 15] });
    V.line($("#f5b"), { series: mk("d_acc", 100), width: 440, height: 280, xname: "slot", xlabel: "draft slot", ylabel: "Δ accuracy (pp)", zero: true,
      xticks: [1, 3, 5, 7, 9, 11, 13, 15], tipfmt: (v) => fmt(v, 2) + " pp" });
  }

  /* ---------- F6: rank ---------- */
  function f6() {
    const P = DATA.points[state.f6];
    if (!P || !P.rank) return;
    const keys = Object.keys(P.rank).sort();
    const series = keys.map((k, i) => {
      const row = P.rank[k];
      const ranks = Object.keys(row).filter((r) => r !== "full").map(Number).sort((a, b) => a - b);
      return { name: k.replace("|", " ε="), color: S((i % 8) + 1), dash: k.endsWith("0.1") ? "" : "5 3",
        x: ranks.concat([4096]), y: ranks.map((r) => row[String(r)].kl12).concat([row.full.kl12]) };
    });
    V.line($("#f6"), { series, logx: true, logy: true, width: 900, height: 340, xname: "rank", xlabel: "rank of ΔW at fixed ‖ΔW‖ (4096 = dense Gaussian)",
      ylabel: "KL(p₁‖p₂) (log)", xticks: [32, 64, 128, 512, 1024, 4096] });
  }

  /* ---------- F7: training paths ---------- */
  function f7() {
    const points = [], lines = [];
    let i = 0;
    for (const [run, d] of Object.entries(DATA.t2)) {
      const col = S([2, 7, 5][i++]);
      const h = d.history.filter((x) => x.all.kl12 > 0);
      lines.push({ name: runLabel[run], color: col, points: h.map((x) => ({ x: x.all.kl12, y: x.all.d_tau })), width: 2 });
      for (const x of h) points.push({ x: x.all.kl12, y: x.all.d_tau, group: run, label: String(x.update),
        html: `<div class="tip-h">${runLabel[run]} · update ${x.update}</div>KL ${fmt(x.all.kl12)} · flips ${pct(x.all.flip, 1)}<br>Δτ <b>${signed(x.all.d_tau, 3)}</b> [${signed(x.ci[0], 2)}, ${signed(x.ci[1], 2)}]` });
    }
    for (const [k, v] of Object.entries(DATA.free)) points.push({ x: v.all.kl12, y: v.all.d_tau, group: "free", label: freeLabel[k],
      html: `<div class="tip-h">${freeLabel[k]}</div>KL ${fmt(v.all.kl12)} · Δτ <b>${signed(v.all.d_tau, 3)}</b> [${signed(v.ci[0], 2)}, ${signed(v.ci[1], 2)}]` });
    const front = (DATA.pareto_by_kind.base || {}).descent || [];
    lines.push({ name: "one gradient step: best Δτ at ≤ movement", color: S(3), points: front.map((q) => ({ x: q.kl, y: q.d_tau })), step: true, dash: "4 3" });
    const groups = {};
    i = 0;
    for (const run of Object.keys(DATA.t2)) groups[run] = { color: S([2, 7, 5][i++]), shape: "diamond", size: 4, label: runLabel[run] };
    groups.free = { color: S(4), shape: "star", size: 4.2, label: "existing adapters (3 epochs)" };
    V.scatter($("#f7"), { points, groups, lines, logx: true, width: 900, height: 400, xlabel: "KL(p_base ‖ p) per position (log)",
      ylabel: "Δτ vs base", label: "Training paths" });
  }

  /* ---------- F8: categories ---------- */
  function f8() {
    const P = DATA.points[state.f8];
    if (!P || !P.cross) return;
    const loc = state.f8loc;
    const cross = P.cross[loc];
    if (!cross) return;
    const srcs = CATS.concat(["all"]).filter((s) => cross[s]);
    const vals = srcs.map((s) => CATS.map((c) => -cross[s][c].d_klT * 1000));
    const flat = vals.flat(), mx = Math.max(...flat.map(Math.abs));
    const pos = V.ramp(["#f0efec", "#b7d3f6", "#5598e7", "#1c5cab"]), neg = V.ramp(["#f0efec", "#f6c3b0", "#e88a64", "#c0441d"]);
    V.heatmap($("#f8"), { rows: srcs.map((s) => (s === "all" ? "step on all" : `step on ${s}`)), cols: CATS.map((c) => `on ${c}`), values: vals,
      color: (v) => (v >= 0 ? pos(v / mx) : neg(-v / mx)), fmt: (v) => fmt(v, 2), labelW: 92, cellW: 70,
      tip: (i, j, v) => { const e = cross[srcs[i]][CATS[j]]; return `gradient step on <b>${srcs[i]}</b> rows (ε=3·10⁻⁴, ${loc}),<br>evaluated on ${CATS[j]}:<br>KL to target ${signed(e.d_klT * 1000, 3)} ×10⁻³ · Δτ ${signed(e.d_tau, 3)}`; },
      label: "Category cross-gain matrix" });
    const g = P.geometry && P.geometry[loc];
    if (g) {
      const rows = Object.keys(g.cos).map((k) => `<tr><td>${k.replace("|", " · ")}</td><td>${g.cos[k].toFixed(3)}</td><td>${g.cos_ceiling[k].toFixed(3)}</td></tr>`).join("");
      $("#f8-cos").innerHTML = `<table><thead><tr><th>category pair (${loc})</th><th>gradient cosine</th><th>noise ceiling</th></tr></thead><tbody>${rows}</tbody></table>`;
    }
  }

  /* ---------- F9: drafter diagram ---------- */
  function f9() {
    const P = DATA.points[state.f9];
    const box = $("#f9");
    box.innerHTML = "";
    if (!P || !P.dose) return;
    const metric = state.f9m;
    const value = (loc) => {
      if (metric === "sens") { const d = P.dose[loc]; return d ? d.kl12[d.eps.indexOf(0.3)] : null; }
      if (metric === "fisher") return P.fisher && P.fisher[loc] ? P.fisher[loc].rand_curv : null;
      if (metric === "gain") { const s = P.steps && P.steps[loc]; return s ? bestDescent(s)[1].d_tau : null; }
      return null;
    };
    const locs = ["fc"];
    for (let i = 0; i < 5; i++) for (const m of ["q", "k@ctx", "k@blk", "v@ctx", "v@blk", "o", "gate", "up", "down"]) locs.push(`l${i}_${m}`);
    const getVal = (loc) => {
      if (loc.includes("@")) { const [l, rest] = loc.split("_"); const [m, path] = rest.split("@"); return value(`${l}_kv@${path}`); }
      return value(loc);
    };
    const vals = locs.map(getVal).filter((v) => v !== null && v !== undefined);
    const logScale = metric !== "gain";
    const tr = (v) => (logScale ? Math.log10(Math.max(v, 1e-9)) : v);
    const lo = Math.min(...vals.map(tr)), hi = Math.max(...vals.map(tr));
    const seq = V.ramp(["#eef4fc", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]);
    const div = V.ramp(["#c0441d", "#e88a64", "#f0efec", "#5598e7", "#1c5cab"]);
    const mxAbs = Math.max(...vals.map(Math.abs));
    const color = (v) => (v === null ? [css("--chip"), css("--muted")] : metric === "gain" ? div(0.5 + 0.5 * v / (mxAbs || 1)) : seq((tr(v) - lo) / (hi - lo || 1)));
    const W = 900, H = 330, el = V.el;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "viz", role: "img", "aria-label": "Drafter diagram" }, box);
    const fmtv = (v) => (v === null ? "–" : metric === "gain" ? signed(v, 2) : fmt(v, 2));
    const cell = (x, y, w, h, label, loc) => {
      const v = getVal(loc);
      const [fill, ink] = color(v);
      const r = el("rect", { x, y, width: w, height: h, rx: 4, fill, stroke: css("--line") }, svg);
      const t = el("text", { x: x + w / 2, y: y + h / 2 - 2, "text-anchor": "middle", class: "cell", fill: ink }, svg);
      t.textContent = label;
      const t2 = el("text", { x: x + w / 2, y: y + h / 2 + 10, "text-anchor": "middle", class: "tick", fill: ink }, svg);
      t2.textContent = fmtv(v);
      t.style.pointerEvents = t2.style.pointerEvents = "none";
      const names = { sens: "KL(p₁‖p₂) of a random ΔW at ε=0.3", fisher: "random-direction curvature ½‖W‖²·tr(F)/n", gain: "best Δτ of one descent step (ε 1e-4 … 3e-3)" };
      r.addEventListener("mousemove", (evt) => V.showTip(evt, `<div class="tip-h">${loc.replace("_", " · ")}</div>${names[metric]}: <b>${fmtv(v)}</b>`));
      r.addEventListener("mouseleave", V.hideTip);
    };
    const ink2 = css("--fg-2");
    const txt = (x, y, s, anchor = "middle") => { const t = el("text", { x, y, "text-anchor": anchor, class: "axlabel", fill: ink2 }, svg); t.textContent = s; };
    txt(58, 20, "target features");
    txt(58, 34, "(5 layers × 4096)");
    cell(18, 120, 80, 64, "fc", "fc");
    el("path", { d: "M98,152 H128", stroke: css("--axis"), "stroke-width": 2, fill: "none" }, svg);
    const colW = 150, x0 = 132;
    for (let i = 0; i < 5; i++) {
      const x = x0 + i * colW;
      txt(x + colW / 2 - 6, 20, `layer ${i + 1}`);
      el("rect", { x: x - 2, y: 28, width: colW - 8, height: H - 36, rx: 8, fill: "none", stroke: css("--line") }, svg);
      const w = (colW - 22) / 2;
      cell(x + 4, 40, colW - 18, 34, "q", `l${i}_q`);
      cell(x + 4, 80, w, 34, "k ctx", `l${i}_k@ctx`);
      cell(x + 8 + w, 80, w, 34, "k blk", `l${i}_k@blk`);
      cell(x + 4, 120, w, 34, "v ctx", `l${i}_v@ctx`);
      cell(x + 8 + w, 120, w, 34, "v blk", `l${i}_v@blk`);
      cell(x + 4, 160, colW - 18, 34, "o", `l${i}_o`);
      cell(x + 4, 206, w, 34, "gate", `l${i}_gate`);
      cell(x + 8 + w, 206, w, 34, "up", `l${i}_up`);
      cell(x + 4, 246, colW - 18, 34, "down", `l${i}_down`);
      const t = el("text", { x: x + colW / 2 - 6, y: H - 14, "text-anchor": "middle", class: "tick", fill: css("--muted") }, svg);
      t.textContent = "attention ↑ · FFN ↓";
    }
  }

  /* ---------- Tables ---------- */
  function tab1() {
    const P = DATA.points[state.t1];
    if (!P || !P.dose) return;
    const locs = Object.keys(P.dose);
    const rows = locs.map((loc) => {
      const d = P.dose[loc], f = P.fisher ? P.fisher[loc] : null, s = P.steps ? P.steps[loc] : null;
      const best = s ? bestDescent(s) : null;
      const cell = (v, cls = "") => `<td class="${cls}">${v}</td>`;
      const sg = (v) => (v === null || v === undefined ? "–" : `<span class="${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${signed(v, 2)}</span>`);
      return `<tr><td>${loc}</td>${cell(f ? fmt(f.trace, 3) : "–")}${cell(f && f.rand_curv ? fmt(f.rand_curv, 3) : "–")}` +
        `${cell(d.eps_kl001 ? fmt(d.eps_kl001, 2) : "> 1")}${cell(d.eps_flip1pp ? fmt(d.eps_flip1pp, 2) : "> 1")}` +
        `${cell(s ? fmt(s.grad_norm * s.weight_norm, 3) : "–")}${cell(s ? sg(s.descent[0.0003].d_tau) : "–")}${cell(s ? sg(s.ascent[0.0003].d_tau) : "–")}` +
        `${cell(best ? `${sg(best[1].d_tau)} <span class="tip-mute">@${best[0]}</span>` : "–")}</tr>`;
    }).join("");
    $("#tab1").innerHTML = `<table><thead><tr><th>location</th><th>Fisher trace</th><th>½‖W‖²tr(F)/n</th><th>ε for KL = 0.01</th><th>ε for flips floor+1pp</th>` +
      `<th>first-order gain ‖g‖‖W‖</th><th>Δτ descent ε=3e-4</th><th>Δτ ascent ε=3e-4</th><th>best descent Δτ</th></tr></thead><tbody>${rows}</tbody></table>`;
  }
  function tab2() {
    const rows = [];
    for (const [run, d] of Object.entries(DATA.t2)) for (const h of d.history) {
      const a = h.all;
      rows.push(`<tr><td>${runLabel[run]}</td><td>${h.update}</td><td>${fmt(a.kl12)}</td><td>${fmt(a.tv12)}</td><td>${pct(a.flip)}</td>` +
        `<td>${signed(a.d_tau, 3)} <span class="tip-mute">[${signed(h.ci[0], 2)}, ${signed(h.ci[1], 2)}]</span></td><td>${signed(a.d_klT, 3)}</td>` +
        `<td>${pct(a.fix)}</td><td>${pct(a.brk)}</td><td>${pp(a.net)}</td></tr>`);
    }
    for (const [k, v] of Object.entries(DATA.free)) {
      const a = v.all;
      rows.push(`<tr><td>${freeLabel[k]}</td><td>3 ep.</td><td>${fmt(a.kl12)}</td><td>${fmt(a.tv12)}</td><td>${pct(a.flip)}</td>` +
        `<td>${signed(a.d_tau, 3)} <span class="tip-mute">[${signed(v.ci[0], 2)}, ${signed(v.ci[1], 2)}]</span></td><td>${signed(a.d_klT, 3)}</td>` +
        `<td>${pct(a.fix)}</td><td>${pct(a.brk)}</td><td>${pp(a.net)}</td></tr>`);
    }
    $("#tab2").innerHTML = `<table><thead><tr><th>run</th><th>update</th><th>KL from base</th><th>TV</th><th>flips</th><th>Δτ [95% CI]</th>` +
      `<th>Δ KL(p_T‖p)</th><th>fixes</th><th>breaks</th><th>net</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
  }
  function tab3() {
    const out = [];
    for (const p of ["base", "static"]) {
      const P = DATA.points[p];
      if (!P || !P.rank) continue;
      for (const [k, row] of Object.entries(P.rank).sort()) {
        const ranks = Object.keys(row).filter((r) => r !== "full").map(Number).sort((a, b) => a - b).map(String).concat(["full"]);
        out.push(`<tr><td>${ptLabel[p]}</td><td>${k.replace("|", " · ε=")}</td>${ranks.map((r) => `<td>${fmt(row[r].kl12, 3)}<br><span class="tip-mute">${pct(row[r].flip, 1)}</span></td>`).join("")}</tr>`);
      }
    }
    $("#tab3").innerHTML = `<table><thead><tr><th>θ₁</th><th>location</th><th>r32</th><th>r64</th><th>r128</th><th>r512</th><th>r1024</th><th>dense</th></tr></thead><tbody>${out.join("")}</tbody></table>`;
  }
  function tab4() {
    const out = [];
    for (const p of ["base", "static"]) {
      const P = DATA.points[p];
      if (!P || !P.cross) continue;
      for (const [loc, cross] of Object.entries(P.cross)) {
        const diag = CATS.map((c) => -cross[c][c].d_klT);
        const off = CATS.map((c) => { const o = CATS.filter((x) => x !== c).map((x) => -cross[x][c].d_klT); return o.reduce((a, b) => a + b, 0) / o.length; });
        const shared = CATS.map((c) => -cross.all[c].d_klT);
        const g = P.geometry ? P.geometry[loc] : null;
        const meanCos = g ? Object.values(g.cos).reduce((a, b) => a + b, 0) / 6 : null;
        out.push(`<tr><td>${ptLabel[p]}</td><td>${loc}</td>${CATS.map((c, i) => `<td>${fmt(diag[i] * 1000, 2)} / ${fmt(off[i] * 1000, 2)} / ${fmt(shared[i] * 1000, 2)}</td>`).join("")}<td>${meanCos === null ? "–" : meanCos.toFixed(3)}</td></tr>`);
      }
    }
    $("#tab4").innerHTML = `<table><thead><tr><th>θ₁</th><th>location</th>${CATS.map((c) => `<th>${c}: own / other / shared</th>`).join("")}<th>mean gradient cosine</th></tr></thead><tbody>${out.join("")}</tbody></table>`;
  }

  function renderAll() {
    seg("#f3-seg", "f3", pts, f3); f3();
    seg("#f1-seg", "f1", pts, f1);
    seg("#f1-m", "f1m", [["kl12", "KL"], ["tv12", "TV"], ["flip", "flips"], ["d_tau", "Δτ"]], f1);
    seg("#f1-v", "f1v", [["groups", "groups"], ["q", "q"], ["k", "k"], ["v", "v"], ["o", "o"], ["gate", "gate"], ["up", "up"], ["down", "down"]], f1); f1();
    seg("#f2-seg", "f2eps", [[0.1, "ε 0.1"], [0.3, "ε 0.3"], [1.0, "ε 1.0"]], f2); f2();
    f4(); f5();
    seg("#f6-seg", "f6", pts, f6); f6();
    f7();
    seg("#f8-seg", "f8", pts, f8);
    const locs = Object.keys((DATA.points.base && DATA.points.base.cross) || {});
    seg("#f8-loc", "f8loc", locs.map((l) => [l, l]), f8); f8();
    seg("#f9-seg", "f9", pts, f9);
    seg("#f9-m", "f9m", [["sens", "sensitivity (random ΔW)"], ["fisher", "curvature (Fisher)"], ["gain", "gain (gradient step)"]], f9); f9();
    seg("#t1-seg", "t1", pts, tab1); tab1();
    tab2(); tab3(); tab4();
  }
  renderAll();
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener && mq.addEventListener("change", renderAll);
  new MutationObserver(renderAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
