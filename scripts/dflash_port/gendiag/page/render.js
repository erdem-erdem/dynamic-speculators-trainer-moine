/* GENDIAG page: renders every figure from the embedded data object D. */
(function () {
  const css = getComputedStyle(document.documentElement);
  const C = (i) => `var(--s${i})`;
  const $ = (id) => document.getElementById(id);
  // Fixed colour per run (identity follows the run, never its rank).
  const RUN = {
    "pt256": { name: "shared heads 0.23B", color: C(1) },
    "pm256": { name: "per-module heads 1.1B", color: C(2) },
    "pt256-twin": { name: "shared, no-input twin", color: C(1), dash: true },
    "pm256-twin": { name: "per-module, no-input twin", color: C(2), dash: true },
    "pt256-fp32": { name: "shared, fp32", color: C(3) },
    "pm256-fp32": { name: "per-module, fp32", color: C(4) },
    "pt256-fp32-centred": { name: "shared, fp32, centred heads", color: C(7) },
    "pt64": { name: "shared 64-wide", color: C(5) },
    "pt1024": { name: "shared 1024-wide", color: C(6) },
    "pm64": { name: "per-module 64-wide", color: C(8) },
  };
  const S = D.series || {};
  const avail = (names) => names.filter((n) => S[n]);

  function seriesOf(names, pick, label) {
    return avail(names).map((n) => ({ name: RUN[n].name, color: RUN[n].color, dash: RUN[n].dash, values: pick(S[n]) }));
  }
  // Runs share update grids (same interval); use the longest.
  function xOf(names) {
    let best = [];
    avail(names).forEach((n) => { if (S[n].update.length > best.length) best = S[n].update; });
    return best;
  }
  function pad(values, len) { const v = values.slice(0, len); while (v.length < len) v.push(null); return v; }
  function dyn(container, names, pick, opts) {
    const x = xOf(names);
    if (!x.length || !$(container)) return;
    lineChart($(container), Object.assign({ x, xlog: true, xlabel: "optimizer update (log)", series: seriesOf(names, (s) => pad(pick(s), x.length)) }, opts));
  }

  const MAIN = ["pt256", "pm256"];
  const ALL = ["pt256", "pm256", "pt256-twin", "pm256-twin", "pt256-fp32", "pm256-fp32", "pt256-fp32-centred", "pt64", "pt1024", "pm64"];
  const PROBES = ["pt256", "pt256-fp32", "pt256-fp32-centred", "pm256", "pm256-fp32"];

  // --- Section: anatomy (parameter share by component) ---
  if ($("fig-params")) {
    const rows = [];
    avail(ALL.filter((n) => !n.includes("twin"))).forEach((n) => {
      const g = S[n].meta.groups;
      const trunk = ["enc1", "enc2", "enc_ln", "cond1", "cond2", "emb", "dec_in", "dec_res"].reduce((a, k) => a + (g[k] || 0), 0);
      const hw = (g.headA_w || 0) + (g.headB_w || 0);
      const hb = (g.headA_b || 0) + (g.headB_b || 0) + (g.bias_A || 0) + (g.bias_B || 0);
      const tot = trunk + hw + hb;
      rows.push({ label: RUN[n].name, parts: [trunk, hw, hb], raw: [`${(trunk / 1e6).toFixed(1)}M`, `${(hw / 1e6).toFixed(0)}M`, `${(hb / 1e6).toFixed(1)}M`], total: tot });
    });
    stackChart($("fig-params"), { rows, left: 190, parts: [{ name: "trunk (encoder, condition, decoder)", color: C(1) }, { name: "head weights", color: C(2) }, { name: "head biases", color: C(3) }] });
  }

  // --- Section: training dynamics ---
  dyn("fig-gshare", MAIN, (s) => s.grad_group_share.trunk, { ylabel: "trunk share of ‖grad‖²", ymin: 0, ymax: 1 });
  dyn("fig-swallow", MAIN.concat(["pt256-fp32", "pm256-fp32"]), (s) => s.swallowed.enc1, { ylabel: "updates lost to bf16 (encoder L1)", ymin: 0, ymax: 1 });
  dyn("fig-swallow-dec", MAIN.concat(["pt256-fp32", "pm256-fp32"]), (s) => s.swallowed.dec_res, { ylabel: "updates lost to bf16 (decoder blocks)", ymin: 0, ymax: 1 });
  dyn("fig-dyn", ALL.filter((n) => !n.includes("twin")), (s) => s.dW.dyn_ratio, { ylabel: "dyn_ratio (per-row share of ‖ΔW‖²)", ymin: 0 });
  dyn("fig-cos", ALL.filter((n) => !n.includes("twin")), (s) => s.dW.cos_cross, { ylabel: "cosine between rows' ΔW (other category)", ymax: 1 });
  dyn("fig-zspread", MAIN, (s) => s.spread.z, { ylabel: "between-row spread of z", ylog: true });
  dyn("fig-bdyn", PROBES, (s) => s.headB_dyn, { ylabel: "B head: per-row / shared part", ymin: 0 });
  dyn("fig-bstatic", MAIN, (s) => s.headB_static.map((v, i) => s.headB_bias[i] / v), { ylabel: "share of B's shared part held by the bias", ymin: 0, ymax: 1 });
  dyn("fig-zgrad", MAIN, (s) => s.zgrad_shared, { ylabel: "share of z-gradient common to all rows", ymin: 0, ymax: 1 });

  // --- Section: controls (reading tests) ---
  const CTRL = [
    ["swap_same", "LoRA of another row, same category"],
    ["swap_cross", "LoRA of a row from another category"],
    ["mean", "one average LoRA for every row"],
    ["centroid", "average LoRA of the row's category"],
    ["zbar", "decoder fed the average input"],
    ["coef_0.5", "per-row part × 0.5"],
    ["coef_1.5", "per-row part × 1.5"],
    ["coef_-1", "per-row part reversed (× −1)"],
    ["bias_only", "head biases only"],
  ];
  const GEN = [
    ["abl24k-pt256_val", "24K shared heads, 0.23B"],
    ["abl24k-pm256_val", "24K per-module heads, 1.1B"],
    ["mix100k-r64_val", "100K shared heads, r64"],
    ["m0-4k_val", "4K per-module, 2.2B (overfits)"],
    ["abl24k-pt256-r64_val", "24K shared heads, r64"],
    ["abl24k-pt32-r512_val", "24K shared, r512 (near-constant)"],
    ["abl24k-hybrid-r512_val", "24K hybrid, r512"],
    ["gd-pt256-fp32_val", "shared, fp32"],
    ["gd-pt256-fp32-centred_val", "shared, fp32, centred heads"],
    ["gd-pm256-fp32_val", "per-module, fp32"],
    ["gd-pt1024_val", "shared 1024-wide"],
    ["gd-pt64_val", "shared 64-wide"],
    ["gd-pm64_val", "per-module 64-wide"],
  ];
  function ctrlPanel(id, key, title) {
    const r = D.controls[key];
    if (!r || !$(id)) return false;
    const rows = CTRL.filter(([k]) => r.summary[k]).map(([k, label]) => {
      const v = r.summary[k].vs_reference;
      return { label, value: v.delta, lo: v.lo, hi: v.hi, color: k.startsWith("coef") ? C(7) : k === "bias_only" ? C(8) : C(1), note: `τ ${r.summary[k].tau.toFixed(3)} vs own ${r.summary[r.summary[k].reference].tau.toFixed(3)} · ${r.rows} rows` };
    });
    const box = document.createElement("div");
    box.className = "fig";
    box.innerHTML = `<h4>${title}</h4><p class="cap">Δτ against the row's own LoRA; 95% CI. Own τ ${r.summary.own.tau.toFixed(3)}.</p>`;
    $(id).appendChild(box);
    ciChart(box, { rows, xmin: -0.06, xmax: 0.03, left: 210, xlabel: "Δτ vs own LoRA (negative = control is worse)", digits: 3 });
    return true;
  }
  GEN.forEach(([k, t]) => ctrlPanel("ctrl-grid", k, t));

  // Own − same-category swap and own − mean, compact, every generator.
  if ($("fig-ctrl-summary")) {
    const rows = [];
    GEN.forEach(([k, t]) => {
      const r = D.controls[k];
      if (!r) return;
      rows.push({ label: t, head: true });
      const s = r.summary.swap_same.vs_reference, m = r.summary.mean.vs_reference;
      rows.push({ label: "vs same-category swap", value: -s.delta, lo: -s.hi, hi: -s.lo, color: C(1) });
      rows.push({ label: "vs one average LoRA", value: -m.delta, lo: -m.hi, hi: -m.lo, color: C(2) });
    });
    ciChart($("fig-ctrl-summary"), { rows, left: 230, xmin: -0.02, xmax: 0.03, xlabel: "own LoRA advantage, Δτ (positive = the prompt helped)", digits: 3, valuename: "own − control", series: [{ name: "own − same-category swap", color: C(1) }, { name: "own − average LoRA", color: C(2) }] });
  }

  // Coefficient curve on the per-row part.
  if ($("fig-coef")) {
    const cs = [-1, 0, 0.5, 1, 1.5, 2];
    const series = [];
    GEN.slice(0, 4).forEach(([k, t], i) => {
      const r = D.controls[k];
      if (!r || !r.summary["coef_0.5"]) return;
      const ref = r.summary.own_dense.tau;
      const val = (c) => c === 1 ? 0 : c === 0 ? r.summary.mean.tau - ref : r.summary[`coef_${c}`].tau - ref;
      series.push({ name: t, color: C(i + 1), values: cs.map(val), points: true });
    });
    lineChart($("fig-coef"), { x: cs, xticks: cs, series, xname: "c", xlabel: "c in ΔW̄ + c·(ΔW_row − ΔW̄)   (0 = average LoRA, 1 = own)", ylabel: "Δτ vs own", zero: true, tipdigits: 4 });
  }

  // PCA truncation.
  if ($("fig-pca")) {
    const ks = [0, 1, 2, 4, 8, 16];
    const series = [];
    GEN.slice(0, 4).forEach(([k, t], i) => {
      const r = D.controls[k];
      if (!r || !r.summary.pca_1) return;
      const ref = r.summary.own_dense.tau;
      series.push({ name: t, color: C(i + 1), points: true, values: ks.map((kk) => kk === 0 ? r.summary.mean.tau - ref : r.summary[`pca_${kk}`].tau - ref) });
    });
    lineChart($("fig-pca"), { x: ks, xticks: ks, series, xname: "k", xlabel: "principal directions of the per-row part kept (k)", ylabel: "Δτ vs own", zero: true, tipdigits: 4 });
  }

  // --- Section: is there a signal (gradient budget) ---
  const DIR = [
    ["abl24k-pt256_val_zero", "base drafter (no LoRA)"],
    ["abl24k-pt256_val", "24K shared heads, 0.23B"],
    ["abl24k-pm256_val", "24K per-module, 1.1B"],
    ["mix100k-r64_val", "100K shared heads, r64"],
    ["m0-4k_val", "4K per-module, 2.2B"],
    ["abl24k-pt256-r64_val", "24K shared, r64"],
    ["abl24k-pt128-r128_val", "24K shared, r128"],
    ["abl24k-pt32-r512_val", "24K shared, r512"],
    ["abl24k-hybrid-r512_val", "24K hybrid, r512"],
    ["gd-pt256-fp32_val", "shared, fp32"],
    ["gd-pt256-fp32-centred_val", "shared, fp32, centred"],
    ["gd-pm256-fp32_val", "per-module, fp32"],
  ];
  if ($("fig-budget")) {
    const rows = DIR.filter(([k]) => D.direction[k]).map(([k, t]) => {
      const b = D.direction[k].budget;
      return { label: t, parts: [Math.max(b.shared, 0), Math.max(b.between_cat, 0), Math.max(b.within_cat, 0), Math.max(b.noise, 0)] };
    });
    stackChart($("fig-budget"), { rows, left: 190, parts: [{ name: "shared by all rows", color: C(1) }, { name: "category-level", color: C(3) }, { name: "row-specific, reproducible", color: C(2) }, { name: "not reproducible (anchor noise)", color: "var(--neutral)" }] });
  }
  if ($("fig-budget-zoom")) {
    const rows = DIR.filter(([k]) => D.direction[k]).map(([k, t]) => {
      const b = D.direction[k].budget;
      return { label: t, parts: [Math.max(b.shared, 0), Math.max(b.between_cat, 0), Math.max(b.within_cat, 0)] };
    });
    stackChart($("fig-budget-zoom"), { rows, left: 190, parts: [{ name: "shared by all rows", color: C(1) }, { name: "category-level", color: C(3) }, { name: "row-specific, reproducible", color: C(2) }] });
  }
  if ($("fig-direction")) {
    const rows = [];
    DIR.filter(([k]) => D.direction[k] && !k.endsWith("_zero")).forEach(([k, t]) => {
      const d = D.direction[k];
      const m = d.own_minus_same.mean, se = d.own_minus_same.se;
      rows.push({ label: t, value: m * 1000, lo: (m - 1.96 * se) * 1000, hi: (m + 1.96 * se) * 1000, color: C(1), note: `own ${(d.direction.own * 1000).toFixed(2)}, same-category ${(d.direction.same * 1000).toFixed(2)}, other ${(d.direction.cross * 1000).toFixed(2)} (×10⁻³) · ${d.rows} rows` });
    });
    ciChart($("fig-direction"), { rows, left: 190, xlabel: "own − same-category donor, first-order loss decrease (×10⁻³)", digits: 2, valuename: "own − same" });
  }

  // --- Section: end-of-training pipeline spread ---
  if ($("fig-stages")) {
    const stages = [["summary", "input"], ["enc_hidden", "enc. hidden"], ["encoder", "encoder out"], ["z", "z"], ["dec_out", "decoder out"], ["A", "A"], ["B", "B"]];
    const keys = [["abl24k-pt256", "24K shared heads"], ["abl24k-pm256", "24K per-module"], ["mix24k-pm256", "24K per-module, 3 ep"], ["gd-pt256-fp32", "shared, fp32"], ["gd-pt256-fp32-centred", "shared, fp32, centred"]];
    const series = keys.filter(([k]) => D.internals[k]).map(([k, t], i) => ({ name: t, color: C(i + 1), values: stages.map(([s]) => (D.internals[k].probe_val[s] || {}).spread ?? null) }));
    barChart($("fig-stages"), { groups: stages.map((s) => s[1]), series, ylog: true, ylabel: "between-row spread (var / ‖mean‖²)", tipdigits: 4 });
  }
  if ($("fig-gradcomp")) {
    const groups = [["enc1", "enc L1"], ["enc2", "enc L2"], ["cond1", "cond L1"], ["cond2", "cond L2"], ["dec_in", "dec in"], ["dec_res", "dec blocks"], ["headA_w", "A head W"], ["headB_w", "B head W"], ["headA_b", "A bias"], ["headB_b", "B bias"]];
    const keys = [["abl24k-pt256", "24K shared heads"], ["abl24k-pm256", "24K per-module"], ["gd-pt256-fp32", "shared, fp32"]];
    const series = keys.filter(([k]) => D.internals[k]).map(([k, t], i) => ({ name: t, color: C(i + 1), values: groups.map(([g]) => (D.internals[k].grad[g] || {}).rms ?? null) }));
    barChart($("fig-gradcomp"), { groups: groups.map((g) => g[1]), series, ylog: true, ylabel: "RMS gradient per parameter", tipdigits: 4 });
  }

  // --- Section: results table ---
  if ($("tbl-runs") && D.results) {
    const rr = D.results.runs || D.results;
    const order = Object.keys(rr);
    const body = order.map((k) => {
      const v = rr[k];
      const val = v.validation ? v.validation.all.tau : null, test = v.test ? v.test.all.tau : null;
      const vr = v.test && v.test.vs_reference ? v.test.vs_reference.tau : null;
      const dv = vr ? { delta: vr.delta, lo: vr.ci[0], hi: vr.ci[1] } : null;
      const sim = (v.similarity || {}).all || {};
      return `<tr><td>${v.display || k}</td><td>${v.trainable_parameters ? (+v.trainable_parameters / 1e6).toFixed(0) + "M" : "–"}</td><td>${val ? val.toFixed(3) : "–"}</td><td>${test ? test.toFixed(3) : "–"}</td><td>${dv ? `${dv.delta >= 0 ? "+" : ""}${dv.delta.toFixed(3)} [${dv.lo.toFixed(3)}, ${dv.hi.toFixed(3)}]` : "–"}</td><td>${sim.dyn_ratio !== undefined ? sim.dyn_ratio.toFixed(3) : "–"}</td><td>${sim.effective_rank !== undefined ? sim.effective_rank.toFixed(2) : "–"}</td></tr>`;
    }).join("");
    $("tbl-runs").innerHTML = `<thead><tr><th>run</th><th>params</th><th>val τ</th><th>test τ</th><th>test Δτ vs static r32 [95% CI]</th><th>dyn_ratio</th><th>eff. rank</th></tr></thead><tbody>${body}</tbody>`;
  }
})();
