/* Renders every ABL24K figure and table from DATA (built by build.py from abl24k_collect.py output). */
(function () {
  const V = window.Viz, css = V.css, fmt = V.fmt;
  const $ = (s) => document.querySelector(s);
  const CATS = ["math", "code", "stem", "chat"];
  const ARMS = DATA.arms.filter((a) => DATA.runs[a.label] && DATA.runs[a.label].test);
  const ALL = DATA.arms.filter((a) => DATA.runs[a.label]);
  const KCOL = { static: "--s1", generator: "--s2", context: "--s7" };
  const col = (a) => css(a.color || KCOL[a.kind] || "--s4");
  const state = { f1: "test", f4: "rank", f7: "static" };
  const sgn = (v, d = 3) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
  const ci = (c) => `[${sgn(c[0])}, ${sgn(c[1])}]`;
  const params = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + "B" : n >= 1e6 ? (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + "M" : n.toLocaleString());
  const R = (a) => DATA.runs[a.label];

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
        box.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        onChange();
      });
      box.appendChild(b);
    }
  }

  /* F1: forest plot of Δτ vs the matched static r32 (KL), 95% CI. */
  function f1() {
    const split = state.f1;
    const rows = ARMS.filter((a) => a.label !== DATA.reference && R(a)[split] && R(a)[split].vs_reference);
    const box = $("#f1");
    if (!rows.length) { box.textContent = "No results yet."; return; }
    const pts = [], lines = [];
    rows.forEach((a, i) => {
      const d = R(a)[split].vs_reference.tau;
      const y = rows.length - i;
      pts.push({ x: d.delta, y, group: a.kind,
        html: `<div class="tip-h">${a.name}</div>Δτ ${sgn(d.delta)} ${ci(d.ci)}<br>τ ${R(a)[split].all.tau.toFixed(3)} · ${params(R(a).trainable_parameters || 0)} params` });
      lines.push({ points: [{ x: d.ci[0], y }, { x: d.ci[1], y }], color: col(a), width: 2.2 });
    });
    const xs = rows.flatMap((a) => R(a)[split].vs_reference.tau.ci);
    const lo = Math.min(-0.02, ...xs) - 0.005, hi = Math.max(0.02, ...xs) + 0.005;
    const names = Object.fromEntries(rows.map((a, i) => [rows.length - i, a.name]));
    V.scatter(box, { width: 760, height: 60 + rows.length * 30, margin: { l: 250, b: 40 },
      points: pts, lines, xdomain: [lo, hi], ydomain: [0.4, rows.length + 0.6],
      yticks: rows.map((_, i) => i + 1), yfmt: (v) => names[v] || "", xfmt: (v) => sgn(v, 2),
      xlabel: `Δτ vs static r32 (KL), ${split}, category mean, 95% paired bootstrap CI`,
      groups: { static: { color: css("--s1"), label: "static LoRA" }, generator: { color: css("--s2"), label: "generator" },
        context: { color: css("--s7"), label: "3-epoch run (context)" } } });
    // zero line
    const svg = box.querySelector("svg");
    const vb = svg.viewBox.baseVal, l = 250, r = 16, x0 = l + (0 - lo) / (hi - lo) * (vb.width - l - r);
    V.el("line", { x1: x0, x2: x0, y1: 14, y2: vb.height - 40, stroke: css("--axis"), "stroke-dasharray": "3 3" }, svg);
  }

  /* F2: τ against rank (static sweep line; generator points). */
  function f2() {
    const box = $("#f2");
    const st = ARMS.filter((a) => a.group === "rank").sort((a, b) => a.rank - b.rank);
    if (!st.length) { box.textContent = "No results yet."; return; }
    const series = ["test", "validation", "ood"].map((s, i) => ({ name: `static KL, ${s}`, color: css(`--s${[1, 3, 7][i]}`),
      x: st.filter((a) => R(a)[s]).map((a) => a.rank), y: st.filter((a) => R(a)[s]).map((a) => R(a)[s].all.tau) }));
    const two = $("#f2-mode").value;
    if (two === "delta") {
      const ss = st.filter((a) => R(a).test.vs_reference || a.label === DATA.reference);
      V.line(box, { width: 640, height: 300, logx: true, xname: "rank", xlabel: "LoRA rank (log scale)", ylabel: "Δτ vs r32 (test)",
        tipfmt: (v) => sgn(v), series: [{ name: "static KL, test Δτ (band = 95% CI)", color: css("--s1"), x: ss.map((a) => a.rank),
          y: ss.map((a) => (a.label === DATA.reference ? 0 : R(a).test.vs_reference.tau.delta)),
          lo: ss.map((a) => (a.label === DATA.reference ? 0 : R(a).test.vs_reference.tau.ci[0])),
          hi: ss.map((a) => (a.label === DATA.reference ? 0 : R(a).test.vs_reference.tau.ci[1])) }],
        xticks: ss.map((a) => a.rank), xfmt: (v) => String(v) });
    } else {
      V.line(box, { width: 640, height: 300, logx: true, xname: "rank", xlabel: "LoRA rank (log scale)", ylabel: "τ (category mean)",
        series: series.filter((s) => s.x.length), xticks: st.map((a) => a.rank), xfmt: (v) => String(v), tipfmt: (v) => v.toFixed(3) });
    }
  }

  /* F3: gain against trainable parameters. */
  function f3() {
    const box = $("#f3");
    const pts = ARMS.filter((a) => R(a).trainable_parameters && R(a).test.vs_base).map((a) => ({
      x: R(a).trainable_parameters, y: R(a).test.vs_base.tau.delta, group: a.kind, label: a.short,
      html: `<div class="tip-h">${a.name}</div>${params(R(a).trainable_parameters)} trainable<br>Δτ vs base ${sgn(R(a).test.vs_base.tau.delta)} ${ci(R(a).test.vs_base.tau.ci)}` }));
    if (!pts.length) { box.textContent = "No results yet."; return; }
    V.scatter(box, { width: 760, height: 340, logx: true, points: pts, xlabel: "trainable parameters (log scale)",
      ylabel: "Δτ vs base drafter (test)", xfmt: (v) => params(v), yfmt: (v) => sgn(v, 2),
      groups: { static: { color: css("--s1"), label: "static LoRA", size: 4.5 }, generator: { color: css("--s2"), label: "generator", shape: "diamond", size: 4.5 },
        context: { color: css("--s7"), label: "3-epoch run (context)", shape: "square" } } });
  }

  /* F4: validation τ over training. */
  function f4() {
    const box = $("#f4");
    const sel = ALL.filter((a) => (a.panels || []).includes(state.f4) && R(a).history && R(a).history.length);
    if (!sel.length) { box.textContent = "No runs in this group yet."; return; }
    V.line(box, { width: 760, height: 320, xname: "update", xlabel: "update (2,667 = 2 epochs)", ylabel: "validation τ (category mean)",
      tipfmt: (v) => v.toFixed(3),
      series: sel.map((a) => ({ name: a.name, color: col(a), dash: a.kind === "context" ? "4 3" : "",
        x: R(a).history.map((h) => h.update), y: R(a).history.map((h) => h.tau) })) });
  }

  /* F5: per-category Δτ vs reference (test). */
  function f5() {
    const box = $("#f5");
    const ref = DATA.runs[DATA.reference];
    const rows = ARMS.filter((a) => a.label !== DATA.reference && a.kind !== "context");
    if (!rows.length || !ref || !ref.test) { box.textContent = "No results yet."; return; }
    const vals = rows.map((a) => [...CATS.map((c) => R(a).test.categories[c].tau - ref.test.categories[c].tau),
      R(a).test.all.tau - ref.test.all.tau]);
    const m = Math.max(0.02, ...vals.flat().map(Math.abs));
    const dark = getComputedStyle(document.documentElement).colorScheme.includes("dark");
    const pos = V.ramp(dark ? ["#1d2226", "#1d5a40", "#2fbf7a"] : ["#f4f6f8", "#9fd3b8", "#1f8a5b"], dark ? -1 : 0.55);
    const neg = V.ramp(dark ? ["#1d2226", "#6b2a27", "#e66767"] : ["#f4f6f8", "#efb2ac", "#b8423a"], dark ? -1 : 0.55);
    V.heatmap(box, { rows: rows.map((a) => a.name), cols: [...CATS, "mean"], values: vals, labelW: 250, cellW: 70,
      fmt: (v) => sgn(v), color: (v) => (v >= 0 ? pos(v / m) : neg(-v / m)),
      tip: (i, j, v) => `${rows[i].name} · ${[...CATS, "mean"][j]}: Δτ <b>${sgn(v)}</b> vs static r32 (test)` });
  }

  /* F6: the D-PACE gate, with q = the base drafter's measured per-slot accuracy. */
  function f6() {
    const box = $("#f6");
    const q = DATA.base_slot_acc;
    if (!q) { box.textContent = "—"; return; }
    const w = (alpha) => {
      const s = q.map((x) => (1 - alpha) * x + alpha);
      const pre = []; s.reduce((acc, x, i) => (pre[i] = acc * x), 1);
      const out = pre.map((_, j) => pre.slice(j).reduce((a, b) => a + b, 0));
      const tot = out.reduce((a, b) => a + b, 0);
      return out.map((x) => x / tot);
    };
    const xs = q.map((_, i) => i + 1);
    V.line(box, { width: 640, height: 280, xname: "draft slot", xlabel: "draft slot in the block", ylabel: "share of the loss weight",
      tipfmt: (v) => (v * 100).toFixed(1) + "%", yfmt: (v) => (v * 100).toFixed(0) + "%",
      series: [
        { name: "uniform (no gate)", color: css("--muted"), dash: "4 3", x: xs, y: xs.map(() => 1 / xs.length) },
        { name: "D-PACE α 0.5 (upstream default)", color: css("--s1"), x: xs, y: w(0.5) },
        { name: "D-PACE α 0.1 (ABL24K label gate)", color: css("--s2"), x: xs, y: w(0.1) },
        { name: "α 0 (exact ∂E[τ] weight)", color: css("--s3"), dash: "2 2", x: xs, y: w(1e-9) } ] });
  }

  /* F7: LoRA coefficient at inference (M4). */
  function f7() {
    const box = $("#f7");
    const sel = ARMS.filter((a) => (a.panels || []).includes(state.f7) && R(a).coefficients && Object.keys(R(a).coefficients).length);
    if (!sel.length) { box.textContent = "No results yet."; return; }
    V.line(box, { width: 640, height: 280, logx: true, xname: "coefficient", xlabel: "LoRA coefficient at inference (training scale 2)",
      ylabel: "validation full_acc", tipfmt: (v) => v.toFixed(4), xticks: [0.5, 1, 2, 4, 8], xfmt: (v) => String(v),
      series: sel.map((a) => { const c = R(a).coefficients; const ks = Object.keys(c).map(Number).sort((x, y) => x - y);
        return { name: a.name, color: col(a), x: ks, y: ks.map((k) => c[String(k)]) }; }) });
  }

  function table() {
    const box = $("#tab1");
    const head = ["Arm", "kind", "loss", "rank", "heads", "trainable", "train min", "val τ", "test τ", "OOD τ", "test Δτ vs r32", "OOD Δτ vs r32",
      "cross-row cos", "eff. rank", "ms / LoRA"];
    const rows = ALL.map((a) => {
      const r = R(a), s = r.similarity && r.similarity.all, c = r.cost || {};
      const dd = (sp) => (r[sp] && r[sp].vs_reference ? `${sgn(r[sp].vs_reference.tau.delta)} <span class="tip-mute">${ci(r[sp].vs_reference.tau.ci)}</span>` : a.label === DATA.reference ? "ref" : "–");
      return [a.name, a.kind, a.loss, a.rank, a.heads || "–", r.trainable_parameters ? params(r.trainable_parameters) : "–",
        r.training_seconds ? (r.training_seconds / 60).toFixed(0) : "–",
        ...["validation", "test", "ood"].map((sp) => (r[sp] ? r[sp].all.tau.toFixed(3) : "…")), dd("test"), dd("ood"),
        s ? s.cos_cross.toFixed(3) : "–", s ? s.effective_rank.toFixed(2) : "–", c.ms_per_call ? c.ms_per_call.toFixed(2) : "–"];
    });
    const base = DATA.base;
    rows.unshift(["Base drafter (no adapter)", "—", "—", "—", "—", "0", "—",
      ...["validation", "test", "ood"].map((sp) => (base[sp] ? base[sp].all.tau.toFixed(3) : "–")), "–", "–", "–", "–", "–"]);
    box.innerHTML = `<table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows
      .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }

  function draw() {
    seg("#f1-seg", "f1", [["test", "test"], ["validation", "validation"], ["ood", "OOD"]], f1);
    seg("#f4-seg", "f4", [["rank", "rank sweep"], ["loss", "loss"], ["generator", "generator"]], f4);
    seg("#f7-seg", "f7", [["static", "static"], ["generator", "generator"]], f7);
    $("#f2-mode").addEventListener("change", f2);
    f1(); f2(); f3(); f4(); f5(); f6(); f7(); table();
  }
  draw();
  let t;
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const redraw = () => { clearTimeout(t); t = setTimeout(() => { f1(); f2(); f3(); f4(); f5(); f6(); f7(); }, 50); };
  mq.addEventListener && mq.addEventListener("change", redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
