/* Renders the MIX100K results page from DATA (built by build.py from abl24k_collect.py output). */
(function () {
  const V = window.Viz, css = V.css;
  const $ = (s) => document.querySelector(s);
  const CATS = ["math", "code", "stem", "chat"];
  const R = (a) => DATA.runs[a.label];
  const ARMS = DATA.arms.filter((a) => DATA.runs[a.label]);
  const EVAL = ARMS.filter((a) => R(a).test);
  const sgn = (v, d = 3) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
  const ci = (c) => `[${sgn(c[0])}, ${sgn(c[1])}]`;
  const params = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + "B" : n >= 1e6 ? (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + "M" : String(n));
  const col = (a) => css(a.color);
  const state = { p1: "test", p3: "both" };

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

  /* P1: Δτ vs base drafter for every arm, 95% CI. */
  function p1() {
    const sp = state.p1, box = $("#p1");
    const rows = EVAL.filter((a) => R(a)[sp] && R(a)[sp].vs_base);
    if (!rows.length) { box.textContent = "No results yet."; return; }
    const pts = [], lines = [];
    rows.forEach((a, i) => {
      const d = R(a)[sp].vs_base.tau, y = rows.length - i;
      pts.push({ x: d.delta, y, group: a.group, html: `<div class="tip-h">${a.name}</div>Δτ vs base ${sgn(d.delta)} ${ci(d.ci)}<br>τ ${R(a)[sp].all.tau.toFixed(3)}` });
      lines.push({ points: [{ x: d.ci[0], y }, { x: d.ci[1], y }], color: col(a), width: 2.2 });
    });
    const xs = rows.flatMap((a) => R(a)[sp].vs_base.tau.ci);
    const lo = Math.min(0, ...xs) - 0.01, hi = Math.max(...xs) + 0.01;
    const names = Object.fromEntries(rows.map((a, i) => [rows.length - i, a.name]));
    V.scatter(box, { width: 760, height: 60 + rows.length * 32, margin: { l: 260, b: 40 }, points: pts, lines,
      xdomain: [lo, hi], ydomain: [0.4, rows.length + 0.6], yticks: rows.map((_, i) => i + 1), yfmt: (v) => names[v] || "",
      xfmt: (v) => sgn(v, 2), xlabel: `Δτ vs base drafter, ${sp}, category mean, 95% paired bootstrap CI`,
      groups: { s100: { color: css("--s1"), label: "100K static" }, g100: { color: css("--s2"), label: "100K generator", shape: "diamond", size: 4.5 },
        s24: { color: css("--s3"), label: "24K static" }, g24: { color: css("--s4"), label: "24K generator", shape: "diamond" } } });
  }

  /* P2: τ against training rows (24K → 100K). */
  function p2() {
    const box = $("#p2");
    const series = DATA.scale.map((s) => {
      const pts = s.points.filter((p) => DATA.runs[p.label] && DATA.runs[p.label].test);
      return { name: s.name, color: css(s.color), x: pts.map((p) => p.rows), y: pts.map((p) => DATA.runs[p.label].test.all.tau) };
    }).filter((s) => s.x.length);
    if (!series.length) { box.textContent = "No results yet."; return; }
    const base = DATA.base.test.all.tau;
    V.line(box, { width: 640, height: 300, logx: true, xname: "training rows", xlabel: "training rows (log scale; 24K = 2 epochs, 100K = 3 epochs)",
      ylabel: "test τ (category mean)", xticks: [24000, 100000], xfmt: (v) => (v / 1000) + "K", tipfmt: (v) => v.toFixed(3),
      refs: [{ y: base, label: `base drafter ${base.toFixed(3)}` }], series });
  }

  /* P3: validation τ over training, in epochs. */
  function p3() {
    const box = $("#p3");
    const sel = ARMS.filter((a) => a.curve && (state.p3 === "both" || a.data === state.p3) && R(a).history && R(a).history.length);
    if (!sel.length) { box.textContent = "No runs yet."; return; }
    V.line(box, { width: 760, height: 320, xname: "epoch", xlabel: "epochs", ylabel: "validation τ (category mean)", tipfmt: (v) => v.toFixed(3),
      series: sel.map((a) => ({ name: a.name, color: col(a), dash: a.data === "24k" ? "4 3" : "",
        x: R(a).history.map((h) => +(h.update / a.updates_per_epoch).toFixed(3)), y: R(a).history.map((h) => h.tau) })) });
  }

  /* P4: per-category gain over base (test). */
  function p4() {
    const box = $("#p4");
    const rows = EVAL.filter((a) => a.heat);
    if (!rows.length) { box.textContent = "No results yet."; return; }
    const b = DATA.base.test.categories;
    const vals = rows.map((a) => [...CATS.map((c) => R(a).test.categories[c].tau - b[c].tau), R(a).test.all.tau - DATA.base.test.all.tau]);
    const m = Math.max(...vals.flat().map(Math.abs));
    const dark = getComputedStyle(document.documentElement).colorScheme.includes("dark");
    const pos = V.ramp(dark ? ["#1d2226", "#1d5a40", "#2fbf7a"] : ["#f4f6f8", "#9fd3b8", "#1f8a5b"], dark ? -1 : 0.55);
    const neg = V.ramp(dark ? ["#1d2226", "#6b2a27", "#e66767"] : ["#f4f6f8", "#efb2ac", "#b8423a"], dark ? -1 : 0.55);
    V.heatmap(box, { rows: rows.map((a) => a.name), cols: [...CATS, "mean"], values: vals, labelW: 260, cellW: 70,
      fmt: (v) => sgn(v), color: (v) => (v >= 0 ? pos(v / m) : neg(-v / m)),
      tip: (i, j, v) => `${rows[i].name} · ${[...CATS, "mean"][j]}: <b>${sgn(v)}</b> τ over base (test)` });
  }

  /* P5: LoRA coefficient at inference. */
  function p5() {
    const box = $("#p5");
    const sel = EVAL.filter((a) => a.curve && R(a).coefficients && Object.keys(R(a).coefficients).length);
    if (!sel.length) { box.textContent = "No results yet."; return; }
    V.line(box, { width: 640, height: 280, logx: true, xname: "coefficient", xlabel: "LoRA coefficient at inference (training scale 2)",
      ylabel: "validation full_acc", xticks: [0.5, 1, 2, 4, 8], xfmt: (v) => String(v), tipfmt: (v) => v.toFixed(4),
      series: sel.map((a) => { const c = R(a).coefficients, ks = Object.keys(c).map(Number).sort((x, y) => x - y);
        return { name: a.name, color: col(a), dash: a.data === "24k" ? "4 3" : "", x: ks, y: ks.map((k) => c[String(k)]) }; }) });
  }

  function direct() {
    const box = $("#direct");
    const rows = DATA.direct.filter((d) => d.result);
    if (!rows.length) { box.innerHTML = "<p class='note'>Pending.</p>"; return; }
    box.innerHTML = `<table><thead><tr><th>Comparison (paired, same rows)</th><th>validation</th><th>test</th><th>OOD</th></tr></thead><tbody>${rows
      .map((d) => `<tr><td>${d.name}</td>${["validation", "test", "ood"].map((s) => (d.result[s] ? `<td class="${d.result[s].ci[0] > 0 ? "pos" : d.result[s].ci[1] < 0 ? "neg" : ""}">${sgn(d.result[s].delta)} <span class="tip-mute">${ci(d.result[s].ci)}</span></td>` : "<td>–</td>")).join("")}</tr>`)
      .join("")}</tbody></table>`;
  }

  function table() {
    const head = ["Arm", "data", "epochs", "trainable", "train h", "val τ", "test τ", "OOD τ", "cross-row cos", "eff. rank", "dyn_ratio", "ms / LoRA"];
    const rows = ARMS.map((a) => { const r = R(a), s = r.similarity && r.similarity.all, c = r.cost || {};
      return [a.name, a.data.toUpperCase(), a.epochs, r.trainable_parameters ? params(r.trainable_parameters) : "–", r.training_seconds ? (r.training_seconds / 3600).toFixed(1) : "…",
        ...["validation", "test", "ood"].map((sp) => (r[sp] ? r[sp].all.tau.toFixed(3) : "…")),
        s ? s.cos_cross.toFixed(3) : "–", s ? s.effective_rank.toFixed(2) : "–", s ? s.dyn_ratio.toFixed(3) : "–", c.ms_per_call ? c.ms_per_call.toFixed(1) : "–"]; });
    const b = DATA.base;
    rows.unshift(["Base drafter", "—", "—", "0", "—", ...["validation", "test", "ood"].map((sp) => b[sp].all.tau.toFixed(3)), "–", "–", "–", "–"]);
    $("#tab1").innerHTML = `<table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }

  function settings() {
    $("#settings").innerHTML = `<table><thead><tr><th>Setting</th>${DATA.settings.arms.map((a) => `<th>${a}</th>`).join("")}</tr></thead><tbody>${DATA.settings.rows
      .map((r) => `<tr><td>${r[0]}</td>${r.slice(1).map((c) => `<td style="text-align:left;white-space:normal">${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }

  function draw() { p1(); p2(); p3(); p4(); p5(); }
  seg("#p1-seg", "p1", [["test", "test"], ["validation", "validation"], ["ood", "OOD"]], p1);
  seg("#p3-seg", "p3", [["both", "100K + 24K"], ["100k", "100K"], ["24k", "24K"]], p3);
  draw(); direct(); table(); settings();
  let t;
  const redraw = () => { clearTimeout(t); t = setTimeout(draw, 50); };
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener && mq.addEventListener("change", redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
