/* GENDIAG page charts: small inline-SVG chart helpers (no library).
   Every chart reads colors from CSS tokens, has a hover tooltip, and a legend
   when it shows more than one series. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  document.body.appendChild(tip);
  function showTip(evt, html) {
    tip.innerHTML = html;
    tip.hidden = false;
    const pad = 14;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    const r = tip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }
  function hideTip() { tip.hidden = true; }
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function fmt(v, d) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    const a = Math.abs(v);
    if (d !== undefined) return v.toFixed(d);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1);
    if (a < 0.1) return v.toFixed(4);
    if (a < 10) return v.toFixed(3);
    return v.toFixed(1);
  }
  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1;
    const step0 = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const t = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) t.push(+v.toFixed(12));
    return t;
  }
  function legend(container, series) {
    if (series.length < 2) return;
    const lg = document.createElement("div");
    lg.className = "legend";
    series.forEach((s) => {
      const i = document.createElement("span");
      i.innerHTML = `<i style="background:${s.color}${s.dash ? ";background:repeating-linear-gradient(90deg," + s.color + " 0 4px,transparent 4px 7px)" : ""}"></i>${s.name}`;
      lg.appendChild(i);
    });
    container.appendChild(lg);
  }
  function frame(container, opts) {
    const W = opts.width || 560, H = opts.height || 260;
    const m = Object.assign({ l: 56, r: 16, t: 14, b: 38 }, opts.margin || {});
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": opts.label || "" });
    container.appendChild(svg);
    return { svg, W, H, m, iw: W - m.l - m.r, ih: H - m.t - m.b };
  }

  // Line chart. opts: {x:[...], series:[{name,color,values,dash}], ylog, xlog, ylabel, xlabel, ymin, ymax}
  window.lineChart = function (container, opts) {
    const f = frame(container, opts);
    const { svg, m, iw, ih } = f;
    const xs = opts.x;
    const tx = opts.xlog ? (v) => Math.log10(Math.max(v, 1)) : (v) => v;
    const ty = opts.ylog ? (v) => Math.log10(v) : (v) => v;
    const vals = [];
    opts.series.forEach((s) => s.values.forEach((v) => { if (v !== null && v !== undefined && isFinite(ty(v))) vals.push(ty(v)); }));
    let ylo = opts.ymin !== undefined ? ty(opts.ymin) : Math.min(...vals);
    let yhi = opts.ymax !== undefined ? ty(opts.ymax) : Math.max(...vals);
    if (ylo === yhi) { ylo -= 1; yhi += 1; }
    if (!opts.ylog && opts.ymin === undefined) { const p = (yhi - ylo) * 0.06; ylo -= p; yhi += p; }
    const xlo = tx(Math.min(...xs)), xhi = tx(Math.max(...xs));
    const X = (v) => m.l + ((tx(v) - xlo) / (xhi - xlo || 1)) * iw;
    const Y = (v) => m.t + ih - ((ty(v) - ylo) / (yhi - ylo || 1)) * ih;
    const g = el("g", { class: "grid" }, svg);
    let yt;
    if (opts.ylog) { yt = []; for (let e = Math.floor(ylo); e <= Math.ceil(yhi); e++) if (e >= ylo - 1e-9 && e <= yhi + 1e-9) yt.push(Math.pow(10, e)); }
    else yt = niceTicks(ylo, yhi, 5);
    yt.forEach((v) => {
      el("line", { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v) }, g);
      const t = el("text", { x: m.l - 6, y: Y(v) + 4, class: "tick", "text-anchor": "end" }, svg);
      t.textContent = opts.ylog ? "1e" + Math.round(Math.log10(v)) : fmt(v, opts.ydigits);
    });
    let xt;
    if (opts.xlog) { xt = [1, 10, 100, 1000].filter((v) => v >= Math.min(...xs) && v <= Math.max(...xs)); xt.push(Math.max(...xs)); }
    else xt = opts.xticks || niceTicks(Math.min(...xs), Math.max(...xs), 6);
    xt.forEach((v) => {
      const t = el("text", { x: X(v), y: m.t + ih + 16, class: "tick", "text-anchor": "middle" }, svg);
      t.textContent = v;
    });
    if (opts.zero && !opts.ylog && ylo < 0 && yhi > 0) el("line", { x1: m.l, x2: m.l + iw, y1: Y(0), y2: Y(0), class: "zero" }, svg);
    el("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, class: "axis" }, svg);
    if (opts.ylabel) { const t = el("text", { x: 12, y: m.t + ih / 2, class: "axlabel", transform: `rotate(-90 12 ${m.t + ih / 2})`, "text-anchor": "middle" }, svg); t.textContent = opts.ylabel; }
    if (opts.xlabel) { const t = el("text", { x: m.l + iw / 2, y: f.H - 4, class: "axlabel", "text-anchor": "middle" }, svg); t.textContent = opts.xlabel; }
    opts.series.forEach((s) => {
      let d = "", pen = false;
      s.values.forEach((v, i) => {
        if (v === null || v === undefined || !isFinite(ty(v))) { pen = false; return; }
        d += (pen ? "L" : "M") + X(xs[i]).toFixed(1) + "," + Y(v).toFixed(1);
        pen = true;
      });
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash ? "5 4" : "none", "stroke-linejoin": "round" }, svg);
      if (s.points) s.values.forEach((v, i) => { if (v !== null && isFinite(ty(v))) el("circle", { cx: X(xs[i]), cy: Y(v), r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }, svg); });
    });
    // crosshair
    const cross = el("line", { y1: m.t, y2: m.t + ih, class: "cross", visibility: "hidden" }, svg);
    const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", (evt) => {
      const r = svg.getBoundingClientRect();
      const px = ((evt.clientX - r.left) / r.width) * f.W;
      let best = 0, bd = Infinity;
      xs.forEach((x, i) => { const dd = Math.abs(X(x) - px); if (dd < bd) { bd = dd; best = i; } });
      cross.setAttribute("x1", X(xs[best])); cross.setAttribute("x2", X(xs[best])); cross.setAttribute("visibility", "visible");
      const rows = opts.series.map((s) => `<div><i style="background:${s.color}"></i>${s.name}<b>${fmt(s.values[best], opts.tipdigits)}</b></div>`).join("");
      showTip(evt, `<div class="tt">${opts.xname || "update"} ${xs[best]}</div>${rows}`);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
    legend(container, opts.series);
  };

  // Dot + CI rows. opts: {rows:[{label, value, lo, hi, color, note}], xlabel, xmin, xmax}
  window.ciChart = function (container, opts) {
    const rowH = 26;
    const H = opts.rows.length * rowH + 44;
    const f = frame(container, Object.assign({}, opts, { height: H, margin: { l: opts.left || 150, r: 18, t: 10, b: 34 } }));
    const { svg, m, iw, ih } = f;
    const all = [];
    opts.rows.forEach((r) => { if (r.lo !== undefined) { all.push(r.lo, r.hi); } all.push(r.value); });
    let lo = opts.xmin !== undefined ? opts.xmin : Math.min(0, ...all);
    let hi = opts.xmax !== undefined ? opts.xmax : Math.max(0, ...all);
    const p = (hi - lo) * 0.05; lo -= p; hi += p;
    const X = (v) => m.l + ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * iw;
    const g = el("g", { class: "grid" }, svg);
    niceTicks(lo, hi, 5).forEach((v) => {
      el("line", { x1: X(v), x2: X(v), y1: m.t, y2: m.t + ih }, g);
      const t = el("text", { x: X(v), y: m.t + ih + 16, class: "tick", "text-anchor": "middle" }, svg);
      t.textContent = fmt(v, opts.digits);
    });
    if (lo < 0 && hi > 0) el("line", { x1: X(0), x2: X(0), y1: m.t, y2: m.t + ih, class: "zero" }, svg);
    if (opts.xlabel) { const t = el("text", { x: m.l + iw / 2, y: f.H - 4, class: "axlabel", "text-anchor": "middle" }, svg); t.textContent = opts.xlabel; }
    opts.rows.forEach((r, i) => {
      const y = m.t + i * rowH + rowH / 2;
      const lab = el("text", { x: m.l - 8, y: y + 4, class: r.head ? "rowlabel head" : "rowlabel", "text-anchor": "end" }, svg);
      lab.textContent = r.label;
      if (r.value === undefined || r.value === null) return;
      const off = r.value < lo || r.value > hi;
      if (r.lo !== undefined) el("line", { x1: X(r.lo), x2: X(r.hi), y1: y, y2: y, stroke: r.color, "stroke-width": 2, "stroke-linecap": "round" }, svg);
      el("circle", { cx: X(r.value), cy: y, r: 5, fill: off ? "var(--surface)" : r.color, stroke: r.color, "stroke-width": 2 }, svg);
      if (off) { const t = el("text", { x: X(r.value) + (r.value < lo ? 9 : -9), y: y + 4, class: "tick", "text-anchor": r.value < lo ? "start" : "end" }, svg); t.textContent = (r.value < lo ? "◀ " : "") + fmt(r.value, opts.digits) + (r.value > hi ? " ▶" : ""); }
      const hit = el("rect", { x: m.l, y: y - rowH / 2, width: iw, height: rowH, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", (evt) => showTip(evt, `<div class="tt">${r.label}</div><div>${opts.valuename || "Δτ"} <b>${fmt(r.value, opts.digits)}</b></div>` + (r.lo !== undefined ? `<div>95% CI <b>[${fmt(r.lo, opts.digits)}, ${fmt(r.hi, opts.digits)}]</b></div>` : "") + (r.note ? `<div class="muted">${r.note}</div>` : "")));
      hit.addEventListener("mouseleave", hideTip);
    });
    if (opts.series) legend(container, opts.series);
  };

  // Horizontal stacked bars (shares). opts: {rows:[{label, parts:[v...]}], parts:[{name,color}], total}
  window.stackChart = function (container, opts) {
    const rowH = 30;
    const H = opts.rows.length * rowH + 40;
    const f = frame(container, Object.assign({}, opts, { height: H, margin: { l: opts.left || 130, r: 18, t: 8, b: 30 } }));
    const { svg, m, iw, ih } = f;
    const X = (v) => m.l + v * iw;
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      el("line", { x1: X(v), x2: X(v), y1: m.t, y2: m.t + ih, class: "gridline" }, svg);
      const t = el("text", { x: X(v), y: m.t + ih + 16, class: "tick", "text-anchor": "middle" }, svg);
      t.textContent = Math.round(v * 100) + "%";
    });
    opts.rows.forEach((r, i) => {
      const y = m.t + i * rowH + 5;
      const lab = el("text", { x: m.l - 8, y: y + 14, class: "rowlabel", "text-anchor": "end" }, svg);
      lab.textContent = r.label;
      const tot = r.parts.reduce((a, b) => a + Math.max(b, 0), 0) || 1;
      let acc = 0;
      r.parts.forEach((v, j) => {
        const w = (Math.max(v, 0) / tot) * iw;
        if (w > 0.2) {
          const rect = el("rect", { x: X(acc / tot) + (acc > 0 ? 1 : 0), y, width: Math.max(w - (acc > 0 ? 1 : 0) - 1, 0.5), height: rowH - 10, fill: opts.parts[j].color, rx: 2 }, svg);
          rect.addEventListener("mousemove", (evt) => showTip(evt, `<div class="tt">${r.label}</div><div><i style="background:${opts.parts[j].color}"></i>${opts.parts[j].name}<b>${(100 * v / tot).toFixed(v / tot < 0.01 ? 2 : 1)}%</b></div>` + (r.raw ? `<div class="muted">${r.raw[j]}</div>` : "")));
          rect.addEventListener("mouseleave", hideTip);
        }
        acc += Math.max(v, 0);
      });
    });
    legend(container, opts.parts);
  };

  // Grouped vertical bars. opts: {groups:[label], series:[{name,color,values}], ylabel, ylog}
  window.barChart = function (container, opts) {
    const f = frame(container, opts);
    const { svg, m, iw, ih } = f;
    const ty = opts.ylog ? (v) => Math.log10(v) : (v) => v;
    const vals = [];
    opts.series.forEach((s) => s.values.forEach((v) => { if (v !== null && v > (opts.ylog ? 0 : -Infinity)) vals.push(ty(v)); }));
    let lo = opts.ylog ? Math.floor(Math.min(...vals)) : Math.min(0, ...vals);
    let hi = opts.ylog ? Math.ceil(Math.max(...vals)) : Math.max(0, ...vals) * 1.08;
    const Y = (v) => m.t + ih - ((ty(v) - lo) / (hi - lo || 1)) * ih;
    const yt = opts.ylog ? Array.from({ length: hi - lo + 1 }, (_, i) => Math.pow(10, lo + i)) : niceTicks(lo, hi, 5);
    yt.forEach((v) => {
      el("line", { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v), class: "gridline" }, svg);
      const t = el("text", { x: m.l - 6, y: Y(v) + 4, class: "tick", "text-anchor": "end" }, svg);
      t.textContent = opts.ylog ? "1e" + Math.round(Math.log10(v)) : fmt(v, opts.ydigits);
    });
    const gw = iw / opts.groups.length;
    const bw = Math.min(26, (gw * 0.8) / opts.series.length);
    const base = opts.ylog ? Math.pow(10, lo) : 0;
    opts.groups.forEach((gname, i) => {
      const t = el("text", { x: m.l + gw * i + gw / 2, y: m.t + ih + 16, class: "tick", "text-anchor": "middle" }, svg);
      t.textContent = gname;
      opts.series.forEach((s, j) => {
        const v = s.values[i];
        if (v === null || v === undefined || (opts.ylog && v <= 0)) return;
        const x = m.l + gw * i + gw / 2 - (bw * opts.series.length) / 2 + j * bw;
        const y0 = Y(base), y1 = Y(v);
        const rect = el("rect", { x: x + 1, y: Math.min(y0, y1), width: bw - 2, height: Math.max(Math.abs(y0 - y1), 1), fill: s.color, rx: 2 }, svg);
        rect.addEventListener("mousemove", (evt) => showTip(evt, `<div class="tt">${gname}</div><div><i style="background:${s.color}"></i>${s.name}<b>${fmt(v, opts.tipdigits)}</b></div>`));
        rect.addEventListener("mouseleave", hideTip);
      });
    });
    el("line", { x1: m.l, x2: m.l + iw, y1: Y(base), y2: Y(base), class: "axis" }, svg);
    if (opts.ylabel) { const t = el("text", { x: 12, y: m.t + ih / 2, class: "axlabel", transform: `rotate(-90 12 ${m.t + ih / 2})`, "text-anchor": "middle" }, svg); t.textContent = opts.ylabel; }
    legend(container, opts.series);
  };
})();
