/* Minimal SVG chart kit for the PERTURB24K results page (no dependencies).
   Every chart reads colors from CSS custom properties so both themes work. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const el = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };
  const fmt = (v, d = 3) => {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e4)) return v.toExponential(1);
    return (+v.toPrecision(d)).toString();
  };
  const signed = (v, d = 3) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v), d);

  function scale(domain, range, log) {
    const [d0, d1] = log ? domain.map(Math.log10) : domain;
    const f = (v) => range[0] + ((log ? Math.log10(v) : v) - d0) / (d1 - d0 || 1) * (range[1] - range[0]);
    f.domain = domain; f.log = log;
    return f;
  }
  function ticks(domain, log, n = 5) {
    if (log) {
      const out = [];
      for (let p = Math.floor(Math.log10(domain[0])); p <= Math.ceil(Math.log10(domain[1])); p++) {
        const v = Math.pow(10, p);
        if (v >= domain[0] * 0.999 && v <= domain[1] * 1.001) out.push(v);
      }
      return out;
    }
    const span = domain[1] - domain[0];
    const step = Math.pow(10, Math.floor(Math.log10(span / n)));
    const m = span / step > n * 2 ? (span / step > n * 4 ? 5 : 2) : 1;
    const s = step * m;
    const out = [];
    for (let v = Math.ceil(domain[0] / s) * s; v <= domain[1] + 1e-12; v += s) out.push(+v.toPrecision(10));
    return out;
  }
  function extent(vals, log, pad = 0.05) {
    let v = vals.filter((x) => x !== null && Number.isFinite(x) && (!log || x > 0));
    let lo = Math.min(...v), hi = Math.max(...v);
    if (log) return [Math.pow(10, Math.floor(Math.log10(lo) * 4) / 4), Math.pow(10, Math.ceil(Math.log10(hi) * 4) / 4)];
    const p = (hi - lo || Math.abs(hi) || 1) * pad;
    return [lo - p, hi + p];
  }

  // Shared tooltip
  let tip;
  function tooltip() {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "viz-tip";
      tip.hidden = true;
      document.body.appendChild(tip);
    }
    return tip;
  }
  function showTip(evt, html) {
    const t = tooltip();
    t.innerHTML = html;
    t.hidden = false;
    const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
    t.style.left = Math.max(8, x) + "px";
    t.style.top = Math.max(8, y) + "px";
  }
  function hideTip() { if (tip) tip.hidden = true; }

  function frame(container, opts) {
    container.innerHTML = "";
    const W = opts.width || 640, H = opts.height || 320;
    const m = Object.assign({ t: 14, r: 16, b: 44, l: 58 }, opts.margin || {});
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "viz", role: "img", "aria-label": opts.label || "" }, container);
    return { svg, W, H, m, iw: W - m.l - m.r, ih: H - m.t - m.b };
  }
  function axes(f, x, y, opts) {
    const g = el("g", {}, f.svg);
    const grid = css("--grid"), axis = css("--axis"), muted = css("--muted");
    for (const v of opts.yticks || ticks(y.domain, y.log)) {
      const yy = y(v);
      if (yy < f.m.t - 1 || yy > f.m.t + f.ih + 1) continue;
      el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: yy, y2: yy, stroke: grid, "stroke-width": 1 }, g);
      const t = el("text", { x: f.m.l - 6, y: yy + 3.5, "text-anchor": "end", class: "tick", fill: muted }, g);
      t.textContent = (opts.yfmt || fmt)(v);
    }
    for (const v of opts.xticks || ticks(x.domain, x.log)) {
      const xx = x(v);
      if (xx < f.m.l - 1 || xx > f.m.l + f.iw + 1) continue;
      el("line", { x1: xx, x2: xx, y1: f.m.t + f.ih, y2: f.m.t + f.ih + 4, stroke: axis }, g);
      const t = el("text", { x: xx, y: f.m.t + f.ih + 16, "text-anchor": "middle", class: "tick", fill: muted }, g);
      t.textContent = (opts.xfmt || fmt)(v);
    }
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: f.m.t + f.ih, y2: f.m.t + f.ih, stroke: axis }, g);
    if (opts.xlabel) {
      const t = el("text", { x: f.m.l + f.iw / 2, y: f.H - 6, "text-anchor": "middle", class: "axlabel", fill: css("--fg-2") }, g);
      t.textContent = opts.xlabel;
    }
    if (opts.ylabel) {
      const t = el("text", { x: 12, y: f.m.t + f.ih / 2, "text-anchor": "middle", class: "axlabel", fill: css("--fg-2"),
        transform: `rotate(-90 12 ${f.m.t + f.ih / 2})` }, g);
      t.textContent = opts.ylabel;
    }
    if (opts.zero && !y.log && y.domain[0] < 0 && y.domain[1] > 0) {
      el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(0), y2: y(0), stroke: axis, "stroke-width": 1.2 }, g);
    }
  }

  /* Line chart. series: [{name, color, x, y, lo?, hi?, dash?}] */
  function line(container, opts) {
    const f = frame(container, opts);
    const all = opts.series.flatMap((s) => [...s.y, ...(s.lo || []), ...(s.hi || [])]);
    const xs = opts.series.flatMap((s) => s.x);
    const x = scale(opts.xdomain || extent(xs, opts.logx, 0.02), [f.m.l, f.m.l + f.iw], opts.logx);
    const ydom = opts.ydomain || extent(all.concat(opts.refs ? opts.refs.map((r) => r.y) : []), opts.logy);
    const y = scale(ydom, [f.m.t + f.ih, f.m.t], opts.logy);
    axes(f, x, y, opts);
    const clamp = (v) => (opts.logy ? Math.max(v, ydom[0]) : v);
    for (const r of opts.refs || []) {
      el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(clamp(r.y)), y2: y(clamp(r.y)), stroke: css("--muted"), "stroke-dasharray": "3 3" }, f.svg);
      const t = el("text", { x: f.m.l + f.iw - 2, y: y(clamp(r.y)) - 4, "text-anchor": "end", class: "tick", fill: css("--muted") }, f.svg);
      t.textContent = r.label;
    }
    for (const s of opts.series) {
      if (s.lo) {
        const pts = s.x.map((v, i) => `${x(v)},${y(clamp(s.hi[i]))}`).concat(s.x.map((v, i) => `${x(v)},${y(clamp(s.lo[i]))}`).reverse());
        el("polygon", { points: pts.join(" "), fill: s.color, "fill-opacity": 0.16, stroke: "none" }, f.svg);
      }
      el("polyline", { points: s.x.map((v, i) => `${x(v)},${y(clamp(s.y[i]))}`).join(" "), fill: "none", stroke: s.color,
        "stroke-width": s.width || 2, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round" }, f.svg);
      s.x.forEach((v, i) => {
        el("circle", { cx: x(v), cy: y(clamp(s.y[i])), r: s.dash ? 2.5 : 3.2, fill: s.color, stroke: css("--surface"), "stroke-width": 1.5 }, f.svg);
      });
    }
    // Hover: nearest x column
    const xsU = [...new Set(xs)].sort((a, b) => a - b);
    const cross = el("line", { y1: f.m.t, y2: f.m.t + f.ih, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden" }, f.svg);
    const hit = el("rect", { x: f.m.l, y: f.m.t, width: f.iw, height: f.ih, fill: "transparent" }, f.svg);
    hit.addEventListener("mousemove", (evt) => {
      const pt = f.svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
      const p = pt.matrixTransform(f.svg.getScreenCTM().inverse());
      let best = xsU[0];
      for (const v of xsU) if (Math.abs(x(v) - p.x) < Math.abs(x(best) - p.x)) best = v;
      cross.setAttribute("x1", x(best)); cross.setAttribute("x2", x(best)); cross.setAttribute("visibility", "visible");
      const rows = opts.series.map((s) => {
        const i = s.x.indexOf(best);
        if (i < 0) return "";
        const band = s.lo ? ` <span class="tip-mute">[${fmt(s.lo[i])}, ${fmt(s.hi[i])}]</span>` : "";
        return `<div><i style="background:${s.color}"></i>${s.name}: <b>${(opts.tipfmt || fmt)(s.y[i])}</b>${band}</div>`;
      }).join("");
      showTip(evt, `<div class="tip-h">${opts.xname || "x"} = ${fmt(best)}</div>${rows}`);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
    legend(container, opts.series);
  }

  function legend(container, series) {
    const div = document.createElement("div");
    div.className = "viz-legend";
    for (const s of series) {
      const span = document.createElement("span");
      span.innerHTML = `<i style="background:${s.color};${s.dash ? "opacity:.6" : ""}"></i>${s.name}`;
      div.appendChild(span);
    }
    container.appendChild(div);
  }

  /* Scatter. points: [{x, y, group, html}], groups: {name: {color, shape, size, label}} */
  function scatter(container, opts) {
    const f = frame(container, opts);
    const pts = opts.points.filter((p) => !opts.logx || p.x > 0);
    const x = scale(opts.xdomain || extent(pts.map((p) => p.x), opts.logx), [f.m.l, f.m.l + f.iw], opts.logx);
    const yd = opts.ydomain || extent(pts.map((p) => p.y), false);
    const y = scale(yd, [f.m.t + f.ih, f.m.t], false);
    axes(f, x, y, Object.assign({ zero: true }, opts));
    const cy = (v) => y(Math.max(yd[0], Math.min(yd[1], v)));
    for (const ln of opts.lines || []) {
      const p = ln.points.filter((q) => !opts.logx || q.x > 0);
      const d = ln.step
        ? p.map((q, i) => (i === 0 ? `M${x(q.x)},${cy(q.y)}` : `H${x(q.x)}V${cy(q.y)}`)).join("") + (ln.extend ? `H${f.m.l + f.iw}` : "")
        : p.map((q, i) => `${i ? "L" : "M"}${x(q.x)},${cy(q.y)}`).join("");
      el("path", { d, fill: "none", stroke: ln.color, "stroke-width": ln.width || 1.6, "stroke-dasharray": ln.dash || "" }, f.svg);
    }
    const order = Object.keys(opts.groups);
    for (const g of order) {
      const st = opts.groups[g];
      for (const p of pts.filter((q) => q.group === g)) {
        const px = x(p.x), py = cy(p.y), r = st.size || 3.5;
        let mk;
        if (st.shape === "square") mk = el("rect", { x: px - r, y: py - r, width: 2 * r, height: 2 * r, rx: 1 }, f.svg);
        else if (st.shape === "diamond") mk = el("path", { d: `M${px},${py - r * 1.3}L${px + r * 1.3},${py}L${px},${py + r * 1.3}L${px - r * 1.3},${py}Z` }, f.svg);
        else if (st.shape === "triangle") mk = el("path", { d: `M${px},${py - r * 1.3}L${px + r * 1.2},${py + r * 0.9}L${px - r * 1.2},${py + r * 0.9}Z` }, f.svg);
        else if (st.shape === "tridown") mk = el("path", { d: `M${px},${py + r * 1.3}L${px + r * 1.2},${py - r * 0.9}L${px - r * 1.2},${py - r * 0.9}Z` }, f.svg);
        else if (st.shape === "star") {
          const s = [];
          for (let k = 0; k < 10; k++) { const a = Math.PI / 5 * k - Math.PI / 2, rr = k % 2 ? r * 0.8 : r * 1.9; s.push(`${px + rr * Math.cos(a)},${py + rr * Math.sin(a)}`); }
          mk = el("polygon", { points: s.join(" ") }, f.svg);
        } else mk = el("circle", { cx: px, cy: py, r }, f.svg);
        mk.setAttribute("fill", st.color);
        mk.setAttribute("fill-opacity", st.opacity || 0.8);
        mk.setAttribute("stroke", css("--surface"));
        mk.setAttribute("stroke-width", 1);
        if (p.html) {
          const hit = el("circle", { cx: px, cy: py, r: Math.max(7, r + 3), fill: "transparent" }, f.svg);
          hit.addEventListener("mousemove", (evt) => showTip(evt, p.html));
          hit.addEventListener("mouseleave", hideTip);
        }
        if (p.label) {
          const t = el("text", { x: px + 7, y: py - 6, class: "plabel", fill: css("--fg-2") }, f.svg);
          t.textContent = p.label;
        }
      }
    }
    legend(container, order.map((g) => ({ name: opts.groups[g].label || g, color: opts.groups[g].color })).concat(
      (opts.lines || []).filter((l) => l.name).map((l) => ({ name: l.name, color: l.color, dash: true }))));
  }

  /* Heatmap. rows, cols, values[r][c], color(v) -> css color, fmt */
  function heatmap(container, opts) {
    container.innerHTML = "";
    const cw = opts.cellW || 62, ch = opts.cellH || 30, lw = opts.labelW || 70, th = 22;
    const W = lw + cw * opts.cols.length + 4, H = th + ch * opts.rows.length + 4;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "viz heat", role: "img", "aria-label": opts.label || "" }, container);
    svg.style.maxWidth = W * 1.25 + "px";
    opts.cols.forEach((c, j) => {
      const t = el("text", { x: lw + cw * j + cw / 2, y: 15, "text-anchor": "middle", class: "tick", fill: css("--fg-2") }, svg);
      t.textContent = c;
    });
    opts.rows.forEach((r, i) => {
      const t = el("text", { x: lw - 8, y: th + ch * i + ch / 2 + 4, "text-anchor": "end", class: "tick", fill: css("--fg-2") }, svg);
      t.textContent = r;
      opts.cols.forEach((c, j) => {
        const v = opts.values[i][j];
        const [fill, ink] = opts.color(v);
        const rect = el("rect", { x: lw + cw * j + 1, y: th + ch * i + 1, width: cw - 2, height: ch - 2, rx: 3, fill }, svg);
        const t2 = el("text", { x: lw + cw * j + cw / 2, y: th + ch * i + ch / 2 + 4, "text-anchor": "middle", class: "cell", fill: ink }, svg);
        t2.textContent = (opts.fmt || fmt)(v);
        const html = opts.tip ? opts.tip(i, j, v) : `${r} · ${c}: <b>${fmt(v)}</b>`;
        rect.addEventListener("mousemove", (evt) => showTip(evt, html));
        rect.addEventListener("mouseleave", hideTip);
        t2.style.pointerEvents = "none";
      });
    });
  }

  /* Sequential color from a ramp of hex stops; returns [fill, ink]. */
  function ramp(stops, inkSwitch = 0.55) {
    const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    const cs = stops.map(hex);
    return (t) => {
      t = Math.max(0, Math.min(1, t));
      const p = t * (cs.length - 1), i = Math.min(Math.floor(p), cs.length - 2), u = p - i;
      const c = cs[i].map((v, k) => Math.round(v + (cs[i + 1][k] - v) * u));
      return [`rgb(${c.join(",")})`, t > inkSwitch ? "#ffffff" : "#0b0b0b"];
    };
  }

  /* Grouped bars. groups: [{label, values: [v per series]}], series: [{name, color}] */
  function bars(container, opts) {
    const f = frame(container, Object.assign({ margin: { b: 92 } }, opts));
    const n = opts.groups.length, k = opts.series.length;
    const all = opts.groups.flatMap((g) => g.values);
    const y = scale(opts.ydomain || [0, Math.max(...all) * 1.12], [f.m.t + f.ih, f.m.t], false);
    const x = scale([0, n], [f.m.l, f.m.l + f.iw], false);
    axes(f, x, y, Object.assign({}, opts, { xticks: [] }));
    const bw = Math.min(16, (f.iw / n - 6) / k);
    opts.groups.forEach((g, i) => {
      const cx = x(i + 0.5);
      g.values.forEach((v, s) => {
        const bx = cx - (k * bw) / 2 + s * bw + 1;
        const top = y(Math.max(v, 0)), h = Math.max(1, y(0) - top);
        const r = Math.min(3, bw / 2 - 1);
        el("path", { d: `M${bx},${y(0)}V${top + r}Q${bx},${top} ${bx + r},${top}H${bx + bw - 2 - r}Q${bx + bw - 2},${top} ${bx + bw - 2},${top + r}V${y(0)}Z`,
          fill: opts.series[s].color }, f.svg);
        const hit = el("rect", { x: bx - 1, y: f.m.t, width: bw, height: f.ih, fill: "transparent" }, f.svg);
        hit.addEventListener("mousemove", (evt) => showTip(evt, g.html || `${g.label}<br>${opts.series[s].name}: <b>${fmt(v)}</b>`));
        hit.addEventListener("mouseleave", hideTip);
      });
      const t = el("text", { x: cx, y: f.m.t + f.ih + 12, class: "tick", fill: css("--fg-2"), "text-anchor": "end",
        transform: `rotate(-35 ${cx} ${f.m.t + f.ih + 12})` }, f.svg);
      t.textContent = g.label;
      if (g.note) {
        const t2 = el("text", { x: cx, y: y(Math.max(...g.values)) - 5, "text-anchor": "middle", class: "plabel", fill: css("--fg-2") }, f.svg);
        t2.textContent = g.note;
      }
    });
    legend(container, opts.series);
  }

  window.Viz = { line, scatter, heatmap, bars, ramp, fmt, signed, css, el, showTip, hideTip };
})();
