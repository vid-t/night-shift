/* HAI attack reader: the write-up view. Plain JavaScript, no build step.
   Data: data/demo.json (check set), data/results_extra.json (test set, huge model).
   Binds the numbers in the static text and draws the charts (SVG). */
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var D = null;          // demo.json
  var X = null;          // results_extra.json
  var redraws = {};      // charts that redraw on resize
  // The write-up is one of two views (js/views.js). Its charts are sized from element widths,
  // so they draw only while it is visible, and redraw when it becomes visible.
  function shown() { return document.body.getAttribute("data-view") === "writeup"; }

  var FAMILIES = [
    ["hints", "With hints beside the curves", "circle"],
    ["nohint", "Newer designs, no hints", "square"],
    ["zscore", "Curve scaled to its own mean and spread, no hints", "diamond"],
    ["range", "The design we started with, and its variants", "triangle"]
  ];

  /* ---------- design names in plain words (raw names come from the search's run table) ----------
     The write-up does not explain "view": each raw view name becomes how the curve is turned into numbers. */
  var VIEW_PLAIN = [
    ["Range view", "Starting design"],
    ["Z-score view", "Curve scaled to its own mean and spread"],
    ["Hybrid", "Curve scaled to its own mean and spread, with the size of its largest change"],
    ["Loop-relative", "Curve against how much its loop normally moves"]
  ];
  var CLAUSE_PLAIN = {
    step: "largest one-second jump and its second",
    law: "how long the valve moved against the controller",
    plant: "gap between a measurement and what its command predicts",
    size: "size of the largest change"
  };
  var MOD_PLAIN = [
    [/^no clause$/, "no extra hints"],
    [/^pass chosen by loss$/, "kept the training pass with the lowest training error"],
    [/^start `har`$/, "started from the OpenTSLM model trained on human activity"],
    [/^start `tsqa`$/, "started from the OpenTSLM model trained on curve questions"],
    [/^starts from another run$/, "started from another run's model"],
    [/^attack share ([\d.]+)$/, function (m) { return Math.round(parseFloat(m[1]) * 100) + "% attack examples in training"; }],
    [/^mirrored rows ([\d.]+)$/, function (m) { return "plus mirrored copies (rises and falls swapped) of " + Math.round(parseFloat(m[1]) * 100) + "% of training rows"; }],
    [/^(\d+) passes$/, function (m) { return m[1] + " passes through the training data"; }],
    [/^verdict only$/, "trained to write only the verdict"],
    [/^verdict first$/, "trained to write the verdict before the sentence"],
    [/^encoder step .*$/, "curve encoder trained more slowly"]
  ];
  function plainDesign(design) {
    var parts = design.split(", ");
    var head = parts.shift();
    var hm = head.match(/^(.*?)(?: \+ (.*))?$/);
    var view = hm[1];
    VIEW_PLAIN.forEach(function (v) { if (view === v[0]) view = v[1]; });
    var out = view;
    var clauses = hm[2] ? hm[2].split(" and ") : [];
    while (parts.length && parts[0].split(" and ").every(function (c) { return CLAUSE_PLAIN[c]; })) {
      clauses = clauses.concat(parts.shift().split(" and "));
    }
    if (clauses.length) out += "; hints: " + clauses.map(function (c) { return CLAUSE_PLAIN[c] || c; }).join(", ");
    parts.forEach(function (p) {
      var done = false;
      MOD_PLAIN.forEach(function (m) {
        if (done) return;
        var mm = p.match(m[0]);
        if (mm) { out += "; " + (typeof m[1] === "function" ? m[1](mm) : m[1]); done = true; }
      });
      if (!done) out += "; " + p.replace(/`/g, "");
    });
    return out;
  }

  /* ---------- small helpers ---------- */
  function $(sel, root) { return (root || document).querySelector(sel); }
  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === "text") n.textContent = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
      else n.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(function (c) { if (c != null) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return n;
  }
  function svg(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function txt(parent, x, y, s, attrs) {
    var a = Object.assign({ x: x, y: y, fill: "var(--ink-2)", "font-size": 12 }, attrs || {});
    var t = svg("text", a, parent);
    t.textContent = s;
    return t;
  }
  function fmt(v) {
    if (typeof v === "number") return v.toLocaleString("en-US", { maximumFractionDigits: 2 }).replace(/^-/, "\u2212");
    return String(v);
  }
  function get(obj, path) {
    return path.split(".").reduce(function (o, k) { return o == null ? undefined : o[k]; }, obj);
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; });
  }
  function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
  function width(node) { return Math.max(280, Math.floor(node.getBoundingClientRect().width)); }

  /* ---------- tooltip ---------- */
  var tip = null;
  function showTip(html, x, y) {
    tip.innerHTML = html;
    tip.hidden = false;
    var r = tip.getBoundingClientRect();
    var left = Math.min(window.innerWidth - r.width - 8, Math.max(8, x + 14));
    var top = y - r.height - 12;
    if (top < 8) top = y + 18;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }
  function hideTip() { tip.hidden = true; }

  /* ---------- text pieces ---------- */
  function hintItems(list, hints) {
    list.innerHTML = "";
    hints.forEach(function (h) {
      var line = h.replace(/:\s*$/, "");
      var m = line.match(/^This is the ([^,]+),(.*)$/);
      var li = el("li");
      if (!m) { li.textContent = line; list.appendChild(li); return; }
      var name = m[1], rest = m[2];
      var parts = rest.split(";");
      var desc = parts.shift();
      var html = 'This is the <span class="hint__sig">' + esc(name) + "</span>," + esc(desc);
      parts.forEach(function (p) { html += "; <mark>" + esc(p.trim()) + "</mark>"; });
      li.innerHTML = html + ".";
      list.appendChild(li);
    });
  }
  function answerHTML(text) {
    var t = esc(text).replace(/Answer:\s*(attack|normal)/i, function (_, v) {
      return '<span class="verdict verdict--' + v.toLowerCase() + '">Answer: ' + v.toLowerCase() + "</span>";
    });
    return "<q>" + t + "</q>";
  }

  /* ---------- the window chart: one band per signal ---------- */
  function seriesChart(host, w, opts) {
    opts = opts || {};
    var W = width(host);
    var compact = opts.compact || W < 520;
    var bandH = compact ? 38 : 52, gapH = 24;
    var padR = 6;
    var truth = w.attack_start;
    var ans = opts.answerStart;
    var both = truth != null && ans != null;
    var top = both ? 58 : 40, bottom = 30;
    var n = w.series.length;
    var H = top + n * (bandH + gapH) + bottom - gapH + 6;
    host.innerHTML = "";
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, host);
    var x = function (sec) { return (sec / 299) * (W - padR); };

    var bandsTop = top, bandsBot = top + n * (bandH + gapH) - gapH;

    // x grid
    for (var g = 0; g <= 300; g += 60) {
      var gx = Math.min(x(g), W - padR);
      svg("line", { x1: gx, x2: gx, y1: bandsTop - 6, y2: bandsBot + 4, stroke: "var(--rule-soft)", "stroke-width": 1 }, s);
      txt(s, gx, H - 8, g === 300 ? "300 s" : String(g), { "text-anchor": g === 0 ? "start" : g === 300 ? "end" : "middle", fill: "var(--ink-3)", "font-size": 11 });
    }

    w.series.forEach(function (ser, i) {
      var y0 = top + i * (bandH + gapH);
      var vals = ser.values;
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
      var constant = hi - lo < 1e-9;
      var y = function (v) {
        if (constant) return y0 + bandH / 2;
        return y0 + bandH - ((v - lo) / (hi - lo)) * bandH;
      };
      svg("line", { x1: 0, x2: W - padR, y1: y0 + bandH, y2: y0 + bandH, stroke: "var(--rule)", "stroke-width": 1 }, s);
      var label = cap(ser.name);
      if (constant) label += " (never changes)";
      txt(s, 0, y0 - 7, label, { fill: "var(--ink)", "font-size": 12, "font-weight": 600 });
      var d = "";
      for (var k = 0; k < vals.length; k++) d += (k ? "L" : "M") + x(k).toFixed(1) + " " + y(vals[k]).toFixed(1);
      var isTarget = ser.role === "target";
      svg("path", {
        d: d, fill: "none", "stroke-linejoin": "round",
        stroke: isTarget ? "var(--ink-3)" : ser.role === "command" ? "var(--ink-2)" : "var(--ink)",
        "stroke-width": isTarget ? 1.5 : 1.75,
        "stroke-dasharray": isTarget ? "4 3" : "none"
      }, s);
    });

    var labelLayer = [];
    function marker(sec, color, dashed, label, side, row) {
      var mx = x(Math.max(0, Math.min(299, sec)));
      svg("line", { x1: mx, x2: mx, y1: 4 + row * 14, y2: bandsBot + 4, stroke: color, "stroke-width": 2, "stroke-dasharray": dashed ? "5 4" : "none" }, s);
      var lw = label.length * 6.4;
      if (side === "right" && mx + 6 + lw > W) side = "left";
      if (side === "left" && mx - 6 - lw < 0) side = "right";
      labelLayer.push([side === "left" ? mx - 6 : mx + 6, 14 + row * 14, label, color, side === "left" ? "end" : "start"]);
    }
    var tSide = "right", aSide = "right";
    if (truth != null && ans != null) { if (ans >= truth) { tSide = "left"; aSide = "right"; } else { tSide = "right"; aSide = "left"; } }
    if (truth != null) marker(truth, "var(--attack)", false, "attack starts, second " + truth, tSide, 0);
    if (ans != null) marker(ans, "var(--model)", true, "answer: second " + ans, aSide, both ? 1 : 0);
    labelLayer.forEach(function (l) {
      txt(s, l[0], l[1], l[2], { fill: l[3], "font-size": 12, "font-weight": 700, "text-anchor": l[4], stroke: "var(--paper)", "stroke-width": 4, "paint-order": "stroke", "stroke-linejoin": "round" });
    });

    // crosshair + tooltip
    var cross = svg("line", { x1: 0, x2: 0, y1: bandsTop - 6, y2: bandsBot + 4, stroke: "var(--ink)", "stroke-width": 1, opacity: 0 }, s);
    var hit = svg("rect", { x: 0, y: bandsTop - 10, width: W, height: bandsBot - bandsTop + 14, fill: "transparent" }, s);
    function move(ev) {
      var r = s.getBoundingClientRect();
      var px = (ev.clientX - r.left) * (W / r.width);
      var sec = Math.max(0, Math.min(299, Math.round((px / (W - padR)) * 299)));
      cross.setAttribute("x1", x(sec)); cross.setAttribute("x2", x(sec)); cross.setAttribute("opacity", .5);
      var html = "<b>Second " + sec + "</b>";
      w.series.forEach(function (ser) {
        html += "<br>" + esc(cap(ser.name)) + ": " + fmt(ser.values[sec]);
      });
      showTip(html, ev.clientX, ev.clientY);
    }
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerleave", function () { cross.setAttribute("opacity", 0); hideTip(); });
  }

  /* ---------- the example window: part 1 (the data) and part 3 (what the model reads and writes) ---------- */
  function heroWindow() { return D.windows.find(function (x) { return x.id === D.hero; }); }
  function renderWindowFigures() {
    var w = heroWindow();
    // Part 1 shows the data only: the true start, no answer yet.
    redraws.window = function () { seriesChart($("#window-chart"), w, {}); };
    redraws.flow = function () { seriesChart($("#flow-chart"), w, { answerStart: w.vote_start, compact: true }); };
    hintItems($("#flow-hints"), w.hints);
    $("#flow-answer").innerHTML = answerHTML(w.vote_example.text) +
      ' <span class="voters__rule">' + w.votes_attack + " of " + w.voters.length + " models said attack. The truth: attack from second " + w.attack_start + ".</span>";
  }

  /* ---------- curve-check bars ---------- */
  function readsChart() {
    var host = $("#reads-chart");
    var W = width(host);
    var t = D.curve_check.totals, max = D.curve_check.rows.attack;
    var rows = [["Normal curve", t.machine.attack_called], ["Blank curve", t.flat.attack_called], ["Shuffled curve", t.curveshuffle.attack_called]];
    var labW = W < 480 ? 0 : 118, barH = 26, gap = W < 480 ? 34 : 16, top = 8;
    var H = top + rows.length * (barH + gap) + 28;
    host.innerHTML = "";
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, host);
    var plotW = W - labW - 44;
    var x = function (v) { return labW + (v / max) * plotW; };
    [0, 36, 72, 108, 144].forEach(function (g) {
      if (g > max) return;
      svg("line", { x1: x(g), x2: x(g), y1: top - 4, y2: H - 22, stroke: g === max ? "var(--ink-3)" : "var(--rule)", "stroke-width": 1, "stroke-dasharray": g === max ? "3 3" : "none" }, s);
      txt(s, x(g), H - 6, g === max ? max + " = all" : String(g), { "text-anchor": tickAnchor(x(g), g === max), fill: "var(--ink-3)", "font-size": 11 });
    });
    rows.forEach(function (r, i) {
      var y = top + i * (barH + gap) + (W < 480 ? 18 : 0);
      if (labW) txt(s, 0, y + barH / 2 + 4, r[0], { fill: "var(--ink)", "font-size": 13, "font-weight": 600 });
      else txt(s, 0, y - 6, r[0], { fill: "var(--ink)", "font-size": 13, "font-weight": 600 });
      var bw = Math.max(2, x(r[1]) - labW);
      var bar = svg("rect", { x: labW, y: y, width: bw, height: barH, rx: 2, fill: i === 0 ? "var(--attack)" : i === 1 ? "#d2475c" : "#e48a98" }, s);
      txt(s, labW + bw + 6, y + barH / 2 + 5, String(r[1]), { fill: "var(--ink)", "font-size": 15, "font-weight": 750 });
      bar.addEventListener("pointermove", function (ev) { showTip("<b>" + r[0] + "</b><br>" + r[1] + " of " + max + " attack windows called attack", ev.clientX, ev.clientY); });
      bar.addEventListener("pointerleave", hideTip);
    });
  }

  /* ---------- check-set scores on one 0 to 144 scale (part 2: the comparisons; part 3: the vote) ---------- */
  function scoreRows(host, rows) {
    var W = width(host);
    var P = D.score.perfect;
    var narrow = W < 640;
    var labW = narrow ? 0 : 300, rowH = narrow ? 58 : 44, top = 22;
    var H = top + rows.length * rowH + 30;
    host.innerHTML = "";
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, host);
    var plotW = W - labW - 16;
    var x = function (v) { return labW + (Math.max(0, v) / P) * plotW; };
    [0, 36, 72, 108, P].forEach(function (g) {
      svg("line", { x1: x(g), x2: x(g), y1: top - 10, y2: H - 24, stroke: g === P ? "var(--ink)" : "var(--rule)", "stroke-width": 1, "stroke-dasharray": g === P || g === 0 ? "none" : "2 3" }, s);
      txt(s, x(g), H - 8, g === P ? "perfect " + P : String(g), { "text-anchor": tickAnchor(x(g), g === P), fill: g === P ? "var(--ink)" : "var(--ink-3)", "font-size": 11, "font-weight": g === P ? 700 : 400 });
    });
    rows.forEach(function (r, i) {
      var cy = top + i * rowH + (narrow ? 34 : rowH / 2);
      var ly = narrow ? cy - 16 : cy + 4;
      txt(s, 0, ly, r.label, { fill: "var(--ink)", "font-size": 13, "font-weight": r.big ? 750 : 600 });
      var tipHtml;
      if (r.range) {
        svg("rect", { x: x(r.range[0]), y: cy - (r.dots ? 2 : 7), width: Math.max(3, x(r.range[1]) - x(r.range[0])), height: r.dots ? 4 : 14, rx: 2, fill: r.color, opacity: r.soft ? .45 : r.dots ? .35 : .8 }, s);
        if (r.dots) r.dots.forEach(function (d) { svg("circle", { cx: x(d), cy: cy, r: 5, fill: r.color, stroke: "var(--paper)", "stroke-width": 2 }, s); });
        txt(s, x(r.range[1]) + (r.dots ? 13 : 10), cy + 4, fmt(r.range[0]) + " to " + fmt(r.range[1]), { fill: "var(--ink-2)", "font-size": 12, "font-weight": 650 });
        tipHtml = "<b>" + esc(r.label) + "</b><br>" + fmt(r.range[0]) + " to " + fmt(r.range[1]) + (r.dots ? "<br>repeats: " + r.dots.join(", ") : "");
      } else if (r.value < 0) {
        // Below zero: an arrow at zero and the number as text, never a stretched axis (as the test chart draws the range alarm).
        var ax = x(0);
        svg("path", { d: "M" + (ax + 16) + " " + cy + "H" + (ax + 2) + "M" + (ax + 8) + " " + (cy - 5) + "L" + (ax + 2) + " " + cy + "L" + (ax + 8) + " " + (cy + 5), stroke: r.color, "stroke-width": 2.5, fill: "none", "stroke-linecap": "round", "stroke-linejoin": "round" }, s);
        txt(s, ax + 24, cy + 5, fmt(r.value), { fill: "var(--ink)", "font-size": 14, "font-weight": 800 });
        tipHtml = "<b>" + esc(r.label) + "</b><br>score " + fmt(r.value) + ", below zero";
      } else {
        svg("circle", { cx: x(r.value), cy: cy, r: r.big ? 9 : 7, fill: r.hollow ? "var(--paper)" : r.color, stroke: r.color, "stroke-width": r.hollow ? 2.5 : 2 }, s);
        txt(s, x(r.value) + (r.big ? 15 : 13), cy + 5, fmt(r.value), { fill: "var(--ink)", "font-size": r.big ? 17 : 14, "font-weight": 800 });
        tipHtml = "<b>" + esc(r.label) + "</b><br>score " + fmt(r.value);
      }
      if (r.counts) tipHtml += " = " + r.counts.caught + " caught − " + r.counts.false_alarms + " false alarms";
      var hit = svg("rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: "transparent" }, s);
      hit.addEventListener("pointermove", function (ev) { showTip(tipHtml, ev.clientX, ev.clientY); });
      hit.addEventListener("pointerleave", hideTip);
    });
    host.setAttribute("aria-label", "Check-set scores, perfect " + P + ": " + rows.map(function (r) {
      return r.label + " " + (r.range ? fmt(r.range[0]) + " to " + fmt(r.range[1]) : fmt(r.value));
    }).join("; ") + ".");
  }
  function gbRows() {
    var sc = D.score;
    return [
      { label: "Gradient boosting", value: sc.gradient_boosting.score, counts: sc.gradient_boosting, color: "var(--ink)", hollow: true },
      { label: "Gradient boosting reading the other loops", value: sc.gradient_boosting_neighbours.score, counts: sc.gradient_boosting_neighbours, color: "var(--ink)", hollow: true }
    ];
  }
  function baselineChart() {
    var ra = D.score.range_alarm;
    scoreRows($("#baseline-chart"), [{ label: "Range alarm", value: ra.score, counts: ra, color: "var(--night-range)" }].concat(gbRows()));
  }
  function scoreChart() {
    var sc = D.score;
    scoreRows($("#score-chart"), [
      { label: "Hint-heavy design, one model, each repeat", range: [sc.voting_design_single.low, sc.voting_design_single.high], dots: sc.voting_design_single.scores, color: "var(--model)" },
      { label: "Seven models voting", value: sc.vote.score, counts: sc.vote, color: "var(--model)", big: true }
    ].concat(gbRows()));
  }

  /* ---------- the night: 64 runs ---------- */
  function shapePath(kind, cx, cy, r) {
    if (kind === "circle") return null;
    if (kind === "square") return "M" + (cx - r) + " " + (cy - r) + "h" + 2 * r + "v" + 2 * r + "h" + -2 * r + "Z";
    if (kind === "diamond") { var q = r * 1.3; return "M" + cx + " " + (cy - q) + "L" + (cx + q) + " " + cy + "L" + cx + " " + (cy + q) + "L" + (cx - q) + " " + cy + "Z"; }
    var t = r * 1.25; return "M" + cx + " " + (cy - t) + "L" + (cx + t) + " " + (cy + t * .8) + "L" + (cx - t) + " " + (cy + t * .8) + "Z";
  }
  function dot(parent, fam, cx, cy, r) {
    var f = FAMILIES.find(function (x) { return x[0] === fam; });
    var color = "var(--night-" + fam + ")";
    var p = shapePath(f[2], cx, cy, r);
    return p ? svg("path", { d: p, fill: color, stroke: "var(--paper)", "stroke-width": 1.5 }, parent)
      : svg("circle", { cx: cx, cy: cy, r: r + .5, fill: color, stroke: "var(--paper)", "stroke-width": 1.5 }, parent);
  }
  function nightLegend() {
    var host = $("#night-legend");
    host.innerHTML = "";
    FAMILIES.forEach(function (f) {
      var sp = el("span");
      var s = svg("svg", { viewBox: "0 0 14 14" }, sp);
      dot(s, f[0], 7, 7, 5);
      sp.appendChild(document.createTextNode(f[1]));
      host.appendChild(sp);
    });
    var sp = el("span");
    var s = svg("svg", { viewBox: "0 0 14 14" }, sp);
    svg("circle", { cx: 7, cy: 7, r: 6, fill: "none", stroke: "var(--ink)", "stroke-width": 1.75 }, s);
    sp.appendChild(document.createTextNode("Voting run"));
    host.appendChild(sp);
  }
  function nightChart() {
    var host = $("#night-chart");
    var W = width(host);
    var runs = D.night.runs, R = D.night.rounds;
    var padL = 34, top = 24, plotH = W < 520 ? 300 : 380, bottom = 34;
    var H = top + plotH + bottom;
    var lo = -10, hi = 100;
    host.innerHTML = "";
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, host);
    // gradient boosting reference lines: the right margin is as wide as the longest label, so it never clips
    var refs = [[D.score.gradient_boosting.score, "gradient boosting " + D.score.gradient_boosting.score], [D.score.gradient_boosting_neighbours.score, "also other loops " + D.score.gradient_boosting_neighbours.score]];
    var refAttrs = { fill: "var(--ink)", "font-size": 11, "font-weight": 650 };
    var refLabels = refs.map(function (l) { return txt(s, 0, -100, W < 520 ? String(l[0]) : l[1], refAttrs); });
    var refW = refLabels.reduce(function (m, t) { var w = 0; try { w = t.getComputedTextLength(); } catch (e) { /* not rendered */ } return Math.max(m, w); }, 0);
    var padR = Math.max(W < 520 ? 34 : 112, Math.ceil(refW) + 12);
    var bandW = (W - padL - padR) / R;
    var y = function (v) { return top + (1 - (v - lo) / (hi - lo)) * plotH; };
    for (var g = 0; g <= 100; g += 20) {
      svg("line", { x1: padL, x2: W - padR, y1: y(g), y2: y(g), stroke: g === 0 ? "var(--ink-3)" : "#cfd8e1", "stroke-width": 1 }, s);
      txt(s, padL - 8, y(g) + 4, String(g), { "text-anchor": "end", fill: "var(--ink-3)", "font-size": 11 });
    }
    txt(s, 0, top - 14, "score", { fill: "var(--ink-3)", "font-size": 11 });
    for (var r = 1; r <= R; r++) {
      var cx = padL + (r - .5) * bandW;
      txt(s, cx, H - 12, (W < 520 ? "" : "round ") + r, { "text-anchor": "middle", fill: "var(--ink-2)", "font-size": 12 });
      if (r > 1) svg("line", { x1: padL + (r - 1) * bandW, x2: padL + (r - 1) * bandW, y1: top, y2: top + plotH, stroke: "#d6dee6", "stroke-width": 1, "stroke-dasharray": "2 4" }, s);
    }
    refs.forEach(function (l, i) {
      svg("line", { x1: padL, x2: W - padR, y1: y(l[0]), y2: y(l[0]), stroke: "var(--ink)", "stroke-width": 1.25, "stroke-dasharray": i ? "2 3" : "6 3" }, s);
      refLabels[i].setAttribute("x", W - padR + 6);
      refLabels[i].setAttribute("y", y(l[0]) + (i ? -2 : 10));
    });
    // dots: spread within the round band by score order
    var rad = W < 520 ? 4 : 5.5;
    var byRound = {};
    runs.forEach(function (run) { (byRound[run.round] = byRound[run.round] || []).push(run); });
    Object.keys(byRound).forEach(function (rk) {
      var list = byRound[rk].slice().sort(function (a, b) { return a.score - b.score || (a.run < b.run ? -1 : 1); });
      list.forEach(function (run, i) {
        var off = (i - (list.length - 1) / 2) * Math.min(bandW / (list.length + 1), rad * 2.4);
        var cx = padL + (run.round - .5) * bandW + off;
        var cy = y(run.score);
        var g = svg("g", { tabindex: "0", role: "img", "aria-label": run.run + ", round " + run.round + ", " + plainDesign(run.design) + ", score " + run.score }, s);
        if (run.voter) svg("circle", { cx: cx, cy: cy, r: rad + 3.5, fill: "none", stroke: "var(--ink)", "stroke-width": 1.75 }, g);
        dot(g, run.family, cx, cy, rad);
        svg("circle", { cx: cx, cy: cy, r: rad + 6, fill: "transparent" }, g);
        var html = "<b>" + esc(run.run) + "</b><br>" + esc(plainDesign(run.design)) + "<br>score " + run.score + " = " + run.caught + " caught − " + run.false_alarms + " false alarms" + (run.voter ? "<br>one of the seven voters" : "");
        g.addEventListener("pointermove", function (ev) { showTip(html, ev.clientX, ev.clientY); });
        g.addEventListener("pointerleave", hideTip);
        g.addEventListener("focus", function () { var b = g.getBoundingClientRect(); showTip(html, b.right, b.top); });
        g.addEventListener("blur", hideTip);
      });
    });
    // annotation pinned to the first hint runs
    var firstHint = runs.filter(function (r) { return r.family === "hints"; }).reduce(function (m, r) { return Math.min(m, r.round); }, 99);
    if (firstHint < 99) {
      var ax = padL + (firstHint - 1) * bandW + 4;
      var ay = y(96);
      txt(s, ax, ay, "hints arrive", { fill: "var(--ink)", "font-size": 12, "font-weight": 700 });
      svg("path", { d: "M" + (ax + 2) + " " + (ay + 5) + "V" + y(76), stroke: "var(--ink)", "stroke-width": 1.25, fill: "none" }, s);
    }
  }
  function nightTable() {
    var t = $("#night-table");
    var fam = {}; FAMILIES.forEach(function (f) { fam[f[0]] = f[1]; });
    var html = '<thead><tr><th>Run</th><th class="n">Round</th><th>What it tried</th><th>Family</th><th class="n">Caught</th><th class="n">False alarms</th><th class="n">Score</th></tr></thead><tbody>';
    D.night.runs.forEach(function (r) {
      html += "<tr><td>" + esc(r.run) + (r.voter ? " (votes)" : "") + '</td><td class="n">' + r.round + "</td><td>" + esc(plainDesign(r.design)) + "</td><td>" + esc(fam[r.family]) +
        '</td><td class="n">' + r.caught + '</td><td class="n">' + r.false_alarms + '</td><td class="n">' + r.score + "</td></tr>";
    });
    t.innerHTML = html + "</tbody>";
  }

  /* ---------- shared row chart pieces (part 4: test set, Kimi, cost) ---------- */
  var KIND_COLOR = { baseline: "var(--ink)", ours: "var(--model)", floor: "var(--night-range)", big: "var(--ink-2)" };
  function grid(s, x, ticks, y1, y2, H, end, endLabel) {
    ticks.forEach(function (g) {
      var isEnd = g === end;
      svg("line", { x1: x(g), x2: x(g), y1: y1, y2: y2, stroke: isEnd ? "var(--ink)" : "var(--rule)", "stroke-width": 1, "stroke-dasharray": isEnd || g === 0 ? "none" : "2 3" }, s);
      txt(s, x(g), H - 8, isEnd ? endLabel : String(g), { "text-anchor": tickAnchor(x(g), isEnd), fill: isEnd ? "var(--ink)" : "var(--ink-3)", "font-size": 11, "font-weight": isEnd ? 700 : 400 });
    });
  }
  // A tick at the chart's left edge (no label column, at narrow widths) starts there instead of centring on it.
  function tickAnchor(px, isEnd) { return isEnd ? "end" : px < 12 ? "start" : "middle"; }
  function rowHit(s, y, h, W, html) {
    var hit = svg("rect", { x: 0, y: y, width: W, height: h, fill: "transparent" }, s);
    hit.addEventListener("pointermove", function (ev) { showTip(html, ev.clientX, ev.clientY); });
    hit.addEventListener("pointerleave", hideTip);
  }
  function wrap2(str, n) {
    var cut = str.lastIndexOf(" ", n);
    return cut > 0 ? [str.slice(0, cut), str.slice(cut + 1)] : [str];
  }
  function label(s, x, y, main, sub, big) {
    txt(s, x, y, main, { fill: "var(--ink)", "font-size": 13, "font-weight": big ? 750 : 600 });
    if (sub) txt(s, x, y + 15, sub, { fill: "var(--ink-3)", "font-size": 11.5 });
  }

  /* ---------- the test set: score on one 0 to 82 scale ---------- */
  function testChart() {
    var host = $("#test-chart"), T = X.test, P = T.perfect;
    var W = width(host), narrow = W < 680;
    var labW = narrow ? 0 : 330, rowH = narrow ? 64 : 50, top = 18;
    var rows = T.systems.map(function (sy) { return { sy: sy }; })
      .concat(T.single.map(function (g) { return { group: g }; }))
      .concat([{ floor: T.floor }]);
    var H = top + rows.length * rowH + 30;
    host.innerHTML = "";
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, host);
    var plotW = W - labW - (narrow ? 8 : 150);
    var x = function (v) { return labW + (Math.max(0, v) / P) * plotW; };
    grid(s, x, [0, 20, 40, 60, P], top - 8, H - 24, H, P, "perfect " + P);
    var summary = [];
    rows.forEach(function (r, i) {
      var y0 = top + i * rowH, cy = y0 + (narrow ? 42 : rowH / 2);
      if (i === T.systems.length || i === rows.length - 1) svg("line", { x1: 0, x2: W, y1: y0 + 1, y2: y0 + 1, stroke: "var(--rule)", "stroke-width": 1 }, s);
      var tipHtml;
      if (r.sy) {
        var sy = r.sy, ours = sy.kind === "ours", col = KIND_COLOR[sy.kind];
        // The data's sub-lines ("hybrid view + law hint") use terms the write-up no longer explains: name only.
        label(s, 0, narrow ? y0 + 14 : cy + 4, sy.name, null, ours);
        svg("circle", { cx: x(sy.score), cy: cy, r: ours ? 8 : 7, fill: ours ? col : "var(--paper)", stroke: col, "stroke-width": ours ? 2 : 2.5 }, s);
        var t = txt(s, x(sy.score) + 14, cy + 5, String(sy.score), { fill: "var(--ink)", "font-size": ours ? 16 : 15, "font-weight": 800 });
        var t2 = svg("tspan", { fill: "var(--ink-3)", "font-size": 12, "font-weight": 450, dx: 6 }, t);
        t2.textContent = "= " + sy.caught + " − " + sy.false_alarms;
        tipHtml = "<b>" + esc(sy.name) + "</b><br>score " + sy.score + " = " + sy.caught + " caught − " + sy.false_alarms + " false alarms";
        summary.push(sy.name + " " + sy.score);
      } else if (r.group) {
        var g = r.group, lo = g.scores_low, hi = g.scores_high;
        label(s, 0, narrow ? y0 + 14 : cy - 2, g.name, narrow ? null : g.repeats + " repeats", false);
        svg("rect", { x: x(lo), y: cy - 2, width: Math.max(3, x(hi) - x(lo)), height: 4, rx: 2, fill: "var(--model)", opacity: .35 }, s);
        g.scores.forEach(function (d) { svg("circle", { cx: x(d), cy: cy, r: 4.5, fill: "var(--model)", stroke: "var(--paper)", "stroke-width": 1.5, opacity: .85 }, s); });
        svg("line", { x1: x(g.mean), x2: x(g.mean), y1: cy - 10, y2: cy + 10, stroke: "var(--ink)", "stroke-width": 2 }, s);
        txt(s, x(hi) + 12, cy + 4, lo + " to " + hi + ", mean " + Math.round(g.mean), { fill: "var(--ink-2)", "font-size": 12, "font-weight": 650 });
        tipHtml = "<b>" + esc(g.name) + "</b><br>" + g.repeats + " repeats: " + g.scores.join(", ") + "<br>mean " + fmt(g.mean);
        summary.push(g.name + " " + lo + " to " + hi);
      } else {
        var f = r.floor;
        label(s, 0, narrow ? y0 + 14 : cy + 4, f.name, null, false);
        var ax = x(0);
        svg("path", { d: "M" + (ax + 16) + " " + cy + "H" + (ax + 2) + "M" + (ax + 8) + " " + (cy - 5) + "L" + (ax + 2) + " " + cy + "L" + (ax + 8) + " " + (cy + 5), stroke: "var(--night-range)", "stroke-width": 2.5, fill: "none", "stroke-linecap": "round", "stroke-linejoin": "round" }, s);
        var tf = txt(s, ax + 24, cy + 5, String(f.score).replace("-", "−"), { fill: "var(--ink)", "font-size": 15, "font-weight": 800 });
        var tf2 = svg("tspan", { fill: "var(--ink-3)", "font-size": 12, "font-weight": 450, dx: 6 }, tf);
        tf2.textContent = "= " + f.caught + " − " + f.false_alarms + ", far below zero";
        tipHtml = "<b>" + esc(f.name) + "</b><br>score " + f.score + " = " + f.caught + " caught − " + f.false_alarms + " false alarms<br>far below the scale";
        summary.push(f.name + " " + f.score);
      }
      rowHit(s, y0, rowH, W, tipHtml);
    });
    host.setAttribute("aria-label", "Test-set scores, perfect " + P + ": " + summary.join("; ") + ".");
  }

  /* ---------- simple horizontal bars (attacks found, start within 30 s, cost) ---------- */
  function barChart(host, rows, max, opts) {
    var W = width(host), narrow = W < 520;
    var labW = narrow ? 0 : (opts.labW || 230), barH = 16, rowH = narrow ? 46 : (opts.rowH || 34), top = opts.ref ? 26 : 8;
    var H = top + rows.length * rowH + 28;
    host.innerHTML = "";
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, host);
    var plotW = W - labW - (opts.padR || 70);
    var x = function (v) { return labW + (Math.max(0, v) / max) * plotW; };
    grid(s, x, opts.ticks, top - 4, H - 24, H, opts.end, opts.endLabel);
    if (opts.ref) {
      var rx = x(opts.ref.value);
      svg("line", { x1: rx, x2: rx, y1: top - 4, y2: H - 24, stroke: "var(--ink)", "stroke-width": 1.5, "stroke-dasharray": "5 3" }, s);
      txt(s, rx, top - 10, opts.ref.label, { "text-anchor": "middle", fill: "var(--ink)", "font-size": 11.5, "font-weight": 700 });
    }
    rows.forEach(function (r, i) {
      var y0 = top + i * rowH, by = narrow ? y0 + 20 : y0 + (rowH - barH) / 2;
      var halo = { stroke: opts.bg || "var(--paper)", "stroke-width": 4, "paint-order": "stroke", "stroke-linejoin": "round" };
      if (narrow) txt(s, 0, y0 + 13, r.label, Object.assign({ fill: "var(--ink)", "font-size": 12.5, "font-weight": r.bold ? 750 : 600 }, halo));
      else txt(s, 0, by + barH / 2 + 4, r.label, { fill: "var(--ink)", "font-size": 13, "font-weight": r.bold ? 750 : 600 });
      var endX;
      if (r.range) {
        svg("rect", { x: x(r.range[0]), y: by + 4, width: Math.max(3, x(r.range[1]) - x(r.range[0])), height: barH - 8, rx: 2, fill: r.color, opacity: .5 }, s);
        endX = x(r.range[1]);
      } else if (r.value != null && r.value > 0) {
        svg("rect", { x: labW, y: by, width: Math.max(2, x(r.value) - labW), height: barH, rx: 2, fill: r.color }, s);
        endX = x(r.value);
      } else endX = labW;
      txt(s, endX + 6, by + barH / 2 + 4.5, r.text, Object.assign({ fill: "var(--ink)", "font-size": 13, "font-weight": 700 }, halo));
      rowHit(s, y0, rowH, W, r.tip);
    });
  }
  function foundChart() {
    var T = X.test, A = T.rows.attacks;
    var rows = T.systems.map(function (sy) {
      return { label: sy.name, value: sy.found, color: KIND_COLOR[sy.kind], bold: sy.kind === "ours", text: String(sy.found), tip: "<b>" + esc(sy.name) + "</b><br>" + sy.found + " of " + A + " attacks found" };
    }).concat(T.single.map(function (g) {
      return { label: g.name, range: g.found, color: "var(--model)", text: g.found[0] + " to " + g.found[1], tip: "<b>" + esc(g.name) + "</b><br>" + g.found[0] + " to " + g.found[1] + " of " + A + " over " + g.repeats + " repeats" };
    })).concat([{ label: T.floor.name, value: T.floor.found, color: KIND_COLOR.floor, text: String(T.floor.found), tip: "<b>" + esc(T.floor.name) + "</b><br>" + T.floor.found + " of " + A }]);
    barChart($("#found-chart"), rows, A, { ticks: [0, 14, 28, 42, A], end: A, endLabel: "all " + A, labW: 300, padR: 64, bg: "var(--paper)" });
    $("#found-chart").setAttribute("aria-label", "Attacks found, of " + A + ": " + rows.map(function (r) { return r.label + " " + r.text; }).join("; ") + ".");
  }
  function startChart() {
    var T = X.test, A = T.rows.attacks;
    var rows = T.start.map(function (r) {
      return { label: r.name, value: r.within_30, color: "var(--model)", bold: r.key.indexOf("one") !== 0, text: r.within_30 + "  (" + r.median_error + " s)",
        tip: "<b>" + esc(r.name) + "</b><br>start within 30 s for " + r.within_30 + " of " + A + " attacks<br>median error " + r.median_error + " s" };
    }).concat([{ label: "Gradient boosting", value: 0, color: "var(--ink)", text: "gives no start time", tip: "<b>Gradient boosting</b><br>says attack or normal; no start second" }]);
    barChart($("#start-chart"), rows, A, { ticks: [0, 14, 28, 42, A], end: A, endLabel: "all " + A, labW: 300, padR: 120, bg: "var(--paper)",
      ref: { value: T.constant.within_30, label: "always “second " + T.constant.second + "”: " + T.constant.within_30 } });
    $("#start-chart").setAttribute("aria-label", "Start named within 30 seconds, of " + A + ": " + rows.map(function (r) { return r.label + " " + r.text; }).join("; ") + "; always answering second " + T.constant.second + ": " + T.constant.within_30 + ".");
  }

  /* ---------- small against huge: caught and false alarms, same windows ---------- */
  function bigChart() {
    var host = $("#big-chart"), B = X.big_model, A = B.rows.attack, N = B.rows.normal;
    var W = width(host), narrow = W < 680;
    var labW = narrow ? 0 : 300, gap = narrow ? 0 : 28, top = 30;
    var rowH = narrow ? 82 : 46, barH = 14;
    var rows = B.systems;
    var H = top + rows.length * rowH + (narrow ? 40 : 28);
    host.innerHTML = "";
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, host);
    var avail = W - labW - gap - (narrow ? 44 : 90);
    var aW = narrow ? avail : Math.round(avail * .3), nW = narrow ? avail : avail - aW;
    var ax0 = labW, nx0 = narrow ? labW : labW + aW + gap + 44;
    var xa = function (v) { return ax0 + (v / A) * aW; };
    var xn = function (v) { return nx0 + (v / N) * nW; };
    if (!narrow) {
      txt(s, ax0, 14, "Attack windows caught, of " + A, { fill: "var(--ink-2)", "font-size": 12, "font-weight": 700 });
      txt(s, nx0, 14, "Normal windows called attack, of " + N, { fill: "var(--ink-2)", "font-size": 12, "font-weight": 700 });
      [[xa, A, ax0 + aW], [xn, N, nx0 + nW]].forEach(function (p) {
        svg("line", { x1: p[2], x2: p[2], y1: top - 6, y2: H - 24, stroke: "var(--ink)", "stroke-width": 1 }, s);
        txt(s, p[2], H - 8, "all " + p[1], { "text-anchor": "end", fill: "var(--ink)", "font-size": 11, "font-weight": 700 });
      });
      svg("line", { x1: ax0, x2: ax0, y1: top - 6, y2: H - 24, stroke: "var(--rule)" }, s);
      svg("line", { x1: nx0, x2: nx0, y1: top - 6, y2: H - 24, stroke: "var(--rule)" }, s);
      var hx = xn(N / 2);
      svg("line", { x1: hx, x2: hx, y1: top - 6, y2: H - 24, stroke: "var(--rule)", "stroke-dasharray": "2 3" }, s);
      txt(s, hx, H - 8, "half", { "text-anchor": "middle", fill: "var(--ink-3)", "font-size": 11 });
    } else top = 0;
    rows.forEach(function (r, i) {
      var y0 = (narrow ? 4 : top) + i * rowH, big = r.kind === "big", ours = r.kind === "ours";
      if (i && rows[i - 1].kind !== "big" && big) svg("line", { x1: 0, x2: W, y1: y0 - 2, y2: y0 - 2, stroke: "var(--rule)" }, s);
      var by = narrow ? y0 + 22 : y0 + (rowH - barH) / 2;
      var by2 = narrow ? by + barH + 14 : by;
      var nm = !narrow && r.name.length > 40 ? wrap2(r.name, 36) : [r.name];
      nm.forEach(function (line, k) {
        txt(s, 0, narrow ? y0 + 14 : by + barH / 2 + 4 + (k - (nm.length - 1) / 2) * 15, line, { fill: "var(--ink)", "font-size": narrow ? 12.5 : 13, "font-weight": ours ? 750 : 600 });
      });
      if (narrow) {
        svg("rect", { x: ax0, y: by, width: avail, height: barH, fill: "var(--panel)" }, s);
        svg("rect", { x: nx0, y: by2, width: avail, height: barH, fill: "var(--panel)" }, s);
      }
      svg("rect", { x: ax0, y: by, width: Math.max(2, xa(r.caught) - ax0), height: barH, rx: 2, fill: "var(--attack)" }, s);
      txt(s, xa(r.caught) + 6, by + barH / 2 + 4.5, String(r.caught), { fill: "var(--ink)", "font-size": 13, "font-weight": 700 });
      svg("rect", { x: nx0, y: by2, width: Math.max(2, xn(r.false_alarms) - nx0), height: barH, rx: 2, fill: "var(--ink)" }, s);
      txt(s, xn(r.false_alarms) + 6, by2 + barH / 2 + 4.5, String(r.false_alarms), { fill: "var(--ink)", "font-size": 13, "font-weight": 700 });
      rowHit(s, y0, rowH, W, "<b>" + esc(r.name) + "</b><br>" + r.caught + " of " + A + " attack windows caught<br>" + r.false_alarms + " of " + N + " normal windows called attack<br>score on these windows " + r.score);
    });
    if (narrow) {
      txt(s, 0, H - 20, "Top bar: caught, of " + A + " attack windows.", { fill: "var(--ink-3)", "font-size": 11 });
      txt(s, 0, H - 6, "Bottom bar: false alarms, of " + N + " normal windows.", { fill: "var(--ink-3)", "font-size": 11 });
    }
    host.setAttribute("aria-label", "On " + B.rows.total + " test windows, caught of " + A + " and false alarms of " + N + ": " + rows.map(function (r) { return r.name + " " + r.caught + " and " + r.false_alarms; }).join("; ") + ".");
  }
  function costChart() {
    // The thinking side run stays in the data file; the page text no longer covers it, so the chart leaves it out.
    var host = $("#cost-chart"), C = X.big_model.cost.filter(function (c) { return c.key !== "kimi_think"; });
    var max = Math.max.apply(null, C.map(function (c) { return c.usd_per_1000 || 0; }));
    var W = width(host), narrow = W < 680;
    var labW = narrow ? 0 : 300, rowH = narrow ? 80 : 46, barH = 14, top = 6;
    var H = top + C.length * rowH + 26;
    host.innerHTML = "";
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, host);
    var plotW = (W - labW) * (narrow ? .62 : .5);
    var x = function (v) { return labW + (v / max) * plotW; };
    var ticks = max > 3 ? [0, 2, 4, 6] : [0, 0.5, 1, 1.5];
    ticks.forEach(function (g) {
      svg("line", { x1: x(g), x2: x(g), y1: top, y2: H - 24, stroke: g ? "var(--rule-soft)" : "var(--rule)", "stroke-dasharray": g ? "2 3" : "none" }, s);
      txt(s, x(g), H - 8, "$" + (g && max <= 3 ? g.toFixed(2) : g), { "text-anchor": tickAnchor(x(g), false), fill: "var(--ink-3)", "font-size": 11 });
    });
    C.forEach(function (c, i) {
      var y0 = top + i * rowH, by = narrow ? y0 + 20 : y0 + 6;
      txt(s, 0, narrow ? y0 + 13 : by + barH / 2 + 4, c.name, { fill: "var(--ink)", "font-size": narrow ? 12.5 : 13, "font-weight": c.kind === "ours" ? 750 : 600 });
      var endX = labW;
      if (c.usd_per_1000 != null) {
        svg("rect", { x: labW, y: by, width: Math.max(2, x(c.usd_per_1000) - labW), height: barH, rx: 2, fill: KIND_COLOR[c.kind] }, s);
        endX = x(c.usd_per_1000);
        txt(s, endX + 6, by + barH / 2 + 4.5, "$" + c.usd_per_1000.toFixed(2), { fill: "var(--ink)", "font-size": 13, "font-weight": 700 });
      }
      var meta = [c.speed, "data " + c.data].filter(Boolean).join(" · ");
      var my = c.usd_per_1000 == null ? by + barH / 2 + 4 : by + barH + 14;
      var mt = txt(s, labW, my, "", { fill: "var(--ink-3)", "font-size": 11.5 });
      if (c.speed) svg("tspan", {}, mt).textContent = c.speed + (narrow ? "" : " · ");
      var em = svg("tspan", { fill: "var(--ink)", "font-weight": c.data === "leaves the site" ? 750 : 450 }, mt);
      if (narrow && c.speed) { em.setAttribute("x", labW); em.setAttribute("dy", 14); }
      em.textContent = "data " + c.data;
      rowHit(s, y0, rowH, W, "<b>" + esc(c.name) + "</b><br>" + (c.usd_per_1000 != null ? "$" + c.usd_per_1000.toFixed(2) + " per 1,000 windows<br>" : "") + esc(meta));
    });
    host.setAttribute("aria-label", "US dollars per 1,000 windows: " + C.map(function (c) { return c.name + " " + (c.usd_per_1000 != null ? c.usd_per_1000 : "not priced") + ", data " + c.data; }).join("; ") + ".");
  }
  /* ---------- numbers in the static text ----------
     data-k="path" reads demo.json, data-x="path" reads results_extra.json (a path starting with "hero."
     reads the example window of demo.json). Options on the same element, all paths in the same file:
       data-minus="path"  shows the value minus this one (the differences in part 5)
       data-of="path"     shows the value as a whole percentage of this one
       data-round         rounds to a whole number
     The static text keeps the same value as a fallback when a file is missing. */
  function bindNumbers() {
    function bind(n, src, path) {
      var look = function (pth) { return pth.indexOf("hero.") === 0 && src === D ? get(heroWindow(), pth.slice(5)) : get(src, pth); };
      var v = look(path);
      if (v == null || typeof v === "object") return;
      if (n.dataset.minus != null) v = v - look(n.dataset.minus);
      if (n.dataset.of != null) v = Math.round(100 * v / look(n.dataset.of));
      if (n.dataset.round != null) v = Math.round(v);
      if (typeof v !== "number") { n.textContent = String(v); return; }
      n.textContent = /usd/.test(path) ? v.toFixed(2) : fmt(v);
    }
    document.querySelectorAll("[data-k]").forEach(function (n) { bind(n, D, n.dataset.k); });
    if (X) document.querySelectorAll("[data-x]").forEach(function (n) { bind(n, X, n.dataset.x); });
  }

  /* ---------- tables: at narrow widths each row stacks as label and value pairs (css/style.css).
     Each cell gets its column header as data-label, read after the numbers are bound. ---------- */
  function labelTables() {
    document.querySelectorAll("#view-writeup .dtab table").forEach(function (t) {
      var heads = Array.prototype.map.call(t.querySelectorAll("thead th"), function (th) { return th.textContent.trim(); });
      // explicit roles: a table restyled with display:block loses its table semantics in some browsers
      t.setAttribute("role", "table");
      t.parentNode.classList.add("is-labelled");
      t.querySelectorAll("thead, tbody").forEach(function (g) { g.setAttribute("role", "rowgroup"); });
      t.querySelectorAll("tr").forEach(function (r) { r.setAttribute("role", "row"); });
      t.querySelectorAll("thead th").forEach(function (c) { c.setAttribute("role", "columnheader"); });
      t.querySelectorAll("tbody th").forEach(function (c) { c.setAttribute("role", "rowheader"); });
      t.querySelectorAll("td").forEach(function (c) { c.setAttribute("role", "cell"); });
      t.querySelectorAll("tbody tr").forEach(function (tr) {
        Array.prototype.forEach.call(tr.children, function (c, i) {
          if (c.tagName === "TD" && heads[i]) c.setAttribute("data-label", heads[i]);
          if (c.tagName === "TD" && !c.textContent.trim()) c.classList.add("is-empty");
        });
      });
    });
  }

  /* ---------- header: mark the part in view, and keep it in sight in the scrolling strip ---------- */
  function navSpy() {
    if (!("IntersectionObserver" in window)) return;
    var links = {};
    document.querySelectorAll(".bar__nav a").forEach(function (a) { links[a.getAttribute("href").slice(1)] = a; });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting && links[en.target.id]) {
          Object.keys(links).forEach(function (k) { links[k].removeAttribute("aria-current"); });
          var a = links[en.target.id];
          a.setAttribute("aria-current", "true");
          var nav = a.parentNode;
          if (nav.scrollWidth > nav.clientWidth + 1) {
            var left = a.offsetLeft - nav.offsetLeft, right = left + a.offsetWidth;
            if (left < nav.scrollLeft || right > nav.scrollLeft + nav.clientWidth) nav.scrollLeft = Math.max(0, left - 16);
          }
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(links).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  }
  // When the part links do not fit (phones), fade the side that has more links, so the strip reads as scrollable.
  function navStrip() {
    var nav = $(".bar__nav");
    if (!nav) return;
    var mark = function () {
      var more = nav.scrollWidth > nav.clientWidth + 1;
      nav.classList.toggle("is-more-right", more && nav.scrollLeft + nav.clientWidth < nav.scrollWidth - 2);
      nav.classList.toggle("is-more-left", more && nav.scrollLeft > 2);
    };
    nav.addEventListener("scroll", mark, { passive: true });
    window.addEventListener("resize", mark);
    document.addEventListener("hai:viewchange", mark);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(mark);
    mark();
  }

  function drawAll() {
    if (!shown()) return;
    Object.keys(redraws).forEach(function (k) { redraws[k](); });
    document.dispatchEvent(new CustomEvent("hai:writeup-drawn"));
  }

  function fail(msg) {
    var m = el("p", { class: "noscript", text: msg });
    (document.getElementById("view-writeup") || document.querySelector("main")).prepend(m);
  }

  /* The control room (js/console.js) loads its own data; nothing here blocks it. */
  function init() {
    tip = $("#tip");
    labelTables();
    navStrip();
    var main = fetch("data/demo.json").then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); });
    var extra = fetch("data/results_extra.json", { cache: "no-cache" }).then(function (r) { return r.ok ? r.json() : {}; }).catch(function () { return {}; });
    Promise.all([main, extra]).then(function (res) {
      D = res[0];
      X = res[1] || {};
      bindNumbers();
      labelTables();
      renderWindowFigures();
      redraws.baseline = baselineChart;
      redraws.score = scoreChart;
      redraws.night = nightChart;
      redraws.reads = readsChart;
      nightLegend();
      nightTable();
      if (X.test && X.big_model) {
        redraws.test = testChart; redraws.found = foundChart; redraws.start = startChart;
        redraws.big = bigChart; redraws.cost = costChart;
      } else {
        fail("The test-set results could not be loaded (data/results_extra.json). The numbers in the text still hold.");
      }
      navSpy();
      drawAll();
      document.addEventListener("hai:viewchange", function (ev) { if (ev.detail.view === "writeup") drawAll(); });
      // the 64-runs chart sizes its right margin from its label widths: measure again with the page font
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(drawAll);
      var lastW = window.innerWidth, timer;
      window.addEventListener("resize", function () {
        if (Math.abs(window.innerWidth - lastW) < 2) return;
        lastW = window.innerWidth;
        clearTimeout(timer);
        timer = setTimeout(drawAll, 120);
      });
    }).catch(function () {
      fail("The demo data could not be loaded. Serve this folder over HTTP (for example: python3 -m http.server) rather than opening the file directly.");
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
