/* HAI attack reader demo. Plain JavaScript, no build step. Data: data/demo.json (check set), data/results_extra.json (test set, huge model). */
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var D = null;          // demo.json
  var state = { id: null, variant: "machine" };
  var redraws = {};      // charts that redraw on resize

  var GROUPS = [
    ["caught", "Attacks caught"],
    ["missed", "Attacks missed"],
    ["false_alarm", "False alarms"],
    ["quiet", "Quiet normal windows"],
    ["neighbour", "Neighbour windows, attack elsewhere"]
  ];
  var VARIANT_NAME = { machine: "normal curve", plain: "normal curve", flat: "blank curve", curveshuffle: "shuffled curve" };
  var FAMILIES = [
    ["hints", "With hints in the text", "circle"],
    ["nohint", "Newer views, no hints", "square"],
    ["zscore", "Spread view, no hints", "diamond"],
    ["range", "Range view (the starting design) and its variants", "triangle"]
  ];

  /* ---------- design names in plain words (raw names come from the night's run table) ---------- */
  var VIEW_PLAIN = [
    ["Range view", "Range view"],
    ["Z-score view", "Spread view"],
    ["Hybrid", "Spread view with the size of its biggest change"],
    ["Loop-relative", "Loop view"]
  ];
  var CLAUSE_PLAIN = {
    step: "largest one-second step",
    law: "seconds against the controller",
    plant: "gap between a measurement and what its command predicts",
    size: "size of the change"
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

  /* ---------- marks: shape = truth (circle attack, square normal), fill = vote said attack ---------- */
  function markSVG(group, size) {
    var s = size || 14, h = s / 2;
    var attackTruth = group === "caught" || group === "missed";
    var saidAttack = group === "caught" || group === "false_alarm";
    var col = attackTruth ? "var(--attack)" : "var(--mk-ink, var(--ink))";
    var fill = saidAttack ? col : "none";
    var shape = attackTruth
      ? '<circle cx="' + h + '" cy="' + h + '" r="' + (h - 1.5) + '" fill="' + fill + '" stroke="' + col + '" stroke-width="2"/>'
      : '<rect x="1.5" y="1.5" width="' + (s - 3) + '" height="' + (s - 3) + '" rx="1" fill="' + fill + '" stroke="' + col + '" stroke-width="2"/>';
    if (group === "neighbour") shape += '<path d="M3.5 ' + (s - 3.5) + 'L' + (s - 3.5) + ' 3.5" stroke="' + col + '" stroke-width="1.6"/>';
    return '<svg viewBox="0 0 ' + s + " " + s + '" aria-hidden="true">' + shape + "</svg>";
  }

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
    var variant = opts.variant || "machine";
    var W = width(host);
    var compact = opts.compact || W < 520;
    var bandH = compact ? 38 : 52, gapH = 24;
    var padR = 6;
    var truth = w.attack_start;
    var ans = opts.answerStart;
    var xs = function (sec) { return (sec / 299) * (W - padR); };
    var both = truth != null && ans != null;
    var top = both ? 58 : 40, bottom = 30;
    var n = w.series.length;
    var H = top + n * (bandH + gapH) + bottom - gapH + 6;
    host.innerHTML = "";
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, host);
    var x = function (sec) { return (sec / 299) * (W - padR); };
    var order = w.curve_check && w.curve_check.order;

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
      var flat = variant === "flat";
      var shown = vals;
      if (variant === "curveshuffle" && order) shown = order.map(function (j) { return vals[j]; });
      var constant = hi - lo < 1e-9;
      var y = function (v) {
        if (flat || constant) return y0 + bandH / 2;
        return y0 + bandH - ((v - lo) / (hi - lo)) * bandH;
      };
      svg("line", { x1: 0, x2: W - padR, y1: y0 + bandH, y2: y0 + bandH, stroke: "var(--rule)", "stroke-width": 1 }, s);
      var label = cap(ser.name);
      if (flat) label += " (blank: all zeros)";
      else if (constant) label += " (never changes)";
      txt(s, 0, y0 - 7, label, { fill: "var(--ink)", "font-size": 12, "font-weight": 600 });
      var d = "";
      for (var k = 0; k < shown.length; k++) d += (k ? "L" : "M") + x(k).toFixed(1) + " " + y(shown[k]).toFixed(1);
      var isTarget = ser.role === "target";
      svg("path", {
        d: d, fill: "none", "stroke-linejoin": "round",
        stroke: flat ? "var(--ink-3)" : isTarget ? "var(--ink-3)" : ser.role === "command" ? "var(--ink-2)" : "var(--ink)",
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
        var v = variant === "flat" ? 0 : (variant === "curveshuffle" && order ? ser.values[order[sec]] : ser.values[sec]);
        html += "<br>" + esc(cap(ser.name)) + ": " + fmt(v);
      });
      if (variant === "flat") html += "<br><i>blank variant: the model saw zeros</i>";
      showTip(html, ev.clientX, ev.clientY);
    }
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerleave", function () { cross.setAttribute("opacity", 0); hideTip(); });
  }

  /* ---------- picker + window panel ---------- */
  function buildPicker() {
    var host = $("#picker");
    host.innerHTML = "";
    GROUPS.forEach(function (g) {
      var items = D.windows.filter(function (w) { return w.group === g[0]; });
      if (!items.length) return;
      var box = el("div", { class: "picker__group" });
      box.appendChild(el("h3", { html: esc(g[1]) + " <span>" + items.length + "</span>" }));
      var list = el("div", { class: "picker__list" });
      items.forEach(function (w) {
        var b = el("button", { type: "button", class: "pick", role: "radio", "aria-checked": "false", "data-id": w.id, tabindex: "-1" });
        b.innerHTML = '<span class="mk">' + markSVG(w.group) + "</span>" +
          '<span class="pick__main"><b>' + esc(w.loop) + "</b> " + esc(w.loop_name) +
          '<span class="pick__sub">vote ' + w.votes_attack + " of 7 said attack</span></span>" +
          (w.curve_check ? '<span class="pick__cc">curve check</span>' : "<span></span>");
        b.addEventListener("click", function () { select(w.id, true); });
        list.appendChild(b);
      });
      box.appendChild(list);
      host.appendChild(box);
    });
    host.addEventListener("keydown", function (ev) {
      if (["ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight", "Home", "End"].indexOf(ev.key) < 0) return;
      var btns = Array.prototype.slice.call(host.querySelectorAll(".pick"));
      var i = btns.findIndex(function (b) { return b.dataset.id === state.id; });
      if (ev.key === "Home") i = 0;
      else if (ev.key === "End") i = btns.length - 1;
      else i = (i + (ev.key === "ArrowDown" || ev.key === "ArrowRight" ? 1 : -1) + btns.length) % btns.length;
      ev.preventDefault();
      select(btns[i].dataset.id, true);
      btns[i].focus();
    });
  }

  function select(id, userAction) {
    var w = D.windows.find(function (x) { return x.id === id; });
    if (!w) return;
    if (state.id !== id) state.variant = "machine";
    state.id = id;
    document.querySelectorAll(".pick").forEach(function (b) {
      var on = b.dataset.id === id;
      b.setAttribute("aria-checked", on ? "true" : "false");
      b.tabIndex = on ? 0 : -1;
    });
    if (userAction) { try { history.replaceState(null, "", "#w=" + encodeURIComponent(id)); } catch (e) { /* ignore */ } }
    renderWindow(w);
  }

  function renderWindow(w) {
    var plantTitle = { boiler: "Boiler", turbine: "Turbine", "water treatment": "Water treatment" }[w.plant] || cap(w.plant);
    $("#win-title").textContent = w.loop + ", " + w.loop_name;
    $("#win-meta").textContent = plantTitle + " plant · HAI dataset release " + w.version + " (recorded " + (w.id.split("/")[2] || "").slice(0, 4) + ") · window " + w.id;

    var right = (w.vote === "attack") === (w.label === "attack");
    var outcomeText = { caught: "Caught", missed: "Missed", false_alarm: "False alarm", quiet: "Correctly quiet", neighbour: "Correctly quiet" }[w.group];
    var oc = $("#win-outcome");
    oc.dataset.good = String(right);
    oc.innerHTML = '<span class="mk">' + markSVG(w.group) + "</span>" + esc(outcomeText);

    // verdict row
    var truthTxt, truthSmall;
    if (w.label === "attack") {
      truthTxt = "Attack";
      truthSmall = w.attack_start != null ? "starts at second " + w.attack_start : "";
      if (w.no_visible_change) truthSmall += (truthSmall ? "; " : "") + "no visible change in these signals";
    } else if (w.kind === "neighbour") {
      truthTxt = "Normal"; truthSmall = "an attack is hitting another loop" + (w.reacting ? "; this loop reacts to it" : "");
    } else { truthTxt = "Normal"; truthSmall = "no attack anywhere in the plant"; }
    var voteSmall = w.votes_attack + " of 7 said attack" + (w.vote_start != null ? "; start near second " + w.vote_start : "");
    var gbSmall = "also reading the other loops: " + w.gb_neighbours + ". It gives no start second.";
    var vd = $("#win-verdicts");
    vd.innerHTML =
      '<div><dt>What was true</dt><dd class="v-' + w.label + '">' + truthTxt + "<small>" + esc(truthSmall) + "</small></dd></div>" +
      '<div><dt>Seven-model vote</dt><dd class="v-' + w.vote + '">' + cap(w.vote) + "<small>" + esc(voteSmall) + "</small></dd></div>" +
      '<div><dt>Gradient boosting</dt><dd class="v-' + w.gb + '">' + cap(w.gb) + "<small>" + esc(gbSmall) + "</small></dd></div>";

    // curve check
    var cc = $("#win-cc");
    if (w.curve_check) {
      cc.hidden = false;
      document.querySelectorAll("#cc-seg button").forEach(function (b) {
        b.setAttribute("aria-checked", b.dataset.v === state.variant ? "true" : "false");
        b.tabIndex = b.dataset.v === state.variant ? 0 : -1;
      });
      var c = w.curve_check;
      var cur = c[state.variant];
      var parts = ["machine", "flat", "curveshuffle"].map(function (v) {
        var on = v === state.variant;
        var s = cap(VARIANT_NAME[v]) + ": " + c[v].label;
        return on ? "<b>" + esc(s) + "</b>" : esc(s);
      }).join(" · ");
      $("#cc-res").innerHTML = "The first of the seven voting models (seed 0) answered this window three times. " + parts +
        '.<br>With the ' + esc(VARIANT_NAME[state.variant]) + ": " + answerHTML(cur.text);
    } else {
      cc.hidden = true;
      state.variant = "machine";
    }

    // chart
    var single = !!w.curve_check;
    var answerStart = single ? w.curve_check[state.variant].start : w.vote_start;
    var drawWin = function () { seriesChart($("#win-chart"), w, { variant: state.variant, answerStart: answerStart }); };
    drawWin();
    redraws.win = drawWin;
    $("#win-chart").setAttribute("aria-label", "Curves of " + w.loop + " over 300 seconds" +
      (w.attack_start != null ? ", attack starting at second " + w.attack_start : "") + ".");
    var key = [];
    if (w.attack_start != null) key.push("<span><i></i>true attack start</span>");
    if (answerStart != null) key.push('<span><i class="k-model"></i>' + (single ? "start that seed 0 gives with the " + VARIANT_NAME[state.variant] : "start the vote gives: the middle value of the starts named by the models that said attack") + "</span>");
    key.push("<span>Hover or touch the curves to read a second.</span>");
    $("#win-key").innerHTML = key.join("");

    // hints and prompt
    hintItems($("#win-hints"), w.hints);
    $("#win-prompt").textContent = w.prompt;

    // voters
    var vo = $("#win-voters");
    vo.innerHTML = "";
    w.voters.forEach(function (v, i) {
      var li = el("li", { class: v.label === "attack" ? "is-attack" : "is-normal", title: v.text });
      li.innerHTML = "seed " + i + "<b>" + (v.label === "attack" ? "attack" : "normal") + "</b>" + (v.start != null ? "at " + v.start : "&nbsp;");
      vo.appendChild(li);
    });

    var seed = w.voters.findIndex(function (v) { return v.run === w.vote_example.run; });
    $("#win-answer").innerHTML = answerHTML(w.vote_example.text) +
      ' <span class="voters__rule">Written by seed ' + seed + ", one of the voters on the winning side.</span>";
    $("#win-ref").innerHTML = answerHTML(w.reference_answer);
  }

  function bindSeg() {
    var seg = $("#cc-seg");
    seg.addEventListener("click", function (ev) {
      var b = ev.target.closest("button");
      if (!b) return;
      state.variant = b.dataset.v;
      select(state.id, false);
    });
    seg.addEventListener("keydown", function (ev) {
      if (["ArrowLeft", "ArrowRight"].indexOf(ev.key) < 0) return;
      var order = ["machine", "flat", "curveshuffle"];
      var i = order.indexOf(state.variant);
      i = (i + (ev.key === "ArrowRight" ? 1 : 2)) % 3;
      state.variant = order[i];
      ev.preventDefault();
      select(state.id, false);
      seg.querySelector('[data-v="' + state.variant + '"]').focus();
    });
  }

  /* ---------- hero ---------- */
  function renderHero() {
    var w = D.windows.find(function (x) { return x.id === D.hero; });
    var draw = function () { seriesChart($("#hero-chart"), w, { variant: "machine", answerStart: w.vote_start, compact: true }); };
    draw();
    redraws.hero = draw;
    hintItems($("#hero-hints"), w.hints);
    $("#hero-answer").innerHTML = answerHTML(w.vote_example.text) +
      ' <span class="voters__rule">' + w.votes_attack + " of 7 models said attack. The truth: attack from second " + w.attack_start + ".</span>";
    $("#hero-id").textContent = w.id;
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
      txt(s, x(g), H - 6, g === max ? max + " = all" : String(g), { "text-anchor": g === max ? "end" : "middle", fill: "var(--ink-3)", "font-size": 11 });
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

  /* ---------- score comparison ---------- */
  function scoreChart() {
    var host = $("#score-chart");
    var W = width(host);
    var sc = D.score, P = sc.perfect;
    var narrow = W < 640;
    var labW = narrow ? 0 : 300, rowH = narrow ? 58 : 44, top = 22;
    var rows = [
      { label: "Start of the night, one model", sub: sc.start_of_night.n + " runs", range: [sc.start_of_night.low, sc.start_of_night.high], color: "var(--night-range)" },
      { label: "Best designs, one model, average of repeats", sub: "6 designs, means", range: [sc.leading_designs.low_mean, sc.leading_designs.high_mean], color: "var(--model)", soft: true },
      { label: "Voting design, one model, each repeat", sub: sc.voting_design_single.n + " repeats", range: [sc.voting_design_single.low, sc.voting_design_single.high], dots: sc.voting_design_single.scores, color: "var(--model)" },
      { label: "Seven models voting", value: sc.vote.score, color: "var(--model)", big: true },
      { label: "Gradient boosting", value: sc.gradient_boosting.score, color: "var(--ink)", hollow: true },
      { label: "Gradient boosting, also reading other loops", value: sc.gradient_boosting_neighbours.score, color: "var(--ink)", hollow: true }
    ];
    var H = top + rows.length * rowH + 30;
    host.innerHTML = "";
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, host);
    var plotW = W - labW - 16;
    var x = function (v) { return labW + (Math.max(0, v) / P) * plotW; };
    [0, 36, 72, 108, P].forEach(function (g) {
      svg("line", { x1: x(g), x2: x(g), y1: top - 10, y2: H - 24, stroke: g === P ? "var(--ink)" : "var(--rule)", "stroke-width": 1, "stroke-dasharray": g === P ? "none" : "2 3" }, s);
      txt(s, x(g), H - 8, g === P ? "perfect " + P : String(g), { "text-anchor": g === P ? "end" : "middle", fill: g === P ? "var(--ink)" : "var(--ink-3)", "font-size": 11, "font-weight": g === P ? 700 : 400 });
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
      } else {
        svg("circle", { cx: x(r.value), cy: cy, r: r.big ? 9 : 7, fill: r.hollow ? "var(--paper)" : r.color, stroke: r.color, "stroke-width": r.hollow ? 2.5 : 2 }, s);
        txt(s, x(r.value) + (r.big ? 15 : 13), cy + 5, String(r.value), { fill: "var(--ink)", "font-size": r.big ? 17 : 14, "font-weight": 800 });
        tipHtml = "<b>" + esc(r.label) + "</b><br>score " + r.value;
      }
      var hit = svg("rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: "transparent" }, s);
      hit.addEventListener("pointermove", function (ev) { showTip(tipHtml, ev.clientX, ev.clientY); });
      hit.addEventListener("pointerleave", hideTip);
    });
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
    return p ? svg("path", { d: p, fill: color, stroke: "var(--band)", "stroke-width": 1.5 }, parent)
      : svg("circle", { cx: cx, cy: cy, r: r + .5, fill: color, stroke: "var(--band)", "stroke-width": 1.5 }, parent);
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
    var padL = 34, padR = W < 520 ? 34 : 112, top = 24, plotH = W < 520 ? 300 : 380, bottom = 34;
    var H = top + plotH + bottom;
    var lo = -10, hi = 100;
    host.innerHTML = "";
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, host);
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
    // gradient boosting reference lines
    [[D.score.gradient_boosting.score, "gradient boosting " + D.score.gradient_boosting.score], [D.score.gradient_boosting_neighbours.score, "also other loops " + D.score.gradient_boosting_neighbours.score]].forEach(function (l, i) {
      svg("line", { x1: padL, x2: W - padR, y1: y(l[0]), y2: y(l[0]), stroke: "var(--ink)", "stroke-width": 1.25, "stroke-dasharray": i ? "2 3" : "6 3" }, s);
      txt(s, W - padR + 6, y(l[0]) + (i ? -2 : 10), W < 520 ? String(l[0]) : l[1], { fill: "var(--ink)", "font-size": 11, "font-weight": 650 });
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

  /* ---------- shared row chart pieces (test set and huge-model sections) ---------- */
  var X = null;          // results_extra.json
  var KIND_COLOR = { baseline: "var(--ink)", ours: "var(--model)", floor: "var(--night-range)", big: "var(--ink-2)" };
  function grid(s, x, ticks, y1, y2, H, end, endLabel) {
    ticks.forEach(function (g) {
      var isEnd = g === end;
      svg("line", { x1: x(g), x2: x(g), y1: y1, y2: y2, stroke: isEnd ? "var(--ink)" : "var(--rule)", "stroke-width": 1, "stroke-dasharray": isEnd || g === 0 ? "none" : "2 3" }, s);
      txt(s, x(g), H - 8, isEnd ? endLabel : String(g), { "text-anchor": isEnd ? "end" : "middle", fill: isEnd ? "var(--ink)" : "var(--ink-3)", "font-size": 11, "font-weight": isEnd ? 700 : 400 });
    });
  }
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
        label(s, 0, narrow ? y0 + 14 : cy - 2, sy.name, narrow ? null : sy.sub, ours);
        svg("circle", { cx: x(sy.score), cy: cy, r: ours ? 8 : 7, fill: ours ? col : "var(--band)", stroke: col, "stroke-width": ours ? 2 : 2.5 }, s);
        var t = txt(s, x(sy.score) + 14, cy + 5, String(sy.score), { fill: "var(--ink)", "font-size": ours ? 16 : 15, "font-weight": 800 });
        var t2 = svg("tspan", { fill: "var(--ink-3)", "font-size": 12, "font-weight": 450, dx: 6 }, t);
        t2.textContent = "= " + sy.caught + " − " + sy.false_alarms;
        tipHtml = "<b>" + esc(sy.name) + "</b><br>score " + sy.score + " = " + sy.caught + " caught − " + sy.false_alarms + " false alarms";
        summary.push(sy.name + " " + sy.score);
      } else if (r.group) {
        var g = r.group, lo = g.scores_low, hi = g.scores_high;
        label(s, 0, narrow ? y0 + 14 : cy - 2, g.name, narrow ? null : g.repeats + " repeats", false);
        svg("rect", { x: x(lo), y: cy - 2, width: Math.max(3, x(hi) - x(lo)), height: 4, rx: 2, fill: "var(--model)", opacity: .35 }, s);
        g.scores.forEach(function (d) { svg("circle", { cx: x(d), cy: cy, r: 4.5, fill: "var(--model)", stroke: "var(--band)", "stroke-width": 1.5, opacity: .85 }, s); });
        svg("line", { x1: x(g.mean), x2: x(g.mean), y1: cy - 10, y2: cy + 10, stroke: "var(--ink)", "stroke-width": 2 }, s);
        txt(s, x(hi) + 12, cy + 4, lo + " to " + hi + ", mean " + fmt(g.mean), { fill: "var(--ink-2)", "font-size": 12, "font-weight": 650 });
        tipHtml = "<b>" + esc(g.name) + "</b><br>" + g.repeats + " repeats: " + g.scores.join(", ") + "<br>mean " + fmt(g.mean);
        summary.push(g.name + " " + lo + " to " + hi);
      } else {
        var f = r.floor;
        label(s, 0, narrow ? y0 + 14 : cy - 2, f.name, narrow ? null : "the floor", false);
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
    barChart($("#found-chart"), rows, A, { ticks: [0, 14, 28, 42, A], end: A, endLabel: "all " + A, labW: 300, padR: 64, bg: "var(--band)" });
    $("#found-chart").setAttribute("aria-label", "Attacks found, of " + A + ": " + rows.map(function (r) { return r.label + " " + r.text; }).join("; ") + ".");
  }
  function startChart() {
    var T = X.test, A = T.rows.attacks;
    var rows = T.start.map(function (r) {
      return { label: r.name, value: r.within_30, color: "var(--model)", bold: r.key.indexOf("one") !== 0, text: r.within_30 + "  (" + r.median_error + " s)",
        tip: "<b>" + esc(r.name) + "</b><br>start within 30 s for " + r.within_30 + " of " + A + " attacks<br>median error " + r.median_error + " s" };
    }).concat([{ label: "Gradient boosting", value: 0, color: "var(--ink)", text: "gives no start time", tip: "<b>Gradient boosting</b><br>says attack or normal; no start second" }]);
    barChart($("#start-chart"), rows, A, { ticks: [0, 14, 28, 42, A], end: A, endLabel: "all " + A, labW: 300, padR: 120, bg: "var(--band)",
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
    var host = $("#cost-chart"), C = X.big_model.cost;
    var max = Math.max.apply(null, C.map(function (c) { return c.usd_per_1000 || 0; }));
    var W = width(host), narrow = W < 680;
    var labW = narrow ? 0 : 300, rowH = narrow ? 80 : 46, barH = 14, top = 6;
    var H = top + C.length * rowH + 26;
    host.innerHTML = "";
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }, host);
    var plotW = (W - labW) * (narrow ? .62 : .5);
    var x = function (v) { return labW + (v / max) * plotW; };
    [0, 2, 4, 6].forEach(function (g) {
      svg("line", { x1: x(g), x2: x(g), y1: top, y2: H - 24, stroke: g ? "var(--rule-soft)" : "var(--rule)", "stroke-dasharray": g ? "2 3" : "none" }, s);
      txt(s, x(g), H - 8, "$" + g, { "text-anchor": "middle", fill: "var(--ink-3)", "font-size": 11 });
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
  function thinkLine() {
    var t = X.big_model.thinking;
    $("#think-line").textContent = "On " + t.rows + " windows (" + t.attack + " attack, " + t.normal + " normal), Kimi cold caught " + t.off.caught + " with " + t.off.false_alarms +
      " false alarms with thinking off, and " + t.on.caught + " with " + t.on.false_alarms + " with thinking on (low), at four times the price. On the same windows gradient boosting caught " +
      t.gb.caught + " with " + t.gb.false_alarms + " false alarms; our votes " + t.vote3.caught + " and " + t.vote7.caught + " with " + t.vote3.false_alarms + ".";
  }

  /* ---------- numbers in the static text ---------- */
  function bindNumbers() {
    document.querySelectorAll("[data-k]").forEach(function (n) {
      var v = get(D, n.dataset.k);
      if (v != null && typeof v !== "object") n.textContent = fmt(v);
    });
    document.querySelectorAll("[data-x]").forEach(function (n) {
      var v = get(X, n.dataset.x);
      if (v == null || typeof v === "object") return;
      n.textContent = typeof v === "number" && /usd/.test(n.dataset.x) ? v.toFixed(2) : fmt(v);
    });
  }

  /* ---------- nav state ---------- */
  function navSpy() {
    if (!("IntersectionObserver" in window)) return;
    var links = {};
    document.querySelectorAll(".bar__nav a").forEach(function (a) { links[a.getAttribute("href").slice(1)] = a; });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting && links[en.target.id]) {
          Object.keys(links).forEach(function (k) { links[k].removeAttribute("aria-current"); });
          links[en.target.id].setAttribute("aria-current", "true");
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(links).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  }

  function drawAll() {
    Object.keys(redraws).forEach(function (k) { redraws[k](); });
  }

  function fail(msg) {
    var m = el("p", { class: "noscript", text: msg });
    document.querySelector("main").prepend(m);
  }

  function init() {
    tip = $("#tip");
    var main = fetch("data/demo.json").then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); });
    var extra = fetch("data/results_extra.json", { cache: "no-cache" }).then(function (r) { return r.ok ? r.json() : {}; }).catch(function () { return {}; });
    Promise.all([main, extra]).then(function (res) {
      D = res[0];
      X = res[1] || {};
      bindNumbers();
      renderHero();
      buildPicker();
      bindSeg();
      var fromHash = decodeURIComponent((location.hash.match(/^#w=(.+)$/) || [])[1] || "");
      var first = D.windows.find(function (w) { return w.id === fromHash; }) || D.windows.find(function (w) { return w.id === D.hero; }) || D.windows[0];
      select(first.id, false);
      if (fromHash) { var t = document.getElementById("try"); if (t) t.scrollIntoView(); }
      redraws.reads = readsChart;
      redraws.score = scoreChart;
      redraws.night = nightChart;
      nightLegend();
      nightTable();
      readsChart(); scoreChart(); nightChart();
      if (X.test && X.big_model) {
        redraws.test = testChart; redraws.found = foundChart; redraws.start = startChart;
        redraws.big = bigChart; redraws.cost = costChart;
        testChart(); foundChart(); startChart(); bigChart(); costChart(); thinkLine();
      } else {
        fail("The test-set results could not be loaded (data/results_extra.json). The headline numbers in the text still hold.");
      }
      navSpy();
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
