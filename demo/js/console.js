/* Control room (demo view): the scene replay inside <section id="view-demo">.
   The view shell is js/views.js: read window.HAI.view at start, listen for "hai:viewchange" on
   document, switch views with window.HAI.showView().

   Honest by construction: every verdict, vote, second, sentence and hint shown here is read from
   data/scenes.json, and the test-set line from data/results_extra.json. Nothing here computes a
   verdict; the page only replays recorded answers. Derived phrases ("17 seconds late") are
   computed from those files.

   One timeline per scene ("tau", milliseconds). The watch is 12000 ms at 25 plant seconds per
   second; the speed setting (10x, 25x, 50x; 50x by default) scales the watch only, so at 50x it takes 6 s.
   The later steps always run at their designed pace, one tau millisecond per real millisecond:
     watch    0 .. 12000   the clock runs over the window; the panel stays neutral
     read     .. 17200     the six windows travel into the central control, the models read
     verdict  .. 22800     votes resolve, every answer travels back to its instrument
     check    .. 28400     the plant log and gradient boosting's call
   Everything on screen is a pure function of (scene, tau), so pausing, scrubbing, stepping and
   deep links (?scene=<number>&step=&t=) all land on the same picture. */
(function () {
  "use strict";

  var root = document.getElementById("cr");
  if (!root) return;

  /* ---------- constants (presentation only, no results) ---------- */
  var NS = "http://www.w3.org/2000/svg";
  var W_MS = 12000, R_MS = 5200, V_MS = 5600, C_MS = 5600;
  var PH = { watch: [0, W_MS], read: [W_MS, W_MS + R_MS], verdict: [W_MS + R_MS, W_MS + R_MS + V_MS] };
  PH.check = [PH.verdict[1], PH.verdict[1] + C_MS];
  var END = PH.check[1];
  var STEPS = ["watch", "read", "verdict", "check"];
  var FLOOR = 3;            // a lane is never shorter than 3 x the curve's usual_span_p90
  var FLAT_FLOOR = 0.05;    // a curve that never moves in normal windows: at least 5% of its check-set range
  var CENTRE_S = 30;        // each curve is centred on its first 30 seconds
  var SIDE = { "P1-PC": "l", "P1-LC": "l", "P1-FC": "l", "P1-TC": "r", "P2-SC": "r", "P3-LC": "r" };
  var GAUGE = { "P1-PC": "dial", "P1-LC": "tank", "P1-FC": "pipe", "P1-TC": "column", "P2-SC": "rotor", "P3-LC": "tank" };
  // loops whose (first) target is in the measurement's units, so the gauge can mark it
  // (P1-FC, P2-SC and P3-LC targets are on another scale: no target mark there)
  var TARGET_ON_GAUGE = { "P1-PC": true, "P1-LC": true, "P1-TC": true };
  var SPIN = 17.28;         // rotor degrees per plant second at the window's starting speed (2.4 turns a second at 50x)
  var FLOWPX = 1.2;         // pipe marks: px per plant second at the window's starting flow
  var CD_MS = 8000;         // countdown to the next scene after the check
  var WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  var RM = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : { matches: false };
  var $ = function (id) { return document.getElementById(id); };

  /* ---------- state ---------- */
  var D = null, X = null;           // scenes.json, results_extra.json
  var SEC = 300;                    // window length, replaced by D.seconds
  var scales = {};                  // loop -> "role|name" -> series scale
  var si = 0, sc = null;            // scene index, scene
  var tau = 0, playing = false, speed = 50, lastTs = 0, raf = 0;  // speed: plant seconds per second, watch only
  var played = {};                  // scene key -> true once its check was reached
  var announced = 0;                // 0 none, 1 verdict, 2 check (per scene)
  var L = [];                       // per loop: data, scale, elements, geometry
  var geo = null;                   // wires and flight geometry (null when not laid out)
  var userScrolled = false, lastPhase = "", wasPlaying = false;
  var aFirst = 0, aLast = 0;        // verdict ms when the first / last answer reaches its instrument
  var cards = [], checks = [], bar = null;
  var cd = { on: false, t0: 0, raf: 0 };
  var stepBtns = Array.prototype.slice.call(document.querySelectorAll(".cr-step"));

  /* ---------- small helpers ---------- */
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function easeOut(p) { return 1 - Math.pow(1 - p, 3); }
  function easeIn(p) { return p * p * p; }
  function easeInOut(p) { return p < .5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2; }
  function prog(local, start, dur) { return clamp((local - start) / dur, 0, 1); }
  function el(tag, cls, text) { var n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }
  function sv(tag, attrs) { var n = document.createElementNS(NS, tag); for (var k in attrs) n.setAttribute(k, attrs[k]); return n; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
  function word(n) { return WORDS[n] != null ? WORDS[n] : String(n); }
  function fmtNum(v) {
    var a = Math.abs(v);
    if (a >= 1000) return String(Math.round(v));
    if (a >= 100) return v.toFixed(1);
    if (a >= 10) return v.toFixed(2);
    return v.toFixed(3);
  }
  function fmtWhen(iso) { // "2021-07-15T02:07:20" -> { date: "15 Jul 2021", time: "02:07:20" }, plant local time
    var m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}:\d{2}:\d{2})$/.exec(iso || "");
    if (!m) return { date: iso || "", time: "" };
    return { date: (+m[3]) + " " + MONTHS[+m[2] - 1] + " " + m[1], time: m[4] };
  }
  function seconds(n) { return n === 1 ? "1 second" : n + " seconds"; }
  function lateText(d) { return d > 0 ? seconds(d) + " late" : d < 0 ? seconds(-d) + " early" : "on the second"; }
  function phaseOf(t) { return t <= PH.read[0] ? "watch" : t <= PH.verdict[0] ? "read" : t <= PH.check[0] ? "verdict" : "check"; } // step ends are inclusive
  function stepEnd(step) { return step === "watch" ? PH.watch[1] : step === "read" ? PH.read[1] : step === "verdict" ? PH.verdict[1] : END; }
  function isNarrow() { return window.matchMedia("(max-width: 1099px)").matches; }
  function rot(a, cx, cy) { return "rotate(" + a.toFixed(2) + " " + cx + " " + cy + ")"; }
  function plantSec(tt) { return clamp(tt / W_MS * SEC, 0, SEC); }

  /* reveal: a pure function of progress p (0 hidden, 1 shown); writes only on change */
  function rv(n, p, dy) {
    if (!n) return;
    p = Math.round(p * 1000) / 1000;
    if (n._p === p) return;
    n._p = p;
    n.style.opacity = p;
    n.style.visibility = p <= 0 ? "hidden" : "visible";
    n.style.transform = p >= 1 || !dy ? "" : "translateY(" + ((1 - easeOut(p)) * dy).toFixed(2) + "px)";
  }
  function setText(n, s) { if (n && n._t !== s) { n._t = s; n.textContent = s; } }
  function setHtml(n, s) { if (n && n._h !== s) { n._h = s; n.innerHTML = s; } }
  function setCls(n, c, on) { if (n && n.classList.contains(c) !== !!on) n.classList.toggle(c, !!on); }
  function setStyle(n, k, v) { if (n && n["_s" + k] !== v) { n["_s" + k] = v; n.style[k] = v; } }

  /* ---------- the scale rule (same for every loop and scene) ---------- */
  function laneScale(values, scale) {
    var n = Math.min(CENTRE_S, values.length), c = 0, i, d = 0;
    for (i = 0; i < n; i++) c += values[i];
    c /= n;
    for (i = 0; i < values.length; i++) d = Math.max(d, Math.abs(values[i] - c));
    var floor = 0;
    if (scale) floor = scale.usual_span_p90 > 0 ? FLOOR * scale.usual_span_p90 : FLAT_FLOOR * Math.max(0, (scale.check_max || 0) - (scale.check_min || 0));
    var span = Math.max(2 * d, floor);
    return { c: c, half: span > 0 ? span / 2 : 1 };
  }

  /* hint parsing: the short chip ("33.4×", "no change") and the clause after the series' own name */
  function chipOf(hint) {
    if (/never changes/.test(hint)) return "flat";
    var m = /(\d+(?:\.\d+)?) times its usual one/.exec(hint);
    return m ? m[1] + "×" : "";
  }
  function timesOf(hint) { var m = /(\d+(?:\.\d+)?) times its usual one/.exec(hint); return m ? +m[1] : -1; }
  function clauseOf(hint) { // verbatim, without the trailing colon
    var i = hint.indexOf("; ");
    return (i >= 0 ? hint.slice(i + 2) : hint).replace(/:\s*$/, "");
  }
  // the one hint line shown under the answer: the "opposite way" line if there is one, else the largest step
  function keyHint(l) {
    var best = -1, bv = -2;
    l.hints.forEach(function (h, j) {
      var v = /opposite way/.test(h) ? 1e9 : timesOf(h);
      if (v > bv) { bv = v; best = j; }
    });
    return best;
  }

  /* ---------- per-scene facts, all derived from scenes.json ---------- */
  function rowFacts(l) {
    var truth = l.kind === "attack", said = l.vote === "attack";
    var outcome = truth ? (said ? "caught" : "missed") : (said ? "false alarm" : "right");
    return {
      truth: truth, said: said, outcome: outcome,
      delta: truth && said && l.vote_start != null && l.attack_start != null ? l.vote_start - l.attack_start : null,
      gbSaid: l.gb === "attack"
    };
  }
  function sceneCounts(s) {
    var c = 0, m = 0, f = 0;
    s.loops.forEach(function (l) { var r = rowFacts(l); if (r.outcome === "caught") c++; else if (r.outcome === "missed") m++; else if (r.outcome === "false alarm") f++; });
    return { c: c, m: m, f: f };
  }
  function sceneOutcome(s) {
    var k = sceneCounts(s), parts = [];
    if (k.c) parts.push(k.c + " caught");
    if (k.m) parts.push(k.m + " missed");
    if (k.f) parts.push(k.f === 1 ? "1 false alarm" : k.f + " false alarms");
    return parts.length ? parts.join(", ") : s.loops.length + " normal, right";
  }

  /* ---------- gauges: instruments driven by the real measurement ---------- */
  function tick(cx, cy, r0, r1, a, cls) {
    var t = a * Math.PI / 180, s = Math.sin(t), c = Math.cos(t);
    return { "class": cls, x1: (cx + s * r0).toFixed(2), y1: (cy - c * r0).toFixed(2), x2: (cx + s * r1).toFixed(2), y2: (cy - c * r1).toFixed(2) };
  }
  function arcPath(cx, cy, r, a0, a1, large) {
    var p = function (a) { var t = a * Math.PI / 180; return (cx + Math.sin(t) * r).toFixed(2) + " " + (cy - Math.cos(t) * r).toFixed(2); };
    return "M" + p(a0) + " A" + r + " " + r + " 0 " + (large ? 1 : 0) + " 1 " + p(a1);
  }
  function buildGauge(kind, hasTarget, uid) {
    var g = sv("svg", { "class": "cr-gauge cr-gauge--" + kind, viewBox: "0 0 96 96", "aria-hidden": "true", focusable: "false" });
    var P = {}, k, y;
    var add = function (tag, a, parent) { var n = sv(tag, a); (parent || g).appendChild(n); return n; };
    if (kind === "dial") {
      add("path", { "class": "g-body", d: arcPath(48, 52, 36, -135, 135, true) });
      for (k = 0; k <= 8; k++) {
        var a = -135 + k * 33.75;
        add("line", tick(48, 52, 27, 36, a, "g-tick"));
        if (k < 8) for (var m = 1; m <= 2; m++) add("line", tick(48, 52, 32, 36, a + m * 11.25, "g-thin"));
      }
      if (hasTarget) P.target = add("path", { "class": "g-target-mark", d: "M48 13l-4.5-7.5h9z" });
      P.needle = add("path", { "class": "g-needle", d: "M45.4 52L48 19l2.6 33zM46.4 52L48 61l1.6-9z" });
      add("circle", { "class": "g-hub", cx: 48, cy: 52, r: 4.5 });
    } else if (kind === "tank") {
      var defs = add("defs", {}), cp = sv("clipPath", { id: uid + "-v" });
      cp.appendChild(sv("rect", { x: 21, y: 13, width: 54, height: 70, rx: 5 }));
      defs.appendChild(cp);
      var inner = add("g", { "clip-path": "url(#" + uid + "-v)" });
      P.water = add("path", { "class": "g-water", d: "" }, inner);
      P.surf = add("path", { "class": "g-surf", d: "" }, inner);
      add("rect", { "class": "g-body", x: 20, y: 12, width: 56, height: 72, rx: 6 });
      [16, 32, 48, 64, 80].forEach(function (yy) { add("line", { "class": "g-thin", x1: 11, x2: 16, y1: yy, y2: yy }); });
      if (hasTarget) P.target = add("line", { "class": "g-target", x1: 13, x2: 83, y1: 48, y2: 48 });
    } else if (kind === "rotor") {
      add("circle", { "class": "g-body", cx: 48, cy: 40, r: 31 });
      P.rotor = add("g", {});
      for (k = 0; k < 5; k++) add("path", { "class": "g-blade", d: "M48 40C49 31 55 22 65 17C63 27 57 35 48 40Z", transform: "rotate(" + (k * 72) + " 48 40)" }, P.rotor);
      add("circle", { "class": "g-hub", cx: 48, cy: 40, r: 5 });
      add("path", { "class": "g-thin", d: arcPath(48, 124, 44, -28, 28, false) });
      add("line", tick(48, 124, 41, 48, 0, "g-thin"));
      P.mark = add("path", { "class": "g-mark", d: "M48 79l-4-7h8z" });
    } else if (kind === "pipe") {
      var defs2 = add("defs", {}), cp2 = sv("clipPath", { id: uid + "-p" });
      cp2.appendChild(sv("rect", { x: 8, y: 38, width: 80, height: 20 }));
      defs2.appendChild(cp2);
      var flow = add("g", { "clip-path": "url(#" + uid + "-p)" });
      P.chev = add("g", {}, flow);
      for (k = -20; k <= 100; k += 10) add("path", { "class": "g-chev", d: "M" + k + " 42l5 6-5 6" }, P.chev);
      add("path", { "class": "g-body", d: "M8 37H88M8 59H88" });
      add("rect", { "class": "g-body", x: 3, y: 31, width: 5, height: 34, rx: 1.5 });
      add("rect", { "class": "g-body", x: 88, y: 31, width: 5, height: 34, rx: 1.5 });
      add("line", { "class": "g-thin", x1: 14, x2: 82, y1: 80, y2: 80 });
      add("line", { "class": "g-thin", x1: 48, x2: 48, y1: 76, y2: 84 });
      P.mark = add("path", { "class": "g-mark", d: "M48 79l-4-7h8z" });
    } else { // column: a thermometer
      add("path", { "class": "g-body", d: "M42 73V14a6 6 0 0 1 12 0V73" });
      add("circle", { "class": "g-body", cx: 48, cy: 81, r: 9 });
      add("circle", { "class": "g-liquid", cx: 48, cy: 81, r: 5.5 });
      P.col = add("rect", { "class": "g-liquid", x: 45.5, y: 40, width: 5, height: 40, rx: 2.5 });
      for (y = 16; y <= 64; y += 12) add("line", { "class": "g-thin", x1: 33, x2: 38, y1: y, y2: y });
      if (hasTarget) P.target = add("path", { "class": "g-target-mark", d: "M58 40l8-4.5v9z" });
    }
    return { svg: g, parts: P, kind: kind };
  }
  function at(arr, t) { // linear interpolation in a per-second array
    var i = Math.floor(t), f = t - i;
    if (i <= 0) return arr[0];
    if (i >= arr.length - 1) return arr[arr.length - 1];
    return arr[i] + (arr[i + 1] - arr[i]) * f;
  }
  function updateGauge(o, t) {
    var G = o.gauge, P = G.parts, g = clamp(at(o.gs, t), -1, 1), gt = o.gt ? clamp(at(o.gt, t), -1, 1) : 0;
    var key = Math.round(t * 30);
    if (G._k === key) return;
    G._k = key;
    if (G.kind === "dial") {
      P.needle.setAttribute("transform", rot(g * 135, 48, 52));
      if (P.target) P.target.setAttribute("transform", rot(gt * 135, 48, 52));
    } else if (G.kind === "tank") {
      var lvl = 48 - g * 32;
      // the surface ripples only while the level changes: amplitude follows the rate of change
      var rate = Math.abs(at(o.gs, Math.min(SEC - 1, t + 1)) - at(o.gs, Math.max(0, t - 1))) / 2;
      var amp = Math.min(3.6, rate * 22), ph = t * 0.3, d = "";
      for (var x = 14; x <= 82; x += 4) d += (x === 14 ? "M" : " L") + x + " " + (lvl + Math.sin(ph + x * .2) * amp).toFixed(2);
      P.surf.setAttribute("d", d);
      P.water.setAttribute("d", d + " L82 90 L14 90Z");
      if (P.target) P.target.setAttribute("transform", "translate(0 " + (-gt * 32).toFixed(2) + ")");
    } else if (G.kind === "rotor") {
      P.rotor.setAttribute("transform", rot((at(o.cum, t) * SPIN) % 360, 48, 40));
      P.mark.setAttribute("transform", rot(g * 28, 48, 124));
    } else if (G.kind === "pipe") {
      P.chev.setAttribute("transform", "translate(" + ((at(o.cum, t) * FLOWPX) % 10).toFixed(2) + " 0)");
      P.mark.setAttribute("transform", "translate(" + (g * 34).toFixed(2) + " 0)");
    } else {
      var top = 40 - g * 26;
      P.col.setAttribute("y", top.toFixed(2)); P.col.setAttribute("height", (80 - top).toFixed(2));
      if (P.target) P.target.setAttribute("transform", "translate(0 " + (-gt * 26).toFixed(2) + ")");
    }
  }

  /* ---------- building a scene ---------- */
  function seriesScale(loop, s) { return (scales[loop] || {})[s.role + "|" + s.name] || null; }
  function smooth(a) { // centred moving average over about 2 seconds
    return a.map(function (v, i) { return (a[Math.max(0, i - 1)] + v + a[Math.min(a.length - 1, i + 1)]) / 3; });
  }

  function prepLoops() {
    L = sc.loops.map(function (l, i) {
      var lanes = l.series.map(function (s) {
        var k = laneScale(s.values, seriesScale(l.loop, s));
        return { role: s.role, name: s.name, values: s.values, c: k.c, half: k.half };
      });
      var mi = 0;
      lanes.forEach(function (ln, j) { if (ln.role === "measurement") mi = j; });
      var m = lanes[mi];
      var sm = smooth(m.values);
      var gs = sm.map(function (v) { return (v - m.c) / m.half; });
      var tgt = null;
      if (TARGET_ON_GAUGE[l.loop]) lanes.forEach(function (ln) { if (!tgt && ln.role === "target") tgt = ln; });
      var gt = tgt ? smooth(tgt.values).map(function (v) { return (v - m.c) / m.half; }) : null;
      // rotor angle and pipe marks: the integral of the measured value, relative to its starting level
      var cum = [0];
      for (var k = 1; k < sm.length; k++) cum.push(cum[k - 1] + (m.c ? clamp(sm[k - 1] / m.c, 0, 3) : 1));
      return { l: l, i: i, f: rowFacts(l), lanes: lanes, mi: mi, gs: gs, gt: gt, cum: cum };
    });
    // answers travel back in this order: normal rows first, flagged rows last, so the alarm lands at the end
    var order = L.filter(function (o) { return !o.f.said; }).concat(L.filter(function (o) { return o.f.said; }));
    order.forEach(function (o, k) { o.ps = 450 + k * 90; o.A = o.ps + 800; });
    var fl = L.filter(function (o) { return o.f.said; });
    aLast = L.length ? L[L.length - 1] && Math.max.apply(null, L.map(function (o) { return o.A; })) : 0;
    aFirst = fl.length ? Math.min.apply(null, fl.map(function (o) { return o.A; })) : aLast;
  }

  function buildInstruments() {
    var left = $("cr-side-l"), right = $("cr-side-r");
    left.textContent = ""; right.textContent = "";
    L.forEach(function (o) {
      var l = o.l;
      var a = el("article", "cr-ins");
      var hid = "cr-ins-h" + o.i;
      a.setAttribute("aria-labelledby", hid);
      var gc = el("div", "cr-ins__gauge");
      o.gauge = buildGauge(GAUGE[l.loop] || "dial", !!o.gt, "crg" + o.i);
      gc.appendChild(o.gauge.svg);
      o.val = el("span", "cr-ins__val", fmtNum(o.lanes[o.mi].values[0]));
      o.val.title = "measurement";
      gc.appendChild(o.val);
      a.appendChild(gc);
      var head = el("div", "cr-ins__head");
      var nm = el("h3", "cr-ins__name", cap(l.loop_name));
      nm.id = hid;
      nm.appendChild(document.createTextNode(" "));
      nm.appendChild(el("small", null, l.loop));
      head.appendChild(nm);
      o.badge = el("span", "cr-badge");
      head.appendChild(o.badge);
      a.appendChild(head);
      o.wrap = el("div", "cr-chart__wrap");
      o.chart = sv("svg", { "class": "cr-chart", role: "img", "aria-label": "" });
      o.wrap.appendChild(o.chart);
      a.appendChild(o.wrap);
      o.ring = el("div", "cr-ins__ring");
      o.ring.setAttribute("aria-hidden", "true");
      a.appendChild(o.ring);
      o.ins = a;
      (SIDE[l.loop] === "r" ? right : left).appendChild(a);
    });
  }

  function buildRows() {
    var ol = $("cr-rows");
    ol.textContent = "";
    L.forEach(function (o) {
      var l = o.l, li = el("li", "cr-row");
      o.row = li;
      var th = el("span", "cr-row__thumb");
      th.setAttribute("aria-hidden", "true");
      o.thumb = th;
      li.appendChild(th);
      var nm = el("span", "cr-row__name", cap(l.loop_name));
      nm.setAttribute("aria-hidden", "true");
      li.appendChild(nm);
      var chips = el("span", "cr-chips");
      chips.setAttribute("aria-hidden", "true");
      o.chips = l.hints.map(function (h) {
        var c = el("span", "cr-chip", chipOf(h));
        c.title = (l.series[l.hints.indexOf(h)] || {}).name + ": " + clauseOf(h);
        chips.appendChild(c);
        return c;
      });
      var mk = el("span", "cr-row__marks");
      mk.setAttribute("aria-hidden", "true");
      o.marks = l.voters.map(function (v) {
        var m = el("span", "cr-mk");
        m._yes = v.label === "attack";
        mk.appendChild(m);
        return m;
      });
      li.appendChild(mk);
      li.appendChild(chips);
      var ans = el("span", "cr-row__ans");
      ans.setAttribute("aria-hidden", "true");
      ans.innerHTML = "<b>" + (o.f.said ? "attack" : "normal") + "</b><small>" + l.votes_attack + " of " + l.voters.length + "</small>";
      o.ans = ans;
      li.appendChild(ans);
      var lg = el("span", "cr-row__log" + (o.f.truth ? " is-attack" : ""));
      lg.setAttribute("aria-hidden", "true");
      lg.appendChild(el("b", null, o.f.truth ? "attack " + l.attack_start : "normal"));
      lg.appendChild(el("small", "is-" + o.f.outcome.replace(" ", "-"), o.f.outcome));
      o.log = lg;
      li.appendChild(lg);
      o.rowSr = el("span", "cr-sr");
      li.appendChild(o.rowSr);
      ol.appendChild(li);
    });
  }

  /* a small copy of one loop's measurement lane with the markers, for narrow screens where the
     instrument is out of sight when the answer arrives */
  function miniChart(o) {
    var w = 320, h = 60, m = o.lanes[o.mi], cy = h / 2 + 4, k = (h / 2 - 8) / m.half, d = "";
    for (var i = 0; i < m.values.length; i++) d += (i ? "L" : "M") + (w * i / (SEC - 1)).toFixed(1) + " " + (cy - (m.values[i] - m.c) * k).toFixed(1);
    var s = sv("svg", { "class": "cr-mini", viewBox: "0 0 " + w + " " + h, preserveAspectRatio: "none", "aria-hidden": "true", focusable: "false" });
    s.appendChild(sv("path", { "class": "cr-tr cr-tr--measurement", d: d }));
    var out = { svg: s, log: null };
    var xs = function (sec) { return w * sec / (SEC - 1); };
    if (o.l.vote_start != null && o.f.said) {
      var gm = sv("g", { "class": "cr-mark cr-mark--model" });
      gm.appendChild(sv("line", { x1: xs(o.l.vote_start), x2: xs(o.l.vote_start), y1: 10, y2: h }));
      var tm = sv("text", { x: xs(o.l.vote_start) + 3, y: 9 }); tm.textContent = "model " + o.l.vote_start; gm.appendChild(tm);
      s.appendChild(gm);
    }
    if (o.f.truth && o.l.attack_start != null) {
      var gl = sv("g", { "class": "cr-mark cr-mark--log" });
      gl.appendChild(sv("line", { x1: xs(o.l.attack_start), x2: xs(o.l.attack_start), y1: 10, y2: h }));
      var tl = sv("text", { x: xs(o.l.attack_start) - 3, y: 9, "text-anchor": "end" }); tl.textContent = "log " + o.l.attack_start; gl.appendChild(tl);
      s.appendChild(gl);
      out.log = gl;
    }
    return out;
  }

  /* the verdict bar: pinned above the details, never scrolls */
  function buildBar() {
    var host = $("cr-bar");
    host.textContent = "";
    var flagged = L.filter(function (o) { return o.f.said; }), n = L.length;
    bar = { lines: [], subs: [], log: null, rule: null };
    bar.rule = el("span", "cr-bar__rule");
    host.appendChild(bar.rule);
    if (!flagged.length) {
      var h = el("h3", "cr-bar__h", "");
      h.innerHTML = "<span class=\"cr-bar__who\">Model:</span> " + esc(cap(word(n))) + " loops, " + esc(word(n)) + " normal.";
      host.appendChild(h);
      var maxv = Math.max.apply(null, L.map(function (o) { return o.l.votes_attack; }));
      var s = el("p", "cr-bar__sub", maxv === 0 ? "No model said attack on any loop." : "No loop reached " + D.votes_needed + " of " + D.voters + " attack votes.");
      host.appendChild(s);
      bar.lines.push({ n: h, at: aLast });
      bar.subs.push({ n: s, at: aLast + 250 });
    } else if (flagged.length === 1) {
      var o = flagged[0], h1 = el("h3", "cr-bar__h", "");
      h1.innerHTML = "<span class=\"cr-bar__who\">Model:</span> <span class=\"is-attack\">Attack</span> on " + esc(o.l.loop_name);
      host.appendChild(h1);
      var s1 = el("p", "cr-bar__sub", "");
      s1.innerHTML = "began near second <b>" + o.l.vote_start + "</b> · " + o.l.votes_attack + " of " + o.l.voters.length + " models said attack";
      host.appendChild(s1);
      bar.lines.push({ n: h1, at: o.A });
      bar.subs.push({ n: s1, at: o.A + 250 });
    } else {
      flagged.forEach(function (o) {
        var hh = el("h3", "cr-bar__h cr-bar__h--multi", "");
        hh.innerHTML = "<span class=\"cr-bar__who\">Model:</span> <span class=\"is-attack\">Attack</span> on " + esc(o.l.loop_name) +
          " <span class=\"cr-bar__more\">· second <b>" + o.l.vote_start + "</b> · " + o.l.votes_attack + " of " + o.l.voters.length + "</span>";
        host.appendChild(hh);
        bar.lines.push({ n: hh, at: o.A });
      });
    }
    var logs = el("div", "cr-bar__logs");
    var lw = el("p", "cr-bar__log cr-bar__log--wait", "");
    lw.innerHTML = "<span class=\"cr-bar__who\">Plant log:</span> checked in the next step.";
    var lg = el("p", "cr-bar__log", "");
    lg.innerHTML = "<span class=\"cr-bar__who\">Plant log:</span> " + logLine();
    logs.appendChild(lw); logs.appendChild(lg);
    host.appendChild(logs);
    bar.log = lg; bar.logWait = lw;
  }

  /* the plant log, in one line (it may wrap): what really happened on the loops that matter */
  function logLine() {
    var caught = L.filter(function (o) { return o.f.outcome === "caught"; });
    var missed = L.filter(function (o) { return o.f.outcome === "missed"; });
    var fa = L.filter(function (o) { return o.f.outcome === "false alarm"; });
    var many = caught.length + missed.length + fa.length > 1;
    var parts = [];
    if (caught.length === 1 && !many) {
      var o = caught[0];
      parts.push("attack from second " + o.l.attack_start + ". <b>Caught</b>, " + lateText(o.f.delta) + ".");
    } else if (caught.length > 1 && !missed.length && !fa.length && caught.every(function (c) { return c.l.attack_start === caught[0].l.attack_start; })) {
      parts.push("attack from second " + caught[0].l.attack_start + " on " + (caught.length === 2 ? "both" : "all " + word(caught.length)) + ". <b>" + (caught.length === 2 ? "Both" : "All") + " caught</b>: " +
        caught.map(function (c) { return esc(c.l.loop_name) + " " + lateText(c.f.delta); }).join(", ") + ".");
    } else {
      caught.forEach(function (c) { parts.push("<b>Caught</b> " + esc(c.l.loop_name) + ": attack from second " + c.l.attack_start + ", " + lateText(c.f.delta) + "."); });
    }
    missed.forEach(function (m) {
      parts.push("<b class=\"is-attack\">Missed</b> " + esc(m.l.loop_name) + ": attack from second " + m.l.attack_start + ".");
    });
    fa.forEach(function (f) {
      parts.push("<b>False alarm:</b> nothing was attacked" + (many || L.some(function (x) { return x.f.truth; }) ? " on " + esc(f.l.loop_name) : "") + ".");
    });
    if (!parts.length) parts.push("<b>Right.</b> Nothing was attacked.");
    return parts.join(" ");
  }

  /* the details under the bar: the sentence, one hint line, and the rest on request */
  function buildAnswer() {
    var box = $("cr-ans");
    box.textContent = "";
    cards = [];
    var flagged = L.filter(function (o) { return o.f.said; });
    if (!flagged.length) {
      var p = el("p", "cr-ans__none", "The " + word(D.voters) + " models read all " + word(L.length) + " windows. A loop is flagged when " + D.votes_needed + " of " + D.voters + " say attack.");
      box.appendChild(p);
      cards.push({ parts: [p], at: aLast + 400, offs: [0] });
      return;
    }
    flagged.forEach(function (o) {
      var l = o.l, card = el("article", "cr-card");
      card.setAttribute("aria-label", "Model's answer for " + l.loop_name);
      var mini = miniChart(o);
      card.appendChild(mini.svg);
      o.miniLog = mini.log;
      var said = el("figure", "cr-said");
      said.appendChild(el("figcaption", "cr-said__who", (flagged.length > 1 ? cap(l.loop_name) + ": one" : "One") + " of the " + word(l.voters.length) + " wrote:"));
      var q = el("p", "cr-said__q");
      var full = "“" + l.vote_example.text + "”";
      var ghost = el("span", "cr-said__ghost", full);
      var vis = el("span", "cr-said__typed", "");
      vis.setAttribute("aria-hidden", "true");
      q.appendChild(ghost);
      q.appendChild(vis);
      said.appendChild(q);
      said.appendChild(el("p", "cr-said__caveat", "Sentences were not checked for correctness."));
      card.appendChild(said);
      var kh = keyHint(l);
      var hint = el("p", "cr-hint");
      hint.innerHTML = "<span class=\"cr-hint__h\">One of the feature lines it was given</span> <b>" + esc((l.series[kh] || {}).name || "") + ":</b> " + esc(clauseOf(l.hints[kh] || ""));
      card.appendChild(hint);
      var more = el("button", "cr-more", "All " + word(l.hints.length) + " feature lines");
      more.type = "button";
      more.setAttribute("aria-expanded", "false");
      var dl = el("dl", "cr-why");
      dl.id = "cr-why-" + o.i;
      dl.hidden = true;
      more.setAttribute("aria-controls", dl.id);
      l.hints.forEach(function (h, j) {
        dl.appendChild(el("dt", null, (l.series[j] || {}).name || ""));
        dl.appendChild(el("dd", null, clauseOf(h)));
      });
      more.addEventListener("click", function () {
        var open = more.getAttribute("aria-expanded") !== "true";
        more.setAttribute("aria-expanded", open ? "true" : "false");
        dl.hidden = !open;
        more.textContent = open ? "Hide the feature lines" : "All " + word(l.hints.length) + " feature lines";
      });
      var mr = el("div", "cr-more__wrap");
      mr.appendChild(more);
      mr.appendChild(dl);
      card.appendChild(mr);
      box.appendChild(card);
      cards.push({ parts: [mini.svg, said, hint, mr], at: o.A + 450, offs: [0, 0, 1150, 1300], typed: vis, full: full });
    });
  }

  /* the check details: what the plant log means for the other loops, and gradient boosting's call */
  function checkLines() {
    var lines = [];            // { html, text, mini }
    var caught = [], missed = [], fa = [];
    L.forEach(function (o) {
      if (o.f.outcome === "caught") caught.push(o); else if (o.f.outcome === "missed") missed.push(o); else if (o.f.outcome === "false alarm") fa.push(o);
    });
    var push = function (html, extra) { var t = document.createElement("div"); t.innerHTML = html; lines.push({ html: html, text: t.textContent, mini: extra || null }); };

    missed.forEach(function (o) {
      var h = esc(cap(o.l.loop_name)) + ": " + o.l.votes_attack + " of " + o.l.voters.length + " models said attack.";
      if (o.l.vote_example && o.l.vote_example.text) {
        h += " <span class=\"cr-said cr-said--small\"><span class=\"cr-said__who\">One of the " + word(o.l.voters.length) + " wrote: </span><span class=\"cr-said__q\">“" + esc(o.l.vote_example.text) + "”</span> <span class=\"cr-said__caveat\">Sentences were not checked for correctness.</span></span>";
      }
      push(h, o);
    });
    fa.forEach(function (o) { push(o.l.votes_attack + " of " + o.l.voters.length + " models said attack on " + esc(o.l.loop_name) + "."); });
    var rest = L.filter(function (o) { return o.f.outcome === "right"; });
    if (caught.length || missed.length || fa.length) {
      if (rest.length) push("The other " + word(rest.length) + " loop" + (rest.length > 1 ? "s" : "") + ": normal, and right.");
    } else push(cap(word(L.length)) + " loops, " + word(L.length) + " normal, and that was right.");

    // gradient boosting, plainly, for the same loops
    var gb = [];
    var cAgree = caught.filter(function (o) { return o.f.gbSaid; });
    if (cAgree.length) gb.push("Gradient boosting also said attack on " + names(cAgree) + ". It gives no start second and no sentence.");
    caught.filter(function (o) { return !o.f.gbSaid; }).forEach(function (o) { gb.push("Gradient boosting said normal on " + esc(o.l.loop_name) + "."); });
    missed.forEach(function (o) {
      gb.push(o.f.gbSaid ? "Gradient boosting caught the " + esc(o.l.loop_name) + " attack the model missed." : "Gradient boosting missed it too.");
    });
    fa.forEach(function (o) {
      gb.push(o.f.gbSaid ? "Gradient boosting also said attack on " + esc(o.l.loop_name) + "." : "Gradient boosting said normal on " + esc(o.l.loop_name) + ", and was right.");
    });
    var restWrong = rest.filter(function (o) { return o.f.gbSaid; });
    if (!caught.length && !missed.length && !fa.length) {
      gb.push(restWrong.length ? "Gradient boosting said attack on " + names(restWrong) + "; nothing was attacked there." : "Gradient boosting said normal on all " + word(L.length) + " too.");
    } else restWrong.forEach(function (o) { gb.push("Gradient boosting said attack on " + esc(o.l.loop_name) + "; nothing was attacked there."); });
    gb.forEach(function (g) { push("<span class=\"cr-gb\">" + g + "</span>"); });
    return lines;
  }
  function names(list) {
    var n = list.map(function (o) { return esc(o.l.loop_name); });
    return n.length < 2 ? n.join("") : n.slice(0, -1).join(", ") + " and " + n[n.length - 1];
  }

  function buildCheck() {
    var box = $("cr-check");
    box.textContent = "";
    checks = [];
    checkLines().forEach(function (ln) {
      var p = el("div", "cr-check__p");
      if (ln.mini) { var mc = miniChart(ln.mini); p.appendChild(mc.svg); ln.mini.miniLog2 = mc.log; }
      var tx = el("p", null);
      tx.innerHTML = ln.html;
      p.appendChild(tx);
      p._text = ln.text;
      box.appendChild(p);
      checks.push(p);
    });
  }

  function verdictAnnouncement() {
    var flagged = L.filter(function (o) { return o.f.said; });
    if (!flagged.length) return "No attack flagged. Model answer: " + word(L.length) + " loops, " + word(L.length) + " normal.";
    var s = "Attack flagged. Model answer: " + flagged.map(function (o) {
      return "attack on " + o.l.loop_name + ", " + o.l.votes_attack + " of " + o.l.voters.length + " models, began near second " + o.l.vote_start;
    }).join("; ") + ".";
    var others = L.length - flagged.length;
    if (others) s += " The other " + word(others) + " loops: normal.";
    return s;
  }
  function checkAnnouncement() {
    return "Plant log: " + bar.log.textContent.replace(/^Plant log:\s*/, "") + " " + checks.map(function (p) { return p._text; }).join(" ");
  }

  /* ---------- charts: drawn from the element's size ---------- */
  function drawChart(o) {
    var svg = o.chart, w = Math.max(40, svg.clientWidth || o.wrap.clientWidth), h = Math.max(40, svg.clientHeight || o.wrap.clientHeight);
    o.w = w; o.h = h;
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);
    svg.textContent = "";
    var axisH = 12, top = 2, plotH = h - axisH - top, gap = 4;
    var wts = o.lanes.map(function (ln) { return ln.role === "measurement" ? 1.6 : 1; });
    var tw = wts.reduce(function (a, b) { return a + b; }, 0);
    var avail = plotH - gap * (o.lanes.length - 1);
    var y0 = top;
    var X0 = 0, X1 = w;
    var xs = function (i) { return X0 + (X1 - X0) * i / (SEC - 1); };
    o.x = xs;
    var uid = "cr" + o.i;
    var defs = sv("defs", {});
    var cp = sv("clipPath", { id: uid + "-ink" });
    o.clipInk = sv("rect", { x: 0, y: -4, width: 0, height: h + 8 });
    cp.appendChild(o.clipInk); defs.appendChild(cp);
    var ca = sv("clipPath", { id: uid + "-alarm" });
    var xa = o.l.vote_start != null ? xs(o.l.vote_start) : w;
    o.alarmX = xa;
    o.clipAlarm = sv("rect", { x: xa, y: -4, width: 0, height: h + 8 });
    ca.appendChild(o.clipAlarm); defs.appendChild(ca);
    svg.appendChild(defs);

    var hide = { "aria-hidden": "true" };
    var mids = sv("g", hide), labels = sv("g", hide), ink = sv("g", { "clip-path": "url(#" + uid + "-ink)", "aria-hidden": "true" }), alarm = sv("g", { "class": "cr-alarm", "clip-path": "url(#" + uid + "-alarm)", "aria-hidden": "true" });
    var pens = sv("g", hide);
    o.pens = [];
    var inkMarkup = "";
    o.lanes.forEach(function (ln, j) {
      var lh = avail * wts[j] / tw, cy = y0 + lh / 2, k = (lh / 2 - 1.5) / ln.half;
      ln.y = function (v) { return cy - (v - ln.c) * k; };
      ln.top = y0; ln.h = lh;
      var labelled = ln.role === "measurement" || lh >= 17;
      // the lane's centre line starts after its label, so it does not show through the label's spaces
      mids.appendChild(sv("line", { "class": "cr-lane__mid", x1: labelled && Math.abs(cy - (y0 + 6)) < 6 ? X0 + 8 + ln.name.length * 5.4 : X0, x2: X1, y1: cy, y2: cy }));
      var d = "", vals = ln.values;
      for (var i = 0; i < vals.length; i++) d += (i ? "L" : "M") + xs(i).toFixed(1) + " " + ln.y(vals[i]).toFixed(1);
      var cls = "cr-tr cr-tr--" + ln.role;
      ink.appendChild(sv("path", { "class": cls, d: d }));
      alarm.appendChild(sv("path", { "class": cls, d: d }));
      inkMarkup += "<path class=\"" + cls + "\" d=\"" + d + "\"/>";
      if (labelled) { // a label only where the lane has room for it
        var t = sv("text", { x: 2, y: y0 + 9 });
        t.textContent = ln.name;
        labels.appendChild(t);
      }
      var pen = sv("circle", { "class": "cr-pen", r: ln.role === "measurement" ? 2.4 : 1.8, cx: 0, cy: 0 });
      pens.appendChild(pen);
      o.pens.push(pen);
      y0 += lh + gap;
    });
    o.inkMarkup = inkMarkup;
    svg.appendChild(mids);
    svg.appendChild(ink);
    svg.appendChild(alarm);
    svg.appendChild(labels);
    o.headLine = sv("line", { "class": "cr-head", x1: 0, x2: 0, y1: top, y2: top + plotH, "aria-hidden": "true" });
    svg.appendChild(o.headLine);
    svg.appendChild(pens);

    // markers: the model's start (dashed, model colour) and the plant log's start (solid, attack colour);
    // hidden (not only transparent) until their step, so no text of them is reachable before
    o.mModel = null; o.mLog = null;
    if (o.l.vote_start != null && o.f.said) {
      var gm = sv("g", { "class": "cr-mark cr-mark--model", "aria-hidden": "true" });
      gm.appendChild(sv("line", { x1: xa, x2: xa, y1: top, y2: top + plotH }));
      var early = o.f.truth && o.l.attack_start != null && o.l.vote_start < o.l.attack_start;
      var tm = sv("text", { x: early ? xa - 3 : xa + 3, y: top + 9, "text-anchor": early ? "end" : "start" });
      tm.textContent = "model " + o.l.vote_start;
      gm.appendChild(tm);
      svg.appendChild(gm);
      o.mModel = gm;
    }
    if (o.f.truth && o.l.attack_start != null) {
      var xl = xs(o.l.attack_start);
      var gl = sv("g", { "class": "cr-mark cr-mark--log", "aria-hidden": "true" });
      gl.appendChild(sv("line", { x1: xl, x2: xl, y1: top, y2: top + plotH }));
      var lateM = o.l.vote_start != null && o.f.said && o.l.vote_start < o.l.attack_start;
      var tl = sv("text", { x: lateM ? xl + 3 : xl - 3, y: top + plotH - 3, "text-anchor": lateM ? "start" : "end" });
      tl.textContent = "log " + o.l.attack_start;
      gl.appendChild(tl);
      svg.appendChild(gl);
      o.mLog = gl;
    }
    var ax = sv("g", { "class": "cr-ax", "aria-hidden": "true" });
    [0, Math.round(SEC / 3), Math.round(SEC * 2 / 3), SEC - 1].forEach(function (s) {
      var t = sv("text", { x: xs(s), y: h - 1, "text-anchor": s === 0 ? "start" : s === SEC - 1 ? "end" : "middle" });
      t.textContent = s === SEC - 1 ? SEC + " s" : String(s);
      ax.appendChild(t);
    });
    svg.appendChild(ax);
    [o.mModel, o.mLog].forEach(function (m) { if (m) { m.style.opacity = 0; m.style.visibility = "hidden"; m._p = null; } });
    o._clipW = o._alarmW = o._head = null;
    o._aria = null;
  }
  function chartLabel(o, stage) {
    var s = cap(o.l.loop_name) + ": " + o.lanes.map(function (ln) { return ln.name; }).join(", ") + ", " + SEC + " seconds.";
    if (stage >= 1) s += o.f.said ? " Model: attack from second " + o.l.vote_start + "." : " Model: normal.";
    if (stage >= 2) s += o.f.truth ? " Plant log: attack from second " + o.l.attack_start + " (" + o.f.outcome + ")." : " Plant log: normal (" + o.f.outcome + ").";
    return s;
  }

  function drawThumb(o) {
    var th = o.thumb, w = 52, h = 24, id = "crt" + o.i;
    var s = "<svg viewBox=\"0 0 " + w + " " + h + "\" preserveAspectRatio=\"none\" aria-hidden=\"true\"><defs><clipPath id=\"" + id + "\"><rect x=\"0\" y=\"0\" width=\"0\" height=\"" + h + "\"/></clipPath></defs><g clip-path=\"url(#" + id + ")\">";
    var n = o.lanes.length, lh = h / n;
    o.lanes.forEach(function (ln, j) {
      var d = "", cy = lh * j + lh / 2, k = (lh / 2 - .8) / ln.half;
      for (var i = 0; i < ln.values.length; i += 3) d += (i ? "L" : "M") + (w * i / (SEC - 1)).toFixed(1) + " " + (cy - (ln.values[i] - ln.c) * k).toFixed(1);
      s += "<path class=\"cr-tr cr-tr--" + ln.role + "\" d=\"" + d + "\"/>";
    });
    th.innerHTML = s + "</g></svg>";
    o.thumbSvg = th.firstChild;
    o.thumbClip = th.querySelector("clipPath rect");
    o.thumbSvg._p = null; o.thumbClip._w = null;
  }

  /* ---------- geometry for wires, flights and the answers' way back ---------- */
  function layoutGeo() {
    var floor = $("cr-floor"), fr = floor.getBoundingClientRect();
    if (!fr.width) { geo = null; return; }
    var wires = $("cr-wires"), fly = $("cr-fly");
    wires.textContent = ""; fly.textContent = "";
    wires.setAttribute("viewBox", "0 0 " + fr.width + " " + fr.height);
    var ctl = $("cr-ctl").getBoundingClientRect();
    var side = !isNarrow();
    geo = { side: side };
    var rel = function (r) { return { x: r.left - fr.left, y: r.top - fr.top, w: r.width, h: r.height }; };
    var open = L.map(function (o) { return { T: rel(o.thumb.getBoundingClientRect()), R: rel(o.row.getBoundingClientRect()) }; });
    var comp = L.map(function (o) { return rel(o.ans.getBoundingClientRect()); });
    var C = rel(ctl);
    L.forEach(function (o, i) {
      var S = rel(o.wrap.getBoundingClientRect()), T = open[i].T, Rw = open[i].R;
      if (!T.w) T = { x: Rw.x, y: Rw.y, w: 52, h: Math.max(20, Rw.h) };
      o.g = { S: S, T: T };
      o.insW = o.ins.offsetWidth; o.insH = o.ins.offsetHeight;
      // the flying card: a copy of the window's traces
      var f = el("div", "cr-flyer");
      f.style.width = S.w + "px"; f.style.height = S.h + "px";
      f.style.left = S.x + "px"; f.style.top = S.y + "px";
      f.innerHTML = "<svg viewBox=\"0 0 " + o.w + " " + o.h + "\" preserveAspectRatio=\"none\" aria-hidden=\"true\">" + o.inkMarkup + "</svg>";
      fly.appendChild(f);
      o.flyer = f; f._p = null;
      // flight path: a cubic Bezier from the window's centre into its row, lifted into an arc
      var P0 = { x: S.x + S.w / 2, y: S.y + S.h / 2 }, P3 = { x: T.x + T.w / 2, y: T.y + T.h / 2 };
      var dx = P3.x - P0.x, lift = Math.max(48, Math.abs(P3.y - P0.y) * .35);
      o.bz = [P0, { x: P0.x + dx * .35, y: Math.min(P0.y, P3.y) - lift }, { x: P3.x - dx * .25, y: Math.min(P0.y, P3.y) - lift * .6 }, P3];
      // the answer's way back: from the compact row's answer cell to the instrument's badge
      var B = rel(o.badge.getBoundingClientRect()), Aa = comp[i];
      var Q0 = { x: Aa.x + Math.min(Aa.w, 64) / 2, y: Aa.y + Aa.h / 2 }, Q2 = { x: B.x + B.w / 2, y: B.y + B.h / 2 };
      o.qz = [Q0, { x: (Q0.x + Q2.x) / 2, y: Math.min(Q0.y, Q2.y) - Math.max(40, Math.abs(Q2.x - Q0.x) * .18) }, Q2];
      var pill = el("div", "cr-pill" + (o.f.said ? " is-attack" : ""), o.f.said ? "attack " + o.l.votes_attack + "/" + o.l.voters.length : "normal");
      pill.style.left = (Q2.x - 32) + "px"; pill.style.top = (Q2.y - 10) + "px";
      fly.appendChild(pill);
      o.pill = pill; pill._p = null;
      // wires only where the control sits beside the instruments
      o.wire = o.flow = null;
      if (side) {
        var fromLeft = SIDE[o.l.loop] === "l";
        var I = rel(o.ins.getBoundingClientRect());
        var x0 = fromLeft ? I.x + I.w + 1 : I.x - 1, y0 = S.y + S.h / 2;
        var x1 = fromLeft ? C.x : C.x + C.w, y1 = Rw.y + Rw.h / 2;
        var mxp = (x0 + x1) / 2;
        var d = "M" + x0.toFixed(1) + " " + y0.toFixed(1) + " C" + mxp.toFixed(1) + " " + y0.toFixed(1) + " " + mxp.toFixed(1) + " " + y1.toFixed(1) + " " + x1.toFixed(1) + " " + y1.toFixed(1);
        o.wire = sv("path", { "class": "cr-wire", d: d });
        o.flow = sv("path", { "class": "cr-wire cr-wire--flow", d: d });
        wires.appendChild(o.wire); wires.appendChild(o.flow);
        o.wireLen = o.wire.getTotalLength();
        o.wire.style.strokeDasharray = o.wireLen; o.wire.style.strokeDashoffset = o.wireLen;
        o.flow.style.opacity = 0; o.wire._v = null; o.flow._v = null;
      }
    });
  }
  function bez(b, t) {
    var u = 1 - t;
    return {
      x: u * u * u * b[0].x + 3 * u * u * t * b[1].x + 3 * u * t * t * b[2].x + t * t * t * b[3].x,
      y: u * u * u * b[0].y + 3 * u * u * t * b[1].y + 3 * u * t * t * b[2].y + t * t * t * b[3].y
    };
  }
  function quad(b, t) {
    var u = 1 - t;
    return { x: u * u * b[0].x + 2 * u * t * b[1].x + t * t * b[2].x, y: u * u * b[0].y + 2 * u * t * b[1].y + t * t * b[2].y };
  }

  /* ---------- render: everything from tau ---------- */
  function render() {
    if (!sc || !L.length || L[0].w == null) return;
    var ph = phaseOf(tau), t = plantSec(tau);
    var rp = tau - PH.read[0], vp = tau - PH.verdict[0], cp = tau - PH.check[0];
    var reduced = RM.matches;
    root.setAttribute("data-phase", ph);
    var stage = cp >= 0 && tau > PH.check[0] ? 2 : vp >= aFirst ? 1 : 0;
    var flagged = L.filter(function (o) { return o.f.said; });
    var tWatch = Math.min(t, SEC - 1);

    // watch: traces, pens, gauges, readouts
    L.forEach(function (o) {
      var xw = o.x(tWatch);
      if (o._clipW !== xw) { o._clipW = xw; o.clipInk.setAttribute("width", Math.max(0, xw + (t > 0 ? 1 : 0)).toFixed(1)); }
      var showHead = t > 0 && t < SEC;
      var hk = showHead ? Math.round(xw * 4) : -1;
      if (o._head !== hk) {
        o._head = hk;
        o.headLine.style.opacity = showHead ? 1 : 0;
        o.headLine.setAttribute("x1", xw.toFixed(1)); o.headLine.setAttribute("x2", xw.toFixed(1));
        o.pens.forEach(function (pen, j) {
          var ln = o.lanes[j];
          pen.style.opacity = showHead ? 1 : 0;
          if (showHead) { pen.setAttribute("cx", xw.toFixed(1)); pen.setAttribute("cy", ln.y(at(ln.values, tWatch)).toFixed(1)); }
        });
      }
      updateGauge(o, tWatch);
      setText(o.val, fmtNum(o.lanes[o.mi].values[Math.min(SEC - 1, Math.floor(t))]));
      // the row's thumbnail writes itself faintly while the clock runs (no colour, no marks)
      var tw = (52 * Math.min(t, SEC) / SEC).toFixed(1);
      if (o.thumbClip._w !== tw) { o.thumbClip._w = tw; o.thumbClip.setAttribute("width", tw); }
      if (o._aria !== stage) { o._aria = stage; o.chart.setAttribute("aria-label", chartLabel(o, stage)); }
    });
    setText($("cr-clock"), (ph === "watch" ? Math.floor(t) : SEC) + " / " + SEC + " s");
    var st = $("cr-state"), alarmOn = flagged.length && vp >= aFirst;
    var stateText = ph === "watch" ? "Watching" : ph === "read" ? "Reading" : vp < aFirst ? "Answering" :
      flagged.length ? "Attack flagged: " + (flagged.length === 1 ? flagged[0].l.loop_name : word(flagged.length) + " loops") : "No attack flagged";
    setText(st, stateText);
    setCls(st, "is-alarm", !!alarmOn);

    // the stage before the answer: waiting, then the pipeline lights up as the models read
    var pre = $("cr-pre");
    rv(pre, 1 - prog(vp, 0, 300));
    rv($("cr-wait-a"), 1 - prog(rp, 0, 300));
    rv($("cr-wait-b"), prog(rp, 150, 300));
    var pb = $("cr-prog");
    setStyle(pb, "transform", "scaleX(" + (Math.min(t, SEC) / SEC).toFixed(4) + ")");
    setText($("cr-prog-t"), Math.floor(Math.min(t, SEC)) + " of " + SEC + " s");
    var lit = [rp >= 200, rp >= 2600, rp >= 4400];
    document.querySelectorAll("#cr-pipe .cr-pipe__box").forEach(function (b, k) { setCls(b, "is-lit", lit[k]); });
    document.querySelectorAll("#cr-pipe .cr-pipe__arrow").forEach(function (b, k) { setCls(b, "is-lit", lit[k + 1]); });

    // read: the floor dims, wires draw, windows fly into the rows, hint chips land, the models read
    setCls($("cr-mx"), "is-reading", rp > 0 && vp <= 0);
    L.forEach(function (o, i) {
      var start = 200 + i * 220, dur = 1100;
      setCls(o.wrap, "is-lifted", rp > 0 && vp < 600 && !reduced);
      if (o.wire) {
        var wd = reduced ? (rp > 0 ? 1 : 0) : easeOut(prog(rp, i * 60, 650));
        var fade = 1 - prog(vp, 0, 400);
        var key = wd.toFixed(3) + fade.toFixed(3);
        if (o.wire._v !== key) {
          o.wire._v = key;
          o.wire.style.strokeDashoffset = (o.wireLen * (1 - wd)).toFixed(1);
          o.wire.style.opacity = fade;
        }
        // dashes flow along the wire only while its window is in flight
        var fon = !reduced && rp > start - 100 && rp < start + dur + 200;
        var fo = fon ? Math.min(1, (rp - start + 100) / 150) * (1 - prog(rp, start + dur, 200)) : 0;
        var fk = fo.toFixed(2) + (fon ? Math.round(rp / 16) : "");
        if (o.flow._v !== fk) { o.flow._v = fk; o.flow.style.opacity = fo; if (fon) o.flow.style.strokeDashoffset = (-rp * .06).toFixed(1); }
      }
      // the window in flight: a card that stays large across the gutter, then is drawn into its row
      var fp = reduced ? (rp > 0 ? 1 : 0) : prog(rp, start, dur);
      var f = o.flyer;
      if (f && o.g) {
        var visible = !reduced && rp > start - 1 && rp < start + dur + 140;
        var mid = .55, sx, sy;
        if (fp < .6) { sx = sy = 1 - (1 - mid) * easeOut(fp / .6); }
        else {
          var q2 = easeIn((fp - .6) / .4);
          sx = mid + (o.g.T.w / o.g.S.w - mid) * q2; sy = mid + (o.g.T.h / o.g.S.h - mid) * q2;
        }
        var pt = bez(o.bz, easeInOut(fp));
        var tx = pt.x - (o.g.S.x + o.g.S.w / 2), ty = pt.y - (o.g.S.y + o.g.S.h / 2);
        var op = visible ? Math.min(1, (rp - start + 1) / 90) * (1 - prog(rp, start + dur, 140)) : 0;
        var fk2 = visible ? tx.toFixed(1) + ty.toFixed(1) + sx.toFixed(3) + sy.toFixed(3) + op.toFixed(2) : "off";
        if (f._p !== fk2) {
          f._p = fk2;
          f.style.visibility = visible ? "visible" : "hidden";
          f.style.opacity = op;
          if (visible) f.style.transform = "translate(" + tx.toFixed(1) + "px," + ty.toFixed(1) + "px) scale(" + sx.toFixed(4) + "," + sy.toFixed(4) + ")";
        }
      }
      // landed: the thumbnail inks in, the hint chips peel off
      var land = reduced ? (rp > 0 ? 1 : 0) : prog(rp, start + dur - 100, 200);
      var lk = land.toFixed(2);
      if (o.thumbSvg._p !== lk) { o.thumbSvg._p = lk; o.thumbSvg.style.setProperty("--land", lk); setCls(o.thumb, "is-landed", land >= 1); }
      // they leave as the answer arrives in the same place
      o.chips.forEach(function (c, j) { rv(c, reduced ? (rp > 0 && vp <= 0 ? 1 : 0) : prog(rp, start + dur + j * 80, 220) * (1 - prog(vp, 0, 200)), 6); });
      // reading sweep across the seven marks, then the votes resolve
      var rowH = 0;
      o.marks.forEach(function (m, c) {
        var hgl = 0;
        if (!reduced && rp > 2500 && rp < R_MS) {
          var centre = 2750 + c * 190 + i * 45;
          hgl = Math.max(0, 1 - Math.abs(rp - centre) / 300) * .85;
        }
        rowH = Math.max(rowH, hgl);
        var hk2 = hgl.toFixed(2);
        if (m._h !== hk2) { m._h = hk2; m.style.setProperty("--h", hk2); }
        var on = vp >= 150 + i * 60 + c * 30;
        setCls(m, "is-on", on);
        setCls(m, "is-yes", on && m._yes);
      });
      var rk = (rowH * .7).toFixed(2);
      if (o.row._h !== rk) { o.row._h = rk; o.row.style.setProperty("--h", rk); }
      rv(o.ans, prog(vp, 300 + i * 60, 200), 2);
      rv(o.log, prog(cp, 350 + i * 90, 260), 2);
      setCls(o.row, "is-flag", vp >= 150 + i * 60 && o.f.said);
      var srT = cap(o.l.loop_name);
      if (vp > 0) srT += ": " + o.l.votes_attack + " of " + o.l.voters.length + " models said attack; answer " + (o.f.said ? "attack" : "normal");
      if (cp > 0) srT += "; plant log " + (o.f.truth ? "attack from second " + o.l.attack_start : "normal") + ", " + o.f.outcome;
      setText(o.rowSr, srT + ".");

      // the answer goes back out: a pill flies from the row to the instrument's badge
      if (o.pill) {
        var pp = prog(vp, o.ps, 800), pv = !reduced && vp > o.ps && vp < o.A + 120;
        var pq = quad(o.qz, easeInOut(pp));
        var pop = pv ? Math.min(1, (vp - o.ps) / 100) * (1 - prog(vp, o.A, 120)) : 0;
        var pk = pv ? (pq.x - o.qz[2].x).toFixed(1) + (pq.y - o.qz[2].y).toFixed(1) + pop.toFixed(2) : "off";
        if (o.pill._p !== pk) {
          o.pill._p = pk;
          o.pill.style.visibility = pv ? "visible" : "hidden";
          o.pill.style.opacity = pop;
          if (pv) o.pill.style.transform = "translate(" + (pq.x - o.qz[2].x).toFixed(1) + "px," + (pq.y - o.qz[2].y).toFixed(1) + "px)";
        }
      }
      var arrived = vp >= o.A;
      var fa = o.f.outcome === "false alarm" && cp >= 600;
      setCls(o.ins, "is-alarm", arrived && o.f.said && !fa);
      setCls(o.ins, "is-fa", fa);
      setCls(o.ins, "is-truth-miss", cp >= 600 && o.f.outcome === "missed");
      // the alarm beat, once: a ring leaves the flagged instrument; the others dim for a moment
      if (o.f.said) {
        var rr = reduced ? 1 : prog(vp, o.A, 600);
        var ron = rr > 0 && rr < 1;
        var rkey = ron ? rr.toFixed(3) : "off";
        if (o.ring._p !== rkey) {
          o.ring._p = rkey;
          o.ring.style.visibility = ron ? "visible" : "hidden";
          if (ron) {
            var e = 10 * easeOut(rr);
            o.ring.style.opacity = (.7 * (1 - rr)).toFixed(3);
            o.ring.style.transform = "scale(" + ((o.insW + 2 * e) / o.insW).toFixed(4) + "," + ((o.insH + 2 * e) / o.insH).toFixed(4) + ")";
          }
        }
        setStyle(o.ins, "opacity", "");
      } else {
        var dim = 0;
        if (!reduced && flagged.length) { var dv = vp - aFirst; dim = dv < 0 ? 0 : dv < 150 ? dv / 150 : dv < 1050 ? 1 : dv < 1250 ? 1 - (dv - 1050) / 200 : 0; }
        setStyle(o.ins, "opacity", dim > 0 ? (1 - .32 * dim).toFixed(3) : "");
      }
      var btxt = "", bcls = "";
      if (cp >= 600 && (o.f.truth || o.f.said)) {
        btxt = o.f.outcome;
        bcls = o.f.outcome === "caught" ? "is-attack" : o.f.outcome === "missed" ? "is-miss" : o.f.outcome === "false alarm" ? "is-fa" : "";
      } else if (arrived) {
        btxt = o.f.said ? "attack · " + o.l.votes_attack + "/" + o.l.voters.length : "normal";
        bcls = o.f.said ? "is-attack" : "";
      }
      if (o.badge._t !== btxt + bcls) {
        o.badge._t = btxt + bcls;
        o.badge.textContent = btxt;
        o.badge.className = "cr-badge" + (btxt ? " is-shown" : "") + (bcls ? " " + bcls : "");
      }
      if (o.mModel) {
        var mp = reduced ? (vp >= o.A ? 1 : 0) : prog(vp, o.A + 50, 450);
        if (o.mModel._p !== mp) { o.mModel._p = mp; o.mModel.style.opacity = mp; o.mModel.style.visibility = mp > 0 ? "visible" : "hidden"; o.mModel.setAttribute("transform", "translate(0 " + ((1 - easeOut(mp)) * -10).toFixed(2) + ")"); }
      }
      if (o.f.said) {
        var ap = reduced ? (vp >= o.A ? 1 : 0) : easeOut(prog(vp, o.A + 200, 1200));
        var aw = (o.w - o.alarmX + 2) * ap;
        if (o._alarmW !== aw) { o._alarmW = aw; o.clipAlarm.setAttribute("width", aw.toFixed(1)); }
      }
      if (o.mLog) {
        var lp = reduced ? (cp > 0 ? 1 : 0) : prog(cp, 150, 500);
        if (o.mLog._p !== lp) { o.mLog._p = lp; o.mLog.style.opacity = lp; o.mLog.style.visibility = lp > 0 ? "visible" : "hidden"; o.mLog.setAttribute("transform", "translate(0 " + ((1 - easeOut(lp)) * 10).toFixed(2) + ")"); }
      }
      [o.miniLog, o.miniLog2].forEach(function (g) { if (g) { var v = cp > 0 ? "visible" : "hidden"; if (g._v !== v) { g._v = v; g.style.visibility = v; } } });
    });

    // the verdict bar: headline(s) when the answer reaches the instrument; the plant log at the check
    var ruleP = reduced ? (vp >= aFirst ? 1 : 0) : prog(vp, aFirst, 300);
    rv(bar.rule, ruleP);
    bar.lines.forEach(function (b) { rv(b.n, reduced ? (vp >= b.at ? 1 : 0) : prog(vp, b.at, 320), 8); });
    bar.subs.forEach(function (b) { rv(b.n, reduced ? (vp >= b.at ? 1 : 0) : prog(vp, b.at, 300), 4); });
    var lp0 = reduced ? (cp > 0 ? 1 : 0) : prog(cp, 350, 300);
    rv(bar.log, lp0, 4);
    rv(bar.logWait, (reduced ? (vp >= aLast + 600 ? 1 : 0) : prog(vp, aLast + 600, 300)) * (1 - (reduced ? lp0 : prog(cp, 0, 300))));

    // the details: the sentence types itself, then one hint line
    cards.forEach(function (c) {
      c.parts.forEach(function (pn, j) {
        var off = c.offs[j] || 0;
        rv(pn, reduced ? (vp >= c.at ? 1 : 0) : prog(vp, c.at + off, 320), 6);
      });
      if (c.typed) {
        var tp = reduced ? (vp >= c.at ? 1 : 0) : prog(vp, c.at + 100, 900);
        setText(c.typed, c.full.slice(0, Math.round(c.full.length * tp)));
      }
    });
    checks.forEach(function (pn, k) { rv(pn, reduced ? (cp > 0 ? 1 : 0) : prog(cp, 900 + k * 500, 350), 6); });

    // the details area is a log: keep its newest shown line in view
    // (but never past the top of the block it belongs to)
    var out = $("cr-out"), lastShown = null, group = null;
    cards.forEach(function (c) { c.parts.forEach(function (pn) { if (pn._p > 0 && pn.offsetHeight) { lastShown = pn; group = c.parts[0].offsetHeight ? c.parts[0] : c.parts[1]; } }); });
    checks.forEach(function (pn) { if (pn._p > 0) { lastShown = pn; group = pn; } }); // the bar pins the verdict, so the check may scroll to its newest line
    var target = lastShown ? Math.max(0, Math.min(lastShown.offsetTop + lastShown.offsetHeight - out.clientHeight + 8, group.offsetTop - 6)) : 0;
    if (out.scrollHeight <= out.clientHeight + 1) target = 0;
    if (target !== out._target) {
      var jump = out._target == null || !playing || reduced;
      out._target = target;
      if (jump) out.scrollTop = target; else out.scrollTo({ top: target, behavior: "smooth" });
    }

    // reduced motion: explicit steps instead of travel
    var rmBox = $("cr-rm");
    if (reduced) {
      var label = tau < PH.verdict[1] ? "Show the model's answer" : tau < END ? "Check against the plant log" : "";
      rmBox.hidden = !label;
      setText($("cr-rm-btn"), label);
    } else rmBox.hidden = true;

    // transport
    var scrub = $("cr-scrub"), sv0 = Math.floor(t);
    if (+scrub.value !== sv0 && document.activeElement !== scrub) scrub.value = sv0;
    if (scrub._p !== sv0) { scrub._p = sv0; scrub.style.setProperty("--p", (sv0 / SEC * 100) + "%"); scrub.setAttribute("aria-valuetext", "second " + sv0 + " of " + SEC); }
    setText($("cr-t"), sv0 + " s");
    stepBtns.forEach(function (b) {
      var stp = b.getAttribute("data-step"), r = PH[stp], p = clamp((tau - r[0]) / (r[1] - r[0]), 0, 1);
      var cur = tau > r[0] && tau <= r[1];
      if (b._p !== p) { b._p = p; b.querySelector("i").style.setProperty("--p", p.toFixed(3)); }
      if (cur) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
      setCls(b, "is-done", tau >= r[1]);
    });
    setCls($("cr-next"), "is-ready", tau >= END - 1800);

    // played: the scene list may now show this scene's outcome
    if (cp >= 900 && !played[sc.key]) { played[sc.key] = true; renderScenes(); }

    // announcements (polite live region); cleared when the reader goes back before them
    var ann = cp >= 1000 ? 2 : vp >= aFirst + 300 ? 1 : 0;
    if (ann > announced) {
      announced = ann;
      $("cr-live").textContent = ann === 1 ? verdictAnnouncement() : checkAnnouncement();
    } else if (ann < announced) {
      announced = ann;
      $("cr-live").textContent = "";
    }

    // narrow screens: bring the central control into view when the reading starts, unless the reader scrolled
    if (ph !== lastPhase) {
      if (ph === "read" && playing && isNarrow() && !userScrolled) {
        var ctl = $("cr-ctl"), r0 = ctl.getBoundingClientRect();
        if (r0.top > window.innerHeight * .5) window.scrollBy({ top: r0.top - 60, behavior: reduced ? "auto" : "smooth" });
      }
      lastPhase = ph;
    }
  }

  /* ---------- the clock ---------- */
  function tickFrame(ts) {
    raf = 0;
    if (!playing) return;
    var dt = lastTs ? Math.min(100, ts - lastTs) : 16;
    lastTs = ts;
    // the speed drives the watch (tau runs at speed / 25 there); the later steps keep their designed pace
    var k = speed / 25;
    if (tau < W_MS && tau + dt * k > W_MS) tau = W_MS + (dt - (W_MS - tau) / k);
    else tau += tau < W_MS ? dt * k : dt;
    tau = Math.min(END, tau);
    render();
    if (tau >= END) { setPlaying(false); startCountdown(); return; }
    raf = requestAnimationFrame(tickFrame);
  }
  function setPlaying(on) {
    on = !!on && (window.HAI ? window.HAI.view !== "writeup" : true);
    stopCountdown();
    if (on && tau >= END) { tau = 0; announced = 0; }
    playing = on;
    setCls(root, "is-playing", on);
    $("cr-play").setAttribute("aria-label", on ? "Pause" : "Play");
    lastTs = 0;
    if (on && !raf) raf = requestAnimationFrame(tickFrame);
    if (!on && raf) { cancelAnimationFrame(raf); raf = 0; }
    render();
  }
  function seek(newTau, keepPlaying) {
    stopCountdown();
    tau = clamp(newTau, 0, END);
    if (!keepPlaying && playing) setPlaying(false);
    lastTs = 0;
    render();
  }

  /* after a scene ends on its own, count down to the next one; any interaction cancels */
  function startCountdown() {
    if (RM.matches || !D || si >= D.scenes.length - 1 || (window.HAI && window.HAI.view === "writeup")) return;
    cd.on = true; cd.t0 = 0;
    setCls($("cr-next"), "is-counting", true);
    var step = function (ts) {
      if (!cd.on) return;
      if (!cd.t0) cd.t0 = ts;
      var p = (ts - cd.t0) / CD_MS;
      $("cr-next").style.setProperty("--cd", Math.min(1, p).toFixed(4));
      if (p >= 1) { stopCountdown(); loadScene(si + 1, { play: true }); return; }
      cd.raf = requestAnimationFrame(step);
    };
    cd.raf = requestAnimationFrame(step);
  }
  function stopCountdown() {
    if (!cd.on) return;
    cd.on = false;
    if (cd.raf) cancelAnimationFrame(cd.raf);
    cd.raf = 0;
    var nx = $("cr-next");
    setCls(nx, "is-counting", false);
    nx.style.setProperty("--cd", "0");
  }

  /* ---------- scenes ---------- */
  function renderScenes() {
    var host = $("cr-scenes");
    if (!host._built) {
      host._built = true;
      host.textContent = "";
      D.scenes.forEach(function (s, i) {
        var b = el("button", "cr-scene");
        b.type = "button";
        b.addEventListener("click", function () { loadScene(i, { play: !RM.matches, user: true }); });
        b.appendChild(el("span", "cr-scene__a"));
        var sub = el("span", "cr-scene__b");
        b.appendChild(sub);
        host.appendChild(b);
      });
    }
    Array.prototype.forEach.call(host.children, function (b, i) {
      var s = D.scenes[i], w = fmtWhen(s.window_start), done = !!played[s.key];
      b.setAttribute("aria-pressed", i === si ? "true" : "false");
      setCls(b, "is-played", done);
      setText(b.children[0], w.date + " " + w.time.slice(0, 5));
      var sub = b.children[1];
      if (done) {
        var k = sceneCounts(s), dots = "";
        if (k.c) dots += "<i class=\"cr-dot cr-dot--caught\"></i>";
        if (k.m) dots += "<i class=\"cr-dot cr-dot--missed\"></i>";
        if (k.f) dots += "<i class=\"cr-dot cr-dot--fa\"></i>";
        if (!dots) dots = "<i class=\"cr-dot cr-dot--quiet\"></i>";
        setHtml(sub, dots + esc(sceneOutcome(s)));
      } else setHtml(sub, "not played yet");
      b.setAttribute("aria-label", "Scene " + (i + 1) + ": " + w.date + ", " + w.time + (done ? ", " + sceneOutcome(s) : ""));
      if (i === si && host.scrollWidth > host.clientWidth && b.scrollIntoView && host._si !== si) {
        host._si = si;
        host.scrollTo({ left: Math.max(0, b.offsetLeft - 16), behavior: "auto" });
      }
    });
  }

  // the URL names the scene by its number only, never by its outcome
  function writeUrl(step) {
    try {
      var q = new URLSearchParams(location.search);
      q.set("scene", String(si + 1));
      if (step) q.set("step", step); else q.delete("step");
      q.delete("t");
      history.replaceState(history.state, "", location.pathname + "?" + q.toString() + location.hash);
    } catch (e) { /* file:// */ }
  }

  function loadScene(i, opt) {
    opt = opt || {};
    stopCountdown();
    si = clamp(i, 0, D.scenes.length - 1);
    sc = D.scenes[si];
    announced = 0; lastPhase = ""; userScrolled = false;
    if (playing) setPlaying(false);
    prepLoops();
    buildInstruments();
    buildRows();
    buildBar();
    buildAnswer();
    buildCheck();
    renderScenes();
    $("cr-next").firstChild.textContent = si < D.scenes.length - 1 ? "Next scene" : "Read the write-up";
    $("cr-live").textContent = "";
    $("cr-out")._target = null;
    layout();
    tau = opt.tau != null ? opt.tau : (RM.matches && !opt.play ? PH.watch[1] : 0);
    if (opt.user) writeUrl(opt.step || null);
    render();
    if (opt.play) setPlaying(true);
  }

  function layout() {
    if (!sc || (window.HAI && window.HAI.view === "writeup")) return;
    L.forEach(function (o) { drawChart(o); drawThumb(o); });
    layoutGeo();
    L.forEach(function (o) { o.marks.forEach(function (m) { m._h = null; }); o.ring._p = null; if (o.gauge) o.gauge._k = null; });
    ["cr-pre", "cr-wait-a", "cr-wait-b"].forEach(function (id) { $(id)._p = null; });
  }

  /* step navigation: playing -> start of the step; paused -> its end state */
  function gotoStep(step, user) {
    var r = PH[step];
    if (playing && !RM.matches) seek(step === "watch" ? 0 : r[0], true);
    else seek(stepEnd(step), false);
    if (user) writeUrl(step);
  }
  function stepBy(dir) {
    var i;
    if (!playing || RM.matches) {
      if (dir > 0) { for (i = 0; i < STEPS.length; i++) if (stepEnd(STEPS[i]) > tau + 1) { gotoStep(STEPS[i], true); return; } }
      else { for (i = STEPS.length - 1; i >= 0; i--) if (stepEnd(STEPS[i]) < tau - 1) { gotoStep(STEPS[i], true); return; } seek(0, false); }
      return;
    }
    var idx = STEPS.indexOf(phaseOf(tau));
    var ni = dir > 0 ? idx + 1 : (tau - PH[STEPS[idx]][0] > 800 ? idx : idx - 1);
    if (ni >= STEPS.length) { seek(END, false); return; }
    seek(PH[STEPS[Math.max(0, ni)]][0], true);
  }

  /* ---------- controls ---------- */
  function wire() {
    $("cr-play").addEventListener("click", function () { setPlaying(!playing); });
    $("cr-replay").addEventListener("click", function () { seek(0, false); announced = 0; if (!RM.matches) setPlaying(true); });
    var scrub = $("cr-scrub");
    scrub.addEventListener("input", function () { seek(+scrub.value / SEC * W_MS, true); });
    stepBtns.forEach(function (b) {
      b.addEventListener("click", function () { gotoStep(b.getAttribute("data-step"), true); });
    });
    document.querySelectorAll(".cr-speed__b").forEach(function (b) {
      b.addEventListener("click", function () {
        speed = +b.getAttribute("data-speed");
        document.querySelectorAll(".cr-speed__b").forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
      });
    });
    $("cr-next").addEventListener("click", function () {
      if (si < D.scenes.length - 1) loadScene(si + 1, { play: !RM.matches, user: true });
      else if (window.HAI && window.HAI.showView) window.HAI.showView("writeup");
    });
    $("cr-rm-btn").addEventListener("click", function () {
      var toCheck = tau >= PH.verdict[1];
      gotoStep(toCheck ? "check" : "verdict", true);
      if (isNarrow()) $("cr-ctl").scrollIntoView({ block: "start", behavior: "auto" });
      if (toCheck) $("cr-next").focus(); // the button leaves with the last step; keep focus in the deck
    });

    // any interaction cancels the countdown to the next scene
    var cancel = function (e) { if (cd.on && !(e.target && e.target.closest && e.target.closest("#cr-next"))) stopCountdown(); };
    ["pointerdown", "keydown", "wheel", "touchstart"].forEach(function (ev) { document.addEventListener(ev, cancel, { capture: true, passive: true }); });

    document.addEventListener("keydown", function (e) {
      if (!window.HAI || window.HAI.view !== "demo" || e.altKey || e.ctrlKey || e.metaKey || !sc) return;
      var tg = e.target, tag = (tg && tg.tagName) || "";
      var native = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (tg && tg.isContentEditable);
      if (e.key === "Escape") {
        var how = document.querySelector("#cr details.cr-how[open]");
        if (how) { how.open = false; how.querySelector("summary").focus(); }
        return;
      }
      if (e.key === " " || e.key === "Spacebar") {
        if (native || tag === "BUTTON" || tag === "A" || tag === "SUMMARY") return;
        e.preventDefault();
        setPlaying(!playing);
      } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        if (native) return;
        e.preventDefault();
        var dir = e.key === "ArrowRight" ? 1 : -1;
        if (tau < W_MS || (tau === W_MS && dir < 0)) {
          var t = clamp(Math.round(tau / W_MS * SEC) + dir * (e.shiftKey ? 30 : 10), 0, SEC);
          seek(t / SEC * W_MS, playing);
        } else stepBy(dir);
      } else if (e.key === "Home" && !native) {
        e.preventDefault(); seek(0, playing);
      }
    });
    document.addEventListener("click", function (e) {
      var how = document.querySelector("#cr details.cr-how[open]");
      if (how && !how.contains(e.target)) how.open = false;
    });

    var mark = function () { userScrolled = true; };
    window.addEventListener("wheel", mark, { passive: true });
    window.addEventListener("touchmove", mark, { passive: true });

    document.addEventListener("hai:viewchange", function (e) {
      if (e.detail.view === "writeup") { wasPlaying = playing; stopCountdown(); if (playing) setPlaying(false); }
      else if (sc) {
        // at once, not on the next frame: the first frame of the switch shows the room drawn
        layout(); render();
        if (wasPlaying) { wasPlaying = false; setPlaying(true); }
      }
    });
    document.addEventListener("visibilitychange", function () { lastTs = 0; });

    var pending = 0;
    var onResize = function () {
      if (pending) return;
      pending = requestAnimationFrame(function () { pending = 0; layout(); render(); });
    };
    if (window.ResizeObserver) new ResizeObserver(onResize).observe($("cr-floor"));
    else window.addEventListener("resize", onResize);
    if (RM.addEventListener) RM.addEventListener("change", function () { if (RM.matches && playing) setPlaying(false); render(); });
  }

  /* ---------- static text bound to the data ---------- */
  function bindHonest() {
    var h = $("cr-honest");
    var sys = {};
    ((X && X.test && X.test.systems) || []).forEach(function (s) { sys[s.key] = s; });
    if (sys.gb && sys.vote3 && sys.vote7 && X.test.perfect != null) {
      h.innerHTML = "Test set, run once: gradient boosting <b>" + sys.gb.score + "</b>, our three-model vote <b>" + sys.vote3.score + "</b>, this seven-model vote <b>" + sys.vote7.score +
        "</b>, of a perfect <b>" + X.test.perfect + "</b>. <a href=\"#test\">The full result in the write-up</a>";
    } else {
      h.innerHTML = "The test-set result could not be loaded. <a href=\"#test\">Read it in the write-up</a>";
    }
  }
  function bindText() {
    SEC = D.seconds || 300;
    var n = D.scenes.length;
    document.querySelectorAll("#cr [data-cr]").forEach(function (nd) {
      var k = nd.getAttribute("data-cr");
      if (k === "seconds") nd.textContent = SEC;
      else if (k === "voters") nd.textContent = cap(word(D.voters));
      else if (k === "voters-l") nd.textContent = word(D.voters);
      else if (k === "loops") nd.textContent = word((D.loops || []).length);
      else if (k === "scenes") nd.textContent = cap(word(n));
    });
    $("cr-scenes").setAttribute("aria-label", "Scenes: " + word(n) + " recorded moments of the plant");
    $("cr-scrub").max = SEC;
    $("cr-rule").textContent = D.votes_needed + " of " + D.voters + " make an attack call.";
    $("cr-how-scale").textContent = "Each curve is centred on its first " + CENTRE_S + " seconds. Its lane fits the whole " + SEC + "-second window and is never shorter than " + FLOOR +
      " times that curve's usual movement in this loop (the span that 9 in 10 normal check-set windows stay within; for a curve that never moves in normal windows, a twentieth of its check-set range). So noise stays small and a real swing fills the lane. The same rule holds for every loop and scene.";
    var cs = D.check_set;
    if (cs && cs.vote && cs.gradient_boosting) {
      $("cr-checkset").innerHTML = "The " + word(n) + " scenes were picked to show each kind of outcome, not at random. All " + cs.window_times + " check-set moments: the vote caught <b>" + cs.vote.caught + "</b> of <b>" + cs.attack_rows + "</b> attacked loops with <b>" +
        cs.vote.false_alarms + "</b> false alarms; gradient boosting caught <b>" + cs.gradient_boosting.caught + "</b> with <b>" + cs.gradient_boosting.false_alarms + "</b>. <a href=\"#score\">Check-set scores</a>";
    }
  }

  /* ---------- start ---------- */
  function fail(msg) {
    root.classList.add("is-failed");
    var p = el("p", "cr-fail");
    p.setAttribute("role", "alert");
    p.innerHTML = esc(msg) + " <a href=\"#results\">Read the results in the write-up</a>";
    var ctl = $("cr-ctl");
    ctl.appendChild(p);
    document.querySelectorAll("#cr-deck button, #cr-deck input").forEach(function (b) { b.disabled = true; });
  }

  function start() {
    var b = fetch("data/results_extra.json").then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; });
    b.then(function (x) { X = x; bindHonest(); });
    fetch("data/scenes.json").then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); }).then(function (d) {
      D = d;
      D.loops.forEach(function (lp) {
        scales[lp.loop] = {};
        lp.series.forEach(function (s) { scales[lp.loop][s.role + "|" + s.name] = s; });
      });
      bindText();
      wire();
      var q = new URLSearchParams(location.search);
      var qs = q.get("scene"), idx = -1;
      if (qs != null) {
        if (/^\d+$/.test(qs) && +qs >= 1 && +qs <= D.scenes.length) idx = +qs - 1;
        else D.scenes.forEach(function (s, i) { if (s.key === qs) idx = i; }); // older links named the scene
      }
      var badScene = qs != null && idx < 0;
      if (idx < 0) idx = 0;
      var step = q.get("step");
      var opt = { play: false };
      var tq = q.get("t");
      if (step && STEPS.indexOf(step) >= 0) {
        opt.tau = stepEnd(step);
        if (step === "watch" && tq != null && tq !== "" && !isNaN(+tq)) opt.tau = clamp(+tq, 0, SEC) / SEC * W_MS;
      } else if (tq != null && tq !== "" && !isNaN(+tq)) {
        opt.tau = clamp(+tq, 0, SEC) / SEC * W_MS;
      } else {
        opt.play = !RM.matches && q.get("autoplay") !== "0" && (!window.HAI || window.HAI.view === "demo");
      }
      loadScene(idx, opt);
      if (badScene) { try { q.delete("scene"); history.replaceState(history.state, "", location.pathname + (q.toString() ? "?" + q.toString() : "") + location.hash); } catch (e) { /* file:// */ } }
    }).catch(function (e) {
      if (window.console && e && e.message && !/^\d+$/.test(e.message) && !/fetch|network/i.test(e.message)) console.error(e);
      fail("The scenes could not be loaded (data/scenes.json), so the replay cannot run.");
    });
  }

  start();
})();
