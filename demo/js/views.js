/* Two views, one page: the control room ("demo") and the write-up ("writeup").
   Loaded without defer at the end of <body>, so the right view is set before the first paint.

   Interface (window.HAI):
     HAI.view                      current view, "demo" or "writeup"
     HAI.showView(view, hash)      switch view and push a history entry. hash is optional:
                                   "#results" opens the write-up at that section; omitted, the
                                   view opens at its top. Query parameters are kept as they are.
     document event "hai:viewchange"  fired on document after every view change (and once at
                                   load), detail = { view, previous }. Fired after body[data-view]
                                   is set and before any scrolling, so listeners can draw.

   Routing by hash: none or #demo opens the demo; #writeup opens the write-up at the top; any
   other hash naming an element inside the write-up (#top, the parts #data #baseline #training
   #results #conclusions, and #kimi #limits) opens the write-up there; a hash naming an element
   inside the control room opens the demo. Back and forward work through hashchange.

   Old anchors from the earlier write-up keep working (ALIAS): #test opens #results, #big #kimi,
   #reads #limits, #score and #night #training, #conclusion #conclusions; #try (the window
   explorer, now replaced by the control room) opens the demo. The hash itself is left as it is.
   These and #writeup also scroll when their view is already open (the browser's own jump needs an
   element with that id).

   The switch (every trigger: the toggle, links, back and forward, HAI.showView): the lights come on,
   the lights go down. With the View Transitions API, html.is-switching names the header and the two
   views (css/style.css): the ground cross-fades, the outgoing view fades and settles back, the
   incoming one fades in and rises, and the header stays live while its colours and the toggle's thumb
   move by CSS transitions. Without the API the same moves run as classes (html.is-fading, .is-leaving,
   .is-entering) with the swap in between. The first load and reduced motion switch at once.
   The swap itself is always apply(), synchronous: body[data-view], hidden, then "hai:viewchange" (the
   charts draw), then the scroll, so the first frame of the incoming view is complete. A request
   during a running switch ends that switch first; only the latest request is applied (commit()). */
(function () {
  "use strict";

  var body = document.body;
  var root = document.documentElement;
  var VIEWS = { demo: "view-demo", writeup: "view-writeup" };
  var current = null;
  var RM = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : { matches: false };
  var SWITCH_MS = 420, OUT_MS = 200;  // keep in step with --switch and --switch-out in css/style.css
  var CLS = ["is-switching", "is-fading", "is-switching-to-demo", "is-switching-to-writeup"];
  var started = false;  // false until the first route: the first load never animates
  var wish = null;      // the latest request { view, target, force }, applied by commit()
  var sw = null;        // the running switch { id, to, vt, timers }
  var swId = 0;
  var ALIAS = { test: "results", big: "kimi", reads: "limits", score: "training", night: "training", conclusion: "conclusions", try: "demo" };

  function viewOfHash(hash) {
    var h = (hash || "").replace(/^#/, "");
    if (ALIAS.hasOwnProperty(h) && !document.getElementById(h)) h = ALIAS[h];
    if (!h || h === "demo") return { view: "demo", target: null };
    if (h === "writeup") return { view: "writeup", target: null };
    var t = null;
    try { t = document.getElementById(decodeURIComponent(h)); } catch (e) { t = document.getElementById(h); }
    if (t) {
      if (t.closest("#" + VIEWS.writeup)) return { view: "writeup", target: t };
      if (t.closest("#" + VIEWS.demo)) return { view: "demo", target: t };
    }
    return null; // unknown hash: leave the view as it is
  }

  function scrollToTarget(target) {
    if (target) target.scrollIntoView({ block: "start", behavior: "instant" });
    else window.scrollTo({ top: 0, left: 0, behavior: "instant" });
  }

  function apply(view, target, forceScroll) {
    var previous = current;
    var changed = previous !== view;
    current = view;
    body.setAttribute("data-view", view);
    Object.keys(VIEWS).forEach(function (v) {
      var sec = document.getElementById(VIEWS[v]);
      if (sec) sec.hidden = v !== view;
    });
    press(view);
    if (sw) holdBar(); // before the scroll: the header's height can change with the view (phones)
    var name = document.querySelector(".bar__name");
    if (name) name.setAttribute("href", view === "demo" ? "#demo" : "#top");
    window.HAI.view = view;
    if (changed) {
      document.dispatchEvent(new CustomEvent("hai:viewchange", { detail: { view: view, previous: previous } }));
    }
    if (changed || forceScroll) scrollToTarget(target);
  }

  /* the toggle: aria-pressed, and the thumb (.viewseg[data-on]) */
  function press(view) {
    document.querySelectorAll("[data-view-to]").forEach(function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-view-to") === view ? "true" : "false");
    });
    var seg = document.querySelector(".viewseg");
    if (seg) seg.setAttribute("data-on", view);
  }

  function commit() {
    if (!wish) return;
    var w = wish;
    wish = null;
    apply(w.view, w.target, w.force);
  }

  /* end a switch: stop its timers and transition, and leave no class or inline style behind */
  function stop(s) {
    if (!s || sw !== s) return;
    sw = null;
    s.timers.forEach(clearTimeout);
    if (s.vt) { try { s.vt.skipTransition(); } catch (e) { /* already done */ } }
    CLS.forEach(function (c) { root.classList.remove(c); });
    root.style.removeProperty("--hai-oy");
    root.style.removeProperty("--hai-bar-h");
    if (!root.getAttribute("style")) root.removeAttribute("style");
    Object.keys(VIEWS).forEach(function (v) {
      var sec = document.getElementById(VIEWS[v]);
      if (sec) sec.classList.remove("is-leaving", "is-entering");
    });
  }
  /* a switch that ran to its end: wait for the header's colour transitions too, then clean up */
  function settle(s) {
    if (sw !== s) return;
    var bar = document.querySelector(".bar");
    var anims = bar && bar.getAnimations ? bar.getAnimations({ subtree: true }) : [];
    anims = anims.filter(function (a) { return a.playState === "running" || a.playState === "pending"; });
    if (!anims.length || (s.waits = (s.waits || 0) + 1) > 4) { stop(s); return; }
    Promise.all(anims.map(function (a) { return a.finished.catch(function () {}); })).then(function () {
      requestAnimationFrame(function () { settle(s); });
    });
  }

  /* the header is fixed during a switch (css/style.css); the body keeps its place */
  function holdBar() {
    var bar = document.querySelector(".bar");
    if (bar) root.style.setProperty("--hai-bar-h", bar.offsetHeight + "px");
  }

  function animated(view) {
    return started && view !== current && !RM.matches && !document.hidden;
  }

  /* every view change goes through here */
  function request(view, target, force) {
    if (sw && sw.to === view) {
      // the running switch already goes there (popstate and hashchange both fire on back/forward)
      if (current === view) apply(view, target, force);
      else wish = { view: view, target: target, force: force };
      return;
    }
    if (sw) stop(sw);
    wish = { view: view, target: target, force: force };
    if (!animated(view)) { commit(); return; }

    var from = document.getElementById(VIEWS[current]);
    var s = { id: ++swId, to: view, vt: null, timers: [] };
    sw = s;
    // the outgoing view settles back towards the middle of the screen, not of its own (tall) box
    if (from) root.style.setProperty("--hai-oy", Math.round(window.innerHeight / 2 - from.getBoundingClientRect().top) + "px");
    holdBar();
    root.classList.add("is-switching", "is-switching-to-" + view);
    // the transitions must be in place before the toggle changes, or the thumb jumps: read a style first
    var seg = document.querySelector(".viewseg");
    if (seg) void getComputedStyle(seg, "::before").transform;
    press(view);

    if (typeof document.startViewTransition === "function") {
      var t;
      try {
        t = document.startViewTransition(function () { if (sw === s) commit(); });
      } catch (e) { t = null; }
      if (t) {
        s.vt = t;
        var quiet = function () {};
        t.ready.catch(quiet);
        t.updateCallbackDone.catch(quiet);
        t.finished.then(function () { settle(s); }, function () { stop(s); });
        return;
      }
    }
    // no View Transitions API: fade out, swap, fade in
    root.classList.add("is-fading");
    if (from) from.classList.add("is-leaving");
    s.timers.push(setTimeout(function () {
      if (sw !== s) return;
      if (from) from.classList.remove("is-leaving");
      commit();
      var into = document.getElementById(VIEWS[current]);
      if (into) into.classList.add("is-entering");
      s.timers.push(setTimeout(function () { settle(s); }, SWITCH_MS - OUT_MS + 40));
    }, OUT_MS));
  }

  function route(forceScroll) {
    var r = viewOfHash(location.hash);
    if (!r) { if (!current) apply("demo", null, false); return; }
    request(r.view, r.target, forceScroll);
  }

  function showView(view, hash) {
    if (!VIEWS[view]) return;
    var h = hash ? (hash.charAt(0) === "#" ? hash : "#" + hash) : "#" + view;
    var r = viewOfHash(h);
    if (!r || r.view !== view) { h = "#" + view; r = viewOfHash(h); }
    if (location.hash !== h) {
      try { history.pushState(null, "", location.pathname + location.search + h); } catch (e) { /* file:// */ }
    }
    request(view, r.target, true);
  }

  window.HAI = window.HAI || {};
  window.HAI.showView = showView;

  document.querySelectorAll("[data-view-to]").forEach(function (b) {
    b.addEventListener("click", function () {
      var v = b.getAttribute("data-view-to");
      if (v !== (sw ? sw.to : current)) showView(v);
    });
  });

  /* During a view transition Chromium hit-tests the whole page as <html>: keep the focus where it is,
     and pass a click on the toggle through to the button under the pointer (the header does not move). */
  root.addEventListener("mousedown", function (e) { if (sw && e.target === root) e.preventDefault(); });
  root.addEventListener("click", function (e) {
    if (!sw || e.target !== root) return;
    document.querySelectorAll("[data-view-to]").forEach(function (b) {
      var r = b.getBoundingClientRect();
      if (e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom) { b.focus({ preventScroll: true }); b.click(); }
    });
  });

  /* The browser jumps to a hash that names an element by itself. #writeup and the old anchors in
     ALIAS name no element, so with the view already open nothing would move: scroll for them. */
  function needsScroll(hash) {
    var h = (hash || "").replace(/^#/, "");
    return h === "writeup" || (ALIAS.hasOwnProperty(h) && !document.getElementById(h));
  }
  window.addEventListener("hashchange", function () { route(needsScroll(location.hash)); });
  window.addEventListener("popstate", function () { route(false); });

  route(false);
  started = true;
  if (RM.addEventListener) RM.addEventListener("change", function () { if (RM.matches && sw) { var s = sw; stop(s); commit(); } });

  /* A deep link into the write-up (#results, #limits): the charts draw after the data arrives and
     push the section down. When the fonts and app.js's charts are in, scroll to the target again,
     unless the reader has scrolled or pressed a key since (scrollY alone cannot tell: the
     browser's scroll anchoring moves it too). */
  var initial = viewOfHash(location.hash);
  if (initial && initial.view === "writeup" && initial.target) {
    var touched = false;
    var mark = function () { touched = true; };
    ["wheel", "touchstart", "keydown", "pointerdown"].forEach(function (t) { window.addEventListener(t, mark, { once: true, passive: true }); });
    var again = function () {
      if (touched || current !== "writeup") return;
      var r = viewOfHash(location.hash);
      if (r && r.target === initial.target) scrollToTarget(initial.target);
    };
    window.addEventListener("load", again, { once: true });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(again);
    document.addEventListener("hai:writeup-drawn", again, { once: true });
  }
})();
