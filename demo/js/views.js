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
   element with that id). */
(function () {
  "use strict";

  var body = document.body;
  var VIEWS = { demo: "view-demo", writeup: "view-writeup" };
  var current = null;
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
    document.querySelectorAll("[data-view-to]").forEach(function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-view-to") === view ? "true" : "false");
    });
    var name = document.querySelector(".bar__name");
    if (name) name.setAttribute("href", view === "demo" ? "#demo" : "#top");
    window.HAI.view = view;
    if (changed) {
      document.dispatchEvent(new CustomEvent("hai:viewchange", { detail: { view: view, previous: previous } }));
    }
    if (changed || forceScroll) scrollToTarget(target);
  }

  function route(forceScroll) {
    var r = viewOfHash(location.hash);
    if (!r) { if (!current) apply("demo", null, false); return; }
    apply(r.view, r.target, forceScroll);
  }

  function showView(view, hash) {
    if (!VIEWS[view]) return;
    var h = hash ? (hash.charAt(0) === "#" ? hash : "#" + hash) : "#" + view;
    var r = viewOfHash(h);
    if (!r || r.view !== view) { h = "#" + view; r = viewOfHash(h); }
    if (location.hash !== h) {
      try { history.pushState(null, "", location.pathname + location.search + h); } catch (e) { /* file:// */ }
    }
    apply(view, r.target, true);
  }

  window.HAI = window.HAI || {};
  window.HAI.showView = showView;

  document.querySelectorAll("[data-view-to]").forEach(function (b) {
    b.addEventListener("click", function () {
      var v = b.getAttribute("data-view-to");
      if (v !== current) showView(v);
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
