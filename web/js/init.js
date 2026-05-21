/*
 * init.js (W11-05, wave-11)
 *
 * Single-source-of-truth boot script for the Prediction Terminal frontend.
 *
 * Why this exists
 * ---------------
 * Wave-10 and Wave-11 shipped ~15+ independent JS modules that each attach to
 * `window.PFM.*`. Most self-initialise on load, but a few need a coordinated
 * kick (theme persistence, tour delay, factor prefetch, microcopy scan). Doing
 * this from inline `<script>` blocks in `index.html` is the recipe for the
 * race conditions we've already paid the price for. This module collects all
 * of it into one ordered, idempotent boot sequence.
 *
 * Public API
 * ----------
 *   window.PFM.boot.isReady()  -> boolean
 *   window.PFM.boot.ready      -> Promise<void>      // resolves after sequence
 *   window.PFM.boot.modules    -> { name: durationMs, ... }   // diagnostic
 *
 * Mount order in index.html
 * -------------------------
 * This file MUST be the LAST <script> mounted in index.html, after every
 * other `web/js/*.js` it depends on. The boot sequence is keyed on
 * `DOMContentLoaded`; if that has already fired by the time this script
 * executes, the sequence runs immediately.
 *
 * Idempotence
 * -----------
 * If `window.PFM.boot.__initialized` is already true, this script is a no-op.
 * Each per-module step is wrapped in a try/catch so a single bad module never
 * stops the rest of the boot.
 *
 * Owner: W11-05 (init-boot). Do not edit without coordinating in
 * `.coordination/active-edits.json`.
 */
(function () {
  "use strict";

  // ── Idempotence guard ────────────────────────────────────────────────────
  window.PFM = window.PFM || {};
  if (window.PFM.boot && window.PFM.boot.__initialized) {
    return;
  }

  // ── Internal state ───────────────────────────────────────────────────────
  var _ready = false;
  var _timings = {};
  var _resolveReady;
  var _readyPromise = new Promise(function (resolve) {
    _resolveReady = resolve;
  });

  function _now() {
    if (typeof performance !== "undefined" && performance.now) {
      return performance.now();
    }
    return Date.now();
  }

  /**
   * Run a single boot step. Catches all errors, records timing in ms, and
   * returns true if the step ran without throwing (regardless of whether the
   * underlying module was actually present).
   */
  function step(name, fn) {
    var t0 = _now();
    var ok = true;
    try {
      fn();
    } catch (err) {
      ok = false;
      // Surface but never re-throw: a single bad module shouldn't halt boot.
      // Prefer console.error so dev tools highlight it.
      try {
        console.error("[PFM.boot] step '" + name + "' threw:", err);
      } catch (e) {
        /* console may be unavailable in some embedded contexts */
      }
    }
    _timings[name] = Math.round((_now() - t0) * 100) / 100;
    return ok;
  }

  /**
   * Defensive accessor: returns the value at the given dotted path under
   * window.PFM, or undefined if any segment is missing. Avoids the verbose
   * `window.PFM && window.PFM.foo && window.PFM.foo.bar` pattern.
   */
  function pfmGet(path) {
    var node = window.PFM;
    if (!node) return undefined;
    var parts = path.split(".");
    for (var i = 0; i < parts.length; i++) {
      if (node == null) return undefined;
      node = node[parts[i]];
    }
    return node;
  }

  // ── Per-module steps ─────────────────────────────────────────────────────
  // Each step is defensive: if the target module isn't loaded, the step is a
  // graceful no-op. This is important because index.html may choose to omit
  // a script (e.g. embed mode) without breaking the boot chain.

  function initTheme() {
    var setFn = pfmGet("theme.set");
    if (typeof setFn !== "function") return;
    var stored = null;
    try {
      stored = window.localStorage && window.localStorage.getItem("pfm:theme");
    } catch (e) {
      // localStorage may be blocked (Safari private mode, iframe sandbox).
      stored = null;
    }
    setFn(stored || "system");
  }

  function initCmdk() {
    // T03 cmdk auto-arms on script load. We only verify presence and emit a
    // console note if it failed to mount, which usually means the script tag
    // is missing or threw on load.
    var cmdk = pfmGet("cmdk");
    if (!cmdk || typeof cmdk.toggle !== "function") {
      try {
        console.warn("[PFM.boot] cmdk not present; '/' and Ctrl+K palette will not work.");
      } catch (e) { /* noop */ }
      return;
    }
    // Touch the module to confirm it's reachable. No-op for the user.
  }

  function initErrors() {
    // T07 error-banner attaches as `window.PFM.errors` on script load. There is
    // no explicit `arm()` — the module wires a global fetch interceptor at
    // module evaluation time. We simply assert presence so a missing module
    // is logged at boot.
    var errs = pfmGet("errors");
    if (!errs) {
      try {
        console.warn("[PFM.boot] errors module not present; global errors will surface in console only.");
      } catch (e) { /* noop */ }
    }
  }

  function initShortcuts() {
    // T15 keyboard-shortcuts module exposes `register`, `help`, `enable`,
    // `disable`. It self-installs its keydown listener at load time. We call
    // `enable()` defensively in case the module was disabled by an earlier
    // script (e.g. embed mode).
    var sc = pfmGet("shortcuts");
    if (!sc) return;
    if (typeof sc.enable === "function") {
      sc.enable();
    }
  }

  function initTour() {
    // T13 onboarding-tour: only start if the user has not completed it. Delay
    // 1500 ms so the rest of the UI has time to settle (data fetches, layout
    // shifts, lazy charts) — otherwise tour anchors point at the wrong spot.
    var tour = pfmGet("tour");
    if (!tour) return;
    var alreadyDone = false;
    try {
      alreadyDone = typeof tour.isCompleted === "function" && tour.isCompleted();
    } catch (e) {
      alreadyDone = false;
    }
    if (alreadyDone) return;
    if (typeof tour.start !== "function") return;
    setTimeout(function () {
      try {
        tour.start();
      } catch (err) {
        try {
          console.error("[PFM.boot] tour.start failed:", err);
        } catch (e) { /* noop */ }
      }
    }, 1500);
  }

  function initConnection() {
    // T60 connection-status auto-mounts its DOM pill and starts heartbeats on
    // module load. We just verify presence.
    var conn = pfmGet("conn");
    if (!conn) return;
    // If a paused state was inherited from a prior session (unlikely but
    // possible during hot-reload), resume on boot.
    if (typeof conn.resume === "function") {
      try { conn.resume(); } catch (e) { /* noop */ }
    }
  }

  function initPinner() {
    // T70 result-pinner self-mounts. Verify presence.
    var pb = pfmGet("pinboard");
    if (!pb) return;
    // No explicit init required; module wires its UI on load.
  }

  function initEventsBridge() {
    // W11-02 events-bridge: armFitBridge() is the explicit arm call. The
    // module also self-arms on load, but calling armFitBridge() is idempotent
    // (the module guards re-entry via the `_armed` flag) and gives us a clean
    // single point of control.
    var arm = pfmGet("events.armFitBridge");
    if (typeof arm !== "function") return;
    arm();
  }

  function initModeRouter() {
    // W11-03 mode-router applies the last-used mode on its own DOMContentLoaded
    // handler. We don't need to call anything explicitly — the module's own
    // init() already runs `applyMode(savedMode)` on boot. We touch the module
    // to confirm presence; future versions may expose `applyLastMode()`.
    var mr = pfmGet("modeRouter");
    if (!mr) return;
    if (typeof mr.applyLast === "function") {
      mr.applyLast();
    }
    // Otherwise rely on module's own DOMContentLoaded handler.
  }

  function initEmptyStates() {
    // W11-60 empty-states: the shipped artifact in this wave is CSS-only
    // (`web/css/empty-states.css`). When a JS companion ships with a
    // `registerDataObserver()` method, this step will call it. Until then,
    // this is a forward-compatible no-op.
    var es = pfmGet("emptyStates");
    if (!es) return;
    if (typeof es.registerDataObserver === "function") {
      es.registerDataObserver();
    }
  }

  function initMicrocopy() {
    // W11-06 microcopy: run the CopyLinter on document.body and log any
    // banned-phrase findings as a single collapsed group. We do NOT auto-edit
    // the DOM — that's a design decision; the lint pass is informational.
    var copy = pfmGet("copy");
    if (!copy || !copy.CopyLinter || typeof copy.CopyLinter.scan !== "function") {
      return;
    }
    var warnings;
    try {
      warnings = copy.CopyLinter.scan(document.body) || [];
    } catch (e) {
      warnings = [];
    }
    if (!warnings.length) return;
    try {
      console.groupCollapsed(
        "[PFM.boot] microcopy lint: " + warnings.length + " banned-phrase findings"
      );
      warnings.forEach(function (w) { console.warn(w); });
      console.groupEnd();
    } catch (e) { /* noop */ }
  }

  function initCurl() {
    // W11-08 copy-as-curl: attaches a global fetch recorder. attach() is the
    // canonical arm call; it is idempotent.
    var curl = pfmGet("curl");
    if (!curl) return;
    if (typeof curl.attach === "function") {
      curl.attach();
    }
  }

  function initFactorSearch() {
    // W11-09 factor-search-fuzzy: there is no public `prefetch()`, but
    // calling `query("")` triggers the internal /factors fetch and warms the
    // 5-minute cache so the first user keystroke is instant. We fire-and-
    // forget — failure to prefetch is non-fatal.
    var fs = pfmGet("factorSearch");
    if (!fs || typeof fs.query !== "function") return;
    try {
      var p = fs.query("");
      if (p && typeof p.catch === "function") {
        p.catch(function () { /* swallowed: cached miss is OK */ });
      }
    } catch (e) { /* noop */ }
  }

  // ── Boot sequence orchestrator ───────────────────────────────────────────
  function runBoot() {
    var bootT0 = _now();

    // The order here matters where one module reads another's state at init
    // time. Theme first so subsequent modules can read the resolved theme.
    // Cmdk before shortcuts because Ctrl+K hands off to cmdk. Errors early so
    // later steps can surface failures through the banner. Tour last among
    // user-facing pieces so the page has settled. Lint and prefetches at the
    // end — they're best-effort.
    step("theme",         initTheme);
    step("cmdk",          initCmdk);
    step("errors",        initErrors);
    step("shortcuts",     initShortcuts);
    step("conn",          initConnection);
    step("pinner",        initPinner);
    step("events-bridge", initEventsBridge);
    step("mode-router",   initModeRouter);
    step("empty-states",  initEmptyStates);
    step("curl",          initCurl);
    step("microcopy",     initMicrocopy);
    step("factor-search", initFactorSearch);
    step("tour",          initTour);

    var totalMs = Math.round((_now() - bootT0) * 100) / 100;
    _timings.__total = totalMs;

    try {
      console.groupCollapsed("[PFM.boot] startup timeline (" + totalMs + " ms)");
      Object.keys(_timings).forEach(function (k) {
        if (k === "__total") return;
        console.log(
          "  " + k.padEnd ? k.padEnd(16, " ") : k,
          _timings[k] + " ms"
        );
      });
      console.log("  ──────────────  ");
      console.log("  total           " + totalMs + " ms");
      console.groupEnd();
    } catch (e) {
      /* console.groupCollapsed may be unavailable in some embedded contexts */
    }

    _ready = true;
    try { _resolveReady(); } catch (e) { /* noop */ }
  }

  // ── Schedule boot ────────────────────────────────────────────────────────
  // If DOMContentLoaded has already fired (which happens when this script tag
  // is placed at the end of <body>, the most common case), run immediately on
  // a microtask so callers that registered `await PFM.boot.ready` synchronously
  // before our exec can still chain in order.
  function schedule() {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", runBoot, { once: true });
    } else {
      // Defer one tick to let any same-frame script attach listeners.
      if (typeof queueMicrotask === "function") {
        queueMicrotask(runBoot);
      } else {
        setTimeout(runBoot, 0);
      }
    }
  }

  // ── Public API ───────────────────────────────────────────────────────────
  window.PFM.boot = {
    __initialized: true,
    isReady: function () { return _ready; },
    ready: _readyPromise,
    modules: _timings,
    // Exposed for tests / debugging only; do not call from production code.
    _runBoot: runBoot
  };

  schedule();
})();
