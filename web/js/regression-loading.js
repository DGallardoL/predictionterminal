/* eslint-disable */
/**
 * web/js/regression-loading.js — T69 (wave-10)
 *
 * Progressive UX for POST /fit. The backend does not stream progress, so we
 * fake staged progress on a time-based schedule calibrated to the user's
 * recent observed durations (EWMA over the last 8 fits, persisted in
 * localStorage under `pfm:fit-avg-ms`).
 *
 * Stages (fractions of estimated duration, must sum to 1.0):
 *   1. "Fetching factor data..."         30%
 *   2. "Fitting OLS..."                   40%
 *   3. "Computing confidence intervals..." 20%
 *   4. "Done"                             10% (visual completion buffer)
 *
 * Public API (mounted at window.PFM.regressionLoading):
 *   - start(estimatedMs?)  Begin the staged progress UI. If estimatedMs is
 *                          omitted we read the EWMA from localStorage (or
 *                          fall back to a 3500 ms baseline).
 *   - stop(success)        Halt the timer, record the actual elapsed
 *                          duration into the EWMA, hide the card. Dispatches
 *                          a `pfm:fit-complete` event on document.
 *   - tick()               Force-advance to the next stage (useful for
 *                          tests / debugging).
 *
 * Wiring (in order of preference):
 *   A. AUTO via window.fetch wrapper. We patch window.fetch once on script
 *      load; any POST whose URL matches /\/fit($|\?)/ triggers start()/stop()
 *      automatically. This is the zero-touch path that works with the
 *      existing runFit() in web/index.html.
 *   B. EVENT hook. Dispatch `pfm:fit-start` (optionally with
 *      `detail.estimatedMs`) on document to start manually, and
 *      `pfm:fit-end` (with `detail.success`) to stop. Use this if the
 *      fetch wrapper is bypassed (e.g. custom XHR or a different transport).
 *
 * Mount instruction for the index.html owner:
 *   <link rel="stylesheet" href="css/regression-loading.css">
 *   <script src="js/regression-loading.js" defer></script>
 *   No further wiring is required. The card auto-injects into the
 *   regression mode-pane (selector: `[data-mode-pane="regression"]`)
 *   immediately after the first child <header>/<section>. If that
 *   selector is missing the card falls back to <body>.
 */
(function () {
  "use strict";

  if (typeof window === "undefined") return;
  window.PFM = window.PFM || {};
  if (window.PFM.regressionLoading && window.PFM.regressionLoading.__t69) {
    return; // already mounted (defensive against double-include)
  }

  // ── Tunables ────────────────────────────────────────────────────────────
  var LS_KEY = "pfm:fit-avg-ms";
  var BASELINE_MS = 3500;
  var EWMA_ALPHA = 0.3; // weight on the latest observation
  var TICK_INTERVAL_MS = 80;
  var STAGES = [
    { label: "Fetching factor data…",        frac: 0.30 },
    { label: "Fitting OLS…",                  frac: 0.40 },
    { label: "Computing confidence intervals…", frac: 0.20 },
    { label: "Done",                          frac: 0.10 },
  ];

  // ── EWMA persistence ────────────────────────────────────────────────────
  function readAvgMs() {
    try {
      var raw = window.localStorage.getItem(LS_KEY);
      var n = raw == null ? NaN : parseFloat(raw);
      if (!isFinite(n) || n <= 0) return BASELINE_MS;
      // Clamp to a sane range so an outlier doesn't poison the UX.
      return Math.max(400, Math.min(60_000, n));
    } catch (_) {
      return BASELINE_MS;
    }
  }
  function writeAvgMs(actualMs) {
    if (!isFinite(actualMs) || actualMs <= 0) return;
    try {
      var prev = readAvgMs();
      var next = (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * actualMs;
      window.localStorage.setItem(LS_KEY, String(Math.round(next)));
    } catch (_) {}
  }

  // ── DOM ─────────────────────────────────────────────────────────────────
  var cardEl = null;
  var dotEls = [];
  var labelEl = null;
  var barEl = null;

  function ensureCard() {
    if (cardEl && document.body.contains(cardEl)) return cardEl;
    var existing = document.getElementById("pfm-regression-loading");
    if (existing) {
      cardEl = existing;
    } else {
      cardEl = document.createElement("div");
      cardEl.id = "pfm-regression-loading";
      cardEl.className = "pfm-rxl-card pfm-rxl-hidden";
      cardEl.setAttribute("role", "status");
      cardEl.setAttribute("aria-live", "polite");
      cardEl.innerHTML =
        '<div class="pfm-rxl-dots" aria-hidden="true">' +
          '<span class="pfm-rxl-dot" data-stage="0"></span>' +
          '<span class="pfm-rxl-bar"><span class="pfm-rxl-bar-fill"></span></span>' +
          '<span class="pfm-rxl-dot" data-stage="1"></span>' +
          '<span class="pfm-rxl-bar"><span class="pfm-rxl-bar-fill"></span></span>' +
          '<span class="pfm-rxl-dot" data-stage="2"></span>' +
          '<span class="pfm-rxl-bar"><span class="pfm-rxl-bar-fill"></span></span>' +
          '<span class="pfm-rxl-dot" data-stage="3"></span>' +
        "</div>" +
        '<div class="pfm-rxl-label" data-pfm-rxl-label>Working…</div>';
      // Mount inside the regression mode pane (or fall back to <body>).
      var pane = document.querySelector('.mode-pane[data-mode-pane="regression"]')
        || document.querySelector('[data-mode-pane="regression"]')
        || document.body;
      // Prefer placing the card at the top of the pane so the sticky
      // position can latch onto the page scroll container.
      if (pane.firstChild) {
        pane.insertBefore(cardEl, pane.firstChild);
      } else {
        pane.appendChild(cardEl);
      }
    }
    dotEls = Array.prototype.slice.call(cardEl.querySelectorAll(".pfm-rxl-dot"));
    labelEl = cardEl.querySelector("[data-pfm-rxl-label]");
    barEl = cardEl; // kept for symmetry; bar fills are per-segment
    return cardEl;
  }

  function applyStageVisuals(stageIdx, fracWithinStage) {
    if (!cardEl) return;
    // 1) Dots
    for (var i = 0; i < dotEls.length; i++) {
      var d = dotEls[i];
      if (i < stageIdx) {
        d.className = "pfm-rxl-dot is-done";
      } else if (i === stageIdx) {
        d.className = "pfm-rxl-dot is-active";
      } else {
        d.className = "pfm-rxl-dot";
      }
    }
    // 2) Connector bar fills (one between each pair of dots)
    var bars = cardEl.querySelectorAll(".pfm-rxl-bar-fill");
    for (var j = 0; j < bars.length; j++) {
      var fill;
      if (j < stageIdx) fill = 1;
      else if (j === stageIdx) fill = Math.max(0, Math.min(1, fracWithinStage));
      else fill = 0;
      bars[j].style.transform = "scaleX(" + fill + ")";
    }
    // 3) Label
    if (labelEl) {
      var stage = STAGES[Math.min(stageIdx, STAGES.length - 1)];
      labelEl.textContent = stage ? stage.label : "Working…";
    }
  }

  function showCard() {
    ensureCard();
    cardEl.classList.remove("pfm-rxl-hidden");
    cardEl.classList.add("pfm-rxl-visible");
  }
  function hideCard() {
    if (!cardEl) return;
    cardEl.classList.add("pfm-rxl-hidden");
    cardEl.classList.remove("pfm-rxl-visible");
  }

  // ── State machine ───────────────────────────────────────────────────────
  var state = {
    running: false,
    t0: 0,
    estimatedMs: BASELINE_MS,
    timerId: null,
    stageIdx: 0,
  };

  function computeStageFromElapsed(elapsed, estimated) {
    // Soft-cap elapsed at 95% of estimated so we never visually "finish"
    // until stop() is actually called.
    var capped = Math.min(elapsed, 0.95 * estimated);
    var frac = capped / Math.max(1, estimated);
    var cum = 0;
    for (var i = 0; i < STAGES.length; i++) {
      var next = cum + STAGES[i].frac;
      if (frac < next || i === STAGES.length - 1) {
        var within = STAGES[i].frac > 0 ? (frac - cum) / STAGES[i].frac : 1;
        return { idx: i, within: within };
      }
      cum = next;
    }
    return { idx: STAGES.length - 1, within: 1 };
  }

  function tickLoop() {
    if (!state.running) return;
    var elapsed = performance.now() - state.t0;
    var s = computeStageFromElapsed(elapsed, state.estimatedMs);
    state.stageIdx = s.idx;
    applyStageVisuals(s.idx, s.within);
  }

  function start(estimatedMs) {
    if (state.running) {
      // Reset rather than refuse — caller may have lost track.
      stop(false);
    }
    state.estimatedMs = (typeof estimatedMs === "number" && estimatedMs > 0)
      ? estimatedMs
      : readAvgMs();
    state.t0 = performance.now();
    state.stageIdx = 0;
    state.running = true;
    ensureCard();
    showCard();
    applyStageVisuals(0, 0);
    if (state.timerId) clearInterval(state.timerId);
    state.timerId = setInterval(tickLoop, TICK_INTERVAL_MS);
    try {
      document.dispatchEvent(new CustomEvent("pfm:fit-loading-start", {
        detail: { estimatedMs: state.estimatedMs }
      }));
    } catch (_) {}
  }

  function stop(success) {
    if (!state.running) {
      // Still emit the complete event so downstream listeners (T63
      // regression-explainer) can react even on a no-op.
      try {
        document.dispatchEvent(new CustomEvent("pfm:fit-complete", {
          detail: { success: !!success }
        }));
      } catch (_) {}
      return;
    }
    var elapsed = performance.now() - state.t0;
    state.running = false;
    if (state.timerId) {
      clearInterval(state.timerId);
      state.timerId = null;
    }
    if (success) {
      // Flash the "Done" stage briefly so the user gets closure, then hide.
      applyStageVisuals(STAGES.length - 1, 1);
      setTimeout(hideCard, 280);
      writeAvgMs(elapsed);
    } else {
      hideCard();
    }
    try {
      document.dispatchEvent(new CustomEvent("pfm:fit-complete", {
        detail: { success: !!success, elapsedMs: elapsed }
      }));
    } catch (_) {}
  }

  function tick() {
    if (!state.running) return;
    state.stageIdx = Math.min(state.stageIdx + 1, STAGES.length - 1);
    applyStageVisuals(state.stageIdx, 0);
  }

  // ── Fetch interceptor ───────────────────────────────────────────────────
  // We wrap window.fetch exactly once. The match condition is intentionally
  // tight so we don't fire on /fit/preview or /event-model/fit.
  var FIT_URL_RE = /\/fit(\?|$)/;

  function isFitRequest(input, init) {
    try {
      var url, method;
      if (typeof input === "string") {
        url = input;
        method = (init && init.method) || "GET";
      } else if (input && typeof input.url === "string") {
        url = input.url;
        method = input.method || (init && init.method) || "GET";
      } else {
        return false;
      }
      if (String(method).toUpperCase() !== "POST") return false;
      return FIT_URL_RE.test(url);
    } catch (_) {
      return false;
    }
  }

  if (typeof window.fetch === "function" && !window.fetch.__pfmRxlWrapped) {
    var origFetch = window.fetch.bind(window);
    var wrapped = function (input, init) {
      var matched = isFitRequest(input, init);
      if (matched) {
        try { start(); } catch (_) {}
      }
      var p = origFetch(input, init);
      if (matched) {
        p.then(
          function (resp) {
            try { stop(!!(resp && resp.ok)); } catch (_) {}
            return resp;
          },
          function (err) {
            try { stop(false); } catch (_) {}
            throw err;
          }
        );
      }
      return p;
    };
    wrapped.__pfmRxlWrapped = true;
    window.fetch = wrapped;
  }

  // ── Event-based fallback hooks ──────────────────────────────────────────
  document.addEventListener("pfm:fit-start", function (ev) {
    var ms = ev && ev.detail && typeof ev.detail.estimatedMs === "number"
      ? ev.detail.estimatedMs : undefined;
    try { start(ms); } catch (_) {}
  });
  document.addEventListener("pfm:fit-end", function (ev) {
    var ok = !!(ev && ev.detail && ev.detail.success);
    try { stop(ok); } catch (_) {}
  });

  // ── Public surface ──────────────────────────────────────────────────────
  window.PFM.regressionLoading = {
    __t69: true,
    start: start,
    stop: stop,
    tick: tick,
    // Test/diagnostic introspection
    _state: function () { return Object.assign({}, state); },
    _readAvgMs: readAvgMs,
  };
})();
