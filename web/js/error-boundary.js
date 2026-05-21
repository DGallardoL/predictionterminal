/* ============================================================
 * error-boundary.js  (W13-39, wave-13)
 *
 * Global JS error boundary. Catches uncaught synchronous errors,
 * unhandled promise rejections, and (optionally) fetch network
 * failures. Surfaces a calm, user-friendly toast via
 * PFM.errors.show() with a trace ID, logs to console with
 * grouping, and keeps the last 50 errors in memory for debug.
 *
 * Public API:
 *   window.PFM.errorBoundary = {
 *     install(opts)           // attach window handlers + optional fetch wrap
 *     uninstall()             // remove handlers, restore fetch
 *     history()               // returns shallow copy of recent errors
 *     report(err, ctx)        // record an error manually
 *     clearHistory()          // reset in-memory ring buffer
 *     isInstalled()
 *   }
 *
 *   opts = {
 *     wrapFetch:   boolean   (default true)  — intercept network failures
 *     reportUrl:   string|null (default null) — POST endpoint for errors
 *     reportEnabled: boolean  (default false) — env-gated remote reporting
 *     maxHistory:  number    (default 50)
 *     toastsEnabled: boolean (default true)  — show PFM.errors toasts
 *   }
 *
 * Skipped patterns (no toast, no remote report — still logged):
 *   - "ResizeObserver loop limit exceeded"
 *   - "ResizeObserver loop completed with undelivered notifications"
 *   - "Script error." (cross-origin opaque)
 *   - "Non-Error promise rejection captured"  (3rd-party noise)
 *   - AbortError (user-cancelled fetches)
 *
 * Pairs with web/js/error-banner.js (PFM.errors).
 * ============================================================ */

(function () {
  "use strict";

  /* --------------------------------------------------
   * Config + state
   * -------------------------------------------------- */
  const DEFAULTS = {
    wrapFetch: true,
    reportUrl: "/api/error-report",
    reportEnabled: false,
    maxHistory: 50,
    toastsEnabled: true,
  };

  const SKIP_PATTERNS = [
    /ResizeObserver loop limit exceeded/i,
    /ResizeObserver loop completed with undelivered notifications/i,
    /^Script error\.?$/i,
    /Non-Error promise rejection captured/i,
    /^AbortError/i,
    /The operation was aborted/i,
  ];

  const state = {
    installed: false,
    opts: Object.assign({}, DEFAULTS),
    history: [],
    originalFetch: null,
    onErrorHandler: null,
    onRejectionHandler: null,
    reportInFlight: 0,
    reportMax: 5,
    toastBudget: { count: 0, windowStart: 0, max: 4, windowMs: 10000 },
  };

  let _nextTrace = 1;
  function newTraceId() {
    const t = Date.now().toString(36);
    const n = (_nextTrace++).toString(36);
    const r = Math.random().toString(36).slice(2, 6);
    return "trc-" + t + "-" + n + r;
  }

  function shouldSkip(message) {
    if (!message) return false;
    const m = String(message);
    for (let i = 0; i < SKIP_PATTERNS.length; i++) {
      if (SKIP_PATTERNS[i].test(m)) return true;
    }
    return false;
  }

  function friendlyMessage(kind, raw) {
    // Map raw technical strings to calmer user-facing phrasing
    if (kind === "network") return "We couldn't reach the server. Check your connection and try again.";
    if (kind === "promise") return "Something failed in the background. Try the action again.";
    if (kind === "syntax") return "The page hit a snag. Reload to continue.";
    if (kind === "type") return "Something didn't load correctly on the page.";
    return "Something went wrong. We logged it.";
  }

  function classify(err, message) {
    if (err && err.name === "TypeError" && /fetch|NetworkError|Failed to fetch/i.test(String(err.message || message))) {
      return "network";
    }
    if (err && err.name === "SyntaxError") return "syntax";
    if (err && err.name === "TypeError") return "type";
    return "generic";
  }

  function pushHistory(entry) {
    state.history.push(entry);
    const cap = state.opts.maxHistory > 0 ? state.opts.maxHistory : 50;
    while (state.history.length > cap) state.history.shift();
  }

  function consoleReport(entry) {
    try {
      const tag = "%c[error-boundary] " + entry.trace_id;
      const style = "color:#e85d24;font-weight:600;";
      // Use grouping so devs can collapse noise
      if (console.groupCollapsed) {
        console.groupCollapsed(tag, style, entry.kind, entry.message);
        console.error("source :", entry.source || "(unknown)");
        console.error("stack  :", entry.stack || "(none)");
        if (entry.context) console.error("context:", entry.context);
        console.error("entry  :", entry);
        console.groupEnd();
      } else {
        console.error(tag, style, entry);
      }
    } catch (_e) {
      // never let logging itself throw
    }
  }

  function toastBudgetOK() {
    const now = Date.now();
    const b = state.toastBudget;
    if (now - b.windowStart > b.windowMs) {
      b.windowStart = now;
      b.count = 0;
    }
    if (b.count >= b.max) return false;
    b.count += 1;
    return true;
  }

  function showToast(entry) {
    if (!state.opts.toastsEnabled) return;
    if (!window.PFM || !window.PFM.errors || typeof window.PFM.errors.show !== "function") return;
    if (!toastBudgetOK()) return; // avoid storms

    try {
      window.PFM.errors.show(friendlyMessage(entry.kind, entry.message), {
        kind: entry.severity || "error",
        title: entry.title || undefined,
        traceId: entry.trace_id,
        autoDismissMs: 12000,
      });
    } catch (_e) {
      // swallow — never throw from the boundary
    }
  }

  function maybeRemoteReport(entry) {
    if (!state.opts.reportEnabled) return;
    if (!state.opts.reportUrl) return;
    if (state.reportInFlight >= state.reportMax) return;

    // Strip noisy fields before POST; cap payload size
    const payload = {
      trace_id: entry.trace_id,
      kind: entry.kind,
      message: String(entry.message || "").slice(0, 1000),
      source: entry.source ? String(entry.source).slice(0, 500) : null,
      line: entry.line || null,
      col: entry.col || null,
      stack: entry.stack ? String(entry.stack).slice(0, 4000) : null,
      ua: (navigator && navigator.userAgent) ? String(navigator.userAgent).slice(0, 250) : null,
      url: (location && location.href) ? String(location.href).slice(0, 500) : null,
      ts: entry.ts,
    };

    state.reportInFlight += 1;
    try {
      // Prefer sendBeacon when available (fire-and-forget, no CORS preflight)
      if (navigator && typeof navigator.sendBeacon === "function") {
        const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
        navigator.sendBeacon(state.opts.reportUrl, blob);
        state.reportInFlight -= 1;
        return;
      }
      const f = state.originalFetch || window.fetch;
      f(state.opts.reportUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        keepalive: true,
      }).catch(function () { /* swallow */ }).finally(function () {
        state.reportInFlight -= 1;
      });
    } catch (_e) {
      state.reportInFlight -= 1;
    }
  }

  /* --------------------------------------------------
   * Core recorder
   * -------------------------------------------------- */
  function record(opts) {
    const o = opts || {};
    const message = o.message != null ? String(o.message) : "(unknown error)";

    if (shouldSkip(message)) {
      // still log silently to history for debug
      pushHistory({
        trace_id: newTraceId(),
        kind: "skipped",
        message: message,
        ts: new Date().toISOString(),
        skipped: true,
      });
      return null;
    }

    const err = o.error || null;
    const kind = o.kind || classify(err, message);
    const trace_id = newTraceId();

    const entry = {
      trace_id: trace_id,
      kind: kind,
      severity: o.severity || "error",
      message: message,
      source: o.source || null,
      line: typeof o.line === "number" ? o.line : null,
      col: typeof o.col === "number" ? o.col : null,
      stack: err && err.stack ? String(err.stack) : (o.stack || null),
      context: o.context || null,
      title: o.title || null,
      ts: new Date().toISOString(),
    };

    pushHistory(entry);
    consoleReport(entry);
    showToast(entry);
    maybeRemoteReport(entry);
    return trace_id;
  }

  /* --------------------------------------------------
   * Handlers
   * -------------------------------------------------- */
  function handleWindowError(event) {
    // event: ErrorEvent
    try {
      const err = event && event.error ? event.error : null;
      record({
        kind: classify(err, event && event.message),
        message: (event && event.message) || (err && err.message) || "Uncaught error",
        source: event && event.filename,
        line: event && event.lineno,
        col: event && event.colno,
        error: err,
      });
    } catch (_e) { /* swallow */ }
    // do not preventDefault — let devtools also see it
  }

  function handleRejection(event) {
    try {
      const reason = event && event.reason;
      let message;
      let err = null;
      if (reason instanceof Error) {
        err = reason;
        message = reason.message || String(reason);
      } else if (typeof reason === "string") {
        message = reason;
      } else {
        try { message = "Unhandled rejection: " + JSON.stringify(reason); }
        catch (_e) { message = "Unhandled rejection: (unserializable)"; }
      }
      record({ kind: "promise", message: message, error: err });
    } catch (_e) { /* swallow */ }
  }

  /* --------------------------------------------------
   * fetch wrapper — catches network failures + 5xx surfaces
   * -------------------------------------------------- */
  function installFetchWrap() {
    if (typeof window.fetch !== "function") return;
    if (state.originalFetch) return; // already wrapped
    state.originalFetch = window.fetch.bind(window);

    window.fetch = function wrappedFetch(input, init) {
      const url = (typeof input === "string") ? input : (input && input.url) || "";
      // Never wrap requests to our own error-report endpoint (avoid recursion)
      if (state.opts.reportUrl && url && url.indexOf(state.opts.reportUrl) !== -1) {
        return state.originalFetch(input, init);
      }
      const started = Date.now();
      return state.originalFetch(input, init).then(function (res) {
        // Surface 5xx as soft errors so the user gets a calm toast;
        // 4xx is generally caller-handled and skipped here.
        if (res && res.status >= 500) {
          record({
            kind: "network",
            severity: "warn",
            message: "HTTP " + res.status + " from " + (url || "(request)"),
            context: { duration_ms: Date.now() - started, status: res.status, url: url },
          });
        }
        return res;
      }).catch(function (err) {
        // Don't toast on AbortError — caller cancelled deliberately
        if (err && (err.name === "AbortError" || /aborted/i.test(String(err.message || "")))) {
          throw err;
        }
        record({
          kind: "network",
          message: (err && err.message) ? err.message : "Network request failed",
          source: url,
          error: err,
          context: { duration_ms: Date.now() - started, url: url },
        });
        throw err; // re-throw so caller's .catch still fires
      });
    };
  }

  function uninstallFetchWrap() {
    if (state.originalFetch) {
      window.fetch = state.originalFetch;
      state.originalFetch = null;
    }
  }

  /* --------------------------------------------------
   * Public API
   * -------------------------------------------------- */
  const api = {
    install: function install(userOpts) {
      if (state.installed) return false;
      state.opts = Object.assign({}, DEFAULTS, userOpts || {});

      state.onErrorHandler = handleWindowError;
      state.onRejectionHandler = handleRejection;

      window.addEventListener("error", state.onErrorHandler, true);
      window.addEventListener("unhandledrejection", state.onRejectionHandler, true);

      if (state.opts.wrapFetch) installFetchWrap();
      state.installed = true;
      return true;
    },

    uninstall: function uninstall() {
      if (!state.installed) return false;
      if (state.onErrorHandler) {
        window.removeEventListener("error", state.onErrorHandler, true);
        state.onErrorHandler = null;
      }
      if (state.onRejectionHandler) {
        window.removeEventListener("unhandledrejection", state.onRejectionHandler, true);
        state.onRejectionHandler = null;
      }
      uninstallFetchWrap();
      state.installed = false;
      return true;
    },

    history: function history() {
      return state.history.slice(); // shallow copy
    },

    report: function report(err, ctx) {
      const e = err instanceof Error ? err : null;
      return record({
        kind: e ? classify(e, e.message) : "generic",
        message: (e && e.message) || String(err || "manual report"),
        error: e,
        context: ctx || null,
      });
    },

    clearHistory: function clearHistory() {
      state.history.length = 0;
    },

    isInstalled: function isInstalled() {
      return state.installed;
    },

    // Exposed for tests; not part of the documented surface.
    _skip: shouldSkip,
    _classify: classify,
  };

  if (typeof window !== "undefined") {
    window.PFM = window.PFM || {};
    window.PFM.errorBoundary = api;

    // Auto-install once the DOM is ready, unless the page opts out
    // via <meta name="pfm-error-boundary" content="manual">.
    function maybeAutoInstall() {
      try {
        const meta = document.querySelector('meta[name="pfm-error-boundary"]');
        if (meta && meta.getAttribute("content") === "manual") return;
        api.install();
      } catch (_e) { /* swallow */ }
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", maybeAutoInstall, { once: true });
    } else {
      maybeAutoInstall();
    }
  }
})();
