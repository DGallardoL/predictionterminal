/* eslint-disable */
/**
 * web/js/events-bridge.js — W11-02 (wave-11)
 *
 * Minimal browser-side event glue that broadcasts the lifecycle of a
 * `POST /fit` request as DOM CustomEvents so that downstream wave-10
 * modules (T61 regression-results-sticky, T62 result-pinner, T63
 * regression-explainer, T70 regression-explainer/onboarding-tour, etc.)
 * can react without each having to wrap `window.fetch` themselves.
 *
 * Events dispatched on `document`:
 *
 *   1. `pfm:fit-start`     fired *before* the fetch is initiated.
 *                          detail = { estimatedMs:number }
 *
 *   2. `pfm:fit-complete`  fired after a 2xx response body has been
 *                          parsed as JSON.
 *                          detail = { result:object, success:true }
 *
 *   3. `pfm:fit-end`       fired on any terminal condition (network
 *                          error, non-2xx response, JSON parse error,
 *                          or after a successful complete).
 *                          detail = { success:boolean, elapsedMs:number }
 *
 * The bridge wraps `window.fetch` ONCE (idempotent via the
 * `__pfmEventsBridgeWrapped` sentinel). It never calls `fetch` itself,
 * so it cannot recurse. The response is `.clone()`d before reading the
 * body so downstream consumers (the original caller) see an intact,
 * unread `Response`.
 *
 * Coexistence with T69 (`web/js/regression-loading.js`):
 *
 *   T69 ALSO wraps `window.fetch` to drive its staged progress UI and
 *   dispatches its own `pfm:fit-complete` with `detail = {success,
 *   elapsedMs}` (no `result`). Both wrappers chain safely because:
 *
 *     - Each wrapper checks `__*Wrapped` on the current `window.fetch`
 *       before installing, so neither is ever applied twice.
 *     - Each wrapper calls the previous `window.fetch` it captured,
 *       i.e. classical decorator chaining.
 *     - The response Promise is `.then()`'d, not awaited, so each
 *       wrapper sees the same Response instance and uses `.clone()`
 *       before reading the body.
 *
 *   MOUNT ORDER (matters for layering, not for correctness):
 *
 *     <script src="js/events-bridge.js" defer></script>      <!-- inner -->
 *     <script src="js/regression-loading.js" defer></script> <!-- outer -->
 *
 *   With this order, T69's wrapper sits OUTSIDE the bridge. Call flow
 *   for `fetch('/fit', {method:'POST'})`:
 *
 *     caller -> T69.wrap()  pfm:fit-loading-start, progress UI starts
 *            -> bridge.wrap()  pfm:fit-start dispatched
 *            -> origFetch()
 *            <- Response
 *            -> bridge.then()  parse JSON, pfm:fit-complete (with result),
 *                              then pfm:fit-end
 *            -> T69.then()     stop UI, pfm:fit-complete (success only)
 *
 *   Both `pfm:fit-complete` events fire; T69's lacks `detail.result`.
 *   Listeners that need the parsed result MUST defensively check
 *   `ev.detail && ev.detail.result` and ignore events without it.
 *
 *   If the scripts are mounted in the OPPOSITE order (bridge after
 *   T69), the chain is symmetric and still safe — only the relative
 *   ordering of the two `pfm:fit-complete` dispatches swaps.
 *
 * Public API: `window.PFM.events = { armFitBridge(), disarm() }`.
 *   - armFitBridge(): install the fetch wrapper if not already armed.
 *     Called automatically once at script load.
 *   - disarm(): restore the original `window.fetch`. Idempotent.
 *     Mainly useful for tests / debugging.
 */
(function () {
  "use strict";

  if (typeof window === "undefined") return;
  window.PFM = window.PFM || {};
  if (window.PFM.events && window.PFM.events.__w11_02) {
    return; // already mounted — defensive against double-include
  }

  // ── Match policy ────────────────────────────────────────────────────────
  // Tight regex so we don't fire on /fit/preview, /event-model/fit, etc.
  // Mirrors the rule used in T69 regression-loading.js for consistency.
  var FIT_URL_RE = /\/fit(\?|$)/;
  // Heuristic baseline used when the user has no recorded EWMA. Kept in
  // sync with T69's BASELINE_MS so progress visuals don't desync.
  var BASELINE_MS = 3500;
  var EWMA_KEY = "pfm:fit-avg-ms"; // shared with T69 (read-only here)

  function readEstimatedMs() {
    try {
      var raw = window.localStorage.getItem(EWMA_KEY);
      var n = raw == null ? NaN : parseFloat(raw);
      if (!isFinite(n) || n <= 0) return BASELINE_MS;
      return Math.max(400, Math.min(60000, n));
    } catch (_) {
      return BASELINE_MS;
    }
  }

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

  function dispatch(name, detail) {
    try {
      document.dispatchEvent(new CustomEvent(name, { detail: detail || {} }));
    } catch (_) {
      // Some very old browsers don't have CustomEvent constructor; we
      // intentionally swallow because the bridge is a best-effort glue.
    }
  }

  // ── Wrapper installation (idempotent) ───────────────────────────────────
  var _origFetch = null; // captured previous fetch (may itself be wrapped)
  var _armed = false;

  function armFitBridge() {
    if (_armed) return;
    if (typeof window.fetch !== "function") return;
    if (window.fetch.__pfmEventsBridgeWrapped) {
      _armed = true;
      return;
    }
    _origFetch = window.fetch.bind(window);
    var wrapped = function (input, init) {
      var matched = false;
      var t0 = 0;
      try {
        matched = isFitRequest(input, init);
      } catch (_) {
        matched = false;
      }
      if (matched) {
        t0 = (typeof performance !== "undefined" && performance.now)
          ? performance.now()
          : Date.now();
        dispatch("pfm:fit-start", { estimatedMs: readEstimatedMs() });
      }
      // IMPORTANT: never invoke window.fetch here; always go through the
      // captured original. Otherwise we'd recurse through our own wrapper.
      var p;
      try {
        p = _origFetch(input, init);
      } catch (syncErr) {
        if (matched) {
          dispatch("pfm:fit-end", { success: false, elapsedMs: 0 });
        }
        throw syncErr;
      }
      if (!matched) return p;
      // Tap the promise without consuming it. We `.then()` and return
      // the original `p`, so the caller still receives the Response
      // (or rejection) untouched.
      p.then(
        function (resp) {
          var elapsed = ((typeof performance !== "undefined" && performance.now)
            ? performance.now()
            : Date.now()) - t0;
          if (!resp || typeof resp.ok !== "boolean") {
            dispatch("pfm:fit-end", { success: false, elapsedMs: elapsed });
            return;
          }
          if (!resp.ok) {
            dispatch("pfm:fit-end", { success: false, elapsedMs: elapsed });
            return;
          }
          // Clone before reading the body so the original caller can
          // still call resp.json() / resp.text() exactly once.
          var clone;
          try {
            clone = resp.clone();
          } catch (_) {
            dispatch("pfm:fit-end", { success: false, elapsedMs: elapsed });
            return;
          }
          clone.json().then(
            function (result) {
              dispatch("pfm:fit-complete", { result: result, success: true });
              dispatch("pfm:fit-end", { success: true, elapsedMs: elapsed });
            },
            function (_parseErr) {
              // 2xx but body not JSON; downstream consumers can't act
              // on this so we report failure for the bridge's purposes
              // (T69 will still flash "Done" off its own signal).
              dispatch("pfm:fit-end", { success: false, elapsedMs: elapsed });
            }
          );
        },
        function (_err) {
          var elapsed = ((typeof performance !== "undefined" && performance.now)
            ? performance.now()
            : Date.now()) - t0;
          dispatch("pfm:fit-end", { success: false, elapsedMs: elapsed });
        }
      );
      return p;
    };
    wrapped.__pfmEventsBridgeWrapped = true;
    window.fetch = wrapped;
    _armed = true;
  }

  function disarm() {
    if (!_armed) return;
    if (window.fetch && window.fetch.__pfmEventsBridgeWrapped && _origFetch) {
      // Only restore if our wrapper is still the current fetch. If a
      // later script (e.g. T69) wrapped us, we cannot safely unwind
      // their wrapper, so we leave it alone — but mark ourselves as
      // disarmed so re-arming is a no-op.
      window.fetch = _origFetch;
    }
    _origFetch = null;
    _armed = false;
  }

  // ── Public surface ──────────────────────────────────────────────────────
  window.PFM.events = {
    __w11_02: true,
    armFitBridge: armFitBridge,
    disarm: disarm,
  };

  // Self-register so that if index.html mounts this script before any
  // user interaction, the bridge is live without further wiring.
  armFitBridge();
})();
