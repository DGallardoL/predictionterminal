/* ============================================================
 * loading-states.js  (W13-38, wave-13)
 *
 * Universal loading state manager. Exposed as window.PFM.loading.
 *
 * API:
 *   window.PFM.loading.start(el, label='Loading')
 *     - Marks the element as loading: dims to 0.6, disables
 *       pointer events, appends overlay with spinner + label.
 *     - Reference-counted: nested start() calls increment a
 *       counter, and stop() only removes UI when counter
 *       reaches zero.
 *
 *   window.PFM.loading.stop(el)
 *     - Decrements counter; removes overlay + class at zero.
 *
 *   window.PFM.loading.wrap(el, asyncFn, label)
 *     - Convenience: starts, awaits asyncFn(), then stops, even
 *       on throw. Returns the asyncFn result (or rethrows).
 *
 *   window.PFM.loading.autoAttach()
 *     - Idempotently wraps window.fetch. For each fetch call,
 *       finds all elements with data-loading-target="<pattern>"
 *       where <pattern> matches the URL (substring or simple
 *       glob `*`), and calls start/stop on each.
 *     - Multiple fetches against the same target are
 *       reference-counted via start/stop, so concurrent
 *       requests collapse into one overlay.
 *
 * Match semantics for data-loading-target:
 *   - "*" matches any URL
 *   - exact prefix "/api/foo" matches URLs containing it
 *   - "/api/STAR/bar" supports STAR (asterisk) wildcards,
 *     where each asterisk is replaced with the regex ".*"
 *     at match time. Bound elements get start()/stop() called.
 *
 * Coordination:
 *   - This file is NEW; window.PFM is created lazily.
 *   - Does NOT modify web/index.html (mounting is the
 *     index-html-owner's responsibility).
 *
 * Accessibility:
 *   - Overlay node gets role="status" and aria-live="polite".
 *   - Reduced motion is handled in CSS.
 *
 * No external deps. Plain ES2019.
 * ============================================================ */

(function () {
  "use strict";

  // ---- Namespace ---------------------------------------------------------
  var root = (typeof window !== "undefined") ? window : globalThis;
  root.PFM = root.PFM || {};
  if (root.PFM.loading && root.PFM.loading.__pfm_v) {
    // Already initialized — keep idempotent.
    return;
  }

  var COUNTER_PROP = "__pfmLoadingCount";
  var OVERLAY_CLASS = "pfm-loading__overlay";
  var HOST_CLASS = "pfm-loading";
  var ACTIVE_CLASS = "is-loading";

  // ---- Helpers -----------------------------------------------------------
  function ensureHostBase(el) {
    if (!el || el.nodeType !== 1) return;
    if (!el.classList.contains(HOST_CLASS)) {
      el.classList.add(HOST_CLASS);
    }
  }

  function buildOverlay(label) {
    var ov = document.createElement("div");
    ov.className = OVERLAY_CLASS;
    ov.setAttribute("role", "status");
    ov.setAttribute("aria-live", "polite");

    var spinner = document.createElement("div");
    spinner.className = "pfm-loading__spinner";
    spinner.setAttribute("aria-hidden", "true");

    var labelEl = document.createElement("p");
    labelEl.className = "pfm-loading__label";
    labelEl.textContent = label || "Loading";

    ov.appendChild(spinner);
    ov.appendChild(labelEl);
    return ov;
  }

  function findOverlay(el) {
    return el.querySelector(":scope > ." + OVERLAY_CLASS);
  }

  // ---- Public: start -----------------------------------------------------
  function start(el, label) {
    if (!el || el.nodeType !== 1) return;
    ensureHostBase(el);
    var n = (el[COUNTER_PROP] || 0) + 1;
    el[COUNTER_PROP] = n;
    if (n === 1) {
      el.classList.add(ACTIVE_CLASS);
      el.setAttribute("aria-busy", "true");
      // Ensure host has positioning so absolute overlay anchors.
      var cs = root.getComputedStyle ? root.getComputedStyle(el) : null;
      if (cs && cs.position === "static") {
        el.style.position = "relative";
        el.__pfmLoadingFixedPos = true;
      }
      var ov = buildOverlay(label);
      el.appendChild(ov);
    } else {
      // Update the label if we already have an overlay.
      var existing = findOverlay(el);
      if (existing && label) {
        var lbl = existing.querySelector(".pfm-loading__label");
        if (lbl) lbl.textContent = label;
      }
    }
  }

  // ---- Public: stop ------------------------------------------------------
  function stop(el) {
    if (!el || el.nodeType !== 1) return;
    var n = el[COUNTER_PROP] || 0;
    if (n <= 0) {
      el[COUNTER_PROP] = 0;
      return;
    }
    n -= 1;
    el[COUNTER_PROP] = n;
    if (n === 0) {
      el.classList.remove(ACTIVE_CLASS);
      el.removeAttribute("aria-busy");
      var ov = findOverlay(el);
      if (ov && ov.parentNode === el) {
        el.removeChild(ov);
      }
      if (el.__pfmLoadingFixedPos) {
        el.style.position = "";
        delete el.__pfmLoadingFixedPos;
      }
    }
  }

  // ---- Public: wrap ------------------------------------------------------
  function wrap(el, asyncFn, label) {
    if (typeof asyncFn !== "function") {
      return Promise.reject(new TypeError("wrap: asyncFn must be a function"));
    }
    start(el, label);
    var done = false;
    function release() {
      if (done) return;
      done = true;
      try { stop(el); } catch (_) { /* swallow */ }
    }
    var p;
    try {
      p = Promise.resolve(asyncFn());
    } catch (e) {
      release();
      return Promise.reject(e);
    }
    return p.then(function (v) { release(); return v; },
                  function (e) { release(); throw e; });
  }

  // ---- URL pattern matching ---------------------------------------------
  function urlMatches(pattern, url) {
    if (!pattern) return false;
    if (pattern === "*") return true;
    if (pattern.indexOf("*") === -1) {
      return url.indexOf(pattern) !== -1;
    }
    // Escape regex special chars except `*`, then replace `*` with `.*`.
    var esc = pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&")
                     .replace(/\*/g, ".*");
    try {
      var re = new RegExp(esc);
      return re.test(url);
    } catch (_) {
      return false;
    }
  }

  function targetsFor(url) {
    if (typeof document === "undefined") return [];
    var nodes = document.querySelectorAll("[data-loading-target]");
    var out = [];
    for (var i = 0; i < nodes.length; i++) {
      var node = nodes[i];
      var raw = node.getAttribute("data-loading-target");
      if (!raw) continue;
      // Allow comma-separated patterns.
      var parts = raw.split(",");
      for (var j = 0; j < parts.length; j++) {
        var p = parts[j].trim();
        if (urlMatches(p, url)) {
          out.push(node);
          break;
        }
      }
    }
    return out;
  }

  function urlFromInput(input) {
    if (typeof input === "string") return input;
    if (input && typeof input === "object") {
      if (typeof input.url === "string") return input.url;
      try { return String(input); } catch (_) { return ""; }
    }
    return "";
  }

  function labelFor(node) {
    var l = node.getAttribute("data-loading-label");
    return l || "Loading";
  }

  // ---- Public: autoAttach -----------------------------------------------
  var ATTACHED_FLAG = "__pfmLoadingFetchPatched";

  function autoAttach() {
    if (typeof root.fetch !== "function") return false;
    if (root[ATTACHED_FLAG]) return true; // idempotent
    var orig = root.fetch.bind(root);

    root.fetch = function (input, init) {
      var url = urlFromInput(input);
      var targets = url ? targetsFor(url) : [];
      var labels = [];
      for (var i = 0; i < targets.length; i++) {
        labels.push(labelFor(targets[i]));
        try { start(targets[i], labels[i]); } catch (_) { /* swallow */ }
      }

      function release() {
        for (var k = 0; k < targets.length; k++) {
          try { stop(targets[k]); } catch (_) { /* swallow */ }
        }
      }

      var pr;
      try {
        pr = orig(input, init);
      } catch (e) {
        release();
        throw e;
      }
      return pr.then(function (resp) { release(); return resp; },
                     function (err)  { release(); throw err; });
    };

    root[ATTACHED_FLAG] = true;
    return true;
  }

  // ---- Export -----------------------------------------------------------
  root.PFM.loading = {
    __pfm_v: 1,
    start: start,
    stop: stop,
    wrap: wrap,
    autoAttach: autoAttach,
    // Exposed for tests / debugging.
    _urlMatches: urlMatches,
    _targetsFor: targetsFor
  };
})();
