/* realtime-tickers.js — tiny SSE → DOM pulse-update helper.
 *
 * Exposes `window.pfmRealtime` with:
 *   - attach({ selector, endpoint, parse, onError? }) → EventSource
 *   - detachAll()
 *
 * Designed against the backend SSE surface documented in
 * docs/sse_inventory.md. The host page (web/index.html) defines the CSS
 * variables `--pos-soft` / `--neg-soft` that the injected stylesheet
 * references, so this works in both light and dark mode automatically.
 *
 * Pure browser APIs only — no imports, no bundler required. Drop the
 * <script src="realtime-tickers.js" defer></script> tag in index.html.
 */
(function () {
  "use strict";

  if (window.pfmRealtime) return; // idempotent — load once

  // ---- one-time CSS injection ---------------------------------------------
  function injectStyle() {
    if (document.getElementById("pfm-rt-style")) return;
    var style = document.createElement("style");
    style.id = "pfm-rt-style";
    style.textContent = [
      "@keyframes rt-flash-up { 0% { background: var(--pos-soft, rgba(22,163,74,0.10)); } 100% { background: transparent; } }",
      "@keyframes rt-flash-down { 0% { background: var(--neg-soft, rgba(220,38,38,0.10)); } 100% { background: transparent; } }",
      ".rt-pulse-up { animation: rt-flash-up 350ms ease-out; }",
      ".rt-pulse-down { animation: rt-flash-down 350ms ease-out; }",
      "[data-rt] { font-variant-numeric: tabular-nums; transition: background 350ms; }",
    ].join("\n");
    document.head.appendChild(style);
  }

  // ---- subscription registry (so detachAll can close them all) -----------
  var handles = []; // [{ es, selector }]

  // ---- helpers -----------------------------------------------------------
  function setPulse(el, prev, next) {
    if (prev == null || next == null) return;
    var p = Number(prev);
    var n = Number(next);
    if (!isFinite(p) || !isFinite(n) || p === n) return;
    var cls = n > p ? "rt-pulse-up" : "rt-pulse-down";
    var other = n > p ? "rt-pulse-down" : "rt-pulse-up";
    el.classList.remove(other);
    // Re-trigger the animation: remove class, force reflow, re-add.
    el.classList.remove(cls);
    void el.offsetWidth; // reflow
    el.classList.add(cls);
  }

  function render(el, parsed) {
    if (parsed == null) return;
    var prev = el.getAttribute("data-rt-prev");
    var value = parsed.value;
    if (value == null) return;
    el.textContent = String(value);
    if (!el.hasAttribute("data-rt")) el.setAttribute("data-rt", "");
    if (parsed.color === "pos") {
      el.classList.add("rt-pulse-up");
      el.classList.remove("rt-pulse-down");
    } else if (parsed.color === "neg") {
      el.classList.add("rt-pulse-down");
      el.classList.remove("rt-pulse-up");
    } else {
      setPulse(el, prev, value);
    }
    el.setAttribute("data-rt-prev", String(value));
  }

  // ---- public API --------------------------------------------------------
  function attach(opts) {
    if (!opts || typeof opts !== "object") {
      throw new Error("pfmRealtime.attach: options object required");
    }
    var selector = opts.selector;
    var endpoint = opts.endpoint;
    var parse = opts.parse;
    var onError = opts.onError;
    if (typeof selector !== "string" || !selector) {
      throw new Error("pfmRealtime.attach: selector (string) required");
    }
    if (typeof endpoint !== "string" || !endpoint) {
      throw new Error("pfmRealtime.attach: endpoint (string) required");
    }
    if (typeof parse !== "function") {
      throw new Error("pfmRealtime.attach: parse (function) required");
    }
    injectStyle();

    var el = document.querySelector(selector);
    if (!el) {
      // Defer element-not-found to console; still open the stream so the
      // caller can attach the element later and see updates on re-render.
      console.warn("pfmRealtime.attach: no element matched", selector);
    }

    var es = new EventSource(endpoint);

    function handle(ev) {
      var target = document.querySelector(selector); // re-query in case DOM re-rendered
      if (!target) return;
      var parsed;
      try {
        parsed = parse(ev);
      } catch (err) {
        console.error("pfmRealtime.parse threw", err);
        return;
      }
      render(target, parsed);
    }

    // The terminal SSE multiplexer emits named events ("tick", "book",
    // "tape", "ready", "hb", "bye"). Browser EventSource only delivers
    // named events to addEventListener(name, ...), not to .onmessage.
    // We register a generic listener for the common payload types and
    // also fall through onmessage for the legacy `/terminal/live-stream`
    // shape which uses `event: tick` too.
    ["tick", "book", "tape", "message"].forEach(function (name) {
      es.addEventListener(name, handle);
    });

    es.onerror = function (err) {
      if (typeof onError === "function") {
        try { onError(err); } catch (e) { console.error(e); }
      } else {
        // EventSource auto-reconnects; just log.
        console.warn("pfmRealtime SSE error on", endpoint, err);
      }
    };

    handles.push({ es: es, selector: selector });
    return es;
  }

  function detachAll() {
    while (handles.length) {
      var h = handles.pop();
      try { h.es.close(); } catch (e) { /* ignore */ }
    }
  }

  window.pfmRealtime = { attach: attach, detachAll: detachAll };
})();
