/* ============================================================
 * results-toast.js  (W11-07, wave-11)
 *
 * Off-screen fit-complete toast.
 *
 * When the regression /fit call finishes while the result panel
 * (#results) is scrolled out of view, surface a small toast in
 * the bottom-right corner so the user notices the new results:
 *
 *     "Fit complete · R² 0.41 · Click to view"
 *
 * Public API:
 *   window.PFM.toast = {
 *     fitComplete(result) -> id | null   // show fit-complete toast
 *     info(msg, opts)     -> id          // generic info toast
 *     dismiss(id)         -> void        // dismiss specific toast
 *   }
 *
 * Inputs:
 *   `result` is the parsed /fit JSON. R² is looked up under
 *   several aliases for resilience:
 *     result.r_squared, result.r2, result.metrics.r_squared,
 *     result.metrics.r2, result.summary.r_squared,
 *     result.fit?.r_squared
 *   If none found, the toast omits the R² segment.
 *
 * Behaviour:
 *   - Listens for `pfm:fit-complete` (CustomEvent dispatched by
 *     W11-02 events-bridge with `detail = { result }`).
 *   - Checks visibility of #results via IntersectionObserver.
 *     If panel is visible (intersectionRatio > 0.1) when the
 *     fit finishes, NO toast is shown — the user is already
 *     looking at it.
 *   - Auto-dismisses after 8s (configurable via opts.ttl).
 *   - Max 1 fit-complete toast at a time; subsequent fit-complete
 *     calls replace any existing fit-complete toast.
 *   - Click toast → smooth-scroll #results into view, focus it.
 *
 * Companion: web/css/results-toast.css.
 *
 * Mount note: file just needs to be loaded once (after DOM ready
 * or with `defer`). No explicit init call required — it self-mounts
 * on first load. Safe to load before #results exists; the observer
 * is attached lazily on the first fitComplete() call.
 * ============================================================ */

(function () {
  "use strict";

  const ROOT_ID = "pfm-rt-root";
  const RESULTS_ID = "results";
  const DEFAULT_TTL_MS = 8000;
  const VISIBILITY_THRESHOLD = 0.1;

  // STORE: id -> { el, kind, timer }
  const STORE = new Map();
  // Track the single active fit-complete toast id (if any).
  let FIT_TOAST_ID = null;

  // Cached visibility state of #results.
  let _resultsVisible = false;
  let _resultsObserver = null;
  let _resultsEl = null;

  let _nextId = 1;
  const _newId = () =>
    "pfm-rt-" + (_nextId++).toString(36) + "-" + Date.now().toString(36);

  /* --------------------------------------------------
   * Root container (lazy)
   * -------------------------------------------------- */
  function ensureRoot() {
    let root = document.getElementById(ROOT_ID);
    if (!root) {
      root = document.createElement("div");
      root.id = ROOT_ID;
      root.className = "pfm-rt-root";
      root.setAttribute("role", "region");
      root.setAttribute("aria-label", "Fit notifications");
      root.setAttribute("aria-live", "polite");
      (document.body || document.documentElement).appendChild(root);
    }
    return root;
  }

  /* --------------------------------------------------
   * Results panel visibility tracking
   * -------------------------------------------------- */
  function attachResultsObserver() {
    if (_resultsObserver) return;
    const el = document.getElementById(RESULTS_ID);
    if (!el) return; // not yet in DOM; will retry on next fitComplete
    _resultsEl = el;
    try {
      _resultsObserver = new IntersectionObserver(
        function (entries) {
          for (const entry of entries) {
            // Visible if any meaningful portion of the panel intersects.
            _resultsVisible = entry.intersectionRatio > VISIBILITY_THRESHOLD;
          }
        },
        { threshold: [0, VISIBILITY_THRESHOLD, 0.5, 1] }
      );
      _resultsObserver.observe(el);
    } catch (_e) {
      // IntersectionObserver unsupported (very old browser). Fall back to
      // a getBoundingClientRect probe per fitComplete call.
      _resultsObserver = null;
    }
  }

  function isResultsVisible() {
    attachResultsObserver();
    if (_resultsObserver) return _resultsVisible;
    // Fallback path
    const el = document.getElementById(RESULTS_ID);
    if (!el) return false;
    const rect = el.getBoundingClientRect();
    const vh = window.innerHeight || document.documentElement.clientHeight;
    const vw = window.innerWidth || document.documentElement.clientWidth;
    if (rect.bottom < 0 || rect.top > vh) return false;
    if (rect.right < 0 || rect.left > vw) return false;
    // Require ~10% of the panel to be inside the viewport.
    const visibleH = Math.min(rect.bottom, vh) - Math.max(rect.top, 0);
    return visibleH / Math.max(rect.height, 1) > VISIBILITY_THRESHOLD;
  }

  /* --------------------------------------------------
   * R² extraction + classification
   * -------------------------------------------------- */
  function extractR2(result) {
    if (!result || typeof result !== "object") return null;
    const candidates = [
      result.r_squared,
      result.r2,
      result.R2,
      result.metrics && result.metrics.r_squared,
      result.metrics && result.metrics.r2,
      result.summary && result.summary.r_squared,
      result.summary && result.summary.r2,
      result.fit && result.fit.r_squared,
      result.fit && result.fit.r2,
    ];
    for (const v of candidates) {
      if (typeof v === "number" && isFinite(v)) return v;
    }
    return null;
  }

  function classifyR2(r2) {
    // Spec: orange ≥ 0.5; gray < 0.2; otherwise mid (neutral ink).
    if (r2 >= 0.5) return "is-strong";
    if (r2 < 0.2) return "is-weak";
    return "is-mid";
  }

  function formatR2(r2) {
    // 2 decimals, no leading 0 stripping (keeps math reading natural).
    return r2.toFixed(2);
  }

  /* --------------------------------------------------
   * Toast DOM builder
   * -------------------------------------------------- */
  function buildToast(id, opts) {
    const o = opts || {};
    const el = document.createElement("div");
    el.id = id;
    el.className = "pfm-rt-toast";
    el.setAttribute("role", "status");
    el.setAttribute("tabindex", "0");
    el.dataset.kind = o.kind || "info";

    const icon = document.createElement("span");
    icon.className = "pfm-rt-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = o.glyph || "▸";

    const body = document.createElement("div");
    body.className = "pfm-rt-body";

    const title = document.createElement("div");
    title.className = "pfm-rt-title";

    // Fit-complete layout: "Fit complete · R² {value}"
    if (o.kind === "fit-complete") {
      const left = document.createElement("span");
      left.textContent = "Fit complete";
      title.appendChild(left);

      if (typeof o.r2 === "number" && isFinite(o.r2)) {
        const sep = document.createElement("span");
        sep.className = "pfm-rt-sep";
        sep.textContent = "·";
        title.appendChild(sep);

        const r2Wrap = document.createElement("span");
        const r2Label = document.createElement("span");
        r2Label.textContent = "R² ";
        const r2Val = document.createElement("span");
        r2Val.className = "pfm-rt-r2 " + classifyR2(o.r2);
        r2Val.textContent = formatR2(o.r2);
        r2Wrap.appendChild(r2Label);
        r2Wrap.appendChild(r2Val);
        title.appendChild(r2Wrap);
      }
    } else {
      title.textContent = o.title || o.message || "Update";
    }

    body.appendChild(title);

    const sub = document.createElement("div");
    sub.className = "pfm-rt-sub";
    sub.textContent = o.subtitle || (o.kind === "fit-complete" ? "Click to view" : "");
    if (sub.textContent) body.appendChild(sub);

    const close = document.createElement("button");
    close.type = "button";
    close.className = "pfm-rt-close";
    close.setAttribute("aria-label", "Dismiss notification");
    close.textContent = "×"; // ×
    close.addEventListener("click", function (ev) {
      ev.stopPropagation();
      dismiss(id);
    });

    el.appendChild(icon);
    el.appendChild(body);
    el.appendChild(close);

    return el;
  }

  /* --------------------------------------------------
   * Mount + animate
   * -------------------------------------------------- */
  function mount(el, ttlMs, onActivate) {
    const root = ensureRoot();
    root.appendChild(el);

    // Trigger enter transition next frame
    requestAnimationFrame(function () {
      el.classList.add("is-enter");
    });

    if (typeof onActivate === "function") {
      el.addEventListener("click", function (ev) {
        // Ignore clicks on the close button (already handled).
        if (ev.target && ev.target.classList && ev.target.classList.contains("pfm-rt-close")) {
          return;
        }
        onActivate(ev);
      });
      el.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          onActivate(ev);
        }
      });
    }

    let timer = null;
    if (typeof ttlMs === "number" && ttlMs > 0) {
      timer = setTimeout(function () {
        dismiss(el.id);
      }, ttlMs);
    }

    return timer;
  }

  function dismiss(id) {
    if (!id) return;
    const rec = STORE.get(id);
    if (!rec) return;
    STORE.delete(id);
    if (FIT_TOAST_ID === id) FIT_TOAST_ID = null;
    if (rec.timer) clearTimeout(rec.timer);
    const el = rec.el;
    if (!el || !el.parentNode) return;
    el.classList.remove("is-enter");
    el.classList.add("is-leave");
    setTimeout(function () {
      if (el && el.parentNode) el.parentNode.removeChild(el);
    }, 220);
  }

  /* --------------------------------------------------
   * Scroll-into-view + focus
   * -------------------------------------------------- */
  function scrollToResults() {
    const el = document.getElementById(RESULTS_ID);
    if (!el) return;
    try {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (_e) {
      el.scrollIntoView();
    }
    // Move focus for screen-reader & keyboard users. Add tabindex if missing.
    if (!el.hasAttribute("tabindex")) {
      el.setAttribute("tabindex", "-1");
    }
    // Defer focus a touch so smooth-scroll has a chance to start.
    setTimeout(function () {
      try { el.focus({ preventScroll: true }); } catch (_e) { el.focus(); }
    }, 40);
  }

  /* --------------------------------------------------
   * Public API
   * -------------------------------------------------- */
  function fitComplete(result) {
    // Attach observer eagerly so the cached state is correct.
    attachResultsObserver();

    // If the results panel is currently visible, no toast needed —
    // the user is already looking at the results area.
    if (isResultsVisible()) {
      return null;
    }

    const r2 = extractR2(result);

    // Replace prior fit-complete toast, if any.
    if (FIT_TOAST_ID) {
      dismiss(FIT_TOAST_ID);
    }

    const id = _newId();
    const el = buildToast(id, {
      kind: "fit-complete",
      r2: r2,
      subtitle: "Click to view",
      glyph: "✓", // ✓
    });

    const timer = mount(el, DEFAULT_TTL_MS, function () {
      scrollToResults();
      dismiss(id);
    });

    STORE.set(id, { el: el, kind: "fit-complete", timer: timer });
    FIT_TOAST_ID = id;
    return id;
  }

  function info(message, opts) {
    const o = opts || {};
    const id = _newId();
    const el = buildToast(id, {
      kind: "info",
      title: message,
      subtitle: o.subtitle || "",
      glyph: o.glyph || "◇", // ◇
    });
    const ttl = typeof o.ttl === "number" ? o.ttl : DEFAULT_TTL_MS;
    const onClick = typeof o.onClick === "function" ? o.onClick : null;
    const timer = mount(el, ttl, function (ev) {
      if (onClick) {
        try { onClick(ev); } catch (_e) { /* swallow */ }
      }
      dismiss(id);
    });
    STORE.set(id, { el: el, kind: "info", timer: timer });
    return id;
  }

  /* --------------------------------------------------
   * Event wiring — listen for pfm:fit-complete
   * -------------------------------------------------- */
  function handleFitCompleteEvent(ev) {
    const detail = ev && ev.detail;
    const result = detail && (detail.result || detail.data || detail);
    fitComplete(result || null);
  }

  // Install listener once.
  window.addEventListener("pfm:fit-complete", handleFitCompleteEvent);

  // Attach observer eagerly if DOM is ready; otherwise wait.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", attachResultsObserver, { once: true });
  } else {
    attachResultsObserver();
  }

  /* --------------------------------------------------
   * Mount on window.PFM.toast
   * -------------------------------------------------- */
  window.PFM = window.PFM || {};
  if (window.PFM.toast && window.PFM.toast.__initialized) {
    // Already mounted — keep first instance authoritative.
    return;
  }
  window.PFM.toast = {
    __initialized: true,
    fitComplete: fitComplete,
    info: info,
    dismiss: dismiss,
  };
})();

/* ============================================================
 * End of results-toast.js
 * ============================================================ */
