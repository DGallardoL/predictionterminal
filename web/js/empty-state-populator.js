/* ============================================================================
 * empty-state-populator.js — auto-render empty-state copy in panels
 * Owner: W11-60
 * Mount: <script src="/js/empty-state-populator.js" defer></script>
 *
 * Auto-injects refined .empty-state markup (T08 classes) into containers that
 * advertise an "empty-kind" via a data attribute, when their data slot is empty
 * or explicitly marked as empty.
 *
 * Public API:
 *   window.PFM.emptyStates = {
 *     attach(selector, opts),        // wrap a container; auto-shows empty state
 *                                    // when [data-empty="true"] or first child is gone
 *     render(targetEl, kind, opts),  // synchronously inject empty-state markup
 *     registerDataObserver(),        // (idempotent) start MutationObserver on body
 *     COPY,                          // the built-in copy dictionary (frozen)
 *     destroy()                      // tear down the observer (tests only)
 *   }
 *
 * Built-in `kind` → copy mapping (uses microcopy.js voice — tight, second-person):
 *   jumps:no-events       → "No significant jumps in the last 7 days —
 *                           try lowering threshold to 3pp"
 *   jumps:no-news         → "Jump detected but no news matched —
 *                           try widening the time window"
 *   clusters:empty        → "Markets moved independently today —
 *                           no clustering detected"
 *   sentiment:no-data     → "No sentiment signal — VADER+financial-lex
 *                           didn't find scoring keywords"
 *   arb:no-opps           → "No arb opportunities right now —
 *                           markets converged after fees"
 *   factors:filtered-empty → "No factors match this filter — try a broader theme"
 *
 * Mount points (auto-detected via attribute on the host container):
 *   <div data-empty-kind="jumps:no-events"></div>
 *   <div data-empty-kind="clusters:empty" data-empty="true"></div>
 *
 * Trigger logic — a container is considered "empty" and gets the empty-state
 * injected when EITHER:
 *   (a) data-empty="true" is set on it (explicit), OR
 *   (b) it has no element children with data-content (or it has no element
 *       children at all, ignoring text whitespace) AND data-empty != "false".
 *
 * Once injected, the empty-state element carries data-empty-injected="true",
 * so subsequent mutations from app code (e.g. re-render of cards) replace it.
 *
 * The observer is a single MutationObserver attached to <body> watching
 * childList + subtree + the relevant attributes. It is debounced via a single
 * rAF flush to avoid layout thrash when a large widget rerenders.
 * ============================================================================ */

(function () {
  "use strict";

  const ATTR_KIND = "data-empty-kind";
  const ATTR_EMPTY = "data-empty";
  const ATTR_INJECTED = "data-empty-injected";

  /* ---------- Built-in copy dictionary --------------------------------- */

  const COPY = Object.freeze({
    "jumps:no-events": Object.freeze({
      title: "No significant jumps in the last 7 days",
      body:
        "Try lowering the threshold to 3pp to surface softer regime shifts.",
      hint: "GET /terminal/jumps?threshold=0.03",
      action: { label: "Set threshold 3pp", data: "jumps-threshold-3pp" },
      icon: "·",
      compact: false,
    }),
    "jumps:no-news": Object.freeze({
      title: "Jump detected but no news matched",
      body:
        "Try widening the time window — sometimes the trigger headline lands a few hours before or after the price move.",
      hint: "GET /terminal/jumps/{slug}?news_window=24h",
      action: null,
      icon: "·",
      compact: false,
    }),
    "clusters:empty": Object.freeze({
      title: "Markets moved independently today",
      body:
        "No clustering detected — every jump traced to its own driver. Check back after the next macro print.",
      hint: "GET /terminal/jumps/cluster",
      action: null,
      icon: "·",
      compact: false,
    }),
    "sentiment:no-data": Object.freeze({
      title: "No sentiment signal",
      body:
        "VADER + financial-lex didn't find scoring keywords. Try a broader query or a longer window.",
      hint: "GET /terminal/sentiment-leaderboard",
      action: null,
      icon: "·",
      compact: false,
    }),
    "arb:no-opps": Object.freeze({
      title: "No arb opportunities right now",
      body:
        "Markets converged after fees. The scanner is still live — new spreads will surface here.",
      hint: "GET /strategies/arb/stream",
      action: null,
      icon: "·",
      compact: false,
    }),
    "factors:filtered-empty": Object.freeze({
      title: "No factors match this filter",
      body:
        "Try a broader theme, or clear the current filter to browse the full 1228-factor catalog.",
      hint: "e.g. fed-rate-cut-2026q3",
      action: { label: "Clear filter", data: "factors-clear-filter" },
      icon: "·",
      compact: true,
    }),
  });

  /* ---------- Helpers -------------------------------------------------- */

  function _isVisibleElementChild(node) {
    return (
      node.nodeType === 1 &&
      !node.hasAttribute(ATTR_INJECTED)
    );
  }

  function _hasRealContent(el) {
    if (!el) return false;
    const children = el.children;
    for (let i = 0; i < children.length; i++) {
      const c = children[i];
      if (c.hasAttribute && c.hasAttribute(ATTR_INJECTED)) continue;
      if (_isVisibleElementChild(c)) return true;
    }
    return false;
  }

  function _isMarkedEmpty(el) {
    const v = el.getAttribute(ATTR_EMPTY);
    if (v == null) return null; // not set
    return v === "true" || v === "1";
  }

  function _escape(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function _resolveCopy(kind, opts) {
    const base = COPY[kind];
    // Prefer microcopy.js lookup for the title if it has a matching key.
    const lookup =
      (window.PFM && window.PFM.copy && typeof window.PFM.copy.get === "function")
        ? window.PFM.copy.get
        : null;
    const microKey = "empty." + (kind || "").split(":")[0];
    const microTitle = lookup ? lookup(microKey) : undefined;
    const merged = Object.assign(
      {
        title: (base && base.title) || (microTitle || "Nothing here yet"),
        body: (base && base.body) || "",
        hint: (base && base.hint) || "",
        action: (base && base.action) || null,
        icon: (base && base.icon) || "·",
        compact: !!(base && base.compact),
      },
      opts || {}
    );
    return merged;
  }

  /* ---------- Render --------------------------------------------------- */

  /**
   * Inject empty-state markup into `targetEl`. Existing content of the target
   * is preserved UNLESS opts.replace === true, in which case the target is
   * emptied first. The injected node carries [data-empty-injected="true"] so
   * the observer can tell it apart from real app content.
   *
   * @param {Element} targetEl
   * @param {string}  kind     a key in COPY (e.g. "jumps:no-events"), or any
   *                           free-form string (falls back to opts overrides)
   * @param {Object}  [opts]   { title, body, hint, action, icon, compact,
   *                            replace }
   * @returns {Element|null}   the injected .empty-state element
   */
  function render(targetEl, kind, opts) {
    if (!targetEl || targetEl.nodeType !== 1) return null;
    const o = _resolveCopy(kind, opts || {});

    // If we already injected one for this kind, refresh it in-place rather
    // than appending a duplicate.
    const existing = targetEl.querySelector(
      "[" + ATTR_INJECTED + '="true"]'
    );
    if (existing) existing.parentNode.removeChild(existing);

    if (opts && opts.replace) {
      // Drop everything else in the container too.
      while (targetEl.firstChild) targetEl.removeChild(targetEl.firstChild);
    }

    const wrap = document.createElement("div");
    wrap.className = "empty-state" + (o.compact ? " empty-state--compact" : "");
    wrap.setAttribute("role", "status");
    wrap.setAttribute(ATTR_INJECTED, "true");
    wrap.setAttribute("data-empty-state-kind", kind || "");

    const parts = [];
    if (o.icon) {
      parts.push(
        '<div class="empty-state__icon" aria-hidden="true">' +
          _escape(o.icon) +
          "</div>"
      );
    }
    parts.push(
      '<h3 class="empty-state__title">' + _escape(o.title) + "</h3>"
    );
    if (o.body) {
      parts.push(
        '<p class="empty-state__body">' + _escape(o.body) + "</p>"
      );
    }
    if (o.action && o.action.label) {
      const dataAttr = o.action.data
        ? ' data-action="' + _escape(o.action.data) + '"'
        : "";
      parts.push(
        '<button class="empty-state__action" type="button"' +
          dataAttr +
          ">" +
          _escape(o.action.label) +
          "</button>"
      );
    }
    if (o.hint) {
      parts.push(
        '<code class="empty-state__hint">' + _escape(o.hint) + "</code>"
      );
    }
    wrap.innerHTML = parts.join("");

    // Optional click hook
    if (o.action && typeof o.action.onClick === "function") {
      const btn = wrap.querySelector(".empty-state__action");
      if (btn) {
        btn.addEventListener("click", function (ev) {
          try {
            o.action.onClick(ev, wrap, targetEl);
          } catch (_) {
            /* swallow — empty-state callbacks must not throw */
          }
        });
      }
    }

    targetEl.appendChild(wrap);
    return wrap;
  }

  /* ---------- Per-container attach ------------------------------------ */

  /**
   * Find every element matching `selector` (or treat selector as an Element)
   * and ensure each has a data-empty-kind. The element is re-evaluated
   * immediately; the global observer takes over from there.
   *
   * @param {string|Element} selector
   * @param {Object} [opts]    { kind, ...renderOpts }
   * @returns {Element[]}      list of containers wired
   */
  function attach(selector, opts) {
    opts = opts || {};
    let els;
    if (typeof selector === "string") {
      els = Array.prototype.slice.call(document.querySelectorAll(selector));
    } else if (selector && selector.nodeType === 1) {
      els = [selector];
    } else {
      return [];
    }
    for (const el of els) {
      if (opts.kind && !el.getAttribute(ATTR_KIND)) {
        el.setAttribute(ATTR_KIND, opts.kind);
      }
      _evaluate(el, opts);
    }
    return els;
  }

  function _evaluate(el, opts) {
    if (!el || el.nodeType !== 1) return;
    const kind = el.getAttribute(ATTR_KIND);
    if (!kind) return;
    const marked = _isMarkedEmpty(el);
    const hasContent = _hasRealContent(el);
    let shouldShow;
    if (marked === true) shouldShow = true;
    else if (marked === false) shouldShow = false;
    else shouldShow = !hasContent;

    const existing = el.querySelector("[" + ATTR_INJECTED + '="true"]');
    if (shouldShow) {
      if (!existing) {
        render(el, kind, opts || {});
      }
    } else if (existing) {
      // Real content arrived — drop the empty state.
      existing.parentNode.removeChild(existing);
    }
  }

  /* ---------- Global MutationObserver --------------------------------- */

  const OBSERVER_STATE = {
    obs: null,
    pendingFlush: false,
    dirtySet: null, // Set<Element>
    started: false,
  };

  function _flush() {
    OBSERVER_STATE.pendingFlush = false;
    const dirty = OBSERVER_STATE.dirtySet;
    OBSERVER_STATE.dirtySet = null;
    if (!dirty) return;
    dirty.forEach(function (el) {
      if (el && el.isConnected) _evaluate(el);
    });
  }

  function _scheduleFlush() {
    if (OBSERVER_STATE.pendingFlush) return;
    OBSERVER_STATE.pendingFlush = true;
    if (typeof window.requestAnimationFrame === "function") {
      window.requestAnimationFrame(_flush);
    } else {
      setTimeout(_flush, 16);
    }
  }

  function _markDirty(el) {
    if (!el || el.nodeType !== 1) return;
    // Walk up to nearest [data-empty-kind] ancestor (most common case:
    // app code inserts/removes inside the container).
    let cur = el;
    while (cur && cur.nodeType === 1) {
      if (cur.hasAttribute && cur.hasAttribute(ATTR_KIND)) break;
      cur = cur.parentNode;
    }
    if (!cur || cur.nodeType !== 1 || !cur.hasAttribute(ATTR_KIND)) return;
    if (!OBSERVER_STATE.dirtySet) OBSERVER_STATE.dirtySet = new Set();
    OBSERVER_STATE.dirtySet.add(cur);
    _scheduleFlush();
  }

  function _onMutations(mutations) {
    for (let i = 0; i < mutations.length; i++) {
      const m = mutations[i];
      if (m.type === "childList") {
        if (m.target) _markDirty(m.target);
        // Newly-added containers themselves need an initial eval.
        for (let j = 0; j < m.addedNodes.length; j++) {
          const node = m.addedNodes[j];
          if (node && node.nodeType === 1) {
            if (node.hasAttribute && node.hasAttribute(ATTR_KIND)) {
              _markDirty(node);
            }
            // Descendants with the attribute (when a bigger panel is
            // injected as a chunk).
            if (typeof node.querySelectorAll === "function") {
              const inner = node.querySelectorAll("[" + ATTR_KIND + "]");
              for (let k = 0; k < inner.length; k++) _markDirty(inner[k]);
            }
          }
        }
      } else if (m.type === "attributes") {
        if (m.target) _markDirty(m.target);
      }
    }
  }

  /**
   * Idempotently start the global MutationObserver. Also performs an initial
   * sweep so existing containers are evaluated right away.
   */
  function registerDataObserver() {
    if (OBSERVER_STATE.started) return;
    if (typeof MutationObserver === "undefined" || !document.body) {
      // Defer until DOM is ready.
      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", registerDataObserver, {
          once: true,
        });
      }
      return;
    }
    const obs = new MutationObserver(_onMutations);
    obs.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: [ATTR_KIND, ATTR_EMPTY],
    });
    OBSERVER_STATE.obs = obs;
    OBSERVER_STATE.started = true;

    // Initial sweep.
    const initial = document.querySelectorAll("[" + ATTR_KIND + "]");
    for (let i = 0; i < initial.length; i++) _evaluate(initial[i]);
  }

  function destroy() {
    if (OBSERVER_STATE.obs) {
      try {
        OBSERVER_STATE.obs.disconnect();
      } catch (_) {
        /* noop */
      }
    }
    OBSERVER_STATE.obs = null;
    OBSERVER_STATE.started = false;
    OBSERVER_STATE.dirtySet = null;
    OBSERVER_STATE.pendingFlush = false;
  }

  /* ---------- Mount on window.PFM.emptyStates ------------------------- */

  const ns = (window.PFM = window.PFM || {});
  ns.emptyStates = Object.freeze({
    attach: attach,
    render: render,
    registerDataObserver: registerDataObserver,
    destroy: destroy,
    COPY: COPY,
  });

  // Auto-start on script load — safe to call multiple times.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", registerDataObserver, {
      once: true,
    });
  } else {
    registerDataObserver();
  }
})();
