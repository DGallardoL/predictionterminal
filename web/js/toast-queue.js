/* ============================================================
 * toast-queue.js  (W12-35, wave-12)
 *
 * Centralised toast queue manager. Coordinates between:
 *   - T07 error-banner.js   (window.PFM.errors)
 *   - W11-07 results-toast  (window.PFM.toast)
 *
 * Design rules:
 *   - Max 3 toasts visible at any time. 4th oldest is evicted
 *     before the new one is added (FIFO).
 *   - Fixed top-right stack with an 8px gap between toasts.
 *   - 8s auto-dismiss per toast (overridable per push).
 *   - Click toast → dismiss. Clicks elsewhere do NOT dismiss.
 *   - Listens for the `pfm:toast` CustomEvent on `window` with
 *     `detail = { message, kind?, autoDismissMs? }`.
 *
 * Public API:
 *   window.PFM.toastQueue = {
 *     push(toast)  -> id            // toast = { message, kind?, autoDismissMs? }
 *     remove(id)   -> void
 *     list()       -> Toast[]       // shallow copies, current order
 *     config(opts) -> currentConfig // { max?, position?, dir? }
 *   }
 *
 * Coexistence:
 *   If `window.PFM.errors` is already mounted by the time this
 *   script runs (T07 has shipped its stylesheet at .pfm-eb-root),
 *   we DEFER to it and do not paint a competing stack. push()
 *   then forwards into PFM.errors.info / PFM.errors.error so
 *   only one visible top-right column ever exists. This keeps the
 *   single-owner styling rule intact.
 *
 *   Otherwise we paint our own minimal stack (inline CSS, <4KB)
 *   so the file is self-contained when loaded standalone.
 *
 * Mount note: load with `defer` after error-banner.js / results-toast.js
 * if both are present, or anywhere if standalone.
 * ============================================================ */

(function () {
  "use strict";

  /* --------------------------------------------------
   * Configuration (mutable via .config())
   * -------------------------------------------------- */
  const CFG = {
    max: 3,
    position: "top-right", // currently only top-right is honoured
    dir: "down",           // "down" stacks newest below; "up" newest above
    defaultAutoDismissMs: 8000,
    gapPx: 8,
    edgePx: 16,
    enterMs: 180,
    leaveMs: 200,
  };

  /* --------------------------------------------------
   * State
   * -------------------------------------------------- */
  // Each record: { id, kind, message, autoDismissMs, el, timer, createdAt }
  const QUEUE = [];
  let _nextId = 1;
  const _newId = () =>
    "pfm-tq-" + (_nextId++).toString(36) + "-" + Date.now().toString(36);

  /* --------------------------------------------------
   * Deferred-mode detection
   *
   * If T07's PFM.errors is mounted we delegate visible-stack
   * rendering to it. We still maintain our own queue so .list()
   * and .remove() work — they just become a thin shim.
   * -------------------------------------------------- */
  function hasErrorsApi() {
    return !!(window.PFM && window.PFM.errors &&
              typeof window.PFM.errors.info === "function");
  }

  /* --------------------------------------------------
   * Inline stylesheet (standalone mode only, < 4KB)
   * Namespaced .pfm-tq-* so it never collides with .pfm-eb-* or .pfm-rt-*.
   * -------------------------------------------------- */
  const STYLE_ID = "pfm-tq-style";
  const CSS_TEXT = [
    ".pfm-tq-root{position:fixed;top:16px;right:16px;z-index:1050;",
    "display:flex;flex-direction:column;gap:8px;width:340px;",
    "max-width:calc(100vw - 32px);pointer-events:none;",
    "font-family:var(--sans,Inter,system-ui,sans-serif);}",
    ".pfm-tq-root.is-dir-up{flex-direction:column-reverse;}",
    ".pfm-tq-toast{pointer-events:auto;box-sizing:border-box;",
    "padding:10px 36px 10px 12px;background:var(--surface,#fff);",
    "color:var(--ink,#0f172a);",
    "border:1px solid var(--hairline,rgba(15,23,42,0.10));",
    "border-radius:8px;box-shadow:0 4px 12px rgba(15,23,42,0.08),",
    "0 1px 2px rgba(15,23,42,0.04);font-size:13px;line-height:1.4;",
    "position:relative;cursor:pointer;user-select:none;",
    "opacity:0;transform:translateX(24px);",
    "transition:opacity 180ms ease-out, transform 200ms ease-out;}",
    ".pfm-tq-toast.is-enter{opacity:1;transform:translateX(0);}",
    ".pfm-tq-toast.is-leave{opacity:0;transform:translateX(24px);",
    "transition:opacity 200ms ease-in, transform 200ms ease-in;}",
    ".pfm-tq-toast[data-kind=error]{",
    "border-left:3px solid var(--danger,#dc2626);}",
    ".pfm-tq-toast[data-kind=warn]{",
    "border-left:3px solid var(--warn,#d97706);}",
    ".pfm-tq-toast[data-kind=success]{",
    "border-left:3px solid var(--success,#059669);}",
    ".pfm-tq-toast[data-kind=info]{",
    "border-left:3px solid var(--orange,#f97316);}",
    ".pfm-tq-msg{display:block;word-break:break-word;}",
    ".pfm-tq-close{position:absolute;top:6px;right:6px;width:22px;",
    "height:22px;border:0;background:transparent;color:inherit;",
    "opacity:0.55;font-size:16px;line-height:1;cursor:pointer;",
    "border-radius:4px;}",
    ".pfm-tq-close:hover{opacity:1;background:rgba(15,23,42,0.06);}",
    ".pfm-tq-close:focus-visible{outline:2px solid var(--orange,#f97316);",
    "outline-offset:1px;opacity:1;}"
  ].join("");

  function injectStyleOnce() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.type = "text/css";
    s.appendChild(document.createTextNode(CSS_TEXT));
    (document.head || document.documentElement).appendChild(s);
  }

  /* --------------------------------------------------
   * Root container
   * -------------------------------------------------- */
  const ROOT_ID = "pfm-tq-root";

  function ensureRoot() {
    let root = document.getElementById(ROOT_ID);
    if (!root) {
      root = document.createElement("div");
      root.id = ROOT_ID;
      root.className = "pfm-tq-root" + (CFG.dir === "up" ? " is-dir-up" : "");
      root.setAttribute("role", "region");
      root.setAttribute("aria-label", "Notifications");
      root.setAttribute("aria-live", "polite");
      // honour position (only top-right styled inline; future-proof hook)
      root.style.top = CFG.edgePx + "px";
      root.style.right = CFG.edgePx + "px";
      root.style.gap = CFG.gapPx + "px";
      (document.body || document.documentElement).appendChild(root);
    } else {
      // sync class if dir changed via .config()
      root.classList.toggle("is-dir-up", CFG.dir === "up");
      root.style.gap = CFG.gapPx + "px";
      root.style.top = CFG.edgePx + "px";
      root.style.right = CFG.edgePx + "px";
    }
    return root;
  }

  /* --------------------------------------------------
   * DOM builder
   * -------------------------------------------------- */
  function buildToastEl(id, kind, message) {
    const el = document.createElement("div");
    el.id = id;
    el.className = "pfm-tq-toast";
    el.dataset.kind = kind || "info";
    el.setAttribute("role", kind === "error" ? "alert" : "status");
    el.setAttribute("tabindex", "0");

    const msg = document.createElement("span");
    msg.className = "pfm-tq-msg";
    msg.textContent = String(message == null ? "" : message);
    el.appendChild(msg);

    const close = document.createElement("button");
    close.type = "button";
    close.className = "pfm-tq-close";
    close.setAttribute("aria-label", "Dismiss notification");
    close.textContent = "×"; // ×
    close.addEventListener("click", function (ev) {
      ev.stopPropagation();
      remove(id);
    });
    el.appendChild(close);

    // Click anywhere on the toast (except close) → dismiss
    el.addEventListener("click", function () { remove(id); });
    // Keyboard activation
    el.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" || ev.key === " " || ev.key === "Escape") {
        ev.preventDefault();
        remove(id);
      }
    });
    return el;
  }

  /* --------------------------------------------------
   * Mount / unmount
   * -------------------------------------------------- */
  function mountStandalone(rec) {
    injectStyleOnce();
    const root = ensureRoot();
    root.appendChild(rec.el);
    requestAnimationFrame(function () {
      rec.el.classList.add("is-enter");
    });
  }

  function unmountStandalone(rec) {
    if (!rec || !rec.el) return;
    const el = rec.el;
    el.classList.remove("is-enter");
    el.classList.add("is-leave");
    setTimeout(function () {
      if (el && el.parentNode) el.parentNode.removeChild(el);
    }, CFG.leaveMs);
  }

  /* --------------------------------------------------
   * FIFO eviction
   * -------------------------------------------------- */
  function evictIfFull() {
    while (QUEUE.length >= CFG.max) {
      const oldest = QUEUE.shift();
      if (!oldest) break;
      if (oldest.timer) clearTimeout(oldest.timer);
      if (oldest._delegated && oldest._delegatedDismiss) {
        try { oldest._delegatedDismiss(); } catch (_e) { /* swallow */ }
      } else {
        unmountStandalone(oldest);
      }
    }
  }

  /* --------------------------------------------------
   * Public — push
   * -------------------------------------------------- */
  function normalise(toast) {
    if (typeof toast === "string") return { message: toast, kind: "info" };
    const t = toast && typeof toast === "object" ? toast : {};
    return {
      message: t.message != null ? t.message : (t.text != null ? t.text : ""),
      kind: t.kind || t.level || "info",
      autoDismissMs: typeof t.autoDismissMs === "number"
        ? t.autoDismissMs
        : (typeof t.ttl === "number" ? t.ttl : CFG.defaultAutoDismissMs),
    };
  }

  function push(toast) {
    const n = normalise(toast);
    if (!n.message) return null;

    const id = _newId();
    const rec = {
      id: id,
      kind: n.kind,
      message: n.message,
      autoDismissMs: n.autoDismissMs,
      el: null,
      timer: null,
      createdAt: Date.now(),
      _delegated: false,
      _delegatedDismiss: null,
    };

    // Evict before adding so the visible count stays <= CFG.max.
    evictIfFull();
    QUEUE.push(rec);

    if (hasErrorsApi()) {
      // Deferred mode: forward to PFM.errors so we don't paint a
      // competing top-right stack. PFM.errors has its own rendering;
      // we just keep a logical record for queue accounting.
      rec._delegated = true;
      try {
        const errsApi = window.PFM.errors;
        const fn = (n.kind === "error" && typeof errsApi.error === "function")
          ? errsApi.error
          : (typeof errsApi.info === "function" ? errsApi.info : null);
        if (fn) {
          const delegatedId = fn.call(errsApi, n.message, {
            ttl: n.autoDismissMs,
          });
          if (delegatedId != null && typeof errsApi.dismiss === "function") {
            rec._delegatedDismiss = function () {
              try { errsApi.dismiss(delegatedId); } catch (_e) { /* swallow */ }
            };
          }
        }
      } catch (_e) {
        // If delegation throws, fall back to standalone render.
        rec._delegated = false;
      }
    }

    if (!rec._delegated) {
      rec.el = buildToastEl(id, rec.kind, rec.message);
      mountStandalone(rec);
    }

    if (typeof rec.autoDismissMs === "number" && rec.autoDismissMs > 0) {
      rec.timer = setTimeout(function () { remove(id); }, rec.autoDismissMs);
    }

    return id;
  }

  /* --------------------------------------------------
   * Public — remove
   * -------------------------------------------------- */
  function remove(id) {
    if (!id) return;
    const idx = QUEUE.findIndex(function (r) { return r.id === id; });
    if (idx === -1) return;
    const rec = QUEUE[idx];
    QUEUE.splice(idx, 1);
    if (rec.timer) { clearTimeout(rec.timer); rec.timer = null; }
    if (rec._delegated) {
      if (typeof rec._delegatedDismiss === "function") {
        try { rec._delegatedDismiss(); } catch (_e) { /* swallow */ }
      }
    } else {
      unmountStandalone(rec);
    }
  }

  /* --------------------------------------------------
   * Public — list
   * -------------------------------------------------- */
  function list() {
    return QUEUE.map(function (r) {
      return {
        id: r.id,
        kind: r.kind,
        message: r.message,
        autoDismissMs: r.autoDismissMs,
        createdAt: r.createdAt,
        delegated: !!r._delegated,
      };
    });
  }

  /* --------------------------------------------------
   * Public — config
   * -------------------------------------------------- */
  function config(opts) {
    if (opts && typeof opts === "object") {
      if (typeof opts.max === "number" && opts.max > 0) {
        CFG.max = Math.floor(opts.max);
        // If the new cap is smaller than the current queue, evict.
        while (QUEUE.length > CFG.max) {
          const oldest = QUEUE.shift();
          if (oldest.timer) clearTimeout(oldest.timer);
          if (oldest._delegated && oldest._delegatedDismiss) {
            try { oldest._delegatedDismiss(); } catch (_e) { /* swallow */ }
          } else {
            unmountStandalone(oldest);
          }
        }
      }
      if (typeof opts.position === "string") {
        CFG.position = opts.position;
        // Only top-right is currently styled; other values are accepted
        // so callers can probe but the root stays top-right.
      }
      if (opts.dir === "up" || opts.dir === "down") {
        CFG.dir = opts.dir;
      }
      if (typeof opts.defaultAutoDismissMs === "number" &&
          opts.defaultAutoDismissMs >= 0) {
        CFG.defaultAutoDismissMs = opts.defaultAutoDismissMs;
      }
      if (typeof opts.gapPx === "number" && opts.gapPx >= 0) {
        CFG.gapPx = opts.gapPx;
      }
      // Re-sync root if it exists already.
      if (!hasErrorsApi() && document.getElementById(ROOT_ID)) {
        ensureRoot();
      }
    }
    return {
      max: CFG.max,
      position: CFG.position,
      dir: CFG.dir,
      defaultAutoDismissMs: CFG.defaultAutoDismissMs,
      gapPx: CFG.gapPx,
    };
  }

  /* --------------------------------------------------
   * Event wiring — listen for `pfm:toast`
   * -------------------------------------------------- */
  function onPfmToast(ev) {
    const detail = ev && ev.detail;
    if (!detail) return;
    push({
      message: detail.message,
      kind: detail.kind,
      autoDismissMs: detail.autoDismissMs,
    });
  }
  window.addEventListener("pfm:toast", onPfmToast);

  /* --------------------------------------------------
   * Mount on window.PFM.toastQueue
   * -------------------------------------------------- */
  window.PFM = window.PFM || {};
  if (window.PFM.toastQueue && window.PFM.toastQueue.__initialized) {
    return; // already mounted by an earlier load — keep first authoritative
  }
  window.PFM.toastQueue = {
    __initialized: true,
    push: push,
    remove: remove,
    list: list,
    config: config,
  };
})();

/* ============================================================
 * End of toast-queue.js
 * ============================================================ */
