/* ============================================================
 * error-banner.js  (T07, wave-10)
 *
 * Friendly error banner system. Calm, helpful — replaces alert().
 *
 * Public API:
 *   window.PFM.errors = {
 *     show(message, opts) -> id
 *     showFromError(err, ctx) -> id
 *     showInline(targetEl, message, opts) -> id
 *     dismiss(id) -> void
 *   }
 *
 *   opts = {
 *     kind:           'error' | 'warn' | 'info'   (default 'error')
 *     title:          string                      (default derived)
 *     traceId:        string                      (renders "Copy trace ID")
 *     action:         { label: string, onClick: fn(banner) }
 *     autoDismissMs:  number | null               (null = sticky)
 *   }
 *
 * Behaviour:
 *   - Stacks at top-right (top:16px right:16px width:360px z:1000)
 *   - Max 3 concurrent banners; oldest auto-dismissed first
 *   - Any banner older than 12s auto-dismisses
 *   - Mobile (<=480px): full-width with 12px margins
 *
 * Pairs with web/css/error-states.css.
 * ============================================================ */

(function () {
  "use strict";

  const ROOT_ID = "pfm-eb-root";
  const MAX_VISIBLE = 3;
  const DEFAULT_AUTO_DISMISS_MS = 12000;
  const STORE = new Map(); // id -> { el, kind, createdAt, timer }
  let _nextId = 1;
  const _newId = () => "pfm-eb-" + (_nextId++).toString(36) + "-" + Date.now().toString(36);

  /* --------------------------------------------------
   * Root container (lazy)
   * -------------------------------------------------- */
  function ensureRoot() {
    let root = document.getElementById(ROOT_ID);
    if (!root) {
      root = document.createElement("div");
      root.id = ROOT_ID;
      root.className = "pfm-eb-root";
      root.setAttribute("role", "region");
      root.setAttribute("aria-label", "Notifications");
      root.setAttribute("aria-live", "polite");
      (document.body || document.documentElement).appendChild(root);
    }
    return root;
  }

  /* --------------------------------------------------
   * Icon glyphs
   * -------------------------------------------------- */
  function iconFor(kind) {
    if (kind === "info") return "◇";   // ◇
    if (kind === "warn") return "⚠";   // ⚠
    return "⚠";                          // ⚠ (error — orange, not red)
  }

  function defaultTitleFor(kind) {
    if (kind === "info") return "Heads up";
    if (kind === "warn") return "Heads up";
    return "Something went wrong";
  }

  /* --------------------------------------------------
   * Build a banner element (NOT yet mounted)
   * -------------------------------------------------- */
  function buildBanner(id, message, opts) {
    const o = opts || {};
    const kind = ["error", "warn", "info"].indexOf(o.kind) >= 0 ? o.kind : "error";
    const title = typeof o.title === "string" && o.title.length ? o.title : defaultTitleFor(kind);

    const wrap = document.createElement("div");
    wrap.className = "pfm-eb-banner";
    wrap.setAttribute("data-kind", kind);
    wrap.setAttribute("data-eb-id", id);
    wrap.setAttribute("role", kind === "error" ? "alert" : "status");

    // Icon
    const icon = document.createElement("span");
    icon.className = "pfm-eb-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = iconFor(kind);
    wrap.appendChild(icon);

    // Body
    const body = document.createElement("div");
    body.className = "pfm-eb-body";
    const titleEl = document.createElement("p");
    titleEl.className = "pfm-eb-title";
    titleEl.textContent = title;
    body.appendChild(titleEl);
    const msgEl = document.createElement("p");
    msgEl.className = "pfm-eb-message";
    msgEl.textContent = message == null ? "" : String(message);
    body.appendChild(msgEl);
    wrap.appendChild(body);

    // Dismiss
    const dismissBtn = document.createElement("button");
    dismissBtn.type = "button";
    dismissBtn.className = "pfm-eb-dismiss";
    dismissBtn.setAttribute("aria-label", "Dismiss notification");
    dismissBtn.textContent = "×"; // ×
    dismissBtn.addEventListener("click", function () {
      api.dismiss(id);
    });
    wrap.appendChild(dismissBtn);

    // Actions row (optional)
    const actions = document.createElement("div");
    actions.className = "pfm-eb-actions";

    if (o.action && typeof o.action.label === "string") {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pfm-eb-action-btn";
      btn.textContent = o.action.label;
      btn.addEventListener("click", function () {
        try {
          if (typeof o.action.onClick === "function") {
            o.action.onClick({ id: id, el: wrap, dismiss: function () { api.dismiss(id); } });
          }
        } catch (e) {
          // swallow — never let an action handler crash the banner
          if (typeof console !== "undefined" && console.error) {
            console.error("[pfm.errors] action onClick threw", e);
          }
        }
      });
      actions.appendChild(btn);
    }

    if (typeof o.traceId === "string" && o.traceId.length) {
      const traceBtn = document.createElement("button");
      traceBtn.type = "button";
      traceBtn.className = "pfm-eb-trace-btn";
      traceBtn.setAttribute("data-trace-id", o.traceId);
      traceBtn.textContent = "Copy trace ID";
      traceBtn.addEventListener("click", function () {
        copyToClipboard(o.traceId).then(function (ok) {
          if (ok) {
            traceBtn.setAttribute("data-copied", "true");
            traceBtn.textContent = "Copied";
            setTimeout(function () {
              traceBtn.removeAttribute("data-copied");
              traceBtn.textContent = "Copy trace ID";
            }, 1800);
          }
        });
      });
      actions.appendChild(traceBtn);
    }

    if (actions.childNodes.length > 0) {
      wrap.appendChild(actions);
    }

    return { wrap: wrap, kind: kind };
  }

  /* --------------------------------------------------
   * Clipboard helper (graceful fallback)
   * -------------------------------------------------- */
  function copyToClipboard(text) {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        return navigator.clipboard.writeText(text).then(function () { return true; }, function () { return legacyCopy(text); });
      }
    } catch (e) { /* fall through */ }
    return Promise.resolve(legacyCopy(text));
  }

  function legacyCopy(text) {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "absolute";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand && document.execCommand("copy");
      document.body.removeChild(ta);
      return !!ok;
    } catch (e) {
      return false;
    }
  }

  /* --------------------------------------------------
   * Stack pruning: cap to MAX_VISIBLE, oldest first
   * -------------------------------------------------- */
  function pruneStack() {
    const root = document.getElementById(ROOT_ID);
    if (!root) return;
    const banners = Array.from(root.querySelectorAll(".pfm-eb-banner:not(.pfm-eb-leaving)"));
    while (banners.length > MAX_VISIBLE) {
      const oldest = banners.shift();
      const oid = oldest.getAttribute("data-eb-id");
      if (oid) api.dismiss(oid);
    }
  }

  /* --------------------------------------------------
   * Specific copy templates for common errors
   * -------------------------------------------------- */
  function deriveErrorCopy(err, ctx) {
    const c = ctx || {};
    const raw = (err && (err.message || err.toString && err.toString())) || "";
    const lower = raw.toLowerCase();

    let status = null;
    if (err && err.response && typeof err.response.status === "number") {
      status = err.response.status;
    } else if (err && typeof err.status === "number") {
      status = err.status;
    } else if (c.status && typeof c.status === "number") {
      status = c.status;
    }

    // Surface-payload detail if FastAPI-shaped
    let detail = "";
    try {
      if (err && err.response && err.response.data) {
        const d = err.response.data;
        if (typeof d === "string") detail = d;
        else if (d.detail) detail = typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail);
      }
    } catch (e) { /* ignore */ }
    const detailLower = (detail || "").toLowerCase();

    // 1. "factor not found"
    if (
      status === 404 ||
      lower.includes("factor not found") ||
      detailLower.includes("factor not found") ||
      detailLower.includes("unknown factor") ||
      detailLower.includes("no such factor")
    ) {
      return {
        kind: "warn",
        title: "Factor not found",
        message: "We couldn't find that factor — try /factors to list available ones.",
      };
    }

    // 2. 502 upstream slow
    if (status === 502 || status === 503 || status === 504 || lower.includes("bad gateway") || lower.includes("gateway timeout")) {
      return {
        kind: "warn",
        title: "Upstream slow",
        message: "Upstream service is slow — retry in a moment.",
      };
    }

    // 3. CORS / network-level connection issue
    const isCors =
      lower.includes("cors") ||
      lower.includes("cross-origin") ||
      lower.includes("blocked by") ||
      lower.includes("network error") ||
      lower.includes("failed to fetch") ||
      (err && err.name === "TypeError" && lower.includes("fetch"));
    if (isCors) {
      return {
        kind: "error",
        title: "Connection issue",
        message: "Connection issue between :8080 and :8000 — check api/config.js.",
      };
    }

    // 4. 422 validation
    if (status === 422) {
      return {
        kind: "warn",
        title: "Request looks off",
        message: detail || "One of the fields didn't validate — check your inputs and try again.",
      };
    }

    // 5. 429 rate-limited
    if (status === 429) {
      return {
        kind: "warn",
        title: "Rate limit",
        message: "You're going a bit fast — give it a few seconds and retry.",
      };
    }

    // 6. 401 / 403
    if (status === 401 || status === 403) {
      return {
        kind: "warn",
        title: "Not authorised",
        message: "This action needs sign-in or a different role.",
      };
    }

    // 7. 5xx generic
    if (status && status >= 500) {
      return {
        kind: "error",
        title: "Server error",
        message: "Server hit an unexpected error — we logged it, please retry.",
      };
    }

    // 8. Abort / cancellation
    if (err && (err.name === "AbortError" || lower.includes("aborted"))) {
      return {
        kind: "info",
        title: "Request cancelled",
        message: "Request was cancelled before completing.",
      };
    }

    // Fallback
    return {
      kind: "error",
      title: "Something went wrong",
      message: detail || raw || "An unexpected error occurred.",
    };
  }

  /* --------------------------------------------------
   * Public API
   * -------------------------------------------------- */
  const api = {
    /**
     * show(message, opts) -> id
     */
    show: function (message, opts) {
      const id = _newId();
      const o = opts || {};
      const root = ensureRoot();
      const built = buildBanner(id, message, o);

      root.appendChild(built.wrap);
      const createdAt = Date.now();
      const autoMs = (o.autoDismissMs === null || o.autoDismissMs === false)
        ? null
        : (typeof o.autoDismissMs === "number" ? o.autoDismissMs : DEFAULT_AUTO_DISMISS_MS);

      let timer = null;
      if (autoMs && autoMs > 0) {
        timer = setTimeout(function () { api.dismiss(id); }, autoMs);
      }

      STORE.set(id, { el: built.wrap, kind: built.kind, createdAt: createdAt, timer: timer });
      pruneStack();
      return id;
    },

    /**
     * showFromError(err, ctx) -> id
     */
    showFromError: function (err, ctx) {
      const c = ctx || {};
      const copy = deriveErrorCopy(err, c);
      const opts = {
        kind: copy.kind,
        title: copy.title,
        traceId: c.traceId || (err && err.response && err.response.headers && err.response.headers["x-request-id"]) || undefined,
        action: c.action,
        autoDismissMs: c.autoDismissMs,
      };
      return api.show(copy.message, opts);
    },

    /**
     * showInline(targetEl, message, opts) -> id
     */
    showInline: function (targetEl, message, opts) {
      if (!targetEl || !targetEl.appendChild) {
        // graceful fallback to top-stack
        return api.show(message, opts);
      }
      const id = _newId();
      const o = opts || {};
      const kind = ["error", "warn", "info"].indexOf(o.kind) >= 0 ? o.kind : "error";
      const title = typeof o.title === "string" && o.title.length ? o.title : defaultTitleFor(kind);

      const wrap = document.createElement("div");
      wrap.className = "pfm-eb-inline";
      wrap.setAttribute("data-kind", kind);
      wrap.setAttribute("data-eb-id", id);
      wrap.setAttribute("role", kind === "error" ? "alert" : "status");

      const icon = document.createElement("span");
      icon.className = "pfm-eb-icon";
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = iconFor(kind);
      wrap.appendChild(icon);

      const body = document.createElement("div");
      body.className = "pfm-eb-body";
      const tEl = document.createElement("p");
      tEl.className = "pfm-eb-title";
      tEl.textContent = title;
      body.appendChild(tEl);
      const mEl = document.createElement("p");
      mEl.className = "pfm-eb-message";
      mEl.textContent = message == null ? "" : String(message);
      body.appendChild(mEl);
      wrap.appendChild(body);

      const dismissBtn = document.createElement("button");
      dismissBtn.type = "button";
      dismissBtn.className = "pfm-eb-dismiss";
      dismissBtn.setAttribute("aria-label", "Dismiss notification");
      dismissBtn.textContent = "×";
      dismissBtn.addEventListener("click", function () { api.dismiss(id); });
      wrap.appendChild(dismissBtn);

      targetEl.appendChild(wrap);
      STORE.set(id, { el: wrap, kind: kind, createdAt: Date.now(), timer: null, inline: true });
      return id;
    },

    /**
     * dismiss(id) -> void
     */
    dismiss: function (id) {
      const entry = STORE.get(id);
      if (!entry) return;
      if (entry.timer) {
        clearTimeout(entry.timer);
        entry.timer = null;
      }
      const el = entry.el;
      if (!el || !el.parentNode) {
        STORE.delete(id);
        return;
      }
      if (entry.inline) {
        // inline: remove without slide animation (would look odd in flow)
        el.parentNode.removeChild(el);
        STORE.delete(id);
        return;
      }
      el.classList.add("pfm-eb-leaving");
      const removeNow = function () {
        if (el.parentNode) el.parentNode.removeChild(el);
        STORE.delete(id);
      };
      // Fallback: even if animationend doesn't fire (e.g. reduced motion), force removal
      const fallback = setTimeout(removeNow, 300);
      el.addEventListener("animationend", function once() {
        clearTimeout(fallback);
        el.removeEventListener("animationend", once);
        removeNow();
      });
    },
  };

  /* --------------------------------------------------
   * Mount on window.PFM.errors
   * -------------------------------------------------- */
  if (typeof window !== "undefined") {
    window.PFM = window.PFM || {};
    window.PFM.errors = api;
  }

})();
