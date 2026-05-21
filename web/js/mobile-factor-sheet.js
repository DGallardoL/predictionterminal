/* W12-42 — Mobile factor bottom-sheet
 *
 * Vanilla JS, no deps. Activates only when viewport width < 768px.
 *
 * What it does:
 *   - Detects an existing factor selector (<select> or <input list="...">).
 *   - Inserts a tap-to-open button right before it; hides the original control.
 *   - Tapping the button opens a bottom-sheet (slides up, 80vh max).
 *     The sheet contains:
 *       - Search input wired to window.PFM.factorSearch (W11-09 fuzzy index)
 *       - Scrollable list of factors with theme chips
 *       - "Cancel" / "Select" buttons (44px tap targets)
 *   - Dismiss: tap backdrop, swipe-down the sheet > 80px, or press Esc.
 *   - On Select, fills original input.value (or matches <select> option),
 *     re-shows the original, and dispatches a bubbling 'change' event so any
 *     downstream listeners (e.g. /fit form) react identically to a normal pick.
 *
 * Public API:
 *   window.PFM.mobileFactorSheet = {
 *     attach(selectEl, opts) -> { detach }    // wraps one element
 *     detach()                                 // detaches everything attached
 *     autoAttachAll()                          // scans known selectors
 *   };
 *
 * Mount: <script defer src="/js/mobile-factor-sheet.js"></script>
 *        Plus  <link rel="stylesheet" href="/css/mobile-factor-sheet.css">
 *        Index-html-owner mounts; this module calls autoAttachAll() on load.
 *
 * No-ops cleanly above 768px (the trigger / sheet are display:none and we
 * never hide the original control unless the media query matches at attach).
 */
(function () {
  "use strict";

  var MOBILE_MAX_PX = 767.98;
  var SHEET_SWIPE_DISMISS_PX = 80;
  var ATTACH_FLAG = "__pfmMfsAttached";

  // Selectors we try to auto-attach. Any input/select participating in factor
  // picking on the regression form. Kept conservative — additional inputs can
  // be wrapped explicitly via attach().
  var AUTO_SELECTORS = [
    'select[data-factor-select]',
    'input[data-factor-input]',
    'input[list="factor-suggestions"]',
    'input[list="factor-list"]',
    'select[name="factor"]',
    'select[name="factors"]',
    'input[name="factor"]',
  ];

  // ---------------- utilities ----------------

  function isMobileViewport() {
    if (!window.matchMedia) return window.innerWidth <= 767;
    return window.matchMedia("(max-width: " + MOBILE_MAX_PX + "px)").matches;
  }

  function escapeHtml(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function makeEl(tag, attrs, children) {
    var el = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === "class") el.className = attrs[k];
        else if (k === "text") el.textContent = attrs[k];
        else if (k.indexOf("on") === 0 && typeof attrs[k] === "function") {
          el.addEventListener(k.slice(2), attrs[k]);
        } else if (attrs[k] != null) el.setAttribute(k, attrs[k]);
      });
    }
    if (children) {
      (Array.isArray(children) ? children : [children]).forEach(function (c) {
        if (c == null) return;
        el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
      });
    }
    return el;
  }

  function fireChange(el) {
    try {
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    } catch (e) {
      // Older browsers: legacy fallback
      var ev = document.createEvent("HTMLEvents");
      ev.initEvent("change", true, true);
      el.dispatchEvent(ev);
    }
  }

  // ---------------- factor source ----------------

  /** Resolve a flat factor list from whichever source is available.
   * Preferred: window.PFM.factorSearch (W11-09) exposes its cached list.
   * Fallback: read options of the <select> we're wrapping, or build from
   * the associated <datalist> if the target is an <input list>. */
  function collectFactors(originalEl) {
    var pfm = window.PFM || {};
    if (pfm.factorSearch && typeof pfm.factorSearch.getFactors === "function") {
      try {
        var f = pfm.factorSearch.getFactors();
        if (f && f.length) return f;
      } catch (e) { /* ignore */ }
    }
    if (pfm.factorSearch && pfm.factorSearch._state && pfm.factorSearch._state.factors) {
      // best-effort access if not exposed; tolerated.
      return pfm.factorSearch._state.factors;
    }

    var out = [];
    if (originalEl.tagName === "SELECT") {
      Array.prototype.forEach.call(originalEl.options, function (o) {
        if (!o.value) return;
        out.push({
          slug: o.value,
          label: o.textContent || o.value,
          theme: o.dataset ? (o.dataset.theme || "") : "",
          description: o.title || "",
        });
      });
    } else {
      var listId = originalEl.getAttribute("list");
      if (listId) {
        var dl = document.getElementById(listId);
        if (dl) {
          Array.prototype.forEach.call(dl.options, function (o) {
            if (!o.value) return;
            out.push({
              slug: o.value,
              label: o.label || o.textContent || o.value,
              theme: o.dataset ? (o.dataset.theme || "") : "",
              description: "",
            });
          });
        }
      }
    }
    return out;
  }

  function filterFactors(factors, query) {
    var q = (query || "").trim().toLowerCase();
    if (!q) return factors.slice(0, 200);
    var pfm = window.PFM || {};
    if (pfm.factorSearch && typeof pfm.factorSearch.query === "function") {
      try {
        var ranked = pfm.factorSearch.query(q, { limit: 50 });
        if (ranked && ranked.length) {
          return ranked
            .map(function (r) { return r.factor || r; })
            .filter(Boolean);
        }
      } catch (e) { /* fallthrough to local */ }
    }
    // Local fallback: simple subseq + token match.
    return factors
      .map(function (f) {
        var hay = (f.slug + " " + (f.label || "") + " " + (f.theme || "")).toLowerCase();
        if (hay.indexOf(q) >= 0) return { f: f, score: 2 };
        // subsequence
        var qi = 0;
        for (var i = 0; i < hay.length && qi < q.length; i++) {
          if (hay.charCodeAt(i) === q.charCodeAt(qi)) qi++;
        }
        return qi === q.length ? { f: f, score: 1 } : null;
      })
      .filter(Boolean)
      .sort(function (a, b) { return b.score - a.score; })
      .slice(0, 50)
      .map(function (r) { return r.f; });
  }

  // ---------------- one attachment ----------------

  /** Internal: build, mount, and wire a sheet for one input/select. */
  function Attachment(originalEl, opts) {
    this.original = originalEl;
    this.opts = opts || {};
    this.factors = collectFactors(originalEl);
    this.selectedSlug = null;
    this.activeIndex = -1;
    this.filtered = this.factors.slice(0, 200);

    this._build();
    this._wire();
    if (isMobileViewport()) this._hideOriginal(true);

    originalEl[ATTACH_FLAG] = this;
  }

  Attachment.prototype._build = function () {
    var self = this;
    var current = this._readCurrent();

    // Trigger button — replaces original visually on small screens.
    this.trigger = makeEl(
      "button",
      {
        type: "button",
        class: "pfm-mfs-trigger",
        "aria-haspopup": "dialog",
        "data-empty": current ? "false" : "true",
      },
      [
        makeEl("span", { class: "pfm-mfs-trigger-label", text: current || (this.opts.placeholder || "Select factor…") }),
        makeEl("span", { class: "pfm-mfs-trigger-caret", text: "▾" }),
      ],
    );
    this.original.parentNode.insertBefore(this.trigger, this.original);

    // Portal root (lives at <body> so transforms / fixed positioning are clean).
    this.root = makeEl("div", { class: "pfm-mfs", role: "dialog", "aria-modal": "true", "aria-hidden": "true" });

    this.backdrop = makeEl("div", { class: "pfm-mfs-backdrop" });
    this.sheet = makeEl("div", { class: "pfm-mfs-sheet" });

    var header = makeEl("div", { class: "pfm-mfs-header" }, [
      makeEl("h3", { class: "pfm-mfs-title", text: this.opts.title || "Pick a factor" }),
      makeEl("span", { class: "pfm-mfs-count", text: this.factors.length + " total" }),
    ]);

    this.search = makeEl("input", {
      type: "search",
      class: "pfm-mfs-search",
      placeholder: "Search factors…",
      autocapitalize: "off",
      autocorrect: "off",
      spellcheck: "false",
      "aria-label": "Search factors",
    });

    this.list = makeEl("div", {
      class: "pfm-mfs-list",
      role: "listbox",
      "aria-label": "Factors",
    });

    var cancelBtn = makeEl("button", { type: "button", class: "pfm-mfs-btn", text: "Cancel" });
    this.selectBtn = makeEl("button", {
      type: "button",
      class: "pfm-mfs-btn pfm-mfs-btn--primary",
      text: "Select",
      disabled: "true",
    });
    var actions = makeEl("div", { class: "pfm-mfs-actions" }, [cancelBtn, this.selectBtn]);

    this.sheet.appendChild(header);
    this.sheet.appendChild(this.search);
    this.sheet.appendChild(this.list);
    this.sheet.appendChild(actions);
    this.root.appendChild(this.backdrop);
    this.root.appendChild(this.sheet);
    document.body.appendChild(this.root);

    // Save event refs for detach.
    this._handlers = {
      triggerClick: function () { self.open(); },
      backdropClick: function () { self.close(); },
      cancelClick: function () { self.close(); },
      selectClick: function () { self._commit(); },
      searchInput: function () { self._refresh(self.search.value); },
      keydown: function (e) { self._onKey(e); },
      touchStart: function (e) { self._onTouchStart(e); },
      touchMove: function (e) { self._onTouchMove(e); },
      touchEnd: function () { self._onTouchEnd(); },
      resize: function () { self._onResize(); },
    };

    this.trigger.addEventListener("click", this._handlers.triggerClick);
    this.backdrop.addEventListener("click", this._handlers.backdropClick);
    cancelBtn.addEventListener("click", this._handlers.cancelClick);
    this.selectBtn.addEventListener("click", this._handlers.selectClick);
    this.search.addEventListener("input", this._handlers.searchInput);
    document.addEventListener("keydown", this._handlers.keydown);
    this.sheet.addEventListener("touchstart", this._handlers.touchStart, { passive: true });
    this.sheet.addEventListener("touchmove", this._handlers.touchMove, { passive: true });
    this.sheet.addEventListener("touchend", this._handlers.touchEnd);
    window.addEventListener("resize", this._handlers.resize);

    this._renderRows();
  };

  Attachment.prototype._wire = function () {
    // Track external value changes so the trigger label stays accurate.
    var self = this;
    this._handlers.externalChange = function () {
      self._updateTriggerLabel();
    };
    this.original.addEventListener("change", this._handlers.externalChange);
  };

  Attachment.prototype._readCurrent = function () {
    if (this.original.tagName === "SELECT") {
      return this.original.value || "";
    }
    return this.original.value || "";
  };

  Attachment.prototype._updateTriggerLabel = function () {
    var v = this._readCurrent();
    var lbl = this.trigger.querySelector(".pfm-mfs-trigger-label");
    if (lbl) lbl.textContent = v || (this.opts.placeholder || "Select factor…");
    this.trigger.setAttribute("data-empty", v ? "false" : "true");
  };

  Attachment.prototype._hideOriginal = function (hide) {
    if (hide) this.original.setAttribute("data-pfm-mfs-hidden", "true");
    else this.original.removeAttribute("data-pfm-mfs-hidden");
  };

  Attachment.prototype._renderRows = function () {
    var rows = this.filtered;
    if (!rows.length) {
      this.list.innerHTML =
        '<div class="pfm-mfs-empty">No factors match.</div>';
      this.selectBtn.disabled = true;
      return;
    }
    var html = rows
      .map(function (f, i) {
        var slug = escapeHtml(f.slug || "");
        var label = escapeHtml(f.label || f.slug || "");
        var theme = escapeHtml(f.theme || "");
        var chip = theme
          ? '<span class="pfm-mfs-chip">' + theme + "</span>"
          : "";
        return (
          '<div class="pfm-mfs-row" role="option" aria-selected="false" data-idx="' + i + '" data-slug="' + slug + '">' +
          '<div class="pfm-mfs-row-top">' +
          '<span class="pfm-mfs-label">' + label + "</span>" +
          chip +
          "</div>" +
          '<div class="pfm-mfs-slug">' + slug + "</div>" +
          "</div>"
        );
      })
      .join("");
    this.list.innerHTML = html;
    var self = this;
    Array.prototype.forEach.call(this.list.querySelectorAll(".pfm-mfs-row"), function (row) {
      row.addEventListener("click", function () {
        var idx = parseInt(row.getAttribute("data-idx"), 10);
        self._setActive(idx);
        // Tap once = select; double-confirm via the Select button.
        // On phones we want a single decisive tap to be a select.
        self._commit();
      });
    });
    // First row preselected for keyboard users.
    this._setActive(0);
  };

  Attachment.prototype._setActive = function (i) {
    this.activeIndex = i;
    var rows = this.list.querySelectorAll(".pfm-mfs-row");
    Array.prototype.forEach.call(rows, function (r, idx) {
      var on = idx === i;
      r.classList.toggle("is-active", on);
      r.setAttribute("aria-selected", on ? "true" : "false");
    });
    var f = this.filtered[i];
    this.selectedSlug = f ? f.slug : null;
    this.selectBtn.disabled = !this.selectedSlug;
    if (rows[i] && rows[i].scrollIntoView) {
      rows[i].scrollIntoView({ block: "nearest" });
    }
  };

  Attachment.prototype._refresh = function (q) {
    this.filtered = filterFactors(this.factors, q);
    this._renderRows();
  };

  Attachment.prototype._commit = function () {
    if (!this.selectedSlug) return;
    var orig = this.original;
    if (orig.tagName === "SELECT") {
      // Ensure the option exists; if not, add it transiently.
      var found = false;
      Array.prototype.forEach.call(orig.options, function (o) {
        if (o.value === this.selectedSlug) { o.selected = true; found = true; }
      }, this);
      if (!found) {
        var opt = document.createElement("option");
        opt.value = this.selectedSlug;
        opt.textContent = this.selectedSlug;
        opt.selected = true;
        orig.appendChild(opt);
      }
    } else {
      orig.value = this.selectedSlug;
    }
    this._updateTriggerLabel();
    fireChange(orig);
    if (typeof this.opts.onSelect === "function") {
      try { this.opts.onSelect(this.selectedSlug); } catch (e) { /* user code */ }
    }
    this.close();
  };

  Attachment.prototype.open = function () {
    if (!isMobileViewport()) {
      // On desktop, the trigger shouldn't even be visible; bail safely.
      return;
    }
    this.search.value = "";
    this.filtered = this.factors.slice(0, 200);
    this._renderRows();
    this.root.setAttribute("data-open", "true");
    this.root.setAttribute("aria-hidden", "false");
    // Defer focus so iOS Safari positions caret correctly under slide-in.
    var s = this.search;
    setTimeout(function () { try { s.focus(); } catch (e) { /* noop */ } }, 220);
    document.body.style.overflow = "hidden";
  };

  Attachment.prototype.close = function () {
    this.root.setAttribute("data-open", "false");
    this.root.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
    // reset transient transform after swipe-cancel
    this.sheet.style.transform = "";
  };

  Attachment.prototype._onKey = function (e) {
    if (this.root.getAttribute("data-open") !== "true") return;
    if (e.key === "Escape") { e.preventDefault(); this.close(); }
    else if (e.key === "ArrowDown") {
      e.preventDefault();
      this._setActive(Math.min(this.filtered.length - 1, this.activeIndex + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      this._setActive(Math.max(0, this.activeIndex - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      this._commit();
    }
  };

  // Swipe-down dismiss
  Attachment.prototype._onTouchStart = function (e) {
    if (!e.touches || !e.touches.length) return;
    // Only start drag if user touched near top of sheet (header / handle area).
    var startY = e.touches[0].clientY;
    var rect = this.sheet.getBoundingClientRect();
    if (startY - rect.top > 48) { this._dragStart = null; return; }
    this._dragStart = startY;
    this._dragDelta = 0;
  };
  Attachment.prototype._onTouchMove = function (e) {
    if (this._dragStart == null || !e.touches || !e.touches.length) return;
    var dy = e.touches[0].clientY - this._dragStart;
    if (dy < 0) dy = 0;
    this._dragDelta = dy;
    this.sheet.style.transform = "translateY(" + dy + "px)";
  };
  Attachment.prototype._onTouchEnd = function () {
    if (this._dragStart == null) return;
    var dy = this._dragDelta || 0;
    this._dragStart = null;
    this._dragDelta = 0;
    if (dy > SHEET_SWIPE_DISMISS_PX) {
      this.close();
    } else {
      this.sheet.style.transform = "";
    }
  };

  Attachment.prototype._onResize = function () {
    // If user rotates / resizes across the breakpoint, sync visibility of the
    // original input.
    if (isMobileViewport()) this._hideOriginal(true);
    else { this._hideOriginal(false); this.close(); }
  };

  Attachment.prototype.refreshFactors = function () {
    this.factors = collectFactors(this.original);
    this.filtered = this.factors.slice(0, 200);
    if (this.root.getAttribute("data-open") === "true") this._renderRows();
  };

  Attachment.prototype.detach = function () {
    try {
      this.trigger.removeEventListener("click", this._handlers.triggerClick);
      this.backdrop.removeEventListener("click", this._handlers.backdropClick);
      this.selectBtn.removeEventListener("click", this._handlers.selectClick);
      this.search.removeEventListener("input", this._handlers.searchInput);
      document.removeEventListener("keydown", this._handlers.keydown);
      this.sheet.removeEventListener("touchstart", this._handlers.touchStart);
      this.sheet.removeEventListener("touchmove", this._handlers.touchMove);
      this.sheet.removeEventListener("touchend", this._handlers.touchEnd);
      window.removeEventListener("resize", this._handlers.resize);
      this.original.removeEventListener("change", this._handlers.externalChange);
    } catch (e) { /* tolerate partial init */ }
    if (this.trigger && this.trigger.parentNode) this.trigger.parentNode.removeChild(this.trigger);
    if (this.root && this.root.parentNode) this.root.parentNode.removeChild(this.root);
    this._hideOriginal(false);
    delete this.original[ATTACH_FLAG];
  };

  // ---------------- public namespace ----------------

  var attached = []; // active Attachment instances (module-scoped registry)

  function attach(el, opts) {
    if (!el || typeof el.addEventListener !== "function") return null;
    if (el[ATTACH_FLAG]) return el[ATTACH_FLAG]; // idempotent
    var a = new Attachment(el, opts || {});
    attached.push(a);
    return {
      detach: function () {
        a.detach();
        var i = attached.indexOf(a);
        if (i >= 0) attached.splice(i, 1);
      },
      refresh: function () { a.refreshFactors(); },
    };
  }

  function detachAll() {
    attached.slice().forEach(function (a) { a.detach(); });
    attached.length = 0;
  }

  function autoAttachAll() {
    var seen = new Set();
    AUTO_SELECTORS.forEach(function (sel) {
      var nodes = document.querySelectorAll(sel);
      Array.prototype.forEach.call(nodes, function (n) {
        if (seen.has(n) || n[ATTACH_FLAG]) return;
        seen.add(n);
        attach(n, {});
      });
    });
    return attached.length;
  }

  window.PFM = window.PFM || {};
  window.PFM.mobileFactorSheet = {
    attach: attach,
    detach: detachAll,
    autoAttachAll: autoAttachAll,
    _attached: attached, // for tests
    _isMobileViewport: isMobileViewport,
  };

  // Self-bootstrap once DOM is ready. Safe no-op above 768px (CSS hides
  // everything; we still wire so a later resize-down keeps working).
  function boot() {
    try { autoAttachAll(); } catch (e) { /* never throw at top level */ }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
