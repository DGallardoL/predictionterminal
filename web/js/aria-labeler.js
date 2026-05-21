/* ============================================================
 * aria-labeler.js  (W13-32, wave-13)
 *
 * Automatic ARIA labelling for screen-reader compatibility.
 *
 * Runs on DOMContentLoaded and again on every DOM mutation
 * (MutationObserver) so dynamically inserted nodes are covered.
 *
 * Behaviours:
 *   - Icon-only buttons        -> aria-label inferred from
 *                                 text glyph, data-action, name,
 *                                 title, or first matching CSS class.
 *   - Charts (Plotly / svg)    -> role="img" + aria-label = chart
 *                                 title (or container heading).
 *   - Loading spinners         -> role="status" + aria-label="Loading"
 *   - Error containers         -> role="alert"
 *   - Tab containers           -> role="tablist", children role="tab"
 *   - Modals / dialogs         -> role="dialog" + aria-modal="true"
 *   - Form inputs              -> ensure <label> linkage or aria-label
 *
 * Public API:
 *   window.PFM.aria.audit()  -> Array<{element, issue, suggestion}>
 *   window.PFM.aria.fix()    -> number of fixes applied
 *   window.PFM.aria.watch()  -> starts MutationObserver (idempotent)
 *
 * Loading instructions (for index-html-owner):
 *   <script defer src="js/aria-labeler.js"></script>
 *   No DOM mount required. Module self-activates.
 *
 * Notes:
 *   - Never overwrites an existing aria-label, aria-labelledby,
 *     or role attribute. Read-only when authored values exist.
 *   - Adds the data-pfm-aria="auto" attribute to any element we
 *     touched so authors can spot auto-injection.
 * ============================================================ */

(function () {
  "use strict";

  if (typeof window === "undefined" || typeof document === "undefined") {
    return;
  }

  window.PFM = window.PFM || {};
  if (window.PFM.aria && window.PFM.aria.__loaded) {
    // already initialised — keep first instance, do nothing.
    return;
  }

  // --------------------------------------------------------
  // Dictionary: glyph / token -> human label
  // --------------------------------------------------------
  // Keys are matched (case-insensitive) against trimmed
  // textContent first, then against data-action / className /
  // id / name attributes (kebab + underscore + space tolerant).
  const GLYPH_LABELS = Object.freeze({
    "⌘k": "Open command palette",
    "⌘ k": "Open command palette",
    "cmd+k": "Open command palette",
    "cmdk": "Open command palette",
    "⚙": "Settings",
    "⚙️": "Settings",
    "gear": "Settings",
    "settings": "Settings",
    "✕": "Close",
    "×": "Close",
    "x": "Close",
    "close": "Close",
    "dismiss": "Close",
    "▼": "Expand",
    "▾": "Expand",
    "v": "Expand",
    "expand": "Expand",
    "▲": "Collapse",
    "▴": "Collapse",
    "collapse": "Collapse",
    "🌙": "Toggle dark mode",
    "☀": "Toggle dark mode",
    "☀️": "Toggle dark mode",
    "moon": "Toggle dark mode",
    "sun": "Toggle dark mode",
    "theme": "Toggle dark mode",
    "dark-mode": "Toggle dark mode",
    "⋯": "More options",
    "…": "More options",
    "...": "More options",
    "more": "More options",
    "menu": "Open menu",
    "≡": "Open menu",
    "☰": "Open menu",
    "hamburger": "Open menu",
    "🔍": "Search",
    "search": "Search",
    "🔔": "Notifications",
    "bell": "Notifications",
    "notifications": "Notifications",
    "+": "Add",
    "add": "Add",
    "plus": "Add",
    "-": "Remove",
    "−": "Remove",
    "remove": "Remove",
    "minus": "Remove",
    "✓": "Confirm",
    "✔": "Confirm",
    "check": "Confirm",
    "confirm": "Confirm",
    "←": "Back",
    "⟵": "Back",
    "back": "Back",
    "→": "Forward",
    "⟶": "Forward",
    "forward": "Forward",
    "↑": "Move up",
    "up": "Move up",
    "↓": "Move down",
    "down": "Move down",
    "↺": "Refresh",
    "↻": "Refresh",
    "refresh": "Refresh",
    "reload": "Refresh",
    "play": "Play",
    "▶": "Play",
    "pause": "Pause",
    "⏸": "Pause",
    "stop": "Stop",
    "⏹": "Stop",
    "copy": "Copy",
    "📋": "Copy",
    "clipboard": "Copy",
    "download": "Download",
    "⬇": "Download",
    "upload": "Upload",
    "⬆": "Upload",
    "share": "Share",
    "edit": "Edit",
    "pencil": "Edit",
    "delete": "Delete",
    "trash": "Delete",
    "🗑": "Delete",
    "info": "More information",
    "?": "Help",
    "help": "Help",
    "pin": "Pin",
    "📌": "Pin",
    "filter": "Filter",
    "sort": "Sort",
    "export": "Export",
  });

  // --------------------------------------------------------
  // Internal state
  // --------------------------------------------------------
  let observer = null;
  let watching = false;
  let pendingFix = false;

  // --------------------------------------------------------
  // Helpers
  // --------------------------------------------------------
  function norm(s) {
    if (!s) return "";
    return String(s).toLowerCase().trim();
  }

  function tokenize(s) {
    // Split kebab / underscore / camelCase into space-separated.
    if (!s) return "";
    return String(s)
      .replace(/[_-]+/g, " ")
      .replace(/([a-z])([A-Z])/g, "$1 $2")
      .toLowerCase()
      .trim();
  }

  function getAccessibleName(el) {
    if (!el || el.nodeType !== 1) return "";
    const al = el.getAttribute("aria-label");
    if (al && al.trim()) return al.trim();
    const labelledby = el.getAttribute("aria-labelledby");
    if (labelledby) {
      const ids = labelledby.split(/\s+/).filter(Boolean);
      const parts = ids
        .map((id) => {
          const ref = document.getElementById(id);
          return ref ? ref.textContent.trim() : "";
        })
        .filter(Boolean);
      if (parts.length) return parts.join(" ");
    }
    const title = el.getAttribute("title");
    if (title && title.trim()) return title.trim();
    const text = (el.textContent || "").trim();
    return text;
  }

  function looksLikeIcon(text) {
    // Pure-symbol short text or empty
    if (!text) return true;
    if (text.length > 3) return false;
    // Allow short alphabetic if all caps with <=2 chars (e.g. "X")
    return /^[ -㌀-\p{Extended_Pictographic}\p{Symbol}\p{Punctuation}A-Za-z]{1,3}$/u.test(
      text
    );
  }

  function inferLabelFromTokens(...sources) {
    for (const raw of sources) {
      if (!raw) continue;
      const direct = norm(raw);
      if (GLYPH_LABELS[direct]) return GLYPH_LABELS[direct];
      const tok = tokenize(raw);
      if (GLYPH_LABELS[tok]) return GLYPH_LABELS[tok];
      // Try every whitespace-separated word.
      const parts = tok.split(/\s+/).filter(Boolean);
      for (const part of parts) {
        if (GLYPH_LABELS[part]) return GLYPH_LABELS[part];
      }
    }
    return "";
  }

  function inferIconButtonLabel(btn) {
    const txt = (btn.textContent || "").trim();
    const dataAction = btn.getAttribute("data-action") || "";
    const dataLabel = btn.getAttribute("data-label") || "";
    const idAttr = btn.id || "";
    const nameAttr = btn.getAttribute("name") || "";
    const titleAttr = btn.getAttribute("title") || "";
    const classAttr = btn.className || "";
    const ariaController = btn.getAttribute("aria-controls") || "";

    // direct title is the highest-quality hint
    if (titleAttr.trim()) return titleAttr.trim();

    const fromDict = inferLabelFromTokens(
      txt,
      dataAction,
      dataLabel,
      idAttr,
      nameAttr,
      classAttr,
      ariaController
    );
    if (fromDict) return fromDict;

    // Inner <img alt="…"> or <svg><title>…</title></svg>?
    const img = btn.querySelector("img[alt]");
    if (img && img.getAttribute("alt").trim()) {
      return img.getAttribute("alt").trim();
    }
    const svgTitle = btn.querySelector("svg > title");
    if (svgTitle && svgTitle.textContent.trim()) {
      return svgTitle.textContent.trim();
    }

    // Fallback: humanise the first non-utility class name.
    const utility = new Set([
      "btn",
      "button",
      "icon",
      "icon-btn",
      "iconbtn",
      "fa",
      "fas",
      "far",
      "fab",
      "material-icons",
      "small",
      "lg",
      "sm",
      "primary",
      "secondary",
      "ghost",
      "muted",
      "active",
      "disabled",
      "pfm-btn",
    ]);
    const classes = classAttr
      .split(/\s+/)
      .map((c) => c.trim())
      .filter((c) => c && !utility.has(c.toLowerCase()));
    if (classes.length) {
      const t = tokenize(classes[0]);
      if (t) return t.replace(/\b\w/g, (m) => m.toUpperCase());
    }

    // Last resort: humanise data-action / id directly.
    const fallbackSrc = dataAction || nameAttr || idAttr;
    if (fallbackSrc) {
      const t = tokenize(fallbackSrc);
      if (t) return t.replace(/\b\w/g, (m) => m.toUpperCase());
    }

    return "";
  }

  function markAuto(el) {
    try {
      el.setAttribute("data-pfm-aria", "auto");
    } catch (_e) {
      // ignore
    }
  }

  // --------------------------------------------------------
  // Element scanners (each returns array of issues, optionally
  // mutating when `apply` is true)
  // --------------------------------------------------------
  function scanIconButtons(root, apply) {
    const issues = [];
    const sel =
      'button, [role="button"], a.btn, a.icon, a.icon-btn, .icon-btn, .icon-button';
    const els = root.querySelectorAll(sel);
    els.forEach((btn) => {
      if (btn.hasAttribute("aria-label") || btn.hasAttribute("aria-labelledby")) {
        return;
      }
      const text = (btn.textContent || "").trim();
      if (!looksLikeIcon(text)) return; // already has visible text
      // Skip buttons that contain a labelled child (e.g., span with text)
      const visibleChildText = Array.from(btn.children)
        .map((c) => (c.textContent || "").trim())
        .filter((t) => t && !looksLikeIcon(t))
        .join(" ");
      if (visibleChildText.length > 0) return;

      const label = inferIconButtonLabel(btn);
      if (!label) {
        issues.push({
          element: btn,
          issue: "icon-button-no-label",
          suggestion: "Add aria-label or visible text",
        });
        return;
      }
      issues.push({
        element: btn,
        issue: "icon-button-missing-aria-label",
        suggestion: `aria-label="${label}"`,
      });
      if (apply) {
        btn.setAttribute("aria-label", label);
        markAuto(btn);
      }
    });
    return issues;
  }

  function scanCharts(root, apply) {
    const issues = [];
    // Plotly + bare SVG charts.
    const sel =
      '.js-plotly-plot, [data-chart], [data-plot], .chart, .pfm-chart, svg.chart';
    const els = root.querySelectorAll(sel);
    els.forEach((el) => {
      if (el.hasAttribute("aria-label") || el.hasAttribute("aria-labelledby")) {
        return;
      }
      // Look for a title: data-chart-title, a sibling/parent heading, or
      // an inner <text class="gtitle"> from Plotly.
      let title = el.getAttribute("data-chart-title") || "";
      if (!title) {
        const plotlyTitle = el.querySelector(".gtitle, .infolayer .g-gtitle text");
        if (plotlyTitle) title = plotlyTitle.textContent.trim();
      }
      if (!title) {
        // ascend to closest container with a heading
        let p = el.parentElement;
        let hops = 0;
        while (p && hops < 4 && !title) {
          const h = p.querySelector(":scope > h1, :scope > h2, :scope > h3, :scope > h4, :scope > .panel-title, :scope > .chart-title");
          if (h && h.textContent.trim()) {
            title = h.textContent.trim();
            break;
          }
          p = p.parentElement;
          hops += 1;
        }
      }
      if (!title) title = "Chart";

      issues.push({
        element: el,
        issue: "chart-missing-aria",
        suggestion: `role="img" aria-label="${title}"`,
      });
      if (apply) {
        if (!el.hasAttribute("role")) el.setAttribute("role", "img");
        el.setAttribute("aria-label", title);
        markAuto(el);
      }
    });
    return issues;
  }

  function scanSpinners(root, apply) {
    const issues = [];
    const sel =
      '.spinner, .loader, .loading, .pfm-spinner, [data-loading="true"], [data-spinner]';
    const els = root.querySelectorAll(sel);
    els.forEach((el) => {
      if (el.hasAttribute("role")) return;
      issues.push({
        element: el,
        issue: "spinner-missing-status-role",
        suggestion: 'role="status" aria-label="Loading"',
      });
      if (apply) {
        el.setAttribute("role", "status");
        if (!el.hasAttribute("aria-label")) {
          el.setAttribute("aria-label", "Loading");
        }
        if (!el.hasAttribute("aria-live")) {
          el.setAttribute("aria-live", "polite");
        }
        markAuto(el);
      }
    });
    return issues;
  }

  function scanErrors(root, apply) {
    const issues = [];
    const sel =
      '.error, .error-banner, .alert-error, .pfm-error, [data-error="true"]';
    const els = root.querySelectorAll(sel);
    els.forEach((el) => {
      if (el.hasAttribute("role")) return;
      issues.push({
        element: el,
        issue: "error-missing-alert-role",
        suggestion: 'role="alert"',
      });
      if (apply) {
        el.setAttribute("role", "alert");
        if (!el.hasAttribute("aria-live")) {
          el.setAttribute("aria-live", "assertive");
        }
        markAuto(el);
      }
    });
    return issues;
  }

  function scanTabs(root, apply) {
    const issues = [];
    const sel =
      '.tabs, .tab-list, .tablist, [data-tabs], .pfm-tabs, .nav-tabs';
    const containers = root.querySelectorAll(sel);
    containers.forEach((cnt) => {
      if (!cnt.hasAttribute("role")) {
        issues.push({
          element: cnt,
          issue: "tab-container-missing-role",
          suggestion: 'role="tablist"',
        });
        if (apply) {
          cnt.setAttribute("role", "tablist");
          markAuto(cnt);
        }
      }
      const tabs = cnt.querySelectorAll(
        ':scope > .tab, :scope > [data-tab], :scope > button, :scope > a'
      );
      tabs.forEach((t) => {
        if (!t.hasAttribute("role")) {
          issues.push({
            element: t,
            issue: "tab-missing-role",
            suggestion: 'role="tab"',
          });
          if (apply) {
            t.setAttribute("role", "tab");
            markAuto(t);
          }
        }
        // selected state
        if (apply) {
          const isActive =
            t.classList.contains("active") ||
            t.classList.contains("selected") ||
            t.getAttribute("aria-current") === "true";
          if (!t.hasAttribute("aria-selected")) {
            t.setAttribute("aria-selected", isActive ? "true" : "false");
          }
        }
      });
    });
    return issues;
  }

  function scanModals(root, apply) {
    const issues = [];
    const sel = '.modal, .pfm-modal, [data-modal], .dialog';
    const els = root.querySelectorAll(sel);
    els.forEach((el) => {
      if (!el.hasAttribute("role")) {
        issues.push({
          element: el,
          issue: "modal-missing-dialog-role",
          suggestion: 'role="dialog" aria-modal="true"',
        });
        if (apply) {
          el.setAttribute("role", "dialog");
          markAuto(el);
        }
      }
      if (!el.hasAttribute("aria-modal")) {
        if (apply) el.setAttribute("aria-modal", "true");
      }
      // Promote a heading to aria-labelledby if absent.
      if (apply && !el.hasAttribute("aria-label") && !el.hasAttribute("aria-labelledby")) {
        const heading = el.querySelector("h1, h2, h3, .modal-title, .dialog-title");
        if (heading) {
          if (!heading.id) {
            heading.id =
              "pfm-modal-title-" +
              Math.random().toString(36).slice(2, 8);
          }
          el.setAttribute("aria-labelledby", heading.id);
        }
      }
    });
    return issues;
  }

  function scanForms(root, apply) {
    const issues = [];
    const sel = "input, select, textarea";
    const els = root.querySelectorAll(sel);
    els.forEach((field) => {
      // Skip hidden / submit / button-style inputs.
      const type = (field.getAttribute("type") || "").toLowerCase();
      if (
        type === "hidden" ||
        type === "submit" ||
        type === "button" ||
        type === "reset"
      ) {
        return;
      }
      if (field.hasAttribute("aria-label") || field.hasAttribute("aria-labelledby")) {
        return;
      }
      // Check for <label for=id> or wrapping <label>
      if (field.id) {
        const lbl = document.querySelector(
          'label[for="' + CSS.escape(field.id) + '"]'
        );
        if (lbl && lbl.textContent.trim()) return;
      }
      const wrappingLabel = field.closest("label");
      if (wrappingLabel && wrappingLabel.textContent.trim()) return;

      // Try placeholder / name / id for fallback label.
      const placeholder = field.getAttribute("placeholder") || "";
      const titleAttr = field.getAttribute("title") || "";
      const nameAttr = field.getAttribute("name") || "";
      const idAttr = field.id || "";
      const guess =
        titleAttr.trim() ||
        placeholder.trim() ||
        tokenize(nameAttr) ||
        tokenize(idAttr);
      if (!guess) {
        issues.push({
          element: field,
          issue: "form-field-no-label",
          suggestion: "Associate <label> or add aria-label",
        });
        return;
      }
      const label = guess.replace(/\b\w/g, (m) => m.toUpperCase());
      issues.push({
        element: field,
        issue: "form-field-missing-aria-label",
        suggestion: `aria-label="${label}"`,
      });
      if (apply) {
        field.setAttribute("aria-label", label);
        markAuto(field);
      }
    });
    return issues;
  }

  // --------------------------------------------------------
  // Public API
  // --------------------------------------------------------
  function runAll(root, apply) {
    const r = root || document.body || document.documentElement;
    if (!r) return [];
    return [
      ...scanIconButtons(r, apply),
      ...scanCharts(r, apply),
      ...scanSpinners(r, apply),
      ...scanErrors(r, apply),
      ...scanTabs(r, apply),
      ...scanModals(r, apply),
      ...scanForms(r, apply),
    ];
  }

  function audit(root) {
    return runAll(root, false);
  }

  function fix(root) {
    const issues = runAll(root, true);
    return issues.length;
  }

  function scheduleFix() {
    if (pendingFix) return;
    pendingFix = true;
    const idle =
      window.requestIdleCallback ||
      function (cb) {
        return setTimeout(cb, 100);
      };
    idle(function () {
      pendingFix = false;
      try {
        fix();
      } catch (e) {
        if (window.console && console.warn) {
          console.warn("[pfm:aria] fix failed", e);
        }
      }
    });
  }

  function watch() {
    if (watching) return;
    watching = true;
    if (typeof MutationObserver === "undefined") return;
    observer = new MutationObserver(function (records) {
      // Skip our own attribute writes (data-pfm-aria) to avoid loops.
      for (const r of records) {
        if (
          r.type === "attributes" &&
          (r.attributeName === "aria-label" ||
            r.attributeName === "aria-labelledby" ||
            r.attributeName === "role" ||
            r.attributeName === "aria-modal" ||
            r.attributeName === "aria-selected" ||
            r.attributeName === "aria-live" ||
            r.attributeName === "data-pfm-aria")
        ) {
          continue;
        }
        scheduleFix();
        return;
      }
      // child/subtree changes
      scheduleFix();
    });
    observer.observe(document.documentElement, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: [
        "class",
        "data-action",
        "data-loading",
        "data-error",
        "data-modal",
        "data-tabs",
        "data-chart",
      ],
    });
  }

  window.PFM.aria = {
    __loaded: true,
    audit: audit,
    fix: fix,
    watch: watch,
    GLYPH_LABELS: GLYPH_LABELS,
  };

  // --------------------------------------------------------
  // Auto-activation
  // --------------------------------------------------------
  function boot() {
    try {
      fix();
    } catch (e) {
      if (window.console && console.warn) {
        console.warn("[pfm:aria] initial fix failed", e);
      }
    }
    watch();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    // Already past DOMContentLoaded
    boot();
  }
})();
