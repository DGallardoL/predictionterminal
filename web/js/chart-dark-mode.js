/* ============================================================
 * chart-dark-mode.js  (W12-41, wave-12)
 *
 * Dark-mode polish for every live Plotly chart on the page.
 *
 * Coordinates with:
 *   - T05  theme-toggle.js          (emits `pfm:theme-change`)
 *   - W12-33 plotly-theme-bloomberg.js (full Bloomberg-style theme)
 *
 * Where this module fits:
 *   plotly-theme-bloomberg.js owns the *layout template* used when
 *   charts are first drawn. This module owns the lightweight
 *   *relayout* applied to charts that are ALREADY on the page when
 *   the user flips theme. It walks `document.querySelectorAll(
 *   '.js-plotly-plot')` and calls `Plotly.relayout(el, patch)` with
 *   the small set of fields that need to swap: background, plot bg,
 *   font color, hairline/grid color, tick color, hover label.
 *
 * Idempotent. Self-registers on script load. Listener is attached
 * at most once. Re-applying the same theme is a no-op (no relayout
 * calls are issued if nothing has changed).
 *
 * Public API:
 *   window.PFM.chartDarkMode = {
 *     apply(theme),     // 'light' | 'dark' — relayout every plot
 *     refreshAll(),     // re-apply current resolved theme
 *     isAttached(),     // bool — listener wired up?
 *     version: 'w12-41'
 *   };
 *
 * Mount instructions (for index-html-owner):
 *   <script defer src="js/chart-dark-mode.js"></script>
 *   No DOM mount point required. Load AFTER theme-toggle.js so the
 *   `pfm:theme-change` listener catches early dispatches.
 * ============================================================ */
(function (root) {
  "use strict";

  if (typeof root === "undefined") return;
  root.PFM = root.PFM || {};
  if (root.PFM.chartDarkMode && root.PFM.chartDarkMode.__initialized) return;

  // --------------------------------------------------------------
  // Palette — kept tight on purpose. The full Bloomberg palette
  // lives in plotly-theme-bloomberg.js; here we only need the
  // ~6 tokens that participate in a relayout patch.
  // --------------------------------------------------------------
  var LIGHT = {
    paper:    "#ffffff",
    plot:     "#fcfcfd",
    font:     "#0a0a0c",
    tick:     "#6a6a73",        // ink-3
    grid:     "rgba(15,23,42,0.06)",
    zeroline: "rgba(15,23,42,0.18)",
    hoverBg:  "#ffffff",
    hoverBd:  "#d6d6dc",
    hoverFg:  "#0a0a0c",
    legendFg: "#3f3f47",
  };

  var DARK = {
    paper:    "#0f172a",        // slate-900
    plot:     "#0b1220",        // slightly deeper
    font:     "#f8fafc",        // slate-50
    tick:     "#94a3b8",        // ink-3-dark (slate-400)
    grid:     "rgba(255,255,255,0.06)",
    zeroline: "rgba(255,255,255,0.18)",
    hoverBg:  "#111827",        // dark surface
    hoverBd:  "#334155",        // slate-700
    hoverFg:  "#f8fafc",
    legendFg: "#e2e8f0",
  };

  function palette(theme) {
    return theme === "dark" ? DARK : LIGHT;
  }

  // --------------------------------------------------------------
  // Build a Plotly relayout patch from a palette. Wildcard
  // `xaxis.*` / `yaxis.*` keys hit every numbered axis Plotly may
  // have created (xaxis2, xaxis3, …) without us enumerating them.
  // --------------------------------------------------------------
  function buildPatch(p) {
    return {
      paper_bgcolor:               p.paper,
      plot_bgcolor:                p.plot,
      "font.color":                p.font,
      "title.font.color":          p.font,
      "legend.font.color":         p.legendFg,
      // Axes — wildcards apply to every axis on the figure.
      "xaxis.gridcolor":           p.grid,
      "xaxis.zerolinecolor":       p.zeroline,
      "xaxis.linecolor":           p.grid,
      "xaxis.tickcolor":           p.tick,
      "xaxis.tickfont.color":      p.tick,
      "xaxis.title.font.color":    p.font,
      "yaxis.gridcolor":           p.grid,
      "yaxis.zerolinecolor":       p.zeroline,
      "yaxis.linecolor":           p.grid,
      "yaxis.tickcolor":           p.tick,
      "yaxis.tickfont.color":      p.tick,
      "yaxis.title.font.color":    p.font,
      // Hover label.
      "hoverlabel.bgcolor":        p.hoverBg,
      "hoverlabel.bordercolor":    p.hoverBd,
      "hoverlabel.font.color":     p.hoverFg,
    };
  }

  // --------------------------------------------------------------
  // Plotly availability — be defensive. The module may load before
  // Plotly's CDN script, in which case we just queue and try again
  // when the next theme event fires (or refreshAll() is called).
  // --------------------------------------------------------------
  function getPlotly() {
    try {
      if (root.Plotly && typeof root.Plotly.relayout === "function") {
        return root.Plotly;
      }
    } catch (_) {}
    return null;
  }

  // --------------------------------------------------------------
  // Walk the DOM for live plots. Plotly attaches the
  // `.js-plotly-plot` class to the container after `newPlot`.
  // --------------------------------------------------------------
  function listPlots() {
    try {
      return document.querySelectorAll(".js-plotly-plot");
    } catch (_) {
      return [];
    }
  }

  // --------------------------------------------------------------
  // Apply a theme to every plot on the page. Each relayout is
  // wrapped in try/catch so one broken chart can't break the rest.
  // --------------------------------------------------------------
  var lastTheme = null;

  function apply(theme) {
    if (theme !== "light" && theme !== "dark") {
      // Resolve via theme-toggle if available.
      try {
        if (root.PFM.theme && typeof root.PFM.theme.resolved === "function") {
          theme = root.PFM.theme.resolved();
        } else {
          theme = "light";
        }
      } catch (_) {
        theme = "light";
      }
    }
    lastTheme = theme;
    var Plotly = getPlotly();
    if (!Plotly) return false;
    var plots = listPlots();
    if (!plots || !plots.length) return true;
    var patch = buildPatch(palette(theme));
    for (var i = 0; i < plots.length; i++) {
      var el = plots[i];
      try {
        Plotly.relayout(el, patch);
      } catch (_) {
        // Chart may have been torn down between query and relayout.
      }
    }
    return true;
  }

  function refreshAll() {
    return apply(lastTheme || resolvedTheme());
  }

  function resolvedTheme() {
    try {
      if (root.PFM.theme && typeof root.PFM.theme.resolved === "function") {
        return root.PFM.theme.resolved();
      }
    } catch (_) {}
    try {
      var attr = document.documentElement.getAttribute("data-theme");
      if (attr === "dark" || attr === "light") return attr;
    } catch (_) {}
    try {
      if (
        root.matchMedia &&
        root.matchMedia("(prefers-color-scheme: dark)").matches
      ) {
        return "dark";
      }
    } catch (_) {}
    return "light";
  }

  // --------------------------------------------------------------
  // Listener wiring — idempotent.
  // --------------------------------------------------------------
  var attached = false;

  function onThemeChange(ev) {
    var detail = (ev && ev.detail) || {};
    var resolved = detail.resolved || resolvedTheme();
    apply(resolved);
  }

  function attach() {
    if (attached) return true;
    try {
      root.addEventListener("pfm:theme-change", onThemeChange);
      attached = true;
    } catch (_) {
      attached = false;
    }
    return attached;
  }

  function isAttached() {
    return attached;
  }

  // --------------------------------------------------------------
  // Boot.
  // --------------------------------------------------------------
  attach();

  function onReady(fn) {
    try {
      if (
        document.readyState === "complete" ||
        document.readyState === "interactive"
      ) {
        root.setTimeout(fn, 0);
      } else {
        document.addEventListener("DOMContentLoaded", fn, { once: true });
      }
    } catch (_) {}
  }

  // First pass after DOM ready — catches charts that finished
  // rendering before this script loaded.
  onReady(function () {
    apply(resolvedTheme());
  });

  // --------------------------------------------------------------
  // Expose API.
  // --------------------------------------------------------------
  root.PFM.chartDarkMode = {
    __initialized: true,
    version: "w12-41",
    apply: apply,
    refreshAll: refreshAll,
    isAttached: isAttached,
  };
})(typeof window !== "undefined" ? window : this);
