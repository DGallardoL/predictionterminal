/* plotly-theme-bloomberg.js — W12-33
 * ============================================================================
 * Refined Plotly theme matching a Bloomberg-terminal aesthetic. Designed to
 * LAYER OVER `web/plotly-theme.js` (the legacy "Quantum Terminal" theme).
 *
 *   What this module does:
 *     1. Sets global Plotly config defaults (`Plotly.setPlotConfig`).
 *     2. Exposes a `layoutTemplate()` factory the existing pfmTheme can deep-
 *        merge against (Plotly itself has no "default theme" hook, but
 *        layoutTemplate() returns a fully merged `layout` ready for
 *        `Plotly.newPlot(el, traces, layout, config)`).
 *     3. Exposes `hoverFmt(d, suffix)` — a tight 11px-mono multi-line hover
 *        string with a bullet separator (Bloomberg-style).
 *     4. Subscribes to `pfm:theme-change` (emitted by `web/js/theme-toggle.js`)
 *        to swap the surface/ink palette and re-render all live Plotly graphs
 *        on the page.
 *
 *   Global surface:
 *     window.PFM = window.PFM || {};
 *     window.PFM.plotlyBloomberg = {
 *       apply(),                              // install global defaults + listener
 *       hoverFmt(d, suffix = ''),             // tight 11px mono hover block
 *       layoutTemplate(custom = {}),          // returns merged layout
 *       palette(),                            // current color tokens (live)
 *       config,                               // ready-to-use plot config
 *       version: 'w12-33'
 *     };
 *
 *   Style guidelines (per task brief):
 *     · Background: white in light, slate-900 in dark (via [data-theme="dark"])
 *     · Gridlines: hairline only (Y), no X grid, no axis lines
 *     · Axis titles: 11px Inter UPPERCASE, letter-spacing 0.04em
 *     · Hover label: 12px JetBrains Mono, surface bg, hairline border
 *     · Custom hover format: 4-line block with • bullet separators
 *     · Color palette: orange primary, then teal / violet / yellow secondaries
 *     · Series stroke width: 1.5px
 *     · Annotation arrows: hairline (width 1)
 *
 *   Idempotent: calling apply() repeatedly is safe; the event listener is
 *   attached at most once.
 * ============================================================================
 */
(function (root) {
  "use strict";

  root.PFM = root.PFM || {};

  // ──────────────────────────────────────────────────────────────────────────
  // Fonts — keep aligned with terminal-premium.css and existing pfmTheme
  // ──────────────────────────────────────────────────────────────────────────
  var FONT_SANS =
    "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif";
  var FONT_MONO =
    "'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace";

  // ──────────────────────────────────────────────────────────────────────────
  // Palettes — light & dark. Order of secondaries: teal → violet → yellow.
  // Primary = orange (Bloomberg amber-style accent).
  // ──────────────────────────────────────────────────────────────────────────
  var LIGHT = {
    bg:        "#ffffff",
    surface:   "#fcfcfd",
    ink:       "#0a0a0c",
    ink2:      "#3f3f47",
    ink3:      "#6a6a73",
    ink4:      "#9c9ca5",
    hairline:  "#ececef",
    hairline2: "#d6d6dc",
    // Series — orange primary, teal/violet/yellow secondaries
    primary:   "#ea580c",   // orange-600
    teal:      "#0d9488",   // teal-600
    violet:    "#7c3aed",   // violet-600
    yellow:    "#ca8a04",   // yellow-600
    // Semantic
    positive:  "#16a34a",
    negative:  "#dc2626",
    neutral:   "#6a6a73",
  };

  var DARK = {
    // slate-900 surface per brief
    bg:        "#0f172a",   // slate-900
    surface:   "#0b1220",   // slightly deeper for plot bg
    ink:       "#f8fafc",   // slate-50
    ink2:      "#e2e8f0",   // slate-200
    ink3:      "#94a3b8",   // slate-400
    ink4:      "#64748b",   // slate-500
    hairline:  "#1e293b",   // slate-800
    hairline2: "#334155",   // slate-700
    primary:   "#fb923c",   // orange-400 — pops on slate
    teal:      "#2dd4bf",   // teal-400
    violet:    "#a78bfa",   // violet-400
    yellow:    "#facc15",   // yellow-400
    positive:  "#4ade80",
    negative:  "#f87171",
    neutral:   "#94a3b8",
  };

  function isDark() {
    try {
      var attr = document.documentElement.getAttribute("data-theme");
      return attr === "dark";
    } catch (_e) {
      return false;
    }
  }

  function palette() {
    return isDark() ? DARK : LIGHT;
  }

  // Sequence used by Plotly's colorway (matches task brief order).
  function colorway(p) {
    return [p.primary, p.teal, p.violet, p.yellow, p.positive, p.negative];
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Axis defaults — hairline grid only on Y; no X grid; no spines; outside
  // ticks; axis titles in 11px UPPERCASE Inter with letter-spacing.
  // ──────────────────────────────────────────────────────────────────────────
  function axisTitleFont(p) {
    return {
      family: FONT_SANS,
      size:   11,
      color:  p.ink3,
    };
  }

  function buildXAxis(p) {
    return {
      showgrid:   false,
      zeroline:   false,
      showline:   false,
      ticks:      "outside",
      tickcolor:  p.hairline2,
      ticklen:    4,
      tickwidth:  1,
      tickfont:   { family: FONT_SANS, size: 10.5, color: p.ink3 },
      // Plotly doesn't honor CSS text-transform; we expose `titleUppercase()`
      // on PFM.plotlyBloomberg for callers to UPPERCASE their title strings.
      titlefont:  axisTitleFont(p),
      automargin: true,
    };
  }

  function buildYAxis(p) {
    return {
      showgrid:   true,
      gridcolor:  p.hairline,
      gridwidth:  1,
      zeroline:   false,
      showline:   false,
      ticks:      "outside",
      tickcolor:  p.hairline2,
      ticklen:    4,
      tickwidth:  1,
      tickfont:   { family: FONT_MONO, size: 10.5, color: p.ink3 },
      titlefont:  axisTitleFont(p),
      automargin: true,
    };
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Layout template
  // ──────────────────────────────────────────────────────────────────────────
  function buildLayout(p) {
    return {
      paper_bgcolor: p.bg,
      plot_bgcolor:  p.surface,
      font: {
        family: FONT_SANS,
        size:   11,
        color:  p.ink2,
      },
      colorway:  colorway(p),
      margin:    { l: 52, r: 18, t: 12, b: 36 },
      autosize:  true,
      xaxis:     buildXAxis(p),
      yaxis:     buildYAxis(p),
      hovermode: "x unified",
      dragmode:  false,
      showlegend: false,
      hoverlabel: {
        bgcolor:     p.surface,
        bordercolor: p.hairline2,
        align:       "left",
        font: {
          family: FONT_MONO,
          size:   12,
          color:  p.ink,
        },
      },
      // Annotations defaults — hairline arrows per brief.
      annotationdefaults: {
        arrowwidth: 1,
        arrowcolor: p.ink3,
        arrowhead:  3,
        font: { family: FONT_SANS, size: 11, color: p.ink2 },
      },
    };
  }

  // Deep-merge helper (objects over primitives; right wins).
  function isPlainObject(v) {
    return v !== null && typeof v === "object" && !Array.isArray(v);
  }
  function deepMerge(target, source) {
    if (!isPlainObject(source)) return source;
    var out = isPlainObject(target) ? Object.assign({}, target) : {};
    Object.keys(source).forEach(function (k) {
      var sv = source[k];
      var tv = out[k];
      if (isPlainObject(sv) && isPlainObject(tv)) {
        out[k] = deepMerge(tv, sv);
      } else if (isPlainObject(sv)) {
        out[k] = deepMerge({}, sv);
      } else {
        out[k] = sv;
      }
    });
    return out;
  }

  function layoutTemplate(custom) {
    return deepMerge(buildLayout(palette()), custom || {});
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Plotly config — modebar on hover, no logo, curated buttons.
  // ──────────────────────────────────────────────────────────────────────────
  var config = {
    displayModeBar: "hover",
    displaylogo:    false,
    responsive:     true,
    scrollZoom:     false,
    showTips:       false,
    doubleClick:    "reset",
    modeBarButtonsToRemove: [
      "lasso2d",
      "select2d",
      "autoScale2d",
    ],
  };

  // ──────────────────────────────────────────────────────────────────────────
  // hoverFmt — tight 11px-mono 4-line block with • bullet separators.
  //
  //   Input shape `d` (all keys optional — missing rows are skipped):
  //     {
  //       title:   string,           // line 1, bold
  //       date:    string|Date,      // line 2
  //       value:   number|string,    // line 3 (primary metric)
  //       delta:   number|string,    // line 4 (Δ vs. baseline)
  //       extra:   string            // appended after line 4
  //     }
  //
  //   `suffix` is appended to value/delta when they are numeric (e.g. '%', 'bps').
  //   Output: HTML string suitable for Plotly hovertemplate (with <extra></extra>).
  // ──────────────────────────────────────────────────────────────────────────
  function _fmtNum(v, suffix) {
    if (v === null || v === undefined) return null;
    if (typeof v === "number") {
      if (!isFinite(v)) return null;
      var sign = v > 0 ? "+" : "";
      var rendered = sign + v.toFixed(Math.abs(v) < 1 ? 3 : 2);
      return suffix ? rendered + suffix : rendered;
    }
    return String(v);
  }

  function _fmtDate(v) {
    if (v === null || v === undefined) return null;
    if (v instanceof Date) {
      try {
        return v.toISOString().slice(0, 10);
      } catch (_e) {
        return String(v);
      }
    }
    return String(v);
  }

  function hoverFmt(d, suffix) {
    var s = suffix || "";
    var data = d || {};
    var lines = [];

    if (data.title) {
      lines.push('<b>' + String(data.title) + '</b>');
    }
    var dateStr = _fmtDate(data.date);
    if (dateStr) {
      lines.push('<span style="color:#94a3b8">' + dateStr + '</span>');
    }
    var valStr = _fmtNum(data.value, s);
    if (valStr !== null) {
      lines.push('• ' + valStr);
    }
    var deltaStr = _fmtNum(data.delta, s);
    if (deltaStr !== null) {
      lines.push('• Δ ' + deltaStr);
    }
    if (data.extra) {
      lines.push('• ' + String(data.extra));
    }

    // Wrap whole thing in mono span so Plotly's hoverlabel font is honored
    // even when the host page overrides it.
    var body =
      '<span style="font-family:' + FONT_MONO + ';font-size:11px">' +
      lines.join('<br>') +
      '</span>' +
      '<extra></extra>';

    return body;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // titleUppercase — helper for axis-title strings (Plotly can't text-transform)
  // ──────────────────────────────────────────────────────────────────────────
  function titleUppercase(text) {
    if (text === null || text === undefined) return "";
    // Letter-spacing 0.04em is a CSS feature; in SVG text it doesn't apply,
    // so we approximate by uppercasing and inserting a thin space between
    // letters of short labels (<= 24 chars). Long labels keep their case-only
    // transform to avoid visual noise.
    var s = String(text).toUpperCase();
    if (s.length <= 24) {
      return s.split("").join(" "); // hair space ≈ 0.04em tracking
    }
    return s;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Re-style all live Plotly graphs after a theme change.
  // ──────────────────────────────────────────────────────────────────────────
  function _restyleAllCharts() {
    if (typeof window === "undefined" || !window.Plotly) return;
    var p = palette();
    var newLayout = {
      "paper_bgcolor": p.bg,
      "plot_bgcolor":  p.surface,
      "font.color":    p.ink2,
      "colorway":      colorway(p),
      "xaxis.tickcolor":  p.hairline2,
      "xaxis.tickfont.color": p.ink3,
      "yaxis.gridcolor":  p.hairline,
      "yaxis.tickcolor":  p.hairline2,
      "yaxis.tickfont.color": p.ink3,
      "hoverlabel.bgcolor":     p.surface,
      "hoverlabel.bordercolor": p.hairline2,
      "hoverlabel.font.color":  p.ink,
    };
    var nodes = document.querySelectorAll(".js-plotly-plot");
    nodes.forEach(function (el) {
      try {
        window.Plotly.relayout(el, newLayout);
      } catch (err) {
        // Non-fatal: chart may have been removed from the DOM mid-restyle.
        console.warn("[pfm-bloomberg] relayout skipped:", err);
      }
    });
  }

  // ──────────────────────────────────────────────────────────────────────────
  // apply() — install global defaults + theme-change listener. Idempotent.
  // ──────────────────────────────────────────────────────────────────────────
  var _applied = false;
  function apply() {
    if (_applied) return;

    // 1) Plotly global config defaults — only if Plotly is loaded.
    if (typeof window !== "undefined" && window.Plotly &&
        typeof window.Plotly.setPlotConfig === "function") {
      try {
        window.Plotly.setPlotConfig({
          displayModeBar: "hover",
          displaylogo:    false,
          modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
        });
      } catch (err) {
        console.warn("[pfm-bloomberg] setPlotConfig failed:", err);
      }
    }

    // 2) Subscribe to theme-change events emitted by web/js/theme-toggle.js.
    if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
      window.addEventListener("pfm:theme-change", function (_evt) {
        _restyleAllCharts();
      });
    }

    _applied = true;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Public surface
  // ──────────────────────────────────────────────────────────────────────────
  root.PFM.plotlyBloomberg = {
    version:         "w12-33",
    apply:           apply,
    hoverFmt:        hoverFmt,
    layoutTemplate:  layoutTemplate,
    palette:         palette,
    colorway:        function () { return colorway(palette()); },
    titleUppercase:  titleUppercase,
    config:          config,
    fonts:           { sans: FONT_SANS, mono: FONT_MONO },
    // Exposed for tests / advanced callers
    _palettes:       { light: LIGHT, dark: DARK },
    _restyleAll:     _restyleAllCharts,
  };

  // Auto-apply on script load if Plotly is already available. Otherwise the
  // host page can call PFM.plotlyBloomberg.apply() once Plotly is loaded.
  try {
    if (typeof window !== "undefined" && window.Plotly) {
      apply();
    } else if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
      // Re-attempt after DOM ready, in case Plotly loads via deferred script.
      window.addEventListener("DOMContentLoaded", function () {
        if (window.Plotly) apply();
      });
    }
  } catch (err) {
    console.warn("[pfm-bloomberg] auto-apply skipped:", err);
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = root.PFM.plotlyBloomberg;
  }
})(typeof window !== "undefined" ? window : this);
