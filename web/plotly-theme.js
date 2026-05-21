/* plotly-theme.js — PREMIUM REVISION
 * ==========================================================================
 * "Quantum Terminal" design language for Plotly charts.
 *
 *   Aesthetic goals:
 *     · Density without clutter (Bloomberg-tight, Linear-clean)
 *     · Confidence without flourish (no gradients, no novelty pills)
 *     · Premium = restraint. Hairline grid only on Y. No axis lines. Ticks
 *       outside, monospaced numbers, calm hovers (dark slate, white text).
 *     · Tokens align with /web/terminal-premium.css ("Quantum Terminal").
 *
 *   Typography:
 *     · Sans:  Inter, with Apple/system fallbacks
 *     · Mono:  JetBrains Mono (used in hover and tick formatting for tabular nums)
 *
 *   Backwards-compat:
 *     The window.pfmTheme API surface is preserved. All prior call sites keep
 *     working. This revision tightens defaults inside lineTrace/barTrace/...
 *     and ADDS helpers: applyPremium, premiumConfig, hoverTemplate, glowTrace,
 *     plus a `subtle` color token. No exports renamed or removed.
 *
 * Exposes a single global: window.pfmTheme
 *
 * Usage (legacy — still works):
 *   const layout = pfmTheme.applyToLayout({ title: "Returns" });
 *   Plotly.newPlot(el, [pfmTheme.lineTrace("AAPL", xs, ys, "positive")], layout, pfmTheme.config);
 *
 * Usage (premium):
 *   const layout = pfmTheme.applyPremium({ title: "Returns" });
 *   const traces = [pfmTheme.lineTrace("AAPL", xs, ys, "positive", { glow: true })];
 *   Plotly.newPlot(el, traces, layout, pfmTheme.premiumConfig);
 * ==========================================================================
 */
(function (root) {
  "use strict";

  // --------------------------------------------------------------------------
  // Environment — reduced motion preference (live-updated via matchMedia)
  // --------------------------------------------------------------------------
  let _rmQuery = null;
  let _rmMatches = false;
  try {
    if (typeof window !== "undefined" && window.matchMedia) {
      _rmQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
      _rmMatches = _rmQuery.matches;
      try {
        _rmQuery.addEventListener("change", function (e) {
          _rmMatches = e.matches;
        });
      } catch (e) {
        console.warn("[pfm-theme]", e);
      }
    }
  } catch (e) {
    console.warn("[pfm-theme]", e);
  }
  // Snapshot for one-shot consumers (e.g. baseLayout.transition below). The
  // live value is exposed as a getter on pfmTheme.prefersReducedMotion.
  const PREFERS_REDUCED_MOTION_AT_LOAD = _rmMatches;

  // --------------------------------------------------------------------------
  // Font stacks — aligned with terminal-premium.css tokens
  // --------------------------------------------------------------------------
  const FONT_SANS =
    "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif";
  const FONT_MONO =
    "'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace";

  // --------------------------------------------------------------------------
  // Colors — Quantum Terminal palette
  // --------------------------------------------------------------------------
  const colors = {
    positive: "#16a34a",
    negative: "#dc2626",
    neutral:  "#6a6a73",   // tx-ink-3
    accent:   "#7c3aed",   // tx-accent (purple)
    peer:     "#0891b2",
    fair:     "#7c3aed",   // restored: purple/accent "fair value" tone
    warn:     "#d97706",   // amber/orange — explicit warn tone
    info:     "#0284c7",
    subtle:   "#cbd5dc",   // inactive / baseline / reference lines

    // Ink scale (mirrors terminal-premium.css)
    ink:      "#0a0a0c",
    ink2:     "#3f3f47",
    ink3:     "#6a6a73",
    ink4:     "#9c9ca5",

    // Surface / structural
    hairline:  "#ececef",
    hairline2: "#d6d6dc",
    surface:   "#fcfcfd",
    bg:        "#ffffff",
  };

  // --------------------------------------------------------------------------
  // Premium axis defaults — hairline grid (Y only), outside ticks, no spine
  // --------------------------------------------------------------------------
  const premiumXAxis = {
    showgrid:   false,
    zeroline:   false,
    showline:   false,
    ticks:      "outside",
    tickcolor:  colors.hairline2,
    ticklen:    4,
    tickwidth:  1,
    tickfont:   { family: FONT_SANS, size: 10.5, color: colors.ink3 },
    title:      null,
    automargin: true,
  };

  const premiumYAxis = {
    showgrid:   true,
    gridcolor:  colors.hairline,
    gridwidth:  1,
    zeroline:   false,
    showline:   false,
    ticks:      "outside",
    tickcolor:  colors.hairline2,
    ticklen:    4,
    tickwidth:  1,
    tickfont:   { family: FONT_SANS, size: 10.5, color: colors.ink3 },
    // No default tickformat — callers opt-in via pfmTheme.tickformat.* so
    // non-percentage charts (Sharpe, equity multiples, volume, bps) render
    // their raw values instead of being silently coerced into "%".
    title:      null,
    automargin: true,
  };

  // Tickformat presets — opt-in by callers via e.g. `yaxis: { tickformat: pfmTheme.tickformat.percent }`
  const tickformat = {
    percent: ".1%",
    dollar:  "$,.2f",
    compact: "~s",
    bps:     ",.0f",
  };

  // --------------------------------------------------------------------------
  // Premium base layout — transparent surfaces, small margins, calm hovers
  // --------------------------------------------------------------------------
  const baseLayout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor:  "rgba(0,0,0,0)",
    font: {
      family: FONT_SANS,
      size: 11,
      color: colors.ink2,
    },
    margin:    { l: 48, r: 16, t: 8, b: 32 },
    autosize:  true,
    xaxis:     Object.assign({}, premiumXAxis),
    yaxis:     Object.assign({}, premiumYAxis),
    hovermode: "x unified",
    dragmode:  false,
    showlegend: false,
    hoverlabel: {
      bgcolor:     colors.ink,
      bordercolor: "transparent",
      align:       "left",
      font:        { family: FONT_SANS, size: 11, color: "#ffffff" },
    },
    // Disable layout transitions for reduced-motion users.
    transition: PREFERS_REDUCED_MOTION_AT_LOAD
      ? { duration: 0 }
      : { duration: 180, easing: "cubic-in-out" },
  };

  // Legacy alias — older call sites read `pfmTheme.layout` directly.
  const layout = baseLayout;

  // --------------------------------------------------------------------------
  // Plotly configs
  // --------------------------------------------------------------------------
  const config = {
    displayModeBar: false,
    responsive: true,
    staticPlot: false,
  };

  const smallChartConfig = {
    displayModeBar: false,
    responsive: true,
    staticPlot: false,
    scrollZoom: false,
  };

  // Premium config — Terminal density: modebar on hover only (so quants can
  // still grab a PNG / pan / reset), curated buttons.
  const premiumConfig = {
    displayModeBar: "hover",
    responsive:     true,
    displaylogo:    false,
    scrollZoom:     false,
    showTips:       false,
    staticPlot:     false,
    doubleClick:    "reset",
    modeBarButtonsToRemove: [
      "lasso2d",
      "select2d",
      "autoScale2d",
      "hoverClosestCartesian",
      "hoverCompareCartesian",
      "toggleSpikelines",
    ],
  };

  // --------------------------------------------------------------------------
  // Deep merge helper (objects over primitives; right wins)
  // --------------------------------------------------------------------------
  function isPlainObject(v) {
    return v !== null && typeof v === "object" && !Array.isArray(v);
  }

  function deepMerge(target, source) {
    if (!isPlainObject(source)) return source;
    const out = isPlainObject(target) ? Object.assign({}, target) : {};
    for (const key of Object.keys(source)) {
      const sv = source[key];
      const tv = out[key];
      if (isPlainObject(sv) && isPlainObject(tv)) {
        out[key] = deepMerge(tv, sv);
      } else if (isPlainObject(sv)) {
        out[key] = deepMerge({}, sv);
      } else {
        out[key] = sv;
      }
    }
    return out;
  }

  function applyToLayout(custom) {
    return deepMerge(baseLayout, custom || {});
  }

  // Premium-grade merge: identical to applyToLayout but explicit so call
  // sites can opt-in semantically. Future-proofing if the two ever diverge.
  function applyPremium(custom) {
    return deepMerge(baseLayout, custom || {});
  }

  // --------------------------------------------------------------------------
  // Color utilities
  // --------------------------------------------------------------------------
  function colorFor(kind) {
    return colors[kind] || colors.neutral;
  }

  // Translate "#rrggbb" -> "rgba(r,g,b,a)"; pass-through if already rgba/rgb
  function withAlpha(color, alpha) {
    if (!color) return "rgba(106,106,115," + alpha + ")";
    if (color.startsWith("rgba") || color.startsWith("rgb")) return color;
    const hex = color.replace("#", "");
    const full = hex.length === 3
      ? hex.split("").map((c) => c + c).join("")
      : hex;
    const r = parseInt(full.slice(0, 2), 16);
    const g = parseInt(full.slice(2, 4), 16);
    const b = parseInt(full.slice(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
  }

  // --------------------------------------------------------------------------
  // Trace builders
  // --------------------------------------------------------------------------
  /**
   * Build a premium line trace. Linear by default (faithful to data; sharp
   * regime breaks, earnings prints, resolution decays render correctly).
   * Callers can opt-in to `shape: "spline"` via opts for decorative charts.
   * Width 2, with dash-cycle support for colorblind safety.
   *
   * @param {string} name      legend / hover name
   * @param {Array}  x         x values (typically Dates)
   * @param {Array}  y         y values
   * @param {string} kind      color key in `colors` (default "positive")
   * @param {object} [opts]    { seriesIndex, dash, width, glow, shape, smoothing, dashFromOpts }
   *                           - shape: "linear" (default) | "spline" | "hv" | "vh" | "hvh" | "vhv"
   *                           - smoothing: only honored when shape === "spline"; default 0.6
   *                           - glow: true -> returns [glow, line] pair when used
   *                             via lineTraceWithGlow(). For backward compat,
   *                             lineTrace() itself still returns a single trace
   *                             but accepts `opts.glow` quietly (ignored here).
   * @returns {object} a Plotly scatter trace
   */
  function lineTrace(name, x, y, kind, opts) {
    const k = kind || "positive";
    const color = colorFor(k);
    const o = opts || {};
    // seriesIndex >0 picks a non-solid dash — colorblind / B&W safety.
    const i = (typeof o.seriesIndex === "number") ? o.seriesIndex : 0;
    const dash = o.dash || dashFor(i);
    const width = (typeof o.width === "number") ? o.width : 2;
    const shape = o.shape || "linear";
    const smoothing = (typeof o.smoothing === "number") ? o.smoothing : 0.6;

    return {
      type: "scatter",
      mode: "lines",
      name: name,
      x: x,
      y: y,
      line: {
        color: color,
        width: width,
        shape: shape,
        smoothing: shape === "spline" ? smoothing : undefined,
        dash: dash,
      },
      hoverlabel: {
        bgcolor:     colors.ink,
        bordercolor: "transparent",
        align:       "left",
        font:        { family: FONT_SANS, size: 11, color: "#ffffff" },
      },
      connectgaps: false,
    };
  }

  /**
   * Build a soft "glow" trace — a wider, very low-alpha echo of a line.
   * Push this trace BEFORE the main line trace so it renders underneath.
   *
   *   const glow = pfmTheme.glowTrace(name, x, y, "positive");
   *   const main = pfmTheme.lineTrace(name, x, y, "positive");
   *   Plotly.newPlot(el, [glow, main], layout, cfg);
   */
  function glowTrace(name, x, y, kind, opts) {
    const k = kind || "positive";
    const color = colorFor(k);
    const o = opts || {};
    const width = (typeof o.width === "number") ? o.width : 6;
    const alpha = (typeof o.alpha === "number") ? o.alpha : 0.08;
    return {
      type: "scatter",
      mode: "lines",
      name: name + " (glow)",
      x: x,
      y: y,
      line: {
        color: withAlpha(color, alpha),
        width: width,
        shape: o.shape || "spline",
        smoothing: 0.6,
      },
      hoverinfo: "skip",
      showlegend: false,
    };
  }

  /**
   * Convenience: returns [glow, line] pair. Use spread:
   *   traces.push(...pfmTheme.lineTraceWithGlow("AAPL", xs, ys, "positive"));
   */
  function lineTraceWithGlow(name, x, y, kind, opts) {
    return [
      glowTrace(name, x, y, kind, opts),
      lineTrace(name, x, y, kind, opts),
    ];
  }

  function barTrace(name, x, y, kind, opts) {
    const k = kind || "positive";
    const color = colorFor(k);
    const o = opts || {};
    const opacity = (typeof o.opacity === "number") ? o.opacity : 0.92;
    return {
      type: "bar",
      name: name,
      x: x,
      y: y,
      marker: {
        color: color,
        line: { width: 0 },
        cornerradius: 3,
        opacity: opacity,
      },
      // Hovering brightens to full opacity — calm but responsive.
      hoverlabel: {
        bgcolor:     colors.ink,
        bordercolor: "transparent",
        align:       "left",
        font:        { family: FONT_SANS, size: 11, color: "#ffffff" },
      },
      // Plotly honors `selected`/`unselected` for marker; we use it lightly so
      // non-hovered bars stay at the configured opacity and hovered ones pop.
      selected: { marker: { opacity: 1 } },
      unselected: { marker: { opacity: opacity } },
    };
  }

  /**
   * Translucent band between y_lo and y_hi. Premium revision drops alpha
   * 18% -> 10% so confidence ribbons feel quieter behind the main line.
   */
  function fillBetween(name, x, y_lo, y_hi, color) {
    const c = color || colors.neutral;
    const fillColor = withAlpha(c, 0.10);
    const lineColor = withAlpha(c, 0);
    const lower = {
      type: "scatter",
      mode: "lines",
      name: name + " (lo)",
      x: x,
      y: y_lo,
      line: { color: lineColor, width: 0 },
      hoverinfo: "skip",
      showlegend: false,
    };
    const upper = {
      type: "scatter",
      mode: "lines",
      name: name,
      x: x,
      y: y_hi,
      line: { color: lineColor, width: 0 },
      fill: "tonexty",
      fillcolor: fillColor,
      hoverinfo: "skip",
      showlegend: false,
    };
    return [lower, upper];
  }

  // --------------------------------------------------------------------------
  // Heatmap helpers
  // --------------------------------------------------------------------------
  // Refined diverging colorscale — slightly desaturated vs. raw ColorBrewer
  // for a calmer, more premium feel. Anchored at white=0.
  const HEATMAP_RDBU_R = [
    [0.0,  "#1e3a5f"],
    [0.1,  "#2c5a8a"],
    [0.2,  "#4e85b5"],
    [0.3,  "#8eb4d4"],
    [0.4,  "#cfe0ec"],
    [0.5,  "#f7f7f8"],
    [0.6,  "#f0d6c4"],
    [0.7,  "#dfa085"],
    [0.8,  "#c66a5b"],
    [0.9,  "#a8333c"],
    [1.0,  "#6d1320"],
  ];

  // Sequential perceptually-uniform colormap for one-sided heatmaps
  // (e.g. p-values). Purple-anchored, dark = low / dense.
  const HEATMAP_SEQ_PURPLE = [
    [0.0,   "#3b0764"],
    [0.05,  "#7c3aed"],
    [0.1,   "#a78bfa"],
    [0.2,   "#c4b5fd"],
    [0.5,   "#ddd6fe"],
    [1.0,   "#f5f3ff"],
  ];

  function heatmapTrace(opts) {
    // opts: { x, y, z, mode: "diverging"|"sequential", zmin, zmax, name,
    //         hovertemplate, colorbar, xgap, ygap }
    const mode = (opts && opts.mode) || "diverging";
    const scale = mode === "sequential" ? HEATMAP_SEQ_PURPLE : HEATMAP_RDBU_R;
    return {
      type: "heatmap",
      x: opts.x,
      y: opts.y,
      z: opts.z,
      zmin: opts.zmin != null ? opts.zmin : (mode === "diverging" ? -1 : 0),
      zmax: opts.zmax != null ? opts.zmax : 1,
      zauto: false,
      colorscale: scale,
      colorbar: Object.assign(
        {
          thickness:    8,
          len:          0.85,
          outlinewidth: 0,
          tickfont:     { family: FONT_SANS, size: 10, color: colors.ink3 },
          ticklen:      3,
          tickcolor:    colors.hairline2,
        },
        opts.colorbar || {}
      ),
      // Hover uses monospaced tabular nums for crisp value alignment.
      hovertemplate:
        opts.hovertemplate ||
        ('<b>%{x}</b> · <b>%{y}</b><br>' +
          '<span style="font-family:' + FONT_MONO + '">%{z:.3f}</span>' +
          '<extra></extra>'),
      hoverlabel: {
        bgcolor:     colors.ink,
        bordercolor: "transparent",
        align:       "left",
        font:        { family: FONT_SANS, size: 11, color: "#ffffff" },
      },
      xgap: opts.xgap != null ? opts.xgap : 1,
      ygap: opts.ygap != null ? opts.ygap : 1,
      name: opts.name || "",
    };
  }

  // --------------------------------------------------------------------------
  // Coefficient bar-chart polish helper (rounded ends + visible whiskers)
  // --------------------------------------------------------------------------
  function coefBarStyle(barColors) {
    return {
      marker: {
        color: barColors,
        line: { width: 0 },
        cornerradius: 3,
      },
    };
  }

  function errorWhiskerStyle(arr, opts) {
    const o = opts || {};
    return {
      type: "data",
      array: arr,
      color: o.color || colors.ink2,
      thickness: o.thickness != null ? o.thickness : 1.4,
      width: o.width != null ? o.width : 6,
      visible: true,
    };
  }

  // --------------------------------------------------------------------------
  // Dash patterns — color-blind-safe series differentiation
  // --------------------------------------------------------------------------
  const DASH_CYCLE = ["solid", "dash", "dot", "longdash", "dashdot", "longdashdot"];
  function dashFor(i) {
    return DASH_CYCLE[(i | 0) % DASH_CYCLE.length];
  }
  // Legacy export under both names — some call sites import `dashCycle`.
  const dashCycle = DASH_CYCLE;

  // --------------------------------------------------------------------------
  // Hover-template helpers — left-aligned, monospaced tabular numbers
  // --------------------------------------------------------------------------
  // Templates assume the dark `hoverlabel` set in baseLayout. The series name
  // sits in muted grey; the value pops to pure white in JetBrains Mono.
  const _monoSpan = function (text) {
    return '<span style="font-family:' + FONT_MONO +
      ';color:#ffffff">' + text + '</span>';
  };

  const hoverTemplate = {
    /** Time-series price/return: "May 14 · NVDA  +2.3%" */
    price: function (name) {
      const label = name ? (name + " ") : "";
      return '<b>%{x|%b %d}</b><br>' + label + _monoSpan("%{y:.1%}") +
        '<extra></extra>';
    },
    /** Delta vs. baseline: "May 14 · Δ NVDA  +0.42%" */
    delta: function (name) {
      const label = name ? ("Δ " + name + " ") : "Δ ";
      return '<b>%{x|%b %d}</b><br>' + label + _monoSpan("%{y:+.2%}") +
        '<extra></extra>';
    },
    /** Probability (0..1) shown as percent with 1dp. */
    prob: function (name) {
      const label = name ? (name + " ") : "";
      return '<b>%{x|%b %d}</b><br>' + label + _monoSpan("%{y:.1%}") +
        '<extra></extra>';
    },
    /** Raw numeric series with configurable decimal places. */
    value: function (name, dp) {
      const d = (typeof dp === "number") ? dp : 2;
      const label = name ? (name + " ") : "";
      return '<b>%{x|%b %d}</b><br>' + label +
        _monoSpan("%{y:." + d + "f}") + '<extra></extra>';
    },
    /** Basis-points formatter (multiply by 10000). */
    bps: function (name) {
      const label = name ? (name + " ") : "";
      // Plotly's d3-format can't multiply, so the caller passes bps already.
      return '<b>%{x|%b %d}</b><br>' + label +
        _monoSpan("%{y:+.0f} bps") + '<extra></extra>';
    },
  };

  // --------------------------------------------------------------------------
  // Tickformat selector — date axes adapt to range length
  // --------------------------------------------------------------------------
  /**
   * Returns "%b %d" for sub-60-day ranges, "%b %Y" otherwise. Pass either
   * a (dateArray) or (startDate, endDate). Falls back to "%b %d" if unknown.
   */
  function dateTickFormat(a, b) {
    let start, end;
    try {
      if (Array.isArray(a) && a.length >= 2) {
        start = new Date(a[0]);
        end = new Date(a[a.length - 1]);
      } else if (a && b) {
        start = new Date(a);
        end = new Date(b);
      } else {
        return "%b %d";
      }
      const days = Math.abs((end - start) / 86400000);
      return days < 60 ? "%b %d" : "%b %Y";
    } catch (e) {
      console.warn("[pfm-theme]", e);
      return "%b %d";
    }
  }

  // --------------------------------------------------------------------------
  // Formatters (display-side string helpers)
  // --------------------------------------------------------------------------
  function _isNum(v) {
    return typeof v === "number" && isFinite(v);
  }

  function formatPercent(v, dp) {
    const d = (typeof dp === "number") ? dp : 1;
    if (!_isNum(v)) return "—";
    return (v * 100).toFixed(d) + "%";
  }

  function formatBps(v) {
    if (!_isNum(v)) return "—";
    return (v * 10000).toFixed(0) + " bps";
  }

  function formatProb(v) {
    if (!_isNum(v)) return "—";
    const clipped = Math.max(0, Math.min(1, v));
    return (clipped * 100).toFixed(1) + "%";
  }

  // --------------------------------------------------------------------------
  // Export — backwards-compatible surface plus new premium helpers
  // --------------------------------------------------------------------------
  const pfmTheme = {
    // ── Legacy (do not rename — existing call sites depend on these) ──
    layout: layout,
    baseLayout: baseLayout,
    colors: colors,
    config: config,
    smallChartConfig: smallChartConfig,
    applyToLayout: applyToLayout,
    lineTrace: lineTrace,
    barTrace: barTrace,
    fillBetween: fillBetween,
    heatmapTrace: heatmapTrace,
    coefBarStyle: coefBarStyle,
    errorWhiskerStyle: errorWhiskerStyle,
    dashFor: dashFor,
    dashCycle: dashCycle,
    HEATMAP_RDBU_R: HEATMAP_RDBU_R,
    HEATMAP_SEQ_PURPLE: HEATMAP_SEQ_PURPLE,
    formatPercent: formatPercent,
    formatBps: formatBps,
    formatProb: formatProb,

    // ── New premium helpers ──
    applyPremium: applyPremium,
    premiumConfig: premiumConfig,
    hoverTemplate: hoverTemplate,
    glowTrace: glowTrace,
    lineTraceWithGlow: lineTraceWithGlow,
    withAlpha: withAlpha,
    dateTickFormat: dateTickFormat,
    premiumXAxis: premiumXAxis,
    premiumYAxis: premiumYAxis,
    tickformat: tickformat,
    fonts: { sans: FONT_SANS, mono: FONT_MONO },
  };

  // `prefersReducedMotion` is a live getter so it reflects user changes after
  // page load (e.g. enabling Reduce Motion in OS settings mid-session).
  Object.defineProperty(pfmTheme, "prefersReducedMotion", {
    get: function () { return _rmMatches; },
    enumerable: true,
    configurable: false,
  });

  root.pfmTheme = pfmTheme;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = pfmTheme;
  }
})(typeof window !== "undefined" ? window : this);
