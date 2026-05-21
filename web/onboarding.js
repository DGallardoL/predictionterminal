/**
 * Prediction Terminal — premium first-time tour ("Welcome cover + spotlight").
 *
 * v2 rewrite — designed to feel like a polished Linear / Stripe / Vercel
 * onboarding, not a stock 5-step popover.
 *
 * Structure
 * ─────────
 *   Step 1 — animated welcome cover (centered, stat counters, gradient hero)
 *   Steps 2-7 — anchored spotlight panels for each mode + ⌘K + power-tools
 *   Step 8 — finish card with three quick-start CTAs
 *
 * Implementation notes
 * ────────────────────
 *   • Uses unique `pt-tour-*` class prefix so it does NOT collide with the
 *     older inline `.pfm-tour-*` styles still living in index.html
 *     (that legacy block remains in place for safety; this file is the
 *     active tour for first-visit + ?tour=1 + restart-button).
 *   • Spotlight = SVG mask cutout with animated rect geometry attrs —
 *     smoother than the legacy `box-shadow: 0 0 0 9999px` trick.
 *   • SVG stage is pointer-events:none so clicks pass straight through to
 *     the underlying UI (Linear-style — users can interact with the
 *     highlighted control without dismissing the tour).
 *   • localStorage `pt-tour-shown` = v2 marks dismissed. We also write the
 *     two legacy keys (`pfm-tour-shown`, `pfm:tour:done`) so the older
 *     tours never double-fire.
 *   • `window.pfmTour.start(true)` re-launches the tour on demand.
 *     We also overwrite the inline `window.pfmStartTour` so ?tour=1
 *     URL replays use this premium version, not the legacy popover.
 *   • Respects prefers-reduced-motion (no counter animation, no spotlight
 *     interpolation, no shimmer) — the page-level CSS rule already kills
 *     keyframe animations, JS just skips RAF interpolations.
 *   • Single self-contained IIFE. No external deps. No build step.
 */
(function () {
  "use strict";

  // ─────────────────────────────────────────────────────────────────────
  //  CONFIGURATION
  // ─────────────────────────────────────────────────────────────────────

  var STORAGE_KEY = "pt-tour-shown";
  var STORAGE_VERSION = "v2";
  // Legacy tour gates we proactively mark so the older popovers + the
  // small bottom "Welcome to PFM" banner never double up with our tour.
  var LEGACY_KEYS = ["pfm-tour-shown", "pfm:tour:done", "pfm:welcomed"];
  var Z_BASE = 2147483000;            // dominate every other layer
  var SPOTLIGHT_PAD = 8;              // px of breathing room around anchor
  var SPOTLIGHT_RADIUS = 12;
  var PANEL_GAP = 18;                 // gap between anchor + panel
  var PANEL_W_COVER = 520;
  var PANEL_W_ANCHOR = 380;

  var STEPS = [
    {
      id: "welcome",
      kind: "cover",
      eyebrow: "Live · prediction-market alpha",
      title: "Welcome to <span class=\"pt-tour-grad\">Prediction Terminal</span>",
      lede:
        "A Bloomberg-style data hub for prediction markets — built for " +
        "traders, researchers, and quants.",
      stats: [
        { value: 1250, label: "live factors", suffix: "" },
        { value: 297,  label: "API endpoints", suffix: "" },
        { value: 4,    label: "deployable alphas", suffix: "" }
      ],
      primary: "Take the 60-second tour →",
      secondary: "Skip"
    },
    {
      id: "modes",
      kind: "anchor",
      anchor: ".mode-switch",
      preferredPos: "below",
      step: "01",
      eyebrow: "The three modes",
      title: "One app, three minds",
      bullets: [
        ["Terminal",   "live market data, top movers, orderbooks"],
        ["Strategies", "curated α Hub, cross-venue arb, crypto micro"],
        ["Regression", "fit your own factor models on prediction data"]
      ],
      hint: "You can switch any time with <kbd>g</kbd>+<kbd>t</kbd> / <kbd>s</kbd> / <kbd>r</kbd>."
    },
    {
      id: "terminal",
      kind: "anchor",
      anchor: ".mode-switch .mode-btn[data-mode=\"terminal\"]",
      preferredPos: "below",
      step: "02",
      eyebrow: "Terminal mode",
      title: "Live data, Bloomberg-style",
      bullets: [
        ["Top movers",   "ranked by 24h Δprob across 500 live markets"],
        ["Deep drill",   "click any row for orderbook, history, peers, fair-price"],
        ["Search",       "by name, slug, theme, or category"],
        ["58 endpoints", "under /terminal/* — fully scriptable via REST"]
      ],
      hint: "Terminal is the default landing tab."
    },
    {
      id: "strategies",
      kind: "anchor",
      anchor: ".mode-switch .mode-btn[data-mode=\"strategies\"]",
      preferredPos: "below",
      step: "03",
      eyebrow: "Strategies — α Hub",
      title: "Validated alphas, ranked by tier",
      bullets: [
        ["Top Alphas",      "rainbow tier pills (A_GOLD = robust, B = paper-only)"],
        ["Calendar & Spreads", "λ-ratio decay plays across resolution windows"],
        ["Cross-venue Arb", "live Kalshi ↔ Polymarket stream, fee-aware sizing"],
        ["Crypto Micro",    "σ + μ model vs market on 5/15m BTC + ETH"]
      ],
      hint: "Click any card for a full tearsheet — equity curve, trade rule, sizing."
    },
    {
      id: "regression",
      kind: "anchor",
      anchor: ".mode-switch .mode-btn[data-mode=\"regression\"]",
      preferredPos: "below",
      step: "04",
      eyebrow: "Regression",
      title: "Why is NVDA moving?",
      lede:
        "Type a ticker into the hero — we stream factor attribution from " +
        "1,250+ prediction markets. Or fit a model from scratch.",
      bullets: [
        ["HAC-corrected SEs", "heteroskedasticity- and autocorrelation-consistent, with automatic bandwidth selection"],
        ["VIF flags",         "multicollinearity warnings inline"],
        ["Bootstrap CI",      "5000 resamples on every β"],
        ["Factor heatmap",    "pairwise ρ across your selection"]
      ]
    },
    {
      id: "search",
      kind: "anchor",
      anchor: ".pfm-cmdk-hint, .nav-links",
      preferredPos: "below",
      step: "05",
      eyebrow: "Instant search",
      title: "<kbd class=\"pt-tour-kbd-lg\">⌘</kbd><kbd class=\"pt-tour-kbd-lg\">K</kbd> — find anything",
      lede:
        "Fuzzy-search every factor, market, recent fit, and command — " +
        "from any view, without leaving your keyboard.",
      bullets: [
        ["⌘ K / Ctrl K",   "open the palette"],
        ["/",              "shortcut to the same palette"],
        ["&gt;",           "switch to command mode (toggle theme, share, recents)"],
        ["? ",             "see the full keyboard cheatsheet"]
      ]
    },
    {
      id: "personalize",
      kind: "anchor",
      anchor: "#pfm-shortcuts-btn, #pfm-theme-btn, .nav-links",
      preferredPos: "below",
      step: "06",
      eyebrow: "Make it yours",
      title: "Dark mode, alerts, shortcuts",
      bullets: [
        ["Theme",      "tap Dark/Light in the nav — persists across sessions"],
        ["Alerts",     "price/threshold notifications from the bell"],
        ["Share view", "copy a deep-link to your current state and filters"],
        ["?",          "open keyboard shortcuts (or read methodology in docs)"]
      ],
      hint: "Everything works offline-first — your watchlist + theme live in localStorage."
    },
    {
      id: "finish",
      kind: "finish",
      eyebrow: "Ready to trade",
      title: "You're set.",
      lede: "Three quick wins to try right now —",
      quickStarts: [
        {
          label: "View the top alpha right now",
          sub: "open Strategies → α Hub",
          action: "openStrategies"
        },
        {
          label: "Ask: “Why is NVDA moving today?”",
          sub: "open Regression → focus ticker",
          action: "askWhy"
        },
        {
          label: "Search the 1,250-factor catalog",
          sub: "press ⌘K — anywhere",
          action: "openCmdK"
        }
      ],
      footer:
        "Re-run this tour any time from the <strong>Tour</strong> button in " +
        "the top nav, or visit any URL with <code>?tour=1</code>.",
      primary: "Start exploring",
      secondary: ""
    }
  ];

  // ─────────────────────────────────────────────────────────────────────
  //  STATE
  // ─────────────────────────────────────────────────────────────────────

  var state = null;   // { stage, cutout, ring, panel, idx, anchorEl, keyHandler, resizeHandler, rafId }

  // ─────────────────────────────────────────────────────────────────────
  //  STORAGE
  // ─────────────────────────────────────────────────────────────────────

  function hasShown() {
    try { return localStorage.getItem(STORAGE_KEY) === STORAGE_VERSION; }
    catch (e) { return true; }
  }
  function markShown() {
    try {
      localStorage.setItem(STORAGE_KEY, STORAGE_VERSION);
      // Also pacify the two legacy tour gates so we never get a double-fire.
      LEGACY_KEYS.forEach(function (k) {
        try { localStorage.setItem(k, "1"); } catch (_e) {}
      });
    } catch (e) {}
  }
  function clearShown() {
    try {
      localStorage.removeItem(STORAGE_KEY);
      LEGACY_KEYS.forEach(function (k) {
        try { localStorage.removeItem(k); } catch (_e) {}
      });
    } catch (e) {}
  }

  // ─────────────────────────────────────────────────────────────────────
  //  HELPERS
  // ─────────────────────────────────────────────────────────────────────

  function $(sel) { return document.querySelector(sel); }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

  function reducedMotion() {
    try {
      return window.matchMedia &&
             window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch (e) { return false; }
  }

  function findAnchor(selectorList) {
    if (!selectorList) return null;
    var parts = selectorList.split(",").map(function (s) { return s.trim(); }).filter(Boolean);
    for (var i = 0; i < parts.length; i++) {
      try {
        var el = document.querySelector(parts[i]);
        if (el && el.offsetParent !== null) return el;   // visible
      } catch (_e) {}
    }
    // fallback: return ANY match even if offscreen — better than nothing
    for (var j = 0; j < parts.length; j++) {
      try {
        var el2 = document.querySelector(parts[j]);
        if (el2) return el2;
      } catch (_e) {}
    }
    return null;
  }

  // ─────────────────────────────────────────────────────────────────────
  //  STYLES (scoped, prefixed, theme-aware via CSS vars)
  // ─────────────────────────────────────────────────────────────────────

  function injectStyles() {
    if (document.getElementById("pt-tour-styles")) return;
    var css = document.createElement("style");
    css.id = "pt-tour-styles";
    css.textContent = [
      // ── stage / overlay ──────────────────────────────────────────────
      ".pt-tour-stage {",
      "  position: fixed; inset: 0; z-index: " + Z_BASE + ";",
      "  pointer-events: none;",
      "  width: 100vw; height: 100vh;",
      "  opacity: 0; transition: opacity 240ms cubic-bezier(0.4,0,0.2,1);",
      "}",
      ".pt-tour-stage.is-open { opacity: 1; }",
      ".pt-tour-stage .pt-tour-veil {",
      "  fill: rgba(8, 10, 18, 0.62);",
      "}",
      "[data-theme='dark'] .pt-tour-stage .pt-tour-veil { fill: rgba(2, 4, 10, 0.72); }",
      ".pt-tour-stage .pt-tour-bloom {",
      "  opacity: 0; transition: opacity 320ms cubic-bezier(0.4,0,0.2,1);",
      "}",
      ".pt-tour-stage.has-cutout .pt-tour-bloom { opacity: 0.55; }",
      ".pt-tour-cutout {",
      "  fill: #000;",
      "  transition: x 320ms cubic-bezier(0.4,0,0.2,1),",
      "              y 320ms cubic-bezier(0.4,0,0.2,1),",
      "              width 320ms cubic-bezier(0.4,0,0.2,1),",
      "              height 320ms cubic-bezier(0.4,0,0.2,1);",
      "}",
      ".pt-tour-ring {",
      "  fill: none; stroke: url(#pt-tour-glow);",
      "  stroke-width: 2;",
      "  opacity: 0;",
      "  transition: x 320ms cubic-bezier(0.4,0,0.2,1),",
      "              y 320ms cubic-bezier(0.4,0,0.2,1),",
      "              width 320ms cubic-bezier(0.4,0,0.2,1),",
      "              height 320ms cubic-bezier(0.4,0,0.2,1),",
      "              opacity 200ms ease;",
      "}",
      ".pt-tour-stage.has-cutout .pt-tour-ring { opacity: 1; }",
      "@keyframes pt-tour-ring-pulse {",
      "  0%,100% { stroke-opacity: 0.95; }",
      "  50%     { stroke-opacity: 0.55; }",
      "}",
      ".pt-tour-stage.has-cutout .pt-tour-ring { animation: pt-tour-ring-pulse 2.4s ease-in-out infinite; }",
      // ── panel chrome ────────────────────────────────────────────────
      ".pt-tour-panel {",
      "  position: fixed; z-index: " + (Z_BASE + 2) + ";",
      "  pointer-events: auto;",
      "  font-family: \"Inter\", -apple-system, BlinkMacSystemFont, \"Segoe UI\", system-ui, sans-serif;",
      "  color: var(--ink, #0a0a0c);",
      "  background: var(--bg, #ffffff);",
      "  border: 1px solid var(--hairline-2, #d8d6dc);",
      "  border-radius: 14px;",
      "  box-shadow:",
      "    0 1px 0 rgba(255,255,255,0.6) inset,",
      "    0 30px 80px -20px rgba(15, 18, 36, 0.32),",
      "    0 12px 28px -10px rgba(15, 18, 36, 0.18),",
      "    0 2px 6px rgba(15, 18, 36, 0.08);",
      "  width: " + PANEL_W_ANCHOR + "px;",
      "  max-width: calc(100vw - 32px);",
      "  overflow: hidden;",
      "  opacity: 0; transform: translateY(6px);",
      "  transition: opacity 240ms cubic-bezier(0.4,0,0.2,1),",
      "              transform 240ms cubic-bezier(0.4,0,0.2,1),",
      "              top 320ms cubic-bezier(0.4,0,0.2,1),",
      "              left 320ms cubic-bezier(0.4,0,0.2,1);",
      "}",
      ".pt-tour-panel.is-shown { opacity: 1; transform: translateY(0); }",
      "[data-theme='dark'] .pt-tour-panel {",
      "  background: #14141c;",
      "  color: #ececf2;",
      "  border-color: rgba(255,255,255,0.10);",
      "  box-shadow:",
      "    0 1px 0 rgba(255,255,255,0.04) inset,",
      "    0 30px 80px -20px rgba(0,0,0,0.7),",
      "    0 12px 28px -10px rgba(0,0,0,0.5);",
      "}",
      ".pt-tour-panel.is-cover { width: " + PANEL_W_COVER + "px; }",
      // gradient top accent strip
      ".pt-tour-panel::before {",
      "  content: \"\";",
      "  position: absolute; top: 0; left: 0; right: 0; height: 2px;",
      "  background: linear-gradient(90deg, #7c3aed 0%, #ec4899 50%, #f97316 100%);",
      "  opacity: 0.9;",
      "}",
      // arrow ("tip") pointing at anchor
      ".pt-tour-panel .pt-tour-tip {",
      "  position: absolute;",
      "  width: 14px; height: 14px;",
      "  background: inherit;",
      "  border: 1px solid var(--hairline-2, #d8d6dc);",
      "  transform: rotate(45deg);",
      "  z-index: -1;",
      "}",
      "[data-theme='dark'] .pt-tour-panel .pt-tour-tip { border-color: rgba(255,255,255,0.10); }",
      ".pt-tour-panel.tip-above .pt-tour-tip { top: -8px; border-right: none; border-bottom: none; }",
      ".pt-tour-panel.tip-below .pt-tour-tip { bottom: -8px; border-left: none; border-top: none; }",
      ".pt-tour-panel.tip-left  .pt-tour-tip { left: -8px; border-top: none; border-right: none; }",
      ".pt-tour-panel.tip-right .pt-tour-tip { right: -8px; border-bottom: none; border-left: none; }",
      // ── inner layout ────────────────────────────────────────────────
      ".pt-tour-body {",
      "  padding: 20px 22px 18px;",
      "  position: relative;",
      "}",
      ".pt-tour-panel.is-cover .pt-tour-body {",
      "  padding: 30px 32px 24px;",
      "}",
      ".pt-tour-eyebrow {",
      "  display: flex; align-items: center; gap: 8px;",
      "  font-family: \"JetBrains Mono\", ui-monospace, \"SF Mono\", Menlo, monospace;",
      "  font-size: 10px; font-weight: 600;",
      "  letter-spacing: 0.12em; text-transform: uppercase;",
      "  color: var(--ink-3, #6b6878);",
      "  margin: 0 0 10px;",
      "}",
      "[data-theme='dark'] .pt-tour-eyebrow { color: rgba(236, 236, 242, 0.62); }",
      ".pt-tour-eyebrow .pt-tour-pulse {",
      "  display: inline-block; width: 7px; height: 7px; border-radius: 999px;",
      "  background: #f97316;",
      "  box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.5);",
      "  animation: pt-tour-pulse 1.8s ease-out infinite;",
      "}",
      "@keyframes pt-tour-pulse {",
      "  0%   { box-shadow: 0 0 0 0   rgba(249, 115, 22, 0.55); }",
      "  70%  { box-shadow: 0 0 0 9px rgba(249, 115, 22, 0); }",
      "  100% { box-shadow: 0 0 0 0   rgba(249, 115, 22, 0); }",
      "}",
      ".pt-tour-eyebrow .pt-tour-step-num {",
      "  display: inline-block;",
      "  font-variant-numeric: tabular-nums;",
      "  color: var(--ink-2, #4a4a55);",
      "  letter-spacing: 0.1em;",
      "  margin-right: 8px;",
      "  padding-right: 8px;",
      "  border-right: 1px solid var(--hairline, #ececef);",
      "}",
      "[data-theme='dark'] .pt-tour-eyebrow .pt-tour-step-num {",
      "  color: rgba(236, 236, 242, 0.82); border-color: rgba(255,255,255,0.10);",
      "}",
      ".pt-tour-title {",
      "  margin: 0 0 8px;",
      "  font-family: \"Instrument Serif\", \"Source Serif 4\", Georgia, serif;",
      "  font-weight: 400;",
      "  font-size: 24px;",
      "  line-height: 1.18;",
      "  letter-spacing: -0.01em;",
      "  color: var(--ink, #0a0a0c);",
      "}",
      "[data-theme='dark'] .pt-tour-title { color: #f5f5fa; }",
      ".pt-tour-panel.is-cover .pt-tour-title { font-size: 34px; line-height: 1.12; margin: 4px 0 12px; }",
      ".pt-tour-panel.is-finish .pt-tour-title { font-size: 30px; line-height: 1.12; }",
      ".pt-tour-grad {",
      "  background: linear-gradient(120deg, #7c3aed 0%, #ec4899 50%, #f97316 100%);",
      "  -webkit-background-clip: text; background-clip: text;",
      "  -webkit-text-fill-color: transparent; color: transparent;",
      "}",
      ".pt-tour-lede {",
      "  margin: 0 0 14px;",
      "  font-size: 14px; line-height: 1.55;",
      "  color: var(--ink-2, #4a4a55);",
      "}",
      "[data-theme='dark'] .pt-tour-lede { color: rgba(236,236,242,0.78); }",
      ".pt-tour-panel.is-cover .pt-tour-lede { font-size: 15px; }",
      // bullet list
      ".pt-tour-bullets {",
      "  margin: 12px 0 16px; padding: 0; list-style: none;",
      "  display: grid; gap: 9px;",
      "}",
      ".pt-tour-bullet {",
      "  display: grid; grid-template-columns: 130px 1fr; gap: 12px;",
      "  align-items: baseline;",
      "  font-size: 13px; line-height: 1.5;",
      "}",
      ".pt-tour-bullet .pt-tour-bullet-key {",
      "  font-family: \"JetBrains Mono\", ui-monospace, \"SF Mono\", Menlo, monospace;",
      "  font-size: 11.5px; font-weight: 500;",
      "  color: var(--ink, #0a0a0c);",
      "  letter-spacing: 0;",
      "  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;",
      "}",
      "[data-theme='dark'] .pt-tour-bullet .pt-tour-bullet-key { color: #ececf2; }",
      ".pt-tour-bullet .pt-tour-bullet-val {",
      "  color: var(--ink-2, #4a4a55);",
      "  font-size: 12.5px;",
      "}",
      "[data-theme='dark'] .pt-tour-bullet .pt-tour-bullet-val { color: rgba(236,236,242,0.72); }",
      // hint row
      ".pt-tour-hint {",
      "  display: flex; align-items: center; gap: 6px;",
      "  margin: -2px 0 14px;",
      "  padding: 8px 11px;",
      "  background: linear-gradient(135deg, rgba(124,58,237,0.07), rgba(249,115,22,0.06));",
      "  border: 1px solid rgba(124,58,237,0.16);",
      "  border-radius: 8px;",
      "  font-size: 12px;",
      "  color: var(--ink-2, #4a4a55);",
      "}",
      "[data-theme='dark'] .pt-tour-hint {",
      "  background: linear-gradient(135deg, rgba(124,58,237,0.14), rgba(249,115,22,0.10));",
      "  border-color: rgba(124,58,237,0.30);",
      "  color: rgba(236,236,242,0.82);",
      "}",
      // stats trio (cover)
      ".pt-tour-stats {",
      "  display: grid; grid-template-columns: repeat(3, 1fr);",
      "  gap: 0;",
      "  margin: 8px 0 22px;",
      "  border-top: 1px solid var(--hairline, #ececef);",
      "  border-bottom: 1px solid var(--hairline, #ececef);",
      "  padding: 14px 0;",
      "}",
      "[data-theme='dark'] .pt-tour-stats { border-color: rgba(255,255,255,0.08); }",
      ".pt-tour-stat {",
      "  text-align: left; padding: 0 14px;",
      "  border-right: 1px solid var(--hairline, #ececef);",
      "}",
      "[data-theme='dark'] .pt-tour-stat { border-color: rgba(255,255,255,0.08); }",
      ".pt-tour-stat:first-child { padding-left: 0; }",
      ".pt-tour-stat:last-child  { border-right: none; padding-right: 0; }",
      ".pt-tour-stat-value {",
      "  font-family: \"Instrument Serif\", \"Source Serif 4\", Georgia, serif;",
      "  font-weight: 400;",
      "  font-size: 30px; line-height: 1;",
      "  letter-spacing: -0.02em;",
      "  color: var(--ink, #0a0a0c);",
      "  font-variant-numeric: tabular-nums;",
      "}",
      "[data-theme='dark'] .pt-tour-stat-value { color: #f5f5fa; }",
      ".pt-tour-stat-label {",
      "  margin-top: 4px;",
      "  font-family: \"JetBrains Mono\", ui-monospace, \"SF Mono\", Menlo, monospace;",
      "  font-size: 10px; text-transform: uppercase; letter-spacing: 0.10em;",
      "  color: var(--ink-3, #6b6878);",
      "}",
      "[data-theme='dark'] .pt-tour-stat-label { color: rgba(236,236,242,0.55); }",
      // quick-start list (finish)
      ".pt-tour-quicks { display: grid; gap: 8px; margin: 14px 0 18px; }",
      ".pt-tour-quick {",
      "  display: flex; align-items: center; gap: 12px;",
      "  padding: 12px 14px;",
      "  background: var(--surface, #f6f5f8);",
      "  border: 1px solid var(--hairline, #ececef);",
      "  border-radius: 10px;",
      "  cursor: pointer;",
      "  font: inherit; text-align: left; width: 100%;",
      "  color: var(--ink, #0a0a0c);",
      "  transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;",
      "}",
      "[data-theme='dark'] .pt-tour-quick {",
      "  background: #1d1d27; border-color: rgba(255,255,255,0.08); color: #ececf2;",
      "}",
      ".pt-tour-quick:hover {",
      "  background: var(--bg, #fff);",
      "  border-color: rgba(124, 58, 237, 0.45);",
      "  transform: translateY(-1px);",
      "}",
      "[data-theme='dark'] .pt-tour-quick:hover { background: #23232f; border-color: rgba(124,58,237,0.55); }",
      ".pt-tour-quick-icon {",
      "  flex: 0 0 28px; width: 28px; height: 28px;",
      "  display: grid; place-items: center;",
      "  background: linear-gradient(135deg, rgba(124,58,237,0.14), rgba(249,115,22,0.14));",
      "  color: #7c3aed; border-radius: 8px;",
      "  font-size: 13px;",
      "}",
      "[data-theme='dark'] .pt-tour-quick-icon { color: #c4a8ff; }",
      ".pt-tour-quick-text { flex: 1 1 auto; min-width: 0; }",
      ".pt-tour-quick-label { font-size: 13px; font-weight: 500; line-height: 1.3; }",
      ".pt-tour-quick-sub {",
      "  font-family: \"JetBrains Mono\", ui-monospace, \"SF Mono\", Menlo, monospace;",
      "  font-size: 10.5px; color: var(--ink-3, #6b6878); margin-top: 2px;",
      "  letter-spacing: 0.02em;",
      "}",
      "[data-theme='dark'] .pt-tour-quick-sub { color: rgba(236,236,242,0.5); }",
      ".pt-tour-quick-go {",
      "  color: var(--ink-3, #6b6878); font-size: 14px; flex: 0 0 auto;",
      "  transition: transform 140ms ease, color 140ms ease;",
      "}",
      ".pt-tour-quick:hover .pt-tour-quick-go { transform: translateX(3px); color: #7c3aed; }",
      // footer text on cover/finish
      ".pt-tour-footer {",
      "  font-size: 11.5px; line-height: 1.5;",
      "  color: var(--ink-3, #6b6878);",
      "  margin: -4px 0 14px;",
      "  padding-top: 12px;",
      "  border-top: 1px dashed var(--hairline, #ececef);",
      "}",
      "[data-theme='dark'] .pt-tour-footer { color: rgba(236,236,242,0.55); border-color: rgba(255,255,255,0.08); }",
      ".pt-tour-footer code, .pt-tour-footer strong {",
      "  font-family: \"JetBrains Mono\", ui-monospace, \"SF Mono\", Menlo, monospace;",
      "  font-size: 10.5px; font-weight: 500;",
      "  background: var(--surface, #f6f5f8); padding: 1px 5px; border-radius: 4px;",
      "  color: var(--ink, #0a0a0c);",
      "}",
      "[data-theme='dark'] .pt-tour-footer code, [data-theme='dark'] .pt-tour-footer strong {",
      "  background: rgba(255,255,255,0.06); color: #ececf2;",
      "}",
      // action row
      ".pt-tour-actions {",
      "  display: flex; align-items: center; justify-content: space-between;",
      "  gap: 10px; margin-top: 2px;",
      "}",
      ".pt-tour-actions-right { display: flex; gap: 8px; }",
      // buttons
      ".pt-tour-btn {",
      "  appearance: none; -webkit-appearance: none;",
      "  font: 500 13px/1 \"Inter\", system-ui, sans-serif;",
      "  padding: 10px 16px; border-radius: 8px; cursor: pointer;",
      "  border: 1px solid transparent;",
      "  background: transparent; color: var(--ink-2, #4a4a55);",
      "  display: inline-flex; align-items: center; gap: 6px;",
      "  transition: background 120ms ease, color 120ms ease, border-color 120ms ease, transform 120ms ease;",
      "}",
      ".pt-tour-btn:focus-visible {",
      "  outline: 2px solid #7c3aed; outline-offset: 2px;",
      "  box-shadow: 0 0 0 4px rgba(124,58,237,0.30);",
      "}",
      ".pt-tour-btn-primary {",
      "  background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);",
      "  color: #fff;",
      "  border-color: rgba(124, 58, 237, 0.40);",
      "  box-shadow: 0 6px 18px -6px rgba(124, 58, 237, 0.55),",
      "              0 1px 0 rgba(255,255,255,0.18) inset;",
      "}",
      ".pt-tour-btn-primary:hover { transform: translateY(-1px); box-shadow: 0 10px 22px -8px rgba(124, 58, 237, 0.65), 0 1px 0 rgba(255,255,255,0.20) inset; }",
      ".pt-tour-btn-primary:active { transform: translateY(0); }",
      ".pt-tour-btn-secondary {",
      "  background: var(--surface, #f6f5f8);",
      "  border-color: var(--hairline-2, #d8d6dc);",
      "  color: var(--ink, #0a0a0c);",
      "}",
      ".pt-tour-btn-secondary:hover { background: var(--bg, #fff); border-color: var(--ink-3, #6b6878); }",
      "[data-theme='dark'] .pt-tour-btn-secondary {",
      "  background: #232330; border-color: rgba(255,255,255,0.10); color: #ececf2;",
      "}",
      "[data-theme='dark'] .pt-tour-btn-secondary:hover { background: #2c2c3c; }",
      ".pt-tour-btn-ghost { color: var(--ink-3, #6b6878); }",
      ".pt-tour-btn-ghost:hover { color: var(--ink, #0a0a0c); background: var(--surface, #f6f5f8); }",
      "[data-theme='dark'] .pt-tour-btn-ghost { color: rgba(236,236,242,0.65); }",
      "[data-theme='dark'] .pt-tour-btn-ghost:hover { color: #ececf2; background: rgba(255,255,255,0.06); }",
      // progress rail
      ".pt-tour-rail {",
      "  display: flex; gap: 4px; padding: 12px 22px 0;",
      "}",
      ".pt-tour-rail-seg {",
      "  flex: 1 1 0; height: 3px; border-radius: 2px;",
      "  background: var(--hairline, #ececef);",
      "  position: relative; overflow: hidden;",
      "  transition: background 200ms ease;",
      "}",
      "[data-theme='dark'] .pt-tour-rail-seg { background: rgba(255,255,255,0.10); }",
      ".pt-tour-rail-seg.is-done {",
      "  background: linear-gradient(90deg, #7c3aed, #f97316);",
      "}",
      ".pt-tour-rail-seg.is-active {",
      "  background: linear-gradient(90deg, #7c3aed, #ec4899);",
      "  height: 4px; margin-top: -0.5px;",
      "}",
      ".pt-tour-rail-seg.is-active::after {",
      "  content: \"\"; position: absolute; inset: 0;",
      "  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);",
      "  transform: translateX(-100%);",
      "  animation: pt-tour-rail-shine 1.6s ease-in-out infinite;",
      "}",
      "@keyframes pt-tour-rail-shine {",
      "  0%   { transform: translateX(-100%); }",
      "  60%  { transform: translateX(120%); }",
      "  100% { transform: translateX(120%); }",
      "}",
      // bottom keyhint row
      ".pt-tour-keyhint {",
      "  display: flex; align-items: center; gap: 10px;",
      "  padding: 10px 22px 14px;",
      "  font-size: 10.5px; color: var(--ink-3, #6b6878);",
      "  font-family: \"JetBrains Mono\", ui-monospace, \"SF Mono\", Menlo, monospace;",
      "  letter-spacing: 0.04em;",
      "}",
      "[data-theme='dark'] .pt-tour-keyhint { color: rgba(236,236,242,0.5); }",
      ".pt-tour-keyhint .pt-tour-keychip {",
      "  display: inline-flex; gap: 4px; align-items: center;",
      "}",
      ".pt-tour-keyhint kbd, .pt-tour-body kbd {",
      "  display: inline-block; padding: 1px 5px;",
      "  background: var(--surface, #f6f5f8);",
      "  border: 1px solid var(--hairline-2, #d8d6dc);",
      "  border-bottom-width: 2px;",
      "  border-radius: 4px;",
      "  font-family: \"JetBrains Mono\", ui-monospace, \"SF Mono\", Menlo, monospace;",
      "  font-size: 10.5px; line-height: 1.3;",
      "  color: var(--ink, #0a0a0c);",
      "}",
      "[data-theme='dark'] .pt-tour-keyhint kbd, [data-theme='dark'] .pt-tour-body kbd {",
      "  background: #1f1f2c; color: #ececf2; border-color: rgba(255,255,255,0.10);",
      "}",
      ".pt-tour-kbd-lg {",
      "  display: inline-block; padding: 4px 10px;",
      "  margin: 0 2px;",
      "  background: linear-gradient(135deg, rgba(124,58,237,0.10), rgba(249,115,22,0.10));",
      "  border: 1px solid rgba(124,58,237,0.30);",
      "  border-bottom-width: 2px;",
      "  border-radius: 8px;",
      "  font-family: \"JetBrains Mono\", ui-monospace, \"SF Mono\", Menlo, monospace;",
      "  font-size: 22px; font-weight: 500; line-height: 1;",
      "  color: var(--ink, #0a0a0c);",
      "  vertical-align: middle;",
      "}",
      "[data-theme='dark'] .pt-tour-kbd-lg { color: #f5f5fa; }",
      // dismiss × in panel
      ".pt-tour-dismiss {",
      "  position: absolute; top: 12px; right: 12px;",
      "  width: 26px; height: 26px;",
      "  display: grid; place-items: center;",
      "  background: transparent; border: 1px solid transparent;",
      "  border-radius: 6px; cursor: pointer;",
      "  color: var(--ink-3, #6b6878); font-size: 16px; line-height: 1;",
      "  transition: background 120ms, border-color 120ms, color 120ms;",
      "}",
      ".pt-tour-dismiss:hover {",
      "  background: var(--surface, #f6f5f8);",
      "  color: var(--ink, #0a0a0c);",
      "  border-color: var(--hairline-2, #d8d6dc);",
      "}",
      "[data-theme='dark'] .pt-tour-dismiss { color: rgba(236,236,242,0.55); }",
      "[data-theme='dark'] .pt-tour-dismiss:hover { background: rgba(255,255,255,0.06); color: #ececf2; border-color: rgba(255,255,255,0.10); }",
      // tour-launch nav button (added dynamically)
      ".pt-tour-launch {",
      "  font: 500 12px/1 \"Inter\", system-ui, sans-serif;",
      "  padding: 6px 10px; border-radius: 6px;",
      "  background: linear-gradient(135deg, rgba(124,58,237,0.10), rgba(249,115,22,0.08));",
      "  border: 1px solid rgba(124,58,237,0.30);",
      "  color: var(--ink, #0a0a0c);",
      "  cursor: pointer;",
      "  display: inline-flex; align-items: center; gap: 5px;",
      "  transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;",
      "}",
      ".pt-tour-launch:hover {",
      "  background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(249,115,22,0.14));",
      "  border-color: rgba(124,58,237,0.50);",
      "  transform: translateY(-1px);",
      "}",
      "[data-theme='dark'] .pt-tour-launch { color: #ececf2; }",
      ".pt-tour-launch::before {",
      "  content: \"\"; width: 6px; height: 6px; border-radius: 999px;",
      "  background: linear-gradient(135deg, #7c3aed, #f97316);",
      "  box-shadow: 0 0 6px rgba(124, 58, 237, 0.55);",
      "}",
      // mobile tweaks
      "@media (max-width: 560px) {",
      "  .pt-tour-panel { width: calc(100vw - 24px) !important; }",
      "  .pt-tour-panel.is-cover { width: calc(100vw - 24px); }",
      "  .pt-tour-bullet { grid-template-columns: 110px 1fr; gap: 10px; }",
      "  .pt-tour-stat-value { font-size: 26px; }",
      "  .pt-tour-body { padding: 18px 18px 14px; }",
      "  .pt-tour-rail, .pt-tour-keyhint { padding-left: 18px; padding-right: 18px; }",
      "  .pt-tour-launch .pt-tour-launch-label { display: none; }",
      "}",
      // reduced-motion: kill the rail shine + ring pulse; transitions remain
      // but page-level reduced-motion rule already collapses them.
      "@media (prefers-reduced-motion: reduce) {",
      "  .pt-tour-rail-seg.is-active::after, .pt-tour-stage.has-cutout .pt-tour-ring,",
      "  .pt-tour-eyebrow .pt-tour-pulse { animation: none !important; }",
      "}"
    ].join("\n");
    document.head.appendChild(css);
  }

  // ─────────────────────────────────────────────────────────────────────
  //  STAGE (SVG overlay with mask cutout)
  // ─────────────────────────────────────────────────────────────────────

  function buildStage() {
    var SVG_NS = "http://www.w3.org/2000/svg";
    var stage = document.createElementNS(SVG_NS, "svg");
    stage.classList.add("pt-tour-stage");
    stage.setAttribute("aria-hidden", "true");
    stage.setAttribute("xmlns", SVG_NS);
    stage.setAttribute("preserveAspectRatio", "none");

    var defs = document.createElementNS(SVG_NS, "defs");

    // Mask with white-everywhere + black cutout rect
    var mask = document.createElementNS(SVG_NS, "mask");
    mask.setAttribute("id", "pt-tour-mask");
    var maskBg = document.createElementNS(SVG_NS, "rect");
    maskBg.setAttribute("x", "0"); maskBg.setAttribute("y", "0");
    maskBg.setAttribute("width", "100%"); maskBg.setAttribute("height", "100%");
    maskBg.setAttribute("fill", "#ffffff");
    var cutout = document.createElementNS(SVG_NS, "rect");
    cutout.classList.add("pt-tour-cutout");
    cutout.setAttribute("rx", String(SPOTLIGHT_RADIUS));
    cutout.setAttribute("ry", String(SPOTLIGHT_RADIUS));
    cutout.setAttribute("x", "-100"); cutout.setAttribute("y", "-100");
    cutout.setAttribute("width", "0"); cutout.setAttribute("height", "0");
    mask.appendChild(maskBg);
    mask.appendChild(cutout);

    // Glow gradient for ring stroke
    var grad = document.createElementNS(SVG_NS, "linearGradient");
    grad.setAttribute("id", "pt-tour-glow");
    grad.setAttribute("x1", "0%"); grad.setAttribute("y1", "0%");
    grad.setAttribute("x2", "100%"); grad.setAttribute("y2", "100%");
    [["0%", "#7c3aed"], ["50%", "#ec4899"], ["100%", "#f97316"]].forEach(function (s) {
      var st = document.createElementNS(SVG_NS, "stop");
      st.setAttribute("offset", s[0]); st.setAttribute("stop-color", s[1]);
      grad.appendChild(st);
    });

    // Radial bloom under the cutout — soft purple→orange glow
    var bloom = document.createElementNS(SVG_NS, "radialGradient");
    bloom.setAttribute("id", "pt-tour-bloom");
    bloom.setAttribute("cx", "50%"); bloom.setAttribute("cy", "50%"); bloom.setAttribute("r", "50%");
    [["0%", "rgba(124,58,237,0.55)"], ["55%", "rgba(249,115,22,0.18)"], ["100%", "rgba(249,115,22,0)"]].forEach(function (s) {
      var st = document.createElementNS(SVG_NS, "stop");
      st.setAttribute("offset", s[0]); st.setAttribute("stop-color", s[1]);
      bloom.appendChild(st);
    });

    defs.appendChild(mask);
    defs.appendChild(grad);
    defs.appendChild(bloom);
    stage.appendChild(defs);

    // Bloom rect (rendered behind veil, faintly)
    var bloomRect = document.createElementNS(SVG_NS, "rect");
    bloomRect.classList.add("pt-tour-bloom");
    bloomRect.setAttribute("fill", "url(#pt-tour-bloom)");
    bloomRect.setAttribute("x", "0"); bloomRect.setAttribute("y", "0");
    bloomRect.setAttribute("width", "0"); bloomRect.setAttribute("height", "0");
    stage.appendChild(bloomRect);

    // Dark veil masked by the cutout
    var veil = document.createElementNS(SVG_NS, "rect");
    veil.classList.add("pt-tour-veil");
    veil.setAttribute("x", "0"); veil.setAttribute("y", "0");
    veil.setAttribute("width", "100%"); veil.setAttribute("height", "100%");
    veil.setAttribute("mask", "url(#pt-tour-mask)");
    stage.appendChild(veil);

    // Animated glow ring outlining the cutout
    var ring = document.createElementNS(SVG_NS, "rect");
    ring.classList.add("pt-tour-ring");
    ring.setAttribute("rx", String(SPOTLIGHT_RADIUS));
    ring.setAttribute("ry", String(SPOTLIGHT_RADIUS));
    ring.setAttribute("x", "-100"); ring.setAttribute("y", "-100");
    ring.setAttribute("width", "0"); ring.setAttribute("height", "0");
    stage.appendChild(ring);

    document.body.appendChild(stage);
    requestAnimationFrame(function () { stage.classList.add("is-open"); });
    return { stage: stage, cutout: cutout, ring: ring, bloom: bloomRect };
  }

  function updateSpotlight(stageRefs, rect) {
    var c = stageRefs.cutout, r = stageRefs.ring, b = stageRefs.bloom, s = stageRefs.stage;
    if (!rect) {
      s.classList.remove("has-cutout");
      c.setAttribute("x", "-100"); c.setAttribute("y", "-100");
      c.setAttribute("width", "0"); c.setAttribute("height", "0");
      r.setAttribute("x", "-100"); r.setAttribute("y", "-100");
      r.setAttribute("width", "0"); r.setAttribute("height", "0");
      b.setAttribute("width", "0"); b.setAttribute("height", "0");
      return;
    }
    var x = rect.left - SPOTLIGHT_PAD;
    var y = rect.top - SPOTLIGHT_PAD;
    var w = rect.width + SPOTLIGHT_PAD * 2;
    var h = rect.height + SPOTLIGHT_PAD * 2;
    c.setAttribute("x", String(x)); c.setAttribute("y", String(y));
    c.setAttribute("width", String(w)); c.setAttribute("height", String(h));
    r.setAttribute("x", String(x)); r.setAttribute("y", String(y));
    r.setAttribute("width", String(w)); r.setAttribute("height", String(h));
    // bloom is wider — softer halo around the cutout
    var bx = x - 120, by = y - 120, bw = w + 240, bh = h + 240;
    b.setAttribute("x", String(bx)); b.setAttribute("y", String(by));
    b.setAttribute("width", String(bw)); b.setAttribute("height", String(bh));
    s.classList.add("has-cutout");
  }

  // ─────────────────────────────────────────────────────────────────────
  //  PANEL CONTENT BUILDERS
  // ─────────────────────────────────────────────────────────────────────

  function escapeAttr(s) {
    return String(s).replace(/[<>&"']/g, function (c) {
      return ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  function buildEyebrow(step, idx, total) {
    var pieces = [];
    pieces.push('<span class="pt-tour-eyebrow">');
    if (step.kind === "cover") {
      pieces.push('<span class="pt-tour-pulse" aria-hidden="true"></span>');
    } else if (step.kind === "anchor" && step.step) {
      pieces.push('<span class="pt-tour-step-num">' + step.step + ' / ' + String(total - 2).padStart(2, "0") + '</span>');
    }
    pieces.push('<span>' + (step.eyebrow || "") + '</span>');
    pieces.push('</span>');
    return pieces.join("");
  }

  function buildBullets(step) {
    if (!step.bullets || !step.bullets.length) return "";
    var rows = step.bullets.map(function (b) {
      return '<li class="pt-tour-bullet">' +
             '  <span class="pt-tour-bullet-key">' + b[0] + '</span>' +
             '  <span class="pt-tour-bullet-val">' + b[1] + '</span>' +
             '</li>';
    }).join("");
    return '<ul class="pt-tour-bullets">' + rows + '</ul>';
  }

  function buildStats(step) {
    if (!step.stats) return "";
    var cells = step.stats.map(function (s, i) {
      return '<div class="pt-tour-stat">' +
             '  <div class="pt-tour-stat-value" data-pt-tour-counter="' + s.value + '" data-pt-tour-suffix="' + (s.suffix || "") + '">0</div>' +
             '  <div class="pt-tour-stat-label">' + s.label + '</div>' +
             '</div>';
    }).join("");
    return '<div class="pt-tour-stats">' + cells + '</div>';
  }

  function buildQuicks(step) {
    if (!step.quickStarts) return "";
    var icons = ["▸", "?", "⌘"];
    var rows = step.quickStarts.map(function (q, i) {
      return '<button type="button" class="pt-tour-quick" data-pt-tour-quick="' + q.action + '">' +
             '  <span class="pt-tour-quick-icon" aria-hidden="true">' + (icons[i] || "▸") + '</span>' +
             '  <span class="pt-tour-quick-text">' +
             '    <span class="pt-tour-quick-label">' + q.label + '</span>' +
             '    <span class="pt-tour-quick-sub">' + q.sub + '</span>' +
             '  </span>' +
             '  <span class="pt-tour-quick-go" aria-hidden="true">→</span>' +
             '</button>';
    }).join("");
    return '<div class="pt-tour-quicks">' + rows + '</div>';
  }

  function buildRail(idx, total) {
    // Skip cover (0) and finish (total-1) — they're meta-steps.
    // Rail shows progress through the substantive anchored steps.
    var n = total;
    var segs = [];
    for (var i = 0; i < n; i++) {
      var cls = "pt-tour-rail-seg";
      if (i < idx) cls += " is-done";
      else if (i === idx) cls += " is-active";
      segs.push('<div class="' + cls + '"></div>');
    }
    return '<div class="pt-tour-rail" aria-hidden="true">' + segs.join("") + '</div>';
  }

  function buildKeyhint(idx, total) {
    var backVisible = idx > 0;
    var pieces = [];
    pieces.push('<div class="pt-tour-keyhint" aria-hidden="true">');
    if (backVisible) {
      pieces.push('<span class="pt-tour-keychip"><kbd>←</kbd> back</span>');
    }
    pieces.push('<span class="pt-tour-keychip"><kbd>→</kbd> next</span>');
    pieces.push('<span class="pt-tour-keychip"><kbd>Esc</kbd> close</span>');
    pieces.push('</div>');
    return pieces.join("");
  }

  function buildActions(step, idx, total) {
    var isLast = idx === total - 1;
    var isCover = step.kind === "cover";
    var primaryLbl = isCover ? (step.primary || "Take the tour →")
                   : isLast  ? (step.primary || "Start exploring")
                             : "Next →";
    var secondaryLbl = isCover ? (step.secondary || "Skip")
                     : isLast  ? ""
                               : "Back";
    var skipLbl = (!isCover && !isLast) ? "Skip" : "";

    var pieces = [];
    pieces.push('<div class="pt-tour-actions">');
    if (skipLbl) {
      pieces.push('<button type="button" class="pt-tour-btn pt-tour-btn-ghost" data-pt-tour-action="skip">' + skipLbl + '</button>');
    } else {
      pieces.push('<span></span>');
    }
    pieces.push('<div class="pt-tour-actions-right">');
    if (secondaryLbl) {
      var actName = isCover ? "skip" : "back";
      pieces.push('<button type="button" class="pt-tour-btn pt-tour-btn-secondary" data-pt-tour-action="' + actName + '">' + secondaryLbl + '</button>');
    }
    pieces.push('<button type="button" class="pt-tour-btn pt-tour-btn-primary" data-pt-tour-action="next">' + primaryLbl + '</button>');
    pieces.push('</div>');
    pieces.push('</div>');
    return pieces.join("");
  }

  function buildPanel(step, idx, total) {
    var panel = document.createElement("div");
    panel.className = "pt-tour-panel";
    if (step.kind === "cover")  panel.classList.add("is-cover");
    if (step.kind === "finish") panel.classList.add("is-finish");
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    panel.setAttribute("aria-labelledby", "pt-tour-title-" + idx);

    var inner = [];
    inner.push('<div class="pt-tour-body">');
    inner.push('  <button type="button" class="pt-tour-dismiss" aria-label="Close tour" data-pt-tour-action="skip">×</button>');
    inner.push('  ' + buildEyebrow(step, idx, total));
    inner.push('  <h2 class="pt-tour-title" id="pt-tour-title-' + idx + '">' + step.title + '</h2>');
    if (step.lede)        inner.push('  <p class="pt-tour-lede">' + step.lede + '</p>');
    if (step.stats)       inner.push('  ' + buildStats(step));
    if (step.bullets)     inner.push('  ' + buildBullets(step));
    if (step.hint)        inner.push('  <div class="pt-tour-hint">' + step.hint + '</div>');
    if (step.quickStarts) inner.push('  ' + buildQuicks(step));
    if (step.footer)      inner.push('  <div class="pt-tour-footer">' + step.footer + '</div>');
    inner.push('  ' + buildActions(step, idx, total));
    inner.push('</div>');
    if (step.kind === "anchor") {
      inner.push(buildRail(idx, total));
      inner.push(buildKeyhint(idx, total));
    }
    inner.push('<div class="pt-tour-tip" aria-hidden="true"></div>');
    panel.innerHTML = inner.join("\n");

    document.body.appendChild(panel);
    requestAnimationFrame(function () { panel.classList.add("is-shown"); });
    return panel;
  }

  // ─────────────────────────────────────────────────────────────────────
  //  PANEL POSITIONING
  // ─────────────────────────────────────────────────────────────────────

  function positionPanel(panel, anchorEl, preferredPos) {
    // Reset tip classes
    panel.classList.remove("tip-above", "tip-below", "tip-left", "tip-right");
    var vw = window.innerWidth;
    var vh = window.innerHeight;
    var pw = panel.offsetWidth;
    var ph = panel.offsetHeight;
    var margin = 12;

    if (!anchorEl) {
      // center
      var cl = Math.round((vw - pw) / 2);
      var ct = Math.round((vh - ph) / 2);
      panel.style.left = clamp(cl, margin, vw - pw - margin) + "px";
      panel.style.top  = clamp(ct, margin, vh - ph - margin) + "px";
      var tip = panel.querySelector(".pt-tour-tip");
      if (tip) tip.style.display = "none";
      return;
    }
    var tipEl = panel.querySelector(".pt-tour-tip");
    if (tipEl) tipEl.style.display = "";

    var r = anchorEl.getBoundingClientRect();
    var pos = preferredPos || "below";
    var left, top;
    var spaceBelow = vh - r.bottom;
    var spaceAbove = r.top;
    var spaceRight = vw - r.right;
    var spaceLeft  = r.left;

    // Auto-flip if not enough room in preferred position
    if (pos === "below" && spaceBelow < ph + PANEL_GAP + margin && spaceAbove > spaceBelow) pos = "above";
    if (pos === "above" && spaceAbove < ph + PANEL_GAP + margin && spaceBelow > spaceAbove) pos = "below";
    if (pos === "right" && spaceRight < pw + PANEL_GAP + margin && spaceLeft > spaceRight) pos = "left";
    if (pos === "left"  && spaceLeft  < pw + PANEL_GAP + margin && spaceRight > spaceLeft) pos = "right";

    if (pos === "below") {
      top = r.bottom + PANEL_GAP;
      left = r.left + r.width / 2 - pw / 2;
      panel.classList.add("tip-above");
    } else if (pos === "above") {
      top = r.top - ph - PANEL_GAP;
      left = r.left + r.width / 2 - pw / 2;
      panel.classList.add("tip-below");
    } else if (pos === "right") {
      top = r.top + r.height / 2 - ph / 2;
      left = r.right + PANEL_GAP;
      panel.classList.add("tip-left");
    } else {
      top = r.top + r.height / 2 - ph / 2;
      left = r.left - pw - PANEL_GAP;
      panel.classList.add("tip-right");
    }

    left = clamp(left, margin, vw - pw - margin);
    top  = clamp(top, margin, vh - ph - margin);
    panel.style.left = Math.round(left) + "px";
    panel.style.top  = Math.round(top) + "px";

    // Position the tip arrow at the anchor's center
    if (tipEl) {
      if (pos === "below" || pos === "above") {
        var tipX = (r.left + r.width / 2) - left - 7;
        tipX = clamp(tipX, 18, pw - 32);
        tipEl.style.left = tipX + "px";
        tipEl.style.right = "";
        tipEl.style.top = "";
        tipEl.style.bottom = "";
      } else {
        var tipY = (r.top + r.height / 2) - top - 7;
        tipY = clamp(tipY, 18, ph - 32);
        tipEl.style.top = tipY + "px";
        tipEl.style.bottom = "";
        tipEl.style.left = "";
        tipEl.style.right = "";
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────
  //  COUNTER ANIMATION
  // ─────────────────────────────────────────────────────────────────────

  function animateCounter(el, target, durationMs) {
    var t0 = performance.now();
    var reduce = reducedMotion();
    var suffix = el.getAttribute("data-pt-tour-suffix") || "";
    if (reduce) {
      el.textContent = target.toLocaleString() + suffix;
      return;
    }
    function step(now) {
      var t = clamp((now - t0) / durationMs, 0, 1);
      var eased = easeOutCubic(t);
      var v = Math.round(target * eased);
      el.textContent = v.toLocaleString() + suffix;
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // ─────────────────────────────────────────────────────────────────────
  //  QUICK-START ACTIONS (final step)
  // ─────────────────────────────────────────────────────────────────────

  function runQuickAction(actionName) {
    finish();   // close the tour first so the action's UI is unobstructed
    setTimeout(function () {
      try {
        switch (actionName) {
          case "openStrategies":
            var sb = document.querySelector('.mode-switch .mode-btn[data-mode="strategies"]');
            if (sb) sb.click();
            break;
          case "askWhy":
            var rb = document.querySelector('.mode-switch .mode-btn[data-mode="regression"]');
            if (rb) rb.click();
            setTimeout(function () {
              var t = document.getElementById("pfm-wow-ticker");
              if (t) { t.focus(); t.select(); }
            }, 280);
            break;
          case "openCmdK":
            if (window.pfmCmdk && typeof window.pfmCmdk.open === "function") {
              window.pfmCmdk.open();
            }
            break;
        }
      } catch (e) { /* noop */ }
    }, 60);
  }

  // ─────────────────────────────────────────────────────────────────────
  //  STEP DRIVER
  // ─────────────────────────────────────────────────────────────────────

  function renderStep(idx) {
    var step = STEPS[idx];
    if (!step) { finish(); return; }
    var prevPanel = state && state.panel;
    var prevAnchor = state && state.anchorEl;

    // Build a fresh panel for this step
    var anchor = step.kind === "anchor" ? findAnchor(step.anchor) : null;
    var panel = buildPanel(step, idx, STEPS.length);
    positionPanel(panel, anchor, step.preferredPos);

    // Wire actions
    panel.querySelectorAll("[data-pt-tour-action]").forEach(function (btn) {
      var action = btn.getAttribute("data-pt-tour-action");
      btn.addEventListener("click", function (ev) {
        ev.preventDefault();
        if (action === "next") next();
        else if (action === "back") back();
        else if (action === "skip") skip();
      });
    });
    panel.querySelectorAll("[data-pt-tour-quick]").forEach(function (btn) {
      btn.addEventListener("click", function (ev) {
        ev.preventDefault();
        runQuickAction(btn.getAttribute("data-pt-tour-quick"));
      });
    });

    // Counter animation for cover step
    var counters = panel.querySelectorAll("[data-pt-tour-counter]");
    counters.forEach(function (c, i) {
      var target = parseInt(c.getAttribute("data-pt-tour-counter"), 10) || 0;
      setTimeout(function () {
        animateCounter(c, target, 1100 + i * 120);
      }, 220 + i * 90);
    });

    // Update spotlight (or clear)
    if (state) {
      if (anchor) {
        updateSpotlight(state.stageRefs, anchor.getBoundingClientRect());
      } else {
        updateSpotlight(state.stageRefs, null);
      }
    }

    // Focus the primary CTA so keyboard users land on it
    setTimeout(function () {
      var p = panel.querySelector(".pt-tour-btn-primary");
      if (p) { try { p.focus({ preventScroll: true }); } catch (e) { try { p.focus(); } catch (_) {} } }
    }, 80);

    // Tear down the previous panel with a fade
    if (prevPanel && prevPanel !== panel) {
      prevPanel.classList.remove("is-shown");
      setTimeout(function () {
        if (prevPanel.parentNode) prevPanel.parentNode.removeChild(prevPanel);
      }, 220);
    }

    state.panel = panel;
    state.anchorEl = anchor;
    state.idx = idx;
  }

  function next() {
    if (!state) return;
    if (state.idx + 1 >= STEPS.length) { finish(); return; }
    renderStep(state.idx + 1);
  }

  function back() {
    if (!state) return;
    if (state.idx <= 0) return;
    renderStep(state.idx - 1);
  }

  function skip() {
    finish();
  }

  function finish() {
    if (!state) return;
    markShown();
    var s = state;
    state = null;
    if (s.keyHandler)    document.removeEventListener("keydown", s.keyHandler, true);
    if (s.resizeHandler) window.removeEventListener("resize", s.resizeHandler);
    if (s.scrollHandler) window.removeEventListener("scroll", s.scrollHandler, true);
    if (s.panel) {
      s.panel.classList.remove("is-shown");
      setTimeout(function () {
        if (s.panel && s.panel.parentNode) s.panel.parentNode.removeChild(s.panel);
      }, 220);
    }
    if (s.stageRefs && s.stageRefs.stage) {
      s.stageRefs.stage.classList.remove("is-open");
      setTimeout(function () {
        if (s.stageRefs.stage.parentNode) s.stageRefs.stage.parentNode.removeChild(s.stageRefs.stage);
      }, 260);
    }
    if (s.prevFocus && typeof s.prevFocus.focus === "function") {
      try { s.prevFocus.focus(); } catch (e) {}
    }
  }

  function reposition() {
    if (!state) return;
    var step = STEPS[state.idx];
    if (!step) return;
    var anchor = step.kind === "anchor" ? findAnchor(step.anchor) : null;
    state.anchorEl = anchor;
    if (state.panel) positionPanel(state.panel, anchor, step.preferredPos);
    if (state.stageRefs) {
      if (anchor) updateSpotlight(state.stageRefs, anchor.getBoundingClientRect());
      else updateSpotlight(state.stageRefs, null);
    }
  }

  // ─────────────────────────────────────────────────────────────────────
  //  ENTRY POINT
  // ─────────────────────────────────────────────────────────────────────

  function start(force) {
    if (state) return;             // already running
    if (!force && hasShown()) return;
    injectStyles();
    var stageRefs = buildStage();
    state = {
      stageRefs: stageRefs,
      panel: null,
      anchorEl: null,
      idx: 0,
      prevFocus: document.activeElement,
      keyHandler: null,
      resizeHandler: null,
      scrollHandler: null
    };
    // Keyboard
    var keyHandler = function (ev) {
      if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); skip(); return; }
      if (ev.key === "ArrowRight" || ev.key === "Enter") {
        // Allow Enter to advance unless it's on a quick-action button
        var t = ev.target;
        if (ev.key === "Enter" && t && t.hasAttribute && t.hasAttribute("data-pt-tour-quick")) return;
        ev.preventDefault(); next(); return;
      }
      if (ev.key === "ArrowLeft") { ev.preventDefault(); back(); return; }
    };
    document.addEventListener("keydown", keyHandler, true);
    state.keyHandler = keyHandler;
    // Resize / scroll → reposition
    var rzId = null;
    var resizeHandler = function () {
      if (rzId) cancelAnimationFrame(rzId);
      rzId = requestAnimationFrame(reposition);
    };
    window.addEventListener("resize", resizeHandler);
    window.addEventListener("scroll", resizeHandler, true);
    state.resizeHandler = resizeHandler;
    state.scrollHandler = resizeHandler;

    renderStep(0);
  }

  // ─────────────────────────────────────────────────────────────────────
  //  NAV LAUNCH BUTTON ("Tour" iconbutton next to "?")
  // ─────────────────────────────────────────────────────────────────────

  function injectNavLaunch() {
    try {
      // Avoid duplicate insertion across HMR / repeat boot
      if (document.querySelector(".pt-tour-launch")) return;
      var shortcuts = document.getElementById("pfm-shortcuts-btn");
      if (!shortcuts || !shortcuts.parentNode) return;
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pt-tour-launch";
      btn.setAttribute("aria-label", "Restart product tour");
      btn.title = "Restart product tour";
      btn.innerHTML = '<span class="pt-tour-launch-label">Tour</span>';
      btn.addEventListener("click", function () { start(true); });
      // Insert just before the "?" button so it reads:
      //   Search ⌘K · Alerts · Theme · Tour · ?
      shortcuts.parentNode.insertBefore(btn, shortcuts);
    } catch (e) { /* noop */ }
  }

  // ─────────────────────────────────────────────────────────────────────
  //  BOOT
  // ─────────────────────────────────────────────────────────────────────

  function shouldForceFromUrl() {
    try {
      var p = new URLSearchParams(window.location.search).get("tour");
      return p && p !== "0" && p !== "false";
    } catch (e) { return false; }
  }

  function boot() {
    injectNavLaunch();
    // ?tour=1 ALWAYS forces a replay
    if (shouldForceFromUrl()) {
      // Preempt the legacy bottom welcome banner — our tour says the same thing.
      try { localStorage.setItem("pfm:welcomed", "1"); } catch (e) {}
      setTimeout(function () { start(true); }, 600);
      return;
    }
    // Wave-7 (2026-05-19): NO auto-start on first visit. The walkthrough audit
    // found this premium tour stacking on top of the legacy coachmark and
    // blocking the entire cold-load experience. The tour is still discoverable
    // via the nav "Tour" link (injected by injectNavLaunch above) and the
    // ?tour=1 URL param. Suppress the auto-fire so first paint is unblocked.
    try { localStorage.setItem("pfm:welcomed", "1"); } catch (e) {}
  }

  // ─────────────────────────────────────────────────────────────────────
  //  PUBLIC API
  // ─────────────────────────────────────────────────────────────────────

  window.pfmTour = {
    start: function (force) { start(force !== false); },
    reset: clearShown,
    isShown: hasShown
  };

  // Override the legacy inline tour entry so ?tour=1 + any code calling
  // pfmStartTour() lands on the new premium tour. The inline IIFE's
  // closure-bound _pfmRenderTourStep becomes unreachable; harmless.
  window.pfmStartTour = function (force) { start(force !== false); };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
