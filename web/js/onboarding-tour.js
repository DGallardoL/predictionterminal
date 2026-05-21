/* ============================================================
 * onboarding-tour.js  (T13, wave-10; extended W13-40 wave-13)
 *
 * Standalone 12-step product tour for the Prediction Terminal.
 * Exposes window.PFM.tour = { start, stop, reset, isCompleted }.
 *
 * MOUNT (handled by index-html-owner, NOT here):
 *   <link rel="stylesheet" href="css/tour.css">
 *   <script defer src="js/onboarding-tour.js"></script>
 *
 *   Then either:
 *     - Wire a help-menu button:  onclick="PFM.tour.start()"
 *     - Or auto-start once:
 *         if (!PFM.tour.isCompleted()) PFM.tour.start();
 *
 *   `PFM.tour.reset()` clears the localStorage completion flag
 *   so the tour will play again on next start().
 *
 * Storage:
 *   localStorage key `pfm:tour:completed` (truthy means seen).
 *
 * Keyboard:
 *   - Esc:        skip tour
 *   - ArrowRight / Enter: next
 *   - ArrowLeft:  back
 *   - Tab:        cycles tooltip buttons (focus trap)
 *
 * No external dependencies. Pure DOM. Idempotent.
 * ============================================================ */
(function () {
  "use strict";

  // ---------------------------------------------------------- constants
  var STORAGE_KEY = "pfm:tour:completed";
  var BACKDROP_CLASS = "pfm-tour-backdrop";
  var SPOTLIGHT_CLASS = "pfm-tour-spotlight";
  var TOOLTIP_CLASS = "pfm-tour-tooltip";
  var TOOLTIP_WIDTH = 280;
  var GAP = 16; // pixels between target and tooltip
  var VIEWPORT_PADDING = 12;

  // ---------------------------------------------------------- steps
  // Selector fallbacks are tried in order; first hit wins.
  // If none resolve, the tooltip floats centred (no spotlight).
  var STEPS = [
    {
      title: "Welcome to Prediction Terminal",
      body:
        "Three modes: Regression for factor models, Strategies for curated alphas, " +
        "Terminal for market data.",
      selectors: [
        "nav.top .inner",
        "nav.top",
        ".pfm-topnav",
        "header[role='banner']",
        "header"
      ]
    },
    {
      title: "Switch between modes",
      body:
        "Use these tabs to jump between Regression, Strategies, and Terminal. " +
        "Try Strategies first if you want to see curated alphas.",
      selectors: [
        ".mode-switch",
        "[data-mode-switch]",
        ".pfm-mode-tabs",
        "nav.top .inner .modes",
        "[role='tablist']"
      ]
    },
    {
      title: "Run a factor model",
      body:
        "Pick a stock ticker, add prediction-market factors, click Fit. " +
        "You'll get betas, t-stats, and a residual chart.",
      selectors: [
        "[data-mode-pane='regression'] form",
        "[data-mode-pane=regression] form",
        "#regression-form",
        "form#fit-form",
        "[data-mode-pane='regression'] .fit-form"
      ]
    },
    {
      title: "Jumps detection",
      body:
        "We detect large price jumps in prediction markets and tag the news " +
        "stories that explain them.",
      selectors: [
        "[data-section='jumps']",
        "#term-jumps",
        "[data-mode-pane='terminal'] [data-pane='jumps']",
        ".term-jumps",
        "#terminal-jumps"
      ]
    },
    {
      title: "Backtest the disagreement",
      body:
        "Backtest the \"news disagrees with market\" signal — sometimes the news " +
        "is right and the market hasn't caught up.",
      selectors: [
        "[data-section='jumps-backtest']",
        "#term-jumps-backtest",
        "[data-jump-backtest]",
        "a[href*='/jumps/'][href*='/backtest']",
        ".jump-backtest-link"
      ]
    },
    {
      title: "Jump clusters",
      body:
        "See which themes are moving today across many markets. " +
        "Clusters group jumps by shared news entities.",
      selectors: [
        "[data-section='jump-clusters']",
        "#term-jump-clusters",
        ".term-cluster-panel",
        "#terminal-cluster",
        "[data-pane='clusters']"
      ]
    },
    {
      title: "Sentiment leaderboard",
      body:
        "Markets ranked by 7-day news sentiment vs. price move. Big mismatches " +
        "are candidate trades.",
      selectors: [
        "[data-section='sentiment-leaderboard']",
        "#term-sentiment-leaderboard",
        ".term-sentlb",
        "#terminal-sentiment-leaderboard",
        "[data-pane='sentiment-leaderboard']"
      ]
    },
    {
      title: "Sentiment leaderboard panel",
      body:
        "The full leaderboard panel, backed by /terminal/sentiment-leaderboard, " +
        "lets you sort by mismatch magnitude and drill into any row.",
      selectors: [
        "[data-endpoint='/terminal/sentiment-leaderboard']",
        "[data-panel='sentiment-leaderboard']",
        "#sentiment-leaderboard-panel",
        ".sentiment-leaderboard-panel",
        "[data-section='sentiment-leaderboard'] .panel"
      ]
    },
    {
      title: "Pinboard",
      body:
        "Save fits for later comparison. Click the pin icon on any fit result to " +
        "stash it in the Pinboard drawer.",
      selectors: [
        "[data-result-pinner-toggle]",
        "#result-pinner-toggle",
        ".result-pinner-toggle",
        "[data-pinboard-toggle]",
        "button[aria-controls='result-pinner']"
      ]
    },
    {
      title: "Command palette",
      body:
        "Press ⌘K (Ctrl+K on Windows) to search factors and slugs from " +
        "anywhere in the app.",
      selectors: [
        "[data-cmdk-trigger]",
        "#cmdk-trigger",
        ".cmdk-trigger",
        "[data-command-palette]",
        "button[aria-keyshortcuts*='K' i]"
      ]
    },
    {
      title: "Theme toggle",
      body:
        "Sun/moon button in top-right switches dark mode. Click again to cycle " +
        "back to light or follow your system preference.",
      selectors: [
        "[data-theme-toggle]",
        "#theme-toggle",
        ".theme-toggle",
        "button[aria-label*='theme' i]",
        "nav.top .inner [data-theme]"
      ]
    },
    {
      title: "Strategy comparison",
      body:
        "Type /compare A B in the command palette to compare two strategies " +
        "side-by-side — PnL, Sharpe, drawdown, factor exposures.",
      selectors: [
        "[data-cmdk-trigger]",
        "#cmdk-trigger",
        ".cmdk-trigger",
        "[data-command-palette]",
        "[data-mode-pane='strategies']"
      ]
    }
  ];

  // ---------------------------------------------------------- state
  var state = {
    running: false,
    index: 0,
    backdrop: null,
    spotlight: null,
    tooltip: null,
    keyHandler: null,
    resizeHandler: null,
    prevFocus: null
  };

  // ---------------------------------------------------------- utils
  function $(sel, root) {
    try {
      return (root || document).querySelector(sel);
    } catch (e) {
      return null;
    }
  }

  function resolveTarget(step) {
    if (!step || !step.selectors) return null;
    for (var i = 0; i < step.selectors.length; i++) {
      var el = $(step.selectors[i]);
      if (el && isVisible(el)) return el;
    }
    return null;
  }

  function isVisible(el) {
    if (!el || !el.getClientRects) return false;
    var rects = el.getClientRects();
    if (rects.length === 0) return false;
    var style = window.getComputedStyle(el);
    if (style.visibility === "hidden" || style.display === "none") return false;
    return true;
  }

  function readStored() {
    try { return window.localStorage.getItem(STORAGE_KEY); }
    catch (e) { return null; }
  }
  function writeStored(value) {
    try { window.localStorage.setItem(STORAGE_KEY, value); }
    catch (e) { /* ignore quota / disabled storage */ }
  }
  function clearStored() {
    try { window.localStorage.removeItem(STORAGE_KEY); }
    catch (e) { /* ignore */ }
  }

  // ---------------------------------------------------------- DOM build
  function buildBackdrop() {
    var el = document.createElement("div");
    el.className = BACKDROP_CLASS;
    el.setAttribute("aria-hidden", "true");
    el.addEventListener("click", function (e) {
      // Click on backdrop = skip; ignore clicks bubbled from tooltip
      if (e.target === el) skip();
    });
    return el;
  }

  function buildSpotlight() {
    var el = document.createElement("div");
    el.className = SPOTLIGHT_CLASS;
    el.setAttribute("aria-hidden", "true");
    return el;
  }

  function buildTooltip() {
    var el = document.createElement("div");
    el.className = TOOLTIP_CLASS;
    el.setAttribute("role", "dialog");
    el.setAttribute("aria-modal", "true");
    el.setAttribute("aria-labelledby", "pfm-tour-title");
    el.setAttribute("aria-describedby", "pfm-tour-body");
    return el;
  }

  function renderTooltip() {
    var step = STEPS[state.index];
    var isFirst = state.index === 0;
    var isLast = state.index === STEPS.length - 1;
    var nextLabel = isLast ? "Done" : "Next";

    state.tooltip.innerHTML = "";

    var header = document.createElement("div");
    header.className = "pfm-tour-header";

    var h = document.createElement("h3");
    h.className = "pfm-tour-title";
    h.id = "pfm-tour-title";
    h.textContent = step.title;
    header.appendChild(h);

    var stepLabel = document.createElement("span");
    stepLabel.className = "pfm-tour-step";
    stepLabel.textContent = (state.index + 1) + " / " + STEPS.length;
    header.appendChild(stepLabel);

    state.tooltip.appendChild(header);

    var body = document.createElement("p");
    body.className = "pfm-tour-body";
    body.id = "pfm-tour-body";
    body.textContent = step.body;
    state.tooltip.appendChild(body);

    var actions = document.createElement("div");
    actions.className = "pfm-tour-actions";

    var skipBtn = document.createElement("button");
    skipBtn.type = "button";
    skipBtn.className = "pfm-tour-skip";
    skipBtn.textContent = "Skip";
    skipBtn.addEventListener("click", skip);
    actions.appendChild(skipBtn);

    var backBtn = document.createElement("button");
    backBtn.type = "button";
    backBtn.className = "pfm-tour-btn pfm-tour-btn--back";
    backBtn.textContent = "Back";
    backBtn.disabled = isFirst;
    backBtn.addEventListener("click", back);
    actions.appendChild(backBtn);

    var nextBtn = document.createElement("button");
    nextBtn.type = "button";
    nextBtn.className = "pfm-tour-btn pfm-tour-btn--next";
    nextBtn.textContent = nextLabel;
    nextBtn.addEventListener("click", next);
    actions.appendChild(nextBtn);

    state.tooltip.appendChild(actions);

    var progress = document.createElement("div");
    progress.className = "pfm-tour-progress";
    for (var i = 0; i < STEPS.length; i++) {
      var dot = document.createElement("span");
      dot.className = "pfm-tour-dot";
      if (i < state.index) dot.className += " pfm-tour-dot--done";
      else if (i === state.index) dot.className += " pfm-tour-dot--current";
      progress.appendChild(dot);
    }
    state.tooltip.appendChild(progress);

    // Defer focus until after positioning so transitions don't fight
    setTimeout(function () {
      if (state.running) nextBtn.focus();
    }, 30);
  }

  // ---------------------------------------------------------- positioning
  function positionFor(target) {
    if (!target) {
      // Float centred; hide spotlight
      state.spotlight.style.display = "none";
      state.tooltip.classList.add("pfm-tour-tooltip--floating");
      state.tooltip.removeAttribute("data-pos");
      state.tooltip.style.top = "";
      state.tooltip.style.left = "";
      return;
    }

    state.spotlight.style.display = "";
    state.tooltip.classList.remove("pfm-tour-tooltip--floating");

    var rect = target.getBoundingClientRect();
    var pad = 6;
    state.spotlight.style.top = (rect.top - pad) + "px";
    state.spotlight.style.left = (rect.left - pad) + "px";
    state.spotlight.style.width = (rect.width + pad * 2) + "px";
    state.spotlight.style.height = (rect.height + pad * 2) + "px";

    // Measure tooltip after content is rendered
    var ttRect = state.tooltip.getBoundingClientRect();
    var ttHeight = ttRect.height || 180;
    var ttWidth = TOOLTIP_WIDTH;

    var vpW = window.innerWidth;
    var vpH = window.innerHeight;

    var pos = "right";
    var top, left;

    // Prefer right
    if (rect.right + GAP + ttWidth + VIEWPORT_PADDING <= vpW) {
      pos = "right";
      left = rect.right + GAP;
      top = rect.top;
    } else if (rect.left - GAP - ttWidth - VIEWPORT_PADDING >= 0) {
      pos = "left";
      left = rect.left - GAP - ttWidth;
      top = rect.top;
    } else if (rect.bottom + GAP + ttHeight + VIEWPORT_PADDING <= vpH) {
      pos = "bottom";
      left = Math.max(VIEWPORT_PADDING, rect.left);
      top = rect.bottom + GAP;
    } else if (rect.top - GAP - ttHeight - VIEWPORT_PADDING >= 0) {
      pos = "top";
      left = Math.max(VIEWPORT_PADDING, rect.left);
      top = rect.top - GAP - ttHeight;
    } else {
      // Fallback: bottom, clamp to viewport
      pos = "bottom";
      left = VIEWPORT_PADDING;
      top = Math.min(vpH - ttHeight - VIEWPORT_PADDING, rect.bottom + GAP);
    }

    // Clamp top to viewport
    top = Math.max(VIEWPORT_PADDING, Math.min(top, vpH - ttHeight - VIEWPORT_PADDING));
    left = Math.max(VIEWPORT_PADDING, Math.min(left, vpW - ttWidth - VIEWPORT_PADDING));

    state.tooltip.setAttribute("data-pos", pos);
    state.tooltip.style.top = top + "px";
    state.tooltip.style.left = left + "px";
  }

  function scrollIntoView(target) {
    if (!target || typeof target.scrollIntoView !== "function") return;
    try {
      target.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
    } catch (e) {
      // Older browsers
      target.scrollIntoView();
    }
  }

  // ---------------------------------------------------------- step machine
  function showStep(idx) {
    if (idx < 0 || idx >= STEPS.length) return;
    state.index = idx;

    var step = STEPS[idx];
    var target = resolveTarget(step);

    renderTooltip();

    if (target) {
      scrollIntoView(target);
      // After smooth-scroll settles, reposition
      setTimeout(function () {
        if (state.running) positionFor(target);
      }, 280);
      // Position immediately too so it doesn't flash off-screen
      positionFor(target);
    } else {
      positionFor(null);
    }
  }

  function next() {
    if (state.index >= STEPS.length - 1) {
      complete();
    } else {
      showStep(state.index + 1);
    }
  }

  function back() {
    if (state.index > 0) showStep(state.index - 1);
  }

  function skip() {
    // Skipping still marks completed so it doesn't auto-replay every load
    writeStored("skipped@" + new Date().toISOString());
    teardown();
  }

  function complete() {
    writeStored("completed@" + new Date().toISOString());
    teardown();
  }

  // ---------------------------------------------------------- keyboard
  function onKeydown(e) {
    if (!state.running) return;
    if (e.key === "Escape") {
      e.preventDefault();
      skip();
      return;
    }
    if (e.key === "ArrowRight" || e.key === "Enter") {
      // Don't intercept Enter when the user is typing in a field
      if (e.key === "Enter") {
        var t = e.target;
        if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
        // If focus is on a tooltip button, let its native click fire
        if (t && t.classList && t.classList.contains("pfm-tour-btn")) return;
      }
      e.preventDefault();
      next();
      return;
    }
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      back();
      return;
    }
    if (e.key === "Tab") {
      // Focus trap within the tooltip
      if (!state.tooltip) return;
      var focusables = state.tooltip.querySelectorAll(
        "button:not(:disabled), [href], [tabindex]:not([tabindex='-1'])"
      );
      if (focusables.length === 0) return;
      var first = focusables[0];
      var last = focusables[focusables.length - 1];
      var active = document.activeElement;
      if (e.shiftKey && active === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      } else if (!state.tooltip.contains(active)) {
        // Bring focus back into the tooltip
        e.preventDefault();
        first.focus();
      }
    }
  }

  function onResize() {
    if (!state.running) return;
    var step = STEPS[state.index];
    var target = resolveTarget(step);
    positionFor(target);
  }

  // ---------------------------------------------------------- lifecycle
  function setup() {
    if (state.running) return;
    state.running = true;
    state.index = 0;
    state.prevFocus = document.activeElement;

    state.backdrop = buildBackdrop();
    state.spotlight = buildSpotlight();
    state.tooltip = buildTooltip();

    document.body.appendChild(state.backdrop);
    document.body.appendChild(state.spotlight);
    document.body.appendChild(state.tooltip);

    state.keyHandler = onKeydown;
    state.resizeHandler = onResize;
    document.addEventListener("keydown", state.keyHandler, true);
    window.addEventListener("resize", state.resizeHandler, true);
    window.addEventListener("scroll", state.resizeHandler, true);

    showStep(0);
  }

  function teardown() {
    state.running = false;
    if (state.keyHandler) {
      document.removeEventListener("keydown", state.keyHandler, true);
      state.keyHandler = null;
    }
    if (state.resizeHandler) {
      window.removeEventListener("resize", state.resizeHandler, true);
      window.removeEventListener("scroll", state.resizeHandler, true);
      state.resizeHandler = null;
    }
    [state.backdrop, state.spotlight, state.tooltip].forEach(function (el) {
      if (el && el.parentNode) el.parentNode.removeChild(el);
    });
    state.backdrop = null;
    state.spotlight = null;
    state.tooltip = null;
    if (state.prevFocus && typeof state.prevFocus.focus === "function") {
      try { state.prevFocus.focus(); } catch (e) { /* ignore */ }
    }
    state.prevFocus = null;
  }

  // ---------------------------------------------------------- public API
  function start() {
    if (state.running) return;
    setup();
  }
  function stop() {
    if (!state.running) return;
    teardown();
  }
  function reset() {
    clearStored();
  }
  function isCompleted() {
    return !!readStored();
  }

  window.PFM = window.PFM || {};
  window.PFM.tour = {
    start: start,
    stop: stop,
    reset: reset,
    isCompleted: isCompleted,
    // Exposed for debug / inspection only (do not rely on shape)
    _steps: STEPS
  };
})();
