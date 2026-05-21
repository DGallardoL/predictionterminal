/* ============================================================
 * quant-glossary.js  (W12-36, wave-12)
 *
 * Inline tooltips for Greek letters and quant terms. Walks text
 * nodes inside an attached root, wraps glossary matches in
 * <span class="qg-term">, and renders a single shared tooltip
 * card on hover / focus.
 *
 * Public API (mounted on window.PFM.glossary):
 *   define(term, definition, reference?)  — extend registry
 *   attach(rootEl = document.body)        — start observing
 *   detach()                              — stop & unwrap
 *   registry                              — current dictionary
 *
 * Definition shape: { term, definition, reference?, group? }
 *
 * Coordination: this file is the sole owner of the
 * `.qg-*` DOM and `qg-glossary` MutationObserver. CSS lives in
 * web/css/quant-glossary.css. Index.html mounts both via the
 * index-html-owner; this module self-bootstraps on DOMContent-
 * Loaded if window.PFM.glossary.autoAttach !== false.
 * ============================================================ */

(function () {
  "use strict";

  // --------------------------------------------------------------
  // Built-in dictionary (60+ terms). Each entry: { definition, reference? }
  // Terms can be Greek glyphs, mixed-case acronyms, or full words.
  // The matcher is case-insensitive but term keys preserve display form.
  // --------------------------------------------------------------
  const BUILTIN = {
    // Greek letters
    "α": { definition: "Alpha — risk-adjusted excess return over a benchmark or factor model.", reference: "Jensen 1968" },
    "β": { definition: "Beta — sensitivity of returns to a factor or market portfolio.", reference: "Sharpe 1964" },
    "γ": { definition: "Gamma — second-order sensitivity (curvature) of an option to spot, or risk-aversion in utility models." },
    "δ": { definition: "Delta — first-order sensitivity of an option to the underlying spot price." },
    "ε": { definition: "Epsilon — idiosyncratic noise term; also the clipping threshold for prediction-market probabilities (default 0.01)." },
    "ζ": { definition: "Zeta — sometimes used for skew-risk exposure or a tail-loading parameter." },
    "η": { definition: "Eta — elasticity, learning rate (SGD), or market-impact coefficient." },
    "θ": { definition: "Theta — time decay of an option; also a general parameter vector in estimation." },
    "ι": { definition: "Iota — vector of ones in linear algebra; rarely used but appears in portfolio constraints." },
    "κ": { definition: "Kappa — vol-of-vol sensitivity, or mean-reversion speed in Ornstein-Uhlenbeck.", reference: "Heston 1993" },
    "λ": { definition: "Lambda — decay rate, shrinkage parameter, or eigenvalue.", reference: "Ledoit-Wolf 2004" },
    "μ": { definition: "Mu — population mean / drift of a return series." },
    "ν": { definition: "Nu — degrees of freedom for a Student-t distribution; also vega in some option notations." },
    "ξ": { definition: "Xi — slack variable in constrained optimization (SVMs, portfolio bounds)." },
    "π": { definition: "Pi — circle constant; also stationary distribution of a Markov chain or implied probability." },
    "ρ": { definition: "Rho — correlation coefficient between two return series; also option sensitivity to rates." },
    "σ": { definition: "Sigma — standard deviation; the canonical measure of volatility for a return series." },
    "σ²": { definition: "Sigma-squared — variance of a return series (square of σ)." },
    "τ": { definition: "Tau — quantile level (e.g. τ=0.05 for 5% VaR), or time-to-resolution for a prediction-market contract." },
    "φ": { definition: "Phi — standard normal PDF, or AR(1) persistence coefficient." },
    "χ": { definition: "Chi — chi-squared statistic; sum of squared standard normals." },
    "ψ": { definition: "Psi — digamma function, or influence function in robust statistics." },
    "ω": { definition: "Omega — Omega ratio (gain-loss above a threshold), or GARCH constant term.", reference: "Keating-Shadwick 2002" },

    // Core return / risk measures
    "R²": { definition: "Coefficient of determination — fraction of dependent-variable variance explained by the model." },
    "adj R²": { definition: "Adjusted R² — R² penalised for the number of regressors; prevents in-sample overfit illusion." },
    "Sharpe": { definition: "Sharpe ratio — annualised mean excess return divided by annualised standard deviation.", reference: "Sharpe 1966" },
    "Sortino": { definition: "Sortino ratio — Sharpe variant using downside deviation only.", reference: "Sortino-Price 1994" },
    "DSR": { definition: "Deflated Sharpe Ratio — corrects observed Sharpe for multiple-testing and non-normality.", reference: "" },
    "PSR": { definition: "Probabilistic Sharpe Ratio — probability the true Sharpe exceeds a benchmark given finite sample.", reference: "" },
    "VaR": { definition: "Value at Risk — loss level not exceeded with probability 1−τ over a horizon." },
    "CVaR": { definition: "Conditional VaR / Expected Shortfall — average loss conditional on exceeding VaR.", reference: "Rockafellar-Uryasev 2000" },
    "MaxDD": { definition: "Maximum drawdown — largest peak-to-trough equity decline." },
    "Calmar": { definition: "Calmar ratio — annualised return divided by maximum drawdown." },
    "Information ratio": { definition: "Active return divided by active risk (tracking error) vs a benchmark." },
    "Treynor": { definition: "Treynor ratio — excess return per unit of systematic (β) risk.", reference: "Treynor 1965" },

    // Econometrics
    "OLS": { definition: "Ordinary Least Squares — linear regression by minimising squared residuals." },
    "GLS": { definition: "Generalized Least Squares — OLS with a non-identity error covariance." },
    "WLS": { definition: "Weighted Least Squares — observations weighted by inverse error variance." },
    "HAC": { definition: "Heteroskedasticity and Autocorrelation Consistent — robust standard errors.", reference: "" },
    "VIF": { definition: "Variance Inflation Factor — multicollinearity diagnostic; values above ~10 signal redundancy." },
    "DW": { definition: "Durbin-Watson statistic — test for first-order residual autocorrelation (≈2 = none)." },
    "ADF": { definition: "Augmented Dickey-Fuller test — null hypothesis is a unit root (non-stationary series)." },
    "KPSS": { definition: "Kwiatkowski-Phillips-Schmidt-Shin test — null hypothesis is stationarity (complements ADF)." },
    "Jarque-Bera": { definition: "Jarque-Bera test — joint test of skewness and excess kurtosis vs normality." },
    "Ljung-Box": { definition: "Ljung-Box Q test — joint test for residual autocorrelation up to lag k." },
    "BH-FDR": { definition: "Benjamini-Hochberg False Discovery Rate — multiple-testing correction controlling expected proportion of false positives.", reference: "Benjamini-Hochberg 1995" },

    // Models
    "GARCH": { definition: "Generalized AutoRegressive Conditional Heteroskedasticity — volatility model where σ²_t depends on lagged σ² and squared returns.", reference: "Bollerslev 1986" },
    "EWMA": { definition: "Exponentially Weighted Moving Average — recent observations weighted more heavily by decay factor λ." },
    "ARMA": { definition: "AutoRegressive Moving Average — linear time series with AR and MA components." },
    "VECM": { definition: "Vector Error Correction Model — multivariate cointegration framework." },
    "Kalman": { definition: "Kalman filter — recursive state estimator for linear Gaussian systems.", reference: "Kalman 1960" },
    "GBM": { definition: "Geometric Brownian Motion — log-returns are Gaussian; underlies Black-Scholes pricing." },
    "OU": { definition: "Ornstein-Uhlenbeck process — mean-reverting continuous-time process; used for spreads." },

    // Sizing / portfolio
    "Kelly": { definition: "Kelly Criterion — fraction of capital that maximises expected log-growth.", reference: "Kelly 1956" },
    "Fractional Kelly": { definition: "A fixed fraction of full Kelly (typically 25–50%) trading geometric growth for lower drawdown." },
    "MVO": { definition: "Mean-Variance Optimization — Markowitz portfolio that minimises variance for target return.", reference: "Markowitz 1952" },
    "Black-Litterman": { definition: "Bayesian portfolio framework blending equilibrium returns with investor views.", reference: "Black-Litterman 1992" },
    "Risk parity": { definition: "Allocation where each asset contributes equal risk; volatility-weighted inverse." },

    // Prediction-market / options / micro
    "Logit": { definition: "Log-odds transform: logit(p) = log(p / (1−p)); maps (0,1) → ℝ." },
    "Δlogit": { definition: "Change in log-odds between consecutive observations; the canonical 'return' for prediction-market contracts." },
    "IV": { definition: "Implied volatility — the σ that equates a model price to the observed option price." },
    "OFI": { definition: "Order Flow Imbalance — signed difference in bid vs ask depth changes; predictor of short-term price moves." },
    "VWAP": { definition: "Volume-Weighted Average Price — sum(price·volume) / sum(volume) over a window." },
    "TWAP": { definition: "Time-Weighted Average Price — equal-weighted average of price over a window." },
    "Slippage": { definition: "Execution-price deviation from a reference price (mid, arrival, or VWAP)." },

    // Inference
    "MLE": { definition: "Maximum Likelihood Estimation — parameter estimator that maximises sample likelihood." },
    "MAP": { definition: "Maximum A Posteriori — Bayesian point estimate at the posterior mode." },
    "Bootstrap": { definition: "Resampling with replacement to estimate sampling distributions empirically.", reference: "Efron 1979" },
    "CV": { definition: "Cross-validation — out-of-sample performance estimate via held-out folds." },
    "Walk-forward": { definition: "Time-respecting expanding-window CV; no future leakage." },
    "PIT": { definition: "Probability Integral Transform — uniform under correct distributional forecast; calibration test." },

    // Other commonly used
    "i.i.d.": { definition: "Independent and identically distributed — assumption underlying many classical estimators." },
    "p-value": { definition: "Probability of observing data as extreme as the sample under the null hypothesis." },
    "t-stat": { definition: "t-statistic — coefficient divided by its standard error; |t| > ~2 is the classical significance bar." },
    "z-score": { definition: "Standardised deviation: (x − μ) / σ." }
  };

  // --------------------------------------------------------------
  // State
  // --------------------------------------------------------------
  const state = {
    registry: {},          // canonical: { lowerKey: { term, definition, reference? } }
    tooltipEl: null,       // shared tooltip DOM node
    rootEl: null,          // currently attached root
    observer: null,        // MutationObserver
    attached: false,
    regex: null,           // compiled term regex
    sortedKeys: [],        // display keys sorted longest-first
    hideTimer: null
  };

  // Skip these tags entirely (we never wrap inside them).
  const SKIP_TAGS = new Set([
    "SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA", "INPUT", "SELECT", "OPTION",
    "CODE", "PRE", "KBD", "SVG", "MATH", "CANVAS", "BUTTON"
  ]);

  // --------------------------------------------------------------
  // Registry helpers
  // --------------------------------------------------------------
  function define(term, definition, reference) {
    if (!term || typeof term !== "string") return;
    if (typeof definition === "string") {
      state.registry[term.toLowerCase()] = { term, definition, reference: reference || null };
    } else if (definition && typeof definition === "object") {
      state.registry[term.toLowerCase()] = {
        term,
        definition: definition.definition || "",
        reference: definition.reference || null
      };
    }
    rebuildRegex();
  }

  function loadBuiltins() {
    for (const k of Object.keys(BUILTIN)) {
      state.registry[k.toLowerCase()] = {
        term: k,
        definition: BUILTIN[k].definition,
        reference: BUILTIN[k].reference || null
      };
    }
    rebuildRegex();
  }

  // Escape regex metacharacters in a term key.
  function escapeRe(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  // A "word char" for our purposes: ASCII alnum, underscore, or Greek/Math letter.
  // We define word boundaries manually because \b in JS regex is ASCII-only and
  // we have non-ASCII glyphs (α, β, σ, …) in the registry.
  const WORDCHAR_RE = /[A-Za-z0-9_²³Ͱ-Ͽἀ-῿]/;

  function rebuildRegex() {
    const keys = Object.keys(state.registry).map((k) => state.registry[k].term);
    // Sort longest-first so "adj R²" beats "R²", "Δlogit" beats "logit", etc.
    state.sortedKeys = keys.slice().sort((a, b) => b.length - a.length);
    if (!state.sortedKeys.length) {
      state.regex = null;
      return;
    }
    const pattern = state.sortedKeys.map(escapeRe).join("|");
    state.regex = new RegExp("(" + pattern + ")", "gi");
  }

  // --------------------------------------------------------------
  // Tooltip rendering
  // --------------------------------------------------------------
  function ensureTooltip() {
    if (state.tooltipEl) return state.tooltipEl;
    const el = document.createElement("div");
    el.className = "qg-tooltip";
    el.setAttribute("role", "tooltip");
    el.setAttribute("aria-hidden", "true");
    el.innerHTML =
      '<div class="qg-tooltip__term"></div>' +
      '<div class="qg-tooltip__def"></div>' +
      '<div class="qg-tooltip__ref"></div>';
    document.body.appendChild(el);
    el.addEventListener("mouseenter", cancelHide);
    el.addEventListener("mouseleave", scheduleHide);
    state.tooltipEl = el;
    return el;
  }

  function showTooltip(target, entry) {
    cancelHide();
    const el = ensureTooltip();
    el.querySelector(".qg-tooltip__term").textContent = entry.term;
    el.querySelector(".qg-tooltip__def").textContent = entry.definition;
    const refEl = el.querySelector(".qg-tooltip__ref");
    if (entry.reference) {
      refEl.textContent = entry.reference;
      refEl.style.display = "";
    } else {
      refEl.textContent = "";
      refEl.style.display = "none";
    }

    // Position above target by default; flip below if not enough room.
    const rect = target.getBoundingClientRect();
    el.style.visibility = "hidden";
    el.style.display = "block";
    el.setAttribute("aria-hidden", "false");
    const tipRect = el.getBoundingClientRect();

    const margin = 8;
    let top = rect.top + window.scrollY - tipRect.height - margin;
    let placement = "top";
    if (top < window.scrollY + 4) {
      top = rect.bottom + window.scrollY + margin;
      placement = "bottom";
    }
    let left = rect.left + window.scrollX + rect.width / 2 - tipRect.width / 2;
    const minLeft = window.scrollX + 8;
    const maxLeft = window.scrollX + document.documentElement.clientWidth - tipRect.width - 8;
    if (left < minLeft) left = minLeft;
    if (left > maxLeft) left = maxLeft;

    el.style.top = top + "px";
    el.style.left = left + "px";
    el.dataset.placement = placement;
    el.style.visibility = "visible";
    el.classList.add("qg-tooltip--visible");
  }

  function hideTooltip() {
    if (!state.tooltipEl) return;
    state.tooltipEl.classList.remove("qg-tooltip--visible");
    state.tooltipEl.setAttribute("aria-hidden", "true");
    // Defer display:none until transition completes so CSS opacity animation runs.
    setTimeout(() => {
      if (state.tooltipEl && !state.tooltipEl.classList.contains("qg-tooltip--visible")) {
        state.tooltipEl.style.display = "none";
      }
    }, 140);
  }

  function scheduleHide() {
    cancelHide();
    state.hideTimer = setTimeout(hideTooltip, 120);
  }

  function cancelHide() {
    if (state.hideTimer) {
      clearTimeout(state.hideTimer);
      state.hideTimer = null;
    }
  }

  // --------------------------------------------------------------
  // Event handlers (delegated on root)
  // --------------------------------------------------------------
  function onPointerEnter(e) {
    const t = e.target;
    if (!(t && t.classList && t.classList.contains("qg-term"))) return;
    const key = (t.dataset.term || "").toLowerCase();
    const entry = state.registry[key];
    if (entry) showTooltip(t, entry);
  }

  function onPointerLeave(e) {
    const t = e.target;
    if (!(t && t.classList && t.classList.contains("qg-term"))) return;
    scheduleHide();
  }

  function onFocusIn(e) {
    const t = e.target;
    if (!(t && t.classList && t.classList.contains("qg-term"))) return;
    const key = (t.dataset.term || "").toLowerCase();
    const entry = state.registry[key];
    if (entry) showTooltip(t, entry);
  }

  function onFocusOut(e) {
    const t = e.target;
    if (!(t && t.classList && t.classList.contains("qg-term"))) return;
    scheduleHide();
  }

  // --------------------------------------------------------------
  // Text-node walker / wrapper
  // --------------------------------------------------------------
  function shouldSkipNode(node) {
    let p = node.parentNode;
    while (p && p.nodeType === 1) {
      if (SKIP_TAGS.has(p.tagName)) return true;
      if (p.classList && (p.classList.contains("qg-term") || p.classList.contains("qg-tooltip") || p.hasAttribute("data-qg-skip"))) {
        return true;
      }
      // Don't reach above the attached root.
      if (p === state.rootEl) break;
      p = p.parentNode;
    }
    return false;
  }

  function wrapNode(textNode) {
    if (!state.regex) return;
    const text = textNode.nodeValue;
    if (!text || text.length < 1) return;
    state.regex.lastIndex = 0;
    if (!state.regex.test(text)) return;
    state.regex.lastIndex = 0;

    const frag = document.createDocumentFragment();
    let lastIndex = 0;
    let m;
    let anyWrapped = false;

    while ((m = state.regex.exec(text)) !== null) {
      const start = m.index;
      const end = start + m[0].length;
      const prev = start > 0 ? text.charAt(start - 1) : "";
      const next = end < text.length ? text.charAt(end) : "";
      // Enforce whole-word boundary using our own word-char definition.
      if (WORDCHAR_RE.test(prev) || WORDCHAR_RE.test(next)) {
        continue;
      }
      if (start > lastIndex) {
        frag.appendChild(document.createTextNode(text.slice(lastIndex, start)));
      }
      const entry = state.registry[m[0].toLowerCase()];
      if (!entry) {
        frag.appendChild(document.createTextNode(text.slice(start, end)));
      } else {
        const span = document.createElement("span");
        span.className = "qg-term";
        span.dataset.term = entry.term;
        span.setAttribute("tabindex", "0");
        span.setAttribute("aria-label", entry.term + " — " + entry.definition);
        span.textContent = text.slice(start, end);
        frag.appendChild(span);
        anyWrapped = true;
      }
      lastIndex = end;
    }
    if (!anyWrapped) return;
    if (lastIndex < text.length) {
      frag.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
    textNode.parentNode.replaceChild(frag, textNode);
  }

  function walkSubtree(root) {
    if (!root || !state.regex) return;
    // Some browsers throw if root is a text node passed to createTreeWalker; guard.
    const startNode = root.nodeType === 1 ? root : root.parentNode;
    if (!startNode) return;
    const walker = document.createTreeWalker(
      startNode,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
          if (shouldSkipNode(node)) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        }
      }
    );
    const queue = [];
    let n;
    while ((n = walker.nextNode())) queue.push(n);
    for (const node of queue) wrapNode(node);
  }

  function unwrapAll(root) {
    if (!root) return;
    const spans = root.querySelectorAll ? root.querySelectorAll("span.qg-term") : [];
    spans.forEach((s) => {
      const parent = s.parentNode;
      if (!parent) return;
      parent.replaceChild(document.createTextNode(s.textContent), s);
      parent.normalize();
    });
  }

  // --------------------------------------------------------------
  // MutationObserver — re-scan newly added subtrees, throttled.
  // --------------------------------------------------------------
  let scanQueue = new Set();
  let scanScheduled = false;
  function scheduleScan(node) {
    scanQueue.add(node);
    if (scanScheduled) return;
    scanScheduled = true;
    const run = () => {
      scanScheduled = false;
      const nodes = Array.from(scanQueue);
      scanQueue.clear();
      for (const n of nodes) {
        if (n && n.isConnected) walkSubtree(n);
      }
    };
    if (window.requestIdleCallback) {
      window.requestIdleCallback(run, { timeout: 250 });
    } else {
      setTimeout(run, 80);
    }
  }

  function onMutations(mutations) {
    for (const m of mutations) {
      if (m.type !== "childList") continue;
      m.addedNodes.forEach((n) => {
        if (n.nodeType === 1) scheduleScan(n);
        else if (n.nodeType === 3 && n.parentNode) scheduleScan(n.parentNode);
      });
    }
  }

  // --------------------------------------------------------------
  // attach / detach
  // --------------------------------------------------------------
  function attach(rootEl) {
    if (state.attached) detach();
    const root = rootEl || document.body;
    if (!root) return;
    state.rootEl = root;
    state.attached = true;

    if (!Object.keys(state.registry).length) loadBuiltins();
    else rebuildRegex();

    walkSubtree(root);

    root.addEventListener("mouseover", onPointerEnter, true);
    root.addEventListener("mouseout", onPointerLeave, true);
    root.addEventListener("focusin", onFocusIn, true);
    root.addEventListener("focusout", onFocusOut, true);

    state.observer = new MutationObserver(onMutations);
    state.observer.observe(root, { childList: true, subtree: true });
  }

  function detach() {
    if (!state.attached) return;
    const root = state.rootEl;
    if (state.observer) {
      state.observer.disconnect();
      state.observer = null;
    }
    if (root) {
      root.removeEventListener("mouseover", onPointerEnter, true);
      root.removeEventListener("mouseout", onPointerLeave, true);
      root.removeEventListener("focusin", onFocusIn, true);
      root.removeEventListener("focusout", onFocusOut, true);
      unwrapAll(root);
    }
    hideTooltip();
    state.rootEl = null;
    state.attached = false;
  }

  // --------------------------------------------------------------
  // Boot — expose on window.PFM.glossary
  // --------------------------------------------------------------
  loadBuiltins();
  window.PFM = window.PFM || {};
  const api = {
    define,
    attach,
    detach,
    get registry() { return state.registry; },
    BUILTIN
  };
  window.PFM.glossary = api;

  // Self-mount on DOMContentLoaded unless caller opts out.
  function maybeAutoAttach() {
    if (window.PFM.glossary.autoAttach === false) return;
    attach(document.body);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", maybeAutoAttach, { once: true });
  } else {
    maybeAutoAttach();
  }
})();
