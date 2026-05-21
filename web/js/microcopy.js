/* ============================================================================
 * microcopy.js — centralized microcopy dictionary + linter
 * Owner: W11-06 (microcopy)
 * Mount: <script src="/js/microcopy.js" defer></script> in web/index.html
 *
 * Exposes:
 *   window.PFM.copy.get(key)          -> string | undefined
 *   window.PFM.copy.replace(el, dict) -> walks textContent of el and substitutes
 *   window.PFM.copy.dict              -> the canonical dictionary
 *   window.PFM.copy.verbs             -> standardized verb table
 *   window.PFM.copy.CopyLinter        -> class with static scan(root) -> warnings[]
 *
 * Design intent: tighter, more confident phrasing throughout. Strings here
 * should be preferred over inline literals everywhere in web/js/*.js. The
 * linter helps QA find banned phrases ("Click here", generic "Submit",
 * "Loading...", "An error occurred") that slipped through.
 * ============================================================================ */

(function () {
  "use strict";

  /* ---------------- Standardized verb table ----------------
   * Use the LHS verb; avoid the RHS variants. This is what the linter checks. */
  const VERBS = Object.freeze({
    fit:      ["compute", "run regression", "calculate model", "calc"],
    fetch:    ["get", "download", "grab", "pull data"],
    pin:      ["save", "store", "bookmark", "favorite"],
    dismiss:  ["close", "x out", "hide it"],
    backtest: ["test history", "historical test", "run history"],
    // Additional non-banned verbs we standardize on:
    run:      [],   // reserved for "run fit" / "run backtest" (compound)
    open:     [],
    refresh:  ["reload data", "re-pull"],
    apply:    ["use these", "set these"],
    cancel:   ["abort", "stop it"],
    retry:    ["try again", "redo"],
    copy:     ["clipboard it", "duplicate"],
    export:   ["save as", "dump"],
  });

  /* ---------------- Canonical microcopy dictionary ----------------
   * 40+ replacements. Keys are the OLD/generic string (lowercase-insensitive
   * lookup via .get) and values are the tighter replacement.
   * Some entries are also keyed by short symbolic IDs (e.g. "error.network")
   * so call-sites can request them by stable key rather than by old text. */
  const DICT = Object.freeze({
    // --- Errors (network / upstream) ---
    "An error occurred":           "We couldn’t reach Polymarket — retry in a moment",
    "Something went wrong":        "Something broke upstream — we’re retrying",
    "Error":                       "Couldn’t complete that",
    "Failed":                      "That didn’t go through",
    "Network error":               "We couldn’t reach the API — check connection and retry",
    "Request failed":              "Request failed — Polymarket may be slow; retry",
    "Timeout":                     "Took too long — retry or narrow the window",
    "Server error":                "Server hiccup — we’re looking at it",
    "Not found":                   "We couldn’t find that contract",
    "404":                         "We couldn’t find that contract",
    "500":                         "Server hiccup — we’re looking at it",

    // Symbolic error keys (preferred call-site)
    "error.network":               "We couldn’t reach Polymarket — retry in a moment",
    "error.timeout":               "Took too long — retry or narrow the window",
    "error.notfound":              "We couldn’t find that contract",
    "error.server":                "Server hiccup — we’re looking at it",
    "error.fit":                   "We couldn’t fit the model — check the slug and date range",
    "error.backtest":              "Backtest failed — sample may be too short",
    "error.factor":                "Factor unavailable — try another slug",

    // --- Loading states ---
    "Loading...":                  "Fitting model…",
    "Loading":                     "Fitting model…",
    "Please wait...":              "Hang tight…",
    "Working...":                  "Working on it…",
    "Processing...":               "Crunching…",

    "loading.fit":                 "Fitting model…",
    "loading.fetch":               "Fetching prices…",
    "loading.backtest":            "Running backtest…",
    "loading.factors":             "Loading factor catalog…",
    "loading.news":                "Pulling latest news…",
    "loading.arb":                 "Scanning cross-venue spreads…",
    "loading.signals":             "Refreshing signals…",

    // --- Empty states ---
    "No data":                     "No prediction-market activity for this contract today",
    "No data available":           "No prediction-market activity for this contract today",
    "No results":                  "Nothing matched — try a different slug or window",
    "No results found":            "Nothing matched — try a different slug or window",
    "Empty":                       "Nothing here yet",
    "Nothing to show":             "Nothing here yet",

    "empty.factors":               "No factors loaded — pick one from the catalog",
    "empty.news":                  "No headlines tagged to this contract today",
    "empty.arb":                   "No live spreads above threshold right now",
    "empty.pinned":                "Nothing pinned — fit a model and pin the result",
    "empty.history":               "No prior fits in this session",

    // --- Buttons / actions (banned generics → action-specific) ---
    "Submit":                      "Run fit",
    "Click here":                  "Open factor catalog",
    "Click":                       "Open factor catalog",
    "OK":                          "Got it",
    "Cancel":                      "Cancel",
    "Save":                        "Pin result",
    "Close":                       "Dismiss",
    "Run":                         "Fit",
    "Get":                         "Fetch",
    "Download":                    "Export",
    "Compute":                     "Fit",
    "Calculate":                   "Fit",

    "action.fit":                  "Run fit",
    "action.backtest":             "Run backtest",
    "action.refit":                "Re-fit with new window",
    "action.pin":                  "Pin result",
    "action.unpin":                "Unpin",
    "action.export":               "Export CSV",
    "action.copy":                 "Copy to clipboard",
    "action.dismiss":              "Dismiss",
    "action.retry":                "Retry",
    "action.refresh":              "Refresh",
    "action.openCatalog":          "Open factor catalog",
    "action.openTerminal":         "Open Terminal",
    "action.openAlphaHub":         "Open α Hub",

    // --- Confirmations ---
    "Are you sure?":               "Pin this fit?",
    "confirm.delete":              "Remove this pin?",
    "confirm.discard":             "Discard unsaved changes?",

    // --- Toasts / status ---
    "Saved":                       "Pinned",
    "Done":                        "Done",
    "Success":                     "Fit complete",
    "Updated":                     "Updated",
    "Copied":                      "Copied to clipboard",
    "toast.fit.success":           "Fit complete — R² and HAC SEs ready",
    "toast.fit.failed":            "Fit failed — see error panel",
    "toast.pinned":                "Pinned to session",
    "toast.unpinned":              "Removed from pins",

    // --- Helper / placeholder ---
    "Enter value":                 "Enter a Polymarket slug",
    "Search...":                   "Search factors, contracts, tickers",
    "placeholder.slug":            "e.g. will-fed-cut-rates-by-march-2026",
    "placeholder.ticker":          "e.g. SPY, NVDA, BTC-USD",
    "placeholder.window":          "Days back (default 90)",

    // --- Tooltips (concise, declarative) ---
    "tip.hac":                     "HAC standard errors — heteroskedasticity- and autocorrelation-consistent correction, lag=5",
    "tip.vif":                     "Variance Inflation Factor — flags collinear factors (>10 is high)",
    "tip.clip":                    "Clipping epsilon — guards logit transform near 0/1; default 0.01",
    "tip.r2":                      "Adjusted R² — penalizes additional factors",
    "tip.pin":                     "Pin holds this fit in your session for side-by-side compare",
  });

  // Build a case-insensitive index for fast text-walk substitution.
  const DICT_LC = Object.freeze(
    Object.keys(DICT).reduce((acc, k) => {
      if (k.indexOf(".") === -1) acc[k.toLowerCase()] = DICT[k];
      return acc;
    }, {})
  );

  /* ---------------- Banned phrases (linter) ---------------- */
  const BANNED = Object.freeze([
    { pat: /\bclick here\b/i,            why: "Use action-specific text (e.g. 'Open factor catalog')." },
    { pat: /^\s*submit\s*$/i,            why: "Use 'Run fit' or another action verb." },
    { pat: /^\s*loading\.{0,3}\s*$/i,    why: "Use context-specific 'Fitting model…' / 'Fetching prices…' etc." },
    { pat: /\ban error occurred\b/i,     why: "Use a specific message, e.g. 'We couldn’t reach Polymarket — retry in a moment'." },
    { pat: /\bsomething went wrong\b/i,  why: "Tell the user what failed and what to do." },
    { pat: /^\s*ok\s*$/i,                why: "Use 'Got it' or an action-specific label." },
    { pat: /^\s*no data\s*$/i,           why: "Use a specific empty-state, e.g. 'No prediction-market activity today'." },
    { pat: /\bplease wait\b/i,           why: "Use 'Hang tight…' or context-specific loading." },
    { pat: /^\s*close\s*$/i,             why: "Use 'Dismiss' (standard verb)." },
    { pat: /^\s*get\s*$/i,               why: "Use 'Fetch' (standard verb)." },
    { pat: /^\s*run\s*$/i,               why: "Use 'Fit' or compound 'Run fit'." },
    { pat: /^\s*save\s*$/i,              why: "Use 'Pin result' (standard verb)." },
    { pat: /\bcompute\b/i,               why: "Use 'fit' for model operations." },
    { pat: /\btest history\b/i,          why: "Use 'backtest' (one word)." },
  ]);

  /* ---------------- Public API ---------------- */

  function get(key) {
    if (key == null) return undefined;
    if (Object.prototype.hasOwnProperty.call(DICT, key)) return DICT[key];
    const lc = String(key).toLowerCase();
    return DICT_LC[lc];
  }

  /**
   * Walk the textContent of `el` (and its descendants) and substitute any
   * text-node that matches a key in `dict` (or the default DICT if omitted).
   * Only replaces nodes whose trimmed text is an exact match — does NOT do
   * substring replacement (too risky for partial matches like "Run query").
   *
   * @param {Element} el        root element
   * @param {Object}  [dict]    optional override dictionary
   * @returns {number}          count of substitutions performed
   */
  function replace(el, dict) {
    if (!el || !el.nodeType) return 0;
    const lookup = dict || DICT;
    const lcLookup = {};
    for (const k of Object.keys(lookup)) {
      if (k.indexOf(".") === -1) lcLookup[k.toLowerCase()] = lookup[k];
    }
    let count = 0;
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        const t = (n.nodeValue || "").trim();
        if (!t) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const n of nodes) {
      const raw = n.nodeValue || "";
      const trimmed = raw.trim();
      if (!trimmed) continue;
      const repl = lookup[trimmed] || lcLookup[trimmed.toLowerCase()];
      if (repl != null && repl !== trimmed) {
        // Preserve surrounding whitespace.
        const leading = raw.match(/^\s*/)[0];
        const trailing = raw.match(/\s*$/)[0];
        n.nodeValue = leading + repl + trailing;
        count += 1;
      }
    }
    return count;
  }

  /* ---------------- CopyLinter ---------------- */

  class CopyLinter {
    /**
     * Scan a DOM root for banned phrases. Returns an array of warnings.
     * Use from devtools: PFM.copy.CopyLinter.scan(document.body).forEach(w => console.warn(w))
     *
     * @param {Element|Document} [root=document]
     * @returns {Array<{phrase:string, why:string, snippet:string, path:string}>}
     */
    static scan(root) {
      root = root || document;
      const out = [];
      const walker = (root.createTreeWalker
        ? root.createTreeWalker(root, NodeFilter.SHOW_TEXT, null)
        : document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null));
      while (walker.nextNode()) {
        const node = walker.currentNode;
        const text = (node.nodeValue || "").trim();
        if (!text) continue;
        // Skip script/style/code blocks.
        const parent = node.parentNode;
        if (!parent) continue;
        const tag = (parent.tagName || "").toUpperCase();
        if (tag === "SCRIPT" || tag === "STYLE" || tag === "CODE" || tag === "PRE") continue;
        for (const b of BANNED) {
          if (b.pat.test(text)) {
            out.push({
              phrase: text.slice(0, 80),
              why: b.why,
              snippet: text.length > 120 ? text.slice(0, 120) + "…" : text,
              path: CopyLinter._pathOf(parent),
            });
            break;
          }
        }
      }
      return out;
    }

    static _pathOf(el) {
      const parts = [];
      let cur = el;
      while (cur && cur.nodeType === 1 && parts.length < 6) {
        let seg = cur.tagName.toLowerCase();
        if (cur.id) {
          seg += "#" + cur.id;
          parts.unshift(seg);
          break;
        }
        if (cur.className && typeof cur.className === "string") {
          const c = cur.className.trim().split(/\s+/)[0];
          if (c) seg += "." + c;
        }
        parts.unshift(seg);
        cur = cur.parentNode;
      }
      return parts.join(" > ");
    }
  }

  /* ---------------- Mount on window.PFM.copy ---------------- */

  const ns = (window.PFM = window.PFM || {});
  ns.copy = Object.freeze({
    get,
    replace,
    dict: DICT,
    verbs: VERBS,
    CopyLinter,
  });
})();
