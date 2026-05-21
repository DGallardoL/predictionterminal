/* ============================================================================
 * alpha-hub-filters.js — chip filter bar for α Hub leaderboard
 * Owner: W12-40
 * Mount: <script src="/js/alpha-hub-filters.js" defer></script>
 *
 * Renders a horizontally-scrollable chip-group bar above the α Hub card grid
 * with three filter dimensions:
 *
 *   - Tier (single-select): All / A_GOLD / A_STRUCTURAL / B_VALIDATED /
 *                            B_FDR_ONLY / C_TENTATIVE / D_RAW
 *   - Theme (single-select): All / Politics / Macro / Crypto / Sports /
 *                            Sentiment
 *   - Min Sharpe (slider): 0.0 - 3.0 step 0.1
 *
 * Public API (mounted at window.PFM.alphaHubFilters):
 *
 *   mount(container | string)        Render into a DOM node or CSS selector.
 *                                    Idempotent: calling twice on the same
 *                                    container replaces the previous bar.
 *
 *   setFilter(name, value)           Programmatic update.
 *                                    name ∈ { "tier", "theme", "min_sharpe" }
 *                                    value: string for tier/theme; number or
 *                                    numeric-string for min_sharpe.
 *                                    Fires onChange callbacks + updates URL.
 *
 *   getFilters() -> { tier, theme, min_sharpe }
 *                                    Returns the current effective filters
 *                                    in the same shape used as query params.
 *                                    Defaults: tier="all", theme="all",
 *                                    min_sharpe=0.
 *
 *   onChange(cb) -> () => off        Register a callback fired whenever any
 *                                    filter changes. Returns an unsubscribe
 *                                    function.
 *
 * URL persistence
 * ---------------
 * Filter state is serialized into the URL hash as `#ahf=tier=X&theme=Y&s=Z`
 * (only non-default keys are written). On mount, the hash is parsed and the
 * initial state is restored — making the current filter view shareable via
 * link copy/paste.
 *
 * No external CSS dependencies beyond the partner stylesheet
 * web/css/alpha-hub-filters.css. No fetch is performed here — consumers wire
 * the onChange callback to the leaderboard re-fetch (the backend already
 * accepts ?tier=&theme=&min_sharpe= per pfm/alpha_hub_router.py).
 * ============================================================================ */

(function () {
  "use strict";

  if (window.PFM && window.PFM.alphaHubFilters && window.PFM.alphaHubFilters.__w1240) {
    // Already mounted in this page. Idempotency guard.
    return;
  }

  /* ---------- Constants -------------------------------------------------- */

  var TIERS = [
    { value: "all",          label: "All" },
    { value: "A_GOLD",       label: "A_GOLD" },
    { value: "A_STRUCTURAL", label: "A_STRUCTURAL" },
    { value: "B_VALIDATED",  label: "B_VALIDATED" },
    { value: "B_FDR_ONLY",   label: "B_FDR_ONLY" },
    { value: "C_TENTATIVE",  label: "C_TENTATIVE" },
    { value: "D_RAW",        label: "D_RAW" }
  ];

  // Theme values are lowercased to match what the backend stores in the
  // strategy objects under `theme` (see alpha_strategies.json). The label
  // is title-cased for display only.
  var THEMES = [
    { value: "all",       label: "All" },
    { value: "politics",  label: "Politics" },
    { value: "macro",     label: "Macro" },
    { value: "crypto",    label: "Crypto" },
    { value: "sports",    label: "Sports" },
    { value: "sentiment", label: "Sentiment" }
  ];

  var SHARPE_MIN = 0.0;
  var SHARPE_MAX = 3.0;
  var SHARPE_STEP = 0.1;
  var SHARPE_DEFAULT = 0.0;

  var TIER_DEFAULT = "all";
  var THEME_DEFAULT = "all";

  var HASH_KEY = "ahf";

  /* ---------- Module state ---------------------------------------------- */

  var state = {
    tier: TIER_DEFAULT,
    theme: THEME_DEFAULT,
    min_sharpe: SHARPE_DEFAULT
  };

  var listeners = [];
  var rootEl = null;          // current mounted container
  var elements = {            // cached refs to live DOM nodes
    tierChips: {},            // value -> button
    themeChips: {},
    slider: null,
    sliderValue: null,
    reset: null
  };
  var hashWriteScheduled = false;
  var suppressHashHandler = false;

  /* ---------- Hash <-> state -------------------------------------------- */

  function parseHash() {
    var hash = String(window.location.hash || "");
    if (!hash) return null;
    // Hash may be "#foo&ahf=tier=...&bar=...". Find our segment.
    var s = hash.replace(/^#/, "");
    // Split by '&' but allow our segment value to contain '=' chars.
    // We look for "ahf=" prefix specifically.
    var segs = s.split("&");
    var i;
    for (i = 0; i < segs.length; i++) {
      if (segs[i].indexOf(HASH_KEY + "=") === 0) {
        var raw = segs[i].slice(HASH_KEY.length + 1);
        try {
          raw = decodeURIComponent(raw);
        } catch (e) {
          /* keep raw */
        }
        return parseFilterString(raw);
      }
    }
    return null;
  }

  function parseFilterString(raw) {
    // raw format: "tier=A_GOLD&theme=macro&s=0.8"
    if (!raw) return null;
    var out = {};
    var parts = raw.split("&");
    var j;
    for (j = 0; j < parts.length; j++) {
      var kv = parts[j].split("=");
      if (kv.length !== 2) continue;
      var k = kv[0].trim();
      var v = kv[1].trim();
      if (k === "tier") out.tier = v;
      else if (k === "theme") out.theme = v;
      else if (k === "s" || k === "min_sharpe") {
        var f = parseFloat(v);
        if (!isNaN(f)) out.min_sharpe = f;
      }
    }
    return out;
  }

  function serializeFilters() {
    var parts = [];
    if (state.tier && state.tier !== TIER_DEFAULT) {
      parts.push("tier=" + state.tier);
    }
    if (state.theme && state.theme !== THEME_DEFAULT) {
      parts.push("theme=" + state.theme);
    }
    if (state.min_sharpe && state.min_sharpe > SHARPE_DEFAULT) {
      parts.push("s=" + state.min_sharpe.toFixed(1));
    }
    return parts.join("&");
  }

  function writeHash() {
    if (hashWriteScheduled) return;
    hashWriteScheduled = true;
    // rAF to coalesce rapid slider input.
    var raf = window.requestAnimationFrame || function (cb) { return setTimeout(cb, 16); };
    raf(function () {
      hashWriteScheduled = false;
      var ahfStr = serializeFilters();
      var hash = String(window.location.hash || "").replace(/^#/, "");
      var segs = hash ? hash.split("&") : [];
      // Drop any existing ahf= segment.
      var kept = [];
      var i;
      for (i = 0; i < segs.length; i++) {
        if (segs[i].indexOf(HASH_KEY + "=") !== 0 && segs[i].length > 0) {
          kept.push(segs[i]);
        }
      }
      if (ahfStr) kept.push(HASH_KEY + "=" + ahfStr);
      var nextHash = kept.length ? "#" + kept.join("&") : "";
      // Avoid pushing duplicate history entries; use replaceState when possible.
      suppressHashHandler = true;
      try {
        if (window.history && typeof window.history.replaceState === "function") {
          var url = window.location.pathname + window.location.search + nextHash;
          window.history.replaceState(window.history.state, "", url);
        } else {
          // Fallback: assign — will fire a hashchange we'll suppress.
          window.location.hash = nextHash;
        }
      } catch (e) {
        /* ignore */
      }
      // Re-enable the listener after the current task.
      setTimeout(function () { suppressHashHandler = false; }, 0);
    });
  }

  function onHashChange() {
    if (suppressHashHandler) return;
    var parsed = parseHash();
    if (!parsed) {
      // Hash cleared by user -> reset to defaults.
      applyState({
        tier: TIER_DEFAULT,
        theme: THEME_DEFAULT,
        min_sharpe: SHARPE_DEFAULT
      }, { fromHash: true });
      return;
    }
    applyState(parsed, { fromHash: true });
  }

  /* ---------- DOM construction ------------------------------------------ */

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      var k;
      for (k in attrs) {
        if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
        var v = attrs[k];
        if (v == null || v === false) continue;
        if (k === "class" || k === "className") {
          node.className = v;
        } else if (k === "text") {
          node.textContent = String(v);
        } else if (k === "style" && typeof v === "object") {
          var sk;
          for (sk in v) {
            if (Object.prototype.hasOwnProperty.call(v, sk)) {
              node.style[sk] = v[sk];
            }
          }
        } else if (k.indexOf("on") === 0 && typeof v === "function") {
          node.addEventListener(k.slice(2).toLowerCase(), v);
        } else if (v === true) {
          node.setAttribute(k, "");
        } else {
          node.setAttribute(k, String(v));
        }
      }
    }
    if (children) {
      var i;
      for (i = 0; i < children.length; i++) {
        var c = children[i];
        if (c == null) continue;
        if (typeof c === "string") {
          node.appendChild(document.createTextNode(c));
        } else {
          node.appendChild(c);
        }
      }
    }
    return node;
  }

  function buildChipGroup(label, options, currentValue, onSelect, refTable) {
    var chips = options.map(function (opt) {
      var chip = el("button", {
        type: "button",
        class: "pfm-ahf-chip",
        "data-value": opt.value,
        "aria-pressed": currentValue === opt.value ? "true" : "false",
        text: opt.label,
        onClick: function () { onSelect(opt.value); }
      });
      refTable[opt.value] = chip;
      return chip;
    });
    return el("div", { class: "pfm-ahf-group", role: "group", "aria-label": label }, [
      el("span", { class: "pfm-ahf-label", text: label }),
      el("div", { class: "pfm-ahf-chips" }, chips)
    ]);
  }

  function buildSliderGroup() {
    var slider = el("input", {
      type: "range",
      class: "pfm-ahf-slider",
      min: String(SHARPE_MIN),
      max: String(SHARPE_MAX),
      step: String(SHARPE_STEP),
      value: String(state.min_sharpe),
      "aria-label": "Minimum Sharpe ratio",
      "aria-valuemin": String(SHARPE_MIN),
      "aria-valuemax": String(SHARPE_MAX)
    });

    var valueEl = el("span", {
      class: "pfm-ahf-slider-value",
      "aria-live": "polite"
    });

    slider.addEventListener("input", function (ev) {
      var f = parseFloat(ev.target.value);
      if (isNaN(f)) f = SHARPE_DEFAULT;
      setFilter("min_sharpe", f);
    });

    elements.slider = slider;
    elements.sliderValue = valueEl;

    return el(
      "div",
      { class: "pfm-ahf-group", role: "group", "aria-label": "Min Sharpe" },
      [
        el("span", { class: "pfm-ahf-label", text: "Min Sharpe" }),
        el("div", { class: "pfm-ahf-slider-wrap" }, [slider, valueEl])
      ]
    );
  }

  function renderSliderFill() {
    if (!elements.slider) return;
    var pct = ((state.min_sharpe - SHARPE_MIN) / (SHARPE_MAX - SHARPE_MIN)) * 100;
    if (pct < 0) pct = 0;
    if (pct > 100) pct = 100;
    elements.slider.style.setProperty("--pfm-ahf-fill", pct.toFixed(2) + "%");
    elements.slider.setAttribute("aria-valuenow", state.min_sharpe.toFixed(1));
    if (elements.sliderValue) {
      var isZero = state.min_sharpe <= SHARPE_DEFAULT + 1e-9;
      elements.sliderValue.textContent =
        (isZero ? "≥ " : "≥ ") + state.min_sharpe.toFixed(1);
      elements.sliderValue.setAttribute("data-zero", isZero ? "true" : "false");
    }
  }

  function renderChipsActive() {
    var v;
    for (v in elements.tierChips) {
      if (Object.prototype.hasOwnProperty.call(elements.tierChips, v)) {
        elements.tierChips[v].setAttribute(
          "aria-pressed",
          v === state.tier ? "true" : "false"
        );
      }
    }
    for (v in elements.themeChips) {
      if (Object.prototype.hasOwnProperty.call(elements.themeChips, v)) {
        elements.themeChips[v].setAttribute(
          "aria-pressed",
          v === state.theme ? "true" : "false"
        );
      }
    }
  }

  function renderResetVisibility() {
    if (!elements.reset) return;
    var dirty =
      state.tier !== TIER_DEFAULT ||
      state.theme !== THEME_DEFAULT ||
      state.min_sharpe > SHARPE_DEFAULT;
    if (dirty) {
      elements.reset.removeAttribute("hidden");
    } else {
      elements.reset.setAttribute("hidden", "");
    }
  }

  function renderAll() {
    renderChipsActive();
    renderSliderFill();
    renderResetVisibility();
  }

  /* ---------- State application ----------------------------------------- */

  function clampSharpe(v) {
    var f = typeof v === "number" ? v : parseFloat(v);
    if (isNaN(f)) f = SHARPE_DEFAULT;
    // Snap to step grid to avoid floating drift.
    f = Math.round(f / SHARPE_STEP) * SHARPE_STEP;
    if (f < SHARPE_MIN) f = SHARPE_MIN;
    if (f > SHARPE_MAX) f = SHARPE_MAX;
    // Fix float artefacts like 0.30000000000000004.
    f = parseFloat(f.toFixed(2));
    return f;
  }

  function normalizeTier(v) {
    if (v == null) return TIER_DEFAULT;
    var s = String(v);
    var i;
    for (i = 0; i < TIERS.length; i++) {
      if (TIERS[i].value === s) return s;
    }
    return TIER_DEFAULT;
  }

  function normalizeTheme(v) {
    if (v == null) return THEME_DEFAULT;
    var s = String(v).toLowerCase();
    var i;
    for (i = 0; i < THEMES.length; i++) {
      if (THEMES[i].value === s) return s;
    }
    return THEME_DEFAULT;
  }

  function applyState(partial, opts) {
    opts = opts || {};
    var changed = false;
    if (Object.prototype.hasOwnProperty.call(partial, "tier")) {
      var nt = normalizeTier(partial.tier);
      if (nt !== state.tier) { state.tier = nt; changed = true; }
    }
    if (Object.prototype.hasOwnProperty.call(partial, "theme")) {
      var nth = normalizeTheme(partial.theme);
      if (nth !== state.theme) { state.theme = nth; changed = true; }
    }
    if (Object.prototype.hasOwnProperty.call(partial, "min_sharpe")) {
      var ns = clampSharpe(partial.min_sharpe);
      if (ns !== state.min_sharpe) { state.min_sharpe = ns; changed = true; }
    }
    if (!changed) return false;

    // Sync slider DOM value if state mutation came from elsewhere.
    if (elements.slider && partial.min_sharpe != null) {
      var cur = parseFloat(elements.slider.value);
      if (cur !== state.min_sharpe) {
        elements.slider.value = String(state.min_sharpe);
      }
    }
    renderAll();
    if (!opts.fromHash) writeHash();
    fire();
    return true;
  }

  /* ---------- Public-API helpers ---------------------------------------- */

  function setFilter(name, value) {
    var partial = {};
    if (name === "tier") partial.tier = value;
    else if (name === "theme") partial.theme = value;
    else if (name === "min_sharpe" || name === "minSharpe" || name === "s") {
      partial.min_sharpe = value;
    } else {
      return false;
    }
    return applyState(partial, {});
  }

  function getFilters() {
    return {
      tier: state.tier,
      theme: state.theme,
      min_sharpe: state.min_sharpe
    };
  }

  function onChange(cb) {
    if (typeof cb !== "function") return function () {};
    listeners.push(cb);
    return function off() {
      var i = listeners.indexOf(cb);
      if (i >= 0) listeners.splice(i, 1);
    };
  }

  function fire() {
    var snap = getFilters();
    var i;
    for (i = 0; i < listeners.length; i++) {
      try {
        listeners[i](snap);
      } catch (e) {
        // Never let one bad subscriber break the bar.
        if (window.console && console.error) {
          console.error("[alpha-hub-filters] onChange listener threw:", e);
        }
      }
    }
    // Also dispatch a DOM event for non-JS consumers (CSS hooks, etc.).
    try {
      var evt;
      if (typeof CustomEvent === "function") {
        evt = new CustomEvent("pfm:alpha-hub-filters-change", { detail: snap });
      } else {
        evt = document.createEvent("CustomEvent");
        evt.initCustomEvent("pfm:alpha-hub-filters-change", false, false, snap);
      }
      document.dispatchEvent(evt);
    } catch (e) {
      /* ignore */
    }
  }

  /* ---------- Mount ----------------------------------------------------- */

  function resolveContainer(target) {
    if (!target) return null;
    if (typeof target === "string") {
      try {
        return document.querySelector(target);
      } catch (e) {
        return null;
      }
    }
    if (target.nodeType === 1) return target;
    return null;
  }

  function mount(target) {
    var container = resolveContainer(target);
    if (!container) {
      if (window.console && console.warn) {
        console.warn("[alpha-hub-filters] mount: container not found", target);
      }
      return null;
    }

    // Restore from URL hash before first render so initial UI matches state.
    var fromHash = parseHash();
    if (fromHash) {
      // Don't fire callbacks during initial restore; just seed state.
      if (Object.prototype.hasOwnProperty.call(fromHash, "tier")) {
        state.tier = normalizeTier(fromHash.tier);
      }
      if (Object.prototype.hasOwnProperty.call(fromHash, "theme")) {
        state.theme = normalizeTheme(fromHash.theme);
      }
      if (Object.prototype.hasOwnProperty.call(fromHash, "min_sharpe")) {
        state.min_sharpe = clampSharpe(fromHash.min_sharpe);
      }
    }

    // Clear element refs (a re-mount replaces the previous bar).
    elements.tierChips = {};
    elements.themeChips = {};
    elements.slider = null;
    elements.sliderValue = null;
    elements.reset = null;

    var tierGroup = buildChipGroup(
      "Tier", TIERS, state.tier,
      function (v) { setFilter("tier", v); },
      elements.tierChips
    );

    var themeGroup = buildChipGroup(
      "Theme", THEMES, state.theme,
      function (v) { setFilter("theme", v); },
      elements.themeChips
    );

    var sliderGroup = buildSliderGroup();

    var reset = el("button", {
      type: "button",
      class: "pfm-ahf-reset",
      text: "Reset",
      hidden: true,
      "aria-label": "Reset all filters",
      onClick: function () {
        applyState(
          { tier: TIER_DEFAULT, theme: THEME_DEFAULT, min_sharpe: SHARPE_DEFAULT },
          {}
        );
      }
    });
    elements.reset = reset;

    var bar = el(
      "div",
      {
        class: "pfm-ahf-bar",
        role: "toolbar",
        "aria-label": "Alpha Hub filters",
        "data-pfm-ahf-root": "true"
      },
      [tierGroup, themeGroup, sliderGroup, reset]
    );

    // Idempotency: if we previously mounted into this container, wipe.
    var prev = container.querySelector('[data-pfm-ahf-root="true"]');
    if (prev && prev.parentNode === container) {
      container.removeChild(prev);
    }
    container.appendChild(bar);
    rootEl = bar;

    renderAll();

    // hashchange wiring (one global handler — re-mounting doesn't double up).
    if (!mount.__hashWired) {
      window.addEventListener("hashchange", onHashChange);
      mount.__hashWired = true;
    }

    return bar;
  }

  /* ---------- Public surface ------------------------------------------- */

  window.PFM = window.PFM || {};
  window.PFM.alphaHubFilters = {
    __w1240: true,
    mount: mount,
    setFilter: setFilter,
    getFilters: getFilters,
    onChange: onChange,
    // Exposed for tests / advanced consumers (not part of the documented API):
    _state: state,
    _tiers: TIERS.slice(),
    _themes: THEMES.slice()
  };
})();

/* ============================================================================
 * End of alpha-hub-filters.js
 * ============================================================================ */
