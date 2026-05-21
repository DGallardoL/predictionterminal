/* ─────────────────────────────────────────────────────────────────────
 * cmdk.js — Command palette (Cmd+K / Ctrl+K)
 *
 * Vanilla ES2020+. No build step. Self-executes on DOMContentLoaded.
 * Exposes window.pfmCmdk = { open, close, toggle, register }.
 * Uses a single internal namespace __pfmCmdkState for module-private data.
 * ───────────────────────────────────────────────────────────────────── */
(() => {
  "use strict";

  if (window.pfmCmdk) return; // idempotent

  // ── State ───────────────────────────────────────────────────────────
  /** @type {any} */
  const state = (window.__pfmCmdkState = {
    open: false,
    query: "",
    results: [],            // flat list of items currently displayed
    selectedIndex: 0,
    debounceTimer: 0,
    requestSeq: 0,
    cache: {
      strategies: null,     // array of strategy items
      strategiesAt: 0,
      factors: new Map(),   // query -> array (legacy /factors endpoint)
      alerts: null,         // array or null
      alertsAt: 0,
    },
    pluginSources: [],      // future plugin sources via register()
    prevFocus: null,
    nodes: {},              // dom refs
  });

  const TTL_MS = 60_000;
  const FACTOR_CATALOG_TTL_MS = 10 * 60_000;  // 10 min, per spec
  const DEBOUNCE_MS = 150;
  const POLYMARKET_BASE = "https://polymarket.com/event/";

  // Theme colour accents (kept light-touch — actual colour is via CSS pill).
  const FACTOR_THEMES = new Set([
    "politics", "macro", "ai", "crypto", "sports", "geopolitics",
    "chips", "commodities", "climate", "health", "equity", "energy",
    "pop_culture", "legal", "weather", "space", "science", "business",
    "other",
  ]);

  // ── API base resolution ─────────────────────────────────────────────
  // The page is served from /ui/. Same-origin GETs are fine. Allow override
  // via window.PFM_API_BASE (some installs set this in config.js).
  const apiBase = () => {
    const b =
      (window.PFM_CONFIG && window.PFM_CONFIG.apiBase) ||
      window.PFM_API_BASE ||
      "";
    return typeof b === "string" ? b.replace(/\/+$/, "") : "";
  };

  const apiUrl = (path, params) => {
    const base = apiBase();
    const url = new URL(
      (base ? base : window.location.origin) + path,
      window.location.origin,
    );
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        if (v !== undefined && v !== null && v !== "") {
          url.searchParams.set(k, String(v));
        }
      }
    }
    return url.toString();
  };

  // ── Static endpoint catalog ─────────────────────────────────────────
  // Used as the "endpoints" source. Kept small and curated; OpenAPI
  // introspection would be overkill for a search UI.
  const STATIC_ENDPOINTS = [
    { path: "/terminal/overview",         desc: "Terminal landing — aggregated market snapshot",  mode: "terminal" },
    { path: "/terminal/search",           desc: "Search markets, contracts and tickers",          mode: "terminal" },
    { path: "/terminal/calendar-curated", desc: "Curated event calendar across factors",          mode: "terminal" },
    { path: "/terminal/news",             desc: "News tape tagged to prediction-market factors",  mode: "terminal" },
    { path: "/factors",                   desc: "List the factor catalog",                        mode: "regression" },
    { path: "/factors/discover",          desc: "Discover candidate factors by ticker / theme",   mode: "regression" },
    { path: "/fit",                       desc: "Fit a factor model on stock returns",            mode: "regression" },
    { path: "/attribution",               desc: "Attribute returns to fitted factors",            mode: "regression" },
  ];

  // ── Icon mapping ────────────────────────────────────────────────────
  const TYPE_ICON = {
    factor: "ƒ",
    strategy: "Σ",
    endpoint: "→",
    alert: "⚠",
  };

  // ── DOM construction ────────────────────────────────────────────────
  function buildDom() {
    const backdrop = document.createElement("div");
    backdrop.className = "pfm-cmdk-backdrop";
    backdrop.setAttribute("data-open", "false");

    const modal = document.createElement("div");
    modal.className = "pfm-cmdk-modal";
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-label", "Command palette");
    modal.tabIndex = -1;

    // input row
    const inputRow = document.createElement("div");
    inputRow.className = "pfm-cmdk-input-row";
    const input = document.createElement("input");
    input.type = "text";
    input.className = "pfm-cmdk-input";
    input.placeholder = "Search factors, strategies, endpoints, alerts…";
    input.setAttribute("autocomplete", "off");
    input.setAttribute("autocorrect", "off");
    input.setAttribute("spellcheck", "false");
    input.setAttribute("aria-controls", "pfm-cmdk-listbox");
    input.setAttribute("aria-autocomplete", "list");
    const kbdHint = document.createElement("span");
    kbdHint.className = "pfm-cmdk-kbdhint";
    kbdHint.textContent = isMac() ? "⌘K" : "Ctrl+K";
    inputRow.appendChild(input);
    inputRow.appendChild(kbdHint);

    const sep = document.createElement("div");
    sep.className = "pfm-cmdk-sep";

    const list = document.createElement("ul");
    list.className = "pfm-cmdk-list";
    list.id = "pfm-cmdk-listbox";
    list.setAttribute("role", "listbox");
    list.setAttribute("aria-label", "Search results");

    const footer = document.createElement("div");
    footer.className = "pfm-cmdk-footer";
    footer.innerHTML =
      '<span><kbd>↑</kbd><kbd>↓</kbd> navigate · <kbd>↵</kbd> select · <kbd>esc</kbd> close</span>' +
      '<span>cmd+k</span>';

    modal.appendChild(inputRow);
    modal.appendChild(sep);
    modal.appendChild(list);
    modal.appendChild(footer);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);

    state.nodes = { backdrop, modal, input, list, footer };

    // Wire events
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) close();
    });
    input.addEventListener("input", onInput);
    input.addEventListener("keydown", onInputKeydown);
    list.addEventListener("click", onListClick);
    list.addEventListener("mousemove", onListHover);
    modal.addEventListener("keydown", onModalKeydown);
  }

  // ── Platform / utility ──────────────────────────────────────────────
  function isMac() {
    return /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent || "");
  }

  function isTypingTarget(el) {
    if (!el) return false;
    const tag = (el.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return true;
    if (el.isContentEditable) return true;
    return false;
  }

  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // ── Data fetchers ───────────────────────────────────────────────────
  async function fetchJSON(url, opts) {
    try {
      const resp = await fetch(url, { credentials: "same-origin", ...(opts || {}) });
      if (!resp.ok) return { ok: false, status: resp.status, data: null };
      const data = await resp.json();
      return { ok: true, status: resp.status, data };
    } catch (err) {
      return { ok: false, status: 0, data: null, error: err };
    }
  }

  async function fetchFactors(query) {
    const q = (query || "").trim();
    const cacheKey = q.toLowerCase();
    const now = Date.now();
    const cached = state.cache.factors.get(cacheKey);
    if (cached && now - cached.at < TTL_MS) return cached.items;

    const params = { limit: 200 };
    if (q) params.search = q;
    const { ok, data } = await fetchJSON(apiUrl("/factors", params));
    if (!ok || !data) return [];
    const facs = Array.isArray(data.factors) ? data.factors : [];
    state.cache.factors.set(cacheKey, { items: facs, at: now });
    return facs;
  }

  /**
   * Fetch the *full* factor catalog once and cache it on window for 10 min.
   * Returns an array of {id, name, slug, theme, source, description}.
   * Falls back to the existing paginated /factors endpoint if /factors/all
   * is unavailable, so the palette stays useful on older backends.
   */
  async function fetchFactorCatalog() {
    const now = Date.now();
    const c = window._cmdkFactorCache;
    if (c && Array.isArray(c.items) && now - c.at < FACTOR_CATALOG_TTL_MS) {
      return c.items;
    }
    let items = [];
    const r = await fetchJSON(apiUrl("/factors/all"));
    if (r.ok && r.data) {
      if (Array.isArray(r.data.factors)) items = r.data.factors;
      else if (Array.isArray(r.data)) items = r.data;
    }
    if (!items.length) {
      // graceful fallback to /factors (paginated, max ~200)
      const r2 = await fetchJSON(apiUrl("/factors", { limit: 1000 }));
      if (r2.ok && r2.data && Array.isArray(r2.data.factors)) items = r2.data.factors;
    }
    window._cmdkFactorCache = { items, at: now };
    return items;
  }

  /**
   * Score a factor against the query.
   *   - id exact:          1000
   *   - id startswith:      700
   *   - id substring:       500
   *   - name startswith:    400
   *   - name substring:     300
   *   - theme exact:        200
   *   - theme substring:    100
   *   - description match:   50
   * Returns 0 when nothing matches (caller filters those out for non-empty q).
   */
  function scoreFactor(f, qLower) {
    if (!qLower) return 1; // empty query — keep all, ordering handled upstream
    const id = String(f.id || "").toLowerCase();
    const name = String(f.name || "").toLowerCase();
    const theme = String(f.theme || "").toLowerCase();
    const desc = String(f.description || "").toLowerCase();
    if (id === qLower) return 1000;
    if (id.startsWith(qLower)) return 700;
    if (id.includes(qLower)) return 500;
    if (name.startsWith(qLower)) return 400;
    if (name.includes(qLower)) return 300;
    if (theme === qLower) return 200;
    if (theme.includes(qLower)) return 100;
    if (desc.includes(qLower)) return 50;
    return 0;
  }

  function searchFactorCatalog(catalog, query, limit) {
    const q = (query || "").trim().toLowerCase();
    if (!q) return catalog.slice(0, limit || 12);
    const scored = [];
    for (const f of catalog) {
      const s = scoreFactor(f, q);
      if (s > 0) scored.push({ f, s });
    }
    scored.sort((a, b) => b.s - a.s);
    return scored.slice(0, limit || 12).map((x) => x.f);
  }

  async function fetchStrategies() {
    const now = Date.now();
    if (state.cache.strategies && now - state.cache.strategiesAt < TTL_MS) {
      return state.cache.strategies;
    }
    const { ok, data } = await fetchJSON(apiUrl("/strategies/list"));
    if (!ok || !data) {
      // graceful degrade — cache an empty list with a short TTL by leaving timestamp 0
      return [];
    }
    const items = Array.isArray(data.items) ? data.items : [];
    state.cache.strategies = items;
    state.cache.strategiesAt = now;
    return items;
  }

  async function fetchAlerts() {
    const now = Date.now();
    if (state.cache.alerts !== null && now - state.cache.alertsAt < TTL_MS) {
      return state.cache.alerts;
    }
    const { ok, status, data } = await fetchJSON(apiUrl("/alerts", { user_id: "default" }));
    if (!ok) {
      // 401 / 422 / etc — silent skip
      state.cache.alerts = [];
      state.cache.alertsAt = now;
      return [];
    }
    // alerts may return a bare list OR an object with items — handle both
    let items = [];
    if (Array.isArray(data)) items = data;
    else if (data && Array.isArray(data.items)) items = data.items;
    else if (data && Array.isArray(data.alerts)) items = data.alerts;
    state.cache.alerts = items;
    state.cache.alertsAt = now;
    return items;
  }

  // ── Item normalisers ────────────────────────────────────────────────
  function factorToItem(f) {
    const slug = f.slug || "";
    const id = f.id || slug || "";
    const theme = String(f.theme || "").toLowerCase();
    // Surface the FACTOR ID prominently (it's what users type when filtering
    // and it's what gets added as a chip), with the human-readable name as
    // the secondary line. This is the inverse of the legacy ordering.
    const title = id || f.name || slug || "(unnamed factor)";
    const sub = f.name && f.name !== id ? f.name : slug;
    const pill = FACTOR_THEMES.has(theme) ? theme : (theme || "factor");
    return {
      kind: "factor",
      icon: TYPE_ICON.factor,
      title,
      sub,
      pill,
      // Highlight A-tier-ish themes with the orange accent so they stand out.
      pillAccent: theme === "macro" || theme === "ai" || theme === "crypto",
      _raw: f,
    };
  }
  function strategyToItem(s) {
    const tag = (s.tag || "").toString();
    return {
      kind: "strategy",
      icon: TYPE_ICON.strategy,
      title: s.id || s.name || "(strategy)",
      sub: s.endpoint || s.description || "",
      pill: tag || "strategy",
      pillAccent: /validated|gold|deployable/i.test(tag),
      _raw: s,
    };
  }
  function endpointToItem(e) {
    return {
      kind: "endpoint",
      icon: TYPE_ICON.endpoint,
      title: e.path,
      sub: e.desc || "",
      pill: "endpoint",
      pillAccent: false,
      _raw: e,
    };
  }
  function alertToItem(a) {
    const name = a.name || a.label || a.id || "Alert";
    const sub = a.expression || a.factor || a.slug || a.condition || "";
    return {
      kind: "alert",
      icon: TYPE_ICON.alert,
      title: String(name),
      sub: String(sub),
      pill: "alert",
      pillAccent: false,
      _raw: a,
    };
  }

  // ── Client-side filter for cached sources ───────────────────────────
  function clientFilter(items, query, fields) {
    const q = (query || "").trim().toLowerCase();
    if (!q) return items;
    return items.filter((it) => {
      for (const f of fields) {
        const v = it[f];
        if (typeof v === "string" && v.toLowerCase().includes(q)) return true;
      }
      return false;
    });
  }

  // Detect "factors-only" filter prefixes: "f " or "f:" (case-insensitive).
  // Returns { onlyFactors: bool, stripped: string }.
  function parseFactorFilter(query) {
    const raw = String(query || "");
    const m = raw.match(/^\s*f\s*[:\s]\s*(.*)$/i);
    if (m) return { onlyFactors: true, stripped: m[1] };
    return { onlyFactors: false, stripped: raw };
  }

  // ── Main search orchestrator ────────────────────────────────────────
  async function runSearch(query) {
    const seq = ++state.requestSeq;
    const rawQ = (query || "").trim();
    const { onlyFactors, stripped } = parseFactorFilter(rawQ);
    const q = stripped.trim();

    // Endpoints: client filter
    const endpointMatches = clientFilter(STATIC_ENDPOINTS, q, ["path", "desc"]);

    // Load all sources in parallel. The factor catalog is fetched once and
    // searched in-memory; legacy fetchFactors stays as a fallback.
    const [strategies, alerts, catalog, plugin] = await Promise.all([
      fetchStrategies(),
      fetchAlerts(),
      fetchFactorCatalog(),
      runPluginSources(q),
    ]);

    if (seq !== state.requestSeq) return; // stale

    const factorMatches = searchFactorCatalog(catalog, q, onlyFactors ? 50 : 12);

    const strategyMatches = clientFilter(
      strategies,
      q,
      ["id", "endpoint", "description", "tag"],
    );
    const alertMatches = clientFilter(
      alerts,
      q,
      ["name", "label", "id", "expression", "factor", "slug", "condition"],
    );

    // Assemble grouped results.
    /** @type {{label:string, items:any[]}[]} */
    const groups = [];

    if (onlyFactors) {
      groups.push({
        label: q ? `Factors matching "${q}"` : "Factors",
        items: factorMatches.map(factorToItem),
      });
    } else if (!q) {
      // Empty query: show a "Recent" placeholder + top of each source.
      // Tickers/markets first (endpoints/strategies act as those entry-points
      // until a dedicated source is wired); factors second per spec.
      groups.push({ label: "Recent", items: [] });
      groups.push({
        label: "Strategies",
        items: strategyMatches.slice(0, 4).map(strategyToItem),
      });
      groups.push({
        label: "Endpoints",
        items: endpointMatches.slice(0, 4).map(endpointToItem),
      });
      groups.push({
        label: "Factors",
        items: factorMatches.slice(0, 6).map(factorToItem),
      });
      if (alertMatches.length) {
        groups.push({
          label: "Alerts",
          items: alertMatches.slice(0, 4).map(alertToItem),
        });
      }
    } else {
      if (strategyMatches.length) {
        groups.push({
          label: "Strategies",
          items: strategyMatches.slice(0, 8).map(strategyToItem),
        });
      }
      if (endpointMatches.length) {
        groups.push({
          label: "Endpoints",
          items: endpointMatches.slice(0, 8).map(endpointToItem),
        });
      }
      if (factorMatches.length) {
        groups.push({
          label: "Factors",
          items: factorMatches.map(factorToItem),
        });
      }
      if (alertMatches.length) {
        groups.push({
          label: "Alerts",
          items: alertMatches.slice(0, 6).map(alertToItem),
        });
      }
      if (plugin && plugin.length) {
        groups.push({ label: "Plugins", items: plugin });
      }
    }

    renderResults(groups);
  }

  async function runPluginSources(query) {
    if (!state.pluginSources.length) return [];
    const out = [];
    for (const src of state.pluginSources) {
      try {
        const items = await src.search(query);
        if (Array.isArray(items)) {
          for (const it of items) {
            out.push({
              kind: src.name || "plugin",
              icon: src.icon || "•",
              title: it.title || "",
              sub: it.sub || "",
              pill: src.name || "plugin",
              pillAccent: false,
              _raw: it,
            });
          }
        }
      } catch (_e) {
        // swallow plugin errors so they never break the palette
      }
    }
    return out;
  }

  // ── Rendering ───────────────────────────────────────────────────────
  function renderResults(groups) {
    const { list } = state.nodes;
    list.innerHTML = "";
    state.results = [];

    const flat = [];
    for (const g of groups) {
      // section header (always render so empty groups read as sections)
      const header = document.createElement("li");
      header.className = "pfm-cmdk-section";
      header.textContent = g.label;
      header.setAttribute("role", "presentation");
      list.appendChild(header);

      if (!g.items.length) {
        const empty = document.createElement("li");
        empty.className = "pfm-cmdk-empty";
        empty.textContent = state.query
          ? "No matches"
          : g.label === "Recent"
          ? "Nothing yet — your recent picks will show up here."
          : "Nothing here yet.";
        empty.setAttribute("role", "presentation");
        list.appendChild(empty);
        continue;
      }

      for (const item of g.items) {
        const li = document.createElement("li");
        li.className = "pfm-cmdk-row";
        li.setAttribute("role", "option");
        li.setAttribute("aria-selected", "false");
        const idx = flat.length;
        li.dataset.idx = String(idx);

        li.innerHTML =
          `<span class="pfm-cmdk-icon" aria-hidden="true">${escapeHtml(item.icon)}</span>` +
          `<span class="pfm-cmdk-text">` +
            `<span class="pfm-cmdk-title">${escapeHtml(item.title)}</span>` +
            `<span class="pfm-cmdk-sub">${escapeHtml(item.sub)}</span>` +
          `</span>` +
          `<span class="pfm-cmdk-pill"${item.pillAccent ? ' data-accent="orange"' : ""}>${escapeHtml(item.pill)}</span>`;

        list.appendChild(li);
        flat.push({ item, node: li });
      }
    }

    state.results = flat;

    if (!flat.length) {
      const empty = document.createElement("li");
      empty.className = "pfm-cmdk-empty";
      empty.textContent = state.query
        ? `No matches for "${state.query}".`
        : "Start typing to search.";
      empty.setAttribute("role", "presentation");
      list.appendChild(empty);
      state.selectedIndex = -1;
      return;
    }

    state.selectedIndex = 0;
    updateSelection();
  }

  function updateSelection() {
    const { results, selectedIndex } = state;
    results.forEach((r, i) => {
      const sel = i === selectedIndex;
      r.node.setAttribute("aria-selected", sel ? "true" : "false");
      if (sel) {
        r.node.scrollIntoView({ block: "nearest" });
      }
    });
    if (state.nodes.input && selectedIndex >= 0 && results[selectedIndex]) {
      state.nodes.input.setAttribute(
        "aria-activedescendant",
        `pfm-cmdk-row-${selectedIndex}`,
      );
      results[selectedIndex].node.id = `pfm-cmdk-row-${selectedIndex}`;
    }
  }

  function move(delta) {
    if (!state.results.length) return;
    const n = state.results.length;
    state.selectedIndex = (state.selectedIndex + delta + n) % n;
    updateSelection();
  }

  // ── Event handlers ──────────────────────────────────────────────────
  function onInput(e) {
    state.query = e.target.value;
    if (state.debounceTimer) clearTimeout(state.debounceTimer);
    state.debounceTimer = setTimeout(() => {
      runSearch(state.query);
    }, DEBOUNCE_MS);
  }

  function onInputKeydown(e) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      move(1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      move(-1);
    } else if (e.key === "Enter") {
      e.preventDefault();
      activateSelection();
    } else if (e.key === "Escape") {
      e.preventDefault();
      close();
    } else if ((e.metaKey || e.ctrlKey) && /^[1-9]$/.test(e.key)) {
      e.preventDefault();
      const idx = parseInt(e.key, 10) - 1;
      if (idx < state.results.length) {
        state.selectedIndex = idx;
        updateSelection();
        activateSelection();
      }
    }
    // do NOT swallow Cmd/Ctrl+K — let onModalKeydown handle close, and let
    // normal text typing happen otherwise.
  }

  function onModalKeydown(e) {
    // close on Cmd/Ctrl+K when palette is already open
    if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) {
      e.preventDefault();
      close();
      return;
    }
    // simple focus trap: Tab stays within modal (we only have one focusable
    // input anyway, so just force focus back to it)
    if (e.key === "Tab") {
      e.preventDefault();
      state.nodes.input.focus();
    }
  }

  function onListClick(e) {
    const row = e.target.closest(".pfm-cmdk-row");
    if (!row) return;
    const idx = parseInt(row.dataset.idx || "-1", 10);
    if (idx >= 0) {
      state.selectedIndex = idx;
      activateSelection();
    }
  }

  function onListHover(e) {
    const row = e.target.closest(".pfm-cmdk-row");
    if (!row) return;
    const idx = parseInt(row.dataset.idx || "-1", 10);
    if (idx >= 0 && idx !== state.selectedIndex) {
      state.selectedIndex = idx;
      updateSelection();
    }
  }

  // Returns the active mode by reading the .mode-switch button state. The
  // index.html mode switch keeps a single .active button; if it's missing
  // we assume "regression" (legacy default).
  function currentMode() {
    try {
      const active = document.querySelector(".mode-switch .mode-btn.active");
      if (active && active.dataset && active.dataset.mode) return active.dataset.mode;
    } catch (_e) {}
    return "regression";
  }

  function showToast(msg, kind) {
    try {
      if (window.pfmToast) { window.pfmToast(msg, kind || "success", 2200); return; }
    } catch (_e) {}
    // Fallback: tiny inline toast so we still confirm the action.
    try {
      const t = document.createElement("div");
      t.textContent = String(msg);
      t.style.cssText =
        "position:fixed;top:80px;right:20px;background:#16a34a;color:white;" +
        "padding:8px 14px;border-radius:6px;z-index:99999;font-size:13px;" +
        "font-family:system-ui,sans-serif;box-shadow:0 4px 12px rgba(0,0,0,.18);";
      document.body.appendChild(t);
      setTimeout(() => { try { t.remove(); } catch (_) {} }, 2200);
    } catch (_e) {}
  }

  // Add a factor id to the Regression chip set, switching mode first if we're
  // not already there. If the page hasn't exposed the helper yet we fall back
  // to opening the underlying Polymarket event so the click is never a no-op.
  function addFactorToRegression(it) {
    const f = it._raw || {};
    const fid = f.id || f.slug || "";
    if (!fid) return;
    const inRegression = currentMode() === "regression";
    const adder = window.addFactorToRegressionChips;

    const doAdd = () => {
      try {
        const added = typeof adder === "function" ? adder(fid) : false;
        if (added) {
          showToast(`Added ${fid} to Regression`, "success");
        } else if (typeof adder === "function") {
          showToast(`${fid} already selected`, "info");
        } else {
          // Helper not present (older page) — open Polymarket instead.
          openFactorOnPolymarket(f);
        }
      } catch (_e) {
        openFactorOnPolymarket(f);
      }
    };

    if (!inRegression) {
      try {
        if (typeof window.setMode === "function") window.setMode("regression");
      } catch (_e) {}
      // setMode flips classes synchronously, but the chip wiring may rely on
      // a tick of layout — defer the add by a frame so handlers settle.
      requestAnimationFrame(() => setTimeout(doAdd, 0));
    } else {
      doAdd();
    }
  }

  function openFactorOnPolymarket(f) {
    const slug = (f && (f.slug || f.id)) || "";
    if (!slug) return;
    const url = POLYMARKET_BASE + encodeURIComponent(slug);
    try { window.open(url, "_blank", "noopener,noreferrer"); } catch (_e) {
      window.location.href = url;
    }
  }

  // ── Selection actions ───────────────────────────────────────────────
  function activateSelection() {
    const sel = state.results[state.selectedIndex];
    if (!sel) return;
    const it = sel.item;
    try {
      switch (it.kind) {
        case "factor": {
          // Primary action: add to Regression chips (switching mode if needed).
          // Fallback inside addFactorToRegression opens Polymarket when the
          // page hasn't exposed the chip helper.
          addFactorToRegression(it);
          break;
        }
        case "strategy": {
          const setMode = window.setMode;  // exposed by index.html (mode-switch handler)
          if (typeof setMode === "function") {
            try { setMode("strategies"); } catch (_) {}
          }
          const target =
            document.getElementById("strategies-mode") ||
            document.getElementById("strategies");
          if (target && typeof target.scrollIntoView === "function") {
            target.scrollIntoView({ behavior: "smooth", block: "start" });
          } else {
            window.location.hash = "#strategies";
          }
          break;
        }
        case "endpoint": {
          const path = it._raw.path;
          // copy to clipboard, best-effort
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(path).catch(() => {});
          }
          // scroll to relevant mode container
          const mode = it._raw.mode;
          const setMode = window.setMode;  // exposed by index.html (mode-switch handler)
          if (mode && typeof setMode === "function") {
            try { setMode(mode); } catch (_) {}
          }
          const target =
            (mode && document.getElementById(mode + "-mode")) ||
            (mode && document.getElementById(mode));
          if (target && typeof target.scrollIntoView === "function") {
            target.scrollIntoView({ behavior: "smooth", block: "start" });
          }
          break;
        }
        case "alert": {
          // no-op for now — graceful
          break;
        }
        default:
          // plugin item: ignore — plugins handle their own actions
          break;
      }
    } catch (_err) {
      // never throw out of a click handler
    }
    close();
  }

  // ── Open / close ────────────────────────────────────────────────────
  function open() {
    if (state.open) return;
    state.open = true;
    state.prevFocus = document.activeElement;
    const { backdrop, input } = state.nodes;
    backdrop.setAttribute("data-open", "true");
    // reset query state every open
    input.value = "";
    state.query = "";
    state.selectedIndex = 0;
    state.results = [];
    // initial render: show empty-query layout immediately, then refresh
    renderResults([{ label: "Recent", items: [] }]);
    runSearch("");
    // focus after the next frame so the fade-in plays
    requestAnimationFrame(() => {
      try { input.focus({ preventScroll: true }); } catch (_) { input.focus(); }
    });
  }

  function close() {
    if (!state.open) return;
    state.open = false;
    const { backdrop } = state.nodes;
    backdrop.setAttribute("data-open", "false");
    if (state.prevFocus && typeof state.prevFocus.focus === "function") {
      try { state.prevFocus.focus(); } catch (_) {}
    }
    state.prevFocus = null;
  }

  function toggle() { state.open ? close() : open(); }

  // ── Global key trigger ──────────────────────────────────────────────
  function onGlobalKeydown(e) {
    // Cmd+K (mac) / Ctrl+K (others)
    const isCmdK = (e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K");
    if (isCmdK) {
      // Don't hijack when typing inside our own input
      if (state.open && e.target === state.nodes.input) {
        e.preventDefault();
        close();
        return;
      }
      e.preventDefault();
      toggle();
      return;
    }
    // "/" opens when no input is focused and the palette is closed
    if (e.key === "/" && !state.open) {
      if (isTypingTarget(e.target)) return;
      e.preventDefault();
      open();
      return;
    }
    // Escape closes if open (in addition to input-level handler)
    if (e.key === "Escape" && state.open) {
      e.preventDefault();
      close();
    }
  }

  // ── Plugin source registration ──────────────────────────────────────
  function register(source) {
    if (!source || typeof source.search !== "function" || !source.name) return;
    // de-dup by name
    state.pluginSources = state.pluginSources.filter((s) => s.name !== source.name);
    state.pluginSources.push(source);
  }

  // ── Init ────────────────────────────────────────────────────────────
  function init() {
    if (state.nodes && state.nodes.backdrop) return; // already built
    buildDom();
    // capture: true so we get Cmd+K even when focus is inside other inputs
    document.addEventListener("keydown", onGlobalKeydown, { capture: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }

  // ── Public API ──────────────────────────────────────────────────────
  window.pfmCmdk = Object.freeze({
    open,
    close,
    toggle,
    register,
  });
})();
