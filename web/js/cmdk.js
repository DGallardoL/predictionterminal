/* T03 — Command Palette (⌘K / Ctrl+K)
 *
 * Vanilla JS, no deps. Self-mounts on DOMContentLoaded.
 *
 * Public API:   window.PFM.cmdk = { open(), close(), toggle(), register(commands) }
 *
 * Keybinds:
 *   ⌘K / Ctrl+K  — toggle palette
 *   Esc          — close
 *   ↑ / ↓        — navigate results
 *   Enter        — execute selected command
 *   Tab          — cycle category filter (All / Modes / Factors / Slugs / Slashes)
 *   /            — insert slash prefix (when input empty)
 *
 * Sources:
 *   1. Modes        — Regression / Strategies / Terminal       (dispatches `pfm:switch-mode`)
 *   2. Slash actions — /jumps /backtest /clusters /fit
 *   3. Factors      — GET /factors (cached 5 min in memory)
 *   4. Slugs        — GET /terminal/search?q=... (200ms debounce)
 *   5. Recents      — localStorage `pfm:cmdk:recents` (last 10)
 *
 * Mount: <link rel="stylesheet" href="/css/cmdk.css"> + <script defer src="/js/cmdk.js"></script>
 * (index-html-owner adds these; the script self-mounts the root DOM.)
 */
(function () {
  "use strict";

  // ---------- config ----------
  var API_BASE =
    (window.PFM && window.PFM.apiBase) ||
    (window.PFM_API_BASE) ||
    (window.location && window.location.port === "8080" ? "http://127.0.0.1:8000" : "");

  var FACTOR_CACHE_TTL_MS = 5 * 60 * 1000;
  var SEARCH_DEBOUNCE_MS = 200;
  var RECENTS_KEY = "pfm:cmdk:recents";
  var RECENTS_MAX = 10;
  var MAX_RESULTS_PER_CAT = 8;

  var CATEGORIES = [
    { id: "all", label: "All" },
    { id: "mode", label: "Modes" },
    { id: "factor", label: "Factors" },
    { id: "slug", label: "Slugs" },
    { id: "slash", label: "Slashes" },
  ];

  // ---------- state ----------
  var state = {
    open: false,
    query: "",
    activeFilter: "all",
    activeIndex: 0,
    results: [], // flattened visible items
    factors: null,
    factorsAt: 0,
    slugs: [],
    slugsLoadingFor: "",
    registered: [], // externally registered commands
    debounceTimer: null,
    abortCtrl: null,
  };

  // ---------- utils ----------
  function isMacLike() {
    return /Mac|iPhone|iPad|iPod/.test(navigator.platform || "");
  }

  function escapeHtml(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function safeReadJSON(key, fallback) {
    try {
      var raw = localStorage.getItem(key);
      if (!raw) return fallback;
      var v = JSON.parse(raw);
      return v == null ? fallback : v;
    } catch (e) {
      return fallback;
    }
  }

  function safeWriteJSON(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* quota */ }
  }

  function debounce(fn, ms) {
    var t = null;
    return function () {
      var ctx = this;
      var args = arguments;
      if (t) clearTimeout(t);
      t = setTimeout(function () { fn.apply(ctx, args); }, ms);
    };
  }

  // ---------- fuzzy matching ----------
  // Subsequence match with positional scoring. Returns { score, indices } or null.
  function fuzzyMatch(query, target) {
    if (!query) return { score: 0, indices: [] };
    if (!target) return null;
    var q = query.toLowerCase();
    var t = String(target).toLowerCase();

    // Fast exact substring boost
    var sub = t.indexOf(q);
    if (sub !== -1) {
      var idxs = [];
      for (var k = 0; k < q.length; k++) idxs.push(sub + k);
      // boost if at start, or word-boundary
      var bonus = (sub === 0 ? 60 : 0) + (sub > 0 && /[\W_]/.test(t.charAt(sub - 1)) ? 25 : 0);
      var penalty = Math.min(40, t.length); // shorter targets score higher
      return { score: 1000 + bonus - penalty - sub * 2, indices: idxs };
    }

    // Subseq match
    var qi = 0, ti = 0, indices = [], lastIdx = -2, score = 0;
    while (qi < q.length && ti < t.length) {
      if (q.charAt(qi) === t.charAt(ti)) {
        indices.push(ti);
        // contiguous bonus
        if (ti === lastIdx + 1) score += 8;
        // word-boundary bonus
        if (ti === 0 || /[\W_]/.test(t.charAt(ti - 1))) score += 12;
        score += 4;
        lastIdx = ti;
        qi++;
      }
      ti++;
    }
    if (qi < q.length) return null;
    // length penalty
    score -= (t.length - q.length) * 0.2;
    return { score: score, indices: indices };
  }

  function highlight(text, indices) {
    if (!text) return "";
    if (!indices || !indices.length) return escapeHtml(text);
    var out = "";
    var set = {};
    for (var i = 0; i < indices.length; i++) set[indices[i]] = 1;
    var inMark = false;
    for (var j = 0; j < text.length; j++) {
      var hit = set[j] === 1;
      if (hit && !inMark) { out += "<mark>"; inMark = true; }
      if (!hit && inMark) { out += "</mark>"; inMark = false; }
      out += escapeHtml(text.charAt(j));
    }
    if (inMark) out += "</mark>";
    return out;
  }

  // ---------- recents ----------
  function readRecents() {
    var arr = safeReadJSON(RECENTS_KEY, []);
    return Array.isArray(arr) ? arr : [];
  }

  function pushRecent(entry) {
    if (!entry || !entry.id) return;
    var rec = readRecents();
    // dedupe
    rec = rec.filter(function (r) { return r && r.id !== entry.id; });
    rec.unshift({
      id: entry.id,
      kind: entry.kind,
      title: entry.title,
      sub: entry.sub || "",
      payload: entry.payload || null,
      at: Date.now(),
    });
    if (rec.length > RECENTS_MAX) rec.length = RECENTS_MAX;
    safeWriteJSON(RECENTS_KEY, rec);
  }

  // ---------- fetchers ----------
  function fetchFactors() {
    var now = Date.now();
    if (state.factors && (now - state.factorsAt) < FACTOR_CACHE_TTL_MS) {
      return Promise.resolve(state.factors);
    }
    return fetch(API_BASE + "/factors", { credentials: "omit" })
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (data) {
        // Accept either { factors: [...] } or [...] shape
        var list = Array.isArray(data) ? data : (data && data.factors) || [];
        state.factors = list;
        state.factorsAt = Date.now();
        return list;
      })
      .catch(function () { return state.factors || []; });
  }

  function fetchSlugs(q) {
    if (state.abortCtrl) {
      try { state.abortCtrl.abort(); } catch (e) { /* noop */ }
    }
    if (!q || q.length < 2) {
      state.slugs = [];
      return Promise.resolve([]);
    }
    var ctl = (typeof AbortController !== "undefined") ? new AbortController() : null;
    state.abortCtrl = ctl;
    state.slugsLoadingFor = q;
    setLoading(true);
    return fetch(API_BASE + "/terminal/search?q=" + encodeURIComponent(q), {
      credentials: "omit",
      signal: ctl ? ctl.signal : undefined,
    })
      .then(function (r) { return r.ok ? r.json() : { results: [] }; })
      .then(function (data) {
        var rows = (data && (data.results || data.markets || data.items)) || [];
        if (!Array.isArray(rows)) rows = [];
        if (state.slugsLoadingFor === q) {
          state.slugs = rows;
          setLoading(false);
          render();
        }
        return rows;
      })
      .catch(function () {
        if (state.slugsLoadingFor === q) {
          state.slugs = [];
          setLoading(false);
        }
        return [];
      });
  }

  var debouncedFetchSlugs = debounce(fetchSlugs, SEARCH_DEBOUNCE_MS);

  // ---------- built-in command sources ----------
  function modeCommands() {
    return [
      { id: "mode:regression", kind: "mode", title: "Go to Regression",  sub: "Factor-model fits",       payload: { mode: "regression" } },
      { id: "mode:strategies", kind: "mode", title: "Go to Strategies",  sub: "α Hub · Arb · Crypto",   payload: { mode: "strategies" } },
      { id: "mode:terminal",   kind: "mode", title: "Go to Terminal",    sub: "Bloomberg-style data hub", payload: { mode: "terminal" } },
    ];
  }

  function slashCommands(query) {
    // shown when input starts with "/" OR as discoverability when empty/filter=slash
    var arg = "";
    var head = "";
    if (query && query.charAt(0) === "/") {
      var sp = query.indexOf(" ");
      if (sp > 0) { head = query.slice(1, sp); arg = query.slice(sp + 1).trim(); }
      else { head = query.slice(1); arg = ""; }
    }
    var cmds = [
      {
        id: "slash:jumps:" + (arg || ""),
        kind: "slash",
        title: "/jumps " + (arg || "<slug>"),
        sub: arg ? ("Open jumps panel for " + arg) : "Open jumps panel for a slug",
        payload: { type: "jumps", slug: arg },
        ready: !!arg || head !== "jumps",
      },
      {
        id: "slash:backtest:" + (arg || ""),
        kind: "slash",
        title: "/backtest " + (arg || "<slug>"),
        sub: arg ? ("Run backtest on " + arg) : "Run jumps backtest on a slug",
        payload: { type: "backtest", slug: arg },
        ready: !!arg || head !== "backtest",
      },
      {
        id: "slash:clusters",
        kind: "slash",
        title: "/clusters",
        sub: "Open factor clusters view",
        payload: { type: "clusters" },
        ready: true,
      },
      {
        id: "slash:fit:" + (arg || ""),
        kind: "slash",
        title: "/fit " + (arg || "<ticker> + <factor>"),
        sub: arg ? ("Prefill fit form: " + arg) : "Prefill /fit form (e.g. /fit NVDA + bitcoin)",
        payload: { type: "fit", expr: arg },
        ready: !!arg || head !== "fit",
      },
    ];
    return cmds;
  }

  function parseFitExpr(expr) {
    // Very small parser: TICKER + factor1 + factor2 ...
    if (!expr) return null;
    var parts = expr.split(/\s*[+,]\s*/).map(function (s) { return s.trim(); }).filter(Boolean);
    if (!parts.length) return null;
    var ticker = parts[0].toUpperCase();
    var factors = parts.slice(1);
    return { ticker: ticker, factors: factors };
  }

  // ---------- ranking ----------
  function score(item, query) {
    var fields = [item.title, item.sub, item.id].filter(Boolean);
    var best = null;
    for (var i = 0; i < fields.length; i++) {
      var m = fuzzyMatch(query, fields[i]);
      if (m && (best === null || m.score > best.score)) {
        best = { score: m.score, indices: m.indices, fieldIdx: i };
      }
    }
    return best;
  }

  function buildResults() {
    var q = state.query.trim();
    var filter = state.activeFilter;
    var slashMode = q.charAt(0) === "/";

    var groups = []; // [{ id, label, items: [...] }]

    // Modes
    if (filter === "all" || filter === "mode") {
      var modes = modeCommands();
      var modeItems = q ? topN(rank(modes, q), MAX_RESULTS_PER_CAT) : modes;
      if (modeItems.length) groups.push({ id: "mode", label: "Modes", items: modeItems });
    }

    // Slashes (always shown when slash mode or filter=slash)
    if (slashMode || filter === "slash") {
      var slashes = slashCommands(q).filter(function (c) { return c.ready !== false; });
      // If slash-mode, narrow to the matching head
      if (slashMode) {
        var head = q.slice(1).split(" ")[0];
        if (head) {
          slashes = slashes.filter(function (c) {
            return c.title.toLowerCase().indexOf("/" + head.toLowerCase()) === 0;
          });
        }
      } else if (q) {
        slashes = topN(rank(slashes, q), MAX_RESULTS_PER_CAT);
      }
      if (slashes.length) groups.push({ id: "slash", label: "Slash Commands", items: slashes });
    } else if (filter === "all" && !q && !slashMode) {
      // discoverability: show slash hints in empty-state path (handled by empty-state UI)
    }

    // Factors
    if ((filter === "all" || filter === "factor") && !slashMode) {
      var factors = (state.factors || []).map(function (f) {
        var slug = f.slug || f.id || f.name || "";
        var src = f.source || f.kind || "";
        var label = f.label || f.title || slug;
        return {
          id: "factor:" + slug,
          kind: "factor",
          title: label,
          sub: slug + (src ? "  ·  " + src : ""),
          payload: { slug: slug, source: src, factor: f },
        };
      });
      if (q) factors = topN(rank(factors, q), MAX_RESULTS_PER_CAT);
      else factors = factors.slice(0, MAX_RESULTS_PER_CAT);
      if (factors.length) groups.push({ id: "factor", label: "Factors", items: factors });
    }

    // Slugs (live search)
    if ((filter === "all" || filter === "slug") && !slashMode) {
      var slugs = (state.slugs || []).map(function (s) {
        var slug = s.slug || s.id || s.market_slug || "";
        var qn = s.question || s.title || s.name || slug;
        return {
          id: "slug:" + slug,
          kind: "slug",
          title: qn,
          sub: slug,
          payload: { slug: slug, market: s },
        };
      });
      if (slugs.length) groups.push({ id: "slug", label: "Prediction Markets", items: slugs.slice(0, MAX_RESULTS_PER_CAT) });
    }

    // Registered (external)
    if (state.registered.length && (filter === "all")) {
      var ext = q ? topN(rank(state.registered, q), MAX_RESULTS_PER_CAT) : state.registered.slice(0, MAX_RESULTS_PER_CAT);
      if (ext.length) groups.push({ id: "ext", label: "Actions", items: ext });
    }

    return groups;
  }

  function rank(items, q) {
    var scored = [];
    for (var i = 0; i < items.length; i++) {
      var s = score(items[i], q);
      if (s) scored.push({ item: items[i], score: s.score, indices: s.indices, fieldIdx: s.fieldIdx });
    }
    scored.sort(function (a, b) { return b.score - a.score; });
    return scored.map(function (r) {
      r.item.__hl = { indices: r.indices, fieldIdx: r.fieldIdx };
      return r.item;
    });
  }

  function topN(arr, n) { return arr.slice(0, n); }

  // ---------- DOM ----------
  var root, panel, input, listEl, filterBar, loadingEl, footEl;

  function mount() {
    if (root) return;
    root = document.createElement("div");
    root.className = "pfm-cmdk-root";
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
    root.setAttribute("aria-label", "Command palette");
    root.setAttribute("data-open", "false");
    root.setAttribute("data-loading", "false");

    var backdrop = document.createElement("div");
    backdrop.className = "pfm-cmdk-backdrop";
    backdrop.addEventListener("mousedown", function (e) { if (e.target === backdrop) close(); });
    root.appendChild(backdrop);

    panel = document.createElement("div");
    panel.className = "pfm-cmdk-panel";

    loadingEl = document.createElement("div");
    loadingEl.className = "pfm-cmdk-loading";
    panel.appendChild(loadingEl);

    // Header
    var header = document.createElement("div");
    header.className = "pfm-cmdk-header";
    header.innerHTML =
      '<svg class="pfm-cmdk-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">' +
        '<circle cx="7" cy="7" r="4.5"></circle><path d="M11 11 L14 14"></path></svg>';
    input = document.createElement("input");
    input.className = "pfm-cmdk-input";
    input.type = "text";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.placeholder = "Search anything · type / for slash commands";
    input.setAttribute("aria-label", "Command palette input");
    input.setAttribute("aria-controls", "pfm-cmdk-list");
    input.addEventListener("input", onInput);
    input.addEventListener("keydown", onInputKey);
    header.appendChild(input);
    var kbd = document.createElement("span");
    kbd.className = "pfm-cmdk-kbd";
    kbd.textContent = "Esc";
    header.appendChild(kbd);
    panel.appendChild(header);

    // Filter bar
    filterBar = document.createElement("div");
    filterBar.className = "pfm-cmdk-filters";
    filterBar.setAttribute("role", "tablist");
    for (var i = 0; i < CATEGORIES.length; i++) {
      var c = CATEGORIES[i];
      var b = document.createElement("button");
      b.type = "button";
      b.className = "pfm-cmdk-filter";
      b.setAttribute("data-filter", c.id);
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", c.id === state.activeFilter ? "true" : "false");
      b.setAttribute("data-active", c.id === state.activeFilter ? "true" : "false");
      b.textContent = c.label;
      (function (id) {
        b.addEventListener("click", function () {
          setFilter(id);
          input.focus();
        });
      })(c.id);
      filterBar.appendChild(b);
    }
    panel.appendChild(filterBar);

    // List
    listEl = document.createElement("ul");
    listEl.id = "pfm-cmdk-list";
    listEl.className = "pfm-cmdk-list";
    listEl.setAttribute("role", "listbox");
    listEl.addEventListener("mousedown", onListMouseDown);
    listEl.addEventListener("mouseover", onListMouseOver);
    panel.appendChild(listEl);

    // Footer
    footEl = document.createElement("div");
    footEl.className = "pfm-cmdk-footer";
    footEl.innerHTML =
      '<div class="pfm-cmdk-foot-group">' +
        '<span class="pfm-cmdk-foot-key"><kbd>↑↓</kbd> Navigate</span>' +
        '<span class="pfm-cmdk-foot-key"><kbd>↵</kbd> Select</span>' +
        '<span class="pfm-cmdk-foot-key"><kbd>Tab</kbd> Filter</span>' +
      '</div>' +
      '<div class="pfm-cmdk-foot-group">' +
        '<span class="pfm-cmdk-foot-key"><kbd>' + (isMacLike() ? "⌘K" : "Ctrl K") + '</kbd> Toggle</span>' +
      '</div>';
    panel.appendChild(footEl);

    root.appendChild(panel);
    document.body.appendChild(root);
  }

  function setLoading(b) {
    if (root) root.setAttribute("data-loading", b ? "true" : "false");
  }

  function setFilter(id) {
    state.activeFilter = id;
    state.activeIndex = 0;
    if (!filterBar) return;
    var btns = filterBar.querySelectorAll(".pfm-cmdk-filter");
    for (var i = 0; i < btns.length; i++) {
      var on = btns[i].getAttribute("data-filter") === id;
      btns[i].setAttribute("data-active", on ? "true" : "false");
      btns[i].setAttribute("aria-selected", on ? "true" : "false");
    }
    render();
  }

  function cycleFilter(direction) {
    var cur = 0;
    for (var i = 0; i < CATEGORIES.length; i++) if (CATEGORIES[i].id === state.activeFilter) cur = i;
    var next = (cur + direction + CATEGORIES.length) % CATEGORIES.length;
    setFilter(CATEGORIES[next].id);
  }

  function onInput() {
    state.query = input.value;
    state.activeIndex = 0;
    var q = state.query.trim();

    // factors lazy-load
    if (!state.factors) fetchFactors().then(render);

    // slug live-search
    if (q && q.charAt(0) !== "/") {
      debouncedFetchSlugs(q);
    } else {
      state.slugs = [];
    }
    render();
  }

  function onInputKey(e) {
    if (e.key === "Escape") { e.preventDefault(); close(); return; }
    if (e.key === "ArrowDown") { e.preventDefault(); moveActive(1); return; }
    if (e.key === "ArrowUp")   { e.preventDefault(); moveActive(-1); return; }
    if (e.key === "Enter")     { e.preventDefault(); selectActive(); return; }
    if (e.key === "Tab")       { e.preventDefault(); cycleFilter(e.shiftKey ? -1 : 1); return; }
    if (e.key === "/" && input.value === "") {
      e.preventDefault();
      input.value = "/";
      onInput();
      return;
    }
    // Focus trap: only the input is focusable inside; nothing to do.
  }

  function moveActive(delta) {
    if (!state.results.length) return;
    state.activeIndex = (state.activeIndex + delta + state.results.length) % state.results.length;
    updateActive();
  }

  function selectActive() {
    var item = state.results[state.activeIndex];
    if (!item) return;
    execute(item);
  }

  function onListMouseDown(e) {
    var row = e.target.closest && e.target.closest(".pfm-cmdk-row");
    if (!row) return;
    e.preventDefault(); // prevent input blur before click resolves
    var idx = parseInt(row.getAttribute("data-idx"), 10);
    if (isFinite(idx)) {
      state.activeIndex = idx;
      var item = state.results[idx];
      if (item) execute(item);
    }
  }

  function onListMouseOver(e) {
    var row = e.target.closest && e.target.closest(".pfm-cmdk-row");
    if (!row) return;
    var idx = parseInt(row.getAttribute("data-idx"), 10);
    if (isFinite(idx) && idx !== state.activeIndex) {
      state.activeIndex = idx;
      updateActive();
    }
  }

  function updateActive() {
    if (!listEl) return;
    var rows = listEl.querySelectorAll(".pfm-cmdk-row");
    for (var i = 0; i < rows.length; i++) {
      var on = parseInt(rows[i].getAttribute("data-idx"), 10) === state.activeIndex;
      rows[i].setAttribute("data-active", on ? "true" : "false");
      rows[i].setAttribute("aria-selected", on ? "true" : "false");
      if (on) {
        // scroll into view
        var rRect = rows[i].getBoundingClientRect();
        var lRect = listEl.getBoundingClientRect();
        if (rRect.top < lRect.top) listEl.scrollTop -= (lRect.top - rRect.top) + 4;
        else if (rRect.bottom > lRect.bottom) listEl.scrollTop += (rRect.bottom - lRect.bottom) + 4;
      }
    }
  }

  // ---------- render ----------
  function render() {
    if (!root) return;
    var groups = buildResults();
    var q = state.query.trim();

    // Flatten with optional recents at top when empty
    var flat = [];
    var html = "";

    if (!q) {
      var recents = readRecents();
      if (recents.length) {
        html += '<li class="pfm-cmdk-group-label" role="presentation">Recent</li>';
        for (var ri = 0; ri < Math.min(5, recents.length); ri++) {
          var r = recents[ri];
          var item = {
            id: r.id,
            kind: r.kind || "recent",
            title: r.title || r.id,
            sub: r.sub || "",
            payload: r.payload || {},
            __hl: null,
            __fromRecent: true,
          };
          flat.push(item);
          html += renderRow(item, flat.length - 1);
        }
      }
    }

    for (var gi = 0; gi < groups.length; gi++) {
      var g = groups[gi];
      if (!g.items.length) continue;
      html += '<li class="pfm-cmdk-group-label" role="presentation">' + escapeHtml(g.label) + '</li>';
      for (var ii = 0; ii < g.items.length; ii++) {
        var it = g.items[ii];
        flat.push(it);
        html += renderRow(it, flat.length - 1);
      }
    }

    state.results = flat;

    if (!flat.length) {
      html = renderEmptyState();
    }

    listEl.innerHTML = html;
    if (state.activeIndex >= flat.length) state.activeIndex = 0;
    updateActive();
  }

  function renderRow(item, idx) {
    var hlIdx = (item.__hl && item.__hl.indices) || null;
    var fieldIdx = (item.__hl && item.__hl.fieldIdx) || 0;
    var title = item.title || "";
    var sub = item.sub || "";
    var titleHtml = fieldIdx === 0 ? highlight(title, hlIdx) : escapeHtml(title);
    var subHtml   = fieldIdx === 1 ? highlight(sub,   hlIdx) : escapeHtml(sub);

    var iconChar = ({
      mode: "M",
      slash: "/",
      factor: "F",
      slug: "S",
      recent: "·",
    })[item.kind] || "·";

    var meta = ({
      mode: "MODE",
      slash: "CMD",
      factor: "FACTOR",
      slug: "SLUG",
      recent: "RECENT",
    })[item.kind] || "";

    return (
      '<li class="pfm-cmdk-row" role="option" data-kind="' + escapeHtml(item.kind || "") + '"' +
        ' data-idx="' + idx + '" id="pfm-cmdk-row-' + idx + '"' +
        ' aria-selected="false" data-active="false">' +
        '<span class="pfm-cmdk-row-icon" aria-hidden="true">' + escapeHtml(iconChar) + '</span>' +
        '<span class="pfm-cmdk-row-body">' +
          '<span class="pfm-cmdk-row-title">' + titleHtml + '</span>' +
          (sub ? '<span class="pfm-cmdk-row-sub">' + subHtml + '</span>' : '') +
        '</span>' +
        (meta ? '<span class="pfm-cmdk-row-meta">' + escapeHtml(meta) + '</span>' : '') +
      '</li>'
    );
  }

  function renderEmptyState() {
    var recents = readRecents();
    var html = '<li class="pfm-cmdk-empty" role="presentation">';
    html +=   '<div class="pfm-cmdk-empty-title">Tips</div>';
    html +=   '<div class="pfm-cmdk-empty-tip"><span>Switch mode</span><code>regression</code></div>';
    html +=   '<div class="pfm-cmdk-empty-tip"><span>Open a jumps panel</span><code>/jumps &lt;slug&gt;</code></div>';
    html +=   '<div class="pfm-cmdk-empty-tip"><span>Run a backtest</span><code>/backtest &lt;slug&gt;</code></div>';
    html +=   '<div class="pfm-cmdk-empty-tip"><span>Open clusters</span><code>/clusters</code></div>';
    html +=   '<div class="pfm-cmdk-empty-tip"><span>Prefill fit</span><code>/fit NVDA + bitcoin</code></div>';
    if (recents.length) {
      html += '<div class="pfm-cmdk-empty-title" style="margin-top:10px;">Recent</div>';
      var n = Math.min(5, recents.length);
      for (var i = 0; i < n; i++) {
        var r = recents[i];
        html += '<div class="pfm-cmdk-empty-tip"><span>' + escapeHtml(r.title || r.id) + '</span><code>' + escapeHtml(r.kind || "") + '</code></div>';
      }
    } else {
      html += '<div class="pfm-cmdk-empty-tip" style="opacity:0.7;"><span>No results</span><code>type to search</code></div>';
    }
    html += '</li>';
    return html;
  }

  // ---------- execute ----------
  function execute(item) {
    if (!item) return;
    var kind = item.kind;
    var p = item.payload || {};

    try {
      if (kind === "mode") {
        document.dispatchEvent(new CustomEvent("pfm:switch-mode", { detail: { mode: p.mode } }));
      } else if (kind === "slash") {
        executeSlash(p);
      } else if (kind === "factor") {
        document.dispatchEvent(new CustomEvent("pfm:open-factor", { detail: { slug: p.slug, source: p.source, factor: p.factor } }));
      } else if (kind === "slug") {
        document.dispatchEvent(new CustomEvent("pfm:open-slug", { detail: { slug: p.slug, market: p.market } }));
      } else if (kind === "recent") {
        // best-effort re-execute by inferring kind from id prefix
        var prefix = (item.id || "").split(":")[0];
        execute({ kind: prefix === "factor" ? "factor"
                       : prefix === "slug"  ? "slug"
                       : prefix === "mode"  ? "mode"
                       : prefix === "slash" ? "slash"
                       : "ext", payload: p });
        return; // pushed in nested call
      } else if (item.run && typeof item.run === "function") {
        item.run(p);
      }
    } catch (e) {
      if (window.console && console.warn) console.warn("[cmdk] execute failed", e);
    }

    pushRecent(item);
    close();
  }

  function executeSlash(p) {
    if (!p || !p.type) return;
    if (p.type === "jumps" && p.slug) {
      document.dispatchEvent(new CustomEvent("pfm:open-jumps", { detail: { slug: p.slug } }));
      // fallback: deep-link via hash so consumers without listener can react
      try { window.location.hash = "#terminal/jumps/" + encodeURIComponent(p.slug); } catch (e) { /* noop */ }
    } else if (p.type === "backtest" && p.slug) {
      document.dispatchEvent(new CustomEvent("pfm:open-backtest", { detail: { slug: p.slug } }));
      try { window.location.hash = "#terminal/jumps/" + encodeURIComponent(p.slug) + "/backtest"; } catch (e) { /* noop */ }
    } else if (p.type === "clusters") {
      document.dispatchEvent(new CustomEvent("pfm:open-clusters", { detail: {} }));
      try { window.location.hash = "#terminal/clusters"; } catch (e) { /* noop */ }
    } else if (p.type === "fit") {
      var parsed = parseFitExpr(p.expr || "");
      document.dispatchEvent(new CustomEvent("pfm:switch-mode", { detail: { mode: "regression" } }));
      document.dispatchEvent(new CustomEvent("pfm:prefill-fit", {
        detail: { expr: p.expr || "", ticker: parsed && parsed.ticker, factors: parsed && parsed.factors },
      }));
    }
  }

  // ---------- open/close ----------
  function open() {
    mount();
    if (state.open) return;
    state.open = true;
    root.setAttribute("data-open", "true");
    // initial loads
    if (!state.factors) fetchFactors().then(render);
    render();
    // focus next tick so transition can begin
    setTimeout(function () { try { input.focus(); input.select(); } catch (e) { /* noop */ } }, 0);
    document.addEventListener("keydown", onDocKey, true);
    document.dispatchEvent(new CustomEvent("pfm:cmdk-open"));
  }

  function close() {
    if (!state.open) return;
    state.open = false;
    if (root) root.setAttribute("data-open", "false");
    document.removeEventListener("keydown", onDocKey, true);
    document.dispatchEvent(new CustomEvent("pfm:cmdk-close"));
  }

  function toggle() { state.open ? close() : open(); }

  function onDocKey(e) {
    if (!state.open) return;
    if (e.key === "Escape") { e.preventDefault(); close(); return; }
    // focus trap — keep focus inside input
    if (e.key === "Tab") {
      // input handles Tab for filter cycling; if focus drifts, pull it back
      if (document.activeElement !== input) {
        e.preventDefault();
        try { input.focus(); } catch (err) { /* noop */ }
      }
      return;
    }
  }

  // ---------- global ⌘K/Ctrl+K shortcut ----------
  function onGlobalKey(e) {
    if (!e) return;
    var mod = e.metaKey || e.ctrlKey;
    if (mod && (e.key === "k" || e.key === "K")) {
      // skip when user is typing in a contenteditable wholly unrelated... actually
      // we want it to work everywhere except inside the palette itself (which has
      // its own handler that wins via input.keydown). So no exclusions here.
      e.preventDefault();
      toggle();
    }
  }

  // ---------- public API ----------
  var api = {
    open: open,
    close: close,
    toggle: toggle,
    register: function (commands) {
      if (!commands) return;
      var arr = Array.isArray(commands) ? commands : [commands];
      for (var i = 0; i < arr.length; i++) {
        var c = arr[i];
        if (!c || !c.id || !c.title) continue;
        var entry = {
          id: String(c.id),
          kind: c.kind || "ext",
          title: String(c.title),
          sub: c.sub ? String(c.sub) : "",
          payload: c.payload || {},
          run: typeof c.run === "function" ? c.run : null,
        };
        // dedupe by id
        var existing = -1;
        for (var j = 0; j < state.registered.length; j++) {
          if (state.registered[j].id === entry.id) { existing = j; break; }
        }
        if (existing >= 0) state.registered[existing] = entry;
        else state.registered.push(entry);
      }
      if (state.open) render();
    },
    _state: state, // for debugging
  };

  window.PFM = window.PFM || {};
  window.PFM.cmdk = api;

  // ---------- boot ----------
  function boot() {
    document.addEventListener("keydown", onGlobalKey, false);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
