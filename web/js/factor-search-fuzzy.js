/* W11-09 (T74) — Factor Fuzzy Search
 *
 * Vanilla JS, no deps (no fuse.js). Self-contained ~100-line fuzzy matcher.
 *
 * Public API:
 *   window.PFM.factorSearch = {
 *     setIndex(factors),                  // factors: [{slug, label, theme, description}]
 *     query(q, opts) -> Result[],
 *     attachInput(inputEl, onSelect)      // wires up an existing search input
 *   }
 *
 * Scoring formula:
 *   subseq = chars of q matched in order against haystack (case-insensitive)
 *   base   = subseq_matches / len(query)             in [0, 1]
 *   token_prefix_bonus = 1 + 0.6 if any whitespace/-/_ token starts with q
 *                       else 1 + 0.3 if any token contains q as substring
 *                       else 1.0
 *   theme_match_bonus  = 1 + 0.25 if theme starts with q (case-insensitive)
 *                       else 1 + 0.10 if theme contains q
 *                       else 1.0
 *   score = base * token_prefix_bonus * theme_match_bonus
 *
 *   Per factor we score slug, label, theme, description independently and
 *   take the max score, remembering which field hit and the matched indices
 *   for highlighting.
 *
 * Behavior:
 *   - Top 20 results, score > 0
 *   - Fetch /factors once, cache 5 min in memory
 *   - attachInput: 100ms debounce, dropdown below input, arrow keys / Enter / Esc
 *
 * Mount: <script defer src="/js/factor-search-fuzzy.js"></script>
 *        (index-html-owner mounts; call attachInput on an existing <input>.)
 */
(function () {
  "use strict";

  var API_BASE =
    (window.PFM && window.PFM.apiBase) ||
    window.PFM_API_BASE ||
    (window.location && window.location.port === "8080" ? "http://127.0.0.1:8000" : "");

  var CACHE_TTL_MS = 5 * 60 * 1000;
  var DEBOUNCE_MS = 100;
  var MAX_RESULTS = 20;

  // ---------- state ----------
  var state = {
    factors: [],     // raw factor objects
    indexed: [],     // [{factor, fields:[{name,text,lower}]}]
    loadedAt: 0,
    fetchPromise: null,
  };

  // ---------- utils ----------
  function escapeHtml(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function lower(s) { return (s == null ? "" : String(s)).toLowerCase(); }

  // ---------- core fuzzy ----------
  // Returns {matches: number, indices: number[]} for the longest in-order
  // subsequence of `q` chars in `hay`. Both args are already lowercased.
  function subseqMatch(q, hay) {
    if (!q) return { matches: 0, indices: [] };
    if (!hay) return { matches: 0, indices: [] };
    var indices = [];
    var qi = 0;
    for (var i = 0; i < hay.length && qi < q.length; i++) {
      if (hay.charCodeAt(i) === q.charCodeAt(qi)) {
        indices.push(i);
        qi++;
      }
    }
    return { matches: qi, indices: indices };
  }

  function tokenPrefixBonus(q, hayLower) {
    // tokens split on whitespace / - / _ / : / .
    var toks = hayLower.split(/[\s\-_:.]+/).filter(Boolean);
    var hasPrefix = false;
    var hasSubstr = false;
    for (var i = 0; i < toks.length; i++) {
      if (toks[i].indexOf(q) === 0) { hasPrefix = true; break; }
      if (!hasSubstr && toks[i].indexOf(q) >= 0) hasSubstr = true;
    }
    if (hasPrefix) return 1.6;
    if (hasSubstr) return 1.3;
    return 1.0;
  }

  function themeBonus(q, themeLower) {
    if (!themeLower) return 1.0;
    if (themeLower.indexOf(q) === 0) return 1.25;
    if (themeLower.indexOf(q) >= 0) return 1.10;
    return 1.0;
  }

  // Score one factor against query. Returns {score, field, indices, text}.
  function scoreFactor(entry, q) {
    var qLower = q;
    var themeLower = lower(entry.factor.theme);
    var tBonus = themeBonus(qLower, themeLower);
    var best = { score: 0, field: null, indices: [], text: "" };
    for (var i = 0; i < entry.fields.length; i++) {
      var f = entry.fields[i];
      if (!f.lower) continue;
      var sub = subseqMatch(qLower, f.lower);
      if (sub.matches < qLower.length) continue; // require full subsequence
      var base = sub.matches / qLower.length; // == 1 when full match, kept for clarity
      var tok = tokenPrefixBonus(qLower, f.lower);
      var score = base * tok * tBonus;
      // field weight: slug > label > theme > description
      var fieldWeight = f.name === "slug" ? 1.10
                      : f.name === "label" ? 1.05
                      : f.name === "theme" ? 1.00
                      : 0.90;
      score *= fieldWeight;
      if (score > best.score) {
        best = { score: score, field: f.name, indices: sub.indices, text: f.text };
      }
    }
    return best;
  }

  function buildIndex(factors) {
    state.factors = Array.isArray(factors) ? factors.slice() : [];
    state.indexed = state.factors.map(function (f) {
      var fields = [
        { name: "slug",        text: f.slug || "",        lower: lower(f.slug) },
        { name: "label",       text: f.label || "",       lower: lower(f.label) },
        { name: "theme",       text: f.theme || "",       lower: lower(f.theme) },
        { name: "description", text: f.description || "", lower: lower(f.description) },
      ];
      return { factor: f, fields: fields };
    });
  }

  function query(q, opts) {
    opts = opts || {};
    var limit = opts.limit || MAX_RESULTS;
    var qLower = lower(q).trim();
    if (!qLower) return [];
    var out = [];
    for (var i = 0; i < state.indexed.length; i++) {
      var best = scoreFactor(state.indexed[i], qLower);
      if (best.score > 0) {
        out.push({
          factor: state.indexed[i].factor,
          score: best.score,
          field: best.field,
          indices: best.indices,
          text: best.text,
        });
      }
    }
    out.sort(function (a, b) { return b.score - a.score; });
    return out.slice(0, limit);
  }

  // Highlight matched indices in `text`. Indices are positions in the
  // lowercased version, which align with the original because we don't
  // change length when lowercasing in BMP.
  function highlight(text, indices) {
    if (!text) return "";
    if (!indices || !indices.length) return escapeHtml(text);
    var set = {};
    for (var i = 0; i < indices.length; i++) set[indices[i]] = true;
    var html = "";
    var inMark = false;
    for (var j = 0; j < text.length; j++) {
      var ch = text.charAt(j);
      if (set[j]) {
        if (!inMark) { html += '<mark class="fzy-mark">'; inMark = true; }
      } else if (inMark) {
        html += "</mark>";
        inMark = false;
      }
      html += escapeHtml(ch);
    }
    if (inMark) html += "</mark>";
    return html;
  }

  // ---------- data load ----------
  function loadFactors() {
    var now = Date.now();
    if (state.indexed.length && (now - state.loadedAt) < CACHE_TTL_MS) {
      return Promise.resolve(state.indexed);
    }
    if (state.fetchPromise) return state.fetchPromise;
    var url = API_BASE + "/factors";
    state.fetchPromise = fetch(url, { headers: { "Accept": "application/json" } })
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (body) {
        // /factors may return {factors:[...]}, [...], or a map.
        var arr = [];
        if (Array.isArray(body)) arr = body;
        else if (body && Array.isArray(body.factors)) arr = body.factors;
        else if (body && typeof body === "object") {
          arr = Object.keys(body).map(function (k) {
            var v = body[k] || {};
            return {
              slug: v.slug || k,
              label: v.label || v.name || k,
              theme: v.theme || v.category || "",
              description: v.description || v.english_description || "",
            };
          });
        }
        // Normalize keys: tolerate english_description, name, category.
        arr = arr.map(function (f) {
          return {
            slug: f.slug || f.id || "",
            label: f.label || f.name || f.title || f.slug || "",
            theme: f.theme || f.category || f.group || "",
            description: f.description || f.english_description || f.summary || "",
          };
        }).filter(function (f) { return f.slug || f.label; });
        buildIndex(arr);
        state.loadedAt = Date.now();
        state.fetchPromise = null;
        return state.indexed;
      })
      .catch(function () {
        state.fetchPromise = null;
        return state.indexed;
      });
    return state.fetchPromise;
  }

  function setIndex(factors) {
    buildIndex(factors);
    state.loadedAt = Date.now();
  }

  // ---------- UI: attachInput ----------
  function attachInput(inputEl, onSelect) {
    if (!inputEl || inputEl.__pfmFuzzyAttached) return;
    inputEl.__pfmFuzzyAttached = true;
    inputEl.setAttribute("autocomplete", "off");
    inputEl.setAttribute("spellcheck", "false");

    var dropdown = document.createElement("div");
    dropdown.className = "fzy-dropdown";
    dropdown.setAttribute("role", "listbox");
    dropdown.style.cssText =
      "position:absolute;z-index:9999;background:#0d1117;color:#e6edf3;" +
      "border:1px solid #30363d;border-radius:6px;max-height:360px;" +
      "overflow-y:auto;display:none;font-family:inherit;font-size:13px;" +
      "box-shadow:0 8px 24px rgba(0,0,0,0.5);min-width:280px;";
    document.body.appendChild(dropdown);

    var currentResults = [];
    var activeIdx = -1;
    var debounceTimer = null;

    function position() {
      var r = inputEl.getBoundingClientRect();
      dropdown.style.left = (window.scrollX + r.left) + "px";
      dropdown.style.top = (window.scrollY + r.bottom + 4) + "px";
      dropdown.style.minWidth = r.width + "px";
    }

    function close() {
      dropdown.style.display = "none";
      activeIdx = -1;
    }

    function render() {
      if (!currentResults.length) { close(); return; }
      position();
      var html = "";
      for (var i = 0; i < currentResults.length; i++) {
        var r = currentResults[i];
        var f = r.factor;
        var isActive = i === activeIdx;
        var hl = r.field && r.text
          ? highlight(r.text, r.indices)
          : escapeHtml(f.label || f.slug);
        // Show slug as primary if match was on slug, else label first.
        var primary = r.field === "slug" ? hl : escapeHtml(f.label || f.slug);
        var secondary = r.field === "slug"
          ? escapeHtml(f.label || "")
          : (r.field === "label" ? "" : hl);
        var meta = escapeHtml(f.theme || "");
        html +=
          '<div class="fzy-item' + (isActive ? " fzy-item--active" : "") + '" ' +
          'role="option" data-idx="' + i + '" ' +
          'style="padding:8px 12px;cursor:pointer;border-bottom:1px solid #21262d;' +
          (isActive ? "background:#1f6feb33;" : "") + '">' +
          '<div class="fzy-primary" style="font-weight:600;color:#58a6ff;">' + primary + '</div>' +
          (secondary ? '<div class="fzy-secondary" style="opacity:0.85;margin-top:2px;">' + secondary + '</div>' : '') +
          (meta ? '<div class="fzy-meta" style="opacity:0.6;font-size:11px;margin-top:2px;">' + meta + '</div>' : '') +
          '</div>';
      }
      dropdown.innerHTML = html;
      dropdown.style.display = "block";
    }

    function runQuery(q) {
      loadFactors().then(function () {
        currentResults = query(q, { limit: MAX_RESULTS });
        activeIdx = currentResults.length ? 0 : -1;
        render();
      });
    }

    function onInput() {
      var v = inputEl.value || "";
      if (debounceTimer) clearTimeout(debounceTimer);
      if (!v.trim()) { currentResults = []; close(); return; }
      debounceTimer = setTimeout(function () { runQuery(v); }, DEBOUNCE_MS);
    }

    function commit(i) {
      var r = currentResults[i];
      if (!r) return;
      close();
      if (typeof onSelect === "function") {
        try { onSelect(r.factor, r); } catch (e) { /* swallow */ }
      }
    }

    inputEl.addEventListener("input", onInput);
    inputEl.addEventListener("focus", function () {
      if (currentResults.length) { render(); }
    });
    inputEl.addEventListener("keydown", function (e) {
      if (dropdown.style.display === "none") return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        activeIdx = Math.min(currentResults.length - 1, activeIdx + 1);
        render();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        activeIdx = Math.max(0, activeIdx - 1);
        render();
      } else if (e.key === "Enter") {
        if (activeIdx >= 0) { e.preventDefault(); commit(activeIdx); }
      } else if (e.key === "Escape") {
        e.preventDefault();
        close();
      }
    });

    dropdown.addEventListener("mousedown", function (e) {
      // mousedown (not click) so blur doesn't fire first and close us
      var el = e.target.closest ? e.target.closest("[data-idx]") : null;
      if (!el) return;
      e.preventDefault();
      var idx = parseInt(el.getAttribute("data-idx"), 10);
      if (!isNaN(idx)) commit(idx);
    });

    document.addEventListener("click", function (e) {
      if (e.target === inputEl) return;
      if (dropdown.contains(e.target)) return;
      close();
    });

    window.addEventListener("resize", function () {
      if (dropdown.style.display !== "none") position();
    });
    window.addEventListener("scroll", function () {
      if (dropdown.style.display !== "none") position();
    }, true);

    // Pre-warm the cache so the first keystroke is instant.
    loadFactors();
  }

  // ---------- expose ----------
  window.PFM = window.PFM || {};
  window.PFM.factorSearch = {
    setIndex: setIndex,
    query: query,
    attachInput: attachInput,
    // exposed for tests / debugging:
    _state: state,
    _scoreFactor: scoreFactor,
    _subseqMatch: subseqMatch,
    _highlight: highlight,
  };
})();
