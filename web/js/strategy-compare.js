/* W11-59 — Strategy Compare (side-by-side)
 *
 * Vanilla JS, no deps (Plotly used opportunistically if window.Plotly is loaded).
 * Self-mounts on DOMContentLoaded. Builds on GET /alpha-hub/strategy/{pair_id}.
 *
 * Public API:
 *   window.PFM.strategyCompare = {
 *     open(pairIdA, pairIdB),
 *     close(),
 *     isOpen()
 *   }
 *
 * UI:
 *   Modal (T12 .modal--xl). 2-col grid 1fr 1fr (A | B) with vertical hairline
 *   divider. 4 sections per strategy: Setup · Risk · Theory · Live.
 *   Diff chip on each numeric row when |Δ%| > 10% (orange, mono 11px, ▲/▼).
 *   Bottom: side-by-side equity-curve plot (Plotly subplot).
 *
 * cmdk integration:
 *   Registers `/compare <id1> <id2>` slash command with window.PFM.cmdk if present.
 *
 * Mount:
 *   <link rel="stylesheet" href="/css/strategy-compare.css">
 *   <script defer src="/js/strategy-compare.js"></script>
 *   (index-html-owner adds these; the script self-mounts the root DOM.)
 */
(function () {
  "use strict";

  // ---------- config ----------
  var API_BASE =
    (window.PFM && window.PFM.apiBase) ||
    (window.PFM_API_BASE) ||
    (window.location && window.location.port === "8080" ? "http://127.0.0.1:8000" : "");

  var DIFF_THRESHOLD = 0.10; // 10% relative delta to show a diff chip

  // ---------- state ----------
  var state = {
    open: false,
    pairA: null,
    pairB: null,
    dataA: null,
    dataB: null,
    error: null,
    loading: false,
    mobileExpanded: { setup: true, risk: false, theory: false, live: false },
  };

  // ---------- DOM refs ----------
  var root, backdrop, modalEl, bodyEl, titleEl, closeBtn;

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

  function fmtNum(v, opts) {
    if (v == null || v === "" || (typeof v === "number" && !isFinite(v))) return "—";
    var o = opts || {};
    var n = Number(v);
    if (!isFinite(n)) return escapeHtml(String(v));
    if (o.pct) return (n * 100).toFixed(o.digits != null ? o.digits : 1) + "%";
    if (o.usd) {
      if (Math.abs(n) >= 1e6) return "$" + (n / 1e6).toFixed(2) + "M";
      if (Math.abs(n) >= 1e3) return "$" + (n / 1e3).toFixed(1) + "k";
      return "$" + n.toFixed(0);
    }
    if (o.signed) return (n >= 0 ? "+" : "") + n.toFixed(o.digits != null ? o.digits : 2);
    return n.toFixed(o.digits != null ? o.digits : 2);
  }

  function diffChip(a, b) {
    // Returns HTML for a diff chip, or "" if no meaningful delta.
    if (a == null || b == null) return "";
    var na = Number(a), nb = Number(b);
    if (!isFinite(na) || !isFinite(nb)) return "";
    if (na === nb) return "";
    var base = Math.max(Math.abs(na), Math.abs(nb));
    if (base === 0) return "";
    var rel = Math.abs(na - nb) / base;
    if (rel < DIFF_THRESHOLD) return "";
    var d = nb - na;
    var arrow = d > 0 ? "▲" : "▼";
    var abs = Math.abs(d);
    var disp;
    if (abs >= 1000) disp = (abs / 1000).toFixed(1) + "k";
    else if (abs >= 10) disp = abs.toFixed(0);
    else disp = abs.toFixed(2);
    return (
      '<span class="sc-chip" data-dir="' + (d > 0 ? "up" : "down") + '"' +
      ' title="Δ ' + (d > 0 ? "+" : "") + d.toFixed(3) + ' (' + (rel * 100).toFixed(0) + '% rel)">' +
        '<span class="sc-chip-arrow">' + arrow + '</span>' +
        '<span class="sc-chip-val">' + escapeHtml(disp) + '</span>' +
      '</span>'
    );
  }

  function textDiffChip(a, b) {
    if (!a && !b) return "";
    if (a === b) return "";
    if (!a || !b) return '<span class="sc-chip sc-chip--text" data-dir="text" title="One side missing">≠</span>';
    return '<span class="sc-chip sc-chip--text" data-dir="text" title="Different references">≠</span>';
  }

  function tierRank(t) {
    // Lower index = higher tier.
    var order = ["A_GOLD", "A", "B_VALIDATED", "B", "C", "D"];
    if (!t) return 99;
    var u = String(t).toUpperCase();
    var i = order.indexOf(u);
    return i === -1 ? 50 : i;
  }

  function tierDiff(a, b) {
    if (!a && !b) return "";
    if (a === b) return "";
    var ra = tierRank(a), rb = tierRank(b);
    if (ra === rb) return "";
    var dir = rb < ra ? "up" : "down";
    var arrow = dir === "up" ? "▲" : "▼";
    return (
      '<span class="sc-chip" data-dir="' + dir + '"' +
      ' title="Tier delta ' + escapeHtml(String(a || "?")) + ' → ' + escapeHtml(String(b || "?")) + '">' +
        '<span class="sc-chip-arrow">' + arrow + '</span>' +
        '<span class="sc-chip-val">tier</span>' +
      '</span>'
    );
  }

  // ---------- fetching ----------
  function fetchStrategy(pairId) {
    var url = API_BASE + "/alpha-hub/strategy/" + encodeURIComponent(pairId);
    return fetch(url, { credentials: "omit" }).then(function (r) {
      if (!r.ok) {
        return r.text().then(function (t) {
          throw new Error("HTTP " + r.status + " " + (t || "").slice(0, 200));
        });
      }
      return r.json();
    });
  }

  // ---------- field accessors ----------
  function pickRule(s) {
    if (!s) return {};
    var r = s.rule || {};
    return {
      window: r.window != null ? r.window : s.rule_window,
      entry_z: r.entry_z != null ? r.entry_z : s.rule_entry_z,
      exit_z: r.exit_z != null ? r.exit_z : s.rule_exit_z,
      stop_z: r.stop_z != null ? r.stop_z : s.rule_stop_z,
    };
  }

  function pickRisk(s) {
    if (!s) return {};
    var r = s.risk || {};
    return {
      grade: r.grade != null ? r.grade : s.risk_grade,
      max_dd: r.max_dd != null ? r.max_dd : (s.max_dd != null ? s.max_dd : s.worst_drawdown_observed),
      best: r.best_conditions != null ? r.best_conditions : s.best_market_conditions,
      worst: r.worst_conditions != null ? r.worst_conditions : s.worst_market_conditions,
    };
  }

  function pickDeploy(s) {
    if (!s) return {};
    var d = s.deployment || {};
    return {
      min_capital: d.min_capital_usd != null ? d.min_capital_usd : s.min_capital_usd,
      holding_days: d.expected_holding_days != null ? d.expected_holding_days : s.expected_holding_days,
      trades_per_year: d.expected_trades_per_year != null ? d.expected_trades_per_year : s.expected_trades_per_year,
      monitoring: d.monitoring_frequency != null ? d.monitoring_frequency : s.monitoring_frequency,
      capacity_usd: s.capacity_usd != null ? s.capacity_usd : s.capacity,
    };
  }

  function pickRecentSignal(s) {
    if (!s) return null;
    return s.recent_signal || s.live_signal || null;
  }

  // ---------- rendering ----------
  function renderHeader(a, b) {
    var na = (a && (a.a_name || a.name || a.title || a.pair_id)) || state.pairA;
    var nb = (b && (b.a_name || b.name || b.title || b.pair_id)) || state.pairB;
    return (
      '<div class="sc-pair-titles">' +
        '<div class="sc-pair-title sc-side sc-side-a">' +
          '<span class="sc-pair-letter">A</span>' +
          '<span class="sc-pair-name">' + escapeHtml(na || "—") + '</span>' +
        '</div>' +
        '<div class="sc-pair-title sc-side sc-side-b">' +
          '<span class="sc-pair-letter">B</span>' +
          '<span class="sc-pair-name">' + escapeHtml(nb || "—") + '</span>' +
        '</div>' +
      '</div>'
    );
  }

  function row(label, valA, valB, chip) {
    return (
      '<div class="sc-row">' +
        '<div class="sc-row-label">' + escapeHtml(label) + '</div>' +
        '<div class="sc-row-val sc-side-a">' + valA + '</div>' +
        '<div class="sc-row-val sc-side-b">' +
          '<span class="sc-row-val-inner">' + valB + '</span>' +
          (chip || "") +
        '</div>' +
      '</div>'
    );
  }

  function section(id, label, html) {
    var open = state.mobileExpanded[id] !== false;
    return (
      '<section class="sc-section" data-section="' + id + '" data-open="' + (open ? "true" : "false") + '">' +
        '<button type="button" class="sc-section-header" data-section-toggle="' + id + '" aria-expanded="' + (open ? "true" : "false") + '">' +
          '<span class="sc-section-label">' + escapeHtml(label) + '</span>' +
          '<span class="sc-section-caret" aria-hidden="true">▾</span>' +
        '</button>' +
        '<div class="sc-section-body">' + html + '</div>' +
      '</section>'
    );
  }

  function renderSetup(a, b) {
    var ra = pickRule(a), rb = pickRule(b);
    var da = pickDeploy(a), db = pickDeploy(b);
    var rows = "";
    rows += row("Tier",
      '<span class="sc-tier" data-tier="' + escapeHtml(String((a && a.tier) || "")) + '">' + escapeHtml((a && a.tier) || "—") + '</span>',
      '<span class="sc-tier" data-tier="' + escapeHtml(String((b && b.tier) || "")) + '">' + escapeHtml((b && b.tier) || "—") + '</span>',
      tierDiff(a && a.tier, b && b.tier)
    );
    rows += row("Theme",
      escapeHtml((a && (a.theme || a.category)) || "—"),
      escapeHtml((b && (b.theme || b.category)) || "—"),
      ""
    );
    rows += row("OOS Sharpe",
      fmtNum(a && a.oos_sharpe, { digits: 2 }),
      fmtNum(b && b.oos_sharpe, { digits: 2 }),
      diffChip(a && a.oos_sharpe, b && b.oos_sharpe)
    );
    rows += row("Full Sharpe",
      fmtNum(a && a.full_sharpe, { digits: 2 }),
      fmtNum(b && b.full_sharpe, { digits: 2 }),
      diffChip(a && a.full_sharpe, b && b.full_sharpe)
    );
    rows += row("Capacity",
      fmtNum(da.capacity_usd, { usd: true }),
      fmtNum(db.capacity_usd, { usd: true }),
      diffChip(da.capacity_usd, db.capacity_usd)
    );
    rows += row("Entry z",
      fmtNum(ra.entry_z, { digits: 2 }),
      fmtNum(rb.entry_z, { digits: 2 }),
      diffChip(ra.entry_z, rb.entry_z)
    );
    rows += row("Exit z",
      fmtNum(ra.exit_z, { digits: 2 }),
      fmtNum(rb.exit_z, { digits: 2 }),
      diffChip(ra.exit_z, rb.exit_z)
    );
    rows += row("Stop z",
      fmtNum(ra.stop_z, { digits: 2 }),
      fmtNum(rb.stop_z, { digits: 2 }),
      diffChip(ra.stop_z, rb.stop_z)
    );
    rows += row("Window (d)",
      fmtNum(ra.window, { digits: 0 }),
      fmtNum(rb.window, { digits: 0 }),
      diffChip(ra.window, rb.window)
    );
    rows += row("Min capital",
      fmtNum(da.min_capital, { usd: true }),
      fmtNum(db.min_capital, { usd: true }),
      diffChip(da.min_capital, db.min_capital)
    );
    rows += row("Holding (d)",
      fmtNum(da.holding_days, { digits: 0 }),
      fmtNum(db.holding_days, { digits: 0 }),
      diffChip(da.holding_days, db.holding_days)
    );
    rows += row("Trades / yr",
      fmtNum(da.trades_per_year, { digits: 0 }),
      fmtNum(db.trades_per_year, { digits: 0 }),
      diffChip(da.trades_per_year, db.trades_per_year)
    );
    return section("setup", "Setup", rows);
  }

  function renderRisk(a, b) {
    var ra = pickRisk(a), rb = pickRisk(b);
    var rows = "";
    rows += row("Risk grade",
      escapeHtml(ra.grade || "—"),
      escapeHtml(rb.grade || "—"),
      ra.grade === rb.grade ? "" : textDiffChip(ra.grade, rb.grade)
    );
    rows += row("Max DD",
      fmtNum(ra.max_dd, { pct: true, digits: 1 }),
      fmtNum(rb.max_dd, { pct: true, digits: 1 }),
      diffChip(ra.max_dd, rb.max_dd)
    );
    rows += row("Worst-Q Sharpe",
      fmtNum(a && a.worst_quarter_sharpe, { digits: 2 }),
      fmtNum(b && b.worst_quarter_sharpe, { digits: 2 }),
      diffChip(a && a.worst_quarter_sharpe, b && b.worst_quarter_sharpe)
    );
    rows += row("Sharpe CI lo",
      fmtNum(a && a.sharpe_ci_lo, { digits: 2 }),
      fmtNum(b && b.sharpe_ci_lo, { digits: 2 }),
      diffChip(a && a.sharpe_ci_lo, b && b.sharpe_ci_lo)
    );
    rows += row("Best regime",
      '<span class="sc-prose">' + escapeHtml(ra.best || "—") + '</span>',
      '<span class="sc-prose">' + escapeHtml(rb.best || "—") + '</span>',
      ""
    );
    rows += row("Worst regime",
      '<span class="sc-prose">' + escapeHtml(ra.worst || "—") + '</span>',
      '<span class="sc-prose">' + escapeHtml(rb.worst || "—") + '</span>',
      ""
    );
    return section("risk", "Risk", rows);
  }

  function renderTheory(a, b) {
    var ta = (a && (a.theory_reference || a.theory_ref)) || "";
    var tb = (b && (b.theory_reference || b.theory_ref)) || "";
    var rationaleA = (a && a.rationale) || "";
    var rationaleB = (b && b.rationale) || "";
    var rows = "";
    rows += row("Theory ref",
      '<span class="sc-prose">' + escapeHtml(ta || "—") + '</span>',
      '<span class="sc-prose">' + escapeHtml(tb || "—") + '</span>',
      textDiffChip(ta, tb)
    );
    rows += row("Rationale",
      '<span class="sc-prose">' + escapeHtml(rationaleA || "—") + '</span>',
      '<span class="sc-prose">' + escapeHtml(rationaleB || "—") + '</span>',
      textDiffChip(rationaleA, rationaleB)
    );
    var corrA = (a && a.correlated_with) || (a && a.correlated_with_strategies) || [];
    var corrB = (b && b.correlated_with) || (b && b.correlated_with_strategies) || [];
    rows += row("Correlated with",
      '<span class="sc-prose">' + escapeHtml(Array.isArray(corrA) ? corrA.join(", ") : (corrA || "—")) + '</span>',
      '<span class="sc-prose">' + escapeHtml(Array.isArray(corrB) ? corrB.join(", ") : (corrB || "—")) + '</span>',
      ""
    );
    return section("theory", "Theory", rows);
  }

  function renderLive(a, b) {
    var sa = pickRecentSignal(a), sb = pickRecentSignal(b);
    var rows = "";
    var zA = sa && (sa.z != null ? sa.z : sa.zscore);
    var zB = sb && (sb.z != null ? sb.z : sb.zscore);
    var stA = (sa && (sa.state || sa.status || sa.signal)) || "";
    var stB = (sb && (sb.state || sb.status || sb.signal)) || "";
    var asOfA = (sa && (sa.as_of || sa.timestamp || sa.observed_at)) || (a && a.updated_at) || "";
    var asOfB = (sb && (sb.as_of || sb.timestamp || sb.observed_at)) || (b && b.updated_at) || "";

    rows += row("Live state",
      '<span class="sc-pill" data-state="' + escapeHtml(String(stA).toLowerCase()) + '">' + escapeHtml(stA || "—") + '</span>',
      '<span class="sc-pill" data-state="' + escapeHtml(String(stB).toLowerCase()) + '">' + escapeHtml(stB || "—") + '</span>',
      textDiffChip(stA, stB)
    );
    rows += row("Current z",
      fmtNum(zA, { digits: 2, signed: true }),
      fmtNum(zB, { digits: 2, signed: true }),
      diffChip(zA, zB)
    );
    rows += row("As of",
      escapeHtml(asOfA || "—"),
      escapeHtml(asOfB || "—"),
      ""
    );
    return section("live", "Live", rows);
  }

  function renderError(err) {
    return (
      '<div class="sc-error">' +
        '<div class="sc-error-title">Could not load comparison</div>' +
        '<div class="sc-error-msg">' + escapeHtml(String(err && err.message || err)) + '</div>' +
        '<button type="button" class="sc-error-retry" data-action="retry">Retry</button>' +
      '</div>'
    );
  }

  function renderLoading() {
    return (
      '<div class="sc-loading">' +
        '<div class="sc-loading-spinner" aria-hidden="true"></div>' +
        '<div class="sc-loading-text">Loading strategies ' +
          escapeHtml(state.pairA || "?") + ' &amp; ' + escapeHtml(state.pairB || "?") +
        '…</div>' +
      '</div>'
    );
  }

  function renderBody() {
    if (state.loading) return renderLoading();
    if (state.error) return renderError(state.error);
    var a = state.dataA, b = state.dataB;
    var hdr = renderHeader(a, b);
    var sections =
      renderSetup(a, b) +
      renderRisk(a, b) +
      renderTheory(a, b) +
      renderLive(a, b);
    var equity =
      '<div class="sc-equity" data-section="equity">' +
        '<div class="sc-equity-header">Side-by-side equity curves</div>' +
        '<div id="sc-equity-plot" class="sc-equity-plot"></div>' +
      '</div>';
    return (
      '<div class="sc-grid">' +
        hdr +
        sections +
      '</div>' +
      equity
    );
  }

  // ---------- equity plot ----------
  function eqXY(curve) {
    if (!Array.isArray(curve) || !curve.length) return { x: [], y: [] };
    var xs = [], ys = [];
    for (var i = 0; i < curve.length; i++) {
      var p = curve[i];
      if (p == null) continue;
      if (Array.isArray(p)) {
        xs.push(p[0]); ys.push(Number(p[1]));
      } else if (typeof p === "object") {
        xs.push(p.date || p.t || p.x || i);
        ys.push(Number(p.value != null ? p.value : (p.equity != null ? p.equity : p.y)));
      } else {
        xs.push(i); ys.push(Number(p));
      }
    }
    return { x: xs, y: ys };
  }

  function plotEquity() {
    var el = document.getElementById("sc-equity-plot");
    if (!el) return;
    if (!window.Plotly) {
      el.innerHTML = '<div class="sc-equity-fallback">Plotly not loaded — equity curves unavailable.</div>';
      return;
    }
    var a = (state.dataA && state.dataA.equity_curve) || [];
    var b = (state.dataB && state.dataB.equity_curve) || [];
    var pa = eqXY(a);
    var pb = eqXY(b);
    var traces = [
      {
        type: "scatter", mode: "lines",
        name: (state.dataA && (state.dataA.a_name || state.dataA.pair_id)) || state.pairA || "A",
        x: pa.x, y: pa.y,
        line: { width: 2, color: "var(--sc-side-a-color, #f97316)" },
        xaxis: "x", yaxis: "y",
        hovertemplate: "<b>A</b> %{x}<br>%{y:.2f}<extra></extra>",
      },
      {
        type: "scatter", mode: "lines",
        name: (state.dataB && (state.dataB.a_name || state.dataB.pair_id)) || state.pairB || "B",
        x: pb.x, y: pb.y,
        line: { width: 2, color: "var(--sc-side-b-color, #0ea5e9)" },
        xaxis: "x2", yaxis: "y2",
        hovertemplate: "<b>B</b> %{x}<br>%{y:.2f}<extra></extra>",
      },
    ];
    var layout = {
      grid: { rows: 1, columns: 2, pattern: "independent" },
      margin: { l: 40, r: 16, t: 28, b: 36 },
      height: 260,
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { family: "system-ui, -apple-system, sans-serif", size: 11 },
      showlegend: false,
      xaxis: { showgrid: false, zeroline: false, automargin: true },
      yaxis: { showgrid: true, gridcolor: "rgba(15,23,42,0.06)", zeroline: false, automargin: true },
      xaxis2: { showgrid: false, zeroline: false, automargin: true },
      yaxis2: { showgrid: true, gridcolor: "rgba(15,23,42,0.06)", zeroline: false, automargin: true },
      annotations: [
        { text: "A", x: 0.0, y: 1.08, xref: "paper", yref: "paper", showarrow: false,
          font: { size: 11, color: "#f97316" }, xanchor: "left" },
        { text: "B", x: 0.55, y: 1.08, xref: "paper", yref: "paper", showarrow: false,
          font: { size: 11, color: "#0ea5e9" }, xanchor: "left" },
      ],
    };
    var config = { displayModeBar: false, responsive: true };
    try {
      window.Plotly.react(el, traces, layout, config);
    } catch (e) {
      try { window.Plotly.newPlot(el, traces, layout, config); } catch (e2) {
        el.innerHTML = '<div class="sc-equity-fallback">Equity plot failed.</div>';
      }
    }
  }

  function repaint() {
    if (!bodyEl) return;
    bodyEl.innerHTML = renderBody();
    wireSectionToggles();
    if (!state.loading && !state.error) plotEquity();
  }

  function wireSectionToggles() {
    if (!bodyEl) return;
    var btns = bodyEl.querySelectorAll("[data-section-toggle]");
    for (var i = 0; i < btns.length; i++) {
      btns[i].addEventListener("click", function (e) {
        var id = e.currentTarget.getAttribute("data-section-toggle");
        var cur = state.mobileExpanded[id] !== false;
        // On mobile (single-column) we behave as "collapse one at a time"
        var isMobile = window.matchMedia && window.matchMedia("(max-width: 720px)").matches;
        if (isMobile) {
          // collapse all, then toggle this one
          Object.keys(state.mobileExpanded).forEach(function (k) { state.mobileExpanded[k] = false; });
          state.mobileExpanded[id] = !cur;
        } else {
          state.mobileExpanded[id] = !cur;
        }
        // Local DOM toggle (avoid full repaint to preserve plot)
        var section = bodyEl.querySelector('[data-section="' + id + '"]');
        if (section) {
          var nowOpen = state.mobileExpanded[id];
          section.setAttribute("data-open", nowOpen ? "true" : "false");
          var hdr = section.querySelector(".sc-section-header");
          if (hdr) hdr.setAttribute("aria-expanded", nowOpen ? "true" : "false");
        }
        if (isMobile) {
          // On mobile we changed multiple sections — sync all
          var all = bodyEl.querySelectorAll(".sc-section");
          for (var k = 0; k < all.length; k++) {
            var sid = all[k].getAttribute("data-section");
            var open = state.mobileExpanded[sid] !== false && state.mobileExpanded[sid] === true;
            all[k].setAttribute("data-open", open ? "true" : "false");
            var h2 = all[k].querySelector(".sc-section-header");
            if (h2) h2.setAttribute("aria-expanded", open ? "true" : "false");
          }
        }
      });
    }
    var retry = bodyEl.querySelector('[data-action="retry"]');
    if (retry) retry.addEventListener("click", function () { reload(); });
  }

  // ---------- mount ----------
  function mount() {
    if (root) return;
    root = document.createElement("div");
    root.className = "sc-root";
    root.setAttribute("data-open", "false");

    backdrop = document.createElement("div");
    backdrop.className = "modal-backdrop sc-backdrop";
    backdrop.setAttribute("role", "presentation");
    backdrop.addEventListener("mousedown", function (e) {
      if (e.target === backdrop) close();
    });

    modalEl = document.createElement("div");
    modalEl.className = "modal modal--xl sc-modal";
    modalEl.setAttribute("role", "dialog");
    modalEl.setAttribute("aria-modal", "true");
    modalEl.setAttribute("aria-label", "Compare strategies");

    var header = document.createElement("header");
    header.className = "modal__header sc-header";
    titleEl = document.createElement("h2");
    titleEl.className = "modal__title";
    titleEl.textContent = "Compare strategies";
    header.appendChild(titleEl);

    closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "modal__close sc-close";
    closeBtn.setAttribute("aria-label", "Close comparison");
    closeBtn.innerHTML = "&times;";
    closeBtn.addEventListener("click", close);
    header.appendChild(closeBtn);

    modalEl.appendChild(header);

    bodyEl = document.createElement("div");
    bodyEl.className = "modal__body sc-body";
    modalEl.appendChild(bodyEl);

    backdrop.appendChild(modalEl);
    root.appendChild(backdrop);
    document.body.appendChild(root);
  }

  // ---------- open/close ----------
  function reload() {
    if (!state.pairA || !state.pairB) return;
    state.loading = true;
    state.error = null;
    state.dataA = null;
    state.dataB = null;
    repaint();
    Promise.all([fetchStrategy(state.pairA), fetchStrategy(state.pairB)])
      .then(function (results) {
        state.dataA = results[0];
        state.dataB = results[1];
        state.loading = false;
        titleEl.textContent = "Compare: " + (results[0].pair_id || state.pairA) +
          " vs " + (results[1].pair_id || state.pairB);
        repaint();
      })
      .catch(function (err) {
        state.error = err;
        state.loading = false;
        repaint();
      });
  }

  function open(pairIdA, pairIdB) {
    if (!pairIdA || !pairIdB) {
      if (window.console && console.warn) console.warn("[strategy-compare] open() requires two pair IDs");
      return;
    }
    mount();
    state.pairA = String(pairIdA);
    state.pairB = String(pairIdB);
    state.open = true;
    root.setAttribute("data-open", "true");
    document.body.classList.add("sc-body-locked");
    document.addEventListener("keydown", onDocKey, true);
    reload();
    document.dispatchEvent(new CustomEvent("pfm:strategy-compare-open",
      { detail: { pairIdA: state.pairA, pairIdB: state.pairB } }));
  }

  function close() {
    if (!state.open) return;
    state.open = false;
    if (root) root.setAttribute("data-open", "false");
    document.body.classList.remove("sc-body-locked");
    document.removeEventListener("keydown", onDocKey, true);
    // Purge plot to free memory
    var el = document.getElementById("sc-equity-plot");
    if (el && window.Plotly && window.Plotly.purge) {
      try { window.Plotly.purge(el); } catch (e) { /* noop */ }
    }
    document.dispatchEvent(new CustomEvent("pfm:strategy-compare-close"));
  }

  function isOpen() { return !!state.open; }

  function onDocKey(e) {
    if (!state.open) return;
    if (e.key === "Escape") { e.preventDefault(); close(); }
  }

  // ---------- cmdk wiring ----------
  function parseCompareExpr(expr) {
    if (!expr) return null;
    var parts = expr.split(/\s+|,|\s+vs\.?\s+/i).map(function (s) { return s.trim(); }).filter(Boolean);
    if (parts.length < 2) return null;
    return { a: parts[0], b: parts[1] };
  }

  function registerCmdk() {
    var cmdk = window.PFM && window.PFM.cmdk;
    if (!cmdk || typeof cmdk.register !== "function") return false;
    cmdk.register({
      id: "slash:compare",
      kind: "slash",
      title: "/compare <id1> <id2>",
      sub: "Side-by-side compare two alpha strategies",
      payload: { type: "compare" },
      run: function () { /* run is handled via the slash listener below */ },
    });
    return true;
  }

  function onSlashCompare(e) {
    var d = (e && e.detail) || {};
    var a = d.pairIdA || d.a;
    var b = d.pairIdB || d.b;
    if ((!a || !b) && d.expr) {
      var parsed = parseCompareExpr(d.expr);
      if (parsed) { a = parsed.a; b = parsed.b; }
    }
    if (a && b) open(a, b);
  }

  // Intercept cmdk slash execution by watching the input text.
  // cmdk emits `pfm:cmdk-open` but not a per-command event; we listen for
  // a custom `pfm:slash-compare` we dispatch ourselves OR a generic
  // `pfm:open-compare` event other code can fire.
  function wireSlashFallback() {
    // Listen on a few custom events for ergonomic integration.
    document.addEventListener("pfm:slash-compare", onSlashCompare);
    document.addEventListener("pfm:open-compare", onSlashCompare);
    document.addEventListener("pfm:compare-strategies", onSlashCompare);

    // Hash-deeplink: #compare/<a>/<b>
    function applyHash() {
      var h = (window.location.hash || "").replace(/^#/, "");
      var m = /^compare\/([^/]+)\/([^/]+)$/.exec(h);
      if (m) open(decodeURIComponent(m[1]), decodeURIComponent(m[2]));
    }
    window.addEventListener("hashchange", applyHash);
    // Defer initial apply so DOM is ready
    setTimeout(applyHash, 0);
  }

  // ---------- public API ----------
  var api = {
    open: open,
    close: close,
    isOpen: isOpen,
  };

  window.PFM = window.PFM || {};
  window.PFM.strategyCompare = api;

  // ---------- boot ----------
  function boot() {
    // Try to register with cmdk now; if it's not ready yet, retry briefly.
    if (!registerCmdk()) {
      var tries = 0;
      var iv = setInterval(function () {
        tries++;
        if (registerCmdk() || tries > 20) clearInterval(iv);
      }, 150);
    }
    wireSlashFallback();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
