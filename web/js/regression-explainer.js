/* ════════════════════════════════════════════════════════════════════════
 * T63 · REGRESSION → PLAIN ENGLISH EXPLAINER
 * ────────────────────────────────────────────────────────────────────────
 * Surface: window.PFM.regressionExplainer
 *
 *   .generate(result)                  → { bullets:[…], significant:[…] }
 *   .render(containerEl, result)       → renders into the element
 *   .toggle('expert'|'plain')          → switches view mode (persists in LS)
 *
 * Auto-mounts: listens for `pfm:fit-complete` on document with the
 * FitResponse payload in event.detail (or event.detail.result), looks up
 * the container at `#pfm-regression-explainer`, and renders.
 *
 * Tone rules enforced here:
 *   - p < 0.001        →  "p<0.001"  (never "p=0.000")
 *   - R²               →  2 decimals (never 3+)
 *   - β units          →  "σ per σ" because /fit standardises factors
 *   - HAC SE           →  named explicitly when regression === "hac"
 *   - DW               →  reported, with a one-clause autocorrelation read
 *
 * No external deps. Works in any browser shipping ES2017+.
 * ──────────────────────────────────────────────────────────────────────── */

(function () {
  "use strict";

  /* ───────────────── helpers ───────────────── */

  // NB: the "<" must be HTML-escaped to "&lt;" because _fmtP() output is
  // injected directly into innerHTML further down. Spec wording is still
  // "p<0.001" — only the on-the-wire encoding changes.
  const _fmtP = (p) => {
    if (p == null || Number.isNaN(p)) return "n/a";
    if (p < 0.001) return "p&lt;0.001";
    if (p < 0.01)  return "p=" + p.toFixed(3);
    return "p=" + p.toFixed(3);
  };

  const _fmtT = (t) => {
    if (t == null || Number.isNaN(t)) return "n/a";
    return "t=" + (t >= 0 ? "" : "") + t.toFixed(2);
  };

  // R² gets EXACTLY 2 decimals — never rounded past.
  const _fmtR2 = (r) => {
    if (r == null || Number.isNaN(r)) return "n/a";
    return r.toFixed(2);
  };

  const _fmtPct = (r) => {
    if (r == null || Number.isNaN(r)) return "n/a";
    return Math.round(r * 100) + "%";
  };

  const _fmtBeta = (b) => {
    if (b == null || Number.isNaN(b)) return "n/a";
    const sign = b >= 0 ? "+" : "−";
    return sign + Math.abs(b).toFixed(2);
  };

  const _fmtCi = (lo, hi) => {
    if (lo == null || hi == null) return "n/a";
    return "[" + lo.toFixed(2) + ", " + hi.toFixed(2) + "]";
  };

  const _ciExcludesZero = (lo, hi) =>
    lo != null && hi != null && (lo > 0 || hi < 0);

  const _esc = (s) =>
    String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  // Pretty-print a factor id like "polymarket:fed-cuts-ge-2-2026" or
  // "sentiment_trump_hawkish". Keeps it short for chip context.
  const _prettyFactor = (id) => {
    if (!id) return "factor";
    const raw = String(id);
    const stripped = raw.replace(/^[a-z]+:/, ""); // drop source: prefix
    if (stripped.length > 32) {
      return stripped.slice(0, 30) + "…";
    }
    return stripped;
  };

  /* ───────────────── core: pick the dominant factor ───────────────── */

  const _dominantFactor = (result) => {
    const factors = (result && Array.isArray(result.factors)) ? result.factors : [];
    if (!factors.length) return null;
    // Prefer the largest |t-stat| amongst factors with finite t. Falls back
    // to |β| if every t-stat is missing.
    let best = null;
    let bestKey = -Infinity;
    for (const f of factors) {
      const t = (f && typeof f.t_stat === "number" && !Number.isNaN(f.t_stat))
        ? Math.abs(f.t_stat) : null;
      const b = (f && typeof f.beta   === "number" && !Number.isNaN(f.beta))
        ? Math.abs(f.beta) : 0;
      const key = (t != null) ? t : (b * 1e-3); // β-only loses to any real t
      if (key > bestKey) { bestKey = key; best = f; }
    }
    return best;
  };

  const _significant = (result) => {
    const factors = (result && Array.isArray(result.factors)) ? result.factors : [];
    return factors
      .filter((f) => f && typeof f.p_value === "number" && f.p_value < 0.05)
      .sort((a, b) => Math.abs(b.t_stat || 0) - Math.abs(a.t_stat || 0));
  };

  /* ───────────────── bullet generators ───────────────── */

  const _bulletDirection = (result, dominant) => {
    if (!dominant) {
      return {
        kind: "info",
        html: "<b>No clear driver.</b> None of the supplied factors moved the ticker in a coherent direction over this window.",
      };
    }
    const ticker = result && result.ticker ? result.ticker : "the stock";
    const beta   = dominant.beta;
    const dir    = beta >= 0 ? "lifts" : "drags";
    const magText = Math.abs(beta).toFixed(2);
    const html =
      "<b>" + _esc(_prettyFactor(dominant.id)) + " is the largest driver.</b> " +
      "A 1σ shock in this factor " + dir + " <b>" + _esc(ticker) + "</b> by " +
      "<span class=\"pfm-rxp-num-mono\">" + _fmtBeta(beta) + "σ</span> on the day " +
      "<span style=\"color:var(--pfm-rxp-ink-4)\">(σ per σ; factors standardised)</span>.";

    let kind = "info";
    if (dominant.p_value != null && dominant.p_value < 0.05 && Math.abs(beta) >= 0.3) kind = "good";
    else if (dominant.p_value != null && dominant.p_value > 0.1) kind = "warn";

    return { kind, html };
  };

  const _bulletConfidence = (result, dominant) => {
    if (!dominant) {
      return {
        kind: "warn",
        html: "<b>No factor passes the significance threshold</b> (all p &gt; 0.05). Treat coefficient signs as noise on this sample.",
      };
    }
    const p   = dominant.p_value;
    const t   = dominant.t_stat;
    const lo  = dominant.ci_low;
    const hi  = dominant.ci_high;
    const sigText = (p != null && p < 0.05)
      ? "is statistically significant"
      : "is <b>not</b> statistically significant at the 5% level";
    const ciClause = (lo != null && hi != null)
      ? (_ciExcludesZero(lo, hi)
          ? " The 95% confidence interval <span class=\"pfm-rxp-num-mono\">" + _fmtCi(lo, hi) + "</span> excludes zero."
          : " The 95% confidence interval <span class=\"pfm-rxp-num-mono\">" + _fmtCi(lo, hi) + "</span> straddles zero, so the sign is not pinned down.")
      : "";
    const html =
      "<b>This relationship " + sigText + "</b> " +
      "<span class=\"pfm-rxp-num-mono\">(" + _fmtP(p) + ", " + _fmtT(t) + ")</span>." +
      ciClause;

    let kind = "info";
    if (p != null) {
      if (p < 0.01) kind = "good";
      else if (p < 0.05) kind = "info";
      else kind = "warn";
    }
    return { kind, html };
  };

  const _bulletFit = (result) => {
    const r2     = result && result.model && typeof result.model.r_squared === "number"
      ? result.model.r_squared : null;
    const ticker = result && result.ticker ? result.ticker : "the ticker";
    const isHac  = result && result.regression === "hac";
    const dw     = result && result.diagnostics && typeof result.diagnostics.durbin_watson === "number"
      ? result.diagnostics.durbin_watson : null;
    const lag    = result && result.diagnostics && typeof result.diagnostics.hac_lag === "number"
      ? result.diagnostics.hac_lag : null;

    let r2Phrase;
    if (r2 == null) r2Phrase = "Model fit could not be computed.";
    else if (r2 >= 0.20) r2Phrase = "The model explains <b>" + _fmtPct(r2) + "</b> of <b>" + _esc(ticker) + "</b>'s daily variance " +
        "<span class=\"pfm-rxp-num-mono\">(R²=" + _fmtR2(r2) + ")</span> — a meaningful share for daily returns.";
    else if (r2 >= 0.05) r2Phrase = "The model explains <b>" + _fmtPct(r2) + "</b> of <b>" + _esc(ticker) + "</b>'s daily variance " +
        "<span class=\"pfm-rxp-num-mono\">(R²=" + _fmtR2(r2) + ")</span> — modest but non-trivial.";
    else r2Phrase = "The model explains only <b>" + _fmtPct(r2) + "</b> of <b>" + _esc(ticker) + "</b>'s daily variance " +
        "<span class=\"pfm-rxp-num-mono\">(R²=" + _fmtR2(r2) + ")</span> — most of the move comes from elsewhere.";

    const seClause = isHac
      ? " <b>HAC-robust standard errors</b> used" + (lag != null ? " (lag " + lag + ")" : "") + ","
      : " Standard OLS errors,";

    let dwClause = "";
    if (dw != null) {
      // DW around 2 ≈ no autocorrelation; well below 1.5 or above 2.5 = flag.
      const ok = dw >= 1.5 && dw <= 2.5;
      const phrase = ok
        ? "residuals show no autocorrelation"
        : "residuals show autocorrelation (check the lag)";
      dwClause = " " + phrase + " <span class=\"pfm-rxp-num-mono\">(DW=" + dw.toFixed(2) + ")</span>.";
    } else {
      dwClause = "";
    }

    const html = r2Phrase + seClause + dwClause;

    let kind;
    if (r2 == null) kind = "warn";
    else if (r2 >= 0.20) kind = "good";
    else if (r2 >= 0.05) kind = "info";
    else kind = "warn";
    return { kind, html };
  };

  /* ───────────────── public: generate ───────────────── */

  function generate(result) {
    if (!result || typeof result !== "object") {
      return { bullets: [], significant: [] };
    }
    const dominant = _dominantFactor(result);
    const bullets  = [
      _bulletDirection(result, dominant),
      _bulletConfidence(result, dominant),
      _bulletFit(result),
    ];
    const sig = _significant(result).map((f) => ({
      id:    f.id,
      beta:  f.beta,
      sign:  (f.beta >= 0 ? "pos" : "neg"),
      p:     f.p_value,
      t:     f.t_stat,
    }));
    return { bullets, significant: sig };
  }

  /* ───────────────── public: render ───────────────── */

  // Module-level state: most-recent container + result, so toggle()
  // can re-render without the caller having to re-pass them.
  let _lastContainer = null;
  let _lastResult    = null;
  const _LS_KEY = "pfm.regressionExplainer.mode";

  function _readMode() {
    try {
      const v = window.localStorage && window.localStorage.getItem(_LS_KEY);
      return (v === "expert" || v === "plain") ? v : "plain";
    } catch (_) {
      return "plain";
    }
  }
  function _writeMode(m) {
    try {
      if (window.localStorage) window.localStorage.setItem(_LS_KEY, m);
    } catch (_) { /* ignore */ }
  }

  // Find sibling "expert" detail block to show/hide. The index-html-owner
  // is expected to mark the existing coefficient table with one of these.
  const _expertSelectors = [
    "[data-pfm-explainer-expert]",
    "#pfm-regression-detail",
    "#regression-detail",
    "#fit-detail",
  ];
  function _findExpertEl() {
    for (const sel of _expertSelectors) {
      const el = document.querySelector(sel);
      if (el) return el;
    }
    return null;
  }

  function _renderCardHTML(result, mode) {
    const { bullets, significant } = generate(result);

    const ticker = result && result.ticker ? _esc(result.ticker) : "—";
    const isExpert = mode === "expert";

    const bulletsHTML = bullets.map((b, i) => {
      const cls = "pfm-rxp-bullet pfm-rxp-bullet--" + (b.kind || "info");
      return (
        '<li class="' + cls + '">' +
          '<span class="pfm-rxp-num" aria-hidden="true">' + (i + 1) + '</span>' +
          '<span class="pfm-rxp-bullet-text">' + b.html + '</span>' +
        '</li>'
      );
    }).join("");

    const chipsHTML = significant.length
      ? significant.map((s) =>
          '<li class="pfm-rxp-chip" data-sign="' + (s.sign === "neg" ? "neg" : "pos") + '">' +
            '<span class="pfm-rxp-chip-dot" aria-hidden="true"></span>' +
            '<span class="pfm-rxp-chip-id">' + _esc(_prettyFactor(s.id)) + '</span>' +
            '<span class="pfm-rxp-chip-beta">' + _fmtBeta(s.beta) + 'σ</span>' +
          '</li>'
        ).join("")
      : '<li class="pfm-rxp-sig-empty">No factor crosses p&lt;0.05 in this window.</li>';

    const sigBlockHTML = isExpert
      ? "" // expert mode hides the chip row too — the detail table is enough
      : (
          '<div class="pfm-rxp-sig-wrap">' +
            '<p class="pfm-rxp-sig-label">Significant predictors (p&lt;0.05, ranked by |t|)</p>' +
            '<ul class="pfm-rxp-sig-list">' + chipsHTML + '</ul>' +
          '</div>'
        );

    const bulletsBlock = isExpert
      ? "" // expert mode hides bullets; detail table elsewhere does the talking
      : '<ul class="pfm-rxp-bullets">' + bulletsHTML + '</ul>';

    const expertHint = isExpert
      ? '<p class="pfm-rxp-empty">Expert view — refer to the detailed coefficient table below.</p>'
      : "";

    return (
      '<section class="pfm-rxp-card" aria-label="Plain-English regression summary">' +
        '<header class="pfm-rxp-head">' +
          '<h3 class="pfm-rxp-title">' +
            '<span class="pfm-rxp-eyebrow">What this means · ' + ticker + '</span>' +
            'Plain-English interpretation' +
          '</h3>' +
          '<div class="pfm-rxp-toggle" role="group" aria-label="View mode">' +
            '<button type="button" class="pfm-rxp-toggle-btn" data-pfm-rxp-mode="plain"  aria-pressed="' + (!isExpert) + '">Plain</button>' +
            '<button type="button" class="pfm-rxp-toggle-btn" data-pfm-rxp-mode="expert" aria-pressed="' + (isExpert)  + '">Expert</button>' +
          '</div>' +
        '</header>' +
        bulletsBlock +
        sigBlockHTML +
        expertHint +
      '</section>'
    );
  }

  function _applyExpertVisibility(mode) {
    const expertEl = _findExpertEl();
    if (!expertEl) return; // mount-side may not be wired yet — that's fine
    if (mode === "expert") {
      expertEl.removeAttribute("hidden");
      expertEl.classList.remove("pfm-rxp-hide");
    } else {
      // Plain mode: keep the detail block visible (user can still scroll),
      // but DO NOT auto-hide — the spec says Expert shows the table; Plain
      // shows just bullets. The card itself swaps; the detail elsewhere
      // is the user's lower-fold reference. We only TOGGLE if the mount
      // explicitly opts in via data-pfm-explainer-hide-on-plain.
      if (expertEl.hasAttribute("data-pfm-explainer-hide-on-plain")) {
        expertEl.setAttribute("hidden", "");
      }
    }
  }

  function _bindToggle(containerEl) {
    if (!containerEl) return;
    const btns = containerEl.querySelectorAll(".pfm-rxp-toggle-btn");
    btns.forEach((btn) => {
      btn.addEventListener("click", function (ev) {
        ev.preventDefault();
        const mode = btn.getAttribute("data-pfm-rxp-mode") || "plain";
        toggle(mode);
      });
    });
  }

  function render(containerEl, result) {
    if (!containerEl) return;
    _lastContainer = containerEl;
    _lastResult    = result || null;
    if (!result) {
      containerEl.innerHTML = '<section class="pfm-rxp-card"><div class="pfm-rxp-empty">Run a fit to see a plain-English summary.</div></section>';
      return;
    }
    const mode = _readMode();
    containerEl.innerHTML = _renderCardHTML(result, mode);
    _bindToggle(containerEl);
    _applyExpertVisibility(mode);
  }

  /* ───────────────── public: toggle ───────────────── */

  function toggle(mode) {
    const next = (mode === "expert") ? "expert" : "plain";
    _writeMode(next);
    if (_lastContainer && _lastResult) {
      _lastContainer.innerHTML = _renderCardHTML(_lastResult, next);
      _bindToggle(_lastContainer);
    }
    _applyExpertVisibility(next);
  }

  /* ───────────────── event wiring ───────────────── */

  function _onFitComplete(ev) {
    const detail = ev && ev.detail;
    if (!detail) return;
    // Accept either `{detail: <FitResponse>}` or `{detail: {result: <FitResponse>}}`
    const result = (detail && typeof detail === "object" && "model" in detail)
      ? detail
      : (detail && detail.result ? detail.result : null);
    if (!result) return;
    const mount = document.getElementById("pfm-regression-explainer");
    if (!mount) return; // index-html-owner hasn't added the mount yet
    render(mount, result);
  }

  function _wire() {
    document.addEventListener("pfm:fit-complete", _onFitComplete);
    // If the mount already exists at script load and a result is cached
    // on window (some panels stash it), do an initial render.
    const mount = document.getElementById("pfm-regression-explainer");
    const cached = (window.PFM && window.PFM._lastFitResult) || null;
    if (mount && cached) render(mount, cached);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", _wire, { once: true });
  } else {
    _wire();
  }

  /* ───────────────── export ───────────────── */

  const api = { generate, render, toggle };

  window.PFM = window.PFM || {};
  window.PFM.regressionExplainer = api;
})();
