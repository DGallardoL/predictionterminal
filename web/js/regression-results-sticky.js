/* ============================================================
 * regression-results-sticky.js  —  T61 (Track H)
 *
 * Sticky summary card for /fit results.
 *
 * Listens for:
 *   - "pfm:fit-complete"  CustomEvent({ detail: { result } })
 *   - "pfm:fit-start"     CustomEvent (optional — collapses prior cards
 *                         immediately when the user clicks Fit again,
 *                         before the network round-trip completes)
 *
 * Renders into [data-mode-pane="regression"] a stack container
 *   <div id="pfm-fit-summary-stack">…</div>
 * inserted directly after the closest <form> inside the regression pane.
 * Newest card stays expanded; previous cards collapse to a 1-line summary
 * row (newest-on-top); a max of 4 cards are retained.
 *
 * Self-registers via window.PFM.registerRegressionResults() and also
 * auto-runs on DOMContentLoaded so the index-html-owner only needs to
 * add a single <script defer src="js/regression-results-sticky.js"></script>.
 *
 * NO external dependencies.
 * ============================================================ */

(function () {
  "use strict";

  const STACK_ID = "pfm-fit-summary-stack";
  const PANE_SELECTOR = '[data-mode-pane="regression"]';
  const REPORT_TARGET = "#results";
  const MAX_CARDS = 4;
  let _registered = false;
  let _fitCounter = 0;

  // ────────── DOM helpers ──────────

  function _qsPane() {
    return document.querySelector(PANE_SELECTOR);
  }

  function _ensureStack() {
    const pane = _qsPane();
    if (!pane) return null;
    let stack = document.getElementById(STACK_ID);
    if (stack) return stack;

    stack = document.createElement("div");
    stack.id = STACK_ID;
    stack.setAttribute("aria-live", "polite");
    stack.setAttribute("aria-label", "Regression fit summary");

    // Insert directly after the first <form> inside the pane.
    // Falls back to "first child" if the pane somehow has no form.
    const form = pane.querySelector("form");
    if (form && form.parentNode) {
      form.parentNode.insertBefore(stack, form.nextSibling);
    } else {
      pane.insertBefore(stack, pane.firstChild);
    }
    return stack;
  }

  // ────────── Number formatting ──────────

  function _fmtR2(x) {
    if (x == null || !isFinite(x)) return "—";
    return x.toFixed(3);
  }
  function _fmtBeta(x) {
    if (x == null || !isFinite(x)) return "—";
    const abs = Math.abs(x);
    if (abs >= 100) return x.toFixed(0);
    if (abs >= 10) return x.toFixed(2);
    if (abs >= 1) return x.toFixed(3);
    return x.toFixed(4);
  }
  function _fmtT(x) {
    if (x == null || !isFinite(x)) return "—";
    const sign = x < 0 ? "−" : "";
    return sign + Math.abs(x).toFixed(2);
  }
  function _fmtP(p) {
    if (p == null || !isFinite(p)) return "p=—";
    if (p < 0.001) return "p<0.001";
    if (p < 0.01) return "p=" + p.toFixed(3);
    if (p < 0.1) return "p=" + p.toFixed(3);
    return "p=" + p.toFixed(2);
  }
  function _gradeR2(r2) {
    if (r2 == null || !isFinite(r2)) return "weak";
    if (r2 >= 0.5) return "strong";
    if (r2 >= 0.2) return "ok";
    return "weak";
  }
  function _escape(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // ────────── Verdict synthesis ──────────

  /**
   * Pick the "dominant" factor: largest |t_stat| among factors with
   * p<0.05; fall back to largest |t_stat| overall when no factor is
   * significant.
   */
  function _pickDominant(result) {
    const factors = Array.isArray(result?.factors) ? result.factors : [];
    if (!factors.length) return null;
    const sig = factors
      .filter((f) => isFinite(f.p_value) && f.p_value < 0.05)
      .sort((a, b) => Math.abs(b.t_stat) - Math.abs(a.t_stat));
    if (sig.length) return sig[0];
    const sorted = factors.slice().sort(
      (a, b) => Math.abs(b.t_stat || 0) - Math.abs(a.t_stat || 0),
    );
    return sorted[0] || null;
  }

  /**
   * Plain-language one-line verdict.
   * Template: "<TICKER> moves <β>σ per unit <factor> shock —
   *            <sig phrase> (<p-value>)."
   * Falls back to a generic readout if no usable dominant factor.
   */
  function _verdict(result, dom) {
    const ticker = (result?.ticker || "ticker").toUpperCase();
    const r2 = result?.model?.r_squared;
    if (!dom) {
      return `${ticker}: model R² ${_fmtR2(r2)} — no factor reached statistical significance.`;
    }
    const beta = dom.beta;
    const p = dom.p_value;
    const isSig = isFinite(p) && p < 0.05;
    const strength = Math.abs(beta || 0);
    const magWord =
      strength >= 1 ? "strongly"
      : strength >= 0.5 ? "materially"
      : strength >= 0.1 ? "modestly"
      : "weakly";
    const sigPhrase = isSig
      ? "statistically significant"
      : "not statistically significant";
    const factorId = _escape(dom.id || "factor");
    return `<strong>${_escape(ticker)}</strong> moves <strong>${_fmtBeta(beta)}σ</strong> ${magWord} per unit <strong>${factorId}</strong> shock — ${sigPhrase} (${_fmtP(p)}).`;
  }

  function _verdictTone(dom) {
    if (!dom) return "weak";
    return isFinite(dom.p_value) && dom.p_value < 0.05 ? "strong" : "weak";
  }

  // Residual t-stat: the FitResponse doesn't ship a single "residual t-stat"
  // scalar, so we compute the closest meaningful proxy: |alpha / residual_std|
  // — i.e. is the intercept distinguishable from noise. If alpha is missing,
  // fall back to the Durbin-Watson |2 - DW| signal which flags residual
  // autocorrelation (a different but legitimate "are residuals well-behaved?"
  // headline).
  function _residualHeadline(result) {
    const alpha = result?.model?.alpha;
    const sd = result?.model?.residual_std;
    if (isFinite(alpha) && isFinite(sd) && sd > 0) {
      const t = alpha / sd;
      return {
        label: "α / σ_ε",
        value: _fmtT(t),
        sub: `α=${_fmtBeta(alpha)} · σ_ε=${_fmtBeta(sd)}`,
        sig: Math.abs(t) >= 2 ? "strong" : "weak",
      };
    }
    const dw = result?.diagnostics?.durbin_watson;
    if (isFinite(dw)) {
      const dist = Math.abs(2 - dw);
      return {
        label: "DW residual",
        value: dw.toFixed(2),
        sub: `|2−DW|=${dist.toFixed(2)}`,
        sig: dist < 0.4 ? "strong" : "weak",
      };
    }
    return { label: "Residuals", value: "—", sub: "", sig: "weak" };
  }

  // ────────── Card markup ──────────

  function _buildCard(result, ordinal) {
    const dom = _pickDominant(result);
    const r2 = result?.model?.r_squared;
    const r2adj = result?.model?.r_squared_adj;
    const n = result?.n_obs_used || result?.n_obs || 0;
    const quality = result?.verdict || "borderline";
    const ticker = result?.ticker || "—";
    const startEnd =
      result?.start && result?.end
        ? `${result.start} → ${result.end}`
        : "";

    const grade = _gradeR2(r2);
    const dominantValue = dom ? `β=${_fmtBeta(dom.beta)}` : "—";
    const dominantSub = dom
      ? `${_escape(dom.id)} <span class="pfm-fs-sig" data-sig="${
          isFinite(dom.p_value) && dom.p_value < 0.05 ? "strong" : "weak"
        }">${_fmtP(dom.p_value)}</span>`
      : "no factor";

    const residual = _residualHeadline(result);
    const verdictHtml = _verdict(result, dom);
    const verdictTone = _verdictTone(dom);

    const card = document.createElement("div");
    card.className = "pfm-fs-card";
    card.setAttribute("role", "status");
    card.dataset.fitOrdinal = String(ordinal);

    card.innerHTML = [
      '<div class="pfm-fs-header">',
      '  <div class="pfm-fs-title">',
      `    Fit summary · <span class="pfm-fs-ticker">${_escape(ticker)}</span>`,
      "  </div>",
      '  <div class="pfm-fs-meta">',
      `    <span class="pfm-fs-quality" data-q="${_escape(quality)}">${_escape(quality.replace(/_/g, " "))}</span>`,
      `    <span class="pfm-fs-dot"></span>n=${n}`,
      startEnd ? `<span class="pfm-fs-dot"></span>${_escape(startEnd)}` : "",
      "  </div>",
      "</div>",

      '<div class="pfm-fs-metrics" role="group" aria-label="Hero metrics">',
      `  <div class="pfm-fs-metric" data-grade="${grade}">`,
      '    <div class="pfm-fs-metric-label">R²</div>',
      `    <div class="pfm-fs-metric-value">${_fmtR2(r2)}</div>`,
      `    <div class="pfm-fs-metric-sub">adj ${_fmtR2(r2adj)}</div>`,
      "  </div>",
      '  <div class="pfm-fs-metric">',
      '    <div class="pfm-fs-metric-label">Dominant factor</div>',
      `    <div class="pfm-fs-metric-value">${dominantValue}</div>`,
      `    <div class="pfm-fs-metric-sub">${dominantSub}</div>`,
      "  </div>",
      `  <div class="pfm-fs-metric" data-sig="${residual.sig}">`,
      `    <div class="pfm-fs-metric-label">${_escape(residual.label)}</div>`,
      `    <div class="pfm-fs-metric-value">${residual.value}</div>`,
      `    <div class="pfm-fs-metric-sub">${residual.sub}</div>`,
      "  </div>",
      "</div>",

      '<div class="pfm-fs-right">',
      '  <button type="button" class="pfm-fs-close" aria-label="Dismiss summary card">✕</button>',
      `  <a class="pfm-fs-jump" href="${REPORT_TARGET}" data-pfm-jump="1">`,
      "    Jump to full report <span class=\"pfm-fs-arrow\" aria-hidden=\"true\">↓</span>",
      "  </a>",
      "</div>",

      `<div class="pfm-fs-verdict ${verdictTone === "weak" ? "is-weak" : ""}">${verdictHtml}</div>`,
    ].join("\n");

    // Wire interactions
    const jump = card.querySelector(".pfm-fs-jump");
    if (jump) {
      jump.addEventListener("click", function (e) {
        e.preventDefault();
        const target = document.querySelector(REPORT_TARGET);
        if (target && typeof target.scrollIntoView === "function") {
          target.scrollIntoView({ behavior: "smooth", block: "start" });
          target.setAttribute("tabindex", "-1");
          try { target.focus({ preventScroll: true }); } catch (_) { /* ignore */ }
        }
      });
    }
    const close = card.querySelector(".pfm-fs-close");
    if (close) {
      close.addEventListener("click", function () {
        card.remove();
      });
    }

    return card;
  }

  // ────────── Stack management ──────────

  function _renderFitComplete(result) {
    if (!result || typeof result !== "object") return;
    const stack = _ensureStack();
    if (!stack) return;

    _fitCounter += 1;

    // Collapse existing cards
    Array.from(stack.querySelectorAll(".pfm-fs-card")).forEach((c) => {
      c.dataset.collapsed = "true";
    });

    const card = _buildCard(result, _fitCounter);
    // Newest goes on top
    stack.insertBefore(card, stack.firstChild);

    // Trim
    const cards = stack.querySelectorAll(".pfm-fs-card");
    if (cards.length > MAX_CARDS) {
      for (let i = MAX_CARDS; i < cards.length; i += 1) cards[i].remove();
    }
  }

  function _handleFitStart() {
    const stack = document.getElementById(STACK_ID);
    if (!stack) return;
    // Collapse all immediately so the freshly-submitted fit's prior result
    // stops competing for attention before the new card arrives.
    Array.from(stack.querySelectorAll(".pfm-fs-card")).forEach((c) => {
      c.dataset.collapsed = "true";
    });
  }

  // ────────── Public registration ──────────

  function register() {
    if (_registered) return;
    _registered = true;

    document.addEventListener("pfm:fit-complete", function (e) {
      try {
        const detail = (e && e.detail) || {};
        const result = detail.result || detail.response || detail;
        _renderFitComplete(result);
      } catch (err) {
        // Never let a render bug break the app — the full report panel
        // is always still rendered downstream.
        if (window.console && console.warn) {
          console.warn("[pfm-fit-summary] render failed:", err);
        }
      }
    });
    document.addEventListener("pfm:fit-start", _handleFitStart);
  }

  // Expose imperative API + namespace hook
  window.PFM = window.PFM || {};
  window.PFM.registerRegressionResults = register;
  // Imperative render hook for callers that already have a result and
  // want to surface it without re-dispatching the event.
  window.PFM.renderRegressionSummary = function (result) {
    register();
    _renderFitComplete(result);
  };

  // Auto-register so the index-html-owner only needs to add one script tag.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", register);
  } else {
    register();
  }
})();
