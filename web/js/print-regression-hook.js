/* =========================================================================
 * web/js/print-regression-hook.js
 * -------------------------------------------------------------------------
 * Companion JS hook for css/print-regression-export.css.
 *
 * On `beforeprint` (when the user invokes Print / Save-as-PDF and the
 * regression mode is the active pane), this script:
 *
 *   1. Tags <body data-print-mode="regression"> so the print stylesheet
 *      activates its page rule and regression-only hide-list.
 *   2. Injects #reg-print-title-block at the top of the regression pane:
 *      a title (ticker + "Factor Model Report"), a sub-line with factors
 *      and date range, and a generated-at timestamp.
 *   3. Injects #reg-print-fitparams: a compact k/v table mirroring the
 *      fit form (epsilon, lag, oos %, bootstrap iterations, etc.).
 *   4. Walks every `.js-plotly-plot` inside [data-mode-pane="regression"]
 *      and, if window.Plotly is available, calls Plotly.toImage at PNG
 *      width=1400 height=420 scale=2, inserts a sibling <img class=
 *      "reg-print-chart" data-print-replaced="1">, and hides the source
 *      div via the same `data-print-replaced` attribute.
 *   5. Sets a `data-print-header-left` / `data-print-header-right` on
 *      the title block so the @page running-header strings populate.
 *   6. Inserts a small italic #reg-print-footer-note at the bottom of
 *      the pane.
 *
 * On `afterprint` everything injected is removed and the Plotly divs
 * are restored.
 *
 * IMPORTANT: This file does not assume Plotly is loaded. If it is missing
 * we silently skip step 4 — the print stylesheet has a fallback that
 * lets the live canvas print (it may be blank in some browsers but that
 * is the best we can do without Plotly).
 *
 * Mount via: <script src="js/print-regression-hook.js" defer></script>
 * (the index-html-owner adds this tag).
 *
 * Author: agent-w13-35 · Wave 13 · task W13-35
 * ========================================================================= */

(function () {
  "use strict";

  var REG_SELECTOR = '[data-mode-pane="regression"]';
  var INJECTED_IDS = [
    "reg-print-title-block",
    "reg-print-fitparams",
    "reg-print-footer-note",
  ];

  function regPane() {
    // The active regression pane is the one not hidden. We grab the first
    // matching node; index.html only has a single regression pane element.
    return document.querySelector(REG_SELECTOR);
  }

  function isRegressionActive() {
    var pane = regPane();
    if (!pane) return false;
    // A pane is "active" when it has the `.active` class OR no `hidden`
    // attribute AND its computed display is not "none".
    if (pane.classList && pane.classList.contains("active")) return true;
    if (pane.hasAttribute("hidden")) return false;
    var cs = window.getComputedStyle(pane);
    return cs && cs.display !== "none";
  }

  function val(id, fallback) {
    var el = document.getElementById(id);
    if (!el) return fallback;
    var v = (el.value != null ? el.value : el.textContent) || "";
    v = String(v).trim();
    return v.length ? v : fallback;
  }

  function nowISO() {
    var d = new Date();
    // Local ISO without ms: 2026-05-16 14:32:07
    function pad(n) { return n < 10 ? "0" + n : "" + n; }
    return (
      d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate()) +
      " " + pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds())
    );
  }

  function buildTitleBlock(pane) {
    var ticker = val("ticker", "—").toUpperCase();
    var start  = val("start", "");
    var end    = val("end", "");
    var eps    = val("epsilon", "0.01");
    var lag    = val("lag", "5");
    var ts     = nowISO();

    // Count factors: prefer the chip list, fall back to the textarea.
    var chips = pane ? pane.querySelectorAll(".factor-chip, .chip-factor, [data-factor-slug]") : [];
    var nFactors = chips ? chips.length : 0;
    if (!nFactors) {
      var ta = document.getElementById("factors") || document.getElementById("factor-list");
      if (ta && ta.value) {
        nFactors = ta.value.split(/[\s,\n]+/).filter(Boolean).length;
      }
    }

    var block = document.createElement("header");
    block.id = "reg-print-title-block";
    block.setAttribute("data-print-injected", "1");
    block.setAttribute("data-print-header-left",  "Prediction Terminal · " + ticker);
    block.setAttribute("data-print-header-right", ts);

    var h1 = document.createElement("h1");
    h1.textContent = ticker + " · Factor Model Report";
    block.appendChild(h1);

    var sub = document.createElement("p");
    sub.className = "sub";
    var parts = [nFactors + " factor" + (nFactors === 1 ? "" : "s")];
    if (start && end) parts.push(start + " → " + end);
    parts.push("ε=" + eps);
    parts.push("lag=" + lag);
    sub.textContent = parts.join(" · ");
    block.appendChild(sub);

    var tsLine = document.createElement("p");
    tsLine.className = "ts";
    tsLine.textContent = "Generated " + ts;
    block.appendChild(tsLine);

    return block;
  }

  function buildFitParams() {
    var rows = [
      ["Ticker",   val("ticker", "—").toUpperCase()],
      ["Start",    val("start", "—")],
      ["End",      val("end", "—")],
      ["Epsilon",  val("epsilon", "0.01")],
      ["HAC lag",  val("lag", "5")],
      ["OOS %",    val("oos", "0")],
      ["Bootstrap", val("bootstrap", "0")],
      ["Rolling",  val("rolling", "0")],
      ["Quantile", val("quantile-input", "—")],
      ["PCA",      val("pca-input", "—")],
    ];

    var table = document.createElement("section");
    table.id = "reg-print-fitparams";
    table.setAttribute("data-print-injected", "1");

    rows.forEach(function (kv) {
      if (!kv[1] || kv[1] === "—" || kv[1] === "0") return; // suppress empties
      var row = document.createElement("div");
      row.className = "row";
      var k = document.createElement("span");
      k.className = "k";
      k.textContent = kv[0];
      var v = document.createElement("span");
      v.className = "v";
      v.textContent = kv[1];
      row.appendChild(k);
      row.appendChild(v);
      table.appendChild(row);
    });

    return table;
  }

  function buildFooterNote() {
    var note = document.createElement("p");
    note.id = "reg-print-footer-note";
    note.setAttribute("data-print-injected", "1");
    note.textContent =
      "This report is informational only. Coefficients are estimated with HAC " +
      "(heteroskedasticity- and autocorrelation-consistent) standard errors and are POC-quality. Verify before deploying capital.";
    return note;
  }

  function plotlyToImg(pane) {
    if (!window.Plotly || typeof window.Plotly.toImage !== "function") return [];
    var divs = pane.querySelectorAll(".js-plotly-plot, .plotly");
    var promises = [];
    divs.forEach(function (div) {
      if (!div || div.getAttribute("data-print-replaced") === "1") return;
      var p = window.Plotly.toImage(div, {
        format: "png",
        width: 1400,
        height: 420,
        scale: 2,
      }).then(function (dataUrl) {
        var img = document.createElement("img");
        img.src = dataUrl;
        img.className = "reg-print-chart";
        img.alt = "Regression chart";
        img.setAttribute("data-print-replaced", "1");
        // Insert directly after the live plotly div.
        if (div.parentNode) div.parentNode.insertBefore(img, div.nextSibling);
        div.setAttribute("data-print-replaced", "1");
      }).catch(function () {
        // Conversion failed — leave the original visible; the CSS
        // fallback constrains its size.
      });
      promises.push(p);
    });
    return promises;
  }

  function cleanup() {
    document.body.removeAttribute("data-print-mode");

    // Remove every injected node.
    INJECTED_IDS.forEach(function (id) {
      var el = document.getElementById(id);
      if (el && el.parentNode) el.parentNode.removeChild(el);
    });

    // Remove every Plotly png replacement and unhide the originals.
    var imgs = document.querySelectorAll('img.reg-print-chart[data-print-replaced="1"]');
    imgs.forEach(function (img) {
      if (img.parentNode) img.parentNode.removeChild(img);
    });
    var hidden = document.querySelectorAll('[data-print-replaced="1"]');
    hidden.forEach(function (el) {
      el.removeAttribute("data-print-replaced");
    });
  }

  function onBeforePrint() {
    if (!isRegressionActive()) return;
    var pane = regPane();
    if (!pane) return;

    // Avoid double-injecting if the user prints twice in a row without
    // an `afterprint` event firing (some browsers swallow it).
    cleanup();

    document.body.setAttribute("data-print-mode", "regression");

    var title = buildTitleBlock(pane);
    pane.insertBefore(title, pane.firstChild);

    var params = buildFitParams();
    // Place fitparams right after the title block.
    title.parentNode.insertBefore(params, title.nextSibling);

    var footer = buildFooterNote();
    pane.appendChild(footer);

    // Fire off Plotly→PNG conversion. We don't await — the browser
    // print dialog is synchronous after beforeprint returns. Most
    // browsers (Chrome/Edge) wait a tick before snapshotting, which
    // is enough for Plotly.toImage to resolve on small charts. If it
    // doesn't resolve in time, the fallback CSS still renders the
    // canvas at constrained size.
    try { plotlyToImg(pane); } catch (e) { /* swallow */ }
  }

  function onAfterPrint() {
    cleanup();
  }

  // Register on both the legacy event AND the modern matchMedia path so
  // Safari and Chrome both fire correctly.
  window.addEventListener("beforeprint", onBeforePrint);
  window.addEventListener("afterprint",  onAfterPrint);

  if (window.matchMedia) {
    try {
      var mql = window.matchMedia("print");
      mql.addEventListener("change", function (ev) {
        if (ev.matches) onBeforePrint();
        else            onAfterPrint();
      });
    } catch (e) {
      // Older Safari: addEventListener on MediaQueryList not supported.
      // beforeprint/afterprint above are sufficient.
    }
  }
})();
