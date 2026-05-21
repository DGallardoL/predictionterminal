/**
 * portfolio-import.js  (W11-58 / T59, wave-11)
 *
 * Drag-and-drop CSV upload UI for the T33 portfolio import endpoint
 * (`POST /portfolio/import`). Renders a friendly drop-zone, parses a
 * CSV client-side, previews the first rows, validates ticker / shares /
 * cost_basis, and submits to the backend with a progress bar.
 *
 * Public API (mounted at window.PFM.portfolio):
 *
 *   window.PFM.portfolio = {
 *     mount(containerEl, opts)
 *       -> { destroy, importCsv(text), getHandle() }
 *     upload(csvText) -> Promise<{handle, row_count, ...}>
 *     parse(csvText)  -> { rows: [...], errors: [...] }
 *   }
 *
 * Auto-mount:
 *   Any element with the `data-pfm-portfolio-import` attribute on the
 *   page at DOMContentLoaded gets `mount()` called automatically. The
 *   index.html owner does NOT need to add a <script> hook; just a
 *   <div data-pfm-portfolio-import></div> placeholder.
 *
 * Mount options (`opts` or `data-*` on the container):
 *   - apiBase            base URL for /portfolio/import (defaults to
 *                        window.PFM.apiBase or "")
 *   - endpoint           path (default "/portfolio/import")
 *   - previewRows        max rows shown in preview table (default 10)
 *   - maxRows            client-side hard cap (default 200, mirrors API)
 *   - maxBytes           upload byte cap (default 256 * 1024)
 *   - viewPortfolioHref  link target for the success "View portfolio"
 *                        anchor (default "#terminal/portfolio")
 *   - onSuccess(payload) callback after successful upload
 *   - onError(err)       callback after failed upload (in addition to
 *                        PFM.errors.show())
 *
 * Pairs with web/css/portfolio-import.css.
 *
 * Endpoint reference (api/src/pfm/portfolio_import_router.py):
 *   - Header: `ticker, shares, cost_basis` (cost_basis optional)
 *   - Ticker: 1-5 uppercase letters
 *   - Shares: positive float
 *   - Cost basis: non-negative float (or blank)
 *   - 200 rows max, 256 KiB max
 *   - Accepts raw text/csv body OR multipart `file=` field
 */
(function () {
  "use strict";

  // ---------- defaults --------------------------------------------------

  var DEFAULTS = Object.freeze({
    apiBase: "",
    endpoint: "/portfolio/import",
    previewRows: 10,
    maxRows: 200,
    maxBytes: 256 * 1024,
    viewPortfolioHref: "#terminal/portfolio",
    onSuccess: null,
    onError: null,
  });

  var TICKER_RE = /^[A-Z]{1,5}$/;
  var SAMPLE_CSV = "ticker,shares,cost_basis\nNVDA,10,500.00\n";

  // ---------- module-level instance tracking ----------------------------

  var INSTANCES = [];

  // ---------- utilities --------------------------------------------------

  function _resolveApiBase(optsBase) {
    if (typeof optsBase === "string" && optsBase.length) return optsBase;
    if (window.PFM && typeof window.PFM.apiBase === "string") {
      return window.PFM.apiBase;
    }
    if (typeof window.PFM_API_BASE === "string") return window.PFM_API_BASE;
    return "";
  }

  function _readDataOpts(el) {
    var out = {};
    if (!el || !el.dataset) return out;
    var ds = el.dataset;
    if (ds.apiBase) out.apiBase = ds.apiBase;
    if (ds.endpoint) out.endpoint = ds.endpoint;
    if (ds.previewRows) out.previewRows = parseInt(ds.previewRows, 10);
    if (ds.maxRows) out.maxRows = parseInt(ds.maxRows, 10);
    if (ds.maxBytes) out.maxBytes = parseInt(ds.maxBytes, 10);
    if (ds.viewPortfolioHref) out.viewPortfolioHref = ds.viewPortfolioHref;
    return out;
  }

  function _mergeOpts(opts, dataOpts) {
    var merged = {};
    var k;
    for (k in DEFAULTS) if (Object.prototype.hasOwnProperty.call(DEFAULTS, k)) merged[k] = DEFAULTS[k];
    for (k in dataOpts) if (Object.prototype.hasOwnProperty.call(dataOpts, k)) merged[k] = dataOpts[k];
    if (opts && typeof opts === "object") {
      for (k in opts) if (Object.prototype.hasOwnProperty.call(opts, k)) merged[k] = opts[k];
    }
    return merged;
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

  function _showError(message, traceId) {
    if (window.PFM && window.PFM.errors && typeof window.PFM.errors.show === "function") {
      window.PFM.errors.show(message, { kind: "error", traceId: traceId || null });
    } else {
      console.warn("[portfolio-import]", message);
    }
  }

  // ---------- CSV parsing ------------------------------------------------

  // Minimal RFC-4180-ish line splitter that tolerates quoted fields and
  // embedded commas. Carriage returns are stripped before line-splitting.
  function _splitCsvRow(line) {
    var out = [];
    var cur = "";
    var inQuotes = false;
    for (var i = 0; i < line.length; i++) {
      var ch = line.charAt(i);
      if (inQuotes) {
        if (ch === "\"") {
          if (i + 1 < line.length && line.charAt(i + 1) === "\"") {
            cur += "\"";
            i++;
          } else {
            inQuotes = false;
          }
        } else {
          cur += ch;
        }
      } else if (ch === "\"") {
        inQuotes = true;
      } else if (ch === ",") {
        out.push(cur);
        cur = "";
      } else {
        cur += ch;
      }
    }
    out.push(cur);
    return out;
  }

  /**
   * Parse a CSV body. Returns { rows, errors, warnings, header }.
   *
   *   rows[i] = {
   *     line: <int>, ticker: str|null, shares: num|null,
   *     cost_basis: num|null, errors: {ticker, shares, cost_basis}
   *   }
   *
   * errors[i] is a top-level error (missing header, empty body, too many
   * rows etc.) — these block submission entirely.
   *
   * Row-level errors live on rows[i].errors and let the user preview
   * what's wrong before fixing the CSV.
   */
  function parseCsv(text, opts) {
    var maxRows = (opts && opts.maxRows) || DEFAULTS.maxRows;
    var result = { rows: [], errors: [], warnings: [], header: null };

    if (typeof text !== "string" || !text.trim()) {
      result.errors.push("CSV is empty.");
      return result;
    }

    var lines = text.replace(/\r/g, "").split("\n");
    // Drop fully blank lines, preserve original line numbers for error msgs.
    var headerLine = null;
    var dataLines = [];
    for (var i = 0; i < lines.length; i++) {
      var raw = lines[i];
      if (raw.trim() === "") continue;
      if (headerLine === null) {
        headerLine = { idx: i + 1, raw: raw };
      } else {
        dataLines.push({ idx: i + 1, raw: raw });
      }
    }

    if (!headerLine) {
      result.errors.push("CSV is empty.");
      return result;
    }

    var headerCells = _splitCsvRow(headerLine.raw).map(function (c) {
      return c.trim().toLowerCase();
    });
    result.header = headerCells;

    var idxTicker = headerCells.indexOf("ticker");
    var idxShares = headerCells.indexOf("shares");
    var idxCost = headerCells.indexOf("cost_basis");
    var missing = [];
    if (idxTicker < 0) missing.push("ticker");
    if (idxShares < 0) missing.push("shares");
    if (missing.length) {
      result.errors.push("Missing required column(s): " + missing.join(", "));
      return result;
    }

    if (dataLines.length === 0) {
      result.errors.push("No data rows after header.");
      return result;
    }
    if (dataLines.length > maxRows) {
      result.errors.push("Too many rows; max " + maxRows + " (got " + dataLines.length + ").");
      // we still parse and show what we can so the user can preview
    }

    var seen = {};
    for (var j = 0; j < dataLines.length; j++) {
      var dl = dataLines[j];
      var cells = _splitCsvRow(dl.raw);

      var row = {
        line: dl.idx,
        ticker: null,
        shares: null,
        cost_basis: null,
        errors: { ticker: null, shares: null, cost_basis: null },
      };

      var tickerRaw = (cells[idxTicker] || "").trim().toUpperCase();
      var sharesRaw = (cells[idxShares] || "").trim();
      var costRaw = idxCost >= 0 ? (cells[idxCost] || "").trim() : "";

      // ---- ticker ----
      if (!tickerRaw) {
        row.errors.ticker = "Required.";
      } else if (!TICKER_RE.test(tickerRaw)) {
        row.errors.ticker = "Must be 1-5 uppercase letters.";
      } else {
        row.ticker = tickerRaw;
        if (seen[tickerRaw]) {
          result.warnings.push("Row " + dl.idx + ": duplicate ticker " + tickerRaw + ".");
        }
        seen[tickerRaw] = true;
      }

      // ---- shares ----
      if (!sharesRaw) {
        row.errors.shares = "Required.";
      } else {
        var sharesVal = Number(sharesRaw);
        if (!isFinite(sharesVal) || isNaN(sharesVal)) {
          row.errors.shares = "Not a number.";
        } else if (!(sharesVal > 0)) {
          row.errors.shares = "Must be > 0.";
        } else {
          row.shares = sharesVal;
        }
      }

      // ---- cost_basis (optional) ----
      if (costRaw === "") {
        row.cost_basis = null;
      } else {
        var costVal = Number(costRaw);
        if (!isFinite(costVal) || isNaN(costVal)) {
          row.errors.cost_basis = "Not a number.";
        } else if (costVal < 0) {
          row.errors.cost_basis = "Must be >= 0.";
        } else {
          row.cost_basis = costVal;
        }
      }

      result.rows.push(row);
    }

    return result;
  }

  function _hasRowErrors(parsed) {
    if (parsed.errors.length) return true;
    for (var i = 0; i < parsed.rows.length; i++) {
      var e = parsed.rows[i].errors;
      if (e.ticker || e.shares || e.cost_basis) return true;
    }
    return false;
  }

  // ---------- upload -----------------------------------------------------

  /**
   * Submit a CSV body. Resolves with the parsed JSON response on 2xx,
   * rejects with an Error (`.detail` populated when the server returned
   * JSON) on non-2xx or network failure.
   *
   * onProgress is optional: receives a number in [0, 1] when upload
   * progress is observable, NaN when the browser only emits load events.
   */
  function uploadCsv(csvText, opts) {
    opts = opts || {};
    var apiBase = _resolveApiBase(opts.apiBase);
    var endpoint = opts.endpoint || DEFAULTS.endpoint;
    var url = apiBase + endpoint;
    var onProgress = typeof opts.onProgress === "function" ? opts.onProgress : null;

    return new Promise(function (resolve, reject) {
      var xhr;
      try {
        xhr = new XMLHttpRequest();
      } catch (e) {
        reject(e);
        return;
      }
      xhr.open("POST", url, true);
      xhr.setRequestHeader("Content-Type", "text/csv");
      xhr.setRequestHeader("Accept", "application/json");

      if (xhr.upload && onProgress) {
        xhr.upload.onprogress = function (evt) {
          if (evt && evt.lengthComputable && evt.total > 0) {
            onProgress(evt.loaded / evt.total);
          } else {
            onProgress(NaN);
          }
        };
      }

      xhr.onload = function () {
        var status = xhr.status;
        var body = xhr.responseText || "";
        var parsedBody = null;
        try { parsedBody = JSON.parse(body); } catch (_) { parsedBody = null; }

        if (status >= 200 && status < 300) {
          if (onProgress) onProgress(1);
          resolve(parsedBody || {});
          return;
        }
        var detail = "";
        if (parsedBody && typeof parsedBody.detail === "string") {
          detail = parsedBody.detail;
        } else if (parsedBody && parsedBody.detail) {
          try { detail = JSON.stringify(parsedBody.detail); } catch (_) { detail = ""; }
        }
        var err = new Error(detail || ("HTTP " + status));
        err.status = status;
        err.detail = detail || body;
        err.body = parsedBody;
        reject(err);
      };

      xhr.onerror = function () {
        var e = new Error("Network error");
        e.status = 0;
        reject(e);
      };
      xhr.onabort = function () {
        var e = new Error("Upload aborted");
        e.status = 0;
        reject(e);
      };

      try {
        xhr.send(csvText);
      } catch (e) {
        reject(e);
      }
    });
  }

  // ---------- DOM builders ----------------------------------------------

  function _buildShell() {
    var root = document.createElement("div");
    root.className = "pfm-pi-widget";
    root.innerHTML = [
      '<div class="pfm-pi-dropzone" data-pfm-pi-drop role="button" tabindex="0" aria-label="Upload CSV — drag a file here or press Enter to browse">',
      '  <input type="file" accept=".csv,text/csv" class="pfm-pi-file-input" data-pfm-pi-file hidden>',
      '  <div class="pfm-pi-dropzone-inner">',
      '    <div class="pfm-pi-dropzone-icon" aria-hidden="true">CSV</div>',
      '    <div class="pfm-pi-dropzone-copy">Drag CSV here or click to browse</div>',
      '    <div class="pfm-pi-dropzone-hint">Header: <code>ticker,shares,cost_basis</code></div>',
      '    <button type="button" class="pfm-pi-link" data-pfm-pi-template>Download sample CSV</button>',
      '  </div>',
      '</div>',
      '<div class="pfm-pi-status" data-pfm-pi-status aria-live="polite"></div>',
      '<div class="pfm-pi-preview" data-pfm-pi-preview hidden>',
      '  <div class="pfm-pi-preview-head">',
      '    <span class="pfm-pi-preview-title">Preview</span>',
      '    <span class="pfm-pi-preview-meta" data-pfm-pi-preview-meta></span>',
      '  </div>',
      '  <div class="pfm-pi-preview-scroll">',
      '    <table class="pfm-pi-table" data-pfm-pi-table>',
      '      <thead><tr><th>#</th><th>Ticker</th><th>Shares</th><th>Cost basis</th></tr></thead>',
      '      <tbody data-pfm-pi-tbody></tbody>',
      '    </table>',
      '  </div>',
      '  <div class="pfm-pi-errlist" data-pfm-pi-errlist hidden></div>',
      '</div>',
      '<div class="pfm-pi-progress" data-pfm-pi-progress hidden>',
      '  <div class="pfm-pi-progress-bar" data-pfm-pi-progress-bar></div>',
      '</div>',
      '<div class="pfm-pi-actions">',
      '  <button type="button" class="pfm-pi-btn pfm-pi-btn-ghost" data-pfm-pi-reset hidden>Clear</button>',
      '  <button type="button" class="pfm-pi-btn pfm-pi-btn-primary" data-pfm-pi-upload disabled>Upload</button>',
      '</div>',
      '<div class="pfm-pi-success" data-pfm-pi-success hidden>',
      '  <div class="pfm-pi-success-row">',
      '    <span class="pfm-pi-success-badge">Imported</span>',
      '    <span class="pfm-pi-success-handle" data-pfm-pi-handle></span>',
      '    <span class="pfm-pi-success-count" data-pfm-pi-count></span>',
      '  </div>',
      '  <a class="pfm-pi-success-link" data-pfm-pi-view href="#">View portfolio →</a>',
      '</div>',
    ].join("\n");
    return root;
  }

  function _renderPreview(tbody, parsed, opts) {
    var maxShown = opts.previewRows;
    var rows = parsed.rows.slice(0, maxShown);
    var html = "";
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      var e = r.errors;
      var tickerCell = _renderCell(r.ticker, e.ticker, "ticker");
      var sharesCell = _renderCell(r.shares != null ? r.shares : "", e.shares, "shares");
      var costRaw = r.cost_basis == null ? "—" : r.cost_basis;
      var costCell = _renderCell(costRaw, e.cost_basis, "cost_basis", r.cost_basis == null);
      html += '<tr data-line="' + r.line + '">';
      html += '<td class="pfm-pi-num">' + r.line + "</td>";
      html += tickerCell + sharesCell + costCell;
      html += "</tr>";
    }
    tbody.innerHTML = html;
  }

  function _renderCell(value, err, kind, isMissing) {
    var cls = "pfm-pi-cell pfm-pi-cell-" + kind;
    if (err) cls += " pfm-pi-cell-bad";
    if (isMissing) cls += " pfm-pi-cell-muted";
    var title = err ? ' title="' + _escape(err) + '"' : "";
    return '<td class="' + cls + '"' + title + ">" + _escape(value) + "</td>";
  }

  function _renderErrList(el, parsed) {
    var msgs = [];
    for (var i = 0; i < parsed.errors.length; i++) msgs.push(parsed.errors[i]);
    var rowBad = 0;
    for (var j = 0; j < parsed.rows.length; j++) {
      var e = parsed.rows[j].errors;
      if (e.ticker || e.shares || e.cost_basis) rowBad++;
    }
    if (rowBad > 0) {
      msgs.push(rowBad + " row" + (rowBad === 1 ? "" : "s") + " have errors. Fix the highlighted cells before uploading.");
    }
    for (var k = 0; k < parsed.warnings.length; k++) {
      msgs.push("Warning: " + parsed.warnings[k]);
    }
    if (!msgs.length) {
      el.hidden = true;
      el.innerHTML = "";
      return;
    }
    el.hidden = false;
    el.innerHTML = msgs.map(function (m) {
      return '<div class="pfm-pi-err">' + _escape(m) + "</div>";
    }).join("");
  }

  // ---------- instance ---------------------------------------------------

  function _createInstance(containerEl, opts) {
    if (!containerEl || containerEl.nodeType !== 1) {
      throw new Error("portfolio-import.mount: containerEl is required");
    }
    var dataOpts = _readDataOpts(containerEl);
    var merged = _mergeOpts(opts, dataOpts);

    var shell = _buildShell();
    containerEl.innerHTML = "";
    containerEl.appendChild(shell);
    containerEl.classList.add("pfm-pi-host");

    var dropEl = shell.querySelector("[data-pfm-pi-drop]");
    var fileInput = shell.querySelector("[data-pfm-pi-file]");
    var templateBtn = shell.querySelector("[data-pfm-pi-template]");
    var statusEl = shell.querySelector("[data-pfm-pi-status]");
    var previewEl = shell.querySelector("[data-pfm-pi-preview]");
    var previewMetaEl = shell.querySelector("[data-pfm-pi-preview-meta]");
    var tbody = shell.querySelector("[data-pfm-pi-tbody]");
    var errListEl = shell.querySelector("[data-pfm-pi-errlist]");
    var progressEl = shell.querySelector("[data-pfm-pi-progress]");
    var progressBar = shell.querySelector("[data-pfm-pi-progress-bar]");
    var resetBtn = shell.querySelector("[data-pfm-pi-reset]");
    var uploadBtn = shell.querySelector("[data-pfm-pi-upload]");
    var successEl = shell.querySelector("[data-pfm-pi-success]");
    var handleEl = shell.querySelector("[data-pfm-pi-handle]");
    var countEl = shell.querySelector("[data-pfm-pi-count]");
    var viewLinkEl = shell.querySelector("[data-pfm-pi-view]");

    var state = {
      destroyed: false,
      csvText: "",
      parsed: null,
      handle: null,
      uploading: false,
    };

    // ---- public methods ----
    function getHandle() {
      return state.handle;
    }

    function setStatus(msg, kind) {
      if (!msg) {
        statusEl.textContent = "";
        statusEl.removeAttribute("data-kind");
        return;
      }
      statusEl.textContent = msg;
      statusEl.setAttribute("data-kind", kind || "info");
    }

    function reset() {
      state.csvText = "";
      state.parsed = null;
      state.handle = null;
      state.uploading = false;
      previewEl.hidden = true;
      tbody.innerHTML = "";
      errListEl.hidden = true;
      errListEl.innerHTML = "";
      progressEl.hidden = true;
      progressBar.style.width = "0%";
      uploadBtn.disabled = true;
      uploadBtn.textContent = "Upload";
      resetBtn.hidden = true;
      successEl.hidden = true;
      handleEl.textContent = "";
      countEl.textContent = "";
      setStatus("");
      fileInput.value = "";
    }

    function applyParse(text) {
      state.csvText = text;
      var parsed = parseCsv(text, merged);
      state.parsed = parsed;

      _renderPreview(tbody, parsed, merged);
      _renderErrList(errListEl, parsed);

      var shownN = Math.min(parsed.rows.length, merged.previewRows);
      var totalN = parsed.rows.length;
      var metaParts = [];
      metaParts.push(totalN + " row" + (totalN === 1 ? "" : "s") + " detected");
      if (totalN > shownN) metaParts.push("showing first " + shownN);
      previewMetaEl.textContent = metaParts.join(" · ");
      previewEl.hidden = false;
      resetBtn.hidden = false;

      var hasErrors = _hasRowErrors(parsed);
      uploadBtn.disabled = hasErrors || totalN === 0 || state.uploading;
      if (parsed.errors.length) {
        setStatus(parsed.errors[0], "error");
      } else if (hasErrors) {
        setStatus("Fix the highlighted cells before uploading.", "warn");
      } else {
        setStatus("Ready to upload " + totalN + " row" + (totalN === 1 ? "" : "s") + ".", "ok");
      }
    }

    function importCsv(text) {
      if (state.destroyed) return;
      applyParse(text || "");
    }

    function setUploading(isUp) {
      state.uploading = isUp;
      progressEl.hidden = !isUp;
      uploadBtn.disabled = isUp;
      uploadBtn.textContent = isUp ? "Uploading…" : "Upload";
      resetBtn.disabled = isUp;
      if (!isUp) progressBar.style.width = "0%";
    }

    function doUpload() {
      if (state.uploading) return Promise.resolve(null);
      if (!state.parsed || _hasRowErrors(state.parsed)) {
        setStatus("Fix the highlighted cells before uploading.", "warn");
        return Promise.resolve(null);
      }
      if (state.csvText.length > merged.maxBytes) {
        var msg = "File too large; max " + merged.maxBytes + " bytes.";
        setStatus(msg, "error");
        _showError(msg);
        return Promise.resolve(null);
      }

      setUploading(true);
      setStatus("Uploading…", "info");

      return uploadCsv(state.csvText, {
        apiBase: merged.apiBase,
        endpoint: merged.endpoint,
        onProgress: function (frac) {
          if (typeof frac === "number" && isFinite(frac)) {
            progressBar.style.width = Math.round(frac * 100) + "%";
          } else {
            // Indeterminate; pulse via CSS class
            progressBar.style.width = "100%";
          }
        },
      }).then(function (payload) {
        setUploading(false);
        state.handle = (payload && payload.handle) || null;
        successEl.hidden = false;
        handleEl.textContent = state.handle || "(no handle)";
        var rc = (payload && payload.row_count) || (state.parsed ? state.parsed.rows.length : 0);
        countEl.textContent = rc + " row" + (rc === 1 ? "" : "s");
        var href = merged.viewPortfolioHref || "#terminal/portfolio";
        if (state.handle) {
          var sep = href.indexOf("?") >= 0 ? "&" : (href.indexOf("#") >= 0 ? "?" : "?");
          // Append handle as a query param only for non-hash anchors
          if (href.charAt(0) === "#") {
            viewLinkEl.href = href + (href.indexOf("?") >= 0 ? "&" : "?") + "handle=" + encodeURIComponent(state.handle);
          } else {
            viewLinkEl.href = href + sep + "handle=" + encodeURIComponent(state.handle);
          }
        } else {
          viewLinkEl.href = href;
        }
        setStatus("Imported " + rc + " row" + (rc === 1 ? "" : "s") + ".", "ok");
        if (typeof merged.onSuccess === "function") {
          try { merged.onSuccess(payload); } catch (_) { /* noop */ }
        }
        return payload;
      }).catch(function (err) {
        setUploading(false);
        var detail = (err && (err.detail || err.message)) || "Upload failed.";
        setStatus("Upload failed: " + detail, "error");
        _showError("Upload failed: " + detail);
        if (typeof merged.onError === "function") {
          try { merged.onError(err); } catch (_) { /* noop */ }
        }
        return null;
      });
    }

    // ---- input wiring ----
    function readFile(file) {
      if (!file) return;
      if (file.size != null && file.size > merged.maxBytes) {
        var msg = "File too large; max " + merged.maxBytes + " bytes.";
        setStatus(msg, "error");
        _showError(msg);
        return;
      }
      var reader = new FileReader();
      reader.onload = function () {
        var text = typeof reader.result === "string" ? reader.result : "";
        applyParse(text);
      };
      reader.onerror = function () {
        setStatus("Could not read file.", "error");
        _showError("Could not read file.");
      };
      reader.readAsText(file);
    }

    function onDragOver(e) {
      e.preventDefault();
      e.stopPropagation();
      dropEl.classList.add("is-dragover");
      if (e.dataTransfer) {
        try { e.dataTransfer.dropEffect = "copy"; } catch (_) {}
      }
    }
    function onDragLeave(e) {
      e.preventDefault();
      e.stopPropagation();
      // Guard against child enter/leave flicker
      if (e.relatedTarget && dropEl.contains(e.relatedTarget)) return;
      dropEl.classList.remove("is-dragover");
    }
    function onDrop(e) {
      e.preventDefault();
      e.stopPropagation();
      dropEl.classList.remove("is-dragover");
      if (!e.dataTransfer || !e.dataTransfer.files) return;
      var f = e.dataTransfer.files[0];
      if (!f) return;
      readFile(f);
    }
    function onDropClick(e) {
      // Ignore clicks coming from the template button or other interactive
      // children — they have their own handlers.
      var t = e.target;
      while (t && t !== dropEl) {
        if (t.hasAttribute && t.hasAttribute("data-pfm-pi-template")) return;
        t = t.parentNode;
      }
      fileInput.click();
    }
    function onDropKey(e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    }
    function onFileChange() {
      var f = fileInput.files && fileInput.files[0];
      if (f) readFile(f);
    }
    function onTemplate(e) {
      e.preventDefault();
      e.stopPropagation();
      try {
        var blob = new Blob([SAMPLE_CSV], { type: "text/csv" });
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = "portfolio-sample.csv";
        document.body.appendChild(a);
        a.click();
        setTimeout(function () {
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        }, 0);
      } catch (_) {
        // Fallback: open in new window
        var w = window.open("", "_blank");
        if (w) {
          w.document.body.innerText = SAMPLE_CSV;
        }
      }
    }
    function onUploadClick() {
      doUpload();
    }
    function onResetClick() {
      reset();
    }

    dropEl.addEventListener("dragover", onDragOver);
    dropEl.addEventListener("dragenter", onDragOver);
    dropEl.addEventListener("dragleave", onDragLeave);
    dropEl.addEventListener("drop", onDrop);
    dropEl.addEventListener("click", onDropClick);
    dropEl.addEventListener("keydown", onDropKey);
    fileInput.addEventListener("change", onFileChange);
    templateBtn.addEventListener("click", onTemplate);
    uploadBtn.addEventListener("click", onUploadClick);
    resetBtn.addEventListener("click", onResetClick);

    function destroy() {
      if (state.destroyed) return;
      state.destroyed = true;
      try {
        dropEl.removeEventListener("dragover", onDragOver);
        dropEl.removeEventListener("dragenter", onDragOver);
        dropEl.removeEventListener("dragleave", onDragLeave);
        dropEl.removeEventListener("drop", onDrop);
        dropEl.removeEventListener("click", onDropClick);
        dropEl.removeEventListener("keydown", onDropKey);
        fileInput.removeEventListener("change", onFileChange);
        templateBtn.removeEventListener("click", onTemplate);
        uploadBtn.removeEventListener("click", onUploadClick);
        resetBtn.removeEventListener("click", onResetClick);
      } catch (_) { /* noop */ }
      try { containerEl.innerHTML = ""; } catch (_) { /* noop */ }
      containerEl.classList.remove("pfm-pi-host");
      var i = INSTANCES.indexOf(api);
      if (i >= 0) INSTANCES.splice(i, 1);
    }

    var api = {
      destroy: destroy,
      importCsv: importCsv,
      getHandle: getHandle,
      _state: state,
      _opts: merged,
    };
    INSTANCES.push(api);
    return api;
  }

  // ---------- auto-mount -------------------------------------------------

  function _autoMount() {
    var nodes = document.querySelectorAll("[data-pfm-portfolio-import]");
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (el.__pfmPortfolioMounted) continue;
      try {
        var inst = _createInstance(el, null);
        el.__pfmPortfolioMounted = inst;
      } catch (e) {
        console.warn("[portfolio-import] auto-mount failed", e);
      }
    }
  }

  // ---------- public namespace ------------------------------------------

  window.PFM = window.PFM || {};
  if (!window.PFM.portfolio || !window.PFM.portfolio.__t59) {
    window.PFM.portfolio = {
      __t59: true,
      mount: function (containerEl, opts) {
        return _createInstance(containerEl, opts);
      },
      upload: function (csvText, opts) {
        return uploadCsv(csvText, opts || {});
      },
      parse: function (csvText, opts) {
        return parseCsv(csvText, opts || {});
      },
      _instances: INSTANCES,
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", _autoMount, { once: true });
  } else {
    // DOM already parsed — schedule async to give the host page a tick.
    setTimeout(_autoMount, 0);
  }
})();
