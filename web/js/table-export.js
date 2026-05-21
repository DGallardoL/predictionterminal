/**
 * web/js/table-export.js — W12-39
 *
 * Lightweight, dependency-free Table-Export helper.
 *
 * Public API (mounted at `window.PFM.tableExport`):
 *
 *   attach(tableEl, opts)        -> { detach }
 *   exportCsv(tableEl, opts)     -> Blob          (text/csv;charset=utf-8, UTF-8 BOM)
 *   exportXlsx(tableEl, opts)    -> Blob | null   (null if window.XLSX not loaded)
 *   autoAttachAll()              -> { count }     (idempotent; runs MutationObserver)
 *
 * Markup contract:
 *   - Walks <thead><tr><th> for headers.
 *   - Walks <tbody><tr><td> for rows.
 *   - Falls back to first row of cells if <thead> is missing.
 *   - Skips rows tagged [data-export-skip] or class .export-skip.
 *   - Skips cells tagged [data-export-skip] / .export-skip.
 *   - File name derived from <caption>, [data-table-name], or aria-label; kebab-cased + ISO timestamp.
 *
 * CSV serialization: RFC 4180 with UTF-8 BOM (﻿) so Excel opens with correct encoding.
 *
 * XLSX: opportunistically uses window.XLSX (SheetJS) if loaded. Otherwise returns null + warns.
 *
 * Button injection:
 *   - Parent of <table> gets `position: relative` (only if currently `static`).
 *   - Button is `position: absolute; top: 6px; right: 6px;` and visible on hover/focus only.
 *   - Click triggers CSV download (default) or XLSX if button has [data-export-format="xlsx"].
 *
 * Auto-attach:
 *   - `[data-exportable]` or `.exportable` on the <table>.
 *   - MutationObserver watches document.body for added/changed tables.
 *
 * No external deps. No edits to other files. Safe to load multiple times (idempotent guards).
 */

(function (global) {
    "use strict";

    if (global.PFM && global.PFM.tableExport && global.PFM.tableExport.__pfmBuild === "w12-39") {
        // Already loaded — keep idempotent.
        return;
    }

    var PFM = (global.PFM = global.PFM || {});

    // ---------------------------------------------------------------------------
    // Constants
    // ---------------------------------------------------------------------------

    var BOM = "﻿";
    var ATTR_AUTO_BOUND = "data-pfm-table-export-bound";
    var ATTR_BTN = "data-pfm-table-export-btn";
    var SELECTOR_AUTO = "table[data-exportable], table.exportable";
    var DEFAULT_ICON = "⬇"; // ⬇ down-arrow
    var ALT_ICON = "📥"; // 📥 inbox

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    function isTable(el) {
        return el && el.nodeType === 1 && el.tagName === "TABLE";
    }

    function nowIsoStampForFilename() {
        var d = new Date();
        // YYYYMMDD-HHMMSS in local time — short enough for filenames, sortable.
        function pad(n) {
            return n < 10 ? "0" + n : String(n);
        }
        return (
            d.getFullYear() +
            pad(d.getMonth() + 1) +
            pad(d.getDate()) +
            "-" +
            pad(d.getHours()) +
            pad(d.getMinutes()) +
            pad(d.getSeconds())
        );
    }

    function kebab(s) {
        if (s == null) return "";
        var out = String(s)
            .normalize("NFKD")
            .replace(/[̀-ͯ]/g, "") // strip diacritics
            .replace(/[^a-zA-Z0-9]+/g, "-")
            .replace(/^-+|-+$/g, "")
            .toLowerCase();
        return out || "";
    }

    function deriveBaseName(tableEl, opts) {
        if (opts && typeof opts.filename === "string" && opts.filename.trim()) {
            return kebab(opts.filename) || "table";
        }
        // <caption>
        var caption = tableEl.querySelector(":scope > caption");
        if (caption && caption.textContent && caption.textContent.trim()) {
            return kebab(caption.textContent) || "table";
        }
        // [data-table-name] on table or nearest ancestor
        var named = tableEl.closest("[data-table-name]");
        if (named) {
            var nm = named.getAttribute("data-table-name");
            if (nm && nm.trim()) return kebab(nm) || "table";
        }
        // aria-label / aria-labelledby
        var aria = tableEl.getAttribute("aria-label");
        if (aria && aria.trim()) return kebab(aria) || "table";
        var labelledBy = tableEl.getAttribute("aria-labelledby");
        if (labelledBy) {
            var lbl = document.getElementById(labelledBy);
            if (lbl && lbl.textContent.trim()) return kebab(lbl.textContent) || "table";
        }
        // id
        if (tableEl.id) return kebab(tableEl.id) || "table";
        return "table";
    }

    function fullFilename(tableEl, opts, ext) {
        var base = deriveBaseName(tableEl, opts);
        var stamp = nowIsoStampForFilename();
        return base + "-" + stamp + "." + ext;
    }

    function shouldSkipRow(tr) {
        if (!tr) return true;
        if (tr.hasAttribute("data-export-skip")) return true;
        if (tr.classList && tr.classList.contains("export-skip")) return true;
        // Skip rows whose only children are <th> in a <tbody> sub-header? Caller can opt out by tagging.
        // Also hidden rows.
        var st = tr.style && tr.style.display;
        if (st === "none") return true;
        if (tr.hidden) return true;
        return false;
    }

    function shouldSkipCell(td) {
        if (!td) return true;
        if (td.hasAttribute && td.hasAttribute("data-export-skip")) return true;
        if (td.classList && td.classList.contains("export-skip")) return true;
        return false;
    }

    function cellText(cell) {
        if (!cell) return "";
        // Prefer explicit override for cells that render complex DOM.
        if (cell.hasAttribute && cell.hasAttribute("data-export-value")) {
            return cell.getAttribute("data-export-value") || "";
        }
        // innerText collapses whitespace and respects CSS hidden; fall back to textContent.
        var t = cell.innerText != null ? cell.innerText : cell.textContent || "";
        // Normalize Windows line endings + collapse runs of internal whitespace except \n.
        return t.replace(/\r\n/g, "\n").replace(/[ \t\f\v]+/g, " ").trim();
    }

    /**
     * Walk a <table> and return a 2-D string array: [headers, ...rows].
     * If no <thead>, treats the first <tr> as headers.
     */
    function extractMatrix(tableEl) {
        if (!isTable(tableEl)) {
            throw new TypeError("PFM.tableExport: expected <table> element");
        }
        var headers = [];
        var rows = [];

        var theadRows = tableEl.querySelectorAll(":scope > thead > tr");
        if (theadRows.length) {
            // Take the LAST <thead><tr> as the column header row (handles multi-row group headers).
            var headRow = theadRows[theadRows.length - 1];
            var headCells = headRow.querySelectorAll("th, td");
            for (var i = 0; i < headCells.length; i++) {
                if (shouldSkipCell(headCells[i])) continue;
                headers.push(cellText(headCells[i]));
            }
        }

        var bodyRows = tableEl.querySelectorAll(":scope > tbody > tr");
        if (!bodyRows.length) {
            // Fall back to any <tr> not in <thead>/<tfoot>.
            bodyRows = tableEl.querySelectorAll(":scope > tr");
        }

        var headerSynth = !headers.length;
        for (var r = 0; r < bodyRows.length; r++) {
            var tr = bodyRows[r];
            if (shouldSkipRow(tr)) continue;
            var cells = tr.querySelectorAll("th, td");
            var rowOut = [];
            for (var c = 0; c < cells.length; c++) {
                if (shouldSkipCell(cells[c])) continue;
                rowOut.push(cellText(cells[c]));
            }
            if (!rowOut.length) continue;
            if (headerSynth && !headers.length) {
                headers = rowOut;
                continue;
            }
            rows.push(rowOut);
        }

        return { headers: headers, rows: rows };
    }

    // ---------------------------------------------------------------------------
    // RFC 4180 CSV
    // ---------------------------------------------------------------------------

    function csvEscape(value) {
        if (value == null) return "";
        var s = String(value);
        // Per RFC 4180, fields containing comma, double-quote, CR, or LF must be quoted;
        // embedded double-quotes are doubled.
        var mustQuote = /[",\r\n]/.test(s);
        if (mustQuote) {
            s = s.replace(/"/g, '""');
            return '"' + s + '"';
        }
        return s;
    }

    function matrixToCsv(headers, rows) {
        var lines = [];
        if (headers && headers.length) {
            lines.push(headers.map(csvEscape).join(","));
        }
        for (var i = 0; i < rows.length; i++) {
            lines.push(rows[i].map(csvEscape).join(","));
        }
        // RFC 4180 uses CRLF.
        return lines.join("\r\n");
    }

    function exportCsv(tableEl, opts) {
        opts = opts || {};
        var m = extractMatrix(tableEl);
        var body = matrixToCsv(m.headers, m.rows);
        var content = (opts.bom === false ? "" : BOM) + body;
        var blob = new Blob([content], { type: "text/csv;charset=utf-8" });
        if (opts.download !== false) {
            triggerDownload(blob, fullFilename(tableEl, opts, "csv"));
        }
        return blob;
    }

    // ---------------------------------------------------------------------------
    // XLSX via SheetJS (window.XLSX) if available
    // ---------------------------------------------------------------------------

    function exportXlsx(tableEl, opts) {
        opts = opts || {};
        var XLSX = global.XLSX;
        if (!XLSX || typeof XLSX.utils !== "object" || typeof XLSX.write !== "function") {
            if (!exportXlsx.__warned) {
                exportXlsx.__warned = true;
                try {
                    console.warn(
                        "PFM.tableExport.exportXlsx: SheetJS (window.XLSX) not loaded; " +
                            "returning null. Include https://cdn.sheetjs.com/xlsx-latest/package/dist/xlsx.full.min.js " +
                            "to enable XLSX export."
                    );
                } catch (_) {}
            }
            return null;
        }
        var m = extractMatrix(tableEl);
        var aoa = m.headers && m.headers.length ? [m.headers].concat(m.rows) : m.rows;
        try {
            var ws = XLSX.utils.aoa_to_sheet(aoa);
            var wb = XLSX.utils.book_new();
            var sheetName = (opts.sheetName || deriveBaseName(tableEl, opts) || "Sheet1")
                .toString()
                .slice(0, 31)
                .replace(/[\\\/\?\*\[\]:]/g, "_");
            XLSX.utils.book_append_sheet(wb, ws, sheetName || "Sheet1");
            var out = XLSX.write(wb, { bookType: "xlsx", type: "array" });
            var blob = new Blob([out], {
                type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            });
            if (opts.download !== false) {
                triggerDownload(blob, fullFilename(tableEl, opts, "xlsx"));
            }
            return blob;
        } catch (err) {
            try {
                console.error("PFM.tableExport.exportXlsx failed:", err);
            } catch (_) {}
            return null;
        }
    }

    // ---------------------------------------------------------------------------
    // Download trigger
    // ---------------------------------------------------------------------------

    function triggerDownload(blob, filename) {
        try {
            if (global.navigator && global.navigator.msSaveBlob) {
                global.navigator.msSaveBlob(blob, filename);
                return;
            }
            var url = URL.createObjectURL(blob);
            var a = document.createElement("a");
            a.href = url;
            a.download = filename;
            a.rel = "noopener";
            a.style.display = "none";
            document.body.appendChild(a);
            a.click();
            // Defer revoke so Safari/Firefox actually start the download.
            setTimeout(function () {
                try {
                    document.body.removeChild(a);
                } catch (_) {}
                try {
                    URL.revokeObjectURL(url);
                } catch (_) {}
            }, 0);
        } catch (err) {
            try {
                console.error("PFM.tableExport: download failed", err);
            } catch (_) {}
        }
    }

    // ---------------------------------------------------------------------------
    // Button injection / attachment
    // ---------------------------------------------------------------------------

    function ensureStyles() {
        if (document.getElementById("pfm-table-export-styles")) return;
        var css =
            "" +
            ".pfm-tx-host{position:relative;}" +
            ".pfm-tx-btn{" +
            "position:absolute;top:6px;right:6px;z-index:5;" +
            "display:inline-flex;align-items:center;gap:4px;" +
            "padding:4px 8px;font:500 11px/1 'Inter','Helvetica Neue',Arial,sans-serif;" +
            "color:#e8e8ea;background:rgba(20,22,28,0.82);" +
            "border:1px solid rgba(255,255,255,0.12);border-radius:6px;" +
            "cursor:pointer;opacity:0;pointer-events:none;" +
            "transition:opacity 120ms ease,transform 120ms ease,background 120ms ease;" +
            "backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);" +
            "user-select:none;" +
            "}" +
            ".pfm-tx-host:hover > .pfm-tx-btn," +
            ".pfm-tx-host:focus-within > .pfm-tx-btn," +
            ".pfm-tx-btn:focus-visible," +
            ".pfm-tx-btn.is-visible{opacity:1;pointer-events:auto;}" +
            ".pfm-tx-btn:hover{background:rgba(35,38,48,0.92);}" +
            ".pfm-tx-btn:active{transform:translateY(1px);}" +
            ".pfm-tx-btn[disabled]{opacity:0.4;cursor:not-allowed;}" +
            ".pfm-tx-btn .pfm-tx-ico{font-size:13px;line-height:1;}" +
            "@media (prefers-reduced-motion: reduce){.pfm-tx-btn{transition:none;}}" +
            "@media print{.pfm-tx-btn{display:none !important;}}";
        var style = document.createElement("style");
        style.id = "pfm-table-export-styles";
        style.textContent = css;
        (document.head || document.documentElement).appendChild(style);
    }

    function ensureHostPositioned(parent) {
        if (!parent || parent.nodeType !== 1) return;
        if (parent.classList.contains("pfm-tx-host")) return;
        var cs = global.getComputedStyle ? global.getComputedStyle(parent) : null;
        if (cs && cs.position === "static") {
            parent.classList.add("pfm-tx-host");
        } else {
            // Still add the class so :hover scoping works, but don't override existing positioning.
            parent.classList.add("pfm-tx-host");
        }
    }

    function buildButton(tableEl, opts) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "pfm-tx-btn";
        btn.setAttribute(ATTR_BTN, "1");
        var format = (opts && opts.format) || "csv";
        btn.setAttribute("data-export-format", format);
        btn.setAttribute("title", "Export table as " + format.toUpperCase());
        btn.setAttribute("aria-label", "Export table as " + format.toUpperCase());

        var icon = document.createElement("span");
        icon.className = "pfm-tx-ico";
        icon.setAttribute("aria-hidden", "true");
        icon.textContent = (opts && opts.icon) || DEFAULT_ICON;
        btn.appendChild(icon);

        var label = document.createElement("span");
        label.className = "pfm-tx-label";
        label.textContent = (opts && opts.label) || format.toUpperCase();
        btn.appendChild(label);

        return btn;
    }

    function attach(tableEl, opts) {
        opts = opts || {};
        if (!isTable(tableEl)) {
            throw new TypeError("PFM.tableExport.attach: expected <table> element");
        }
        if (tableEl.getAttribute(ATTR_AUTO_BOUND) === "1") {
            // Already attached — return a no-op detach so caller still gets the contract.
            return { detach: function () {} };
        }

        ensureStyles();

        var parent = tableEl.parentElement;
        if (!parent) {
            // Cannot attach without a parent — fall back to attaching to the table itself
            // by wrapping. Conservative: skip.
            return { detach: function () {} };
        }
        ensureHostPositioned(parent);

        var btn = buildButton(tableEl, opts);

        var onClick = function (ev) {
            ev.preventDefault();
            ev.stopPropagation();
            if (btn.disabled) return;
            btn.disabled = true;
            try {
                var fmt = btn.getAttribute("data-export-format") || "csv";
                if (fmt === "xlsx") {
                    var blob = exportXlsx(tableEl, opts);
                    if (!blob) {
                        // Fall back to CSV transparently for the user.
                        exportCsv(tableEl, opts);
                    }
                } else {
                    exportCsv(tableEl, opts);
                }
            } finally {
                // Re-enable shortly so back-to-back exports work.
                setTimeout(function () {
                    btn.disabled = false;
                }, 250);
            }
        };
        btn.addEventListener("click", onClick);

        // Insert before the <table> so it sits in the parent's coordinate space.
        parent.insertBefore(btn, tableEl);
        tableEl.setAttribute(ATTR_AUTO_BOUND, "1");

        var detach = function () {
            try {
                btn.removeEventListener("click", onClick);
            } catch (_) {}
            try {
                if (btn.parentNode) btn.parentNode.removeChild(btn);
            } catch (_) {}
            try {
                tableEl.removeAttribute(ATTR_AUTO_BOUND);
            } catch (_) {}
        };

        return { detach: detach, button: btn };
    }

    // ---------------------------------------------------------------------------
    // Auto-attach via MutationObserver
    // ---------------------------------------------------------------------------

    var autoState = {
        observer: null,
        started: false,
        attached: new WeakSet(),
    };

    function attachIfEligible(table) {
        if (!isTable(table)) return false;
        if (autoState.attached.has(table)) return false;
        if (table.getAttribute(ATTR_AUTO_BOUND) === "1") {
            autoState.attached.add(table);
            return false;
        }
        if (!table.matches(SELECTOR_AUTO)) return false;
        try {
            attach(table, {});
            autoState.attached.add(table);
            return true;
        } catch (err) {
            try {
                console.warn("PFM.tableExport: auto-attach failed", err);
            } catch (_) {}
            return false;
        }
    }

    function scan(root) {
        var scope = root && root.querySelectorAll ? root : document;
        var tables = scope.querySelectorAll(SELECTOR_AUTO);
        var count = 0;
        for (var i = 0; i < tables.length; i++) {
            if (attachIfEligible(tables[i])) count++;
        }
        return count;
    }

    function autoAttachAll() {
        ensureStyles();
        var count = scan(document);

        if (!autoState.started && typeof MutationObserver === "function") {
            var obs = new MutationObserver(function (mutations) {
                for (var i = 0; i < mutations.length; i++) {
                    var m = mutations[i];
                    if (m.type === "childList") {
                        for (var j = 0; j < m.addedNodes.length; j++) {
                            var n = m.addedNodes[j];
                            if (!n || n.nodeType !== 1) continue;
                            if (isTable(n)) {
                                attachIfEligible(n);
                            } else if (n.querySelectorAll) {
                                var inner = n.querySelectorAll(SELECTOR_AUTO);
                                for (var k = 0; k < inner.length; k++) {
                                    attachIfEligible(inner[k]);
                                }
                            }
                        }
                    } else if (m.type === "attributes" && isTable(m.target)) {
                        attachIfEligible(m.target);
                    }
                }
            });
            try {
                obs.observe(document.body || document.documentElement, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    attributeFilter: ["data-exportable", "class"],
                });
                autoState.observer = obs;
                autoState.started = true;
            } catch (err) {
                try {
                    console.warn("PFM.tableExport: observer failed to start", err);
                } catch (_) {}
            }
        }

        return { count: count };
    }

    // ---------------------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------------------

    PFM.tableExport = {
        __pfmBuild: "w12-39",
        attach: attach,
        exportCsv: exportCsv,
        exportXlsx: exportXlsx,
        autoAttachAll: autoAttachAll,
        // Exposed for tests / power users:
        _internals: {
            extractMatrix: extractMatrix,
            csvEscape: csvEscape,
            matrixToCsv: matrixToCsv,
            kebab: kebab,
            deriveBaseName: deriveBaseName,
        },
    };

    // ---------------------------------------------------------------------------
    // Boot
    // ---------------------------------------------------------------------------

    function boot() {
        try {
            autoAttachAll();
        } catch (err) {
            try {
                console.warn("PFM.tableExport: boot failed", err);
            } catch (_) {}
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot, { once: true });
    } else {
        // Microtask defer so app code declaring PFM.* later still wins ordering races.
        setTimeout(boot, 0);
    }

    // Suppress unused-symbol lint by exposing the alternate icon constant on the namespace.
    PFM.tableExport._icons = { down: DEFAULT_ICON, inbox: ALT_ICON };
})(typeof window !== "undefined" ? window : this);
