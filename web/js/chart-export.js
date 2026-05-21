/**
 * chart-export.js - Export-as-PNG / SVG / CSV button group for Plotly charts.
 *
 * Public API (window.PFM.chartExport):
 *   attach(plotlyDiv, opts)         -> { detach }
 *   exportPng(plotlyDiv, opts)      -> Promise<Blob>
 *   exportSvg(plotlyDiv, opts)      -> Promise<Blob>
 *   exportCsv(plotlyDiv, opts)      -> Promise<Blob>
 *   autoAttachAll()                 -> scans `.chart-card` and attaches to any
 *                                       Plotly div inside (idempotent)
 *
 * Behavior:
 *   - Button group appears top-right of `.chart-card` on hover (CSS handles
 *     opacity 0 -> 0.85).
 *   - Three buttons: PNG, SVG, CSV (data only).
 *   - Filename: derived from chart title (Plotly layout.title.text), parent
 *     `.chart-card__title` text, or div ID; kebab-cased + ISO timestamp.
 *   - Triggers download via an invisible `<a download>` click.
 *
 * Mount: <script src="/js/chart-export.js" defer></script>
 *        <link rel="stylesheet" href="/css/chart-export.css">
 *   The module self-installs a MutationObserver and attaches automatically.
 *
 * Detection of a Plotly div: any element with `._fullLayout` (set by Plotly
 * after the first plot) OR with class `js-plotly-plot`. We poll briefly on
 * attach because some charts mount after the card.
 */
(function () {
  'use strict';

  // ---------- helpers ----------

  function _getPlotly() {
    return (typeof window !== 'undefined' && window.Plotly) || null;
  }

  function _kebab(s) {
    if (!s) return '';
    return String(s)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 80);
  }

  function _isoStamp() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return (
      d.getFullYear() +
      pad(d.getMonth() + 1) +
      pad(d.getDate()) +
      '-' +
      pad(d.getHours()) +
      pad(d.getMinutes()) +
      pad(d.getSeconds())
    );
  }

  function _isPlotly(div) {
    if (!div || div.nodeType !== 1) return false;
    if (div._fullLayout) return true;
    if (div.classList && div.classList.contains('js-plotly-plot')) return true;
    return false;
  }

  function _findPlotlyDivInCard(card) {
    if (!card) return null;
    if (_isPlotly(card)) return card;
    const candidates = card.querySelectorAll('.js-plotly-plot, .chart, .plotly-graph-div, [id]');
    for (const el of candidates) {
      if (_isPlotly(el)) return el;
    }
    return null;
  }

  function _resolveTitle(div) {
    try {
      const layout = div && div.layout;
      if (layout && layout.title) {
        if (typeof layout.title === 'string') return layout.title;
        if (layout.title.text) return layout.title.text;
      }
      const fl = div && div._fullLayout;
      if (fl && fl.title && fl.title.text) return fl.title.text;
    } catch (_) {}
    const card = div && div.closest && div.closest('.chart-card');
    if (card) {
      const t = card.querySelector('.chart-card__title, .chart-card__header h4, .chart-card__header h3');
      if (t && t.textContent) return t.textContent.trim();
    }
    return (div && (div.id || '')) || 'chart';
  }

  function _filename(div, ext) {
    const base = _kebab(_resolveTitle(div)) || 'chart';
    return base + '-' + _isoStamp() + '.' + ext;
  }

  function _triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.rel = 'noopener';
    a.style.display = 'none';
    document.body.appendChild(a);
    try {
      a.click();
    } finally {
      setTimeout(() => {
        try {
          document.body.removeChild(a);
        } catch (_) {}
        URL.revokeObjectURL(url);
      }, 60);
    }
  }

  function _dataUrlToBlob(dataUrl) {
    // Plotly.toImage returns "data:image/png;base64,...." or an SVG data URI.
    if (!dataUrl || typeof dataUrl !== 'string') {
      return new Blob([], { type: 'application/octet-stream' });
    }
    const commaIdx = dataUrl.indexOf(',');
    if (commaIdx < 0) return new Blob([dataUrl]);
    const meta = dataUrl.slice(5, commaIdx);
    const payload = dataUrl.slice(commaIdx + 1);
    const isBase64 = /;base64$/i.test(meta);
    const mime = meta.replace(/;base64$/i, '') || 'application/octet-stream';
    if (isBase64) {
      try {
        const bin = atob(payload);
        const len = bin.length;
        const buf = new Uint8Array(len);
        for (let i = 0; i < len; i++) buf[i] = bin.charCodeAt(i);
        return new Blob([buf], { type: mime });
      } catch (e) {
        return new Blob([payload], { type: mime });
      }
    }
    return new Blob([decodeURIComponent(payload)], { type: mime });
  }

  // ---------- image exports ----------

  function _toImage(div, format, opts) {
    const Plotly = _getPlotly();
    if (!Plotly || !Plotly.toImage) {
      return Promise.reject(new Error('Plotly.toImage unavailable'));
    }
    const o = opts || {};
    const w = o.width || (div && div.clientWidth) || 800;
    const h = o.height || (div && div.clientHeight) || 480;
    const scale = o.scale || 2;
    return Plotly.toImage(div, { format: format, width: w, height: h, scale: scale });
  }

  function exportPng(div, opts) {
    return _toImage(div, 'png', opts).then((url) => _dataUrlToBlob(url));
  }

  function exportSvg(div, opts) {
    return _toImage(div, 'svg', opts).then((url) => _dataUrlToBlob(url));
  }

  // ---------- CSV export ----------

  function _csvEscape(v) {
    if (v === null || v === undefined) return '';
    let s;
    if (v instanceof Date) {
      s = v.toISOString();
    } else if (typeof v === 'number') {
      s = Number.isFinite(v) ? String(v) : '';
    } else {
      s = String(v);
    }
    if (s.indexOf('"') >= 0 || s.indexOf(',') >= 0 || s.indexOf('\n') >= 0 || s.indexOf('\r') >= 0) {
      return '"' + s.replace(/"/g, '""') + '"';
    }
    return s;
  }

  function _traceLabel(trace, idx) {
    if (trace && trace.name) return String(trace.name);
    if (trace && trace.legendgroup) return String(trace.legendgroup);
    return 'trace_' + (idx + 1);
  }

  function _buildCsv(div) {
    const traces = (div && div.data) || [];
    if (!traces.length) return 'x\n';

    // Single-trace fast path.
    if (traces.length === 1) {
      const t = traces[0];
      const xs = t.x || [];
      const ys = t.y || [];
      const yLabel = _traceLabel(t, 0);
      const rows = ['x,' + _csvEscape(yLabel)];
      const len = Math.max(xs.length, ys.length);
      for (let i = 0; i < len; i++) {
        rows.push(_csvEscape(xs[i]) + ',' + _csvEscape(ys[i]));
      }
      return rows.join('\n') + '\n';
    }

    // Multi-trace: align by x using insertion order (stable).
    // Build canonical x index from the union of all trace x arrays.
    const xKey = new Map(); // serialized x -> { idx, raw }
    const xOrder = [];
    for (const t of traces) {
      const xs = t.x || [];
      for (const x of xs) {
        const k = x instanceof Date ? x.toISOString() : String(x);
        if (!xKey.has(k)) {
          xKey.set(k, { idx: xOrder.length, raw: x });
          xOrder.push(k);
        }
      }
    }

    const headers = ['x'];
    const cols = [];
    traces.forEach((t, i) => {
      headers.push(_traceLabel(t, i));
      const map = new Map();
      const xs = t.x || [];
      const ys = t.y || [];
      const n = Math.min(xs.length, ys.length);
      for (let j = 0; j < n; j++) {
        const k = xs[j] instanceof Date ? xs[j].toISOString() : String(xs[j]);
        map.set(k, ys[j]);
      }
      cols.push(map);
    });

    const out = [headers.map(_csvEscape).join(',')];
    for (const k of xOrder) {
      const raw = xKey.get(k).raw;
      const row = [_csvEscape(raw)];
      for (const m of cols) {
        row.push(_csvEscape(m.has(k) ? m.get(k) : ''));
      }
      out.push(row.join(','));
    }
    return out.join('\n') + '\n';
  }

  function exportCsv(div, _opts) {
    try {
      const text = _buildCsv(div);
      return Promise.resolve(new Blob([text], { type: 'text/csv;charset=utf-8' }));
    } catch (e) {
      return Promise.reject(e);
    }
  }

  // ---------- UI: button group ----------

  const ATTACHED_FLAG = '__pfmChartExportAttached';

  function _makeBtn(label, title, onClick) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'pfm-chart-export__btn';
    b.setAttribute('data-export', label.toLowerCase());
    b.setAttribute('aria-label', title);
    b.title = title;
    b.textContent = label;
    b.addEventListener('click', (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      try {
        b.classList.add('is-busy');
        Promise.resolve(onClick()).finally(() => {
          setTimeout(() => b.classList.remove('is-busy'), 300);
        });
      } catch (_) {
        b.classList.remove('is-busy');
      }
    });
    return b;
  }

  function _resolveHost(plotlyDiv) {
    // Buttons are absolutely positioned inside the .chart-card if there is one,
    // otherwise inside the plotly div's parent.
    if (!plotlyDiv) return null;
    const card = plotlyDiv.closest && plotlyDiv.closest('.chart-card');
    if (card) return card;
    return plotlyDiv.parentElement || plotlyDiv;
  }

  function attach(plotlyDiv, opts) {
    if (!plotlyDiv) return { detach() {} };
    if (plotlyDiv[ATTACHED_FLAG]) {
      return { detach: plotlyDiv[ATTACHED_FLAG].detach };
    }

    const o = opts || {};
    const host = _resolveHost(plotlyDiv);
    if (!host) return { detach() {} };

    // Ensure host is a positioning ancestor.
    const cs = window.getComputedStyle(host);
    if (cs && cs.position === 'static') {
      host.style.position = 'relative';
    }
    host.classList.add('pfm-chart-export-host');

    const group = document.createElement('div');
    group.className = 'pfm-chart-export';
    group.setAttribute('role', 'group');
    group.setAttribute('aria-label', 'Export chart');

    const tryDownload = (ext, blobPromise) => {
      return blobPromise
        .then((blob) => {
          _triggerDownload(blob, _filename(plotlyDiv, ext));
        })
        .catch((err) => {
          console.warn('[chart-export] ' + ext + ' failed', err);
        });
    };

    const png = _makeBtn('PNG', 'Export as PNG', () =>
      tryDownload('png', exportPng(plotlyDiv, o)),
    );
    const svg = _makeBtn('SVG', 'Export as SVG', () =>
      tryDownload('svg', exportSvg(plotlyDiv, o)),
    );
    const csv = _makeBtn('CSV', 'Export data as CSV', () =>
      tryDownload('csv', exportCsv(plotlyDiv, o)),
    );

    group.appendChild(png);
    group.appendChild(svg);
    group.appendChild(csv);
    host.appendChild(group);

    const detach = () => {
      try {
        if (group.parentNode) group.parentNode.removeChild(group);
      } catch (_) {}
      try {
        delete plotlyDiv[ATTACHED_FLAG];
      } catch (_) {
        plotlyDiv[ATTACHED_FLAG] = null;
      }
    };

    plotlyDiv[ATTACHED_FLAG] = { detach, group };
    return { detach };
  }

  // ---------- auto-attach ----------

  function _scanCard(card) {
    if (!card || card.nodeType !== 1) return;
    if (card.dataset && card.dataset.chartExport === 'off') return;
    const div = _findPlotlyDivInCard(card);
    if (!div) {
      // Re-try briefly; Plotly might not have rendered yet.
      let tries = 0;
      const iv = setInterval(() => {
        tries += 1;
        const d = _findPlotlyDivInCard(card);
        if (d) {
          clearInterval(iv);
          if (!d[ATTACHED_FLAG]) attach(d, {});
        } else if (tries > 20) {
          // ~10s; give up silently.
          clearInterval(iv);
        }
      }, 500);
      return;
    }
    if (!div[ATTACHED_FLAG]) attach(div, {});
  }

  function autoAttachAll() {
    const cards = document.querySelectorAll('.chart-card');
    cards.forEach(_scanCard);
  }

  let _observer = null;
  function _installObserver() {
    if (_observer || typeof MutationObserver === 'undefined') return;
    _observer = new MutationObserver((muts) => {
      for (const m of muts) {
        if (!m.addedNodes) continue;
        for (const n of m.addedNodes) {
          if (n.nodeType !== 1) continue;
          if (n.classList && n.classList.contains('chart-card')) {
            _scanCard(n);
          } else if (n.querySelectorAll) {
            const inner = n.querySelectorAll('.chart-card');
            inner.forEach(_scanCard);
          }
        }
      }
    });
    _observer.observe(document.body, { childList: true, subtree: true });
  }

  function _onReady(fn) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn, { once: true });
    } else {
      fn();
    }
  }

  // ---------- public API ----------

  const api = {
    attach: attach,
    exportPng: exportPng,
    exportSvg: exportSvg,
    exportCsv: exportCsv,
    autoAttachAll: autoAttachAll,
  };

  window.PFM = window.PFM || {};
  window.PFM.chartExport = api;

  _onReady(() => {
    try {
      autoAttachAll();
      _installObserver();
    } catch (e) {
      console.warn('[chart-export] init failed', e);
    }
  });
})();
