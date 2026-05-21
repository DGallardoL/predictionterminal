/**
 * result-pinner.js - Regression-fit pinboard.
 *
 * Saves up to 12 fits to localStorage and renders a right-side slide-in
 * drawer so users can compare betas/SE/p across two fits side-by-side.
 *
 * Public API (window.PFM.pinboard):
 *   add(result)              -> pinId
 *   remove(pinId)            -> bool
 *   list()                   -> Pin[]
 *   compare(pinIdA, pinIdB)  -> renders side-by-side card
 *   exportJSON()             -> string
 *   open()                   -> open drawer
 *   close()                  -> close drawer
 *
 * localStorage key: 'pfm:pinboard:v1'
 * Schema (per pin):
 *   { id: string, ts: number, ticker: string, factors: string[],
 *     R2: number|null, betas: [{name, beta, se, p}], note: string }
 *
 * Mount: <script src="/js/result-pinner.js" defer></script>
 *        <link rel="stylesheet" href="/css/result-pinner.css">
 * No HTML mount needed - the drawer is injected on init.
 *
 * Dispatches/listens:
 *   listens 'pfm:fit-complete' (CustomEvent, detail.result = fit response)
 *   dispatches 'pfm:pinboard:change' after add/remove/note
 */
(function () {
  'use strict';

  const STORAGE_KEY = 'pfm:pinboard:v1';
  const MAX_PINS = 12;
  const PIN_BTN_TTL_MS = 8000;
  const SIG_DELTA_BETA = 0.25;   // fractional difference threshold
  const SIG_DELTA_P = 0.05;      // p-value gap threshold

  // ---------- storage ----------
  function _read() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const arr = JSON.parse(raw);
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      console.warn('[pinboard] read failed', e);
      return [];
    }
  }

  function _write(arr) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(arr));
    } catch (e) {
      console.warn('[pinboard] write failed', e);
    }
  }

  function _genId() {
    return 'pin_' + Date.now().toString(36) + '_' +
           Math.random().toString(36).slice(2, 8);
  }

  function _emit() {
    try {
      window.dispatchEvent(new CustomEvent('pfm:pinboard:change',
        { detail: { pins: list() } }));
    } catch (_) { /* noop */ }
  }

  // ---------- shape extraction ----------
  function _extractPin(result) {
    if (!result || typeof result !== 'object') return null;
    const ticker = result.ticker || result.symbol || result.target || '—';
    const factors = Array.isArray(result.factors)
      ? result.factors.map((f) => (typeof f === 'string' ? f : (f && (f.slug || f.id || f.name)) || ''))
      : (Array.isArray(result.factor_slugs) ? result.factor_slugs.slice() : []);
    let R2 = null;
    if (typeof result.r_squared === 'number') R2 = result.r_squared;
    else if (typeof result.R2 === 'number') R2 = result.R2;
    else if (result.metrics && typeof result.metrics.r_squared === 'number') R2 = result.metrics.r_squared;

    let betas = [];
    if (Array.isArray(result.coefficients)) {
      betas = result.coefficients.map((c) => ({
        name: c.name || c.factor || c.slug || '?',
        beta: _num(c.beta ?? c.coef ?? c.coefficient),
        se: _num(c.se ?? c.std_err ?? c.standard_error),
        p: _num(c.p ?? c.pvalue ?? c.p_value),
      }));
    } else if (Array.isArray(result.betas)) {
      betas = result.betas.map((c) => ({
        name: c.name || c.factor || c.slug || '?',
        beta: _num(c.beta),
        se: _num(c.se),
        p: _num(c.p),
      }));
    }

    return {
      id: _genId(),
      ts: Date.now(),
      ticker: String(ticker),
      factors: factors.filter(Boolean),
      R2: R2,
      betas: betas,
      note: '',
    };
  }

  function _num(v) {
    if (v === null || v === undefined) return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  // ---------- core API ----------
  function add(result) {
    const pin = _extractPin(result);
    if (!pin) return null;
    const all = _read();
    all.push(pin);
    // FIFO evict oldest
    while (all.length > MAX_PINS) all.shift();
    _write(all);
    _emit();
    _renderDrawerList();
    return pin.id;
  }

  function remove(pinId) {
    const all = _read();
    const next = all.filter((p) => p.id !== pinId);
    if (next.length === all.length) return false;
    _write(next);
    _emit();
    _renderDrawerList();
    return true;
  }

  function list() {
    return _read();
  }

  function setNote(pinId, note) {
    const all = _read();
    const p = all.find((x) => x.id === pinId);
    if (!p) return false;
    p.note = String(note || '');
    _write(all);
    _emit();
    _renderDrawerList();
    return true;
  }

  function exportJSON() {
    return JSON.stringify(_read(), null, 2);
  }

  // ---------- compare ----------
  function compare(pinIdA, pinIdB) {
    const all = _read();
    const a = all.find((p) => p.id === pinIdA);
    const b = all.find((p) => p.id === pinIdB);
    if (!a || !b) {
      console.warn('[pinboard] compare: pin not found');
      return false;
    }
    _renderCompare(a, b);
    open();
    return true;
  }

  function _factorUnion(a, b) {
    const names = new Set();
    a.betas.forEach((c) => names.add(c.name));
    b.betas.forEach((c) => names.add(c.name));
    return Array.from(names);
  }

  function _findBeta(pin, name) {
    return pin.betas.find((c) => c.name === name) || null;
  }

  function _isSigDelta(rowA, rowB) {
    if (!rowA || !rowB) return true;
    const bA = rowA.beta, bB = rowB.beta;
    if (typeof bA === 'number' && typeof bB === 'number') {
      const denom = Math.max(Math.abs(bA), Math.abs(bB), 1e-9);
      if (Math.abs(bA - bB) / denom >= SIG_DELTA_BETA) return true;
    }
    const pA = rowA.p, pB = rowB.p;
    if (typeof pA === 'number' && typeof pB === 'number') {
      // significance flip across 0.05 OR large absolute gap
      if ((pA < 0.05) !== (pB < 0.05)) return true;
      if (Math.abs(pA - pB) >= SIG_DELTA_P) return true;
    }
    return false;
  }

  function _fmt(v, digits) {
    if (v === null || v === undefined || !Number.isFinite(v)) return '—';
    return v.toFixed(digits === undefined ? 3 : digits);
  }

  function _fmtPct(v) {
    if (v === null || v === undefined || !Number.isFinite(v)) return '—';
    return (v * 100).toFixed(1) + '%';
  }

  function _shortDate(ts) {
    try {
      const d = new Date(ts);
      const mm = String(d.getMonth() + 1).padStart(2, '0');
      const dd = String(d.getDate()).padStart(2, '0');
      const hh = String(d.getHours()).padStart(2, '0');
      const mi = String(d.getMinutes()).padStart(2, '0');
      return `${d.getFullYear()}-${mm}-${dd} ${hh}:${mi}`;
    } catch (_) { return String(ts); }
  }

  function _esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // ---------- DOM: drawer ----------
  let _drawer = null;
  let _backdrop = null;
  let _compareHost = null;
  let _listHost = null;
  let _toastEl = null;
  let _toastTimer = null;

  function _ensureDrawer() {
    if (_drawer) return;
    _backdrop = document.createElement('div');
    _backdrop.className = 'pfm-pin-backdrop';
    _backdrop.addEventListener('click', close);

    _drawer = document.createElement('aside');
    _drawer.className = 'pfm-pin-drawer';
    _drawer.setAttribute('role', 'complementary');
    _drawer.setAttribute('aria-label', 'Regression pinboard');
    _drawer.innerHTML = `
      <header class="pfm-pin-head">
        <h3>Pinboard</h3>
        <div class="pfm-pin-head-actions">
          <button type="button" class="pfm-pin-btn-mini" data-action="export" title="Copy JSON">Export</button>
          <button type="button" class="pfm-pin-btn-close" data-action="close" aria-label="Close">×</button>
        </div>
      </header>
      <div class="pfm-pin-body">
        <section class="pfm-pin-list-wrap">
          <div class="pfm-pin-meta">
            <span class="pfm-pin-count"></span>
            <span class="pfm-pin-hint">Pick A then B to compare</span>
          </div>
          <ul class="pfm-pin-list" role="listbox"></ul>
        </section>
        <section class="pfm-pin-compare" hidden></section>
      </div>
    `;
    document.body.appendChild(_backdrop);
    document.body.appendChild(_drawer);

    _listHost = _drawer.querySelector('.pfm-pin-list');
    _compareHost = _drawer.querySelector('.pfm-pin-compare');

    _drawer.addEventListener('click', _onDrawerClick);
    _renderDrawerList();
  }

  let _selectedA = null;
  let _selectedB = null;

  function _onDrawerClick(ev) {
    const t = ev.target;
    if (!(t instanceof Element)) return;
    const action = t.getAttribute('data-action');
    if (action === 'close') { close(); return; }
    if (action === 'export') {
      const json = exportJSON();
      try {
        navigator.clipboard.writeText(json);
        _toast('Pinboard JSON copied');
      } catch (_) {
        _toast('Copy failed; see console');
        console.log(json);
      }
      return;
    }
    const pinId = t.getAttribute('data-pin');
    if (action === 'unpin' && pinId) {
      remove(pinId);
      if (_selectedA === pinId) _selectedA = null;
      if (_selectedB === pinId) _selectedB = null;
      return;
    }
    if (action === 'note' && pinId) {
      const all = _read();
      const p = all.find((x) => x.id === pinId);
      if (!p) return;
      const note = window.prompt('Note for this pin:', p.note || '');
      if (note !== null) setNote(pinId, note);
      return;
    }
    if (action === 'select' && pinId) {
      if (_selectedA === pinId) { _selectedA = null; }
      else if (_selectedB === pinId) { _selectedB = null; }
      else if (!_selectedA) { _selectedA = pinId; }
      else if (!_selectedB) { _selectedB = pinId; }
      else { _selectedA = pinId; _selectedB = null; }
      _renderDrawerList();
      if (_selectedA && _selectedB) compare(_selectedA, _selectedB);
      return;
    }
    if (action === 'clear-compare') {
      _selectedA = null; _selectedB = null;
      _compareHost.hidden = true;
      _compareHost.innerHTML = '';
      _renderDrawerList();
      return;
    }
  }

  function _renderDrawerList() {
    if (!_listHost) return;
    const pins = _read().slice().sort((a, b) => b.ts - a.ts);
    const countEl = _drawer.querySelector('.pfm-pin-count');
    if (countEl) countEl.textContent = `${pins.length} / ${MAX_PINS}`;
    if (!pins.length) {
      _listHost.innerHTML = '<li class="pfm-pin-empty">No pins yet. Run a fit and use “Pin this”.</li>';
      return;
    }
    _listHost.innerHTML = pins.map((p) => {
      const flag = p.id === _selectedA ? 'A' : (p.id === _selectedB ? 'B' : '');
      const flagCls = flag ? ' is-selected slot-' + flag.toLowerCase() : '';
      const noteHtml = p.note ? `<div class="pfm-pin-note" title="${_esc(p.note)}">${_esc(p.note)}</div>` : '';
      return `
        <li class="pfm-pin-item${flagCls}" data-pin="${_esc(p.id)}">
          <button type="button" class="pfm-pin-select" data-action="select" data-pin="${_esc(p.id)}" title="Select for compare">
            <span class="pfm-pin-tick">${_esc(p.ticker)}</span>
            <span class="pfm-pin-r2">${p.R2 != null ? 'R² ' + _fmt(p.R2, 3) : 'R² —'}</span>
            <span class="pfm-pin-date">${_esc(_shortDate(p.ts))}</span>
            ${flag ? `<span class="pfm-pin-flag">${flag}</span>` : ''}
          </button>
          <div class="pfm-pin-menu">
            <details>
              <summary aria-label="Pin menu">▼</summary>
              <div class="pfm-pin-menu-body">
                <button type="button" data-action="select" data-pin="${_esc(p.id)}">Compare</button>
                <button type="button" data-action="note" data-pin="${_esc(p.id)}">Note</button>
                <button type="button" data-action="unpin" data-pin="${_esc(p.id)}">Unpin</button>
              </div>
            </details>
          </div>
          ${noteHtml}
        </li>
      `;
    }).join('');
  }

  function _renderCompare(a, b) {
    if (!_compareHost) return;
    const names = _factorUnion(a, b);
    const rows = names.map((name) => {
      const rA = _findBeta(a, name);
      const rB = _findBeta(b, name);
      const sig = _isSigDelta(rA, rB);
      const cls = sig ? ' is-sig' : '';
      const cellA = rA
        ? `<span class="b">${_fmt(rA.beta, 3)}</span><span class="se">±${_fmt(rA.se, 3)}</span><span class="p">p=${_fmt(rA.p, 3)}</span>`
        : '<span class="b muted">—</span>';
      const cellB = rB
        ? `<span class="b">${_fmt(rB.beta, 3)}</span><span class="se">±${_fmt(rB.se, 3)}</span><span class="p">p=${_fmt(rB.p, 3)}</span>`
        : '<span class="b muted">—</span>';
      return `
        <tr class="pfm-pin-cmp-row${cls}">
          <th scope="row">${_esc(name)}</th>
          <td class="cmp-a">${cellA}</td>
          <td class="cmp-b">${cellB}</td>
        </tr>
      `;
    }).join('');
    _compareHost.hidden = false;
    _compareHost.innerHTML = `
      <header class="pfm-pin-cmp-head">
        <h4>Compare</h4>
        <button type="button" class="pfm-pin-btn-mini" data-action="clear-compare">Clear</button>
      </header>
      <div class="pfm-pin-cmp-cards">
        <div class="pfm-pin-cmp-card slot-a">
          <div class="lbl">A</div>
          <div class="tk">${_esc(a.ticker)}</div>
          <div class="r2">R² ${_fmt(a.R2, 3)}</div>
          <div class="dt">${_esc(_shortDate(a.ts))}</div>
        </div>
        <div class="pfm-pin-cmp-card slot-b">
          <div class="lbl">B</div>
          <div class="tk">${_esc(b.ticker)}</div>
          <div class="r2">R² ${_fmt(b.R2, 3)}</div>
          <div class="dt">${_esc(_shortDate(b.ts))}</div>
        </div>
      </div>
      <table class="pfm-pin-cmp-table">
        <thead><tr><th>Factor</th><th>A · β / SE / p</th><th>B · β / SE / p</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="3" class="muted">No betas to compare</td></tr>'}</tbody>
      </table>
      <p class="pfm-pin-cmp-legend">Highlighted in <span class="lg-sig">orange</span> when |Δβ|/max ≥ ${SIG_DELTA_BETA.toFixed(2)}, p-significance flips, or |Δp| ≥ ${SIG_DELTA_P.toFixed(2)}.</p>
    `;
  }

  function open() {
    _ensureDrawer();
    document.body.classList.add('pfm-pin-open');
    _drawer.classList.add('is-open');
    _backdrop.classList.add('is-open');
    _renderDrawerList();
  }

  function close() {
    if (!_drawer) return;
    document.body.classList.remove('pfm-pin-open');
    _drawer.classList.remove('is-open');
    _backdrop.classList.remove('is-open');
  }

  function toggle() {
    if (!_drawer || !_drawer.classList.contains('is-open')) open();
    else close();
  }

  // ---------- "Pin this" floating button ----------
  let _pinBtn = null;
  let _pinBtnTimer = null;
  let _pendingResult = null;

  function _showPinPrompt(result) {
    _pendingResult = result;
    if (!_pinBtn) {
      _pinBtn = document.createElement('button');
      _pinBtn.type = 'button';
      _pinBtn.className = 'pfm-pin-prompt';
      _pinBtn.innerHTML = '<span class="pfm-pin-prompt-icon" aria-hidden="true">📌</span><span>Pin this fit</span>';
      _pinBtn.addEventListener('click', () => {
        if (!_pendingResult) { _hidePinPrompt(); return; }
        const id = add(_pendingResult);
        _pendingResult = null;
        _hidePinPrompt();
        if (id) {
          _toast('Pinned. Open pinboard to compare.');
          open();
        }
      });
      document.body.appendChild(_pinBtn);
    }
    _pinBtn.classList.add('is-visible');
    clearTimeout(_pinBtnTimer);
    _pinBtnTimer = setTimeout(_hidePinPrompt, PIN_BTN_TTL_MS);
  }

  function _hidePinPrompt() {
    if (_pinBtn) _pinBtn.classList.remove('is-visible');
    clearTimeout(_pinBtnTimer);
  }

  // ---------- toast ----------
  function _toast(msg) {
    if (!_toastEl) {
      _toastEl = document.createElement('div');
      _toastEl.className = 'pfm-pin-toast';
      document.body.appendChild(_toastEl);
    }
    _toastEl.textContent = msg;
    _toastEl.classList.add('is-visible');
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => _toastEl.classList.remove('is-visible'), 2400);
  }

  // ---------- init ----------
  function _init() {
    _ensureDrawer();
    window.addEventListener('pfm:fit-complete', (ev) => {
      const result = ev && ev.detail && ev.detail.result;
      if (!result) return;
      _showPinPrompt(result);
    });
    // Keyboard: 'p' toggles drawer when not inside an input
    window.addEventListener('keydown', (ev) => {
      if (ev.key !== 'p' && ev.key !== 'P') return;
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      const t = ev.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || (t.isContentEditable))) return;
      if (ev.shiftKey) { toggle(); ev.preventDefault(); }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _init, { once: true });
  } else {
    _init();
  }

  // ---------- export to window ----------
  window.PFM = window.PFM || {};
  window.PFM.pinboard = {
    add: add,
    remove: remove,
    list: list,
    compare: compare,
    exportJSON: exportJSON,
    setNote: setNote,
    open: open,
    close: close,
    toggle: toggle,
    _STORAGE_KEY: STORAGE_KEY,
    _MAX_PINS: MAX_PINS,
  };
})();
