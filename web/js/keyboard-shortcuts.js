/**
 * keyboard-shortcuts.js — T15
 *
 * Global keyboard shortcuts manager. Vanilla JS, no deps.
 *
 * Public API (window.PFM.shortcuts):
 *   register(key, handler, label)  -> registers a single-key shortcut
 *                                     `key` may be a token (e.g. "?", "/", "n", "Shift+P", "Ctrl+K", "g r")
 *                                     `g r` style strings register a 2-key Gmail-style sequence.
 *   help()                         -> opens the help sheet
 *   enable()                       -> re-enables the global listener
 *   disable()                      -> disables the global listener
 *
 * Built-in shortcuts:
 *   ?            help sheet
 *   /            focus first visible search input, else PFM.cmdk.open()
 *   g then r     switch to Regression mode
 *   g then s     switch to Strategies mode
 *   g then t     switch to Terminal mode
 *   Esc          close any open modal/sheet/drawer; dispatch pfm:escape
 *   Ctrl+K/Cmd+K command palette (PFM.cmdk.open)
 *   Shift+P      pinboard toggle (PFM.pinboard.toggle)
 *   n            "next" in list contexts ([data-list])
 *   p            "previous" in list contexts ([data-list])
 *   t            theme toggle (PFM.theme.toggle)
 *
 * Mount:
 *   <link rel="stylesheet" href="/css/shortcuts-help.css">
 *   <script src="/js/keyboard-shortcuts.js" defer></script>
 *
 * Events:
 *   dispatches CustomEvent 'pfm:escape' on Esc.
 *   dispatches CustomEvent 'pfm:switch-mode' with detail { mode: 'regression'|'strategies'|'terminal' }.
 */
(function () {
  'use strict';

  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------
  const SEQUENCE_TIMEOUT_MS = 1000;
  const _registry = new Map();      // normalized-key  -> { handler, label, key }
  const _sequences = new Map();     // "g r" style     -> { handler, label, key }
  const _seqPrefixes = new Set();   // first chars of any sequence (e.g. "g")
  let _seqState = null;             // { prefix: 'g', timer: setTimeout id }
  let _enabled = true;
  let _helpEl = null;

  // -------------------------------------------------------------------------
  // Key normalization
  // -------------------------------------------------------------------------
  function _isSequenceKey(key) {
    return typeof key === 'string' && /\s/.test(key.trim());
  }

  function _normalizeSingle(key) {
    if (typeof key !== 'string') return '';
    // Build canonical "mod+key" form. Mods sorted: Ctrl, Alt, Shift, Meta.
    const parts = key.split('+').map((s) => s.trim()).filter(Boolean);
    if (parts.length === 0) return '';
    const mods = new Set();
    let main = '';
    for (const p of parts) {
      const low = p.toLowerCase();
      if (low === 'ctrl' || low === 'control') mods.add('ctrl');
      else if (low === 'alt' || low === 'option') mods.add('alt');
      else if (low === 'shift') mods.add('shift');
      else if (low === 'meta' || low === 'cmd' || low === 'command' || low === 'super') mods.add('meta');
      else {
        // last non-mod wins
        main = p.length === 1 ? p.toLowerCase() : p;
      }
    }
    if (!main) main = parts[parts.length - 1];
    const ordered = [];
    if (mods.has('ctrl')) ordered.push('ctrl');
    if (mods.has('alt')) ordered.push('alt');
    if (mods.has('shift')) ordered.push('shift');
    if (mods.has('meta')) ordered.push('meta');
    ordered.push(main);
    return ordered.join('+');
  }

  function _normalizeSequence(key) {
    return key
      .trim()
      .split(/\s+/)
      .map((tok) => tok.toLowerCase())
      .join(' ');
  }

  // Convert a KeyboardEvent into a canonical "mods+key" token.
  function _eventToToken(ev) {
    const k = ev.key;
    if (!k) return '';
    // Treat ' ' (Space) explicitly; otherwise lowercase single chars.
    let main;
    if (k === ' ') main = 'space';
    else if (k.length === 1) main = k.toLowerCase();
    else main = k; // 'Escape', 'ArrowUp', etc.
    const parts = [];
    if (ev.ctrlKey) parts.push('ctrl');
    if (ev.altKey) parts.push('alt');
    // Only include shift when the main key is a non-letter or letter is uppercase
    // (avoids "Shift+a" when the user types "A").
    if (ev.shiftKey && (main.length !== 1 || !/^[a-z]$/i.test(main))) parts.push('shift');
    if (ev.metaKey) parts.push('meta');
    // Special case for uppercase letters typed with shift: treat as "shift+<lower>"
    if (ev.shiftKey && main.length === 1 && /^[a-z]$/.test(main)) {
      parts.push('shift');
    }
    parts.push(main);
    return parts.join('+');
  }

  // -------------------------------------------------------------------------
  // Should-ignore detection
  // -------------------------------------------------------------------------
  function _isEditableTarget(target) {
    if (!target || target.nodeType !== 1) return false;
    if (target.closest && target.closest('[data-shortcut-passthrough]')) return false;
    const tag = (target.tagName || '').toUpperCase();
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
    if (target.isContentEditable) return true;
    return false;
  }

  // -------------------------------------------------------------------------
  // Help sheet rendering
  // -------------------------------------------------------------------------
  function _ensureHelpEl() {
    if (_helpEl && document.body.contains(_helpEl)) return _helpEl;
    const root = document.createElement('div');
    root.className = 'pfm-shortcuts-help';
    root.setAttribute('data-open', 'false');
    root.setAttribute('role', 'dialog');
    root.setAttribute('aria-modal', 'true');
    root.setAttribute('aria-label', 'Keyboard shortcuts');
    root.innerHTML = `
      <div class="pfm-shortcuts-help__backdrop" data-close></div>
      <div class="pfm-shortcuts-help__sheet" role="document">
        <header class="pfm-shortcuts-help__header">
          <h2 class="pfm-shortcuts-help__title">Keyboard shortcuts</h2>
          <button type="button" class="pfm-shortcuts-help__close" data-close aria-label="Close">&times;</button>
        </header>
        <div class="pfm-shortcuts-help__body">
          <div class="pfm-shortcuts-help__grid" data-grid></div>
        </div>
        <footer class="pfm-shortcuts-help__footer">
          <span class="pfm-shortcuts-help__hint">Press <kbd>Esc</kbd> to close</span>
        </footer>
      </div>
    `;
    root.addEventListener('click', (ev) => {
      const t = ev.target;
      if (t && (t.hasAttribute && t.hasAttribute('data-close'))) {
        _closeHelp();
      }
    });
    document.body.appendChild(root);
    _helpEl = root;
    return root;
  }

  function _renderHelpRows() {
    const root = _ensureHelpEl();
    const grid = root.querySelector('[data-grid]');
    if (!grid) return;
    const rows = [];
    // Sequences first (more memorable)
    for (const [seq, entry] of _sequences.entries()) {
      rows.push({ key: seq, label: entry.label || '' });
    }
    for (const [key, entry] of _registry.entries()) {
      rows.push({ key, label: entry.label || '' });
    }
    grid.innerHTML = rows.map((r) => {
      return `
        <div class="pfm-shortcuts-help__row">
          <div class="pfm-shortcuts-help__keys">${_formatKey(r.key)}</div>
          <div class="pfm-shortcuts-help__label">${_escapeHtml(r.label)}</div>
        </div>
      `;
    }).join('');
  }

  function _formatKey(key) {
    // Render each token as <kbd>, separating with "then" for sequences and "+" for combos.
    if (_isSequenceKey(key)) {
      const tokens = key.split(/\s+/);
      return tokens.map((t) => `<kbd>${_escapeHtml(_prettyToken(t))}</kbd>`).join(' <span class="pfm-shortcuts-help__sep">then</span> ');
    }
    const parts = key.split('+');
    return parts.map((t) => `<kbd>${_escapeHtml(_prettyToken(t))}</kbd>`).join(' <span class="pfm-shortcuts-help__plus">+</span> ');
  }

  function _prettyToken(t) {
    const low = t.toLowerCase();
    if (low === 'ctrl') return 'Ctrl';
    if (low === 'alt') return 'Alt';
    if (low === 'shift') return 'Shift';
    if (low === 'meta') return _isMac() ? 'Cmd' : 'Win';
    if (low === 'escape') return 'Esc';
    if (low === 'space') return 'Space';
    if (low.length === 1) return low.toUpperCase();
    return t;
  }

  function _isMac() {
    try {
      return /Mac|iPhone|iPad/.test(navigator.platform || '');
    } catch (_e) {
      return false;
    }
  }

  function _escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function _openHelp() {
    const root = _ensureHelpEl();
    _renderHelpRows();
    root.setAttribute('data-open', 'true');
    // Trap focus minimally — focus close button.
    const close = root.querySelector('.pfm-shortcuts-help__close');
    if (close) {
      try { close.focus(); } catch (_e) { /* noop */ }
    }
  }

  function _closeHelp() {
    if (_helpEl) _helpEl.setAttribute('data-open', 'false');
  }

  function _helpIsOpen() {
    return !!(_helpEl && _helpEl.getAttribute('data-open') === 'true');
  }

  // -------------------------------------------------------------------------
  // Built-in handlers
  // -------------------------------------------------------------------------
  function _switchMode(mode) {
    try {
      document.dispatchEvent(new CustomEvent('pfm:switch-mode', { detail: { mode } }));
    } catch (_e) { /* noop */ }
  }

  function _focusSearchOrCmdk() {
    // Find a visible search input — common selectors.
    const sels = [
      'input[type="search"]',
      'input[role="searchbox"]',
      'input[data-role="search"]',
      'input[name="q"]',
      'input[name="search"]',
      'input[placeholder*="Search" i]',
    ];
    for (const sel of sels) {
      const inputs = document.querySelectorAll(sel);
      for (const el of inputs) {
        if (_isVisible(el)) {
          try { el.focus(); el.select && el.select(); } catch (_e) { /* noop */ }
          return true;
        }
      }
    }
    // Fallback: command palette.
    const cmdk = window.PFM && window.PFM.cmdk;
    if (cmdk && typeof cmdk.open === 'function') {
      try { cmdk.open(); } catch (_e) { /* noop */ }
      return true;
    }
    return false;
  }

  function _isVisible(el) {
    if (!el || !el.getBoundingClientRect) return false;
    if (el.disabled) return false;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    const style = window.getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none') return false;
    return true;
  }

  function _openCmdk() {
    const cmdk = window.PFM && window.PFM.cmdk;
    if (cmdk && typeof cmdk.open === 'function') {
      try { cmdk.open(); } catch (_e) { /* noop */ }
    }
  }

  function _togglePinboard() {
    const pb = window.PFM && window.PFM.pinboard;
    if (pb && typeof pb.toggle === 'function') {
      try { pb.toggle(); } catch (_e) { /* noop */ }
    } else if (pb) {
      // Fallback: open/close pair.
      try {
        // Heuristic: if there is an open drawer, close it.
        const opened = document.querySelector('.pfm-pinboard[data-open="true"], #pfm-pinboard[data-open="true"]');
        if (opened && typeof pb.close === 'function') pb.close();
        else if (typeof pb.open === 'function') pb.open();
      } catch (_e) { /* noop */ }
    }
  }

  function _toggleTheme() {
    const th = window.PFM && window.PFM.theme;
    if (th && typeof th.toggle === 'function') {
      try { th.toggle(); } catch (_e) { /* noop */ }
    }
  }

  function _closeAllModals() {
    // Help sheet first (we own this one).
    if (_helpIsOpen()) {
      _closeHelp();
      return true;
    }
    let closed = false;
    // Generic patterns: anything with [data-open=true] inside known shells.
    const sels = [
      '.modal[data-open="true"]',
      '.pfm-modal[data-open="true"]',
      '[role="dialog"][data-open="true"]',
      '.pfm-cmdk-root[data-open="true"]',
      '.alphahub-fs-overlay[data-open="true"]',
      '.pfm-pinboard[data-open="true"]',
      '#pfm-pinboard[data-open="true"]',
      '.drawer[data-open="true"]',
      '.sheet[data-open="true"]',
    ];
    for (const sel of sels) {
      document.querySelectorAll(sel).forEach((el) => {
        el.setAttribute('data-open', 'false');
        closed = true;
      });
    }
    return closed;
  }

  function _listNav(direction) {
    // Operate on the active (focused) [data-list] or the first visible one.
    let list = null;
    const active = document.activeElement;
    if (active && active.closest) list = active.closest('[data-list]');
    if (!list) {
      const lists = document.querySelectorAll('[data-list]');
      for (const el of lists) {
        if (_isVisible(el)) { list = el; break; }
      }
    }
    if (!list) return false;
    const items = Array.from(list.querySelectorAll('[data-list-item]'));
    if (items.length === 0) return false;
    const currentIdx = items.findIndex((it) => it.getAttribute('aria-selected') === 'true' || it.classList.contains('is-active') || it === active);
    let nextIdx;
    if (currentIdx < 0) {
      nextIdx = direction > 0 ? 0 : items.length - 1;
    } else {
      nextIdx = (currentIdx + direction + items.length) % items.length;
    }
    items.forEach((it) => {
      it.removeAttribute('aria-selected');
      it.classList.remove('is-active');
    });
    const target = items[nextIdx];
    target.setAttribute('aria-selected', 'true');
    target.classList.add('is-active');
    if (typeof target.focus === 'function') {
      try { target.focus({ preventScroll: false }); } catch (_e) { /* noop */ }
    }
    if (target.scrollIntoView) {
      try { target.scrollIntoView({ block: 'nearest' }); } catch (_e) { /* noop */ }
    }
    return true;
  }

  // -------------------------------------------------------------------------
  // Sequence machinery
  // -------------------------------------------------------------------------
  function _resetSequence() {
    if (_seqState && _seqState.timer) clearTimeout(_seqState.timer);
    _seqState = null;
  }

  function _startSequence(prefix) {
    _resetSequence();
    _seqState = {
      prefix,
      timer: setTimeout(() => { _seqState = null; }, SEQUENCE_TIMEOUT_MS),
    };
  }

  // -------------------------------------------------------------------------
  // Main listener
  // -------------------------------------------------------------------------
  function _onKeyDown(ev) {
    if (!_enabled) return;
    // Special-case: Esc always works, even inside text inputs.
    if (ev.key === 'Escape') {
      const did = _closeAllModals();
      try { document.dispatchEvent(new CustomEvent('pfm:escape')); } catch (_e) { /* noop */ }
      if (did) {
        ev.preventDefault();
      }
      return;
    }
    // Allow built-in nav keys to pass through inside text inputs unless override.
    if (_isEditableTarget(ev.target)) return;

    const token = _eventToToken(ev);
    if (!token) return;

    // Sequence in progress?
    if (_seqState) {
      // Only consider sequence completion for single-letter / printable keys (no mods).
      if (!ev.ctrlKey && !ev.altKey && !ev.metaKey) {
        const second = ev.key && ev.key.length === 1 ? ev.key.toLowerCase() : '';
        if (second) {
          const seqKey = `${_seqState.prefix} ${second}`;
          const entry = _sequences.get(seqKey);
          _resetSequence();
          if (entry && typeof entry.handler === 'function') {
            ev.preventDefault();
            try { entry.handler(ev); } catch (e) { /* swallow */ console && console.error && console.error('shortcut handler error', e); }
            return;
          }
          // No matching sequence — fall through to normal handling for `second`.
        } else {
          _resetSequence();
        }
      } else {
        _resetSequence();
      }
    }

    // Does this single key open a sequence prefix?
    if (!ev.ctrlKey && !ev.altKey && !ev.metaKey && ev.key && ev.key.length === 1) {
      const lower = ev.key.toLowerCase();
      if (_seqPrefixes.has(lower)) {
        _startSequence(lower);
        ev.preventDefault();
        return;
      }
    }

    // Single-key lookup.
    const entry = _registry.get(token);
    if (entry && typeof entry.handler === 'function') {
      ev.preventDefault();
      try { entry.handler(ev); } catch (e) { /* swallow */ console && console.error && console.error('shortcut handler error', e); }
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------
  function register(key, handler, label) {
    if (typeof key !== 'string' || typeof handler !== 'function') return false;
    if (_isSequenceKey(key)) {
      const norm = _normalizeSequence(key);
      const parts = norm.split(' ');
      if (parts.length !== 2) return false;
      _sequences.set(norm, { handler, label: label || '', key: norm });
      _seqPrefixes.add(parts[0]);
      return true;
    }
    const norm = _normalizeSingle(key);
    if (!norm) return false;
    _registry.set(norm, { handler, label: label || '', key: norm });
    return true;
  }

  function help() { _openHelp(); }
  function enable() { _enabled = true; }
  function disable() { _enabled = false; }

  // -------------------------------------------------------------------------
  // Bootstrap
  // -------------------------------------------------------------------------
  window.PFM = window.PFM || {};
  window.PFM.shortcuts = { register, help, enable, disable };

  // Register built-ins.
  register('?', () => _openHelp(), 'Show this help');
  register('/', () => _focusSearchOrCmdk(), 'Focus search / open command palette');
  register('Ctrl+K', () => _openCmdk(), 'Open command palette');
  register('Meta+K', () => _openCmdk(), 'Open command palette');
  register('Shift+P', () => _togglePinboard(), 'Toggle pinboard');
  register('t', () => _toggleTheme(), 'Toggle theme');
  register('n', () => _listNav(1), 'Next item in list');
  register('p', () => _listNav(-1), 'Previous item in list');
  register('g r', () => _switchMode('regression'), 'Go to Regression mode');
  register('g s', () => _switchMode('strategies'), 'Go to Strategies mode');
  register('g t', () => _switchMode('terminal'), 'Go to Terminal mode');

  // Attach single document-level listener.
  document.addEventListener('keydown', _onKeyDown, true);
})();
