/**
 * copy-as-curl.js - "Copy as cURL" buttons for every panel with a known
 *                   backing API request. (Task W11-08, wave-11.)
 *
 * Public API (window.PFM.curl):
 *   attach(targetEl, requestDescriptor)
 *       - Mounts a 24x24 ghost icon-button into `targetEl` (typically a
 *         `.dc-card` or `.chart-card` host). Clicking copies the cURL
 *         string for `requestDescriptor` directly to the clipboard.
 *
 *   copyFromLastFetch()
 *       - Opens a dropdown listing the last 10 (method, url, body)
 *         tuples remembered from the global `fetch` wrap. Each row
 *         has a copy-icon that copies the cURL for that request.
 *
 *   record(method, url, body, headers)
 *       - Manually push an entry into the recent-requests ring (used by
 *         non-fetch transports like SSE EventSource subscribers).
 *
 *   build(requestDescriptor) -> string
 *       - Pure helper. Returns the cURL command without copying.
 *
 * Request descriptor schema:
 *   { method: string,          // 'GET' | 'POST' | 'PUT' | 'DELETE' ...
 *     url:    string,          // absolute or relative
 *     body:   any,             // object/array -> JSON.stringify; string passthrough
 *     headers: Record<string, string> | undefined
 *   }
 *
 * Auto-detect mode:
 *   The module installs a non-destructive wrap around `window.fetch`. Every
 *   call adds a `{ method, url, body, ts }` entry to a 10-deep in-memory
 *   ring buffer. The wrap is reentrancy-safe (no infinite recursion) and
 *   defaults to the un-wrapped fetch if anything throws.
 *
 * Mount:
 *   <script src="/js/copy-as-curl.js" defer></script>
 *   <link  rel="stylesheet" href="/css/copy-as-curl.css">
 *
 *   No HTML mount needed. The floating dropdown root + "Copied" badge are
 *   injected on first use.
 *
 * Coordination note:
 *   This file is sole-owner of the `.pfm-curl-*` CSS class namespace.
 *   The companion stylesheet (`web/css/copy-as-curl.css`) provides all
 *   visual styling; this script only sets minimal inline styles for the
 *   dynamic dropdown position.
 */
(function () {
  'use strict';

  // ---------------- constants ----------------
  const RING_MAX = 10;
  const BADGE_TTL_MS = 1500;
  const DEFAULT_HEADERS_CT = 'application/json';

  // ---------------- state ----------------
  /** @type {{method:string,url:string,body:any,ts:number}[]} */
  const ring = [];
  let dropdownEl = null;
  let badgeEl = null;
  let badgeTimer = null;

  // ---------------- helpers ----------------
  function _absUrl(url) {
    if (!url) return '';
    try {
      // If `url` is already absolute (http(s)://...) URL() keeps it.
      // Otherwise resolve against the current page origin.
      return new URL(url, window.location.origin).toString();
    } catch (_e) {
      return String(url);
    }
  }

  function _isJsonBody(body) {
    if (body == null) return false;
    if (typeof body === 'string') {
      const s = body.trim();
      return (s.startsWith('{') && s.endsWith('}')) ||
             (s.startsWith('[') && s.endsWith(']'));
    }
    return typeof body === 'object';
  }

  function _prettyBody(body) {
    if (body == null) return '';
    if (typeof body === 'string') {
      if (_isJsonBody(body)) {
        try {
          return JSON.stringify(JSON.parse(body), null, 2);
        } catch (_e) {
          return body;
        }
      }
      return body;
    }
    try {
      return JSON.stringify(body, null, 2);
    } catch (_e) {
      return String(body);
    }
  }

  function _shellEscape(s) {
    // Single-quote escape for POSIX shell. Inside single quotes everything
    // is literal except the single quote itself which we close-escape-reopen.
    if (s == null) return "''";
    return "'" + String(s).replace(/'/g, "'\\''") + "'";
  }

  function _build(desc) {
    if (!desc || !desc.url) return '';
    const method = (desc.method || 'GET').toUpperCase();
    const url = _absUrl(desc.url);
    const parts = ['curl'];
    if (method !== 'GET') {
      parts.push('-X', method);
    }
    parts.push(_shellEscape(url));

    const headers = Object.assign({}, desc.headers || {});
    const hasBody = desc.body != null && desc.body !== '';
    if (hasBody && !Object.keys(headers).some(h => h.toLowerCase() === 'content-type')) {
      headers['Content-Type'] = DEFAULT_HEADERS_CT;
    }
    Object.keys(headers).forEach(k => {
      parts.push('-H', _shellEscape(k + ': ' + headers[k]));
    });

    if (hasBody) {
      parts.push('-d', _shellEscape(_prettyBody(desc.body)));
    }
    return parts.join(' ');
  }

  function _writeClipboard(text) {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        return navigator.clipboard.writeText(text);
      }
    } catch (_e) {
      // fall through to fallback
    }
    // Fallback for non-secure contexts: hidden textarea + execCommand.
    return new Promise((resolve, reject) => {
      try {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.top = '-1000px';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand('copy');
        document.body.removeChild(ta);
        ok ? resolve() : reject(new Error('execCommand returned false'));
      } catch (e) {
        reject(e);
      }
    });
  }

  function _ensureBadge() {
    if (badgeEl) return badgeEl;
    badgeEl = document.createElement('div');
    badgeEl.className = 'pfm-curl-badge';
    badgeEl.setAttribute('role', 'status');
    badgeEl.setAttribute('aria-live', 'polite');
    badgeEl.textContent = 'Copied';
    document.body.appendChild(badgeEl);
    return badgeEl;
  }

  function _flashBadge(anchorEl) {
    const el = _ensureBadge();
    // Position above anchor; fall back to top-right of viewport.
    if (anchorEl && anchorEl.getBoundingClientRect) {
      const r = anchorEl.getBoundingClientRect();
      el.style.left = Math.round(r.left + r.width / 2) + 'px';
      el.style.top = Math.round(r.top - 8) + 'px';
      el.style.transform = 'translate(-50%, -100%)';
    } else {
      el.style.left = 'auto';
      el.style.right = '16px';
      el.style.top = '16px';
      el.style.transform = 'none';
    }
    el.classList.remove('is-fading');
    // Force reflow so re-adding the class restarts the animation.
    // eslint-disable-next-line no-unused-expressions
    el.offsetHeight;
    el.classList.add('is-visible');
    if (badgeTimer) clearTimeout(badgeTimer);
    badgeTimer = setTimeout(() => {
      el.classList.add('is-fading');
      setTimeout(() => {
        el.classList.remove('is-visible', 'is-fading');
      }, 250);
    }, BADGE_TTL_MS);
  }

  function _icon() {
    // Inline SVG, currentColor so CSS controls hue. Matches a "terminal
    // prompt" glyph (>_), chosen to read as "shell command".
    return (
      '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" ' +
      'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ' +
      'stroke-linejoin="round" aria-hidden="true">' +
      '<path d="M3 5l2.5 2.5L3 10"/>' +
      '<path d="M8 11h5"/>' +
      '</svg>'
    );
  }

  function _copyIcon() {
    return (
      '<svg viewBox="0 0 16 16" width="12" height="12" fill="none" ' +
      'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ' +
      'stroke-linejoin="round" aria-hidden="true">' +
      '<rect x="5" y="5" width="8" height="9" rx="1.2"/>' +
      '<path d="M3 11V3a1 1 0 0 1 1-1h7"/>' +
      '</svg>'
    );
  }

  // ---------------- recent-requests ring ----------------
  function _push(method, url, body) {
    try {
      // Avoid recording our own internal calls (none today, but defensive).
      ring.push({
        method: (method || 'GET').toUpperCase(),
        url: _absUrl(url),
        body: body == null ? null : body,
        ts: Date.now(),
      });
      while (ring.length > RING_MAX) ring.shift();
    } catch (_e) {
      // never let recording break the host fetch
    }
  }

  function _wrapFetch() {
    if (typeof window.fetch !== 'function' || window.__pfmCurlFetchWrapped) {
      return;
    }
    const orig = window.fetch.bind(window);
    window.__pfmCurlFetchWrapped = true;
    window.fetch = function (input, init) {
      try {
        let method = 'GET';
        let url = '';
        let body = null;
        if (typeof input === 'string') {
          url = input;
        } else if (input && typeof input === 'object') {
          // Request instance
          url = input.url || '';
          method = input.method || method;
        }
        if (init && typeof init === 'object') {
          if (init.method) method = init.method;
          if (init.body !== undefined) body = init.body;
        }
        // Body can be FormData / Blob / URLSearchParams / ArrayBuffer / string / object.
        // For non-string-friendly types, fall back to a stable placeholder.
        if (body && typeof body !== 'string') {
          if (typeof FormData !== 'undefined' && body instanceof FormData) {
            body = '<FormData>';
          } else if (typeof Blob !== 'undefined' && body instanceof Blob) {
            body = '<Blob>';
          } else if (typeof URLSearchParams !== 'undefined' &&
                     body instanceof URLSearchParams) {
            body = body.toString();
          } else if (typeof ArrayBuffer !== 'undefined' &&
                     body instanceof ArrayBuffer) {
            body = '<ArrayBuffer>';
          }
        }
        _push(method, url, body);
      } catch (_e) {
        // ignore - we never block the real fetch
      }
      return orig(input, init);
    };
  }

  // ---------------- dropdown UI ----------------
  function _ensureDropdown() {
    if (dropdownEl) return dropdownEl;
    dropdownEl = document.createElement('div');
    dropdownEl.className = 'pfm-curl-dropdown';
    dropdownEl.setAttribute('role', 'menu');
    dropdownEl.setAttribute('aria-label', 'Recent API requests');
    dropdownEl.innerHTML = '';
    document.body.appendChild(dropdownEl);
    // Click-outside closes.
    document.addEventListener('mousedown', (ev) => {
      if (!dropdownEl || !dropdownEl.classList.contains('is-open')) return;
      if (!dropdownEl.contains(ev.target) &&
          !(ev.target.closest && ev.target.closest('.pfm-curl-btn'))) {
        _closeDropdown();
      }
    });
    document.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && dropdownEl &&
          dropdownEl.classList.contains('is-open')) {
        _closeDropdown();
      }
    });
    return dropdownEl;
  }

  function _closeDropdown() {
    if (dropdownEl) dropdownEl.classList.remove('is-open');
  }

  function _renderDropdown() {
    const root = _ensureDropdown();
    if (ring.length === 0) {
      root.innerHTML =
        '<div class="pfm-curl-dropdown-empty">No requests recorded yet.</div>';
      return;
    }
    const rows = [];
    rows.push('<div class="pfm-curl-dropdown-head">Recent requests</div>');
    // Most recent first.
    for (let i = ring.length - 1; i >= 0; i--) {
      const r = ring[i];
      const path = r.url.replace(/^https?:\/\/[^/]+/, '') || r.url;
      rows.push(
        '<button type="button" class="pfm-curl-row" data-idx="' + i + '" ' +
        'role="menuitem" title="Copy cURL">' +
          '<span class="pfm-curl-row-method pfm-curl-m-' +
            r.method.toLowerCase() + '">' + r.method + '</span>' +
          '<span class="pfm-curl-row-url">' +
            _escapeHtml(path) +
          '</span>' +
          '<span class="pfm-curl-row-copy" aria-hidden="true">' +
            _copyIcon() +
          '</span>' +
        '</button>'
      );
    }
    root.innerHTML = rows.join('');
    Array.prototype.forEach.call(
      root.querySelectorAll('.pfm-curl-row'),
      (rowEl) => {
        rowEl.addEventListener('click', () => {
          const idx = parseInt(rowEl.getAttribute('data-idx'), 10);
          const entry = ring[idx];
          if (!entry) return;
          const cmd = _build({
            method: entry.method,
            url: entry.url,
            body: entry.body,
          });
          _writeClipboard(cmd)
            .then(() => _flashBadge(rowEl))
            .catch((e) => console.warn('[copy-as-curl] clipboard failed', e));
          _closeDropdown();
        });
      }
    );
  }

  function _escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function _openDropdownNear(anchorEl) {
    _renderDropdown();
    const root = _ensureDropdown();
    if (anchorEl && anchorEl.getBoundingClientRect) {
      const r = anchorEl.getBoundingClientRect();
      const top = Math.round(r.bottom + window.scrollY + 6);
      // Right-align under the icon button.
      let right = Math.round(window.innerWidth - r.right);
      if (right < 8) right = 8;
      root.style.top = top + 'px';
      root.style.right = right + 'px';
      root.style.left = 'auto';
    } else {
      root.style.top = '64px';
      root.style.right = '16px';
      root.style.left = 'auto';
    }
    root.classList.add('is-open');
  }

  // ---------------- public: attach single-descriptor button ----------------
  function _ensureRelativeHost(targetEl) {
    // For position:absolute to anchor correctly we need a positioned host.
    // If the target has static positioning, switch to relative.
    try {
      const cs = window.getComputedStyle(targetEl);
      if (cs && cs.position === 'static') {
        targetEl.style.position = 'relative';
      }
    } catch (_e) {
      targetEl.style.position = 'relative';
    }
  }

  function _makeButton(opts) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'pfm-curl-btn';
    btn.setAttribute('aria-label', opts.label || 'Copy as cURL');
    btn.setAttribute('title', opts.title || 'Copy as cURL');
    btn.innerHTML = _icon();
    return btn;
  }

  function attach(targetEl, requestDescriptor) {
    if (!targetEl || !targetEl.appendChild) return null;
    if (!requestDescriptor || !requestDescriptor.url) return null;
    _ensureRelativeHost(targetEl);
    // Avoid double-mount: re-use existing button for this descriptor key.
    const key = (requestDescriptor.method || 'GET').toUpperCase() + ' ' +
                requestDescriptor.url;
    const existing = targetEl.querySelector(
      '.pfm-curl-btn[data-curl-key="' + CSS.escape(key) + '"]'
    );
    if (existing) {
      existing.__pfmCurlDesc = requestDescriptor;
      return existing;
    }
    const btn = _makeButton({ title: 'Copy as cURL: ' + key });
    btn.setAttribute('data-curl-key', key);
    btn.__pfmCurlDesc = requestDescriptor;
    btn.addEventListener('click', (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      const desc = btn.__pfmCurlDesc;
      const cmd = _build(desc);
      _writeClipboard(cmd)
        .then(() => _flashBadge(btn))
        .catch((e) => console.warn('[copy-as-curl] clipboard failed', e));
    });
    targetEl.appendChild(btn);
    return btn;
  }

  // ---------------- public: copyFromLastFetch ----------------
  function copyFromLastFetch(anchorEl) {
    // When invoked without an anchor (e.g. programmatic), copy the single
    // most recent entry directly to clipboard.
    if (!anchorEl) {
      const last = ring[ring.length - 1];
      if (!last) return Promise.resolve(false);
      const cmd = _build({
        method: last.method, url: last.url, body: last.body,
      });
      return _writeClipboard(cmd).then(() => {
        _flashBadge(null);
        return true;
      });
    }
    _openDropdownNear(anchorEl);
    return Promise.resolve(true);
  }

  // ---------------- public: record (for non-fetch transports) ----------------
  function record(method, url, body, headers) {
    _push(method, url, body);
    // headers ignored in ring; preserved only in descriptor-mode.
    void headers;
  }

  // ---------------- floating "recent requests" launcher ----------------
  function _ensureLauncher() {
    if (document.querySelector('.pfm-curl-launcher')) return;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'pfm-curl-launcher';
    btn.setAttribute('aria-label', 'Copy recent API request as cURL');
    btn.setAttribute('title', 'Recent requests → Copy as cURL');
    btn.innerHTML = _icon() +
      '<span class="pfm-curl-launcher-label">cURL</span>';
    btn.addEventListener('click', (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      const root = _ensureDropdown();
      if (root.classList.contains('is-open')) {
        _closeDropdown();
      } else {
        _openDropdownNear(btn);
      }
    });
    document.body.appendChild(btn);
  }

  // ---------------- bootstrap ----------------
  function _init() {
    _wrapFetch();
    _ensureLauncher();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _init, { once: true });
  } else {
    _init();
  }

  // ---------------- export ----------------
  window.PFM = window.PFM || {};
  window.PFM.curl = {
    attach: attach,
    copyFromLastFetch: copyFromLastFetch,
    record: record,
    build: _build,
    // exposed for tests / power-users; do not mutate from outside.
    _ring: ring,
  };
})();
