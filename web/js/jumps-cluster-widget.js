/**
 * jumps-cluster-widget.js — "Today's jump clusters" panel for the Terminal homepage.
 * Task T58.
 *
 * Consumes:
 *   GET /terminal/jumps/cluster  (see api/src/pfm/terminal/jumps_cluster.py)
 *   Response: JumpsClusterResponse
 *     { slugs, days, time_tol_minutes, kw_min_jaccard,
 *       n_jumps_total, n_clusters,
 *       clusters: [
 *         { cluster_id, ts_iso, n_markets, n_articles,
 *           dominant_terms: [str], representative_headline,
 *           member_jumps: [{ slug, ts_iso, delta_pp, sentiment_alignment }] }
 *       ] }
 *
 * Public API:
 *   window.PFM.clusterWidget = {
 *     mount(containerEl, opts?),  // opts: { refreshMs, viewAllHref, apiBase, ... }
 *     refresh(),                  // force a re-fetch
 *     pause(),                    // stop auto-refresh
 *     resume(),                   // resume auto-refresh
 *     unmount(),                  // tear down (for tests)
 *   }
 *
 * Behaviour:
 *   - Auto-refresh every 60 s (configurable via opts.refreshMs or
 *     data-refresh-ms on the container).
 *   - visibilitychange: pauses fetch loop when document is hidden,
 *     resumes (and triggers an immediate refresh if stale) on visible.
 *   - Skeleton state (T04 .skel) on initial load, swapped in-place once
 *     data arrives. Empty state if clusters[] is empty.
 *   - Click row → toggles expanded view that lists ALL member markets.
 *   - "View all" link → navigates to opts.viewAllHref (defaults to
 *     "#terminal/clusters") so the host page can wire its own router.
 *
 * Mount instruction (host code, separate file):
 *   <script src="/js/jumps-cluster-widget.js" defer></script>
 *   <link rel="stylesheet" href="/css/jumps-cluster-widget.css">
 *   ...
 *   <div id="jc-host"></div>
 *   <script>
 *     window.PFM.clusterWidget.mount(document.getElementById('jc-host'));
 *   </script>
 */
(function () {
  'use strict';

  // ---------- defaults ----------------------------------------------------

  const DEFAULTS = Object.freeze({
    apiBase: '',                       // same-origin by default
    endpoint: '/terminal/jumps/cluster',
    params: '',                        // raw query string ("?slugs=...")
    refreshMs: 60000,                  // 60 s
    viewAllHref: '#terminal/clusters',
    slugTruncate: 24,                  // chars before ellipsis on slug
    topMarkets: 3,                     // collapsed-row member count
    skeletonRows: 3,                   // count while initial load
    initialFetch: true,
    fetchTimeoutMs: 12000,
    onError: null,                     // optional callback(err)
    onRefresh: null,                   // optional callback(payload)
  });

  // ---------- state -------------------------------------------------------

  // A single global widget instance is what the spec calls for (one widget
  // per Terminal homepage). If a second mount() comes in we tear the old
  // one down first so we don't leak intervals.
  const STATE = {
    instance: null,
  };

  // ---------- utilities ---------------------------------------------------

  function _truncate(str, n) {
    if (typeof str !== 'string') return '';
    if (str.length <= n) return str;
    if (n <= 1) return str.slice(0, n);
    return str.slice(0, n - 1) + '…';
  }

  function _esc(s) {
    if (s == null) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // delta_pp = percentage-points jump (0..100 scale). 1 pp == 100 bps.
  function _ppToBps(deltaPp) {
    if (typeof deltaPp !== 'number' || !isFinite(deltaPp)) return null;
    return Math.round(deltaPp * 100);
  }

  function _signBps(bps) {
    if (bps == null) return '';
    if (bps > 0) return '+' + bps + ' bps';
    if (bps < 0) return bps + ' bps';      // includes '-'
    return '0 bps';
  }

  function _formatTimestamp(d) {
    // Compact HH:MM:SS UTC; mirrors the Terminal's tabular-nums style.
    const hh = String(d.getUTCHours()).padStart(2, '0');
    const mm = String(d.getUTCMinutes()).padStart(2, '0');
    const ss = String(d.getUTCSeconds()).padStart(2, '0');
    return hh + ':' + mm + ':' + ss + ' UTC';
  }

  function _now() {
    return Date.now();
  }

  // Direction: majority sign of delta_pp; ties → flat.
  function _dominantDirection(members) {
    let up = 0;
    let down = 0;
    let sumAbs = 0;
    let n = 0;
    for (const m of members || []) {
      const d = typeof m.delta_pp === 'number' ? m.delta_pp : 0;
      if (d > 0) up += 1;
      else if (d < 0) down += 1;
      sumAbs += Math.abs(d);
      n += 1;
    }
    const avgPp = n > 0 ? sumAbs / n : 0;
    const avgBps = Math.round(avgPp * 100);
    let dir = 'flat';
    if (up > down) dir = 'up';
    else if (down > up) dir = 'down';
    return { dir: dir, avgBps: avgBps };
  }

  function _topMembersByMagnitude(members, k) {
    const arr = Array.isArray(members) ? members.slice() : [];
    arr.sort(function (a, b) {
      const da = Math.abs(typeof a.delta_pp === 'number' ? a.delta_pp : 0);
      const db = Math.abs(typeof b.delta_pp === 'number' ? b.delta_pp : 0);
      return db - da;
    });
    return arr.slice(0, k);
  }

  function _themeTag(cluster) {
    const terms = Array.isArray(cluster.dominant_terms) ? cluster.dominant_terms : [];
    if (terms.length === 0) return 'mixed';
    return terms.slice(0, 2).join(' / ');
  }

  // ---------- HTML builders ----------------------------------------------

  function _skeletonHTML(opts) {
    const rows = [];
    for (let i = 0; i < opts.skeletonRows; i += 1) {
      rows.push(
        '<div class="pfm-jc-skel-row" aria-hidden="true">' +
          '<div class="skel skel--chip"></div>' +
          '<div class="skel skel--line-md"></div>' +
          '<div class="skel skel--line-sm"></div>' +
          '<div class="skel skel--line-sm"></div>' +
        '</div>'
      );
    }
    return rows.join('');
  }

  function _arrowHTML(dir) {
    if (dir === 'up') return '<span class="pfm-jc-arrow pfm-jc-arrow--up" aria-label="up">▲</span>';
    if (dir === 'down') return '<span class="pfm-jc-arrow pfm-jc-arrow--down" aria-label="down">▼</span>';
    return '<span class="pfm-jc-arrow pfm-jc-arrow--flat" aria-label="flat">•</span>';
  }

  function _memberLineHTML(m, opts) {
    const slugTrunc = _truncate(m.slug || '', opts.slugTruncate);
    const bps = _ppToBps(m.delta_pp);
    let cls = 'pfm-jc-member-bps';
    if (bps != null && bps > 0) cls += ' pfm-jc-member-bps--up';
    else if (bps != null && bps < 0) cls += ' pfm-jc-member-bps--down';
    const bpsStr = bps == null ? '—' : _signBps(bps);
    return (
      '<li class="pfm-jc-member">' +
        '<span class="pfm-jc-member-slug" title="' + _esc(m.slug || '') + '">' +
          _esc(slugTrunc) +
        '</span>' +
        '<span class="' + cls + '">' + _esc(bpsStr) + '</span>' +
      '</li>'
    );
  }

  function _clusterRowHTML(cluster, opts) {
    const members = Array.isArray(cluster.member_jumps) ? cluster.member_jumps : [];
    const top = _topMembersByMagnitude(members, opts.topMarkets);
    const dom = _dominantDirection(members);
    const theme = _themeTag(cluster);
    const headlineSafe = cluster.representative_headline
      ? '<div class="pfm-jc-headline" style="margin-top:6px;font-size:12px;color:var(--ink-2,#334155);font-style:italic;">' +
          _esc(_truncate(cluster.representative_headline, 140)) +
        '</div>'
      : '';

    const topListHTML = top.map(function (m) { return _memberLineHTML(m, opts); }).join('');
    const allListHTML = members.map(function (m) { return _memberLineHTML(m, opts); }).join('');

    return (
      '<button type="button" class="pfm-jc-row" data-cluster-id="' + _esc(String(cluster.cluster_id)) + '" aria-expanded="false">' +
        '<div class="pfm-jc-row-head">' +
          '<span class="pfm-jc-chip">' + _esc(theme) + '</span>' +
          '<span class="pfm-jc-size"><strong>' + _esc(String(cluster.n_markets)) + '</strong> markets moved together</span>' +
          '<span class="pfm-jc-summary">' +
            _arrowHTML(dom.dir) +
            '<span>' + _esc(String(dom.avgBps)) + ' bps avg</span>' +
          '</span>' +
        '</div>' +
        '<ul class="pfm-jc-members">' + topListHTML + '</ul>' +
        headlineSafe +
        '<div class="pfm-jc-extra">' +
          '<div class="pfm-jc-extra-inner">' +
            '<ul class="pfm-jc-members">' + allListHTML + '</ul>' +
          '</div>' +
        '</div>' +
      '</button>'
    );
  }

  function _emptyStateHTML() {
    return (
      '<div class="pfm-jc-empty">' +
        'No clusters detected today — markets moved independently' +
      '</div>'
    );
  }

  function _errorStateHTML(msg) {
    return (
      '<div class="pfm-jc-error" role="alert">' +
        _esc(msg || 'Failed to load clusters') +
      '</div>'
    );
  }

  // refresh icon (inline SVG so we don't depend on a font/icon set)
  const REFRESH_SVG =
    '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" ' +
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
      '<path d="M13.5 3.5A6 6 0 1 0 14 8" stroke="currentColor" stroke-width="1.6" ' +
        'stroke-linecap="round" fill="none"/>' +
      '<path d="M10 2.5h3.5V6" stroke="currentColor" stroke-width="1.6" ' +
        'stroke-linecap="round" stroke-linejoin="round" fill="none"/>' +
    '</svg>';

  function _shellHTML(opts) {
    return (
      '<section class="pfm-jc-widget" data-loaded="false">' +
        '<header class="pfm-jc-header">' +
          '<h3 class="pfm-jc-title">Today\'s jump clusters</h3>' +
          '<div class="pfm-jc-meta">' +
            '<span class="pfm-jc-updated" data-pfm-jc-updated>—</span>' +
            '<button type="button" class="pfm-jc-refresh" data-pfm-jc-refresh ' +
              'title="Refresh" aria-label="Refresh clusters">' +
              REFRESH_SVG +
            '</button>' +
          '</div>' +
        '</header>' +
        '<div class="pfm-jc-body" data-pfm-jc-body>' +
          _skeletonHTML(opts) +
        '</div>' +
        '<footer class="pfm-jc-footer">' +
          '<a class="pfm-jc-view-all" href="' + _esc(opts.viewAllHref) + '" ' +
            'data-pfm-jc-view-all>View all clusters →</a>' +
        '</footer>' +
      '</section>'
    );
  }

  // ---------- fetch -------------------------------------------------------

  function _fetchClusters(opts) {
    const url = (opts.apiBase || '') + opts.endpoint + (opts.params || '');
    const ctrl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
    const t = setTimeout(function () {
      if (ctrl) {
        try { ctrl.abort(); } catch (_) { /* noop */ }
      }
    }, opts.fetchTimeoutMs);
    const init = {
      method: 'GET',
      credentials: 'same-origin',
      headers: { 'Accept': 'application/json' },
    };
    if (ctrl) init.signal = ctrl.signal;
    return fetch(url, init).then(function (res) {
      clearTimeout(t);
      if (!res.ok) {
        const err = new Error('HTTP ' + res.status);
        err.status = res.status;
        throw err;
      }
      return res.json();
    }).catch(function (e) {
      clearTimeout(t);
      throw e;
    });
  }

  // ---------- instance ----------------------------------------------------

  function _createInstance(containerEl, userOpts) {
    if (!containerEl || typeof containerEl !== 'object') {
      throw new Error('clusterWidget.mount(): containerEl required');
    }

    const dataMs = parseInt(containerEl.getAttribute('data-refresh-ms') || '', 10);
    const opts = Object.assign({}, DEFAULTS, userOpts || {});
    if (!isNaN(dataMs) && dataMs > 0) opts.refreshMs = dataMs;
    if (opts.refreshMs < 5000) opts.refreshMs = 5000;     // floor

    // build shell
    containerEl.innerHTML = _shellHTML(opts);
    const root = containerEl.querySelector('.pfm-jc-widget');
    const bodyEl = containerEl.querySelector('[data-pfm-jc-body]');
    const updatedEl = containerEl.querySelector('[data-pfm-jc-updated]');
    const refreshBtn = containerEl.querySelector('[data-pfm-jc-refresh]');
    const viewAllEl = containerEl.querySelector('[data-pfm-jc-view-all]');

    // internal state
    const state = {
      paused: false,
      destroyed: false,
      inFlight: false,
      timerId: null,
      lastFetchAt: 0,
      lastPayload: null,
      lastError: null,
    };

    function _setUpdatedNow() {
      try {
        updatedEl.textContent = 'Updated ' + _formatTimestamp(new Date());
      } catch (_) { /* noop */ }
    }

    function _setSpinning(on) {
      if (!refreshBtn) return;
      if (on) refreshBtn.setAttribute('data-spinning', 'true');
      else refreshBtn.removeAttribute('data-spinning');
    }

    function _render(payload) {
      if (state.destroyed) return;
      const clusters = (payload && Array.isArray(payload.clusters)) ? payload.clusters : [];
      if (clusters.length === 0) {
        bodyEl.innerHTML = _emptyStateHTML();
      } else {
        const html = clusters.map(function (c) { return _clusterRowHTML(c, opts); }).join('');
        bodyEl.innerHTML = html;
      }
      root.setAttribute('data-loaded', 'true');
    }

    function _renderError(err) {
      if (state.destroyed) return;
      // Keep the prior data on screen if we have it; only show the error
      // banner when we have nothing useful to display.
      if (state.lastPayload && Array.isArray(state.lastPayload.clusters) &&
          state.lastPayload.clusters.length > 0) {
        // Inject a small transient banner above existing rows.
        const banner = document.createElement('div');
        banner.className = 'pfm-jc-error';
        banner.setAttribute('role', 'alert');
        banner.textContent = 'Refresh failed: ' + (err && err.message ? err.message : 'unknown');
        const existing = bodyEl.querySelector('.pfm-jc-error');
        if (existing) existing.remove();
        bodyEl.insertBefore(banner, bodyEl.firstChild);
        setTimeout(function () {
          if (banner.parentNode) banner.parentNode.removeChild(banner);
        }, 4000);
      } else {
        bodyEl.innerHTML = _errorStateHTML(err && err.message ? err.message : 'Failed to load clusters');
        root.setAttribute('data-loaded', 'true');
      }
    }

    function _doFetch() {
      if (state.destroyed || state.inFlight) return Promise.resolve();
      state.inFlight = true;
      _setSpinning(true);
      return _fetchClusters(opts).then(function (payload) {
        if (state.destroyed) return;
        state.lastPayload = payload;
        state.lastError = null;
        state.lastFetchAt = _now();
        _render(payload);
        _setUpdatedNow();
        if (typeof opts.onRefresh === 'function') {
          try { opts.onRefresh(payload); } catch (_) { /* noop */ }
        }
      }).catch(function (err) {
        if (state.destroyed) return;
        state.lastError = err;
        _renderError(err);
        if (typeof opts.onError === 'function') {
          try { opts.onError(err); } catch (_) { /* noop */ }
        }
      }).then(function () {
        state.inFlight = false;
        _setSpinning(false);
      });
    }

    function _scheduleNext() {
      if (state.destroyed || state.paused) return;
      if (state.timerId) clearTimeout(state.timerId);
      state.timerId = setTimeout(function () {
        _doFetch().then(_scheduleNext);
      }, opts.refreshMs);
    }

    function _start() {
      if (opts.initialFetch) _doFetch().then(_scheduleNext);
      else _scheduleNext();
    }

    // ---- event wiring --------------------------------------------------

    function _onRowClick(ev) {
      const row = ev.target && ev.target.closest && ev.target.closest('.pfm-jc-row');
      if (!row || !bodyEl.contains(row)) return;
      const expanded = row.getAttribute('aria-expanded') === 'true';
      row.setAttribute('aria-expanded', expanded ? 'false' : 'true');
    }

    function _onRefreshClick(ev) {
      ev.preventDefault();
      // Manual refresh — reset the timer so the next auto-tick is in a full
      // refreshMs window from "now" (not a few ms after a manual click).
      if (state.timerId) clearTimeout(state.timerId);
      _doFetch().then(_scheduleNext);
    }

    function _onVisibilityChange() {
      if (state.destroyed) return;
      if (document.hidden) {
        // Pause auto-refresh while hidden, but don't flip the user's
        // explicit pause() call.
        if (state.timerId) {
          clearTimeout(state.timerId);
          state.timerId = null;
        }
      } else if (!state.paused) {
        // If we've been hidden longer than refreshMs, force a fresh fetch
        // now; otherwise just resume the schedule from where we left off.
        const stale = (_now() - state.lastFetchAt) >= opts.refreshMs;
        if (stale) {
          _doFetch().then(_scheduleNext);
        } else {
          _scheduleNext();
        }
      }
    }

    bodyEl.addEventListener('click', _onRowClick);
    if (refreshBtn) refreshBtn.addEventListener('click', _onRefreshClick);
    if (viewAllEl) {
      // If host doesn't override href, leave default navigation. Hosts that
      // want SPA behaviour can listen for the 'pfm:clusters:view-all' event.
      viewAllEl.addEventListener('click', function (ev) {
        try {
          window.dispatchEvent(new CustomEvent('pfm:clusters:view-all', {
            detail: { href: opts.viewAllHref },
          }));
        } catch (_) { /* noop */ }
      });
    }
    document.addEventListener('visibilitychange', _onVisibilityChange);

    // ---- public methods on the instance --------------------------------

    const api = {
      refresh: function () {
        if (state.destroyed) return Promise.resolve();
        if (state.timerId) clearTimeout(state.timerId);
        return _doFetch().then(_scheduleNext);
      },
      pause: function () {
        state.paused = true;
        if (state.timerId) {
          clearTimeout(state.timerId);
          state.timerId = null;
        }
      },
      resume: function () {
        if (state.destroyed) return;
        if (!state.paused) return;
        state.paused = false;
        const stale = (_now() - state.lastFetchAt) >= opts.refreshMs;
        if (stale) _doFetch().then(_scheduleNext);
        else _scheduleNext();
      },
      unmount: function () {
        state.destroyed = true;
        if (state.timerId) {
          clearTimeout(state.timerId);
          state.timerId = null;
        }
        bodyEl.removeEventListener('click', _onRowClick);
        if (refreshBtn) refreshBtn.removeEventListener('click', _onRefreshClick);
        document.removeEventListener('visibilitychange', _onVisibilityChange);
        try { containerEl.innerHTML = ''; } catch (_) { /* noop */ }
      },
      // Diagnostics (handy for tests / devtools)
      _state: state,
      _opts: opts,
    };

    _start();
    return api;
  }

  // ---------- public namespace -------------------------------------------

  const PFM = window.PFM = window.PFM || {};

  PFM.clusterWidget = {
    mount: function (containerEl, opts) {
      // Tear down a prior instance bound to *any* container — this widget is
      // a singleton on the Terminal homepage.
      if (STATE.instance && typeof STATE.instance.unmount === 'function') {
        try { STATE.instance.unmount(); } catch (_) { /* noop */ }
        STATE.instance = null;
      }
      STATE.instance = _createInstance(containerEl, opts);
      return STATE.instance;
    },
    refresh: function () {
      if (STATE.instance) return STATE.instance.refresh();
      return Promise.resolve();
    },
    pause: function () {
      if (STATE.instance) STATE.instance.pause();
    },
    resume: function () {
      if (STATE.instance) STATE.instance.resume();
    },
    unmount: function () {
      if (STATE.instance) {
        STATE.instance.unmount();
        STATE.instance = null;
      }
    },
  };
})();
