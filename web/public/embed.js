/* PFM embed loader.
 *
 * Drop one of these on any page:
 *
 *   <script src="https://your-pfm-host.example.com/embed.js" async></script>
 *
 *   <div data-pfm-market="will-bitcoin-hit-100k"></div>
 *   <div data-pfm-strategy="fed_cuts_10_2026__fed_cuts_7_2026" data-pfm-theme="dark"></div>
 *   <div data-pfm-compare="will-x-happen,will-y-happen"></div>
 *
 * Supported attributes per element:
 *   data-pfm-market    — single-market mini card (slug)
 *   data-pfm-strategy  — alpha-strategy card (pair_id)
 *   data-pfm-compare   — overlay card (comma-separated slugs)
 *   data-pfm-theme     — "light" (default) | "dark"
 *   data-pfm-height    — initial pixel height (default 200)
 */
(function () {
  'use strict';

  // The script tag itself tells us where the API lives — derive the base URL
  // so embeds don't have to hardcode the host.
  var scriptEl = document.currentScript || (function () {
    var arr = document.getElementsByTagName('script');
    return arr[arr.length - 1];
  })();
  var apiBase = (function () {
    if (!scriptEl || !scriptEl.src) return '';
    try {
      var u = new URL(scriptEl.src);
      return u.protocol + '//' + u.host;
    } catch (e) { return ''; }
  })();

  function makeIframe(src, initialHeight) {
    var f = document.createElement('iframe');
    f.src = src;
    f.loading = 'lazy';
    f.setAttribute('frameborder', '0');
    f.setAttribute('scrolling', 'no');
    f.style.cssText =
      'width:100%;height:' + (initialHeight || 200) + 'px;border:0;' +
      'border-radius:8px;display:block;background:transparent;';
    return f;
  }

  function buildUrl(path, params) {
    var qs = Object.keys(params || {})
      .filter(function (k) { return params[k] !== undefined && params[k] !== null && params[k] !== ''; })
      .map(function (k) { return encodeURIComponent(k) + '=' + encodeURIComponent(params[k]); })
      .join('&');
    return apiBase + path + (qs ? ('?' + qs) : '');
  }

  function mountAll() {
    document.querySelectorAll('[data-pfm-market]:not([data-pfm-mounted])').forEach(function (el) {
      var slug = el.getAttribute('data-pfm-market');
      var theme = el.getAttribute('data-pfm-theme') || 'light';
      var height = parseInt(el.getAttribute('data-pfm-height') || '200', 10);
      var url = buildUrl('/embed/market/' + encodeURIComponent(slug), { theme: theme });
      el.appendChild(makeIframe(url, height));
      el.setAttribute('data-pfm-mounted', '1');
    });
    document.querySelectorAll('[data-pfm-strategy]:not([data-pfm-mounted])').forEach(function (el) {
      var pid = el.getAttribute('data-pfm-strategy');
      var theme = el.getAttribute('data-pfm-theme') || 'light';
      var height = parseInt(el.getAttribute('data-pfm-height') || '180', 10);
      var url = buildUrl('/embed/strategy/' + encodeURIComponent(pid), { theme: theme });
      el.appendChild(makeIframe(url, height));
      el.setAttribute('data-pfm-mounted', '1');
    });
    document.querySelectorAll('[data-pfm-compare]:not([data-pfm-mounted])').forEach(function (el) {
      var slugs = el.getAttribute('data-pfm-compare');
      var theme = el.getAttribute('data-pfm-theme') || 'light';
      var height = parseInt(el.getAttribute('data-pfm-height') || '240', 10);
      var url = buildUrl('/embed/compare', { slugs: slugs, theme: theme });
      el.appendChild(makeIframe(url, height));
      el.setAttribute('data-pfm-mounted', '1');
    });
  }

  // Auto-resize iframes when child cards postMessage their measured height.
  window.addEventListener('message', function (e) {
    var data = e.data;
    if (!data || data.pfm !== 'resize' || typeof data.height !== 'number') return;
    var iframes = document.querySelectorAll('iframe');
    for (var i = 0; i < iframes.length; i++) {
      var f = iframes[i];
      if (f.contentWindow === e.source) {
        f.style.height = data.height + 'px';
        break;
      }
    }
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountAll);
  } else {
    mountAll();
  }

  // Re-scan when the page mutates (SPA hosts): observe body for new elements.
  if (typeof MutationObserver !== 'undefined') {
    var obs = new MutationObserver(mountAll);
    obs.observe(document.body || document.documentElement, { childList: true, subtree: true });
  }
})();
