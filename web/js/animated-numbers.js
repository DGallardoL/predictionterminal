/* ============================================================
 * animated-numbers.js  (W12-34, wave-12)
 *
 * Vanilla CountUp-style number transitions for metric cards.
 * No deps. Respects prefers-reduced-motion.
 *
 * Public API:
 *   window.PFM.animatedNumbers = {
 *     attach(el, opts) -> { update, destroy },
 *     update(el, newValue),
 *     detach(el),
 *     autoAttachAll(selector='[data-animated-number]')
 *   }
 *
 * Per-element data attributes (all optional):
 *   data-decimals="2"        // fraction digits
 *   data-prefix="$"          // string prepended
 *   data-suffix="%"          // string appended
 *   data-show-sign="true"    // "+" prefix on positive
 *   data-duration="600"      // ms (overrides opts.duration)
 *   data-animated-number     // marker for autoAttachAll()
 *
 * MutationObserver: if the element's text changes (e.g. server
 * re-render writes a new raw number), we parse it and animate.
 * Set `el._pfmAnimSilent = true` around your own writes to skip.
 * ============================================================ */
(function () {
  'use strict';
  const W = window;
  W.PFM = W.PFM || {};
  if (W.PFM.animatedNumbers) return;

  const REDUCED = W.matchMedia && W.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
  const registry = new WeakMap();

  function parseNum(str) {
    if (str == null) return NaN;
    const cleaned = String(str).replace(/[^0-9.\-eE]/g, '');
    const n = parseFloat(cleaned);
    return Number.isFinite(n) ? n : NaN;
  }

  function fmt(value, el) {
    const decimals = parseInt(el.dataset.decimals || '0', 10) || 0;
    const prefix = el.dataset.prefix || '';
    const suffix = el.dataset.suffix || '';
    const showSign = el.dataset.showSign === 'true';
    const sign = showSign && value > 0 ? '+' : '';
    return `${sign}${prefix}${value.toFixed(decimals)}${suffix}`;
  }

  function writeText(el, text) {
    el._pfmAnimSilent = true;
    el.textContent = text;
    el._pfmAnimSilent = false;
  }

  function attach(el, opts) {
    if (!el || el.nodeType !== 1) return null;
    if (registry.has(el)) return registry.get(el).handle;

    const defaults = { duration: 600 };
    const cfg = Object.assign({}, defaults, opts || {});
    const dur = parseInt(el.dataset.duration, 10) || cfg.duration;

    const state = {
      current: parseNum(el.textContent) || 0,
      target: parseNum(el.textContent) || 0,
      raf: null,
    };

    function animateTo(newValue) {
      if (!Number.isFinite(newValue)) return;
      state.target = newValue;
      if (REDUCED) {
        state.current = newValue;
        writeText(el, fmt(newValue, el));
        return;
      }
      const start = state.current;
      const delta = newValue - start;
      const t0 = performance.now();
      if (state.raf) cancelAnimationFrame(state.raf);
      const tick = (now) => {
        const t = Math.min(1, (now - t0) / dur);
        const v = start + delta * easeOutCubic(t);
        state.current = v;
        writeText(el, fmt(t === 1 ? newValue : v, el));
        if (t < 1) state.raf = requestAnimationFrame(tick);
        else state.raf = null;
      };
      state.raf = requestAnimationFrame(tick);
    }

    const mo = new MutationObserver(() => {
      if (el._pfmAnimSilent) return;
      const next = parseNum(el.textContent);
      if (Number.isFinite(next) && next !== state.target) animateTo(next);
    });
    mo.observe(el, { childList: true, characterData: true, subtree: true });

    writeText(el, fmt(state.current, el));

    const handle = {
      update: (v) => animateTo(typeof v === 'number' ? v : parseNum(v)),
      destroy: () => {
        if (state.raf) cancelAnimationFrame(state.raf);
        mo.disconnect();
        registry.delete(el);
      },
    };
    registry.set(el, { handle, state, mo });
    return handle;
  }

  function update(el, newValue) {
    const rec = registry.get(el);
    if (rec) rec.handle.update(newValue);
    else attach(el).update(newValue);
  }

  function detach(el) {
    const rec = registry.get(el);
    if (rec) rec.handle.destroy();
  }

  function autoAttachAll(selector) {
    const sel = selector || '[data-animated-number]';
    document.querySelectorAll(sel).forEach((el) => attach(el));
  }

  W.PFM.animatedNumbers = { attach, update, detach, autoAttachAll };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => autoAttachAll());
  } else {
    autoAttachAll();
  }
})();
