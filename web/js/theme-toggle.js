/* ============================================================
 * theme-toggle.js  (T05, wave-10)
 *
 * Light / dark / system theme management for the Prediction
 * Terminal frontend.
 *
 * Public API (idempotent; safe to load once per page):
 *
 *   window.PFM.theme.current()         -> 'light' | 'dark' | 'system'
 *   window.PFM.theme.resolved()        -> 'light' | 'dark'  (after system resolution)
 *   window.PFM.theme.set(mode)         -> 'light' | 'dark' | 'system'
 *   window.PFM.theme.toggle()          -> flips between light <-> dark explicitly
 *   window.PFM.theme.onChange(cb)      -> () => off, cb is called with { mode, resolved }
 *
 * Persistence:
 *   localStorage key:   'pfm:theme'
 *   accepted values:    'light' | 'dark' | 'system'
 *   default (unset):    'system'  (matches prefers-color-scheme)
 *
 * Activation surface:
 *   The resolved theme is reflected on <html> via the
 *   `data-theme` attribute, which is consumed by:
 *     - web/css/tokens.css                (variable redefinition)
 *     - web/css/dark-mode.css             (mode-specific overrides)
 *
 * Toggle button:
 *   Auto-injected on DOMContentLoaded as a 36x36px fixed-position
 *   pill in the top-right of the viewport. Class .pfm-theme-toggle.
 *   Sun / moon SVG swaps via CSS based on [data-theme="dark"].
 *
 * Events:
 *   'pfm:theme-change' (window-level CustomEvent)
 *     detail: { mode: 'light'|'dark'|'system', resolved: 'light'|'dark' }
 *   Listeners include plotly-theme.js (will re-theme open charts
 *   via window.PFM.replotAll?.()).
 *
 * Mount instructions (for index-html-owner):
 *   <link rel="stylesheet" href="css/dark-mode.css">
 *   <script defer src="js/theme-toggle.js"></script>
 *   No DOM mount point required (button self-injects).
 *
 * Keybinds: NONE shipped here. The global keyboard-shortcuts
 * module (T15) may register Shift+D -> PFM.theme.toggle() later.
 * ============================================================ */

(function () {
  "use strict";

  if (typeof window === "undefined") return;
  window.PFM = window.PFM || {};
  if (window.PFM.theme && window.PFM.theme.__initialized) return;

  var STORAGE_KEY = "pfm:theme";
  var VALID_MODES = { light: 1, dark: 1, system: 1 };
  var TRANSITION_CLASS = "pfm-theme-transition";

  var listeners = [];
  var mediaQuery = null;

  // ---------------------------------------------------------
  // Storage helpers (defensive: localStorage may be unavailable
  // in private mode or sandboxed iframes).
  // ---------------------------------------------------------
  function readStored() {
    try {
      var v = window.localStorage.getItem(STORAGE_KEY);
      if (v && VALID_MODES[v]) return v;
    } catch (_) {}
    return "system";
  }

  function writeStored(mode) {
    try {
      window.localStorage.setItem(STORAGE_KEY, mode);
    } catch (_) {}
  }

  // ---------------------------------------------------------
  // Resolution: turn 'system' into a concrete 'light' | 'dark'.
  // ---------------------------------------------------------
  function systemPrefersDark() {
    try {
      return (
        window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: dark)").matches
      );
    } catch (_) {
      return false;
    }
  }

  function resolveMode(mode) {
    if (mode === "dark") return "dark";
    if (mode === "light") return "light";
    return systemPrefersDark() ? "dark" : "light";
  }

  // ---------------------------------------------------------
  // Apply the theme to <html>. Enables the smooth-transition
  // class briefly so background/color/border swaps are eased,
  // then removes it so the class doesn't penalize unrelated
  // animations.
  // ---------------------------------------------------------
  function apply(mode, opts) {
    var resolved = resolveMode(mode);
    var html = document.documentElement;
    if (!html) return resolved;

    var skipTransition = opts && opts.skipTransition;

    if (!skipTransition) {
      html.classList.add(TRANSITION_CLASS);
      window.setTimeout(function () {
        html.classList.remove(TRANSITION_CLASS);
      }, 240);
    }

    html.setAttribute("data-theme", resolved);
    // Mirror the user's selection (including 'system') so other
    // scripts can distinguish between explicit-dark and OS-dark.
    html.setAttribute("data-theme-mode", mode);

    return resolved;
  }

  // ---------------------------------------------------------
  // Notify subscribers.
  // ---------------------------------------------------------
  function dispatch(mode, resolved) {
    var detail = { mode: mode, resolved: resolved };
    try {
      window.dispatchEvent(
        new CustomEvent("pfm:theme-change", { detail: detail })
      );
    } catch (_) {}
    for (var i = 0; i < listeners.length; i++) {
      try {
        listeners[i](detail);
      } catch (_) {}
    }
    // Convenience hook for Plotly re-theming.
    try {
      if (typeof window.PFM.replotAll === "function") {
        window.PFM.replotAll();
      }
    } catch (_) {}
  }

  // ---------------------------------------------------------
  // Public state.
  // ---------------------------------------------------------
  var currentMode = readStored();

  function setMode(next, opts) {
    if (!VALID_MODES[next]) next = "system";
    currentMode = next;
    writeStored(next);
    var resolved = apply(next, opts);
    dispatch(next, resolved);
    syncButton(resolved);
    return next;
  }

  function toggleMode() {
    // Explicit toggle skips 'system' so the user gets a
    // predictable flip. If currently following system, jump
    // to the opposite of the current resolution.
    var resolved = resolveMode(currentMode);
    return setMode(resolved === "dark" ? "light" : "dark");
  }

  function onChange(cb) {
    if (typeof cb !== "function") return function () {};
    listeners.push(cb);
    return function off() {
      var idx = listeners.indexOf(cb);
      if (idx >= 0) listeners.splice(idx, 1);
    };
  }

  // ---------------------------------------------------------
  // Listen to OS preference flips while user is in 'system'.
  // ---------------------------------------------------------
  function bindMedia() {
    try {
      mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
      var handler = function () {
        if (currentMode === "system") {
          var resolved = apply("system");
          dispatch("system", resolved);
          syncButton(resolved);
        }
      };
      if (typeof mediaQuery.addEventListener === "function") {
        mediaQuery.addEventListener("change", handler);
      } else if (typeof mediaQuery.addListener === "function") {
        // Safari < 14 fallback.
        mediaQuery.addListener(handler);
      }
    } catch (_) {}
  }

  // ---------------------------------------------------------
  // Toggle button (auto-injected).
  // ---------------------------------------------------------
  var BUTTON_ID = "pfm-theme-toggle-btn";

  var SUN_SVG =
    '<svg class="pfm-theme-toggle-sun" viewBox="0 0 24 24" fill="none"' +
    ' stroke="currentColor" stroke-width="1.8" stroke-linecap="round"' +
    ' stroke-linejoin="round" aria-hidden="true">' +
    '<circle cx="12" cy="12" r="4.2"></circle>' +
    '<path d="M12 2.5v2.2M12 19.3v2.2M4.2 4.2l1.6 1.6M18.2 18.2l1.6 1.6' +
    'M2.5 12h2.2M19.3 12h2.2M4.2 19.8l1.6-1.6M18.2 5.8l1.6-1.6"></path>' +
    "</svg>";

  var MOON_SVG =
    '<svg class="pfm-theme-toggle-moon" viewBox="0 0 24 24" fill="none"' +
    ' stroke="currentColor" stroke-width="1.8" stroke-linecap="round"' +
    ' stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M20.5 14.5A8 8 0 0 1 9.5 3.5a0.6 0.6 0 0 0-.8-.7 9.2 9.2 0 1 0' +
    " 12.5 12.5 0.6 0.6 0 0 0-.7-.8z\"></path>" +
    "</svg>";

  function ensureButton() {
    if (!document.body) return null;
    var btn = document.getElementById(BUTTON_ID);
    if (btn) return btn;
    btn = document.createElement("button");
    btn.id = BUTTON_ID;
    btn.className = "pfm-theme-toggle";
    btn.type = "button";
    btn.setAttribute("aria-label", "Toggle dark mode");
    btn.setAttribute("title", "Toggle dark mode");
    btn.innerHTML = SUN_SVG + MOON_SVG;
    btn.addEventListener("click", function (ev) {
      ev.preventDefault();
      toggleMode();
    });
    document.body.appendChild(btn);
    return btn;
  }

  function syncButton(resolved) {
    var btn = document.getElementById(BUTTON_ID);
    if (!btn) return;
    var label =
      resolved === "dark" ? "Switch to light mode" : "Switch to dark mode";
    btn.setAttribute("aria-label", label);
    btn.setAttribute("title", label);
    btn.setAttribute("data-resolved", resolved);
    btn.setAttribute("data-mode", currentMode);
  }

  // ---------------------------------------------------------
  // Boot: apply persisted theme as early as possible (avoid
  // flash-of-wrong-theme). Inject button on DOMContentLoaded.
  // ---------------------------------------------------------
  apply(currentMode, { skipTransition: true });
  bindMedia();

  function onReady(fn) {
    if (
      document.readyState === "complete" ||
      document.readyState === "interactive"
    ) {
      window.setTimeout(fn, 0);
    } else {
      document.addEventListener("DOMContentLoaded", fn, { once: true });
    }
  }

  onReady(function () {
    ensureButton();
    syncButton(resolveMode(currentMode));
  });

  // ---------------------------------------------------------
  // Expose API.
  // ---------------------------------------------------------
  window.PFM.theme = {
    __initialized: true,
    current: function () {
      return currentMode;
    },
    resolved: function () {
      return resolveMode(currentMode);
    },
    set: function (mode) {
      return setMode(mode);
    },
    toggle: function () {
      return toggleMode();
    },
    onChange: onChange,
  };
})();
