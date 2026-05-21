/* mode-router.js — SPA-like mode switching for Prediction Terminal
 *
 * Owns transitions between the three top-level modes:
 *   - regression  (factor-model fits)
 *   - strategies  (alpha hub)
 *   - terminal    (Bloomberg-style data hub, DEFAULT)
 *
 * Public API (attached to window.PFM.modeRouter):
 *   current()          -> 'regression' | 'strategies' | 'terminal'
 *   switch(mode)       -> sets mode, updates DOM, URL, fires events
 *   onChange(cb)       -> subscribes; returns unsubscribe function
 *
 * Side effects of switch(mode):
 *   - <body data-pfm-mode="<mode>">
 *   - Panes with [data-mode-pane="<mode>"] become visible, others hidden
 *   - URL hash becomes #<mode> via history.pushState (no reload)
 *   - CustomEvent 'pfm:switch-mode' dispatched on window with detail {mode}
 *   - localStorage 'pfm:last-mode' persisted
 *
 * Inputs the router reacts to:
 *   - Clicks on [data-mode-tab="<mode>"] elements
 *   - 'pfm:switch-mode' CustomEvent (from cmdk, shortcuts, etc.)
 *   - popstate (browser back/forward)
 *
 * Coordination scope: W11-03-mode-router. Does NOT modify index.html.
 */
(function () {
  "use strict";

  var VALID_MODES = ["regression", "strategies", "terminal"];
  var DEFAULT_MODE = "terminal";
  var STORAGE_KEY = "pfm:last-mode";
  var EVENT_NAME = "pfm:switch-mode";

  // Guard: only one instance.
  if (window.PFM && window.PFM.modeRouter) {
    return;
  }
  window.PFM = window.PFM || {};

  var listeners = [];
  // Re-entrancy guard so dispatching pfm:switch-mode from inside switch()
  // does not cause infinite recursion when the same event listener calls back.
  var dispatching = false;

  function isValid(mode) {
    return typeof mode === "string" && VALID_MODES.indexOf(mode) !== -1;
  }

  function readHashMode() {
    var h = (window.location.hash || "").replace(/^#/, "").toLowerCase();
    return isValid(h) ? h : null;
  }

  function readBodyMode() {
    if (!document.body) return null;
    var m = document.body.getAttribute("data-pfm-mode");
    return isValid(m) ? m : null;
  }

  function readStoredMode() {
    try {
      var m = window.localStorage.getItem(STORAGE_KEY);
      return isValid(m) ? m : null;
    } catch (_e) {
      return null;
    }
  }

  function detectInitialMode() {
    // Priority: explicit body attribute -> URL hash -> localStorage -> default.
    return (
      readBodyMode() ||
      readHashMode() ||
      readStoredMode() ||
      DEFAULT_MODE
    );
  }

  function persist(mode) {
    try {
      window.localStorage.setItem(STORAGE_KEY, mode);
    } catch (_e) {
      // ignore (private mode, quota, etc.)
    }
  }

  function applyPaneVisibility(mode) {
    var panes = document.querySelectorAll("[data-mode-pane]");
    for (var i = 0; i < panes.length; i++) {
      var pane = panes[i];
      var paneMode = pane.getAttribute("data-mode-pane");
      var active = paneMode === mode;
      // Toggle a class for CSS-driven transitions, plus hidden attribute as a
      // robust fallback so the pane really disappears even without styles.
      if (active) {
        pane.classList.add("is-active-mode");
        pane.removeAttribute("hidden");
        pane.setAttribute("aria-hidden", "false");
      } else {
        pane.classList.remove("is-active-mode");
        pane.setAttribute("hidden", "");
        pane.setAttribute("aria-hidden", "true");
      }
    }
  }

  function applyTabState(mode) {
    var tabs = document.querySelectorAll("[data-mode-tab]");
    for (var i = 0; i < tabs.length; i++) {
      var tab = tabs[i];
      var tabMode = tab.getAttribute("data-mode-tab");
      var active = tabMode === mode;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    }
  }

  function updateBody(mode) {
    if (!document.body) return;
    document.body.setAttribute("data-pfm-mode", mode);
  }

  function updateHash(mode) {
    // Avoid pushing duplicate history entries.
    var current = (window.location.hash || "").replace(/^#/, "");
    if (current === mode) return;
    try {
      var url =
        window.location.pathname + window.location.search + "#" + mode;
      window.history.pushState({ pfmMode: mode }, "", url);
    } catch (_e) {
      // Some environments (file://, sandboxes) block pushState.
      try {
        window.location.hash = "#" + mode;
      } catch (_e2) {
        /* give up silently */
      }
    }
  }

  function notify(mode, options) {
    // Call onChange subscribers.
    for (var i = 0; i < listeners.length; i++) {
      try {
        listeners[i](mode);
      } catch (err) {
        // Swallow listener errors so one bad subscriber can't break others.
        if (window.console && window.console.error) {
          window.console.error("[mode-router] listener error", err);
        }
      }
    }
    // Dispatch DOM event for non-subscriber consumers (cmdk, shortcuts).
    if (!options || !options.silent) {
      dispatching = true;
      try {
        var ev = new CustomEvent(EVENT_NAME, {
          detail: { mode: mode, source: (options && options.source) || "router" },
        });
        window.dispatchEvent(ev);
      } catch (_e) {
        // Older browsers: fall back to document.createEvent. Best-effort.
        try {
          var legacy = document.createEvent("CustomEvent");
          legacy.initCustomEvent(EVENT_NAME, false, false, { mode: mode });
          window.dispatchEvent(legacy);
        } catch (_e2) {
          /* ignore */
        }
      } finally {
        dispatching = false;
      }
    }
  }

  function switchMode(mode, options) {
    if (!isValid(mode)) {
      if (window.console && window.console.warn) {
        window.console.warn("[mode-router] invalid mode:", mode);
      }
      return false;
    }
    var prev = readBodyMode();
    var changed = prev !== mode;

    updateBody(mode);
    applyPaneVisibility(mode);
    applyTabState(mode);

    if (!options || !options.skipHash) {
      updateHash(mode);
    }
    persist(mode);

    if (changed) {
      notify(mode, options);
    }
    return true;
  }

  function current() {
    return readBodyMode() || DEFAULT_MODE;
  }

  function onChange(cb) {
    if (typeof cb !== "function") return function () {};
    listeners.push(cb);
    return function unsubscribe() {
      var idx = listeners.indexOf(cb);
      if (idx !== -1) listeners.splice(idx, 1);
    };
  }

  // --- Wire DOM listeners ---------------------------------------------------

  function handleTabClick(ev) {
    // Walk up to find an ancestor with [data-mode-tab].
    var node = ev.target;
    while (node && node !== document.body) {
      if (node.nodeType === 1 && node.hasAttribute && node.hasAttribute("data-mode-tab")) {
        var mode = node.getAttribute("data-mode-tab");
        if (isValid(mode)) {
          ev.preventDefault();
          switchMode(mode, { source: "tab-click" });
        }
        return;
      }
      node = node.parentNode;
    }
  }

  function handleSwitchEvent(ev) {
    if (dispatching) return; // ignore our own dispatch
    var detail = ev && ev.detail ? ev.detail : {};
    if (detail.source === "router") return; // already came from us
    if (isValid(detail.mode)) {
      switchMode(detail.mode, { source: detail.source || "event", silent: true });
      // We still call notify subscribers (above) with silent:true preventing
      // re-dispatch, so external listeners triggered from cmdk see one event.
    }
  }

  function handlePopState(_ev) {
    var mode = readHashMode() || readStoredMode() || DEFAULT_MODE;
    // Don't push another history entry — we're responding to navigation.
    switchMode(mode, { skipHash: true, source: "popstate" });
  }

  function init() {
    var initial = detectInitialMode();
    // skipHash on initial apply if URL already matches, otherwise normalize.
    var hash = readHashMode();
    switchMode(initial, {
      skipHash: hash === initial,
      source: "init",
      silent: false,
    });

    document.addEventListener("click", handleTabClick, false);
    window.addEventListener(EVENT_NAME, handleSwitchEvent, false);
    window.addEventListener("popstate", handlePopState, false);
  }

  // Expose public API immediately so other scripts loaded after this one can
  // bind regardless of DOM readiness.
  window.PFM.modeRouter = {
    current: current,
    switch: function (mode) {
      return switchMode(mode, { source: "api" });
    },
    onChange: onChange,
    // Internal helpers, useful for tests / debugging.
    _validModes: VALID_MODES.slice(),
    _defaultMode: DEFAULT_MODE,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, false);
  } else {
    init();
  }
})();
