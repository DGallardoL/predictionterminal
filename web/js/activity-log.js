/* ============================================================
 * activity-log.js  (W12-37, wave-12)
 *
 * Recent-activity log. Tracks last 20 user actions (with a
 * FIFO buffer of 50 in localStorage `pfm:activity:v1`),
 * renders a right-side slide-in drawer (320px wide), and
 * supports re-triggering ("replay") of recorded entries.
 *
 * Public API:
 *   window.PFM.activityLog.push(entry)      // {type,label,detail,replay?}
 *   window.PFM.activityLog.list()  -> Entry[]
 *   window.PFM.activityLog.clear()
 *   window.PFM.activityLog.replay(entryId)
 *   window.PFM.activityLog.mount(containerEl?)
 *   window.PFM.activityLog.open() / close() / toggle()
 *
 * Auto-tracked events (best-effort, defensive against missing
 * DOM hooks):
 *   - Mode switch  (clicks on [data-mode], [data-mode-tab],
 *     [data-tab-id])
 *   - /fit submit  (submit on #fit-form, #regression-form,
 *     form[data-fit-form], or form containing #fit-submit)
 *   - Factor add/remove (custom events `pfm:factor:add`,
 *     `pfm:factor:remove`; plus DOM mutation fallback)
 *   - Pin/unpin    (clicks on [data-pin-action])
 *   - Theme toggle (click on [data-theme-toggle], or watch
 *     <html data-theme> attr changes)
 *   - cmdk open    (custom `pfm:cmdk:open`, or Ctrl/Cmd+K key)
 *   - Sort change  (custom `pfm:sort:change` or click on
 *     [data-sort-key])
 *   - Filter change (custom `pfm:filter:change`, or change on
 *     [data-filter])
 *
 * Replay: handlers stored in a private map keyed by entry.id.
 * Replay function is NOT serialized (can't survive reload);
 * persisted entries marked `replayable:false` after reload.
 *
 * Coordination: claim W12-37. Sole owner of the
 * `window.PFM.activityLog` namespace. Does NOT modify
 * index.html — to mount in the page add this line in any
 * downstream init script:
 *     window.PFM.activityLog.mount()
 * ============================================================ */

(function () {
  "use strict";

  // ---- Namespace bootstrap (defensive) ----
  window.PFM = window.PFM || {};
  if (window.PFM.activityLog && window.PFM.activityLog.__W12_37) {
    return; // idempotent: already loaded
  }

  // ---- Constants ----
  var STORAGE_KEY = "pfm:activity:v1";
  var MAX_PERSIST = 50;   // FIFO cap on localStorage
  var MAX_DISPLAY = 20;   // shown in the drawer
  var DRAWER_ID = "pfm-activity-drawer-root";

  // ---- Internal state ----
  var entries = loadFromStorage();     // newest-first array
  var replayHandlers = {};             // id -> fn (in-memory only)
  var mounted = false;
  var rootEl = null;
  var drawerEl = null;
  var backdropEl = null;
  var listEl = null;
  var toggleEl = null;
  var countEl = null;
  var subtitleEl = null;
  var openState = false;
  var listeners = [];                  // event-listener teardown handles

  // ---- Storage helpers ----
  function loadFromStorage() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      var parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      // After reload, replay handlers are gone — keep entries but mark non-replayable
      return parsed.map(function (e) {
        return Object.assign({}, e, { replayable: false });
      });
    } catch (err) {
      console.warn("[activity-log] failed to load storage:", err);
      return [];
    }
  }

  function persist() {
    try {
      var capped = entries.slice(0, MAX_PERSIST);
      var serializable = capped.map(function (e) {
        return {
          id: e.id,
          ts: e.ts,
          type: e.type,
          label: e.label,
          detail: e.detail || null,
        };
      });
      localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
    } catch (err) {
      // Quota or disabled storage — non-fatal
      console.warn("[activity-log] failed to persist:", err);
    }
  }

  // ---- Utils ----
  function genId() {
    return (
      Date.now().toString(36) +
      "-" +
      Math.random().toString(36).slice(2, 8)
    );
  }

  function fmtTime(ts) {
    try {
      var d = new Date(ts);
      var hh = String(d.getHours()).padStart(2, "0");
      var mm = String(d.getMinutes()).padStart(2, "0");
      var ss = String(d.getSeconds()).padStart(2, "0");
      return hh + ":" + mm + ":" + ss;
    } catch (err) {
      return "--:--:--";
    }
  }

  var ICON_GLYPHS = {
    mode: "M",
    fit: "F",
    factor: "+",
    "factor-remove": "-",
    pin: "P",
    theme: "T",
    cmdk: "K",
    sort: "S",
    filter: "f",
  };

  function iconFor(type) {
    return ICON_GLYPHS[type] || "*";
  }

  function safeText(s) {
    if (s == null) return "";
    return String(s);
  }

  // ---- Core API ----
  function pushEntry(entry) {
    if (!entry || typeof entry !== "object") return null;
    var rec = {
      id: entry.id || genId(),
      ts: entry.ts || Date.now(),
      type: safeText(entry.type) || "event",
      label: safeText(entry.label) || "(unlabeled)",
      detail: entry.detail != null ? entry.detail : null,
      replayable: typeof entry.replay === "function",
    };
    if (rec.replayable) {
      replayHandlers[rec.id] = entry.replay;
    }
    // newest first
    entries.unshift(rec);
    if (entries.length > MAX_PERSIST) {
      var removed = entries.splice(MAX_PERSIST);
      removed.forEach(function (e) {
        delete replayHandlers[e.id];
      });
    }
    persist();
    render();
    return rec.id;
  }

  function listEntries() {
    // Return a defensive copy
    return entries.map(function (e) {
      return Object.assign({}, e);
    });
  }

  function clearAll() {
    entries = [];
    replayHandlers = {};
    persist();
    render();
  }

  function replay(entryId) {
    var fn = replayHandlers[entryId];
    if (typeof fn !== "function") {
      console.info("[activity-log] entry " + entryId + " is not replayable");
      return false;
    }
    try {
      fn();
      // log the replay itself
      pushEntry({
        type: "fit",
        label: "Replayed entry",
        detail: { sourceId: entryId },
      });
      return true;
    } catch (err) {
      console.warn("[activity-log] replay failed:", err);
      return false;
    }
  }

  // ---- Rendering ----
  function render() {
    if (!mounted || !listEl) return;
    var slice = entries.slice(0, MAX_DISPLAY);
    if (slice.length === 0) {
      listEl.innerHTML =
        '<div class="pfm-activity-empty">' +
        '<p class="pfm-activity-empty__title">No activity yet</p>' +
        '<p class="pfm-activity-empty__hint">Switch modes, run a fit, or pin a result to see it here.</p>' +
        "</div>";
    } else {
      var html = "";
      for (var i = 0; i < slice.length; i++) {
        var e = slice[i];
        var detailStr = "";
        if (e.detail) {
          try {
            detailStr =
              typeof e.detail === "string"
                ? e.detail
                : JSON.stringify(e.detail, null, 2);
          } catch (err) {
            detailStr = String(e.detail);
          }
        }
        html +=
          '<li class="pfm-activity-row" role="listitem" data-id="' +
          escapeAttr(e.id) +
          '" data-replayable="' +
          (e.replayable ? "true" : "false") +
          '" tabindex="0">' +
          '<span class="pfm-activity-icon" data-type="' +
          escapeAttr(e.type) +
          '">' +
          escapeText(iconFor(e.type)) +
          "</span>" +
          '<span class="pfm-activity-time">' +
          escapeText(fmtTime(e.ts)) +
          "</span>" +
          '<span class="pfm-activity-label">' +
          escapeText(e.label) +
          "</span>" +
          '<span class="pfm-activity-replay">' +
          (e.replayable ? "Replay" : "") +
          "</span>" +
          (detailStr
            ? '<span class="pfm-activity-detail">' +
              escapeText(detailStr) +
              "</span>"
            : "") +
          "</li>";
      }
      listEl.innerHTML = '<ul style="list-style:none;margin:0;padding:0;">' + html + "</ul>";
    }
    if (countEl) {
      countEl.textContent = String(entries.length);
    }
    if (subtitleEl) {
      subtitleEl.textContent =
        entries.length === 0
          ? "0 events"
          : entries.length + " events (showing " + slice.length + ")";
    }
    // BUGFIX (wave-7): hide the floating "Activity 0" pill when there are no
    // entries — the persistent "0" badge in screenshots was visual noise that
    // looked broken. The drawer remains accessible via the toggleDrawer()
    // public API for future-Claude / power users.
    if (toggleEl) {
      if (entries.length === 0) {
        toggleEl.style.display = "none";
      } else {
        toggleEl.style.display = "";
      }
    }
  }

  function escapeText(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
  function escapeAttr(s) {
    return String(s).replace(/"/g, "&quot;");
  }

  // ---- Drawer DOM ----
  function buildDom(container) {
    rootEl = document.createElement("div");
    rootEl.id = DRAWER_ID;

    // Toggle button (FAB)
    toggleEl = document.createElement("button");
    toggleEl.type = "button";
    toggleEl.className = "pfm-activity-toggle";
    toggleEl.setAttribute("aria-label", "Open activity log");
    toggleEl.innerHTML =
      '<span class="pfm-activity-toggle__dot" aria-hidden="true"></span>' +
      '<span>Activity</span>' +
      '<span class="pfm-activity-toggle__count">0</span>';
    countEl = toggleEl.querySelector(".pfm-activity-toggle__count");
    toggleEl.addEventListener("click", openDrawer);

    // Backdrop
    backdropEl = document.createElement("div");
    backdropEl.className = "pfm-activity-backdrop";
    backdropEl.addEventListener("click", closeDrawer);

    // Drawer
    drawerEl = document.createElement("aside");
    drawerEl.className = "pfm-activity-drawer";
    drawerEl.setAttribute("role", "complementary");
    drawerEl.setAttribute("aria-label", "Recent activity log");
    drawerEl.innerHTML =
      '<header class="pfm-activity-header">' +
      "<div>" +
      '<h2 class="pfm-activity-title">Activity</h2>' +
      '<p class="pfm-activity-subtitle">0 events</p>' +
      "</div>" +
      '<div class="pfm-activity-actions">' +
      '<button type="button" class="pfm-activity-btn" data-act="clear">Clear</button>' +
      '<button type="button" class="pfm-activity-btn pfm-activity-btn--close" data-act="close" aria-label="Close">&times;</button>' +
      "</div>" +
      "</header>" +
      '<div class="pfm-activity-list" role="list"></div>' +
      '<footer class="pfm-activity-footer">' +
      "<span>last 20 of 50</span>" +
      "<span>pfm:activity:v1</span>" +
      "</footer>";

    listEl = drawerEl.querySelector(".pfm-activity-list");
    subtitleEl = drawerEl.querySelector(".pfm-activity-subtitle");

    drawerEl
      .querySelector('[data-act="close"]')
      .addEventListener("click", closeDrawer);
    drawerEl
      .querySelector('[data-act="clear"]')
      .addEventListener("click", function () {
        if (confirm("Clear all activity entries?")) clearAll();
      });

    // Delegated row click → replay
    listEl.addEventListener("click", function (ev) {
      var row = ev.target.closest && ev.target.closest(".pfm-activity-row");
      if (!row) return;
      if (row.dataset.replayable !== "true") return;
      replay(row.dataset.id);
    });
    // Keyboard support
    listEl.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter" && ev.key !== " ") return;
      var row =
        ev.target.classList &&
        ev.target.classList.contains("pfm-activity-row")
          ? ev.target
          : null;
      if (!row || row.dataset.replayable !== "true") return;
      ev.preventDefault();
      replay(row.dataset.id);
    });

    rootEl.appendChild(toggleEl);
    rootEl.appendChild(backdropEl);
    rootEl.appendChild(drawerEl);

    (container || document.body).appendChild(rootEl);
  }

  function openDrawer() {
    if (!mounted) return;
    openState = true;
    drawerEl.classList.add("is-open");
    backdropEl.classList.add("is-open");
    toggleEl.setAttribute("aria-label", "Close activity log");
    render();
  }

  function closeDrawer() {
    if (!mounted) return;
    openState = false;
    drawerEl.classList.remove("is-open");
    backdropEl.classList.remove("is-open");
    toggleEl.setAttribute("aria-label", "Open activity log");
  }

  function toggleDrawer() {
    if (openState) closeDrawer();
    else openDrawer();
  }

  // ---- Mount ----
  function mount(container) {
    if (mounted) return;
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () {
        mount(container);
      });
      return;
    }
    buildDom(container);
    mounted = true;
    attachAutoTrackers();
    render();
  }

  // ---- Auto-tracked events ----
  function on(target, ev, fn, opts) {
    if (!target || !target.addEventListener) return;
    target.addEventListener(ev, fn, opts || false);
    listeners.push({ target: target, ev: ev, fn: fn });
  }

  function attachAutoTrackers() {
    // Mode switch
    on(document, "click", function (ev) {
      var el =
        ev.target.closest &&
        ev.target.closest("[data-mode],[data-mode-tab],[data-tab-id]");
      if (!el) return;
      var mode =
        el.dataset.mode ||
        el.dataset.modeTab ||
        el.dataset.tabId ||
        el.textContent.trim();
      pushEntry({
        type: "mode",
        label: "Mode switch -> " + mode,
        detail: { mode: mode },
        replay: function () {
          try {
            el.click();
          } catch (e) {}
        },
      });
    });

    // /fit submit
    on(
      document,
      "submit",
      function (ev) {
        var form = ev.target;
        if (!form || !form.matches) return;
        var match =
          form.matches(
            "#fit-form,#regression-form,form[data-fit-form]"
          ) ||
          (form.querySelector && form.querySelector("#fit-submit"));
        if (!match) return;
        var fd;
        var summary = {};
        try {
          fd = new FormData(form);
          fd.forEach(function (v, k) {
            // skip CSRF / huge inputs
            if (k.toLowerCase().includes("csrf")) return;
            summary[k] =
              typeof v === "string" && v.length > 120 ? v.slice(0, 120) + "..." : v;
          });
        } catch (e) {}
        pushEntry({
          type: "fit",
          label: "Submitted /fit",
          detail: summary,
          replay: function () {
            try {
              if (typeof form.requestSubmit === "function") form.requestSubmit();
              else form.submit();
            } catch (e) {}
          },
        });
      },
      true
    );

    // Factor add/remove (custom events preferred)
    on(document, "pfm:factor:add", function (ev) {
      var d = (ev && ev.detail) || {};
      pushEntry({
        type: "factor",
        label: "Factor added: " + (d.slug || d.name || "?"),
        detail: d,
      });
    });
    on(document, "pfm:factor:remove", function (ev) {
      var d = (ev && ev.detail) || {};
      pushEntry({
        type: "factor-remove",
        label: "Factor removed: " + (d.slug || d.name || "?"),
        detail: d,
      });
    });

    // Pin/unpin
    on(document, "click", function (ev) {
      var el = ev.target.closest && ev.target.closest("[data-pin-action]");
      if (!el) return;
      var action = el.dataset.pinAction || "pin";
      var id = el.dataset.pinId || el.dataset.id || "";
      pushEntry({
        type: "pin",
        label: action === "unpin" ? "Unpinned " + id : "Pinned " + id,
        detail: { action: action, id: id },
        replay: function () {
          try {
            el.click();
          } catch (e) {}
        },
      });
    });

    // Theme toggle — via dedicated button
    on(document, "click", function (ev) {
      var el = ev.target.closest && ev.target.closest("[data-theme-toggle]");
      if (!el) return;
      // Defer reading the resulting theme until after click handlers run
      setTimeout(function () {
        var theme =
          document.documentElement.getAttribute("data-theme") ||
          (window.matchMedia &&
          window.matchMedia("(prefers-color-scheme: dark)").matches
            ? "dark"
            : "light");
        pushEntry({
          type: "theme",
          label: "Theme -> " + theme,
          detail: { theme: theme },
          replay: function () {
            try {
              el.click();
            } catch (e) {}
          },
        });
      }, 0);
    });
    // Theme toggle — via attribute mutation (covers programmatic toggles)
    try {
      var lastTheme = document.documentElement.getAttribute("data-theme");
      var obs = new MutationObserver(function (muts) {
        for (var i = 0; i < muts.length; i++) {
          if (
            muts[i].type === "attributes" &&
            muts[i].attributeName === "data-theme"
          ) {
            var t = document.documentElement.getAttribute("data-theme");
            if (t && t !== lastTheme) {
              lastTheme = t;
              pushEntry({
                type: "theme",
                label: "Theme -> " + t,
                detail: { theme: t, source: "mutation" },
              });
            }
          }
        }
      });
      obs.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["data-theme"],
      });
    } catch (err) {
      /* non-fatal */
    }

    // cmdk open — custom event
    on(document, "pfm:cmdk:open", function () {
      pushEntry({
        type: "cmdk",
        label: "Opened cmdk",
        detail: null,
        replay: function () {
          try {
            document.dispatchEvent(new CustomEvent("pfm:cmdk:open"));
          } catch (e) {}
        },
      });
    });
    // cmdk open — keyboard fallback
    on(document, "keydown", function (ev) {
      var k = (ev.key || "").toLowerCase();
      if ((ev.metaKey || ev.ctrlKey) && k === "k") {
        pushEntry({
          type: "cmdk",
          label: "Opened cmdk (Ctrl/Cmd+K)",
          detail: null,
        });
      }
    });

    // Sort change
    on(document, "pfm:sort:change", function (ev) {
      var d = (ev && ev.detail) || {};
      pushEntry({
        type: "sort",
        label: "Sort -> " + (d.key || "?") + (d.dir ? " " + d.dir : ""),
        detail: d,
      });
    });
    on(document, "click", function (ev) {
      var el = ev.target.closest && ev.target.closest("[data-sort-key]");
      if (!el) return;
      var key = el.dataset.sortKey;
      pushEntry({
        type: "sort",
        label: "Sort by " + key,
        detail: { key: key },
        replay: function () {
          try {
            el.click();
          } catch (e) {}
        },
      });
    });

    // Filter change — custom event
    on(document, "pfm:filter:change", function (ev) {
      var d = (ev && ev.detail) || {};
      pushEntry({
        type: "filter",
        label: "Filter " + (d.key || "?") + " = " + (d.value || ""),
        detail: d,
      });
    });
    // Filter change — DOM change listener
    on(
      document,
      "change",
      function (ev) {
        var el = ev.target;
        if (!el || !el.matches) return;
        if (!el.matches("[data-filter]")) return;
        var key = el.dataset.filter || el.name || "filter";
        var val =
          el.type === "checkbox" || el.type === "radio"
            ? el.checked
            : el.value;
        pushEntry({
          type: "filter",
          label: "Filter " + key + " = " + val,
          detail: { key: key, value: val },
        });
      },
      true
    );
  }

  // ---- Public namespace ----
  window.PFM.activityLog = {
    __W12_37: true,
    push: pushEntry,
    list: listEntries,
    clear: clearAll,
    replay: replay,
    mount: mount,
    open: openDrawer,
    close: closeDrawer,
    toggle: toggleDrawer,
  };

  // ---- Auto-mount on DOM ready (defensive: only if body exists) ----
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      try {
        mount();
      } catch (e) {
        console.warn("[activity-log] auto-mount failed:", e);
      }
    });
  } else {
    try {
      mount();
    } catch (e) {
      console.warn("[activity-log] auto-mount failed:", e);
    }
  }
})();

/* ============================================================
 * End of activity-log.js
 * ============================================================ */
