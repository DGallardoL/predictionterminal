/* ============================================================
 * connection-status.js  (T60, wave-10)
 *
 * Hardened connection-status pill for the Prediction Terminal
 * frontend. Replaces the older `.pfm-conn-status` indicator
 * defined inline in web/index.html, which flickered "Degraded"
 * because the first ping fired before detectApi() resolved.
 *
 * STATE MACHINE (tri-state):
 *   'live'    — all upstream sources OK; overall response < 800ms
 *   'slow'    — any source down, OR latency in [800ms, 5s)
 *   'offline' — request failed, OR timed out (≥ 5s)
 *
 *   An additional pseudo-state 'init' is used internally before
 *   the first successful reading; the pill is hidden during it
 *   so the user never sees a misleading flash.
 *
 * DEBOUNCE:
 *   State changes require **2 consecutive readings** of the new
 *   state before they are committed. This prevents a single
 *   slow request (e.g. a transient laptop wake) from flipping
 *   the pill. Reverting to 'live' from 'slow' / 'offline' is
 *   debounced symmetrically so a single fast reply doesn't
 *   prematurely declare recovery.
 *
 *   Special-case: a single 'slow' tick sandwiched between two
 *   'live' ticks is silently auto-dismissed (never rendered).
 *
 * PUBLIC API (idempotent — safe to load once per page):
 *
 *   window.PFM.conn.current()    -> { state, lastSeenAt, lastLatency, sources }
 *   window.PFM.conn.heartbeat()  -> Promise resolving after one immediate probe
 *   window.PFM.conn.pause()      -> stop heartbeats (no-op if already paused)
 *   window.PFM.conn.resume()     -> resume; triggers an immediate probe
 *   window.PFM.conn.onChange(cb) -> () => off
 *
 * VISIBILITY HANDLING:
 *   While document.hidden the heartbeat is paused (no network);
 *   on `visibilitychange` -> visible, we probe immediately and
 *   reset the debounce counter so a stale state from a tab that
 *   slept for 30 minutes can't linger.
 *
 * MOUNT INSTRUCTIONS (for index-html-owner, when T60 ships):
 *   <link rel="stylesheet" href="css/connection-status.css">
 *   <script defer src="js/connection-status.js"></script>
 *   No DOM mount point required (pill self-injects on
 *   DOMContentLoaded). The legacy `<div id="pfm-conn-status">`
 *   should be removed in the same edit; the JS below also
 *   defensively hides any element with that id if present.
 *
 * COORDINATION:
 *   This script does NOT modify index.html. It self-injects a
 *   `<button class="pfm-conn-pill">` and the diagnostics
 *   drawer into <body>.
 * ============================================================ */

(function () {
  "use strict";

  if (typeof window === "undefined") return;
  window.PFM = window.PFM || {};
  if (window.PFM.conn && window.PFM.conn.__initialized) return;

  // ---------------------------------------------------------
  // Constants
  // ---------------------------------------------------------
  var HEARTBEAT_INTERVAL_MS = 30 * 1000;   // 30s between probes when visible
  var REQUEST_TIMEOUT_MS    = 5 * 1000;    // hard timeout per probe
  var FAST_THRESHOLD_MS     = 800;         // < 800ms => live (if all OK)
  var SLOW_CEILING_MS       = 8000;        // >= 8s   => offline (allow GDELT slowness)
  var DEBOUNCE_TICKS        = 2;           // need 2 consecutive readings to flip
  var DETECT_API_TIMEOUT_MS = 5 * 1000;    // poll PFM_API_BASE up to 5s
  var DETECT_API_INTERVAL   = 200;         // every 200ms

  var STATES = { init: 1, live: 1, slow: 1, offline: 1 };

  // ---------------------------------------------------------
  // Internal state
  // ---------------------------------------------------------
  var state = {
    current: "init",
    candidate: null,        // proposed next state being debounced
    candidateCount: 0,      // consecutive readings of `candidate`
    lastSeenAt: null,       // ISO when last reading completed (any)
    lastOkAt: null,         // ISO when last 'live' reading completed
    lastLatencyMs: null,
    lastError: null,
    sources: [],            // [{name, ok, latency_ms, error}]
    deepEndpoint: true,     // true => /health/deep, fallback to /health
    paused: false,
    inflight: false,
    timerId: null,
    pillEl: null,
    drawerEl: null,
    backdropEl: null,
    listEl: null,
    summaryEl: null,
    refreshBtnEl: null,
    listeners: []
  };

  // ---------------------------------------------------------
  // Utility: emit
  // ---------------------------------------------------------
  function emit() {
    var snap = currentSnapshot();
    for (var i = 0; i < state.listeners.length; i++) {
      try { state.listeners[i](snap); } catch (_) {}
    }
    try {
      window.dispatchEvent(new CustomEvent("pfm:conn-change", { detail: snap }));
    } catch (_) {}
  }

  function currentSnapshot() {
    return {
      state: state.current,
      lastSeenAt: state.lastSeenAt,
      lastOkAt: state.lastOkAt,
      lastLatency: state.lastLatencyMs,
      lastError: state.lastError,
      sources: state.sources.slice(),
      paused: state.paused
    };
  }

  // ---------------------------------------------------------
  // Utility: nowIso
  // ---------------------------------------------------------
  function nowIso() {
    try { return new Date().toISOString(); } catch (_) { return null; }
  }

  // ---------------------------------------------------------
  // API base resolution. We must wait for window.PFM_API_BASE
  // to be set (by web/config.js or by the in-page detectApi).
  // Polls up to DETECT_API_TIMEOUT_MS then returns "" so the
  // probe at least tries a same-origin relative URL.
  // ---------------------------------------------------------
  function waitForApiBase() {
    return new Promise(function (resolve) {
      if (window.PFM_API_BASE != null) {
        resolve(String(window.PFM_API_BASE));
        return;
      }
      var elapsed = 0;
      var iv = setInterval(function () {
        elapsed += DETECT_API_INTERVAL;
        if (window.PFM_API_BASE != null) {
          clearInterval(iv);
          resolve(String(window.PFM_API_BASE));
        } else if (elapsed >= DETECT_API_TIMEOUT_MS) {
          clearInterval(iv);
          // Last-ditch: resolve with empty string so same-origin
          // relative URLs work in single-port dev setups.
          resolve("");
        }
      }, DETECT_API_INTERVAL);
    });
  }

  function getApiBase() {
    // After initial detection, always read fresh — detectApi can
    // upgrade PFM_API_BASE mid-session.
    return (window.PFM_API_BASE != null) ? String(window.PFM_API_BASE) : "";
  }

  // ---------------------------------------------------------
  // Fetch with timeout (AbortController)
  // ---------------------------------------------------------
  function fetchWithTimeout(url, ms) {
    return new Promise(function (resolve) {
      var controller = null;
      var signal = undefined;
      try {
        controller = new AbortController();
        signal = controller.signal;
      } catch (_) {}

      var tid = setTimeout(function () {
        try { if (controller) controller.abort(); } catch (_) {}
        resolve({ ok: false, status: 0, timedOut: true, error: "timeout", body: null });
      }, ms);

      var started = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();

      fetch(url, {
        method: "GET",
        signal: signal,
        credentials: "omit",
        cache: "no-store",
        headers: { "Accept": "application/json" }
      })
        .then(function (r) {
          clearTimeout(tid);
          var ended = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
          var latency = Math.max(0, Math.round(ended - started));
          if (!r.ok) {
            resolve({ ok: false, status: r.status, latency: latency, error: "http_" + r.status, body: null });
            return;
          }
          r.json().then(function (j) {
            resolve({ ok: true, status: r.status, latency: latency, body: j });
          }).catch(function () {
            // body wasn't JSON; treat as still up but with no detail
            resolve({ ok: true, status: r.status, latency: latency, body: {} });
          });
        })
        .catch(function (err) {
          clearTimeout(tid);
          var ended = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
          var latency = Math.max(0, Math.round(ended - started));
          var msg = (err && err.name === "AbortError") ? "aborted" : (err && err.message) || "network_error";
          resolve({ ok: false, status: 0, latency: latency, error: msg, body: null });
        });
    });
  }

  // ---------------------------------------------------------
  // Classify a probe result -> 'live' | 'slow' | 'offline'
  //   live    : ok && latency<FAST_THRESHOLD_MS && every source ok
  //   offline : !ok && (timedOut || network_error || status === 0)
  //             OR latency >= SLOW_CEILING_MS
  //   slow    : everything else (any source down, latency in
  //             [FAST_THRESHOLD_MS, SLOW_CEILING_MS), http error)
  // ---------------------------------------------------------
  function classify(probe) {
    if (!probe) return "offline";
    if (probe.timedOut) return "offline";
    if (probe.latency != null && probe.latency >= SLOW_CEILING_MS) return "offline";
    if (!probe.ok) {
      // status 0 + error like "network_error"/"Failed to fetch" => offline.
      if (probe.status === 0) return "offline";
      // HTTP 5xx with response = still reachable but degraded.
      return "slow";
    }
    // ok=true, evaluate sources if the body looks like /health/deep
    var sources = extractSources(probe.body);
    var allOk = sources.length === 0 || sources.every(function (s) { return s.ok; });
    var fast = (probe.latency != null) && (probe.latency < FAST_THRESHOLD_MS);
    if (allOk && fast) return "live";
    if (allOk) return "slow";   // healthy but slow
    // Some source flagged down → slow (we don't go offline if we still got a response)
    return "slow";
  }

  // ---------------------------------------------------------
  // Try to find a per-source list inside the response body.
  // Supports /health/deep (preferred) and /health (best-effort).
  // /health/deep is expected to return something like:
  //   { ok: true, sources: [
  //       { name: "polymarket", ok: true,  latency_ms: 120, error: null },
  //       { name: "kalshi",     ok: false, latency_ms: 5012, error: "..." },
  //       ...
  //     ] }
  // ---------------------------------------------------------
  function extractSources(body) {
    if (!body || typeof body !== "object") return [];
    var raw = body.sources || body.upstreams || body.checks || body.dependencies;
    if (!raw) return [];

    // Accept either an array [{name, ok, ...}] or a dict keyed by name
    // ({polymarket: {ok, ...}, kalshi: {...}}). The Python backend returns
    // the dict form; legacy callers shipped arrays.
    var items;
    if (Array.isArray(raw)) {
      items = raw;
    } else if (typeof raw === "object") {
      items = Object.keys(raw).map(function (k) {
        var v = raw[k];
        if (!v || typeof v !== "object") return null;
        // Inject the dict key as `name` if the value doesn't already have one.
        return Object.assign({ name: k }, v);
      });
    } else {
      return [];
    }

    return items.map(function (s) {
      if (!s || typeof s !== "object") return null;
      var name = String(s.name || s.id || s.source || "unknown");
      var ok = (s.ok === true) || (s.status === "ok") || (s.healthy === true);
      var latency = (typeof s.latency_ms === "number") ? s.latency_ms
                  : (typeof s.latency === "number")    ? s.latency
                  : null;
      var error = s.error || s.last_error || s.message || null;
      return { name: name, ok: !!ok, latency_ms: latency, error: error ? String(error) : null };
    }).filter(Boolean);
  }

  // ---------------------------------------------------------
  // Apply a new reading with debounce.
  // Rule: a state change is committed only after 2 consecutive
  // identical candidate readings. A single `slow` reading
  // sandwiched between `live` reads never reaches the user.
  // ---------------------------------------------------------
  function applyReading(reading) {
    state.lastSeenAt = nowIso();
    state.lastLatencyMs = reading.latency != null ? reading.latency : null;
    state.lastError = reading.error || null;
    state.sources = reading.sources || [];

    var classified = reading.classified;
    if (!STATES[classified]) classified = "offline";

    // First reading ever: commit immediately so we exit 'init'.
    if (state.current === "init") {
      state.current = classified;
      state.candidate = null;
      state.candidateCount = 0;
      if (classified === "live") state.lastOkAt = state.lastSeenAt;
      renderPill();
      renderDrawerIfOpen();
      emit();
      return;
    }

    if (classified === state.current) {
      // Confirms current state; clear any pending candidate.
      state.candidate = null;
      state.candidateCount = 0;
      if (classified === "live") state.lastOkAt = state.lastSeenAt;
      renderDrawerIfOpen();
      // No emit on identical state (avoids listener spam) but
      // still refresh the in-flight indicator.
      renderPill();
      return;
    }

    // Different from current — debounce.
    if (state.candidate === classified) {
      state.candidateCount += 1;
    } else {
      state.candidate = classified;
      state.candidateCount = 1;
    }

    if (state.candidateCount >= DEBOUNCE_TICKS) {
      state.current = classified;
      state.candidate = null;
      state.candidateCount = 0;
      if (classified === "live") state.lastOkAt = state.lastSeenAt;
      renderPill();
      renderDrawerIfOpen();
      emit();
    } else {
      // Reading recorded but state unchanged. Still refresh the
      // drawer (so per-source latency numbers update live) and
      // the pill (to clear the in-flight dim).
      renderPill();
      renderDrawerIfOpen();
    }
  }

  // ---------------------------------------------------------
  // Perform one probe.
  // ---------------------------------------------------------
  function probeOnce() {
    if (state.inflight) return Promise.resolve(currentSnapshot());
    state.inflight = true;
    if (state.pillEl) state.pillEl.setAttribute("data-checking", "1");

    var base = getApiBase();
    var path = state.deepEndpoint ? "/health/deep" : "/health";
    var url = (base || "") + path;

    return fetchWithTimeout(url, REQUEST_TIMEOUT_MS).then(function (probe) {
      // If /health/deep returned 404, downgrade and try /health
      // next time. (Most installs ship /health long before
      // /health/deep.)
      if (state.deepEndpoint && probe.status === 404) {
        state.deepEndpoint = false;
      }

      var sources = extractSources(probe.body);
      var classified = classify(probe);

      applyReading({
        classified: classified,
        latency: probe.latency,
        error: probe.ok ? null : probe.error,
        sources: sources
      });
    }).catch(function () {
      // Defensive: classify as offline if the promise itself rejects
      applyReading({
        classified: "offline",
        latency: null,
        error: "unhandled",
        sources: []
      });
    }).then(function () {
      state.inflight = false;
      if (state.pillEl) state.pillEl.removeAttribute("data-checking");
      return currentSnapshot();
    });
  }

  // ---------------------------------------------------------
  // Heartbeat loop
  // ---------------------------------------------------------
  function scheduleNext() {
    if (state.paused) return;
    if (state.timerId != null) {
      clearTimeout(state.timerId);
    }
    state.timerId = setTimeout(function () {
      state.timerId = null;
      tick();
    }, HEARTBEAT_INTERVAL_MS);
  }

  function tick() {
    if (state.paused) return;
    if (document.hidden) {
      // Don't probe while hidden. Re-schedule for after the tab
      // becomes visible (handled in visibilitychange).
      return;
    }
    probeOnce().finally(scheduleNext);
  }

  // ---------------------------------------------------------
  // DOM injection: pill
  // ---------------------------------------------------------
  function buildPill() {
    if (state.pillEl) return state.pillEl;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pfm-conn-pill";
    btn.setAttribute("data-state", "init");
    btn.setAttribute("aria-live", "polite");
    btn.setAttribute("aria-label", "Connection status — click to open diagnostics");
    btn.setAttribute("title", "Connection status");

    var dot = document.createElement("span");
    dot.className = "pfm-conn-pill__dot";

    var label = document.createElement("span");
    label.className = "pfm-conn-pill__label";
    label.textContent = "Checking…";

    btn.appendChild(dot);
    btn.appendChild(label);

    btn.addEventListener("click", function (ev) {
      ev.preventDefault();
      openDrawer();
    });

    document.body.appendChild(btn);
    state.pillEl = btn;
    return btn;
  }

  function labelFor(s) {
    switch (s) {
      case "live": return "Live";
      case "slow": return "Slow";
      case "offline": return "Offline";
      case "init":
      default: return "Checking…";
    }
  }

  function renderPill() {
    var pill = state.pillEl;
    if (!pill) return;
    pill.setAttribute("data-state", state.current);
    var label = pill.querySelector(".pfm-conn-pill__label");
    if (label) label.textContent = labelFor(state.current);
    // Reveal once we leave 'init' (CSS handles the fade).
    if (state.current === "init") {
      pill.setAttribute("hidden", "");
    } else {
      pill.removeAttribute("hidden");
    }
    // Tooltip with latency if we have one
    if (state.lastLatencyMs != null) {
      pill.setAttribute("title", "Connection: " + labelFor(state.current)
        + " · " + state.lastLatencyMs + " ms · click for details");
    } else {
      pill.setAttribute("title", "Connection: " + labelFor(state.current) + " · click for details");
    }
  }

  // ---------------------------------------------------------
  // DOM injection: diagnostics drawer (uses T12 modal tokens
  // when present, otherwise its own CSS).
  // ---------------------------------------------------------
  function buildDrawer() {
    if (state.drawerEl) return state.drawerEl;

    var backdrop = document.createElement("div");
    backdrop.className = "pfm-conn-drawer__backdrop";
    backdrop.addEventListener("click", closeDrawer);

    var drawer = document.createElement("aside");
    drawer.className = "pfm-conn-drawer";
    drawer.setAttribute("role", "dialog");
    drawer.setAttribute("aria-modal", "true");
    drawer.setAttribute("aria-label", "Connection diagnostics");

    var head = document.createElement("header");
    head.className = "pfm-conn-drawer__head";
    var title = document.createElement("h2");
    title.className = "pfm-conn-drawer__title";
    title.textContent = "Connection diagnostics";
    var close = document.createElement("button");
    close.type = "button";
    close.className = "pfm-conn-drawer__close";
    close.setAttribute("aria-label", "Close");
    close.innerHTML = "&times;";
    close.addEventListener("click", closeDrawer);
    head.appendChild(title);
    head.appendChild(close);

    var body = document.createElement("section");
    body.className = "pfm-conn-drawer__body";

    var summary = document.createElement("div");
    summary.className = "pfm-conn-drawer__summary";
    summary.innerHTML = ""
      + '<span class="pfm-conn-drawer__summary-dot"></span>'
      + '<span class="pfm-conn-drawer__summary-label">Status…</span>'
      + '<span class="pfm-conn-drawer__summary-meta"></span>';

    var list = document.createElement("ul");
    list.className = "pfm-conn-drawer__list";

    var footer = document.createElement("div");
    footer.className = "pfm-conn-drawer__footer";
    var refresh = document.createElement("button");
    refresh.type = "button";
    refresh.className = "pfm-conn-drawer__refresh";
    refresh.textContent = "Run heartbeat now";
    refresh.addEventListener("click", function () {
      refresh.disabled = true;
      probeOnce().finally(function () { refresh.disabled = false; });
    });
    var endpointLine = document.createElement("div");
    endpointLine.className = "pfm-conn-drawer__endpoint";
    footer.appendChild(endpointLine);
    footer.appendChild(refresh);

    body.appendChild(summary);
    body.appendChild(list);
    body.appendChild(footer);

    drawer.appendChild(head);
    drawer.appendChild(body);

    document.body.appendChild(backdrop);
    document.body.appendChild(drawer);

    state.drawerEl = drawer;
    state.backdropEl = backdrop;
    state.listEl = list;
    state.summaryEl = summary;
    state.refreshBtnEl = refresh;
    state._endpointLineEl = endpointLine;

    // ESC closes
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && drawer.classList.contains("is-open")) {
        closeDrawer();
      }
    });

    return drawer;
  }

  function openDrawer() {
    buildDrawer();
    renderDrawerIfOpen(true);
    state.drawerEl.classList.add("is-open");
    state.backdropEl.classList.add("is-open");
    // Probe immediately on open so user gets fresh data
    probeOnce();
  }

  function closeDrawer() {
    if (!state.drawerEl) return;
    state.drawerEl.classList.remove("is-open");
    state.backdropEl.classList.remove("is-open");
  }

  function renderDrawerIfOpen(force) {
    if (!state.drawerEl) return;
    if (!force && !state.drawerEl.classList.contains("is-open")) return;
    var s = state.current;
    state.summaryEl.setAttribute("data-state", s);
    var labelEl = state.summaryEl.querySelector(".pfm-conn-drawer__summary-label");
    var metaEl = state.summaryEl.querySelector(".pfm-conn-drawer__summary-meta");
    if (labelEl) labelEl.textContent = labelFor(s) + (s === "live" ? "  ·  all systems normal" : "");
    if (metaEl) {
      var bits = [];
      if (state.lastLatencyMs != null) bits.push(state.lastLatencyMs + " ms");
      if (state.lastSeenAt) bits.push(prettyTime(state.lastSeenAt));
      metaEl.textContent = bits.join("  ·  ");
    }

    // Per-source rows
    var list = state.listEl;
    list.innerHTML = "";
    if (!state.sources.length) {
      var empty = document.createElement("li");
      empty.className = "pfm-conn-drawer__empty";
      if (s === "offline") {
        empty.textContent = "No upstreams reachable. Check network / API server.";
      } else if (!state.deepEndpoint) {
        empty.textContent = "/health/deep not available — using legacy /health. Per-source detail will appear once /health/deep ships.";
      } else {
        empty.textContent = "No per-source data in response.";
      }
      list.appendChild(empty);
    } else {
      state.sources.forEach(function (src) {
        var li = document.createElement("li");
        li.className = "pfm-conn-drawer__row";
        li.setAttribute("data-ok", src.ok ? "1" : "0");

        var name = document.createElement("span");
        name.className = "pfm-conn-drawer__row-name";
        name.textContent = src.name;

        var lat = document.createElement("span");
        lat.className = "pfm-conn-drawer__row-latency";
        lat.textContent = (src.latency_ms != null) ? (src.latency_ms + " ms") : "—";

        var badge = document.createElement("span");
        badge.className = "pfm-conn-drawer__row-badge";
        badge.textContent = src.ok ? "OK" : "DOWN";

        li.appendChild(name);
        li.appendChild(lat);
        li.appendChild(badge);

        if (!src.ok && src.error) {
          var err = document.createElement("div");
          err.className = "pfm-conn-drawer__row-error";
          err.textContent = src.error;
          li.appendChild(err);
        }

        list.appendChild(li);
      });
    }

    if (state._endpointLineEl) {
      var base = getApiBase() || "(same-origin)";
      var path = state.deepEndpoint ? "/health/deep" : "/health";
      state._endpointLineEl.textContent = "Probing  " + base + path + "  every 30s";
    }
  }

  function prettyTime(iso) {
    try {
      var d = new Date(iso);
      var hh = String(d.getHours()).padStart(2, "0");
      var mm = String(d.getMinutes()).padStart(2, "0");
      var ss = String(d.getSeconds()).padStart(2, "0");
      return hh + ":" + mm + ":" + ss;
    } catch (_) { return iso || ""; }
  }

  // ---------------------------------------------------------
  // Visibility handling: pause when hidden, resume on visible.
  // ---------------------------------------------------------
  function onVisibilityChange() {
    if (document.hidden) {
      // Just suspend the timer; don't clear state.
      if (state.timerId != null) {
        clearTimeout(state.timerId);
        state.timerId = null;
      }
    } else if (!state.paused) {
      // Tab became visible. Probe immediately and reset debounce
      // counters so a long-stale state doesn't linger.
      state.candidate = null;
      state.candidateCount = 0;
      probeOnce().finally(scheduleNext);
    }
  }

  // ---------------------------------------------------------
  // Hide any legacy `.pfm-conn-status` element shipped inline
  // in web/index.html so the two don't visually fight.
  // (Defensive — index-html-owner should remove it eventually.)
  // ---------------------------------------------------------
  function hideLegacy() {
    try {
      var legacy = document.getElementById("pfm-conn-status");
      if (legacy) {
        legacy.style.display = "none";
        legacy.setAttribute("aria-hidden", "true");
      }
    } catch (_) {}
  }

  // ---------------------------------------------------------
  // Public API
  // ---------------------------------------------------------
  var api = {
    __initialized: true,
    current: currentSnapshot,
    heartbeat: function () {
      return probeOnce();
    },
    pause: function () {
      state.paused = true;
      if (state.timerId != null) {
        clearTimeout(state.timerId);
        state.timerId = null;
      }
    },
    resume: function () {
      if (!state.paused) {
        // already running — no-op
        return;
      }
      state.paused = false;
      probeOnce().finally(scheduleNext);
    },
    onChange: function (cb) {
      if (typeof cb !== "function") return function () {};
      state.listeners.push(cb);
      return function () {
        var idx = state.listeners.indexOf(cb);
        if (idx >= 0) state.listeners.splice(idx, 1);
      };
    }
  };
  window.PFM.conn = api;

  // ---------------------------------------------------------
  // Bootstrap
  // ---------------------------------------------------------
  function start() {
    hideLegacy();
    buildPill();
    renderPill();
    document.addEventListener("visibilitychange", onVisibilityChange, false);
    waitForApiBase().then(function () {
      // First probe AFTER PFM_API_BASE is resolved (or timeout).
      // This is the central fix for the "flickers Degraded on
      // first paint" bug.
      probeOnce().finally(scheduleNext);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
