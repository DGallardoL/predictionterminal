/* ============================================================
 * research-tab.js  (T57, wave-research-tab)
 *
 * Frontend for the α Hub → Research sub-tab. Renders a grid of
 * "Research Reports" cards from T31's GET /research/reports
 * endpoint. Clicking a card opens a modal with the full markdown
 * body fetched from GET /research/reports/{version}?format=html.
 *
 * Public API:
 *   window.PFM.research = {
 *     mount(containerEl, opts?),   // mount UI into containerEl
 *     refresh(),                   // re-fetch + re-render
 *   }
 *
 * Mount target:
 *   The Strategies pane already has a sub-pane:
 *     <div class="strat-pane" data-spane="research-reports">
 *   Index.html ships an empty placeholder there (T57 does NOT
 *   edit index.html). Callers (or an index-html-owner) invoke:
 *     window.PFM.research.mount(
 *       document.querySelector('[data-spane="research-reports"]')
 *     );
 *
 * Graceful degradation:
 *   - If GET /research/reports 404s or errors, show empty state
 *     with "Research API not available yet — re-run after T31".
 *   - If the API returns [], show "No research reports found".
 *
 * Dependencies:
 *   - web/css/research-tab.css (T57)
 *   - web/css/data-cards.css (T06, .dc-card primitives)
 *   - web/css/modal.css (T12, .modal / .modal-backdrop)
 *   - web/css/skeletons.css (T04, .skel primitives)
 *   - window.PFM.apiBase (optional) — defaults to "" (same-origin)
 * ============================================================ */

(function () {
  "use strict";

  if (window.PFM && window.PFM.research && window.PFM.research.__t57) {
    /* already mounted by another loader */
    return;
  }

  /* ------------------------------------------------------------
   * Constants + small helpers
   * ------------------------------------------------------------ */

  const API_BASE_FALLBACKS = [
    () => (window.PFM && window.PFM.apiBase) || null,
    () => window.PFM_API_BASE || null,
    () => "",
  ];

  function apiBase() {
    // Walk the fallbacks and pick the FIRST non-null value. Previously each
    // fallback `|| ""`-ed itself, so the first one always returned the empty
    // string and short-circuited the loop — meaning window.PFM_API_BASE was
    // never consulted, and requests hit :8080 instead of the API origin.
    for (const fn of API_BASE_FALLBACKS) {
      try {
        const v = fn();
        if (typeof v === "string") return v.replace(/\/+$/, "");
      } catch (_e) { /* try next */ }
    }
    return "";
  }

  function escapeHtml(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fmtDate(iso) {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return String(iso);
      return d.toLocaleDateString(undefined, {
        year: "numeric", month: "short", day: "2-digit",
      });
    } catch (_e) {
      return String(iso);
    }
  }

  function dateSortValue(report) {
    /* Prefer published_at; fall back to version number so ordering
       still makes sense if the API omits dates. */
    const t = Date.parse(report.published_at || report.date || "");
    if (!Number.isNaN(t)) return t;
    return (typeof report.version === "number" ? report.version : 0) * 86400000;
  }

  function readVersion(r) {
    /* The endpoint may return either an integer ("version": 17) or
       a string ("v17"). Normalize to integer where possible. */
    if (typeof r.version === "number") return r.version;
    if (typeof r.version === "string") {
      const m = r.version.match(/(\d+)/);
      if (m) return parseInt(m[1], 10);
    }
    return null;
  }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const k of Object.keys(attrs)) {
        if (k === "className") node.className = attrs[k];
        else if (k === "html") node.innerHTML = attrs[k];
        else if (k === "text") node.textContent = attrs[k];
        else if (k.startsWith("on") && typeof attrs[k] === "function") {
          node.addEventListener(k.slice(2).toLowerCase(), attrs[k]);
        } else if (attrs[k] != null) {
          node.setAttribute(k, attrs[k]);
        }
      }
    }
    if (children) {
      for (const c of [].concat(children)) {
        if (c == null) continue;
        node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
      }
    }
    return node;
  }

  /* ------------------------------------------------------------
   * Module-level state (one container per page)
   * ------------------------------------------------------------ */

  const state = {
    container: null,
    gridEl: null,
    sortEl: null,
    refreshBtn: null,
    sortMode: "newest", // "newest" | "oldest" | "alpha-count"
    reports: null,     // null = not loaded yet; [] = empty
    loading: false,
    error: null,
    apiAvailable: null, // null = unknown, true/false after first fetch
  };

  /* ------------------------------------------------------------
   * Fetch layer
   * ------------------------------------------------------------ */

  async function fetchReports() {
    const url = apiBase() + "/research/reports";
    let res;
    try {
      res = await fetch(url, {
        cache: "no-cache",
        headers: { "Accept": "application/json" },
      });
    } catch (e) {
      throw new Error("network:" + (e && e.message ? e.message : "unknown"));
    }
    if (res.status === 404) {
      throw new Error("not-implemented");
    }
    if (!res.ok) {
      throw new Error("http:" + res.status);
    }
    const data = await res.json();
    /* Endpoint may return {reports: [...]} or [...] directly. */
    const list = Array.isArray(data) ? data : (data && Array.isArray(data.reports) ? data.reports : []);
    return list.map(normaliseReport);
  }

  function normaliseReport(r) {
    /* Defensive normaliser so card render code can rely on shape. */
    return {
      version: readVersion(r),
      version_label: r.version_label || (typeof r.version === "string" ? r.version : (r.version != null ? "v" + r.version : "v?")),
      title: r.title || r.name || "Untitled report",
      published_at: r.published_at || r.date || r.published || null,
      summary: r.summary || r.abstract || r.description || "",
      deployable_count: typeof r.deployable_count === "number" ? r.deployable_count
        : (typeof r.deployable === "number" ? r.deployable
          : (Array.isArray(r.deployable) ? r.deployable.length : null)),
      anti_alpha_count: typeof r.anti_alpha_count === "number" ? r.anti_alpha_count
        : (typeof r.anti_alpha === "number" ? r.anti_alpha
          : (Array.isArray(r.anti_alpha) ? r.anti_alpha.length : null)),
      slug: r.slug || (r.version != null ? String(r.version) : null),
      current: !!r.current,
    };
  }

  async function fetchReportHtml(versionKey) {
    const url = apiBase() + "/research/reports/" + encodeURIComponent(versionKey) + "?format=html";
    const res = await fetch(url, {
      cache: "no-cache",
      headers: { "Accept": "text/html,application/json" },
    });
    if (!res.ok) {
      throw new Error("http:" + res.status);
    }
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    if (ct.indexOf("application/json") !== -1) {
      const j = await res.json();
      if (typeof j === "string") return j;
      if (j && typeof j.html === "string") return j.html;
      if (j && typeof j.body === "string") return j.body;
      return "<pre>" + escapeHtml(JSON.stringify(j, null, 2)) + "</pre>";
    }
    const text = await res.text();
    /* If the API fell back to text/markdown (e.g. the optional
       `markdown` library isn't installed in the API env), wrap it
       in <pre> so the modal body still renders legibly. */
    if (ct.indexOf("text/markdown") !== -1 || ct.indexOf("text/plain") !== -1) {
      return "<pre>" + escapeHtml(text) + "</pre>";
    }
    return text;
  }

  /* ------------------------------------------------------------
   * Rendering
   * ------------------------------------------------------------ */

  function renderHeader(rootEl) {
    const header = el("div", { className: "pfm-research__header" }, [
      el("h2", { className: "pfm-research__title", text: "Research Reports" }),
      el("div", { className: "pfm-research__controls" }, [
        (function () {
          const sel = el("select", {
            className: "pfm-research__sort",
            "aria-label": "Sort research reports",
            title: "Sort reports",
            onchange: (e) => { state.sortMode = e.target.value; renderGrid(); },
          }, [
            el("option", { value: "newest", text: "Newest first" }),
            el("option", { value: "oldest", text: "Oldest first" }),
            el("option", { value: "alpha-count", text: "By alpha count" }),
          ]);
          sel.value = state.sortMode;
          state.sortEl = sel;
          return sel;
        })(),
        (function () {
          const btn = el("button", {
            type: "button",
            className: "pfm-research__refresh",
            "aria-label": "Refresh research reports",
            title: "Refresh",
            onclick: () => { api.refresh(); },
          });
          btn.innerHTML =
            '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
            '<path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9"/>' +
            '<polyline points="13.5 2 13.5 4.1 11.4 4.1"/>' +
            "</svg>";
          state.refreshBtn = btn;
          return btn;
        })(),
      ]),
    ]);
    rootEl.appendChild(header);
  }

  function renderSkeletonGrid() {
    const grid = el("div", { className: "pfm-research__grid", role: "list" });
    for (let i = 0; i < 6; i++) {
      const card = el("div", { className: "pfm-research__skel-card", "aria-hidden": "true" }, [
        el("div", { className: "pfm-research__skel-row" }, [
          el("span", { className: "skel skel--line-md" }),
          el("span", { className: "skel skel--line-sm" }),
        ]),
        el("span", { className: "skel skel--line-lg" }),
        el("span", { className: "skel skel--line-md" }),
        el("span", { className: "skel skel--line-sm" }),
        el("div", { className: "pfm-research__skel-row", style: "margin-top:auto;" }, [
          el("span", { className: "skel skel--line-sm" }),
          el("span", { className: "skel skel--line-sm" }),
        ]),
      ]);
      grid.appendChild(card);
    }
    return grid;
  }

  function renderEmptyState(message, hint) {
    return el("div", { className: "pfm-research__empty", role: "status" }, [
      el("p", { className: "pfm-research__empty-title", text: message }),
      hint ? el("p", { className: "pfm-research__empty-body", text: hint }) : null,
    ]);
  }

  function buildCard(report) {
    const verLabel = report.version_label || ("v" + (report.version != null ? report.version : "?"));
    const dateStr = fmtDate(report.published_at);
    const dep = (report.deployable_count != null) ? report.deployable_count : "—";
    const anti = (report.anti_alpha_count != null) ? report.anti_alpha_count : "—";

    const card = el("button", {
      type: "button",
      className: "dc-card pfm-research-card is-interactive",
      "data-version": String(report.version != null ? report.version : (report.slug || "")),
      "aria-label": "Open research report " + verLabel + (report.title ? " — " + report.title : ""),
      onclick: () => openReportModal(report),
    }, [
      el("div", { className: "pfm-research-card__top" }, [
        el("h3", { className: "pfm-research-card__title", text: report.title }),
        el("span", { className: "pfm-research-card__version", text: verLabel }),
      ]),
      el("div", { className: "pfm-research-card__date", text: dateStr || "—" }),
      el("p", { className: "pfm-research-card__summary",
        text: report.summary || "(no summary)" }),
      el("div", { className: "pfm-research-card__counts", role: "group", "aria-label": "Report counts" }, [
        el("span", { className: "pfm-research-card__count" }, [
          el("span", { className: "pfm-research-card__count-value is-pos", text: String(dep) }),
          el("span", { className: "pfm-research-card__count-label", text: "deployable" }),
        ]),
        el("span", { className: "pfm-research-card__count" }, [
          el("span", { className: "pfm-research-card__count-value is-neg", text: String(anti) }),
          el("span", { className: "pfm-research-card__count-label", text: "anti-α" }),
        ]),
      ]),
    ]);
    return card;
  }

  function sortReports(reports, mode) {
    const arr = reports.slice();
    if (mode === "oldest") {
      arr.sort((a, b) => dateSortValue(a) - dateSortValue(b));
    } else if (mode === "alpha-count") {
      arr.sort((a, b) => (b.deployable_count || 0) - (a.deployable_count || 0));
    } else {
      arr.sort((a, b) => dateSortValue(b) - dateSortValue(a));
    }
    return arr;
  }

  function renderGrid() {
    if (!state.gridEl) return;
    const slot = state.gridEl;
    slot.innerHTML = "";

    if (state.loading) {
      slot.appendChild(renderSkeletonGrid());
      return;
    }

    if (state.error === "not-implemented") {
      slot.appendChild(renderEmptyState(
        "Research API not available yet",
        "Re-run after T31 (GET /research/reports) lands."
      ));
      return;
    }
    if (state.error) {
      slot.appendChild(renderEmptyState(
        "Could not load research reports",
        "Error: " + state.error + ". Click refresh to retry."
      ));
      return;
    }

    if (!state.reports || state.reports.length === 0) {
      slot.appendChild(renderEmptyState("No research reports found", null));
      return;
    }

    const grid = el("div", { className: "pfm-research__grid", role: "list" });
    const sorted = sortReports(state.reports, state.sortMode);
    for (const r of sorted) {
      grid.appendChild(buildCard(r));
    }
    slot.appendChild(grid);
  }

  /* ------------------------------------------------------------
   * Modal (full markdown body)
   * ------------------------------------------------------------ */

  function closeModal(backdrop) {
    if (!backdrop || !backdrop.parentNode) return;
    backdrop.classList.add("is-leaving");
    const modal = backdrop.querySelector(".modal");
    if (modal) modal.classList.add("is-leaving");
    setTimeout(() => {
      if (backdrop.parentNode) backdrop.parentNode.removeChild(backdrop);
    }, 180);
    document.removeEventListener("keydown", backdrop.__pfmEsc);
  }

  function openReportModal(report) {
    const verLabel = report.version_label || ("v" + report.version);
    const verKey = (report.version != null) ? String(report.version) : (report.slug || verLabel);

    const titleEl = el("div", { className: "modal__title" }, [
      el("span", { text: verLabel + " — " }),
      el("span", { text: report.title }),
    ]);

    const closeBtn = el("button", {
      type: "button",
      className: "modal__close",
      "aria-label": "Close research report",
      text: "×",
    });

    const header = el("header", { className: "modal__header" }, [
      titleEl, closeBtn,
    ]);

    const metaBits = [];
    if (report.published_at) {
      metaBits.push(el("span", {}, [
        el("strong", { text: "Published " }),
        document.createTextNode(fmtDate(report.published_at)),
      ]));
    }
    if (report.deployable_count != null) {
      metaBits.push(el("span", {}, [
        el("strong", { text: String(report.deployable_count) + " " }),
        document.createTextNode("deployable"),
      ]));
    }
    if (report.anti_alpha_count != null) {
      metaBits.push(el("span", {}, [
        el("strong", { text: String(report.anti_alpha_count) + " " }),
        document.createTextNode("anti-α"),
      ]));
    }
    const meta = el("div", { className: "pfm-research-modal__meta" }, metaBits);

    const bodyHost = el("div", { className: "pfm-research-modal__body" });
    bodyHost.innerHTML =
      '<div class="pfm-research-modal__loading">' +
      '<span class="skel skel--line-lg"></span>' +
      '<span class="skel skel--line-md"></span>' +
      '<span class="skel skel--line-md"></span>' +
      '<span class="skel skel--line-sm"></span>' +
      '<span class="skel skel--line-md"></span>' +
      "</div>";

    const body = el("div", { className: "modal__body" }, [meta, bodyHost]);

    const modal = el("div", {
      className: "modal modal--lg",
      role: "dialog",
      "aria-modal": "true",
      "aria-labelledby": "pfm-research-modal-title",
      "data-modal-depth": "1",
    }, [header, body]);

    titleEl.id = "pfm-research-modal-title";

    const backdrop = el("div", {
      className: "modal-backdrop",
      role: "presentation",
      "data-modal-depth": "1",
    }, [modal]);

    /* close handlers */
    function onEsc(e) { if (e.key === "Escape") closeModal(backdrop); }
    backdrop.__pfmEsc = onEsc;
    document.addEventListener("keydown", onEsc);
    closeBtn.addEventListener("click", () => closeModal(backdrop));
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) closeModal(backdrop);
    });

    document.body.appendChild(backdrop);

    /* Fetch the rendered HTML body. */
    fetchReportHtml(verKey).then((html) => {
      /* The API is trusted (same-origin internal service). If a future
         maintainer wires this to a third-party, swap to a sanitiser. */
      bodyHost.innerHTML = html || '<p class="pfm-research-modal__body">(empty)</p>';
    }).catch((err) => {
      bodyHost.innerHTML =
        '<p style="color:var(--neg, #dc2626); font-family:var(--sans, Inter, system-ui, sans-serif); font-size:13px;">' +
        "Could not load report body: " + escapeHtml(err && err.message ? err.message : String(err)) +
        "</p>";
    });
  }

  /* ------------------------------------------------------------
   * Public API
   * ------------------------------------------------------------ */

  async function loadAndRender() {
    if (!state.container) return;
    state.loading = true;
    state.error = null;
    if (state.refreshBtn) state.refreshBtn.classList.add("is-spinning");
    renderGrid();

    try {
      const reports = await fetchReports();
      state.reports = reports;
      state.apiAvailable = true;
    } catch (e) {
      const msg = (e && e.message) ? e.message : String(e);
      if (msg === "not-implemented" || /^http:404/.test(msg)) {
        state.error = "not-implemented";
        state.apiAvailable = false;
      } else {
        state.error = msg;
      }
      state.reports = state.reports || [];
    } finally {
      state.loading = false;
      if (state.refreshBtn) state.refreshBtn.classList.remove("is-spinning");
      renderGrid();
    }
  }

  const api = {
    __t57: true,

    mount(containerEl, _opts) {
      if (!containerEl) {
        console.warn("[research-tab] mount() called without a container");
        return;
      }
      /* Idempotent: if already mounted into the same container, just
         re-render. */
      if (state.container === containerEl && state.gridEl) {
        renderGrid();
        return;
      }
      state.container = containerEl;

      /* Build a fresh root wrapper. We REPLACE existing children of
         the pane only when the pane has no .pfm-research already. */
      let root = containerEl.querySelector(".pfm-research");
      if (!root) {
        root = el("div", { className: "pfm-research" });
        /* Clear loading placeholders that may have been hardcoded in
           index.html (#research-reports-summary, #research-reports-list).
           We leave any sibling node Damian might have added intentionally
           alongside, so this only removes the two known placeholders. */
        const stale = containerEl.querySelectorAll(
          "#research-reports-summary, #research-reports-list, .term-empty-mini"
        );
        stale.forEach((n) => { if (n && n.parentNode) n.parentNode.removeChild(n); });
        containerEl.appendChild(root);
      } else {
        root.innerHTML = "";
      }

      renderHeader(root);
      state.gridEl = el("div", { className: "pfm-research__grid-slot" });
      root.appendChild(state.gridEl);

      /* Initial render with skeletons then fetch. */
      state.loading = true;
      renderGrid();
      loadAndRender();
    },

    refresh() {
      if (!state.container) return;
      loadAndRender();
    },
  };

  window.PFM = window.PFM || {};
  window.PFM.research = api;

  /* ------------------------------------------------------------
   * Auto-mount when the Research sub-tab becomes active.
   *
   * The α Hub already wires tab clicks to swap `.strat-pane`
   * visibility via [data-spane="research-reports"]. We do NOT
   * fight that; we simply auto-mount the first time the pane is
   * present in the DOM (and not yet mounted). Safe to call mount
   * again from external code — it's idempotent.
   * ------------------------------------------------------------ */
  function autoMountWhenAvailable() {
    const pane = document.querySelector('[data-spane="research-reports"]');
    if (!pane) return false;
    if (pane.querySelector(".pfm-research")) return true;
    api.mount(pane);
    return true;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => { autoMountWhenAvailable(); });
  } else {
    /* defer one tick so other late-init scripts complete first */
    setTimeout(autoMountWhenAvailable, 0);
  }

  /* Observe DOM in case the pane appears later (e.g. lazy-rendered
     mode panes). Disconnect once mounted. */
  try {
    const mo = new MutationObserver(() => {
      if (autoMountWhenAvailable()) mo.disconnect();
    });
    mo.observe(document.documentElement, { childList: true, subtree: true });
    /* Safety: stop after 30 s of unsuccessful waiting. */
    setTimeout(() => mo.disconnect(), 30000);
  } catch (_e) { /* MutationObserver unsupported - ignore */ }

})();
