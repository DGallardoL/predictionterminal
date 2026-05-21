/* ============================================================
 * a11y-fixes.js  (W13-31, wave-13)
 *
 * Runtime keyboard-accessibility patcher. Companion to
 * web/css/a11y-fixes.css.
 *
 * RESPONSIBILITIES
 *   1. Inject skip-to-main-content link as the FIRST focusable
 *      element on the page (if not already present).
 *   2. Inject #a11y-main anchor on a sensible main landmark
 *      (the first <main>, or the first .pane.active, or body).
 *   3. Inject ARIA live regions (#a11y-live polite,
 *      #a11y-live-assertive) and expose
 *      window.A11y.announce(msg, urgent=false).
 *   4. Scan the document for interactive-looking elements that
 *      lack tabindex / role / aria-label and PATCH them at
 *      runtime. Each patch logs a console.warn so the team
 *      can fix the source HTML over time.
 *   5. Universal ESC handler: closes any open .modal, .sheet,
 *      .drawer, [role="dialog"], [aria-modal="true"], or
 *      element marked [data-a11y-dismissible].
 *   6. Tab-key sanity: no logic needed beyond browsers'
 *      default, BUT we ensure no element has tabindex > 0
 *      (which corrupts logical tab order). Anything > 0 is
 *      rewritten to 0 + warning.
 *   7. Arrow-key roving-tabindex for groups marked with
 *      [data-a11y-arrow-nav] (tablists, menus, card grids).
 *   8. Space / Enter activation for custom controls that have
 *      role="button" but no native click semantics.
 *
 * COORDINATION
 *   - Self-contained. Reads window/document only.
 *   - Idempotent: safe to load multiple times.
 *   - Uses MutationObserver so SPA-style mode switches are
 *     re-patched without re-load.
 * ============================================================ */

(function () {
  "use strict";

  if (window.__a11yFixesInstalled) {
    return; // idempotent
  }
  window.__a11yFixesInstalled = true;

  // -------------------------------------------------------- log
  var DEBUG = true;
  function warn() {
    if (!DEBUG) return;
    try {
      var args = ["[a11y-fixes]"].concat(Array.prototype.slice.call(arguments));
      console.warn.apply(console, args);
    } catch (e) {
      /* noop */
    }
  }

  // ============================================================
  // 1. SKIP LINK + MAIN ANCHOR
  // ============================================================
  function ensureSkipLink() {
    if (document.querySelector(".a11y-skip-link")) return;
    var a = document.createElement("a");
    a.className = "a11y-skip-link";
    a.href = "#a11y-main";
    a.textContent = "Skip to main content";
    a.setAttribute("data-a11y-injected", "true");
    if (document.body.firstChild) {
      document.body.insertBefore(a, document.body.firstChild);
    } else {
      document.body.appendChild(a);
    }
  }

  function ensureMainAnchor() {
    if (document.getElementById("a11y-main")) return;
    var target =
      document.querySelector("main") ||
      document.querySelector('[role="main"]') ||
      document.querySelector(".pane.active") ||
      document.querySelector(".pane") ||
      document.querySelector("#app") ||
      document.body;
    if (!target) return;
    target.setAttribute("id", target.id || "a11y-main");
    if (target.id !== "a11y-main") {
      // alias: add a tiny invisible anchor
      var anchor = document.createElement("span");
      anchor.id = "a11y-main";
      anchor.tabIndex = -1;
      anchor.setAttribute("data-a11y-injected", "true");
      target.parentNode.insertBefore(anchor, target);
    } else {
      target.tabIndex = -1;
    }
  }

  // ============================================================
  // 2. ARIA LIVE REGIONS
  // ============================================================
  function ensureLiveRegions() {
    if (!document.getElementById("a11y-live")) {
      var p = document.createElement("div");
      p.id = "a11y-live";
      p.setAttribute("role", "status");
      p.setAttribute("aria-live", "polite");
      p.setAttribute("aria-atomic", "true");
      document.body.appendChild(p);
    }
    if (!document.getElementById("a11y-live-assertive")) {
      var a = document.createElement("div");
      a.id = "a11y-live-assertive";
      a.setAttribute("role", "alert");
      a.setAttribute("aria-live", "assertive");
      a.setAttribute("aria-atomic", "true");
      document.body.appendChild(a);
    }
  }

  function announce(msg, urgent) {
    if (typeof msg !== "string" || !msg) return;
    var id = urgent ? "a11y-live-assertive" : "a11y-live";
    var el = document.getElementById(id);
    if (!el) return;
    // toggle to force re-announce
    el.textContent = "";
    setTimeout(function () {
      el.textContent = msg;
    }, 50);
  }

  // ============================================================
  // 3. INTERACTIVE-ELEMENT PATCHER
  // ============================================================
  // Selectors for elements that LOOK interactive but may lack
  // proper a11y attrs. We try to infer the intended role.
  var CUSTOM_BUTTON_SELECTORS = [
    ".btn",
    ".button",
    ".pill",
    ".chip",
    ".tab",
    ".card-clickable",
    "[data-action]",
    "[data-toggle]",
    "[data-href]",
  ].join(",");

  function isNativelyInteractive(el) {
    var tag = (el.tagName || "").toLowerCase();
    return (
      tag === "button" ||
      tag === "a" ||
      tag === "input" ||
      tag === "select" ||
      tag === "textarea" ||
      tag === "summary" ||
      tag === "details"
    );
  }

  function inferAriaLabel(el) {
    if (el.getAttribute("aria-label")) return null;
    if (el.getAttribute("aria-labelledby")) return null;
    if (el.getAttribute("title")) return el.getAttribute("title");
    var text = (el.textContent || "").trim().replace(/\s+/g, " ");
    if (text && text.length < 80) return text;
    // icon-only?
    var icon = el.querySelector("svg, .icon, [class*='icon']");
    if (icon) {
      var data = el.getAttribute("data-action") || el.getAttribute("data-toggle");
      if (data) return data.replace(/[-_]/g, " ");
    }
    return null;
  }

  function patchOne(el) {
    if (!el || el.nodeType !== 1) return;
    if (el.classList.contains("a11y-patched")) return;
    if (el.hasAttribute("data-a11y-skip")) return;

    var changed = false;

    // 1) Force tabindex onto custom controls
    if (!isNativelyInteractive(el)) {
      var tab = el.getAttribute("tabindex");
      if (tab === null || tab === undefined) {
        el.setAttribute("tabindex", "0");
        changed = true;
        warn("Added tabindex=0 to custom control:", el);
      } else if (parseInt(tab, 10) > 0) {
        // Positive tabindex destroys logical tab order; rewrite to 0.
        warn(
          "Rewriting tabindex=" + tab + " to 0 (positive values harm tab order):",
          el
        );
        el.setAttribute("tabindex", "0");
        changed = true;
      }
      // 2) Role
      if (!el.getAttribute("role")) {
        if (el.matches('[role="link"], [data-href]')) {
          el.setAttribute("role", "link");
        } else {
          el.setAttribute("role", "button");
        }
        changed = true;
      }
    } else {
      // Native: still check for stray positive tabindex
      var ntab = el.getAttribute("tabindex");
      if (ntab !== null && parseInt(ntab, 10) > 0) {
        warn(
          "Native element has positive tabindex=" + ntab + "; rewriting to 0:",
          el
        );
        el.setAttribute("tabindex", "0");
        changed = true;
      }
    }

    // 3) aria-label fallback
    if (!el.getAttribute("aria-label") && !el.getAttribute("aria-labelledby")) {
      var inferred = inferAriaLabel(el);
      if (inferred) {
        // Only inject if the visible text is missing (icon-only) OR
        // the element is a custom control. Native <button> with
        // text content needs no aria-label.
        var visible = (el.textContent || "").trim();
        if (!visible) {
          el.setAttribute("aria-label", inferred);
          changed = true;
          warn("Added aria-label='" + inferred + "' to:", el);
        }
      } else if (!isNativelyInteractive(el)) {
        warn("Custom control has no inferable label:", el);
      }
    }

    if (changed) {
      el.classList.add("a11y-patched");
    }
  }

  function patchAll(root) {
    root = root || document;
    var nodes;
    try {
      nodes = root.querySelectorAll(CUSTOM_BUTTON_SELECTORS);
    } catch (e) {
      return;
    }
    for (var i = 0; i < nodes.length; i++) {
      patchOne(nodes[i]);
    }
    // Also scan tabindex>0 across all elements
    var positives = root.querySelectorAll("[tabindex]");
    for (var j = 0; j < positives.length; j++) {
      var t = positives[j].getAttribute("tabindex");
      if (t !== null && parseInt(t, 10) > 0) {
        patchOne(positives[j]);
      }
    }
  }

  // ============================================================
  // 4. UNIVERSAL ESC HANDLER
  // ============================================================
  var DISMISSIBLE_SELECTORS = [
    '.modal.is-open',
    '.modal.open',
    '.modal[aria-hidden="false"]',
    '.sheet.is-open',
    '.drawer.is-open',
    '.drawer.open',
    '[role="dialog"][aria-modal="true"]',
    '[data-a11y-dismissible="true"]',
  ].join(",");

  function dismissTopmost() {
    var candidates = document.querySelectorAll(DISMISSIBLE_SELECTORS);
    if (!candidates.length) return false;
    // pick topmost by z-index then DOM order
    var topmost = null;
    var topZ = -Infinity;
    for (var i = 0; i < candidates.length; i++) {
      var c = candidates[i];
      var cs = window.getComputedStyle(c);
      if (cs.display === "none" || cs.visibility === "hidden") continue;
      var z = parseInt(cs.zIndex, 10);
      if (isNaN(z)) z = 0;
      if (z >= topZ) {
        topZ = z;
        topmost = c;
      }
    }
    if (!topmost) return false;
    // Prefer a close button click for app-managed lifecycle
    var closeBtn = topmost.querySelector(
      "[data-close], .close-btn, .modal-close, [aria-label='Close']"
    );
    if (closeBtn) {
      try {
        closeBtn.click();
        return true;
      } catch (e) {
        /* fall through */
      }
    }
    // Fallback: hide directly
    topmost.classList.remove("is-open", "open");
    topmost.setAttribute("aria-hidden", "true");
    try {
      topmost.dispatchEvent(new CustomEvent("a11y:dismiss", { bubbles: true }));
    } catch (e) {
      /* noop */
    }
    return true;
  }

  function onKeyDown(e) {
    // ESC: close topmost dismissible
    if (e.key === "Escape" || e.keyCode === 27) {
      if (dismissTopmost()) {
        e.stopPropagation();
      }
      return;
    }

    var target = e.target;
    if (!target || target.nodeType !== 1) return;

    // Space / Enter activate custom buttons
    if (
      (e.key === " " || e.key === "Spacebar" || e.key === "Enter") &&
      !isNativelyInteractive(target) &&
      (target.getAttribute("role") === "button" ||
        target.getAttribute("role") === "tab" ||
        target.getAttribute("role") === "menuitem")
    ) {
      // Don't hijack textareas / contenteditable
      if (target.isContentEditable) return;
      e.preventDefault();
      try {
        target.click();
      } catch (err) {
        /* noop */
      }
      return;
    }

    // Arrow-key roving navigation
    var group = target.closest && target.closest("[data-a11y-arrow-nav]");
    if (group && /^Arrow/.test(e.key)) {
      handleArrowNav(e, group, target);
    }
  }

  // ============================================================
  // 5. ARROW-KEY ROVING NAVIGATION
  // ============================================================
  function getRovingItems(group) {
    var sel =
      group.getAttribute("data-a11y-arrow-items") ||
      '[role="tab"], [role="menuitem"], [role="option"], .a11y-rove-item';
    return Array.prototype.slice.call(group.querySelectorAll(sel)).filter(function (el) {
      return !el.hasAttribute("disabled") && el.tabIndex !== -1;
    });
  }

  function handleArrowNav(e, group, current) {
    var items = getRovingItems(group);
    if (!items.length) return;
    var idx = items.indexOf(current);
    if (idx === -1) return;
    var orientation =
      group.getAttribute("aria-orientation") ||
      group.getAttribute("data-a11y-orientation") ||
      "horizontal";
    var horizontal = orientation === "horizontal";
    var next = idx;
    if ((horizontal && e.key === "ArrowRight") || (!horizontal && e.key === "ArrowDown")) {
      next = (idx + 1) % items.length;
    } else if ((horizontal && e.key === "ArrowLeft") || (!horizontal && e.key === "ArrowUp")) {
      next = (idx - 1 + items.length) % items.length;
    } else if (e.key === "Home") {
      next = 0;
    } else if (e.key === "End") {
      next = items.length - 1;
    } else {
      return; // not a key we handle
    }
    e.preventDefault();
    // Update roving state
    items.forEach(function (it, i) {
      it.setAttribute("tabindex", i === next ? "0" : "-1");
      if (i === next) {
        it.setAttribute("data-a11y-roving-active", "true");
      } else {
        it.removeAttribute("data-a11y-roving-active");
      }
    });
    try {
      items[next].focus();
    } catch (err) {
      /* noop */
    }
  }

  // ============================================================
  // 6. MUTATION OBSERVER (SPA dynamic content)
  // ============================================================
  function observe() {
    if (!window.MutationObserver) return;
    var obs = new MutationObserver(function (muts) {
      for (var i = 0; i < muts.length; i++) {
        var m = muts[i];
        if (m.type === "childList") {
          for (var j = 0; j < m.addedNodes.length; j++) {
            var n = m.addedNodes[j];
            if (n.nodeType === 1) {
              patchAll(n);
            }
          }
        }
      }
    });
    obs.observe(document.body, { childList: true, subtree: true });
  }

  // ============================================================
  // INIT
  // ============================================================
  function init() {
    try {
      ensureSkipLink();
      ensureMainAnchor();
      ensureLiveRegions();
      patchAll(document);
      document.addEventListener("keydown", onKeyDown, true);
      observe();

      // Expose minimal public API
      window.A11y = window.A11y || {};
      window.A11y.announce = announce;
      window.A11y.patch = patchAll;
      window.A11y.dismissTopmost = dismissTopmost;
      window.A11y.version = "W13-31";

      // Small first-load announcement (polite)
      announce("Keyboard accessibility enabled.", false);
    } catch (err) {
      try {
        console.error("[a11y-fixes] init failed:", err);
      } catch (e) {
        /* noop */
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
