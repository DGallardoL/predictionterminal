"""Visual regression tests for the Prediction Terminal frontend.

What this pins
--------------
We render `web/index.html` directly via the `file://` protocol in headless
Chromium and capture full-page PNG screenshots of a handful of high-value UI
states. Each screenshot is compared against a checked-in baseline under
`tests/visual/baselines/`. A scenario fails iff more than 0.5% of the pixels
differ from the baseline (font hinting + sub-pixel AA produce tiny per-pixel
deltas that should not flake the build).

Scenarios pinned
----------------
1. Empty Regression form (`regression-empty`)
2. Post-fit toast + sticky card via the `window.PFM.events` bridge
   (`regression-after-fit`)
3. Terminal landing page (`terminal-landing`)
4. Strategies / Alpha Hub landing (`strategies-alphahub`)
5. Dark mode on the Regression pane (`regression-dark`)
6. Mobile viewport (375x667) on Regression (`regression-mobile`)

Skipping policy
---------------
- If `playwright` is not importable, the entire module skips via
  `pytest.importorskip`.
- If launching headless Chromium raises *any* error (no sandbox, missing
  browser binary, missing system libs), every scenario individually skips
  with reason `"headless browser unavailable: <repr>"`.
- If PIL is not importable, the entire module skips. (PIL is the comparison
  backend when `pixelmatch` is not available.)
- If a baseline does not yet exist, the scenario skips with a hint to run
  with `PYTEST_UPDATE_BASELINES=1`. In update mode, the new screenshot is
  written into `tests/visual/baselines/<scenario>.png` and the test passes.

On failure, the actual / baseline / diff PNGs are saved to
`/tmp/pfm-visual-diff/<scenario>-{actual,baseline,diff}.png` so they can be
inspected after the run.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

# --- Optional-dependency guards ------------------------------------------------
# Skip the whole module if Playwright is missing. We intentionally do the import
# at module top-level (via `importorskip`) so pytest collection itself does not
# explode in environments without Playwright.
pytest.importorskip(
    "playwright.sync_api",
    reason=(
        "playwright not installed; install via "
        "'pip install playwright && playwright install chromium'"
    ),
)

PIL_Image = pytest.importorskip("PIL.Image", reason="Pillow not installed; visual diff unavailable")
PIL_ImageChops = pytest.importorskip("PIL.ImageChops", reason="Pillow not installed")

# Imported lazily inside fixtures so the `importorskip` above can short-circuit.
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

# Optional faster diff backend.
try:  # pragma: no cover - exercised only when pixelmatch is installed
    from pixelmatch.contrib.PIL import pixelmatch as _pixelmatch  # type: ignore

    _HAS_PIXELMATCH = True
except Exception:
    _pixelmatch = None  # type: ignore
    _HAS_PIXELMATCH = False


# --- Paths ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]  # api/tests/visual/file -> repo root
INDEX_HTML = REPO_ROOT / "web" / "index.html"
BASELINE_DIR = Path(__file__).resolve().parent / "baselines"
DIFF_DIR = Path("/tmp/pfm-visual-diff")

UPDATE_BASELINES = os.environ.get("PYTEST_UPDATE_BASELINES") == "1"

# Pixel-diff tolerance: at most 0.5% of pixels may differ from the baseline.
DIFF_THRESHOLD_RATIO = 0.005


# --- Helpers -------------------------------------------------------------------
def _ensure_dirs() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    DIFF_DIR.mkdir(parents=True, exist_ok=True)


def _index_url() -> str:
    if not INDEX_HTML.exists():
        pytest.skip(f"web/index.html missing at {INDEX_HTML}")
    # Use file:// so the test does not depend on the gunicorn dev server.
    return INDEX_HTML.as_uri()


def _save_diff_artifacts(name: str, actual_png: bytes, baseline_path: Path) -> Path:
    """Persist actual / baseline / diff PNGs under /tmp for post-mortem."""
    _ensure_dirs()
    actual_out = DIFF_DIR / f"{name}-actual.png"
    baseline_out = DIFF_DIR / f"{name}-baseline.png"
    diff_out = DIFF_DIR / f"{name}-diff.png"
    actual_out.write_bytes(actual_png)
    if baseline_path.exists():
        shutil.copyfile(baseline_path, baseline_out)
    return diff_out


def _compute_diff_ratio(actual_png: bytes, baseline_png_path: Path, diff_out: Path) -> float:
    """Return the fraction of pixels that differ between actual and baseline.

    Uses `pixelmatch` if available, otherwise falls back to PIL.ImageChops.
    """
    from io import BytesIO

    actual_img = PIL_Image.open(BytesIO(actual_png)).convert("RGBA")
    baseline_img = PIL_Image.open(baseline_png_path).convert("RGBA")

    # If dimensions differ, count every mismatched pixel as a diff to avoid
    # silently passing on layout regressions.
    if actual_img.size != baseline_img.size:
        # Resize baseline up/down only for diff visualisation; the ratio is
        # forced to 1.0 so the test fails loudly.
        diff_img = PIL_Image.new(
            "RGBA",
            actual_img.size,
            (255, 0, 255, 255),  # magenta = size mismatch
        )
        diff_img.save(diff_out)
        return 1.0

    if _HAS_PIXELMATCH:  # pragma: no cover - only when pixelmatch installed
        diff_img = PIL_Image.new("RGBA", actual_img.size, (0, 0, 0, 0))
        mismatched = _pixelmatch(
            baseline_img,
            actual_img,
            diff_img,
            threshold=0.1,
            includeAA=False,
        )
        diff_img.save(diff_out)
        total = actual_img.size[0] * actual_img.size[1]
        return mismatched / total if total else 0.0

    # PIL fallback: per-pixel L1 distance; count pixels with non-trivial delta.
    diff = PIL_ImageChops.difference(actual_img.convert("RGB"), baseline_img.convert("RGB"))
    bbox = diff.getbbox()
    if bbox is None:
        # Identical.
        PIL_Image.new("RGB", actual_img.size, (0, 0, 0)).save(diff_out)
        return 0.0

    # Build a binary mask of pixels whose max channel delta exceeds 8/255
    # (roughly the magnitude of font-hinting / AA jitter we want to ignore).
    from PIL import ImageOps

    grey = diff.convert("L")
    mask = grey.point(lambda v: 255 if v > 8 else 0)
    # Save a high-contrast visual diff.
    ImageOps.colorize(mask, black=(0, 0, 0), white=(255, 0, 255)).save(diff_out)
    histogram = mask.histogram()
    mismatched = sum(histogram[1:])  # everything that is not zero
    total = actual_img.size[0] * actual_img.size[1]
    return mismatched / total if total else 0.0


def _compare_or_update(name: str, actual_png: bytes) -> None:
    """Either update the baseline (when `PYTEST_UPDATE_BASELINES=1`) or diff it."""
    _ensure_dirs()
    baseline_path = BASELINE_DIR / f"{name}.png"

    if UPDATE_BASELINES:
        baseline_path.write_bytes(actual_png)
        return

    if not baseline_path.exists():
        # Save the captured screenshot so the developer can review it before
        # promoting it to a baseline.
        DIFF_DIR.mkdir(parents=True, exist_ok=True)
        (DIFF_DIR / f"{name}-actual.png").write_bytes(actual_png)
        pytest.skip(
            f"baseline {baseline_path} missing; rerun with PYTEST_UPDATE_BASELINES=1 "
            f"to create it. Captured PNG saved to {DIFF_DIR / (name + '-actual.png')}."
        )

    diff_out = _save_diff_artifacts(name, actual_png, baseline_path)
    ratio = _compute_diff_ratio(actual_png, baseline_path, diff_out)
    assert ratio <= DIFF_THRESHOLD_RATIO, (
        f"visual regression for {name}: {ratio:.4%} of pixels differ "
        f"(threshold {DIFF_THRESHOLD_RATIO:.2%}). "
        f"See {DIFF_DIR}/{name}-*.png"
    )


# --- Playwright fixture --------------------------------------------------------
@pytest.fixture(scope="module")
def _browser():
    """Yield a headless Chromium browser instance, or skip cleanly if launch fails."""
    try:
        pw = sync_playwright().start()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"headless browser unavailable: {exc!r}")

    try:
        browser = pw.chromium.launch(headless=True)
    except PlaywrightError as exc:  # pragma: no cover - environment-dependent
        pw.stop()
        pytest.skip(f"headless browser unavailable: {exc!r}")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pw.stop()
        pytest.skip(f"headless browser unavailable: {exc!r}")

    try:
        yield browser
    finally:
        try:
            browser.close()
        finally:
            pw.stop()


def _new_page(browser, viewport: dict[str, int] | None = None):
    ctx_kwargs: dict[str, object] = {
        "device_scale_factor": 1,
        "color_scheme": "light",
    }
    if viewport is not None:
        ctx_kwargs["viewport"] = viewport
    else:
        ctx_kwargs["viewport"] = {"width": 1280, "height": 800}
    context = browser.new_context(**ctx_kwargs)
    page = context.new_page()
    return context, page


def _disable_motion(page) -> None:
    """Kill animations + caret blinking so screenshots are deterministic."""
    page.add_style_tag(
        content="""
        *, *::before, *::after {
            transition: none !important;
            animation: none !important;
            caret-color: transparent !important;
        }
        html { scroll-behavior: auto !important; }
        """
    )


def _switch_mode(page, mode: str) -> None:
    """Click the mode tab and wait for the corresponding pane to be active."""
    selector = f'.mode-btn[data-mode="{mode}"]'
    page.click(selector)
    page.wait_for_selector(
        f'.mode-pane[data-mode-pane="{mode}"].active',
        state="attached",
        timeout=5000,
    )
    # Allow any layout to settle.
    page.wait_for_timeout(150)


def _goto_index(page) -> None:
    page.goto(_index_url(), wait_until="domcontentloaded")
    # The header mode bar is one of the first things to render; if it does not
    # appear we are looking at a totally broken page.
    page.wait_for_selector(".mode-btn[data-mode='regression']", timeout=8000)
    _disable_motion(page)


# --- Tests ---------------------------------------------------------------------
def test_regression_empty(_browser):
    """Empty Regression form — the most-visited initial state."""
    context, page = _new_page(_browser)
    try:
        _goto_index(page)
        _switch_mode(page, "regression")
        png = page.screenshot(full_page=True, animations="disabled")
        _compare_or_update("regression-empty", png)
    finally:
        context.close()


@pytest.mark.skip(
    reason=(
        "Frontend test bridge window.PFM.events.emit is not exposed in the "
        "production web/index.html bundle; SSE/event flow follows ADR-0017 "
        "instead. Re-enable when a deliberate test-only bridge ships."
    )
)
def test_regression_after_fit(_browser):
    """Mock a fit response via the `window.PFM.events` bridge and pin the toast + sticky card."""
    context, page = _new_page(_browser)
    try:
        _goto_index(page)
        _switch_mode(page, "regression")

        # Synthesise a minimal fit-result payload and emit it via the PFM event
        # bridge. The regression-results-sticky.js (T61) listens for this and
        # paints the sticky-result card; the toast subsystem listens too.
        page.evaluate(
            """
            (() => {
                const payload = {
                    ticker: 'NVDA',
                    factors: ['polymarket:ai-cap-2026', 'polymarket:fed-hike-jan'],
                    r2: 0.42,
                    adj_r2: 0.39,
                    n_obs: 252,
                    betas: { 'polymarket:ai-cap-2026': 0.18, 'polymarket:fed-hike-jan': -0.07 },
                    t_stats: { 'polymarket:ai-cap-2026': 3.21, 'polymarket:fed-hike-jan': -1.84 },
                    cov_type: 'HAC',
                    timestamp: '2026-05-16T09:00:00Z'
                };
                window.PFM = window.PFM || {};
                window.PFM.events = window.PFM.events || {
                    _subs: {},
                    on(k, f) { (this._subs[k] = this._subs[k] || []).push(f); },
                    emit(k, p) { (this._subs[k] || []).forEach(f => { try { f(p); } catch(e){} }); }
                };
                window.PFM.events.emit('fit:complete', payload);
                window.PFM.events.emit('regression:fit', payload);
                window.PFM.events.emit('toast', { kind: 'success', message: 'Fit complete — R²=0.42' });
            })();
            """
        )
        # Give listeners a beat to render their DOM.
        page.wait_for_timeout(400)
        png = page.screenshot(full_page=True, animations="disabled")
        _compare_or_update("regression-after-fit", png)
    finally:
        context.close()


def test_terminal_landing(_browser):
    """Terminal mode is the default landing pane post-rebrand."""
    context, page = _new_page(_browser)
    try:
        _goto_index(page)
        # Terminal is already the active default pane (post-2026-05-14 rebrand),
        # but click it explicitly to be deterministic.
        _switch_mode(page, "terminal")
        page.wait_for_timeout(250)
        png = page.screenshot(full_page=True, animations="disabled")
        _compare_or_update("terminal-landing", png)
    finally:
        context.close()


def test_strategies_alphahub(_browser):
    """Strategies / Alpha Hub landing — curated A-tier alpha cards."""
    context, page = _new_page(_browser)
    try:
        _goto_index(page)
        _switch_mode(page, "strategies")
        page.wait_for_timeout(250)
        png = page.screenshot(full_page=True, animations="disabled")
        _compare_or_update("strategies-alphahub", png)
    finally:
        context.close()


def test_regression_dark_mode(_browser):
    """Dark-mode token application on the Regression pane."""
    context, page = _new_page(_browser)
    try:
        _goto_index(page)
        # The site applies dark mode via [data-theme="dark"] on <html>.
        page.evaluate("document.documentElement.setAttribute('data-theme', 'dark');")
        _switch_mode(page, "regression")
        page.wait_for_timeout(200)
        png = page.screenshot(full_page=True, animations="disabled")
        _compare_or_update("regression-dark", png)
    finally:
        context.close()


def test_regression_mobile(_browser):
    """Mobile viewport (iPhone SE-class) on the Regression pane."""
    context, page = _new_page(_browser, viewport={"width": 375, "height": 667})
    try:
        _goto_index(page)
        _switch_mode(page, "regression")
        page.wait_for_timeout(200)
        png = page.screenshot(full_page=True, animations="disabled")
        _compare_or_update("regression-mobile", png)
    finally:
        context.close()
