"""Playwright-based visual regression tests for the Prediction Terminal frontend.

These tests render `web/index.html` in headless Chromium and pixel-diff full-page
screenshots against checked-in baselines. The whole suite skips cleanly when
Playwright (or its browser binaries) is unavailable, so it is safe to ship in CI
sandboxes that do not allow downloading Chromium.

Regenerate baselines with:

    PYTEST_UPDATE_BASELINES=1 pytest api/tests/visual -q --noconftest
"""
