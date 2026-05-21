"""Alpha-Hub package.

Per CLAUDE.md "α Hub is the product surface" the alpha-hub area collects
curated, validated-alpha surfaces shown on the strategies tab. Most of
the historical routing still lives in :mod:`pfm.alpha_hub_router` (a
sibling module, intentionally not folded in yet to avoid disturbing the
shipped FastAPI surface). New focused submodules — like
:mod:`pfm.alpha_hub.sentiment_alert` — land here so the package can grow
without bloating the legacy 26 kB router file.

This ``__init__`` deliberately re-exports nothing eagerly so importing
the package is cheap (no Polymarket / numpy fan-out). Sub-modules are
imported explicitly by their consumers.
"""

from __future__ import annotations

__all__: list[str] = []
