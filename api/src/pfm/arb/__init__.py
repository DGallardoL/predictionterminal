"""``pfm.arb`` — sub-package for arb routers.

Currently houses :mod:`pfm.arb.quality_router` which exposes a single
``GET /arb/quality-audit`` endpoint. Other ``/arb`` related logic lives in
its historical home (``pfm.arb_scanner``, ``pfm.arb_matching``,
``pfm.strategies_arb_router``) — this package is the umbrella for
future arb-audit / match-quality endpoints carved out of the monolith.
"""

__all__: list[str] = []
