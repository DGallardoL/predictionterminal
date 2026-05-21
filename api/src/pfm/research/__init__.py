"""Research package — alpha-report discovery and rendering.

Exposes a single FastAPI router (:mod:`pfm.research.router`) that lists the
versioned ``docs/alpha-reports/alpha-report-v*.md`` files as JSON cards and
returns individual report bodies (markdown or, when the optional ``markdown``
library is installed, rendered HTML).

The router is intentionally not auto-mounted from ``pfm.main`` — Damian wires
it via a single ``app.include_router(...)`` line when the ``main.py:routes``
section is unclaimed.
"""

from __future__ import annotations

from pfm.research.router import router

__all__ = ["router"]
