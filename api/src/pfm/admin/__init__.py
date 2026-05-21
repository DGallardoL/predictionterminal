"""Admin / introspection routers.

This sub-package holds operational endpoints not intended for end users:
cache statistics, manual invalidation, runtime knobs, etc. Every router
here is expected to be gated behind the admin auth dependency before
being exposed on a production deployment (mounting is the caller's job;
this package only defines the routers).
"""

from __future__ import annotations

from pfm.admin import cache_stats_router as _cache_stats_module

cache_stats_router_obj = _cache_stats_module.router

__all__ = ["cache_stats_router_obj"]
