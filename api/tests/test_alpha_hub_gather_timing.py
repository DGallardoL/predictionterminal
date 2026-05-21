"""Async-timing assertion for the parallelized strategy-detail builder (W11-18).

`_build_strategy_detail_async` fans out the spread-series build, live-signal
lookup, and catalog-updated fallback via :func:`asyncio.gather`. With three
unit-tasks each sleeping ``t``, the wall-clock must be ``~t`` (max) rather
than ``~3t`` (sum). We patch the three unit helpers to sleep a known
interval and assert the gathered call beats the serial-sum lower bound by
a generous margin.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from pfm import alpha_hub_router
from pfm.alpha_hub_router import _build_strategy_detail_async, _load_strategies
from pfm.cache_utils import reset_caches


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()
    yield
    reset_caches()


def _first_pair_id_without_last_updated() -> str:
    """Pick a pair_id; force `last_updated_iso` to be missing to exercise the
    updated-fallback branch (the third gather task)."""
    return str(_load_strategies()[0]["pair_id"])


def test_async_builder_runs_units_concurrently() -> None:
    """Three 50ms unit-tasks under gather must complete in <= ~120ms total."""
    pair_id = _first_pair_id_without_last_updated()
    sleep_s = 0.05

    def slow_spread(pid: str, hedge: object) -> list[dict[str, object]]:
        time.sleep(sleep_s)
        return [{"date": "2026-01-01", "spread": 0.0, "p_a": 0.5, "p_b": 0.5, "z_score": 0.0}]

    def slow_signal(pid: str) -> None:
        time.sleep(sleep_s)

    def slow_updated() -> str:
        time.sleep(sleep_s)
        return "2026-01-01"

    # Force the updated-fallback path by stripping last_updated_iso.
    real_find = alpha_hub_router._find_strategy_src

    def find_no_updated(pid: str) -> dict[str, object]:
        src = dict(real_find(pid))
        src["last_updated_iso"] = None
        return src

    with (
        patch.object(alpha_hub_router, "_build_spread", side_effect=slow_spread),
        patch.object(alpha_hub_router, "_build_recent_signal", side_effect=slow_signal),
        patch.object(alpha_hub_router, "_load_catalog_updated_iso", side_effect=slow_updated),
        patch.object(alpha_hub_router, "_find_strategy_src", side_effect=find_no_updated),
    ):
        t0 = time.perf_counter()
        payload = asyncio.run(_build_strategy_detail_async(pair_id))
        elapsed = time.perf_counter() - t0

    # Serial would be ~3 * 50 = 150ms; gather should be ~50ms.
    # Allow 120ms ceiling for thread-pool overhead / GIL scheduling.
    assert elapsed < 0.120, (
        f"expected gather <120ms (max(t_i)+overhead); got {elapsed * 1000:.1f}ms — looks serial"
    )
    assert payload["spread_series"]
    assert payload["recent_signal"] is None
    assert payload["updated_at"] == "2026-01-01"


def test_async_builder_handles_unit_exception_gracefully() -> None:
    """A failing unit-task must downgrade to its safe default, not 500."""
    pair_id = _first_pair_id_without_last_updated()

    def boom_spread(pid: str, hedge: object) -> list[dict[str, object]]:
        raise RuntimeError("spread boom")

    with patch.object(alpha_hub_router, "_build_spread", side_effect=boom_spread):
        payload = asyncio.run(_build_strategy_detail_async(pair_id))

    # Failing spread is substituted with [] and the rest of the payload survives.
    assert payload["spread_series"] == []
    assert payload["pair_id"] == pair_id
    assert "risk" in payload
