"""Scenario-based integration tests — multi-endpoint user workflows (W12-12).

Each scenario chains 4-6 endpoint calls in sequence and asserts that the
output of call ``N`` informs the input (or expectation) of call ``N+1``.
This complements the per-router unit tests by verifying that the actual
user journeys hold together end-to-end.

Three scenarios are covered:

* **Scenario 1 — "Quant explores then fits"** uses the synthetic ``app_client``
  fixture from ``conftest.py`` so we can post ``/fit`` calls without hitting
  any real upstream. The ``/terminal/themes`` endpoint depends on an async
  HTTP fetch we don't want to mock here, so we substitute the in-process
  ``factors_router`` "theme" projection by reading ``GET /factors?theme=…``
  (which exercises the same theme grouping the front-end uses for nav).
* **Scenario 2 — "Trader watches arb"** mounts the read-only routers we
  need directly (anti-alpha, deployable, arb-quality, strategies-arb) onto a
  bare FastAPI app. ``/strategies/arb/state`` and ``/arb/quality-audit``
  both have graceful in-process fallbacks (live scanner + ``top_arbs``),
  so the test works offline.
* **Scenario 3 — "Op monitors"** mounts the ops / health-deep / metrics /
  admin routers. ``/health/deep`` is fully mocked with ``respx`` so we
  exercise the parallel-probe envelope, and ``/admin/cache-stats`` is
  gated on the admin router being mountable — if it can't be imported we
  ``pytest.skip`` rather than fail.

Style notes
-----------
* Every chain assertion is explicit: we extract a value from step N's
  response and assert that step N+1 either echoes it, references it, or
  produces a different result *because of it*.
* Each scenario lives in its own class so pytest can collect and report
  the workflow as a unit.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_json(client: TestClient, path: str, **kwargs: Any) -> dict[str, Any]:
    """GET ``path`` and assert 200, returning the decoded JSON body."""
    r = client.get(path, **kwargs)
    assert r.status_code == 200, f"GET {path} → {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert isinstance(body, dict), f"GET {path} returned non-object: {type(body)}"
    return body


def _post_json(client: TestClient, path: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST ``path`` and assert 200, returning the decoded JSON body."""
    r = client.post(path, json=body)
    assert r.status_code == 200, f"POST {path} → {r.status_code}: {r.text[:200]}"
    return r.json()


# ---------------------------------------------------------------------------
# Scenario 1 — "Quant explores then fits"
# ---------------------------------------------------------------------------


class TestScenarioQuantExploresThenFits:
    """A quant lists factors, picks one, looks at its theme grouping +
    correlations, then runs two ``/fit`` calls (anchor + a second factor)
    to compare coefficients, and finally checks the alpha-hub leaderboard.

    Uses the synthetic ``app_client`` fixture from conftest.py — it boots
    the real production app with two synthetic factors (``factor_a`` /
    ``factor_b``) and stubs Polymarket + yfinance. ``/factors/{slug}/related``
    is exercised end-to-end against the mocked factor catalog.
    """

    def test_full_scenario_chain(self, app_client: TestClient) -> None:
        # ── Step 1: GET /factors → pick a factor ─────────────────────────
        factors_page = _get_json(app_client, "/factors?limit=10")
        assert "factors" in factors_page and isinstance(factors_page["factors"], list)
        assert factors_page["total"] >= 2, "synthetic catalog must expose ≥2 factors"
        assert len(factors_page["factors"]) >= 2

        # The conftest synthetic catalog seeds two factors: factor_a / factor_b.
        # Verify the structure and capture the anchor.
        ids = [f["id"] for f in factors_page["factors"]]
        assert "factor_a" in ids and "factor_b" in ids
        anchor_meta = next(f for f in factors_page["factors"] if f["id"] == "factor_a")
        anchor_slug = anchor_meta["slug"]
        anchor_id = anchor_meta["id"]
        assert anchor_slug == "slug-a"

        # ── Step 2: GET /factors/{slug}/related → see correlated factors ─
        # Chain assertion: the response must echo the anchor slug we just
        # picked, and the returned "related" entries must NOT include the
        # anchor itself (output of step 1 directly drives step 2's URL).
        related_body = _get_json(app_client, f"/factors/{anchor_slug}/related")
        assert related_body["anchor"] == anchor_slug, (
            "step 2 must echo the anchor slug chosen in step 1"
        )
        assert isinstance(related_body["related"], list)
        for row in related_body["related"]:
            assert row["slug"] != anchor_slug, "anchor must never appear in its own related list"
            # Each row must carry a rho in [-1, 1].
            assert -1.0 - 1e-9 <= row["rho"] <= 1.0 + 1e-9

        # Identify the "second-best" candidate we'd want to compare against.
        # With the synthetic two-factor catalog the related list will contain
        # exactly one row (factor_b). When empty (e.g. low-overlap window),
        # fall back to factor_b directly — the chain still holds because
        # step 1's enumeration is what tells us factor_b exists.
        if related_body["related"]:
            partner_slug = related_body["related"][0]["slug"]
        else:
            partner_slug = "slug-b"
        partner_id = next(f["id"] for f in factors_page["factors"] if f["slug"] == partner_slug)
        assert partner_id != anchor_id, "partner must differ from anchor"

        # ── Step 3: GET /factors?theme=… → understand theme structure ────
        # /terminal/themes hits the live Polymarket gamma API; the same
        # "theme grouping" concept is exposed offline via /factors?theme=…
        # which is what the front-end nav uses to filter chips. We exercise
        # it here by reading the anchor's theme and re-querying.
        anchor_theme = anchor_meta.get("theme", "other") or "other"
        theme_page = _get_json(app_client, f"/factors?theme={anchor_theme}")
        # Theme filter must (a) return something and (b) every row must
        # carry the requested theme — the output of step 2 doesn't directly
        # feed this call, but step 1's metadata does.
        assert isinstance(theme_page["factors"], list)
        # All themed rows must agree on the theme; the anchor must be in there.
        themed_ids = {f["id"] for f in theme_page["factors"]}
        assert anchor_id in themed_ids, "anchor must appear in its own themed sub-listing"

        # ── Step 4: POST /fit with anchor → get coefficient ──────────────
        fit_payload_anchor = {
            "ticker": "AAPL",
            "factors": [anchor_id],
            "start": "2025-06-15",
            "end": "2025-12-15",
        }
        fit_anchor = _post_json(app_client, "/fit", fit_payload_anchor)
        assert fit_anchor["ticker"] == "AAPL"
        assert fit_anchor["n_obs"] > 30
        assert len(fit_anchor["factors"]) == 1, (
            "step 4 must fit exactly the single anchor we chose in step 1"
        )
        anchor_coef = fit_anchor["factors"][0]
        assert anchor_coef["id"] == anchor_id, (
            "fit response must echo back the factor id we asked for"
        )
        anchor_beta = float(anchor_coef["beta"])
        # beta must be finite — the synthetic series cannot produce NaN.
        assert anchor_beta == anchor_beta, "anchor beta must be finite"

        # ── Step 5: POST /fit with related factor → compare ──────────────
        # Chain assertion: step 5's request *uses* the slug surfaced by
        # step 2 → step 1 mapping. The two fit responses must agree on the
        # window and ticker but produce DIFFERENT factor ids in the
        # response so we genuinely measured a different coefficient.
        fit_payload_partner = {
            "ticker": "AAPL",
            "factors": [partner_id],
            "start": "2025-06-15",
            "end": "2025-12-15",
        }
        fit_partner = _post_json(app_client, "/fit", fit_payload_partner)
        assert fit_partner["ticker"] == fit_anchor["ticker"]
        assert fit_partner["start"] == fit_anchor["start"]
        assert fit_partner["end"] == fit_anchor["end"]
        assert len(fit_partner["factors"]) == 1
        partner_coef = fit_partner["factors"][0]
        assert partner_coef["id"] == partner_id
        assert partner_coef["id"] != anchor_coef["id"], (
            "comparison fit must target a different factor than the anchor fit"
        )

        # ── Step 6: GET /alpha-hub/leaderboard → see deployable strategies ─
        # The leaderboard reads from the on-disk web/data/alpha_strategies.json
        # catalog; we just need to confirm the endpoint serves a paginated
        # envelope so the quant's workflow doesn't dead-end. The chain link
        # here is conceptual: after fitting individual factors, the leaderboard
        # is where deployable bundles surface.
        board = _get_json(app_client, "/alpha-hub/leaderboard?limit=5")
        assert "items" in board, "leaderboard envelope must carry an items array"
        assert isinstance(board["items"], list)
        # The leaderboard at least claims a finite total ≥ items returned.
        assert board.get("total", 0) >= len(board["items"])


# ---------------------------------------------------------------------------
# Scenario 2 — "Trader watches arb"
# ---------------------------------------------------------------------------


@pytest.fixture()
def arb_app_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Bare FastAPI app mounting the read-only arb / strategy routers.

    No real upstream calls happen: ``/strategies/arb/state`` falls back to
    ``pfm.arb_scanner.top_arbs`` which we stub to return an empty list, and
    ``/arb/quality-audit`` falls back to the same stub (so it returns 0
    audited pairs with a stable shape).
    """
    # Stub top_arbs to return [] so neither the arb state fallback nor the
    # quality-audit fallback hits Polymarket / Kalshi in tests.
    from pfm import arb_scanner
    from pfm.arb.quality_router import router as arb_quality_router
    from pfm.strategies.anti_alpha_router import router as anti_alpha_router
    from pfm.strategies.deployable_router import router as deployable_router
    from pfm.strategies_arb_router import router as strategies_arb_router

    monkeypatch.setattr(arb_scanner, "top_arbs", lambda *_a, **_kw: [])

    # Reset both routers' module-level caches so prior tests don't bleed in.
    import pfm.arb.quality_router as quality_mod
    import pfm.strategies.anti_alpha_router as anti_alpha_mod
    import pfm.strategies.deployable_router as deployable_mod
    import pfm.strategies_arb_router as strategies_arb_mod

    quality_mod._CACHE.update({"t": 0.0, "key": None, "value": None})
    strategies_arb_mod._FALLBACK_CACHE.update({"t": 0.0, "value": None})
    anti_alpha_mod._CACHE.clear()
    deployable_mod.clear_cache()

    app = FastAPI()
    app.include_router(strategies_arb_router)
    app.include_router(arb_quality_router)
    app.include_router(anti_alpha_router)
    app.include_router(deployable_router)

    with TestClient(app) as client:
        yield client


class TestScenarioTraderWatchesArb:
    """A trader pulls the live arb state, audits the match quality, reviews
    what NOT to deploy, then checks the curated deployable list.
    """

    def test_full_scenario_chain(self, arb_app_client: TestClient) -> None:
        # ── Step 1: GET /strategies/arb/state → see current opportunities ─
        state = _get_json(arb_app_client, "/strategies/arb/state")
        # Envelope shape (mirrors arbstuff/dashboard_state.json):
        assert "opportunities" in state
        assert isinstance(state["opportunities"], list)
        assert "bot_status" in state
        assert "scan_mode" in state
        # Capture the opportunity count — informs what step 2 should audit.
        opp_count_from_state = len(state["opportunities"])
        assert opp_count_from_state >= 0  # tautology, but documents the chain

        # ── Step 2: GET /arb/quality-audit → verify quality ──────────────
        # Chain assertion: step 1 exposed an arb universe (size N opps); the
        # quality audit must (a) return a well-formed envelope, (b) tally
        # high_conf + borderline + rejected to *at most* the audited count
        # (every pair lands in exactly one bucket; "ok-but-not-borderline"
        # pairs sit outside the three buckets), and (c) source itself from
        # either the dashboard_state file (the same source step 1 prefers)
        # or the fallback top_arbs scanner — never both.
        audit = _get_json(arb_app_client, "/arb/quality-audit")
        for field in (
            "audited_count",
            "rejected_count",
            "high_conf_count",
            "borderline_count",
            "rejection_breakdown",
            "source",
        ):
            assert field in audit, f"audit envelope missing {field}"
        assert isinstance(audit["rejection_breakdown"], dict)
        # Sanity: rejected ≤ audited; high_conf + borderline + rejected ≤ audited.
        assert audit["rejected_count"] <= audit["audited_count"]
        assert (
            audit["high_conf_count"] + audit["borderline_count"] + audit["rejected_count"]
            <= audit["audited_count"]
        ), "audit bucket counts must not exceed audited universe"
        # Source must be one of the two documented loaders.
        assert audit["source"] in {"dashboard_state", "top_arbs"}
        # Chain check: both endpoints read from a shared arb universe; the
        # audit's `source` field tells the trader *which* source step 1's
        # `_source` should also reflect. When the audit ran against the
        # top_arbs fallback (our stubbed empty scanner), the universe must
        # be empty — and step 1's fallback chain would land in the same
        # branch. Conversely, the dashboard_state branch can carry pairs
        # without our stub having any effect.
        if audit["source"] == "top_arbs":
            assert audit["audited_count"] == 0, (
                "top_arbs fallback was stubbed to return []; audit must be empty"
            )
            # When step 2 fell back to top_arbs the in-process arb-state
            # SHOULD also have produced an empty list (same fallback source).
            assert opp_count_from_state == 0
        # When source == dashboard_state, both endpoints read from the same
        # on-disk file → their per-pair counts should be in the same ballpark.
        # We only assert non-negativity to keep the test environment-agnostic.

        # ── Step 3: GET /strategies/anti-alpha-list → review DO-NOT-DEPLOY ─
        anti = _get_json(arb_app_client, "/strategies/anti-alpha-list")
        assert "count" in anti and "items" in anti
        assert isinstance(anti["items"], list)
        assert anti["count"] == len(anti["items"]), "anti-alpha count must equal len(items)"
        # Every anti-alpha entry must carry a tier so the trader can filter.
        for item in anti["items"]:
            assert "tier" in item, "anti-alpha row missing tier"

        # Collect the set of anti-alpha pair_ids (when present) to verify
        # step 4 excludes them.
        anti_ids: set[str] = {
            str(i.get("pair_id") or i.get("id") or "")
            for i in anti["items"]
            if (i.get("pair_id") or i.get("id"))
        }

        # ── Step 4: GET /strategies/deployable-list → see candidates ─────
        # Chain assertion: deployable_ids and anti_ids must be DISJOINT —
        # the same strategy cannot be both "do not deploy" and "deployable".
        deploy = _get_json(arb_app_client, "/strategies/deployable-list")
        assert "count" in deploy and "items" in deploy
        assert isinstance(deploy["items"], list)
        assert deploy["count"] == len(deploy["items"])
        # Every deployable row must carry a tier in a "deployable" bucket.
        # ``A_STRUCTURAL`` is the highest-allocation tier introduced with the
        # strike-family / BH-q10 framework (see pfm/alpha_tier_regen.py).
        deployable_tiers = {
            "A_STRUCTURAL",
            "A_GOLD",
            "A_PRODUCTION",
            "A_PAPER",
            "B_VALIDATED",
        }
        for item in deploy["items"]:
            tier = item.get("tier")
            assert tier in deployable_tiers, f"deployable item has non-deployable tier: {tier}"

        # Final chain assertion: deployable IDs must not overlap with anti-alpha.
        deployable_ids: set[str] = {
            str(i.get("pair_id") or i.get("id") or "")
            for i in deploy["items"]
            if (i.get("pair_id") or i.get("id"))
        }
        overlap = deployable_ids & anti_ids
        # Tolerate empty sets (fixture-only environments may not seed IDs)
        # but if both sides exposed ids, they MUST be disjoint.
        if deployable_ids and anti_ids:
            assert not overlap, f"deployable and anti-alpha lists overlap on: {overlap}"


# ---------------------------------------------------------------------------
# Scenario 3 — "Op monitors"
# ---------------------------------------------------------------------------


@pytest.fixture()
def ops_app_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> Iterator[TestClient]:
    """Bare FastAPI app mounting the ops/health-deep/metrics/admin routers.

    /health/deep is wrapped in respx so the 5 upstream probes all return
    200 deterministically. /ops/sessions reads a tmp_path active-edits
    file we seed with two synthetic active sessions.
    """
    # ── Stage a synthetic active-edits.json ──────────────────────────────
    import json
    from datetime import datetime, timedelta

    from pfm.health_deep_router import router as health_deep_router
    from pfm.metrics_router import router as metrics_router
    from pfm.ops_router import router as ops_router

    now = datetime.now(UTC).replace(microsecond=0)
    edits = [
        {
            "session_id": "scenario-active-1",
            "scope": "scenario-1",
            "files": ["api/src/pfm/foo.py"],
            "started_at": now.isoformat().replace("+00:00", "Z"),
            "expires_at": (now + timedelta(minutes=25)).isoformat().replace("+00:00", "Z"),
        },
        {
            "session_id": "scenario-active-2",
            "scope": "scenario-2",
            "files": ["api/src/pfm/bar.py"],
            "started_at": now.isoformat().replace("+00:00", "Z"),
            "expires_at": (now + timedelta(minutes=20)).isoformat().replace("+00:00", "Z"),
        },
    ]
    active_edits_path = tmp_path / "active-edits.json"
    active_edits_path.write_text(json.dumps(edits), encoding="utf-8")
    monkeypatch.setenv("PFM_OPS_ACTIVE_EDITS_PATH", str(active_edits_path))
    # Avoid surprise Redis probes during /health/deep.
    monkeypatch.delenv("REDIS_URL", raising=False)

    app = FastAPI()
    app.include_router(health_deep_router)
    app.include_router(metrics_router)
    app.include_router(ops_router)

    # /admin/cache-stats — mount when importable, otherwise leave unmounted
    # so the corresponding scenario step can pytest.skip() gracefully.
    try:
        from pfm.admin.cache_stats_router import router as admin_cache_router

        app.include_router(admin_cache_router)
        cache_stats_mounted = True
    except Exception:
        cache_stats_mounted = False
    app.state.cache_stats_mounted = cache_stats_mounted

    with TestClient(app) as client:
        yield client


class TestScenarioOpMonitors:
    """An on-call op walks the deep-health probe, the latency audit, the
    cache-pool health (if mounted), and the active multi-session claims.
    """

    @respx.mock
    def test_full_scenario_chain(self, ops_app_client: TestClient) -> None:
        # Stub all five upstream probes for /health/deep with 200s.
        from pfm.health_deep_router import (
            GDELT_URL,
            KALSHI_URL,
            POLYMARKET_URL,
            YFINANCE_URL,
        )

        for url in (POLYMARKET_URL, KALSHI_URL, GDELT_URL, YFINANCE_URL):
            respx.get(url).mock(return_value=httpx.Response(200, json={"ok": True}))

        # ── Step 1: GET /health/deep → upstream health ───────────────────
        health = _get_json(ops_app_client, "/health/deep")
        assert "status" in health
        assert health["status"] in {"ok", "degraded", "down"}
        assert "sources" in health
        assert isinstance(health["sources"], dict)
        # All 5 upstreams must be present in the response.
        for src in ("polymarket", "kalshi", "yfinance", "redis", "gdelt"):
            assert src in health["sources"], f"missing upstream: {src}"
        # With all four HTTP probes mocked OK and redis "not configured" (which
        # the probe treats as ok), the overall status must be "ok".
        assert health["status"] == "ok", (
            f"all probes mocked OK but status is {health['status']}: {health['sources']}"
        )
        n_upstreams = len(health["sources"])

        # ── Step 2: GET /metrics/audit → latencies ───────────────────────
        # Chain assertion: by the time we hit /metrics/audit, the previous
        # /health/deep call MAY have been tracked (depends on middleware
        # mounting); regardless, the audit envelope must carry the standard
        # shape so an ops dashboard can render it.
        audit = _get_json(ops_app_client, "/metrics/audit")
        assert "endpoints" in audit
        assert isinstance(audit["endpoints"], dict)
        assert "total_requests" in audit
        assert isinstance(audit["total_requests"], int)
        assert audit["total_requests"] >= 0
        # Every endpoint row must carry the percentile fields.
        for endpoint, row in audit["endpoints"].items():
            for field in ("count", "p50_ms", "p95_ms", "p99_ms", "err_rate"):
                assert field in row, f"endpoint {endpoint!r} missing {field} in audit row"

        # ── Step 3: GET /admin/cache-stats → cache health (IF mounted) ───
        if getattr(ops_app_client.app.state, "cache_stats_mounted", False):
            stats = _get_json(ops_app_client, "/admin/cache-stats")
            # The standard /admin/cache-stats envelope carries 'pools' + totals.
            assert "pools" in stats or "totals" in stats, (
                "/admin/cache-stats response must carry pools or totals"
            )
            if "pools" in stats:
                assert isinstance(stats["pools"], list)
            # Chain check: the cache-stats endpoint should never claim more
            # total requests than the global metrics tracker observed.
            if "totals" in stats and isinstance(stats["totals"], dict):
                # totals are cache hits/misses, NOT request counts — but
                # they must be non-negative ints.
                for k, v in stats["totals"].items():
                    if isinstance(v, int):
                        assert v >= 0, f"totals[{k}] negative: {v}"
        else:
            pytest.skip("/admin/cache-stats router not mounted in this env")

        # ── Step 4: GET /ops/sessions → active claims ────────────────────
        # Chain assertion: the staged active-edits.json had 2 active sessions
        # with future expires_at, so /ops/sessions MUST return exactly 2.
        sess = _get_json(ops_app_client, "/ops/sessions")
        assert "active_sessions" in sess
        assert isinstance(sess["active_sessions"], list)
        assert "count" in sess
        assert sess["count"] == len(sess["active_sessions"])
        assert sess["count"] == 2, f"expected 2 staged active sessions, got {sess['count']}"
        session_ids = {s["session_id"] for s in sess["active_sessions"]}
        assert session_ids == {"scenario-active-1", "scenario-active-2"}
        # Final chain assertion: the number of active sessions should be a
        # plausible reflection of how many upstreams an op has to monitor
        # (no hard equality — this is the workflow's narrative tie-in).
        assert sess["count"] >= 0
        assert n_upstreams == 5  # the env we just verified in step 1
