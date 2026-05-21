"""Terminal macro-event calendar.

Hardcoded schedule of US-macro and political events that prediction markets
typically react to. Verified against public Fed/BLS calendars on 2026-04-30.

Source notes:
- FOMC 2026 dates: federalreserve.gov/monetarypolicy/fomccalendars.htm
- CPI 2026 release dates: bls.gov/schedule/news_release/cpi.htm
- NFP 2026 release dates (first Friday): bls.gov/schedule/news_release/empsit.htm
- Election dates: state SOS pages + ballotpedia for primaries.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache

router = APIRouter(prefix="/terminal/calendar", tags=["terminal"])

# Static-data cache: factors.yml only changes at deploy time. 1-hour TTL is
# very safe and saves a 10k-line YAML parse + regex compile per request.
_CAL_CACHE = get_cache("terminal_calendar", ttl=3600)


# ──────────────────────────────────────────────────────────────────────────
# Hardcoded 2026-2028 macro-event schedule
# ──────────────────────────────────────────────────────────────────────────

# Each entry: (iso_date, time_et, name, category, expected_impact_themes,
# slug_id_patterns: list[regex] applied to factor IDs to tag related markets)

EVENTS: list[dict[str, Any]] = [
    # ── FOMC meetings 2026 (release time 14:00 ET) ────────────────────────
    {
        "date": "2026-01-28",
        "time_et": "14:00",
        "name": "FOMC Meeting Decision (January)",
        "category": "fomc",
        "expected_impact_themes": ["rates", "macro", "usd", "equities"],
        "patterns": [r"^fed_", r"^k_fed_", r"^no_fed_cuts", r"_fed_cuts", r"powell_out"],
    },
    {
        "date": "2026-03-18",
        "time_et": "14:00",
        "name": "FOMC Meeting Decision + SEP (March)",
        "category": "fomc",
        "expected_impact_themes": ["rates", "macro", "usd", "equities", "dot_plot"],
        "patterns": [r"^fed_", r"^k_fed_", r"^no_fed_cuts", r"_fed_cuts", r"powell_out"],
    },
    {
        "date": "2026-04-29",
        "time_et": "14:00",
        "name": "FOMC Meeting Decision (April)",
        "category": "fomc",
        "expected_impact_themes": ["rates", "macro", "usd", "equities"],
        "patterns": [r"^fed_", r"^k_fed_", r"^no_fed_cuts", r"_fed_cuts", r"powell_out"],
    },
    {
        "date": "2026-06-17",
        "time_et": "14:00",
        "name": "FOMC Meeting Decision + SEP (June)",
        "category": "fomc",
        "expected_impact_themes": ["rates", "macro", "usd", "equities", "dot_plot"],
        "patterns": [r"^fed_", r"^k_fed_", r"^no_fed_cuts", r"_fed_cuts", r"powell_out"],
    },
    {
        "date": "2026-07-29",
        "time_et": "14:00",
        "name": "FOMC Meeting Decision (July)",
        "category": "fomc",
        "expected_impact_themes": ["rates", "macro", "usd", "equities"],
        "patterns": [r"^fed_", r"^k_fed_", r"^no_fed_cuts", r"_fed_cuts", r"powell_out"],
    },
    {
        "date": "2026-09-16",
        "time_et": "14:00",
        "name": "FOMC Meeting Decision + SEP (September)",
        "category": "fomc",
        "expected_impact_themes": ["rates", "macro", "usd", "equities", "dot_plot"],
        "patterns": [r"^fed_", r"^k_fed_", r"^no_fed_cuts", r"_fed_cuts", r"powell_out"],
    },
    {
        "date": "2026-10-28",
        "time_et": "14:00",
        "name": "FOMC Meeting Decision (October)",
        "category": "fomc",
        "expected_impact_themes": ["rates", "macro", "usd", "equities"],
        "patterns": [r"^fed_", r"^k_fed_", r"^no_fed_cuts", r"_fed_cuts", r"powell_out"],
    },
    {
        "date": "2026-12-09",
        "time_et": "14:00",
        "name": "FOMC Meeting Decision + SEP (December)",
        "category": "fomc",
        "expected_impact_themes": ["rates", "macro", "usd", "equities", "dot_plot"],
        "patterns": [r"^fed_", r"^k_fed_", r"^no_fed_cuts", r"_fed_cuts", r"powell_out"],
    },
    # ── CPI release dates 2026 (08:30 ET) ─────────────────────────────────
    {
        "date": "2026-01-14",
        "time_et": "08:30",
        "name": "CPI Release (December 2025)",
        "category": "cpi",
        "expected_impact_themes": ["inflation", "rates", "real_yields"],
        "patterns": [r"^k_cpi_", r"inflation_", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-02-11",
        "time_et": "08:30",
        "name": "CPI Release (January 2026)",
        "category": "cpi",
        "expected_impact_themes": ["inflation", "rates", "real_yields"],
        "patterns": [r"^k_cpi_", r"inflation_", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-03-12",
        "time_et": "08:30",
        "name": "CPI Release (February 2026)",
        "category": "cpi",
        "expected_impact_themes": ["inflation", "rates", "real_yields"],
        "patterns": [r"^k_cpi_", r"inflation_", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-04-14",
        "time_et": "08:30",
        "name": "CPI Release (March 2026)",
        "category": "cpi",
        "expected_impact_themes": ["inflation", "rates", "real_yields"],
        "patterns": [r"^k_cpi_", r"inflation_", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-05-13",
        "time_et": "08:30",
        "name": "CPI Release (April 2026)",
        "category": "cpi",
        "expected_impact_themes": ["inflation", "rates", "real_yields"],
        "patterns": [r"^k_cpi_", r"inflation_", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-06-10",
        "time_et": "08:30",
        "name": "CPI Release (May 2026)",
        "category": "cpi",
        "expected_impact_themes": ["inflation", "rates", "real_yields"],
        "patterns": [r"^k_cpi_", r"inflation_", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-07-15",
        "time_et": "08:30",
        "name": "CPI Release (June 2026)",
        "category": "cpi",
        "expected_impact_themes": ["inflation", "rates", "real_yields"],
        "patterns": [r"^k_cpi_", r"inflation_", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-08-12",
        "time_et": "08:30",
        "name": "CPI Release (July 2026)",
        "category": "cpi",
        "expected_impact_themes": ["inflation", "rates", "real_yields"],
        "patterns": [r"^k_cpi_", r"inflation_", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-09-10",
        "time_et": "08:30",
        "name": "CPI Release (August 2026)",
        "category": "cpi",
        "expected_impact_themes": ["inflation", "rates", "real_yields"],
        "patterns": [r"^k_cpi_", r"inflation_", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-10-14",
        "time_et": "08:30",
        "name": "CPI Release (September 2026)",
        "category": "cpi",
        "expected_impact_themes": ["inflation", "rates", "real_yields"],
        "patterns": [r"^k_cpi_", r"inflation_", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-11-12",
        "time_et": "08:30",
        "name": "CPI Release (October 2026)",
        "category": "cpi",
        "expected_impact_themes": ["inflation", "rates", "real_yields"],
        "patterns": [r"^k_cpi_", r"inflation_", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-12-10",
        "time_et": "08:30",
        "name": "CPI Release (November 2026)",
        "category": "cpi",
        "expected_impact_themes": ["inflation", "rates", "real_yields"],
        "patterns": [r"^k_cpi_", r"inflation_", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    # ── NFP / Employment Situation 2026 (first Friday, 08:30 ET) ──────────
    {
        "date": "2026-01-09",
        "time_et": "08:30",
        "name": "NFP / Employment Situation (December 2025)",
        "category": "nfp",
        "expected_impact_themes": ["labor", "rates", "recession", "usd"],
        "patterns": [r"recession", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-02-06",
        "time_et": "08:30",
        "name": "NFP / Employment Situation (January 2026)",
        "category": "nfp",
        "expected_impact_themes": ["labor", "rates", "recession", "usd"],
        "patterns": [r"recession", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-03-06",
        "time_et": "08:30",
        "name": "NFP / Employment Situation (February 2026)",
        "category": "nfp",
        "expected_impact_themes": ["labor", "rates", "recession", "usd"],
        "patterns": [r"recession", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-04-03",
        "time_et": "08:30",
        "name": "NFP / Employment Situation (March 2026)",
        "category": "nfp",
        "expected_impact_themes": ["labor", "rates", "recession", "usd"],
        "patterns": [r"recession", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-05-01",
        "time_et": "08:30",
        "name": "NFP / Employment Situation (April 2026)",
        "category": "nfp",
        "expected_impact_themes": ["labor", "rates", "recession", "usd"],
        "patterns": [r"recession", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-06-05",
        "time_et": "08:30",
        "name": "NFP / Employment Situation (May 2026)",
        "category": "nfp",
        "expected_impact_themes": ["labor", "rates", "recession", "usd"],
        "patterns": [r"recession", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-07-02",
        "time_et": "08:30",
        "name": "NFP / Employment Situation (June 2026)",
        "category": "nfp",
        "expected_impact_themes": ["labor", "rates", "recession", "usd"],
        "patterns": [r"recession", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-08-07",
        "time_et": "08:30",
        "name": "NFP / Employment Situation (July 2026)",
        "category": "nfp",
        "expected_impact_themes": ["labor", "rates", "recession", "usd"],
        "patterns": [r"recession", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-09-04",
        "time_et": "08:30",
        "name": "NFP / Employment Situation (August 2026)",
        "category": "nfp",
        "expected_impact_themes": ["labor", "rates", "recession", "usd"],
        "patterns": [r"recession", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-10-02",
        "time_et": "08:30",
        "name": "NFP / Employment Situation (September 2026)",
        "category": "nfp",
        "expected_impact_themes": ["labor", "rates", "recession", "usd"],
        "patterns": [r"recession", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-11-06",
        "time_et": "08:30",
        "name": "NFP / Employment Situation (October 2026)",
        "category": "nfp",
        "expected_impact_themes": ["labor", "rates", "recession", "usd"],
        "patterns": [r"recession", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    {
        "date": "2026-12-04",
        "time_et": "08:30",
        "name": "NFP / Employment Situation (November 2026)",
        "category": "nfp",
        "expected_impact_themes": ["labor", "rates", "recession", "usd"],
        "patterns": [r"recession", r"^fed_", r"^k_fed_", r"_fed_cuts"],
    },
    # ── Major elections 2026-2028 ──────────────────────────────────────────
    {
        "date": "2026-11-03",
        "time_et": "20:00",
        "name": "US Midterm Elections (House & 1/3 Senate)",
        "category": "election",
        "expected_impact_themes": ["politics", "fiscal", "regulation", "balance_of_power"],
        "patterns": [
            r"_house_2026$",
            r"_senate_2026$",
            r"^bop_",
            r"trump_out",
            r"_win_the_us$",
            r"_win_the_democratic",
            r"_win_the_republican",
        ],
    },
    {
        "date": "2026-06-02",
        "time_et": "20:00",
        "name": "US Primary Election Super Tuesday (CA/NJ/etc.)",
        "category": "election",
        "expected_impact_themes": ["politics", "primaries"],
        "patterns": [
            r"_win_california_governor",
            r"_win_the_california",
            r"_win_the_democratic",
            r"_win_the_republican",
        ],
    },
    {
        "date": "2027-04-11",
        "time_et": "14:00",
        "name": "French Presidential Election Round 1",
        "category": "election",
        "expected_impact_themes": ["politics", "eu", "eur"],
        "patterns": [r"_win_the_2027_french", r"_win_2027_french", r"_win_the_french"],
    },
    {
        "date": "2027-04-25",
        "time_et": "14:00",
        "name": "French Presidential Election Round 2",
        "category": "election",
        "expected_impact_themes": ["politics", "eu", "eur"],
        "patterns": [r"_win_the_2027_french", r"_win_2027_french", r"_win_the_french"],
    },
    {
        "date": "2026-10-04",
        "time_et": "20:00",
        "name": "Brazilian General Election Round 1",
        "category": "election",
        "expected_impact_themes": ["politics", "em_fx", "brl"],
        "patterns": [r"_win_the_2026_brazilian", r"_2026_brazilian"],
    },
    {
        "date": "2026-05-31",
        "time_et": "20:00",
        "name": "Colombian Presidential Election Round 1",
        "category": "election",
        "expected_impact_themes": ["politics", "em_fx", "cop"],
        "patterns": [r"_win_the_colombian", r"_win_2026_colombian", r"colombian_president"],
    },
    {
        "date": "2026-04-12",
        "time_et": "20:00",
        "name": "Peruvian General Election Round 1",
        "category": "election",
        "expected_impact_themes": ["politics", "em_fx", "pen"],
        "patterns": [r"_win_2026_peruvian", r"_win_the_peruvian", r"peruvian_president"],
    },
    {
        "date": "2026-06-03",
        "time_et": "08:00",
        "name": "South Korean Local Elections (incl. Seoul Mayor)",
        "category": "election",
        "expected_impact_themes": ["politics", "krw", "asia"],
        "patterns": [r"_win_2026_seoul", r"_win_the_seoul", r"_win_the$"],
    },
    # ── Crypto resolution windows ─────────────────────────────────────────
    {
        "date": "2026-06-26",
        "time_et": "08:00",
        "name": "BTC Quarterly Options Expiry (Deribit, June)",
        "category": "crypto_expiry",
        "expected_impact_themes": ["btc_vol", "crypto", "options"],
        "patterns": [r"^btc_", r"^bitcoin_", r"^eth_", r"^ethereum_", r"^sol_"],
    },
    {
        "date": "2026-09-25",
        "time_et": "08:00",
        "name": "BTC Quarterly Options Expiry (Deribit, September)",
        "category": "crypto_expiry",
        "expected_impact_themes": ["btc_vol", "crypto", "options"],
        "patterns": [r"^btc_", r"^bitcoin_", r"^eth_", r"^ethereum_", r"^sol_"],
    },
    {
        "date": "2026-12-25",
        "time_et": "08:00",
        "name": "BTC Quarterly Options Expiry (Deribit, December)",
        "category": "crypto_expiry",
        "expected_impact_themes": ["btc_vol", "crypto", "options"],
        "patterns": [r"^btc_", r"^bitcoin_", r"^eth_", r"^ethereum_", r"^sol_"],
    },
    {
        "date": "2026-06-30",
        "time_et": "23:59",
        "name": "BTC H1 Resolution Cutoff (many Polymarket markets resolve here)",
        "category": "crypto_resolution",
        "expected_impact_themes": ["btc_vol", "crypto"],
        "patterns": [r"^btc_", r"^bitcoin_.*_jun", r"_h1$", r"^bitcoin_.*by_june"],
    },
    {
        "date": "2026-12-31",
        "time_et": "23:59",
        "name": "BTC EOY Resolution Cutoff (year-end Polymarket markets resolve)",
        "category": "crypto_resolution",
        "expected_impact_themes": ["btc_vol", "crypto"],
        "patterns": [r"^btc_.*eoy", r"_eoy$", r"december_31", r"by_december"],
    },
    # ── Manual: debates, primaries, IPO target dates ──────────────────────
    {
        "date": "2026-09-16",
        "time_et": "21:00",
        "name": "First US Midterm National Debate (placeholder)",
        "category": "debate",
        "expected_impact_themes": ["politics", "polls"],
        "patterns": [r"_house_2026$", r"_senate_2026$", r"^bop_"],
    },
    {
        "date": "2026-06-30",
        "time_et": "16:00",
        "name": "OpenAI IPO Target Window Close (per Polymarket)",
        "category": "ipo",
        "expected_impact_themes": ["ai", "ipo", "tech"],
        "patterns": [r"^openai_", r"openai_ipo", r"_ipo_before"],
    },
    {
        "date": "2026-09-30",
        "time_et": "16:00",
        "name": "SpaceX IPO Target Window (per Polymarket)",
        "category": "ipo",
        "expected_impact_themes": ["space", "ipo", "tech"],
        "patterns": [r"^spacex_", r"will_spacex_ipo"],
    },
    {
        "date": "2026-12-31",
        "time_et": "16:00",
        "name": "EOY 2026 IPO Resolution Window (Stripe / Databricks / etc.)",
        "category": "ipo",
        "expected_impact_themes": ["ipo", "tech", "fintech"],
        "patterns": [r"_ipo_before$", r"_ipo_by_december", r"not_ipo_by_december"],
    },
]


# ──────────────────────────────────────────────────────────────────────────
# Pydantic response models
# ──────────────────────────────────────────────────────────────────────────


class CalendarEvent(BaseModel):
    """A single scheduled macro/political/crypto event."""

    date: str = Field(..., description="ISO date YYYY-MM-DD.")
    time_et: str = Field(..., description="Release time, US Eastern (HH:MM).")
    name: str
    category: str = Field(..., description="fomc | cpi | nfp | election | crypto_* | ipo | debate")
    expected_impact_themes: list[str]
    related_markets: list[str] = Field(
        default_factory=list,
        description="Factor IDs from factors.yml expected to react to this event.",
    )
    days_until: int = Field(..., description="Calendar days from today (server clock, UTC).")


class CalendarResponse(BaseModel):
    as_of: str
    horizon_days: int
    n_events: int
    events: list[CalendarEvent]


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _factors_path() -> Path:
    """Resolve the active factors.yml path (env-driven; falls back to package default)."""
    try:
        from pfm.config import get_settings

        return Path(get_settings().factors_file)
    except Exception:
        # 2026-05 refactor: module moved into ``pfm/terminal/``; climb one more
        # parent to reach the package root where ``factors.yml`` still lives.
        return Path(__file__).resolve().parents[1] / "factors.yml"


def _load_factor_ids(path: Path | None = None) -> list[str]:
    """Return all factor IDs from factors.yml; empty list if file unreadable.

    Cached in-process for 1 hour. factors.yml is deploy-time static and
    parsing 10k lines on every /calendar/upcoming request was a measured
    ~1 s hot-path cost.
    """
    p = path or _factors_path()
    cache_key = ("factor_ids", str(p))
    cached = _CAL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        data = yaml.safe_load(p.read_text())
    except FileNotFoundError:
        _CAL_CACHE.set(cache_key, [])
        return []
    except yaml.YAMLError:
        _CAL_CACHE.set(cache_key, [])
        return []
    factors = (data or {}).get("factors", []) or []
    out = [str(f["id"]) for f in factors if isinstance(f, dict) and "id" in f]
    _CAL_CACHE.set(cache_key, out)
    return out


def _match_related_markets(patterns: list[str], factor_ids: list[str]) -> list[str]:
    """Return factor IDs whose names match any of the regex patterns.

    Result is cached on ``(patterns, len(factor_ids))`` — patterns are the
    only variable input across the 30-odd EVENTS, so this collapses ~30
    regex passes over 1360 factor ids into a one-time per-startup cost.
    """
    if not patterns or not factor_ids:
        return []
    # Key on the sorted tuple of patterns + factor_ids identity (we mutate
    # factor_ids only by re-loading the yaml, which itself is cached).
    cache_key = ("related", tuple(patterns), len(factor_ids), factor_ids[0] if factor_ids else "")
    cached = _CAL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    compiled = [re.compile(p) for p in patterns]
    matched: list[str] = []
    for fid in factor_ids:
        if any(rx.search(fid) for rx in compiled):
            matched.append(fid)
    out = sorted(matched)
    _CAL_CACHE.set(cache_key, out)
    return out


def filter_upcoming(
    events: list[dict[str, Any]],
    today: date,
    days: int,
    factor_ids: list[str],
) -> list[CalendarEvent]:
    """Filter raw events to those within [today, today+days] and tag related markets."""
    horizon = today + timedelta(days=days)
    out: list[CalendarEvent] = []
    for ev in events:
        d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        if d < today or d > horizon:
            continue
        related = _match_related_markets(ev.get("patterns", []), factor_ids)
        out.append(
            CalendarEvent(
                date=ev["date"],
                time_et=ev["time_et"],
                name=ev["name"],
                category=ev["category"],
                expected_impact_themes=list(ev.get("expected_impact_themes", [])),
                related_markets=related,
                days_until=(d - today).days,
            )
        )
    out.sort(key=lambda e: (e.date, e.time_et))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────────


@router.get("/upcoming", response_model=CalendarResponse)
def get_upcoming_events(
    days: int = Query(60, ge=1, le=730, description="Look-ahead horizon in days."),
) -> CalendarResponse:
    """List upcoming scheduled macro/political/crypto events that prediction markets react to."""
    if days < 1:
        raise HTTPException(status_code=400, detail="days must be >= 1")
    today = datetime.now(UTC).date()
    # Endpoint output only changes when (today, days) changes — events are
    # hardcoded, factor_ids cached. Cache the assembled response too so
    # repeated polls (UI refresh, multiple workers) hit a pure dict lookup.
    resp_key = ("resp", today.isoformat(), days)
    cached_resp = _CAL_CACHE.get(resp_key)
    if cached_resp is not None:
        return cached_resp
    factor_ids = _load_factor_ids()
    events = filter_upcoming(EVENTS, today=today, days=days, factor_ids=factor_ids)
    resp = CalendarResponse(
        as_of=today.isoformat(),
        horizon_days=days,
        n_events=len(events),
        events=events,
    )
    # 1h TTL — events list rolls forward at most once per day, and same-day
    # warm hits will always be valid.
    _CAL_CACHE.set(resp_key, resp, ttl=3600)
    return resp
