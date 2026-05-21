"""Schemas for /terminal/* (legacy inline terminal handlers in pfm.main)."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---- /terminal/* ------------------------------------------------------------
# Yahoo-Finance-style data hub: one endpoint per UI panel so the frontend can
# render market detail / overview / search with minimal client-side logic.


class TerminalLive(BaseModel):
    """Live order-book snapshot for a single market (from Polymarket Gamma)."""

    best_bid: float | None = None
    best_ask: float | None = None
    midpoint: float | None = None
    last_trade_price: float | None = None
    spread_cents: float | None = None
    volume_24hr: float | None = None
    volume_total: float | None = None
    liquidity: float | None = None
    one_day_price_change: float | None = None
    one_week_price_change: float | None = None


class TerminalMeta(BaseModel):
    """Static / slowly-changing metadata for a market."""

    slug: str
    question: str
    description: str | None = None
    theme: str | None = None
    resolution_source: str | None = None
    end_date: str | None = None
    start_date: str | None = None
    created_at: str | None = None
    days_to_resolve: int | None = None
    age_days: int | None = None
    active: bool = True
    closed: bool = False


class TerminalStats(BaseModel):
    """Mean-reversion / persistence diagnostics computed from cached history."""

    n_obs: int = 0
    half_life_days: float | None = None
    dfa_alpha: float | None = None
    dfa_interpretation: str | None = None
    variance_ratio: float | None = None
    variance_ratio_verdict: str | None = None
    realized_vol_30d: float | None = None
    current_price: float | None = None


class TerminalPeer(BaseModel):
    """One cointegrated peer market surfaced from the alpha-hunter sweep cache."""

    peer_id: str
    half_life_days: float | None = None
    adf_pvalue: float | None = None
    beta_hedge: float | None = None
    oos_sharpe: float | None = None
    full_sharpe: float | None = None
    perm_p: float | None = None
    verdict: str | None = None
    sweep: str | None = None
    fair_price: float | None = None  # rolling-EG implied fair value vs this peer


class TerminalMarketResponse(BaseModel):
    """Aggregate response for ``GET /terminal/market/{slug}``.

    Top-level convenience fields (``question``, ``price``, ``volume_24h``,
    ``theme``, ``resolution_iso``) are projected from ``meta``/``live`` so
    the front-end doesn't have to know the nested layout. The UX audit
    (2026-05-14) found code paths that read each name in three different
    spots — the aliases keep one canonical source of truth here.
    """

    slug: str
    live: TerminalLive
    meta: TerminalMeta
    stats: TerminalStats
    peers: list[TerminalPeer] = Field(default_factory=list)
    # Convenience aliases (projected at construction time in main.py).
    question: str | None = None
    theme: str | None = None
    price: float | None = None
    volume_24h: float | None = None
    resolution_iso: str | None = None


class TerminalHistoryBar(BaseModel):
    t: int
    p: float


class TerminalHistoryResponse(BaseModel):
    """Pass-through response for ``GET /terminal/market/{slug}/history``."""

    slug: str
    yes_token_id: str
    fidelity: int
    n_bars: int
    history: list[TerminalHistoryBar]


class TerminalThemeBucket(BaseModel):
    theme: str
    n_markets: int
    median_24h_change: float | None = None
    median_volume_24hr: float | None = None
    total_volume_24hr: float | None = None
    # Median YES-side probability across the theme's markets. Useful as a
    # 4th heatmap dimension (color by sentiment) — values near 0.5 are
    # markets in genuine doubt; near 0 / 1 are near-resolved consensus.
    median_yes_price: float | None = None


class TerminalMover(BaseModel):
    slug: str
    question: str
    theme: str | None = None
    price: float | None = None
    one_day_price_change: float | None = None
    volume_24hr: float | None = None


class TerminalNewMarket(BaseModel):
    slug: str
    question: str
    theme: str | None = None
    price: float | None = None
    created_at: str | None = None
    age_days: int | None = None


class TerminalUpcomingResolution(BaseModel):
    slug: str
    question: str
    theme: str | None = None
    price: float | None = None
    end_date: str | None = None
    days_to_resolve: int | None = None
    conviction: float | None = None  # |price − 0.5| · 2 ∈ [0, 1]


class TerminalOverviewResponse(BaseModel):
    """Aggregate response for ``GET /terminal/overview``."""

    n_markets_considered: int
    theme_heatmap: list[TerminalThemeBucket]
    top_movers: list[TerminalMover]
    most_traded: list[TerminalMover]
    recently_launched: list[TerminalNewMarket]
    upcoming_resolutions: list[TerminalUpcomingResolution]


class TerminalSearchHit(BaseModel):
    factor_id: str
    name: str
    slug: str
    theme: str | None = None
    score: float
    current_price: float | None = None
    # Alias of ``current_price`` — the UX audit (2026-05-14) flagged that the
    # front-end and the ⌘-K palette both look for ``price``. Keep both so the
    # contract remains backward-compatible for any existing client.
    price: float | None = None
    # 24-hour notional volume from the gamma prewarm (top ~1000 markets by
    # volume). ``None`` for factors that are not currently active on
    # Polymarket — the front-end should treat null as "no live data".
    volume_24h: float | None = None


class TerminalSearchResponse(BaseModel):
    query: str
    n_results: int
    results: list[TerminalSearchHit]
