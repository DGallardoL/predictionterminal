"""``pfm.crypto5min`` — model probability vs market probability for the
Polymarket short-dated (5m / 15m) BTC & ETH up/down markets.

This module is the *trading edge* surface for the cryptostuff microstructure
engine: it consumes the per-symbol live state (OFI, σ_short, last price)
produced by ``pfm.crypto_events_engine`` and produces a calibrated
``P(symbol up by end of window)`` that we compare to the live Polymarket
midpoint. The frontend (Strategies → Crypto Micro) renders the comparison
as a table with an edge bar and a BUY YES / BUY NO / WAIT pill.

Public surface
--------------
``predict_up_prob``     — closed-form predictor (pure function).
``predict_for_window``  — high-level wrapper that returns a structured
                           prediction for the current 5m/15m boundary.
``CryptoFiveMinState``  — rolling spot buffer + boundary cache.
``discover_active_markets`` — auto-discovers active btc-updown / eth-updown
                              markets via the Polymarket Gamma API.
``compare_market_vs_model`` — pairs a model prediction with a Polymarket
                              midpoint and emits a discrete signal.
``router``              — FastAPI sub-router that gets mounted on the app.
"""

from __future__ import annotations

from pfm.crypto5min.comparator import (
    DEFAULT_EDGE_THRESHOLD,
    ComparisonResult,
    compare_market_vs_model,
    decide_signal,
)
from pfm.crypto5min.confidence import (
    ConfidenceBreakdown,
    ConfidenceResult,
    build_confidence_result,
    compute_confidence_score,
    compute_z_edge,
    compute_z_model,
    signal_strength_from_confidence,
)
from pfm.crypto5min.market_fetcher import (
    ActiveMarket,
    discover_active_markets,
    fetch_clob_midpoint,
    parse_active_market,
)
from pfm.crypto5min.predictor import (
    ModelPrediction,
    PredictorInputs,
    predict_for_window,
    predict_up_prob,
)
from pfm.crypto5min.router import router
from pfm.crypto5min.state import CryptoFiveMinState, WindowAnchor, get_state

__all__ = [
    "DEFAULT_EDGE_THRESHOLD",
    "ActiveMarket",
    "ComparisonResult",
    "ConfidenceBreakdown",
    "ConfidenceResult",
    "CryptoFiveMinState",
    "ModelPrediction",
    "PredictorInputs",
    "WindowAnchor",
    "build_confidence_result",
    "compare_market_vs_model",
    "compute_confidence_score",
    "compute_z_edge",
    "compute_z_model",
    "decide_signal",
    "discover_active_markets",
    "fetch_clob_midpoint",
    "get_state",
    "parse_active_market",
    "predict_for_window",
    "predict_up_prob",
    "router",
    "signal_strength_from_confidence",
]
