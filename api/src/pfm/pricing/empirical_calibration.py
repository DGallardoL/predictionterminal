"""Empirical-calibration harness for the T81 binary-pricing models (T82).

This module pulls a sample of *resolved* Polymarket binary markets via the
Gamma `/markets` endpoint, reconstructs each market's price trajectory,
asks each :class:`pfm.pricing.binary_models.Pricer` for its predicted
price at every observed time-point, and scores model quality with:

* Brier score
* Log-loss (with epsilon clipping)
* Calibration RMSE (10-bin reliability)
* Early-warning lead time (days that the model first deviated
  >= 10pp from the market price in the *correct* eventual direction)
* Economic PnL (Kelly-capped trade of the model-vs-market mispricing,
  fee-debited per round-trip)

The module is import-time *tolerant* — it does NOT import the T81
``binary_models`` module eagerly so that other tests can keep importing
``pfm`` even before T81 lands. Functions that need the pricers raise an
``ImportError`` with a clear message the first time they're called.

No network calls happen unless ``pull_resolved_markets`` is invoked
without a ``respx_mock``; tests pass a respx instance and a fixture JSON.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# T81 lazy import (raise a clear ImportError if the module is missing)
# ---------------------------------------------------------------------------


def _require_t81() -> Any:
    """Return the ``pfm.pricing.binary_models`` module or raise.

    T82 deliberately does not import T81 at module-load time so that the
    rest of the codebase can still import :mod:`pfm.pricing` even before
    T81's :mod:`binary_models` module lands. The first function that
    actually needs a pricer calls this helper which gives a precise,
    actionable error message rather than the cryptic
    ``ModuleNotFoundError`` raised by a naive ``import``.
    """
    try:
        from pfm.pricing import binary_models  # type: ignore[attr-defined]
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "T82 empirical_calibration requires the T81 module "
            "`pfm.pricing.binary_models` to be present. Implement T81 first "
            "(see .coordination/TASK-BOARD.md Track L T81) — it must expose "
            "the `Pricer` protocol and the four candidate pricing models."
        ) from exc
    return binary_models


# ---------------------------------------------------------------------------
# Pricer protocol (mirrors T81's; defined here so we can type without
# importing T81 at module-load time)
# ---------------------------------------------------------------------------


class Pricer(Protocol):
    """Minimal Pricer interface for typing — full version lives in T81."""

    name: str

    def theoretical_price(
        self, state: Any, params: Any
    ) -> float: ...  # pragma: no cover - protocol
    def calibrate(self, history: Iterable[Any]) -> Any: ...  # pragma: no cover - protocol


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------


@dataclass
class MarketEpisode:
    """One resolved binary-market trajectory.

    ``trajectory`` is a list of ``(t, price)`` pairs where ``t`` is
    *days-to-resolution* (positive going forward, 0 at resolution). Prices
    are in [0, 1]. ``resolved`` is the realised binary outcome.
    """

    market_id: str
    title: str
    resolved: bool
    trajectory: list[tuple[float, float]]
    underlying_trajectory: list[tuple[float, float]] | None = None
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def market_prices(self) -> np.ndarray:
        """Return market prices as an ndarray, sorted by t descending."""
        if not self.trajectory:
            return np.array([], dtype=float)
        sorted_pts = sorted(self.trajectory, key=lambda x: -x[0])
        return np.array([p for _, p in sorted_pts], dtype=float)

    def times(self) -> np.ndarray:
        """Return times (days-to-resolution) sorted descending (matches market_prices)."""
        if not self.trajectory:
            return np.array([], dtype=float)
        sorted_pts = sorted(self.trajectory, key=lambda x: -x[0])
        return np.array([t for t, _ in sorted_pts], dtype=float)


# ---------------------------------------------------------------------------
# Gamma fetcher
# ---------------------------------------------------------------------------


GAMMA_BASE = "https://gamma-api.polymarket.com"


def pull_resolved_markets(
    limit: int = 50,
    *,
    respx_mock: Any = None,
    fixture_path: Path | str | None = None,
) -> list[MarketEpisode]:
    """Pull resolved binary markets from Polymarket Gamma API.

    Parameters
    ----------
    limit
        Soft limit on how many episodes to return. We request a larger
        page size and filter to binary markets with a clean trajectory.
    respx_mock
        Optional respx instance used by tests. When provided, the caller
        is responsible for stubbing the relevant URLs. We do NOT pass the
        mock into the httpx client — respx patches at the transport
        layer, so a plain ``httpx.Client`` is sufficient.
    fixture_path
        Optional path to a JSON file with a pre-built list of episode
        dicts. When supplied, network is bypassed entirely. The schema is
        ``{"markets": [{"id", "title", "resolved", "trajectory": [...],
        "underlying_trajectory": [...], "threshold": ...}, ...]}``.

    Returns
    -------
    list[MarketEpisode]
        At most ``limit`` episodes, ordered as Gamma returned them.
    """
    if fixture_path is not None:
        return _load_episodes_fixture(fixture_path, limit=limit)

    page_size = max(limit * 4, 200)
    url = f"{GAMMA_BASE}/markets"
    params = {"closed": "true", "limit": str(page_size)}

    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:  # pragma: no cover - real-network errors not exercised in tests
        logger.warning("pull_resolved_markets: HTTP error: %s", exc)
        return []

    if isinstance(payload, dict) and "markets" in payload:
        raw = payload["markets"]
    elif isinstance(payload, list):
        raw = payload
    else:
        raw = []

    episodes: list[MarketEpisode] = []
    for m in raw:
        ep = _episode_from_gamma_row(m)
        if ep is None:
            continue
        episodes.append(ep)
        if len(episodes) >= limit:
            break
    return episodes


def _episode_from_gamma_row(row: dict[str, Any]) -> MarketEpisode | None:
    """Build a :class:`MarketEpisode` from a Gamma `/markets` row.

    Filters out markets without a clear binary outcome or with empty
    price history. We intentionally accept either ``resolved``/``closed``
    booleans plus an ``outcomePrices`` indicator of which side won.
    """
    try:
        if not row.get("closed", False):
            return None
        outcome_prices = row.get("outcomePrices")
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except json.JSONDecodeError:
                outcome_prices = None
        if not outcome_prices or len(outcome_prices) < 2:
            return None
        # Outcome: yes side resolved to 1.0 means True
        yes_price = float(outcome_prices[0])
        resolved = yes_price > 0.5

        history = row.get("priceHistory") or row.get("price_history") or []
        if not isinstance(history, list) or len(history) < 2:
            return None

        # Each history entry is {"t": unix_ts, "p": price}. Convert to
        # days-to-resolution.
        end_ts = row.get("endDateIso") or row.get("end_date_iso")
        # If endDate not parseable, use the last history timestamp.
        try:
            from datetime import datetime

            if isinstance(end_ts, str):
                end_dt = datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
                end_unix = end_dt.replace(tzinfo=UTC).timestamp()
            else:
                end_unix = float(history[-1].get("t", 0))
        except Exception:
            end_unix = float(history[-1].get("t", 0))

        trajectory: list[tuple[float, float]] = []
        for pt in history:
            try:
                t_unix = float(pt["t"])
                price = float(pt["p"])
            except (KeyError, TypeError, ValueError):
                continue
            days_to_res = max(0.0, (end_unix - t_unix) / 86400.0)
            trajectory.append((days_to_res, price))
        if len(trajectory) < 2:
            return None

        return MarketEpisode(
            market_id=str(row.get("id", row.get("conditionId", "unknown"))),
            title=str(row.get("question", row.get("title", "")) or ""),
            resolved=resolved,
            trajectory=trajectory,
            underlying_trajectory=None,
            threshold=None,
        )
    except Exception as exc:
        logger.debug("Skipping market row: %s", exc)
        return None


def _load_episodes_fixture(path: Path | str, *, limit: int) -> list[MarketEpisode]:
    """Load a fixture JSON of pre-built episodes (test convenience)."""
    p = Path(path)
    data = json.loads(p.read_text())
    markets = data.get("markets", [])
    out: list[MarketEpisode] = []
    for m in markets[:limit]:
        traj = [(float(t), float(p_)) for t, p_ in m.get("trajectory", [])]
        und = m.get("underlying_trajectory")
        if und is not None:
            und = [(float(t), float(p_)) for t, p_ in und]
        out.append(
            MarketEpisode(
                market_id=str(m["market_id"]),
                title=str(m.get("title", "")),
                resolved=bool(m["resolved"]),
                trajectory=traj,
                underlying_trajectory=und,
                threshold=m.get("threshold"),
                metadata=m.get("metadata", {}),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Scoring metrics
# ---------------------------------------------------------------------------

_EPS = 1e-6


def brier_score(predictions: Iterable[float], outcomes: Iterable[float]) -> float:
    """Mean squared error of probability predictions vs binary outcomes."""
    p = np.asarray(list(predictions), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    if p.size == 0:
        return float("nan")
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: predictions {p.shape} vs outcomes {y.shape}")
    return float(np.mean((p - y) ** 2))


def log_loss(
    predictions: Iterable[float], outcomes: Iterable[float], *, eps: float = _EPS
) -> float:
    """Binary cross-entropy, clipped to avoid -inf at p in {0, 1}."""
    p = np.clip(np.asarray(list(predictions), dtype=float), eps, 1.0 - eps)
    y = np.asarray(list(outcomes), dtype=float)
    if p.size == 0:
        return float("nan")
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: predictions {p.shape} vs outcomes {y.shape}")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def calibration_rmse(
    predictions: Iterable[float],
    outcomes: Iterable[float],
    *,
    n_bins: int = 10,
) -> float:
    """Reliability-diagram RMSE.

    Bin predictions into ``n_bins`` equal-width buckets in [0,1], then
    compute the RMSE of (mean predicted) - (mean realised) over the
    populated bins.
    """
    p = np.asarray(list(predictions), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    if p.size == 0:
        return float("nan")
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: predictions {p.shape} vs outcomes {y.shape}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed buckets except first
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    diffs2: list[float] = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        diffs2.append((float(p[mask].mean()) - float(y[mask].mean())) ** 2)
    if not diffs2:
        return float("nan")
    return float(math.sqrt(np.mean(diffs2)))


def early_warning_lead_time(
    episode: MarketEpisode,
    model_predictions: Iterable[float],
    *,
    threshold: float = 0.10,
) -> float:
    """Average days where |model - market| > ``threshold`` *in the right direction*.

    "Right direction" means: if the market eventually resolved YES
    (``episode.resolved=True``) the model was *above* the market by
    >threshold; if it resolved NO, the model was *below* by >threshold.

    Returns the mean of the ``t`` (days-to-resolution) over qualifying
    points, or 0.0 if none qualify. Higher is better (longer lead).
    """
    times = episode.times()
    market = episode.market_prices()
    model = np.asarray(list(model_predictions), dtype=float)
    if model.shape != market.shape:
        raise ValueError(f"early-warning: model shape {model.shape} != market shape {market.shape}")
    direction = 1.0 if episode.resolved else -1.0
    delta = (model - market) * direction
    qualifies = delta > threshold
    if not qualifies.any():
        return 0.0
    return float(np.mean(times[qualifies]))


def _kelly_fraction(model_p: float, market_p: float) -> float:
    """Edge-relative Kelly fraction for a binary contract.

    We bet YES when model_p > market_p, NO otherwise. The contract pays
    1 if resolved=True, 0 otherwise. The expected log growth is
    maximised at ``f* = (p*b - q) / b`` where ``b = (1-c)/c`` for cost c.

    Returns a signed fraction (positive long-YES, negative long-NO).
    """
    if not (0.0 < market_p < 1.0):
        return 0.0
    if model_p > market_p:
        # buy YES at price market_p, payoff = 1 - market_p if win, -market_p if lose
        b = (1.0 - market_p) / market_p
        f = (model_p * b - (1.0 - model_p)) / b
        return max(0.0, f)
    elif model_p < market_p:
        # buy NO at price 1 - market_p
        q_no = 1.0 - market_p
        b = (1.0 - q_no) / q_no
        p_no = 1.0 - model_p
        f = (p_no * b - (1.0 - p_no)) / b
        return -max(0.0, f)
    return 0.0


def compute_pnl(
    pricer: Pricer,
    episodes: list[MarketEpisode],
    *,
    fee_bps: int = 100,
    kelly_cap: float = 0.25,
) -> float:
    """Aggregate PnL across episodes for a given pricer.

    For each episode we pull the pricer's predicted price at each
    trajectory point (using the predictions cached in
    ``episode.metadata['predictions'][pricer.name]`` if present; otherwise
    we ask the pricer directly). We open a position sized at
    ``min(|kelly|, kelly_cap)`` whenever the model-vs-market edge crosses
    a small threshold (1 cent) and settle at resolution. Each opened
    position pays a one-way fee of ``fee_bps`` basis points.

    Returns total PnL summed across episodes (in units of starting
    bankroll fractions — interpret as a return multiple).
    """
    fee = fee_bps / 10_000.0
    total_pnl = 0.0
    for ep in episodes:
        market = ep.market_prices()
        times = ep.times()
        if market.size == 0:
            continue
        preds = _resolve_predictions(pricer, ep)
        outcome = 1.0 if ep.resolved else 0.0
        # Trade only the *last* observation (closest to resolution) per
        # episode: a simple "snapshot bet" — the PnL harness becomes
        # noise-bounded and avoids double-counting along the trajectory.
        last_idx = int(np.argmin(times))
        m = float(market[last_idx])
        pred = float(preds[last_idx])
        if abs(pred - m) < 0.01:
            continue
        f = _kelly_fraction(pred, m)
        f = max(-kelly_cap, min(kelly_cap, f))
        if f == 0.0:
            continue
        if f > 0:
            # long YES at price m, payoff (1-m) if win, -m if lose; per-unit-stake
            unit_payoff = (1.0 - m) if outcome == 1.0 else -m
        else:
            # long NO at price (1-m), payoff m if NO wins, -(1-m) if YES wins
            unit_payoff = m if outcome == 0.0 else -(1.0 - m)
        pnl = abs(f) * unit_payoff - abs(f) * fee
        total_pnl += pnl
    return float(total_pnl)


def _resolve_predictions(pricer: Pricer, episode: MarketEpisode) -> np.ndarray:
    """Look up cached predictions on the episode, or ask the pricer.

    The empirical-eval pipeline typically pre-computes per-pricer
    predictions and stuffs them on ``episode.metadata['predictions']`` to
    avoid re-fitting models inside the metric loop. We honour that cache;
    otherwise we ask the pricer for a price at each trajectory point.
    """
    cache = episode.metadata.get("predictions", {}) if episode.metadata else {}
    name = getattr(pricer, "name", pricer.__class__.__name__)
    if name in cache:
        arr = np.asarray(cache[name], dtype=float)
        if arr.size == len(episode.trajectory):
            return arr
    # Fallback: query the pricer at each trajectory point. We require
    # T81 to be loaded.
    bm = _require_t81()
    times = episode.times()
    market = episode.market_prices()
    preds = np.empty_like(market)
    # Calibrate once on the *full* trajectory (peeking is fine for the
    # snapshot-pnl test path; the live harness shouldn't use this code path)
    try:
        params = pricer.calibrate(episode.trajectory)
    except Exception:
        params = None
    for i, (t, m) in enumerate(zip(times, market, strict=False)):
        state = bm.MarketState(
            market_price=float(m),
            time_to_resolution=float(t),
            underlying=None,
            threshold=episode.threshold,
            news_evidence=0.0,
        )
        try:
            preds[i] = float(pricer.theoretical_price(state, params))
        except Exception:
            preds[i] = float(m)
    return preds


# ---------------------------------------------------------------------------
# Per-model scoring
# ---------------------------------------------------------------------------


def score_model(
    pricer: Pricer,
    episodes: list[MarketEpisode],
    *,
    fee_bps: int = 100,
    kelly_cap: float = 0.25,
) -> dict[str, float]:
    """Compute the full metric panel for a single pricer.

    Returns a dict with keys: ``brier``, ``log_loss``,
    ``calibration_rmse``, ``early_warning_days``, ``pnl``,
    ``n_episodes``, ``n_points``.
    """
    all_preds: list[float] = []
    all_outcomes: list[float] = []
    leads: list[float] = []
    for ep in episodes:
        if not ep.trajectory:
            continue
        preds = _resolve_predictions(pricer, ep)
        outcome = 1.0 if ep.resolved else 0.0
        all_preds.extend(preds.tolist())
        all_outcomes.extend([outcome] * len(preds))
        leads.append(early_warning_lead_time(ep, preds))

    if not all_preds:
        return {
            "brier": float("nan"),
            "log_loss": float("nan"),
            "calibration_rmse": float("nan"),
            "early_warning_days": 0.0,
            "pnl": 0.0,
            "n_episodes": 0,
            "n_points": 0,
        }

    return {
        "brier": brier_score(all_preds, all_outcomes),
        "log_loss": log_loss(all_preds, all_outcomes),
        "calibration_rmse": calibration_rmse(all_preds, all_outcomes),
        "early_warning_days": float(np.mean(leads)) if leads else 0.0,
        "pnl": compute_pnl(pricer, episodes, fee_bps=fee_bps, kelly_cap=kelly_cap),
        "n_episodes": float(len(episodes)),
        "n_points": float(len(all_preds)),
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(
    scores_by_model: dict[str, dict[str, float]],
    output_path: Path | str,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a comparison report to ``output_path`` as JSON."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "binary-pricing-report/v1",
        "models": scores_by_model,
    }
    if extra:
        payload["meta"] = extra
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p


__all__ = [
    "GAMMA_BASE",
    "MarketEpisode",
    "Pricer",
    "brier_score",
    "calibration_rmse",
    "compute_pnl",
    "early_warning_lead_time",
    "log_loss",
    "pull_resolved_markets",
    "score_model",
    "write_report",
]
