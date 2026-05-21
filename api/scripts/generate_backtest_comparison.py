"""Generate ``web/data/backtest_comparison.json`` for the frontend.

For each *deployable* strategy in ``web/data/alpha_strategies.json`` (tier in
``A_STRUCTURAL``, ``A_GOLD``, ``B_VALIDATED``), the script runs a small
synthetic backtest grounded in the strategy's own published statistics:

* Daily mean return is anchored on the strategy's ``oos_sharpe`` (annualised),
  rescaled to daily with ``μ_d = sharpe · σ_d`` where ``σ_d = 0.01`` is a
  conservative per-day vol.
* Path noise is reproducible via a ``pair_id``-derived seed so this script is
  deterministic across runs (important for CI snapshots).
* If callers provide pre-baked PnL fixtures via ``--fixtures path.json``, the
  script loads the daily returns from there instead of simulating.

The resulting JSON has the shape:

.. code-block:: json

    {
      "generated_at": "2026-05-16T00:00:00Z",
      "n_strategies": 33,
      "n_days": 60,
      "strategies": [
        {
          "pair_id": "...",
          "tier": "...",
          "metrics": {
            "sharpe": 0.62,
            "deflated_sharpe": 0.38,
            "max_drawdown": -0.07,
            "win_rate": 0.55,
            "n_trades": 32,
            "total_return": 0.12,
            "annual_vol": 0.16
          },
          "equity_curve": [1.0, 1.001, ...]   /* length = n_days */
        }
      ],
      "comparison": {
        "best_sharpe": {"pair_id": "...", "value": 1.4},
        "best_deflated_sharpe": {...},
        "best_max_drawdown": {...},
        "best_win_rate": {...},
        "worst_max_drawdown": {...}
      }
    }

Run from the repo root::

    python3 api/scripts/generate_backtest_comparison.py

Custom output path::

    python3 api/scripts/generate_backtest_comparison.py --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants — exported for tests.
# ---------------------------------------------------------------------------

DEPLOYABLE_TIERS: frozenset[str] = frozenset({"A_STRUCTURAL", "A_GOLD", "B_VALIDATED"})
DEFAULT_N_DAYS: int = 60
DEFAULT_DAILY_VOL: float = 0.01  # 1% per day → ~16% annualised; conservative
TRADING_DAYS_PER_YEAR: int = 252

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "web" / "data" / "alpha_strategies.json"
DEFAULT_OUTPUT = ROOT / "web" / "data" / "backtest_comparison.json"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _seed_for_pair(pair_id: str) -> int:
    """Deterministic, well-distributed 32-bit seed from a pair_id string."""
    digest = hashlib.sha256(pair_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def simulate_daily_returns(
    sharpe: float,
    n_days: int = DEFAULT_N_DAYS,
    daily_vol: float = DEFAULT_DAILY_VOL,
    seed: int = 0,
) -> np.ndarray:
    """Generate a synthetic daily-return series with the requested ex-ante Sharpe.

    The series is ``r_d = μ_d + σ_d · ε_t`` with ``ε_t ~ N(0,1)`` and
    ``μ_d = (sharpe / sqrt(252)) · σ_d``. ``sharpe`` is the *target* annualised
    Sharpe; the realised Sharpe will drift from it by finite-sample noise,
    which is the point — that drift is what the deflated-Sharpe haircut
    captures.
    """
    if n_days <= 0:
        raise ValueError("n_days must be positive")
    if daily_vol <= 0:
        raise ValueError("daily_vol must be positive")
    rng = np.random.default_rng(seed)
    mu_d = (sharpe / math.sqrt(TRADING_DAYS_PER_YEAR)) * daily_vol
    eps = rng.standard_normal(n_days)
    return mu_d + daily_vol * eps


def equity_curve_from_returns(returns: Iterable[float]) -> list[float]:
    """Cumulative compounded equity, starting at 1.0, one entry per day."""
    curve: list[float] = []
    nav = 1.0
    for r in returns:
        nav *= 1.0 + float(r)
        curve.append(nav)
    return curve


def sharpe_ratio(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd == 0.0 or not math.isfinite(sd):
        return 0.0
    return float(np.mean(returns)) / sd * math.sqrt(TRADING_DAYS_PER_YEAR)


def deflated_sharpe(sharpe: float, n_obs: int, n_trials: int = 1) -> float:
    """Bailey/Lopez de Prado deflated Sharpe haircut (simplified).

    We apply two penalties:

    * Finite-sample variance of the SR estimator: ``var(SR) ≈ (1 + 0.5·SR²)/N``.
    * Multiple-testing inflation: subtract ``E[max SR over n_trials]`` using the
      Bailey-LdP normal approximation
      ``E[max] ≈ √2·log(N_trials) − γ/√(2·log(N_trials))``.

    Returns a non-finite-safe float (always finite).
    """
    if n_obs <= 1:
        return 0.0
    var_sr = (1.0 + 0.5 * sharpe * sharpe) / n_obs
    if var_sr <= 0:
        return sharpe
    se = math.sqrt(var_sr)
    if n_trials <= 1:
        emax = 0.0
    else:
        gamma = 0.5772156649015329  # Euler-Mascheroni
        log_n = math.log(n_trials)
        if log_n <= 0:
            emax = 0.0
        else:
            emax = math.sqrt(2.0 * log_n) - gamma / math.sqrt(2.0 * log_n)
    return float(sharpe - emax * se)


def max_drawdown(equity: list[float]) -> float:
    """Maximum peak-to-trough drawdown as a negative number (or 0.0)."""
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for nav in equity:
        peak = max(peak, nav)
        if peak <= 0:
            continue
        dd = (nav - peak) / peak
        worst = min(worst, dd)
    return float(worst)


def win_rate(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float((returns > 0).sum()) / float(returns.size)


def annual_vol(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    return float(np.std(returns, ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)


def count_trades(returns: np.ndarray, threshold: float = 0.0) -> int:
    """Count days where ``|r| > threshold`` — a reasonable proxy for active days."""
    if returns.size == 0:
        return 0
    return int((np.abs(returns) > threshold).sum())


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def backtest_strategy(
    strategy: dict,
    n_days: int = DEFAULT_N_DAYS,
    fixture_returns: list[float] | None = None,
) -> dict:
    """Run a synthetic (or fixture-driven) backtest for one strategy.

    Returns a dict with ``pair_id``, ``tier``, ``metrics``, ``equity_curve``.
    """
    pair_id = str(strategy.get("pair_id") or "")
    tier = str(strategy.get("tier") or "")
    target_sharpe = _as_float(strategy.get("oos_sharpe"))

    if fixture_returns is not None:
        returns = np.asarray(fixture_returns, dtype=float)
        if returns.size == 0:
            returns = simulate_daily_returns(target_sharpe, n_days, seed=_seed_for_pair(pair_id))
    else:
        returns = simulate_daily_returns(
            target_sharpe,
            n_days=n_days,
            seed=_seed_for_pair(pair_id),
        )

    curve = equity_curve_from_returns(returns)
    sr = sharpe_ratio(returns)
    n_trials = max(int(_as_float(strategy.get("n_obs"), default=returns.size)), returns.size)
    metrics = {
        "sharpe": round(sr, 4),
        "deflated_sharpe": round(deflated_sharpe(sr, returns.size, n_trials=n_trials), 4),
        "max_drawdown": round(max_drawdown(curve), 4),
        "win_rate": round(win_rate(returns), 4),
        "n_trades": count_trades(returns),
        "total_return": round(curve[-1] - 1.0, 4) if curve else 0.0,
        "annual_vol": round(annual_vol(returns), 4),
        "target_sharpe": round(target_sharpe, 4),
    }
    return {
        "pair_id": pair_id,
        "tier": tier,
        "metrics": metrics,
        "equity_curve": [round(v, 6) for v in curve],
    }


def filter_deployable(strategies: list[dict]) -> list[dict]:
    """Return only strategies whose tier is in ``DEPLOYABLE_TIERS``."""
    return [s for s in strategies if str(s.get("tier", "")) in DEPLOYABLE_TIERS]


def aggregate_comparison(rows: list[dict]) -> dict:
    """Compute cross-strategy best/worst metrics.

    Each value points back to the strategy's ``pair_id`` for the UI to deep-link.
    Empty input returns an empty mapping (no None/NaN leaks).
    """
    if not rows:
        return {}

    def best_by(key: str, *, want_max: bool = True) -> dict:
        ranked = sorted(
            rows,
            key=lambda r: r["metrics"].get(key, float("-inf") if want_max else float("inf")),
            reverse=want_max,
        )
        top = ranked[0]
        return {"pair_id": top["pair_id"], "value": top["metrics"].get(key)}

    return {
        "best_sharpe": best_by("sharpe", want_max=True),
        "best_deflated_sharpe": best_by("deflated_sharpe", want_max=True),
        "best_max_drawdown": best_by("max_drawdown", want_max=True),  # closest to 0
        "worst_max_drawdown": best_by("max_drawdown", want_max=False),
        "best_win_rate": best_by("win_rate", want_max=True),
        "best_total_return": best_by("total_return", want_max=True),
    }


def build_comparison(
    strategies: list[dict],
    n_days: int = DEFAULT_N_DAYS,
    fixtures: dict[str, list[float]] | None = None,
    generated_at: str | None = None,
) -> dict:
    """Compose the full output dict consumed by the frontend."""
    fixtures = fixtures or {}
    deployable = filter_deployable(strategies)
    rows: list[dict] = []
    for s in deployable:
        pair_id = str(s.get("pair_id") or "")
        fr = fixtures.get(pair_id)
        rows.append(backtest_strategy(s, n_days=n_days, fixture_returns=fr))

    # Stable sort: best Sharpe first so the JSON is ready to display
    rows.sort(key=lambda r: r["metrics"].get("sharpe", 0.0), reverse=True)

    return {
        "generated_at": generated_at
        or datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "n_strategies": len(rows),
        "n_days": n_days,
        "deployable_tiers": sorted(DEPLOYABLE_TIERS),
        "strategies": rows,
        "comparison": aggregate_comparison(rows),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_strategies(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    strategies = payload.get("strategies")
    if not isinstance(strategies, list):
        raise ValueError(f"{path}: expected top-level 'strategies' array")
    return strategies


def _load_fixtures(path: Path | None) -> dict[str, list[float]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected mapping of pair_id → [returns]")
    out: dict[str, list[float]] = {}
    for k, v in raw.items():
        if not isinstance(v, list):
            continue
        out[str(k)] = [float(x) for x in v]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-days", type=int, default=DEFAULT_N_DAYS)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="Optional JSON file mapping pair_id → daily returns list (overrides simulation)",
    )
    args = parser.parse_args(argv)

    strategies = _load_strategies(args.input)
    fixtures = _load_fixtures(args.fixtures)
    payload = build_comparison(strategies, n_days=args.n_days, fixtures=fixtures)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        f"Wrote {args.out} — {payload['n_strategies']} deployable strategies, "
        f"{payload['n_days']} days each."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
