"""4-quarter Sharpe stability stress test.

CLAUDE.md anti-alpha rule:

    Every "wow" backtest from a single window must be cross-validated
    against >=4 disjoint quarters. If sign flips or Sharpe collapses in
    any quarter, it goes on the anti-alpha list, not the deployable list.

This script implements that gate. For each disjoint calendar quarter from
``--start`` for ``--quarters`` quarters it computes:

* annualised Sharpe ratio
* Sortino ratio (downside-only volatility)
* sign of mean PnL
* t-stat of mean PnL (scipy.stats.ttest_1samp against zero)
* Bailey & Lopez de Prado (2014) deflated Sharpe (per-period scale)

A quarter is marked FAIL when its Sharpe < 0.5 OR the sign of its mean PnL
flips relative to the full-sample mean. Any FAIL quarter fails the whole
run.

Usage
-----

::

    python scripts/stress_test.py --strategy buy-and-hold --start 2024-01 --quarters 4

The pass/fail table is printed to stdout; a machine-readable JSON report is
written to ``/tmp/stress_<strategy>_<YYYYMMDD>.json``.

Synthetic price data is generated deterministically (seeded from the
strategy name) so the script is reproducible without external data
dependencies. Real strategies should hook their own price series in by
registering themselves in :mod:`pfm.strategies_registry` and overriding
``--prices``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure ``src/`` is importable without requiring a pip install.
_API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_API_ROOT / "src"))

from pfm.multitest import deflated_sharpe_full
from pfm.strategies_registry import Strategy, get

logger = logging.getLogger("pfm.stress_test")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def annualised_sharpe(
    pnl: pd.Series, *, risk_free: float = 0.0, ann_factor: float = 252.0
) -> float:
    """Annualised Sharpe over a daily PnL series.

    Per-period excess return mean over std, scaled by ``sqrt(ann_factor)``.
    Returns ``0.0`` when std is zero or NaN (degenerate / flat strategy).
    """
    if pnl is None or len(pnl) < 2:
        return 0.0
    excess = pnl - (risk_free / ann_factor)
    sd = float(excess.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(ann_factor))


def annualised_sortino(
    pnl: pd.Series, *, risk_free: float = 0.0, ann_factor: float = 252.0
) -> float:
    """Annualised Sortino — Sharpe using *downside* deviation only."""
    if pnl is None or len(pnl) < 2:
        return 0.0
    excess = pnl - (risk_free / ann_factor)
    downside = excess[excess < 0.0]
    if len(downside) < 2:
        return 0.0
    dd = float(downside.std(ddof=1))
    if not np.isfinite(dd) or dd <= 0.0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(ann_factor))


def t_stat(pnl: pd.Series) -> tuple[float, float]:
    """One-sample t-stat of mean PnL against zero. Returns ``(t, p)``."""
    from scipy import stats as sci_stats

    if pnl is None or len(pnl) < 2:
        return 0.0, 1.0
    arr = np.asarray(pnl.dropna(), dtype=float)
    if arr.size < 2 or float(np.std(arr, ddof=1)) <= 0.0:
        return 0.0, 1.0
    res = sci_stats.ttest_1samp(arr, popmean=0.0)
    return float(res.statistic), float(res.pvalue)


def mean_sign(pnl: pd.Series) -> int:
    """Sign of mean PnL: +1 / -1 / 0."""
    if pnl is None or len(pnl) == 0:
        return 0
    m = float(pnl.mean())
    if not np.isfinite(m) or m == 0.0:
        return 0
    return 1 if m > 0.0 else -1


# ---------------------------------------------------------------------------
# Quarter slicing
# ---------------------------------------------------------------------------


def parse_start(start: str) -> pd.Timestamp:
    """Accept ``YYYY-MM`` or ``YYYY-MM-DD``. Normalises to start of month."""
    s = start.strip()
    if len(s) == 7:  # YYYY-MM
        return pd.Timestamp(f"{s}-01", tz="UTC")
    return pd.Timestamp(s, tz="UTC").normalize()


def quarter_windows(start: pd.Timestamp, n: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """``n`` disjoint quarter (start, end-exclusive) windows from ``start``."""
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    for _ in range(n):
        nxt = cur + pd.DateOffset(months=3)
        out.append((cur, nxt))
        cur = nxt
    return out


# ---------------------------------------------------------------------------
# Synthetic prices (deterministic per strategy name)
# ---------------------------------------------------------------------------


def synthetic_prices(
    start: pd.Timestamp,
    n_days: int,
    *,
    seed_token: str = "default",
    drift: float = 0.0002,
    vol: float = 0.012,
) -> pd.DataFrame:
    """GBM-ish daily close series, seeded reproducibly from ``seed_token``."""
    seed_int = int(hashlib.sha256(seed_token.encode()).hexdigest()[:8], 16) % (2**32)
    rng = np.random.default_rng(seed_int)
    idx = pd.date_range(start.tz_convert("UTC"), periods=n_days, freq="D", tz="UTC")
    log_ret = rng.normal(loc=drift, scale=vol, size=n_days)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame({"close": close}, index=idx)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def evaluate_quarter(
    pnl: pd.Series,
    *,
    full_sample_sign: int,
    sharpe_floor: float = 0.5,
    risk_free: float = 0.0,
    ann_factor: float = 252.0,
    n_trials: int = 1,
) -> dict[str, float | str | int | bool]:
    """Compute the per-quarter metric row + verdict."""
    sharpe = annualised_sharpe(pnl, risk_free=risk_free, ann_factor=ann_factor)
    sortino = annualised_sortino(pnl, risk_free=risk_free, ann_factor=ann_factor)
    t, p = t_stat(pnl)
    sign = mean_sign(pnl)

    # Deflated Sharpe needs higher moments to be honest.
    arr = np.asarray(pnl.dropna(), dtype=float)
    if arr.size >= 5:
        from scipy import stats as sci_stats

        skew_val = float(sci_stats.skew(arr, bias=False))
        # scipy.stats.kurtosis returns excess by default; we want Pearson.
        kurt_val = float(sci_stats.kurtosis(arr, bias=False) + 3.0)
    else:
        skew_val, kurt_val = 0.0, 3.0
    dsr = deflated_sharpe_full(
        sharpe_observed=sharpe,
        n_obs=int(arr.size),
        n_trials=n_trials,
        skew=skew_val,
        kurtosis=kurt_val,
        annualisation=ann_factor,
    )

    sign_flip = sign != 0 and full_sample_sign != 0 and sign != full_sample_sign
    fail_sharpe = sharpe < sharpe_floor
    fail = bool(sign_flip or fail_sharpe)
    reason = []
    if fail_sharpe:
        reason.append(f"Sharpe {sharpe:.2f} < {sharpe_floor}")
    if sign_flip:
        reason.append(f"sign flip vs full ({sign} vs {full_sample_sign})")
    return {
        "n_obs": int(arr.size),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "sign": int(sign),
        "t_stat": float(t),
        "p_value": float(p),
        "deflated_sharpe": float(dsr["deflated_sharpe"]),
        "deflated_p_value": float(dsr["deflated_p_value"]),
        "expected_max_sharpe_under_null": float(dsr["expected_max_sharpe_under_null"]),
        "fail": fail,
        "fail_reason": "; ".join(reason) if reason else "",
    }


def run_stress(
    strategy: Strategy,
    *,
    start: pd.Timestamp,
    quarters: int,
    prices: pd.DataFrame | None = None,
    sharpe_floor: float = 0.5,
    risk_free: float = 0.0,
    ann_factor: float = 252.0,
) -> dict:
    """Run the stress test and return a JSON-serialisable report."""
    windows = quarter_windows(start, quarters)
    end = windows[-1][1]

    if prices is None:
        n_days = int((end - start).days) + 5
        prices = synthetic_prices(start, n_days, seed_token=strategy.name)
    prices = prices.sort_index()

    full_pnl = strategy.compute_pnl(prices).dropna()
    full_sharpe = annualised_sharpe(full_pnl, risk_free=risk_free, ann_factor=ann_factor)
    full_sign = mean_sign(full_pnl)

    rows: list[dict] = []
    for i, (qs, qe) in enumerate(windows, start=1):
        mask = (full_pnl.index >= qs) & (full_pnl.index < qe)
        q_pnl = full_pnl.loc[mask]
        row = evaluate_quarter(
            q_pnl,
            full_sample_sign=full_sign,
            sharpe_floor=sharpe_floor,
            risk_free=risk_free,
            ann_factor=ann_factor,
            n_trials=quarters,
        )
        row.update(
            {
                "quarter": i,
                "start": qs.strftime("%Y-%m-%d"),
                "end": qe.strftime("%Y-%m-%d"),
            }
        )
        rows.append(row)

    full_arr = np.asarray(full_pnl.dropna(), dtype=float)
    if full_arr.size >= 5:
        from scipy import stats as sci_stats

        skew_full = float(sci_stats.skew(full_arr, bias=False))
        kurt_full = float(sci_stats.kurtosis(full_arr, bias=False) + 3.0)
    else:
        skew_full, kurt_full = 0.0, 3.0
    full_dsr = deflated_sharpe_full(
        sharpe_observed=full_sharpe,
        n_obs=int(full_arr.size),
        n_trials=quarters,
        skew=skew_full,
        kurtosis=kurt_full,
        annualisation=ann_factor,
    )

    overall_fail = any(r["fail"] for r in rows)
    verdict = "FAIL" if overall_fail else "PASS"

    return {
        "strategy": strategy.name,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "start": start.strftime("%Y-%m-%d"),
        "quarters": quarters,
        "sharpe_floor": float(sharpe_floor),
        "risk_free": float(risk_free),
        "ann_factor": float(ann_factor),
        "full_sample": {
            "n_obs": int(full_arr.size),
            "sharpe": float(full_sharpe),
            "sign": int(full_sign),
            "deflated_sharpe": float(full_dsr["deflated_sharpe"]),
            "deflated_p_value": float(full_dsr["deflated_p_value"]),
        },
        "quarter_rows": rows,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def format_table(report: dict) -> str:
    """Compact stdout pass/fail table."""
    lines: list[str] = []
    head = (
        f"Stress test: strategy={report['strategy']!r}  "
        f"start={report['start']}  quarters={report['quarters']}  "
        f"floor={report['sharpe_floor']:.2f}"
    )
    lines.append(head)
    lines.append("-" * len(head))
    lines.append(
        f"{'Q':>2}  {'window':<24}  {'n':>4}  "
        f"{'Sharpe':>8}  {'Sortino':>8}  {'sign':>5}  "
        f"{'t':>7}  {'DSR':>7}  verdict"
    )
    for r in report["quarter_rows"]:
        verdict = "FAIL" if r["fail"] else "PASS"
        win = f"{r['start']} → {r['end']}"
        lines.append(
            f"{r['quarter']:>2}  {win:<24}  {r['n_obs']:>4}  "
            f"{r['sharpe']:>8.3f}  {r['sortino']:>8.3f}  {r['sign']:>5d}  "
            f"{r['t_stat']:>7.2f}  {r['deflated_sharpe']:>7.3f}  {verdict}"
        )
        if r["fail_reason"]:
            lines.append(f"     reason: {r['fail_reason']}")
    full = report["full_sample"]
    lines.append("-" * len(head))
    lines.append(
        f"FULL  n={full['n_obs']}  Sharpe={full['sharpe']:.3f}  "
        f"sign={full['sign']:+d}  DSR={full['deflated_sharpe']:.3f}  "
        f"DSR-p={full['deflated_p_value']:.3f}"
    )
    lines.append(f"VERDICT: {report['verdict']}")
    return "\n".join(lines)


def default_report_path(strategy: str, today: date | None = None) -> Path:
    """``/tmp/stress_<strategy>_<YYYYMMDD>.json``."""
    today = today or date.today()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in strategy)
    return Path(f"/tmp/stress_{safe}_{today.strftime('%Y%m%d')}.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4-quarter Sharpe stability stress test (CLAUDE.md anti-alpha rule)."
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="Strategy name registered in pfm.strategies_registry.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Quarter-window start, YYYY-MM or YYYY-MM-DD (UTC).",
    )
    parser.add_argument(
        "--quarters",
        type=int,
        default=4,
        help="Number of disjoint quarter windows (default: 4).",
    )
    parser.add_argument(
        "--sharpe-floor",
        type=float,
        default=0.5,
        help="Fail threshold for per-quarter Sharpe (default: 0.5).",
    )
    parser.add_argument(
        "--risk-free",
        type=float,
        default=0.0,
        help="Annualised risk-free rate (default: 0).",
    )
    parser.add_argument(
        "--ann-factor",
        type=float,
        default=252.0,
        help="Annualisation factor (default: 252).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: /tmp/stress_<strategy>_<YYYYMMDD>.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(argv)
    try:
        strategy = get(args.strategy)
    except KeyError as exc:
        logger.error("%s", exc)
        return 2

    start = parse_start(args.start)
    report = run_stress(
        strategy,
        start=start,
        quarters=args.quarters,
        sharpe_floor=args.sharpe_floor,
        risk_free=args.risk_free,
        ann_factor=args.ann_factor,
    )
    print(format_table(report))

    out_path = args.output or default_report_path(args.strategy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Wrote JSON report to %s", out_path)

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
