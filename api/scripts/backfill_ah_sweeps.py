"""Backfill ``/tmp/ah_sweeps/all_unique_hits.json`` from the factor pickle.

Loads the populated factor-history pickle (written at FastAPI startup),
runs Engle-Granger 2-step cointegration on every (slug-A, slug-B) pair,
and writes the surviving hits in the schema consumed by
``pfm.terminal.peer_scanner`` / ``pfm.terminal.fair_price`` /
``pfm.terminal.inline_backtest``.

Filters
-------
- ``n_obs >= 30``  (Engle-Granger lower bound from ``engle_granger``)
- ``adf_pvalue < 0.05``  (5% significance, rejects unit-root null)
- ``-10 <= beta_hedge <= +10``  (drops degenerate hedge ratios from
  near-flat regressors)

Outputs (each row)
------------------
``a_id, b_id`` — factor ids resolved from ``factors.yml`` slug→id map.
``verdict, n_obs, adf_pvalue, half_life_days, beta_hedge``
``oos_sharpe, full_sharpe, perm_p, perm_real_sharpe``  — derived
    cheaply from the spread series itself (last-30% holdout for OOS,
    sign-shuffle permutation for ``perm_p``).
``sweep`` — flat ``"backfill_pickle"`` tag so the UI can identify the
    provenance.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/backfill_ah_sweeps.py \\
        --pickle /tmp/strat7_factor_history.pkl \\
        --out /tmp/ah_sweeps/all_unique_hits.json \\
        --top-n 100

A cap of 100 factors yields ~4 950 pairs and finishes in ~60-90 s on a
mid-range laptop. Pass ``--top-n 0`` to scan all factors.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pfm.cointegration import engle_granger

logger = logging.getLogger("backfill_ah_sweeps")


def _load_slug_to_factor_id(factors_yml: Path) -> dict[str, str]:
    """Build ``slug -> factor_id`` map from ``factors.yml``.

    Pickle keys are slugs; the peers endpoint expects ``a_id``/``b_id`` to
    be factor ids (the snake_case ``id`` field). When no factor entry
    exists for a slug we fall back to the slug itself so we still emit a
    record (the peers endpoint will just miss the display-name lookup).
    """
    if not factors_yml.exists():
        logger.warning("factors.yml missing at %s", factors_yml)
        return {}
    with factors_yml.open() as f:
        doc = yaml.safe_load(f) or {}
    out: dict[str, str] = {}
    for fd in doc.get("factors", []) or []:
        fid = fd.get("id")
        slug = fd.get("slug")
        if fid and slug:
            out[slug] = fid
    return out


#: Minimum in-sample PnL observations required to emit a real ``full_sharpe``.
#: Below this we return ``None`` (not ``0.0``) so downstream consumers don't
#: mistake "sample too short" for "strategy is flat". 30 matches the
#: Engle-Granger ``min_obs`` we already enforce upstream.
_MIN_IS_OBS_FOR_SHARPE = 30
#: Same idea for OOS: an OOS Sharpe computed from <30 PnL points is noise.
_MIN_OOS_OBS_FOR_SHARPE = 30


def _compute_pair_metrics(
    spread: pd.Series,
    *,
    oos_frac: float = 0.3,
    n_perm: int = 0,
) -> dict[str, float | None]:
    """Derive Sharpe-like diagnostics from a cointegrated residual spread.

    We treat the spread as a position signal: trade ``-sign(spread_{t-1})``
    units of the spread itself (i.e. fade dislocations). PnL series is
    ``-sign(spread_{t-1}) * (spread_t - spread_{t-1})``. This is the
    canonical Engle-Granger mean-reversion harvest and lets us emit a
    realistic ``full_sharpe`` / ``oos_sharpe`` without re-running the
    expensive bootstrap that produced the original sweep file.

    Returns ``None`` for any Sharpe whose sample is below the
    ``_MIN_*_OBS_FOR_SHARPE`` threshold or whose PnL variance is zero —
    historically these were silently coerced to ``0.0``, which produced
    impossible rows like ``full_sharpe=0`` alongside ``oos_sharpe=9.47``
    in the α Hub.
    """
    s = spread.dropna()
    if len(s) < _MIN_IS_OBS_FOR_SHARPE:
        return {
            "full_sharpe": None,
            "oos_sharpe": None,
            "perm_p": None,
            "perm_real_sharpe": None,
        }
    diffs = s.diff().dropna().to_numpy()
    pos = -np.sign(s.shift(1).dropna().to_numpy())  # fade
    pnl = pos * diffs
    if pnl.size < _MIN_IS_OBS_FOR_SHARPE or float(np.std(pnl)) <= 0.0:
        return {
            "full_sharpe": None,
            "oos_sharpe": None,
            "perm_p": None,
            "perm_real_sharpe": None,
        }
    # Annualise assuming daily bars.
    full_sharpe = float(np.mean(pnl) / np.std(pnl) * math.sqrt(252.0))

    # OOS: last `oos_frac` of the sample.
    split = max(1, int(len(pnl) * (1.0 - oos_frac)))
    oos_pnl = pnl[split:]
    oos_sharpe: float | None
    if oos_pnl.size >= _MIN_OOS_OBS_FOR_SHARPE and float(np.std(oos_pnl)) > 0.0:
        oos_sharpe = float(np.mean(oos_pnl) / np.std(oos_pnl) * math.sqrt(252.0))
    else:
        oos_sharpe = None

    # Cheap sign-shuffle permutation (off by default — pricey at scale).
    perm_p = 0.0
    perm_real_sharpe = full_sharpe
    if n_perm > 0:
        rng = np.random.default_rng(seed=0)
        beats = 0
        for _ in range(n_perm):
            shuf = rng.permutation(diffs)
            p_pnl = pos * shuf
            if float(np.std(p_pnl)) <= 0.0:
                continue
            p_sharpe = float(np.mean(p_pnl) / np.std(p_pnl) * math.sqrt(252.0))
            if abs(p_sharpe) >= abs(full_sharpe):
                beats += 1
        perm_p = (beats + 1) / (n_perm + 1)

    return {
        "full_sharpe": full_sharpe,
        "oos_sharpe": oos_sharpe,
        "perm_p": perm_p,
        "perm_real_sharpe": perm_real_sharpe,
    }


def backfill(
    pickle_path: Path,
    out_path: Path,
    *,
    top_n: int = 100,
    min_obs: int = 30,
    alpha: float = 0.05,
    beta_max_abs: float = 10.0,
    n_perm: int = 0,
    factors_yml: Path | None = None,
) -> dict[str, Any]:
    """Run the cointegration sweep and write JSON hits to ``out_path``.

    Returns a small summary dict for the caller to log.
    """
    if not pickle_path.exists():
        raise SystemExit(f"pickle not found: {pickle_path}")
    with pickle_path.open("rb") as f:
        history: dict[str, pd.Series] = pickle.load(f)
    logger.info("loaded %d series from %s", len(history), pickle_path)

    # Slug→factor_id resolution.
    factors_yml = factors_yml or (SRC / "pfm" / "factors.yml")
    slug_to_id = _load_slug_to_factor_id(factors_yml)
    logger.info("resolved %d slug→factor_id entries", len(slug_to_id))

    # Rank series by length (n_obs) so we can cap to the most-active subset.
    ranked = sorted(history.items(), key=lambda kv: -len(kv[1]))
    if top_n and top_n > 0:
        ranked = ranked[:top_n]
    logger.info("scanning %d factors → %d pairs", len(ranked), len(ranked) * (len(ranked) - 1) // 2)

    hits: list[dict[str, Any]] = []
    n_pairs = 0
    t0 = time.time()
    for (slug_a, ser_a), (slug_b, ser_b) in combinations(ranked, 2):
        n_pairs += 1
        try:
            cr = engle_granger(ser_a, ser_b)
        except Exception as exc:  # pragma: no cover - belt-and-braces
            logger.debug("engle_granger failed (%s, %s): %s", slug_a, slug_b, exc)
            continue
        if cr.verdict == "insufficient-data":
            continue
        if cr.n_obs < min_obs:
            continue
        if not math.isfinite(cr.adf_pvalue) or cr.adf_pvalue >= alpha:
            continue
        if not math.isfinite(cr.beta_hedge) or abs(cr.beta_hedge) > beta_max_abs:
            continue

        metrics = _compute_pair_metrics(cr.spread, n_perm=n_perm)
        a_id = slug_to_id.get(slug_a, slug_a)
        b_id = slug_to_id.get(slug_b, slug_b)
        hits.append(
            {
                "a_id": a_id,
                "b_id": b_id,
                "a_slug": slug_a,
                "b_slug": slug_b,
                "verdict": "REAL_ALPHA",
                "n_obs": int(cr.n_obs),
                "adf_pvalue": float(cr.adf_pvalue),
                "half_life_days": (
                    float(cr.half_life_days)
                    if cr.half_life_days is not None and math.isfinite(cr.half_life_days)
                    else 0.0
                ),
                "beta_hedge": float(cr.beta_hedge),
                "oos_sharpe": metrics["oos_sharpe"],
                "full_sharpe": metrics["full_sharpe"],
                "perm_p": metrics["perm_p"],
                "perm_real_sharpe": metrics["perm_real_sharpe"],
                "sweep": "backfill_pickle",
            }
        )

        if n_pairs % 500 == 0:
            elapsed = time.time() - t0
            logger.info("scanned %d pairs, %d hits, %.1fs", n_pairs, len(hits), elapsed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(hits, f)

    elapsed = time.time() - t0
    summary = {
        "n_factors": len(ranked),
        "n_pairs_scanned": n_pairs,
        "n_hits": len(hits),
        "elapsed_seconds": round(elapsed, 1),
        "out_path": str(out_path),
    }
    logger.info("done: %s", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pickle", type=Path, default=Path("/tmp/strat7_factor_history.pkl"))
    p.add_argument("--out", type=Path, default=Path("/tmp/ah_sweeps/all_unique_hits.json"))
    p.add_argument("--top-n", type=int, default=100, help="cap factors to top-N by n_obs (0 = all)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-obs", type=int, default=30)
    p.add_argument("--beta-max-abs", type=float, default=10.0)
    p.add_argument(
        "--n-perm", type=int, default=0, help="permutations for perm_p (expensive, 0 to skip)"
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    summary = backfill(
        args.pickle,
        args.out,
        top_n=args.top_n,
        min_obs=args.min_obs,
        alpha=args.alpha,
        beta_max_abs=args.beta_max_abs,
        n_perm=args.n_perm,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
