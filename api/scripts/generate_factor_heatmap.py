"""Generate a hierarchically-clustered factor-correlation heatmap PNG.

This is a janitor/diagnostic script (not a runtime endpoint). For the
top-N most-active factors in ``factors.yml`` it fetches a rolling
``window``-day log-return series, computes the pairwise Pearson
correlation matrix, runs SciPy hierarchical clustering on the
``1 - |corr|`` distance, reorders the matrix by cluster, and writes a
matplotlib PNG to ``docs/static/factor-correlation-heatmap.png``.

"Most-active" is defined as the factors with the highest realised
volatility (population std of daily log returns) over the chosen
window. Constant or near-constant factors (which would correlate
trivially with anything) are pushed to the bottom of the ranking.

Two run modes:

* **Live** (default) — for each factor, dispatch to the appropriate
  cached history fetcher (``pfm.factors.fetch_factor_history_dispatch``)
  and slice the last ``--window`` daily log returns. Slow for the full
  1228-factor catalogue (network + disk cache).

* **Fixture** (``--fixture path.json``) — read a JSON dict
  ``{factor_id: [r_1, r_2, ...]}`` of pre-computed return series. Used
  by the unit tests (and any offline reproducibility run) so the math
  + plotting path can be exercised without touching live data.

Run from the ``api/`` directory::

    .venv/bin/python scripts/generate_factor_heatmap.py \\
        --fixture tests/fixtures/factor_heatmap_returns.json \\
        --top-n 50 --window 30 \\
        --out ../docs/static/factor-correlation-heatmap.png

CLI flags mirror the task spec: ``--top-n 100``, ``--window 30``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Make ``pfm`` importable when invoked as
# ``python scripts/generate_factor_heatmap.py`` from the ``api/`` dir.
_HERE = Path(__file__).resolve().parent
_API_ROOT = _HERE.parent
_SRC = _API_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 100
DEFAULT_WINDOW = 30
DEFAULT_OUT = _API_ROOT.parent / "docs" / "static" / "factor-correlation-heatmap.png"
DEFAULT_FACTORS_YML = _SRC / "pfm" / "factors.yml"
# scipy.cluster.hierarchy linkage method. "average" works well on
# ``1 - |corr|`` distances and is robust to outliers.
DEFAULT_LINKAGE = "average"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeatmapResult:
    """Outputs of the heatmap pipeline (excluding the PNG bytes)."""

    factor_ids: list[str]  # ordered as plotted
    correlation: np.ndarray  # (n, n) reordered correlation matrix
    linkage: np.ndarray  # scipy linkage matrix
    skipped: list[str]  # factors dropped before clustering


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def rank_by_volatility(
    returns_by_factor: dict[str, list[float] | np.ndarray],
    *,
    window: int,
    top_n: int,
) -> tuple[list[str], list[str]]:
    """Return the top-N factor ids by recent realised volatility.

    Factors with fewer than ``window`` finite return observations are
    excluded from the ranking and reported in ``skipped``. Ranking is
    deterministic: ties break on factor id (ascending).

    Args:
        returns_by_factor: Mapping from factor id to a sequence of daily
            log returns. Only the trailing ``window`` entries are used.
        window: Trailing observations per factor.
        top_n: Number of factors to keep.

    Returns:
        ``(ranked_ids, skipped)``. ``ranked_ids`` has at most ``top_n``
        entries, sorted by descending volatility (stable). ``skipped``
        lists the factor ids that lacked enough history.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window!r}")
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n!r}")

    scored: list[tuple[float, str]] = []
    skipped: list[str] = []
    for fid, series in returns_by_factor.items():
        arr = np.asarray(series, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size < window:
            logger.warning(
                "factor %s: only %d finite return observations (<%d), skipping",
                fid,
                int(finite.size),
                window,
            )
            skipped.append(fid)
            continue
        tail = finite[-window:]
        vol = float(np.std(tail))
        scored.append((vol, fid))

    # Sort: vol descending, factor id ascending for tie-break determinism.
    scored.sort(key=lambda t: (-t[0], t[1]))
    ranked = [fid for _vol, fid in scored[:top_n]]
    return ranked, sorted(skipped)


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


def build_correlation_matrix(
    returns_by_factor: dict[str, list[float] | np.ndarray],
    factor_ids: list[str],
    *,
    window: int,
) -> np.ndarray:
    """Compute the pairwise Pearson correlation matrix for ``factor_ids``.

    Each row of the input matrix is the trailing-``window`` log-return
    vector for the corresponding factor. NaNs are coerced to 0 before
    correlation so a single bad observation does not collapse a whole
    row to NaN.

    Returns:
        ``(n, n)`` ndarray with ones on the diagonal. The off-diagonal
        entries are clipped to ``[-1, 1]`` (numpy can return e.g.
        ``1.0000000002`` due to floating point).
    """
    n = len(factor_ids)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    matrix = np.zeros((n, window), dtype=float)
    for i, fid in enumerate(factor_ids):
        arr = np.asarray(returns_by_factor[fid], dtype=float)
        finite = arr[np.isfinite(arr)]
        tail = finite[-window:]
        matrix[i, :] = np.nan_to_num(tail, nan=0.0, posinf=0.0, neginf=0.0)

    # numpy.corrcoef raises a warning for constant rows; suppress and
    # post-process: replace NaN entries (which arise from zero-variance
    # rows) with 0 off-diagonal and 1 on-diagonal.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(matrix)
    if corr.ndim == 0:
        # corrcoef on 1×k returns a scalar; promote to (1,1).
        corr = np.array([[float(corr)]], dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Hierarchical clustering / reordering
# ---------------------------------------------------------------------------


def hierarchical_order(
    correlation: np.ndarray,
    *,
    method: str = DEFAULT_LINKAGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute hierarchical clustering and return (order, linkage).

    The distance metric is ``1 - |corr|`` so highly anti-correlated
    pairs are treated as similar (they share a factor of variation,
    just with opposite sign). ``order`` is a permutation of
    ``range(n)`` that places clustered factors next to each other.

    A 0×0 or 1×1 input returns ``(arange(n), empty_linkage)`` without
    calling scipy.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    n = correlation.shape[0]
    if n <= 1:
        return np.arange(n, dtype=int), np.zeros((0, 4), dtype=float)

    # 1 - |corr| in [0, 1]; zero out the diagonal so squareform accepts.
    dist = 1.0 - np.abs(correlation)
    np.fill_diagonal(dist, 0.0)
    # squareform symmetrises; tiny floating-point asymmetry from corrcoef
    # is fine, but ensure exact symmetry to keep scipy happy.
    dist = (dist + dist.T) / 2.0
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method=method)
    order = leaves_list(z).astype(int)
    return order, z


def reorder_matrix(
    correlation: np.ndarray,
    factor_ids: list[str],
    order: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Permute both axes of ``correlation`` and reorder ``factor_ids``."""
    if order.size == 0:
        return correlation, list(factor_ids)
    reordered = correlation[np.ix_(order, order)]
    reordered_ids = [factor_ids[i] for i in order.tolist()]
    return reordered, reordered_ids


# ---------------------------------------------------------------------------
# PNG rendering
# ---------------------------------------------------------------------------


def render_heatmap(
    correlation: np.ndarray,
    factor_ids: list[str],
    out_path: Path,
    *,
    title: str | None = None,
    dpi: int = 120,
) -> Path:
    """Render ``correlation`` to a PNG at ``out_path``.

    The colormap is ``RdBu_r`` (red = positive, blue = negative)
    clamped to ``[-1, 1]`` so the legend is comparable across runs.
    Tick labels are dropped when there are >40 factors (otherwise the
    plot is unreadable); a colorbar always appears.

    Returns ``out_path`` (after writing) for caller convenience.
    """
    # Use a non-interactive backend so this works on CI / headless boxes.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = correlation.shape[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Square figure scaled with n; cap so very-large-N still produces a
    # readable image.
    side = max(6.0, min(18.0, 0.18 * max(n, 1) + 4.0))
    fig, ax = plt.subplots(figsize=(side, side), dpi=dpi)
    # Empty input: render a placeholder so callers still get a valid PNG
    # instead of an imshow warning + blank axes.
    plot_data = correlation if n > 0 else np.zeros((1, 1), dtype=float)
    im = ax.imshow(
        plot_data,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(
        title or f"Factor correlation heatmap (n={n}, hierarchical)",
        fontsize=11,
    )
    if n <= 40:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(factor_ids, rotation=90, fontsize=6)
        ax.set_yticklabels(factor_ids, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{n} factors (labels suppressed)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson ρ")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, format="png")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Return-series sourcing (live vs. fixture)
# ---------------------------------------------------------------------------


def load_fixture(path: Path) -> dict[str, list[float]]:
    """Read a JSON fixture of pre-computed return series.

    The fixture format is a plain object ``{factor_id: [r_1, r_2, ...]}``;
    missing factors are simply omitted (treated as "no history"
    downstream).
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            f"fixture at {path} must be a JSON object {{factor_id: [returns]}}; "
            f"got {type(raw).__name__}"
        )
    out: dict[str, list[float]] = {}
    for fid, series in raw.items():
        if not isinstance(series, list):
            raise ValueError(
                f"fixture entry {fid!r}: expected list of floats, got {type(series).__name__}"
            )
        out[str(fid)] = [float(x) for x in series]
    return out


def fetch_live_returns(
    factor_ids: list[str],
    *,
    window: int,
    factors_yml: Path,
) -> dict[str, list[float]]:
    """Fetch the last ``window`` log returns for each factor from cache/API.

    Wraps :func:`pfm.factors.fetch_factor_history_dispatch`. Errors per
    factor are caught and yield "no history" (the caller will skip the
    factor with a warning).

    .. note::
        Slow path. With 1228 factors and a cold cache this can take
        minutes. Use ``--fixture`` (or ``--limit``) for development.
    """
    import pandas as pd

    from pfm.factors import (
        FactorConfig,
        fetch_factor_history_dispatch,
        load_factors,
    )

    factors = load_factors(factors_yml)
    # Pull a slightly longer window than ``window`` to give the returns
    # diff a buffer (we drop the first observation when differencing).
    end = pd.Timestamp(datetime.now(UTC))
    start = end - pd.Timedelta(days=window * 3 + 5)

    out: dict[str, list[float]] = {}
    for fid in factor_ids:
        fc: FactorConfig | None = factors.get(fid)
        if fc is None:
            logger.warning("factor %s: not in factors.yml; skipping", fid)
            continue
        try:
            df = fetch_factor_history_dispatch(fc, start=start, end=end)
        except Exception as exc:  # network/cache failure → skip
            logger.warning("factor %s: history fetch failed (%s); skipping", fid, exc)
            continue
        if df is None or df.empty or "price" not in df.columns:
            logger.warning("factor %s: empty history frame; skipping", fid)
            continue
        prices = np.asarray(df["price"].to_numpy(), dtype=float)
        if prices.size < window + 1:
            logger.warning(
                "factor %s: only %d price observations (<%d+1), skipping",
                fid,
                int(prices.size),
                window,
            )
            continue
        safe = np.clip(prices, 1e-6, None)
        log_rets = np.diff(np.log(safe))
        out[fid] = log_rets[-window:].tolist()
    return out


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def generate_heatmap(
    returns_by_factor: dict[str, list[float] | np.ndarray],
    *,
    top_n: int,
    window: int,
    method: str = DEFAULT_LINKAGE,
) -> HeatmapResult:
    """Run the full math pipeline (rank → correlate → cluster → reorder).

    Returns a :class:`HeatmapResult`. The caller is responsible for
    rendering the PNG (so the math can be exercised separately in
    tests).
    """
    ranked, skipped = rank_by_volatility(returns_by_factor, window=window, top_n=top_n)
    correlation = build_correlation_matrix(returns_by_factor, ranked, window=window)
    order, linkage_matrix = hierarchical_order(correlation, method=method)
    reordered, reordered_ids = reorder_matrix(correlation, ranked, order)
    return HeatmapResult(
        factor_ids=reordered_ids,
        correlation=reordered,
        linkage=linkage_matrix,
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a hierarchically-clustered factor-correlation heatmap "
            "PNG. Top-N most-active factors are picked by recent volatility."
        )
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top-volatility factors to plot (default: {DEFAULT_TOP_N}).",
    )
    p.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Trailing daily observations per factor (default: {DEFAULT_WINDOW}).",
    )
    p.add_argument(
        "--method",
        default=DEFAULT_LINKAGE,
        help=f"scipy linkage method (default: {DEFAULT_LINKAGE}).",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output PNG path (default: {DEFAULT_OUT}).",
    )
    p.add_argument(
        "--fixture",
        default=None,
        help=(
            "Path to a JSON fixture {factor_id: [returns]}. When set, "
            "the script does not hit any live source — used for offline "
            "tests and reproducibility runs."
        ),
    )
    p.add_argument(
        "--factors-yml",
        default=str(DEFAULT_FACTORS_YML),
        help="Override path to factors.yml (live mode only).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit live-mode fetch to the first N factor ids (sorted).",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Override plot title.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="PNG output DPI (default: 120).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-run stdout summary.",
    )
    return p.parse_args(argv)


def _print_summary(result: HeatmapResult, *, out_path: Path) -> None:
    n = len(result.factor_ids)
    print()
    print(f"factors plotted : {n}")
    print(f"skipped         : {len(result.skipped)}")
    if result.correlation.size > 0:
        # Off-diagonal stats — diagonal is always 1 and would skew the mean.
        mask = ~np.eye(n, dtype=bool)
        if mask.any():
            off = result.correlation[mask]
            print(
                f"off-diag ρ      : mean={float(off.mean()):+0.3f}  "
                f"min={float(off.min()):+0.3f}  max={float(off.max()):+0.3f}"
            )
    print(f"PNG written     : {out_path}")


def run(args: argparse.Namespace) -> int:
    """Execute the pipeline; return the desired process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.fixture:
        fixture_path = Path(args.fixture)
        if not fixture_path.exists():
            print(f"ERROR: fixture {fixture_path} not found.", file=sys.stderr)
            return 2
        returns_by_factor = load_fixture(fixture_path)
    else:
        from pfm.factors import load_factors

        factors_yml = Path(args.factors_yml)
        factors = load_factors(factors_yml)
        factor_ids = sorted(factors.keys())
        if args.limit is not None:
            factor_ids = factor_ids[: args.limit]
        returns_by_factor = fetch_live_returns(
            factor_ids, window=args.window, factors_yml=factors_yml
        )

    result = generate_heatmap(
        returns_by_factor,
        top_n=args.top_n,
        window=args.window,
        method=args.method,
    )

    out_path = Path(args.out)
    render_heatmap(
        result.correlation,
        result.factor_ids,
        out_path,
        title=args.title,
        dpi=args.dpi,
    )

    if not args.quiet:
        _print_summary(result, out_path=out_path)

    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
