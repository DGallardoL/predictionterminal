"""KMeans-cluster every factor in ``factors.yml`` by its rolling-30d return series.

This is a janitor/diagnostic script (not a runtime endpoint). It loads the
factor catalogue, builds a ``(n_factors x window)`` matrix of recent log
returns, z-scores each row, runs scikit-learn KMeans, and writes a JSON
report to ``/tmp/factor-clusters.json`` summarising cluster membership.

Two run modes:

* **Live** (default) — for each factor, dispatch to the appropriate cached
  history fetcher and slice the last ``--window`` daily log returns. Slow
  for the full 1228-factor catalogue (network + disk cache). The script is
  deliberately permissive: factors with insufficient history are skipped
  with a single ``logger.warning`` per slug, and the report records them
  under ``meta.skipped``.

* **Fixture** (``--fixture path.json``) — read a JSON dict ``{factor_id:
  [r_1, r_2, ...]}`` of pre-computed return series. Used by the unit
  tests (and any offline reproducibility run) so the math can be
  exercised without touching live data.

The output schema is documented in the spec for task W11-56:

.. code-block:: json

    {
      "generated_at": "2026-05-16T12:34:56Z",
      "k": 20,
      "factor_count": 1228,
      "window": 30,
      "seed": 42,
      "clusters": [
        {
          "id": 0,
          "size": 87,
          "factors": ["slug1", "slug2", ...],
          "centroid_summary": "mean=+0.013 std=0.41 min=-1.20 max=+1.18"
        }
      ],
      "skipped": ["slug-with-no-history", ...]
    }

The CLI prints (a) cluster sizes and (b) up to five member factors per
cluster as a quick eyeball summary.

Run from the ``api/`` directory::

    .venv/bin/python scripts/factor_cluster_report.py \\
        --fixture tests/fixtures/factor_cluster_returns.json \\
        --k 4 --window 30 --out /tmp/factor-clusters.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Make ``pfm`` importable when invoked as ``python scripts/factor_cluster_report.py``
# from the ``api/`` directory.
_HERE = Path(__file__).resolve().parent
_API_ROOT = _HERE.parent
_SRC = _API_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logger = logging.getLogger(__name__)

DEFAULT_WINDOW = 30
DEFAULT_K = 20
DEFAULT_SEED = 42
DEFAULT_OUT = Path("/tmp/factor-clusters.json")
DEFAULT_N_INIT = 10  # KMeans n_init for stability (spec requirement)

# Path is computed lazily so ``--factors-yml`` overrides remain testable.
FACTORS_YML = _SRC / "pfm" / "factors.yml"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterRow:
    """One row of the output ``clusters`` array."""

    id: int
    size: int
    factors: list[str]
    centroid_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "size": self.size,
            "factors": self.factors,
            "centroid_summary": self.centroid_summary,
        }


# ---------------------------------------------------------------------------
# Feature-matrix construction
# ---------------------------------------------------------------------------


def _zscore_row(row: np.ndarray) -> np.ndarray:
    """Z-score a 1-D vector; constant rows collapse to all zeros (not NaN).

    Spec requires per-row z-scoring before clustering so factors are
    compared on shape, not level/volatility. A constant series has no
    variance, so we fall back to zeros (which then sit at the origin of
    feature space and naturally cluster with other flat factors).
    """
    mu = float(np.nanmean(row))
    sigma = float(np.nanstd(row))
    if sigma == 0.0 or not np.isfinite(sigma):
        return np.zeros_like(row, dtype=float)
    out = (row - mu) / sigma
    # NaN→0 so KMeans (which can't ingest NaN) still gets a defined point.
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_matrix(
    returns_by_factor: dict[str, list[float] | np.ndarray],
    *,
    window: int = DEFAULT_WINDOW,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Build the ``(n, window)`` z-scored feature matrix.

    Factors with fewer than ``window`` return observations are skipped
    (a warning is logged once per slug).

    Args:
        returns_by_factor: Mapping from factor id to a sequence of daily
            log returns. The **last** ``window`` entries are used so callers
            can pass longer histories without preprocessing.
        window: Number of trailing observations to keep per factor.

    Returns:
        ``(factor_ids, matrix, skipped)`` where ``factor_ids`` is the
        deterministic ordering of rows in ``matrix`` (sorted), ``matrix``
        has shape ``(len(factor_ids), window)`` and is z-scored row-wise,
        and ``skipped`` lists the factor ids that lacked enough history.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window!r}")

    rows: list[np.ndarray] = []
    kept: list[str] = []
    skipped: list[str] = []

    # Deterministic ordering for reproducibility.
    for fid in sorted(returns_by_factor.keys()):
        series = returns_by_factor[fid]
        arr = np.asarray(series, dtype=float)
        # Drop NaNs from the tail count so a factor with 30 obs of which 5
        # are NaN doesn't masquerade as having enough data.
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
        rows.append(_zscore_row(tail))
        kept.append(fid)

    if not rows:
        return kept, np.zeros((0, window), dtype=float), skipped

    matrix = np.vstack(rows)
    return kept, matrix, skipped


# ---------------------------------------------------------------------------
# KMeans wrapper
# ---------------------------------------------------------------------------


def run_kmeans(
    matrix: np.ndarray,
    *,
    k: int,
    seed: int = DEFAULT_SEED,
    n_init: int = DEFAULT_N_INIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit sklearn KMeans and return ``(labels, centroids)``.

    A separate function so tests can call it directly without the YAML
    or fixture-loading scaffold.

    Empty matrices return empty arrays without raising. Asking for more
    clusters than there are rows is clamped down to ``n_rows`` (sklearn
    raises in that case; clamping is friendlier to the script's CLI).
    """
    # Lazy import so the file is importable in environments without
    # scikit-learn (e.g. lightweight CI lint jobs).
    from sklearn.cluster import KMeans

    n = matrix.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=int), np.zeros((0, matrix.shape[1]), dtype=float)
    effective_k = max(1, min(k, n))
    km = KMeans(
        n_clusters=effective_k,
        n_init=n_init,
        random_state=seed,
    )
    labels = km.fit_predict(matrix)
    return labels.astype(int), km.cluster_centers_.astype(float)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _summarise_centroid(centroid: np.ndarray) -> str:
    """Concise one-liner describing a centroid's shape statistics."""
    if centroid.size == 0:
        return "empty"
    return (
        f"mean={float(centroid.mean()):+0.3f} "
        f"std={float(centroid.std()):0.3f} "
        f"min={float(centroid.min()):+0.3f} "
        f"max={float(centroid.max()):+0.3f}"
    )


def assemble_report(
    factor_ids: list[str],
    labels: np.ndarray,
    centroids: np.ndarray,
    *,
    k: int,
    window: int,
    seed: int,
    factor_count: int,
    skipped: list[str],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Bundle the KMeans output into the JSON-serialisable report shape."""
    n_centroids = int(centroids.shape[0])
    cluster_members: list[list[str]] = [[] for _ in range(n_centroids)]
    for fid, label in zip(factor_ids, labels.tolist(), strict=True):
        if 0 <= label < n_centroids:
            cluster_members[label].append(fid)

    clusters: list[ClusterRow] = []
    for cid in range(n_centroids):
        members = sorted(cluster_members[cid])
        clusters.append(
            ClusterRow(
                id=cid,
                size=len(members),
                factors=members,
                centroid_summary=_summarise_centroid(centroids[cid]),
            )
        )

    return {
        "generated_at": generated_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "k": int(k),
        "factor_count": int(factor_count),
        "window": int(window),
        "seed": int(seed),
        "clusters": [c.to_dict() for c in clusters],
        "skipped": sorted(skipped),
    }


def print_summary(report: dict[str, Any], *, top_n: int = 5) -> None:
    """Echo a human-friendly cluster overview to stdout."""
    print()
    print(
        f"factor_count={report['factor_count']}  "
        f"k={report['k']}  window={report['window']}  seed={report['seed']}"
    )
    print(f"  clusters    : {len(report['clusters'])}")
    print(f"  skipped     : {len(report['skipped'])}")
    print(f"  generated_at: {report['generated_at']}")
    print()
    print("  cluster sizes:")
    for c in report["clusters"]:
        head = ", ".join(c["factors"][:top_n])
        suffix = f", … (+{c['size'] - top_n} more)" if c["size"] > top_n else ""
        print(f"    [{c['id']:>3}] size={c['size']:<4} {head}{suffix}")


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
) -> dict[str, list[float]]:
    """Fetch the last ``window`` log returns for each factor from cache/API.

    Wraps :func:`pfm.factors.fetch_factor_history_dispatch`, which already
    handles per-source dispatch + on-disk caching. Errors per factor are
    caught and yield "no history" (the caller will skip the factor with
    a warning).

    .. note::
        This is the slow path. With 1228 factors and a cold cache this
        can take minutes. The spec explicitly says **not** to run this
        end-to-end during development; use ``--fixture`` instead.
    """
    # Import lazily so the unit tests (which only touch math) don't pay the
    # cost of pulling in the full pfm package graph.
    from pfm.factors import FactorConfig, fetch_factor_history_dispatch, load_factors

    factors = load_factors(FACTORS_YML)
    out: dict[str, list[float]] = {}
    for fid in factor_ids:
        fc: FactorConfig | None = factors.get(fid)
        if fc is None:
            logger.warning("factor %s: not in factors.yml; skipping", fid)
            continue
        try:
            df = fetch_factor_history_dispatch(fc)
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
        # Log returns. Clip to avoid log(0) blowups on probability series.
        safe = np.clip(prices, 1e-6, None)
        log_rets = np.diff(np.log(safe))
        out[fid] = log_rets[-window:].tolist()
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "KMeans-cluster every factor in factors.yml by its rolling 30-day "
            "log-return series. Writes a JSON report to /tmp/factor-clusters.json."
        )
    )
    p.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of KMeans clusters (default: {DEFAULT_K}).",
    )
    p.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Trailing observations per factor (default: {DEFAULT_WINDOW}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    p.add_argument(
        "--n-init",
        type=int,
        default=DEFAULT_N_INIT,
        help=f"KMeans n_init (default: {DEFAULT_N_INIT}).",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output JSON path (default: {DEFAULT_OUT}).",
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
        default=str(FACTORS_YML),
        help="Override path to factors.yml (live mode only).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit live-mode fetch to the first N factor ids (sorted).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-cluster stdout summary.",
    )
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Execute clustering; return the desired process exit code."""
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
        factor_count = len(returns_by_factor)
    else:
        # Live mode — pull from cached fetchers via pfm.factors dispatcher.
        from pfm.factors import load_factors

        factors = load_factors(Path(args.factors_yml))
        factor_ids = sorted(factors.keys())
        if args.limit is not None:
            factor_ids = factor_ids[: args.limit]
        factor_count = len(factor_ids)
        returns_by_factor = fetch_live_returns(factor_ids, window=args.window)

    kept, matrix, skipped = build_feature_matrix(returns_by_factor, window=args.window)
    labels, centroids = run_kmeans(matrix, k=args.k, seed=args.seed, n_init=args.n_init)
    report = assemble_report(
        kept,
        labels,
        centroids,
        k=args.k,
        window=args.window,
        seed=args.seed,
        factor_count=factor_count,
        skipped=skipped,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    if not args.quiet:
        print_summary(report)
        print(f"  report written: {out_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
