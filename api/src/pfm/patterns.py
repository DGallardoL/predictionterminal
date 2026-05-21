"""Pattern-finder: cross-pair PnL correlation, day-of-week effects,
pre-resolution regime shifts, structural clustering.

These are *meta* analyses — they look at the *portfolio* of alpha pairs
and surface structural patterns that aren't visible in any single-pair
backtest:

1.  **Pair-PnL correlation**: are the OOS-validated pairs *independent*
    sources of alpha or are they correlated? Independence ⇒ √k portfolio
    Sharpe boost; correlation ⇒ much smaller diversification benefit.
2.  **Day-of-week effect**: many financial series exhibit DOW seasonality
    (Monday-effect, Friday-effect). For prediction-market spreads, news
    flow concentrates around weekday business hours; weekend bars are
    quieter. Detect via mean PnL per weekday with t-statistic.
3.  **Pre-resolution regime shift**: as a market approaches resolution,
    its dynamics change — spread vol typically explodes, mean-reversion
    half-life shortens. Compare "far" (>30d to resolution) vs "near"
    (≤30d) statistics.
4.  **Pair clustering**: k-means on (Sharpe, half_life, hit_rate, n_obs)
    surfaces natural groupings — e.g., "fast-revert / many-trade" vs
    "slow-revert / few-trade".

All non-parametric (no model assumptions on the underlying probability
process). References:
    Cross, F. (1973). "The Behavior of Stock Prices on Fridays and Mondays."
    Lakonishok, J. & Smidt, S. (1988). "Are Seasonal Anomalies Real?"
    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning* §4 (clustering).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ───────────────────── PnL correlation ───────────────────────────────


@dataclass(frozen=True)
class PnlCorrelationMatrix:
    """Pairwise correlation matrix + summary stats."""

    pair_labels: list[str]
    correlation_matrix: list[list[float]]
    mean_off_diagonal: float  # average |ρ| across distinct pairs
    max_off_diagonal: float  # maximum |ρ|
    most_correlated: tuple[str, str, float] | None
    diversification_ratio: float  # √k / sum_of_eigenvalues^0.5; 1.0 = perfect dive


def correlate_pair_pnls(
    pnls: dict[str, pd.Series],
) -> PnlCorrelationMatrix:
    """Compute pairwise correlation of per-bar PnL series across pairs."""
    if len(pnls) < 2:
        return PnlCorrelationMatrix(
            pair_labels=list(pnls.keys()),
            correlation_matrix=[[1.0]] if pnls else [],
            mean_off_diagonal=0.0,
            max_off_diagonal=0.0,
            most_correlated=None,
            diversification_ratio=1.0,
        )
    df = pd.DataFrame(pnls).dropna()
    if len(df) < 10:
        return PnlCorrelationMatrix(
            pair_labels=list(pnls.keys()),
            correlation_matrix=df.corr().values.tolist() if not df.empty else [],
            mean_off_diagonal=0.0,
            max_off_diagonal=0.0,
            most_correlated=None,
            diversification_ratio=1.0,
        )
    corr = df.corr()
    n = corr.shape[0]
    # Off-diagonal stats
    mask = ~np.eye(n, dtype=bool)
    off = corr.values[mask]
    abs_off = np.abs(off)
    mean_abs = float(abs_off.mean()) if len(abs_off) else 0.0
    max_abs = float(abs_off.max()) if len(abs_off) else 0.0
    # Most-correlated pair
    most: tuple[str, str, float] | None = None
    if len(abs_off):
        i_max, j_max = np.unravel_index(
            np.argmax(np.where(mask, np.abs(corr.values), -np.inf)), corr.shape
        )
        most = (corr.index[i_max], corr.columns[j_max], float(corr.values[i_max, j_max]))
    # Diversification ratio via the participation-ratio of eigenvalues:
    # eff_n = (Σ λ_i)² / Σ λ_i² ∈ [1, n]; ratio = (eff_n − 1) / (n − 1).
    # 1 ⇒ fully independent (eff_n = n), 0 ⇒ perfectly correlated (eff_n = 1).
    eigvals = np.linalg.eigvalsh(corr.values)
    eigvals = np.clip(eigvals, 0.0, None)
    sum_l = eigvals.sum()
    sum_l2 = (eigvals**2).sum()
    if sum_l2 > 0 and n > 1:
        eff_n = (sum_l**2) / sum_l2
        div_ratio = float((eff_n - 1.0) / (n - 1.0))
    else:
        div_ratio = 1.0 if n == 1 else 0.0

    return PnlCorrelationMatrix(
        pair_labels=list(corr.index),
        correlation_matrix=corr.values.tolist(),
        mean_off_diagonal=mean_abs,
        max_off_diagonal=max_abs,
        most_correlated=most,
        diversification_ratio=div_ratio,
    )


# ───────────────────── day-of-week effect ────────────────────────────


@dataclass(frozen=True)
class DayOfWeekEffect:
    """Mean PnL per weekday + significance."""

    means: dict[str, float]
    counts: dict[str, int]
    t_stats: dict[str, float]
    p_values: dict[str, float]
    best_day: tuple[str, float] | None
    worst_day: tuple[str, float] | None
    significant_days: list[str]  # |t| > 1.96


_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def day_of_week_effect(pnl: pd.Series) -> DayOfWeekEffect:
    """Test for day-of-week seasonality in a per-bar PnL series.

    Per weekday d: t = (μ_d − μ_global) / (σ_d / √n_d). Compare to a
    Student-t distribution. Multiple-testing correction is the user's
    responsibility (Bonferroni at α=0.05 ⇒ |t| > ~2.7).
    """
    s = pnl.dropna()
    if len(s) < 14:
        return DayOfWeekEffect(
            means={},
            counts={},
            t_stats={},
            p_values={},
            best_day=None,
            worst_day=None,
            significant_days=[],
        )
    by_dow = s.groupby(s.index.dayofweek)
    global_mean = float(s.mean())

    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    t_stats: dict[str, float] = {}
    p_values: dict[str, float] = {}
    significant: list[str] = []
    for dow_int, group in by_dow:
        if dow_int >= 7:
            continue
        label = _WEEKDAYS[dow_int]
        n_d = len(group)
        if n_d < 3:
            means[label] = float(group.mean())
            counts[label] = n_d
            t_stats[label] = 0.0
            p_values[label] = 1.0
            continue
        mu_d = float(group.mean())
        sd_d = float(group.std(ddof=1))
        # One-sample t against global mean.
        se = sd_d / np.sqrt(n_d) if sd_d > 0 else 1e-12
        t_val = (mu_d - global_mean) / se
        p_val = float(2.0 * (1.0 - t_dist.cdf(abs(t_val), df=n_d - 1)))
        means[label] = mu_d
        counts[label] = n_d
        t_stats[label] = float(t_val)
        p_values[label] = p_val
        if abs(t_val) > 1.96:
            significant.append(label)

    if means:
        best = max(means.items(), key=lambda kv: kv[1])
        worst = min(means.items(), key=lambda kv: kv[1])
    else:
        best = worst = None
    return DayOfWeekEffect(
        means=means,
        counts=counts,
        t_stats=t_stats,
        p_values=p_values,
        best_day=best,
        worst_day=worst,
        significant_days=significant,
    )


# ─────────────── pre-resolution regime shift ─────────────────────────


@dataclass(frozen=True)
class PreResolutionRegime:
    """Comparison of "far" vs "near-resolution" spread statistics."""

    far_n: int
    near_n: int
    far_mean: float
    near_mean: float
    far_std: float
    near_std: float
    vol_ratio: float  # near_std / far_std
    mean_shift: float  # near_mean − far_mean
    vol_shift_significant: bool  # F-test p < 0.05
    f_stat: float
    f_p_value: float


def pre_resolution_regime(
    spread: pd.Series,
    *,
    days_to_resolution: int = 30,
) -> PreResolutionRegime:
    """Split the spread into "far" (early bars) and "near" (last
    ``days_to_resolution`` bars) windows. Report vol and mean shifts.

    Many prediction-market spreads exhibit *vol explosion* in the last
    weeks before resolution — a sign the cointegration is breaking down
    as the underlying event resolves. Don't trade pairs in their
    "near-resolution" regime without re-validating cointegration on a
    fresh window.
    """
    s = spread.dropna()
    n = len(s)
    if n < 2 * days_to_resolution:
        return PreResolutionRegime(
            far_n=0,
            near_n=0,
            far_mean=float("nan"),
            near_mean=float("nan"),
            far_std=float("nan"),
            near_std=float("nan"),
            vol_ratio=float("nan"),
            mean_shift=float("nan"),
            vol_shift_significant=False,
            f_stat=float("nan"),
            f_p_value=float("nan"),
        )
    near = s.iloc[-days_to_resolution:]
    far = s.iloc[:-days_to_resolution]
    far_std = float(far.std(ddof=1))
    near_std = float(near.std(ddof=1))
    far_mean = float(far.mean())
    near_mean = float(near.mean())
    vol_ratio = near_std / far_std if far_std > 0 else float("nan")

    # F-test for variance equality. F = larger_var / smaller_var.
    if far_std > 0 and near_std > 0:
        var_f = far_std**2
        var_n = near_std**2
        if var_n >= var_f:
            f_stat = var_n / var_f
            df1 = len(near) - 1
            df2 = len(far) - 1
        else:
            f_stat = var_f / var_n
            df1 = len(far) - 1
            df2 = len(near) - 1
        # Two-sided p
        from scipy.stats import f as f_dist

        p_value = float(
            2.0
            * min(
                1.0 - f_dist.cdf(f_stat, df1, df2),
                f_dist.cdf(f_stat, df1, df2),
            )
        )
        f_stat = float(f_stat)
    else:
        f_stat = float("nan")
        p_value = float("nan")

    return PreResolutionRegime(
        far_n=len(far),
        near_n=len(near),
        far_mean=far_mean,
        near_mean=near_mean,
        far_std=far_std,
        near_std=near_std,
        vol_ratio=vol_ratio,
        mean_shift=near_mean - far_mean,
        vol_shift_significant=(not np.isnan(p_value) and p_value < 0.05),
        f_stat=f_stat,
        f_p_value=p_value,
    )


# ──────────────────── pair-signature clustering ──────────────────────


@dataclass(frozen=True)
class PairCluster:
    """One cluster of pairs sharing structural signatures."""

    cluster_id: int
    pair_labels: list[str]
    centroid: dict[str, float]
    n_members: int


@dataclass(frozen=True)
class PairClusteringResult:
    n_clusters: int
    clusters: list[PairCluster]
    silhouette_proxy: float  # 1 - within/between ratio (rough)


def cluster_pairs_by_signature(
    signatures: dict[str, dict[str, float]],
    *,
    n_clusters: int = 3,
    seed: int = 42,
) -> PairClusteringResult:
    """K-means on standardised pair signatures.

    ``signatures`` maps pair_label → {feature_name: value}. Suggested
    features: sharpe, half_life_days, hit_rate, n_obs, max_drawdown.
    Returns clusters with centroids in the original-units feature space.
    """
    pairs = list(signatures.keys())
    if len(pairs) < n_clusters:
        return PairClusteringResult(n_clusters=0, clusters=[], silhouette_proxy=0.0)
    # Build matrix
    feature_names = sorted({k for v in signatures.values() for k in v})
    X = np.array([[signatures[p].get(f, 0.0) for f in feature_names] for p in pairs])
    # Drop rows with any NaN
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]
    pairs_clean = [p for p, m in zip(pairs, mask, strict=True) if m]
    if len(pairs_clean) < n_clusters:
        return PairClusteringResult(n_clusters=0, clusters=[], silhouette_proxy=0.0)
    X_std = StandardScaler().fit_transform(X_clean)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=5).fit(X_std)
    labels = km.labels_
    # Reverse-standardise centroids back to original units.
    sd = np.std(X_clean, axis=0, ddof=1)
    mu = np.mean(X_clean, axis=0)
    centroids_orig = km.cluster_centers_ * sd + mu

    clusters: list[PairCluster] = []
    for k in range(n_clusters):
        members = [pairs_clean[i] for i in range(len(pairs_clean)) if labels[i] == k]
        centroid_dict = {
            feature_names[j]: float(centroids_orig[k, j]) for j in range(len(feature_names))
        }
        clusters.append(
            PairCluster(
                cluster_id=k,
                pair_labels=members,
                centroid=centroid_dict,
                n_members=len(members),
            )
        )
    # Rough silhouette proxy: 1 - (within-cluster / between-cluster) variance.
    # A higher value means tighter clusters.
    within = float(
        np.sum(
            [np.sum((X_std[labels == k] - km.cluster_centers_[k]) ** 2) for k in range(n_clusters)]
        )
    )
    grand_centroid = X_std.mean(axis=0)
    between = (
        float(
            np.sum(
                km.n_features_in_
                * len(X_std[labels == k])
                * ((km.cluster_centers_[k] - grand_centroid) ** 2).sum()
                for k in range(n_clusters)
            )
        )
        if False
        else float(
            np.sum(
                [
                    len(X_std[labels == k]) * ((km.cluster_centers_[k] - grand_centroid) ** 2).sum()
                    for k in range(n_clusters)
                ]
            )
        )
    )
    silhouette = 1.0 - (within / (within + between)) if (within + between) > 0 else 0.0

    return PairClusteringResult(
        n_clusters=n_clusters,
        clusters=clusters,
        silhouette_proxy=float(silhouette),
    )


__all__ = [
    "DayOfWeekEffect",
    "PairCluster",
    "PairClusteringResult",
    "PnlCorrelationMatrix",
    "PreResolutionRegime",
    "cluster_pairs_by_signature",
    "correlate_pair_pnls",
    "day_of_week_effect",
    "pre_resolution_regime",
]
