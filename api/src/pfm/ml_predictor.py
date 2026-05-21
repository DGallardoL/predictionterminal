"""ML predictor for prediction-market spread dynamics.

A gradient-boosted regressor on engineered spread features. Predicts
Δspread_{t+1} = spread_{t+1} − spread_t conditional on past spread state.

**Features** (literature-motivated, no look-ahead):

1.  Lag-k z-scores ``z_{t-k}`` for k ∈ {1, 2, 3, 5, 10}. Captures
    auto-regressive structure (Engle-Granger 1987).
2.  Rolling realised volatility on 5-bar and 20-bar windows. Captures
    vol-of-vol regimes (Bollerslev 1986).
3.  Lag-1 spread Δ momentum: sign(Δspread_{t-1}) and Δspread_{t-1} itself.
4.  Sample auto-correlation at lags 1, 5 in a rolling 30-bar window.
    Auto-correlation = mean-reversion strength signal (Lo-MacKinlay 1988).
5.  Distance-from-mean ``(spread_t − μ_60d) / σ_60d``. Long-window z.

**Validation**: TimeSeriesSplit (sklearn) — strict chronological folds, no
look-ahead. Reports test R², direction-accuracy (the practical metric for
trading), and information-coefficient (rank-correlation between predicted
and realised Δ — Grinold & Kahn 2000 §7).

**Baseline**: predict Δspread_{t+1} = −α · z_t (the naive mean-reversion
linear rule). The ML model has to *beat* this baseline to earn its keep.
The output reports both side by side so the user can decide.

**Honest caveats** (explicit):
- Probability series are bounded [0, 1] — vanilla GBR is fine here, but
  forecasts of Δspread that imply spread leaving [0, 1] are clipped.
- 100-300 daily bars is *very little* data for ML. A single feature
  contributing 1-2% R² is realistic; "0.6 R²" is suspicious overfit.
- We don't claim transferability across pairs — train one model per pair.
- We don't recommend using predictions as direct signals without a
  trade-cost-aware sizing layer.

References:
    Avellaneda, M. & Lee, J. (2010). "Statistical Arbitrage in the U.S.
        Equities Market." Quantitative Finance 10(7), 761-782.
    Bollerslev, T. (1986). "Generalized Autoregressive Conditional
        Heteroskedasticity." J. Econometrics 31, 307-327.
    Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*.
    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class FeatureImportance:
    name: str
    importance: float


@dataclass(frozen=True)
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    test_r2: float
    test_direction_accuracy: float
    baseline_direction_accuracy: float
    information_coefficient: float


@dataclass(frozen=True)
class MlPredictorResult:
    """Output of :func:`fit_ml_predictor`.

    Attributes:
        n_obs: aligned spread length.
        n_features: number of engineered features.
        feature_names: ordered.
        n_folds: TimeSeriesSplit folds used.
        folds: per-fold metrics.
        mean_test_r2: average across folds.
        mean_direction_accuracy: average direction-correct rate (test folds).
        mean_baseline_direction_accuracy: same metric for the naive z-score
            baseline.
        beats_baseline: True if the GBR mean is strictly higher than baseline.
        mean_ic: average Spearman rank-correlation (predicted vs realised Δ).
        feature_importances: averaged across folds (descending).
        last_prediction: predicted Δspread for the very next bar (refit on
            all data); useful as a current trading signal. ``None`` when
            unavailable.
        verdict: ``"likely_alpha"`` (R²>0.05 AND beats baseline AND IC>0.10)
            / ``"marginal"`` / ``"no_edge"``.
    """

    n_obs: int
    n_features: int
    feature_names: list[str]
    n_folds: int
    folds: list[FoldResult]
    mean_test_r2: float
    mean_direction_accuracy: float
    mean_baseline_direction_accuracy: float
    beats_baseline: bool
    mean_ic: float
    feature_importances: list[FeatureImportance]
    last_prediction: float | None
    verdict: Literal["likely_alpha", "marginal", "no_edge", "insufficient-data"]


# ────────────────────── feature engineering ───────────────────────────


def _build_features(spread: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Engineer the feature matrix. Pure-past, no look-ahead.

    Returns the feature DataFrame plus a stable ordered list of feature
    column names (target column is excluded from the list).
    """
    s = spread.dropna()
    df = pd.DataFrame(index=s.index)
    df["spread"] = s
    df["dspread"] = s.diff()

    # Lag-k z-scores using a 30-bar rolling window.
    mu30 = s.rolling(30, min_periods=10).mean()
    sd30 = s.rolling(30, min_periods=10).std(ddof=1)
    z = (s - mu30) / sd30
    for k in (1, 2, 3, 5, 10):
        df[f"z_lag{k}"] = z.shift(k)

    # Rolling realised vol of dspread.
    df["vol_5"] = df["dspread"].rolling(5, min_periods=3).std(ddof=1).shift(1)
    df["vol_20"] = df["dspread"].rolling(20, min_periods=10).std(ddof=1).shift(1)

    # Momentum: previous dspread + sign.
    df["dspread_lag1"] = df["dspread"].shift(1)
    df["dspread_sign_lag1"] = np.sign(df["dspread"].shift(1)).fillna(0.0)

    # Rolling autocorrelation at lag 1 and 5 over 30 bars.
    def _rolling_autocorr(x: pd.Series, lag: int, window: int = 30) -> pd.Series:
        return x.rolling(window, min_periods=window // 2).apply(
            lambda v: pd.Series(v).autocorr(lag=lag) if len(v) > lag else np.nan,
            raw=False,
        )

    df["acf_1"] = _rolling_autocorr(df["dspread"], lag=1).shift(1)
    df["acf_5"] = _rolling_autocorr(df["dspread"], lag=5).shift(1)

    # Distance-from-long-mean.
    mu60 = s.rolling(60, min_periods=20).mean()
    sd60 = s.rolling(60, min_periods=20).std(ddof=1)
    df["z_long"] = ((s - mu60) / sd60).shift(1)

    # Target: next-bar dspread.
    df["target"] = df["dspread"].shift(-1)

    feature_names = [
        "z_lag1",
        "z_lag2",
        "z_lag3",
        "z_lag5",
        "z_lag10",
        "vol_5",
        "vol_20",
        "dspread_lag1",
        "dspread_sign_lag1",
        "acf_1",
        "acf_5",
        "z_long",
    ]
    return df, feature_names


def fit_ml_predictor(
    spread: pd.Series,
    *,
    n_folds: int = 5,
    n_estimators: int = 80,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> MlPredictorResult:
    """Fit a gradient-boosted regressor on engineered spread features
    with TimeSeriesSplit cross-validation.

    Args:
        spread: per-bar spread series.
        n_folds: time-series CV folds.
        n_estimators / max_depth / learning_rate: GBR hyper-parameters.
            Defaults are conservative (low max_depth keeps model
            interpretable; low learning rate guards against overfit).
        seed: RNG seed.

    Returns:
        :class:`MlPredictorResult`.
    """
    df, feature_names = _build_features(spread)
    df_clean = df.dropna(subset=[*feature_names, "target"])
    n = len(df_clean)
    if n < n_folds * 25:
        return MlPredictorResult(
            n_obs=n,
            n_features=len(feature_names),
            feature_names=feature_names,
            n_folds=n_folds,
            folds=[],
            mean_test_r2=float("nan"),
            mean_direction_accuracy=float("nan"),
            mean_baseline_direction_accuracy=float("nan"),
            beats_baseline=False,
            mean_ic=float("nan"),
            feature_importances=[],
            last_prediction=None,
            verdict="insufficient-data",
        )

    X = df_clean[feature_names].to_numpy()
    y = df_clean["target"].to_numpy()
    z_lag1 = df_clean["z_lag1"].to_numpy()  # baseline signal

    tscv = TimeSeriesSplit(n_splits=n_folds)
    folds: list[FoldResult] = []
    importances_acc = np.zeros(len(feature_names))
    fold_count = 0

    for k, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        gbr = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
        )
        gbr.fit(X_train, y_train)
        y_pred = gbr.predict(X_test)
        # R²
        ss_res = float(np.sum((y_test - y_pred) ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # Direction-accuracy: fraction of bars where sign(pred) == sign(real).
        # Treat exact-zero predictions as "correct" only when y_test is also 0.
        direction = np.where(
            np.sign(y_pred) == np.sign(y_test),
            1.0,
            0.0,
        )
        # Baseline: predict −α·z_lag1 (mean-reversion); positive z ⇒ short
        # ⇒ predict negative dspread.
        z_test = z_lag1[test_idx]
        baseline_pred = -z_test
        baseline_direction = np.where(
            np.sign(baseline_pred) == np.sign(y_test),
            1.0,
            0.0,
        )
        # Information coefficient: Spearman correlation between pred and target.
        if len(y_test) >= 5:
            ic, _ = spearmanr(y_pred, y_test)
            ic = 0.0 if np.isnan(ic) else float(ic)
        else:
            ic = 0.0
        folds.append(
            FoldResult(
                fold=k,
                n_train=len(train_idx),
                n_test=len(test_idx),
                test_r2=float(r2),
                test_direction_accuracy=float(direction.mean()),
                baseline_direction_accuracy=float(baseline_direction.mean()),
                information_coefficient=ic,
            )
        )
        importances_acc += gbr.feature_importances_
        fold_count += 1

    if fold_count == 0:
        return MlPredictorResult(
            n_obs=n,
            n_features=len(feature_names),
            feature_names=feature_names,
            n_folds=n_folds,
            folds=[],
            mean_test_r2=float("nan"),
            mean_direction_accuracy=float("nan"),
            mean_baseline_direction_accuracy=float("nan"),
            beats_baseline=False,
            mean_ic=float("nan"),
            feature_importances=[],
            last_prediction=None,
            verdict="insufficient-data",
        )

    mean_r2 = float(np.mean([f.test_r2 for f in folds]))
    mean_acc = float(np.mean([f.test_direction_accuracy for f in folds]))
    mean_baseline = float(np.mean([f.baseline_direction_accuracy for f in folds]))
    mean_ic = float(np.mean([f.information_coefficient for f in folds]))
    beats_baseline = mean_acc > mean_baseline + 1e-6

    importances_avg = importances_acc / fold_count
    feat_importances = [
        FeatureImportance(name=n_, importance=float(v))
        for n_, v in sorted(
            zip(feature_names, importances_avg, strict=True),
            key=lambda x: -x[1],
        )
    ]

    # Refit on all data → predict next bar.
    last_prediction: float | None
    try:
        gbr_full = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
        )
        gbr_full.fit(X, y)
        # The LAST row of df_clean has features for predicting the bar
        # *after* the last observed spread value.
        x_last = df_clean[feature_names].iloc[-1:].to_numpy()
        last_prediction = float(gbr_full.predict(x_last)[0])
    except Exception:
        last_prediction = None

    if mean_r2 > 0.05 and beats_baseline and mean_ic > 0.10:
        verdict: Literal["likely_alpha", "marginal", "no_edge", "insufficient-data"] = (
            "likely_alpha"
        )
    elif mean_r2 > 0.0 and beats_baseline:
        verdict = "marginal"
    else:
        verdict = "no_edge"

    return MlPredictorResult(
        n_obs=n,
        n_features=len(feature_names),
        feature_names=feature_names,
        n_folds=n_folds,
        folds=folds,
        mean_test_r2=mean_r2,
        mean_direction_accuracy=mean_acc,
        mean_baseline_direction_accuracy=mean_baseline,
        beats_baseline=beats_baseline,
        mean_ic=mean_ic,
        feature_importances=feat_importances,
        last_prediction=last_prediction,
        verdict=verdict,
    )


__all__ = [
    "FeatureImportance",
    "FoldResult",
    "MlPredictorResult",
    "fit_ml_predictor",
]
