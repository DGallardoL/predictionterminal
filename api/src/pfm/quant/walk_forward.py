"""Walk-forward optimization / backtesting framework.

A common backtesting pattern: fit a model on an in-sample window, generate
out-of-sample (OOS) predictions on the next slice, then roll forward.
Two modes are supported:

* **Rolling** (default) — train window is a fixed-size sliding window;
  fold ``k+1``'s train start = fold ``k``'s train start + ``step``.
* **Expanding** — train window starts at index 0 and grows by ``step``
  each fold; test window slides forward.

This module purposefully knows nothing about returns, Sharpe, or
prediction-market mechanics: it accepts a generic ``fit_fn`` / ``predict_fn``
pair and returns concatenated OOS predictions plus per-fold metrics. The
caller is responsible for choosing an appropriate model.

References
----------
- Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*,
  2nd ed., Wiley. (Chapter 11 on walk-forward analysis.)
- Bailey, D. H., et al. (2014). *Pseudo-Mathematics and Financial
  Charlatanism: The Effects of Backtest Overfitting on OOS Performance.*
  Notices of the AMS, 61(5).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["WalkForwardResult", "walk_forward"]


@dataclass(frozen=True)
class WalkForwardResult:
    """Result bundle for :func:`walk_forward`.

    Attributes
    ----------
    predictions:
        Concatenated out-of-sample predictions across every fold, indexed
        by the original ``y.index`` slice they correspond to. When folds
        overlap (because ``step < test_window``) the *last* prediction
        for any given index wins — but the default ``step = test_window``
        produces disjoint folds and therefore no overlap.
    fold_metrics:
        One dict per fold, with keys ``fold`` (int, 0-based), ``r2``,
        ``mse``, ``n_train``, ``n_test``, ``train_start``, ``train_end``,
        ``test_start``, ``test_end``. ``r2`` is ``np.nan`` when the test
        target has zero variance.
    summary:
        Overall metrics over the concatenated OOS predictions: ``r2``,
        ``mse``, ``ic`` (Spearman rank correlation between predictions
        and realised ``y``; the information coefficient).
    folds:
        Number of completed folds (length of ``fold_metrics``).
    """

    predictions: pd.Series
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    folds: int = 0


def _spearman_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank correlation; returns 0.0 when either side is constant."""
    if pred.size < 2 or actual.size < 2:
        return 0.0
    pr = pd.Series(pred).rank(method="average").to_numpy()
    ar = pd.Series(actual).rank(method="average").to_numpy()
    pr_std = pr.std()
    ar_std = ar.std()
    if pr_std == 0.0 or ar_std == 0.0:
        return 0.0
    return float(np.corrcoef(pr, ar)[0, 1])


def _r2(pred: np.ndarray, actual: np.ndarray) -> float:
    """Coefficient of determination, returns NaN when actual has zero variance."""
    if actual.size == 0:
        return float("nan")
    ss_res = float(np.sum((actual - pred) ** 2))
    mean = float(actual.mean())
    ss_tot = float(np.sum((actual - mean) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _mse(pred: np.ndarray, actual: np.ndarray) -> float:
    if actual.size == 0:
        return float("nan")
    return float(np.mean((actual - pred) ** 2))


def walk_forward(
    fit_fn: Callable[[pd.DataFrame, pd.Series], Any],
    predict_fn: Callable[[Any, pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_window: int = 252,
    test_window: int = 21,
    step: int | None = None,
    expanding: bool = False,
) -> WalkForwardResult:
    """Run a walk-forward backtest of ``fit_fn`` / ``predict_fn``.

    Parameters
    ----------
    fit_fn:
        Callable ``(X_train, y_train) -> model``. The returned ``model``
        is opaque to this function — it is forwarded straight to
        ``predict_fn``.
    predict_fn:
        Callable ``(model, X_test) -> np.ndarray`` of length
        ``len(X_test)``. Predictions must be coercible to ``float``.
    X:
        Feature matrix. Rows are observations; index is shared with ``y``.
    y:
        Target series; must align with ``X`` index and have the same length.
    train_window:
        Initial training window size (number of rows). In **rolling** mode
        this is the fixed train size on every fold; in **expanding** mode
        this is the train size of fold 0 only.
    test_window:
        Test window size (number of rows) per fold.
    step:
        Stride between consecutive folds. Defaults to ``test_window``
        (disjoint OOS slices). Must be ``>= 1``.
    expanding:
        If ``True``, the training window starts at index 0 and grows by
        ``step`` each fold. If ``False`` (default), training is a fixed
        sliding window.

    Returns
    -------
    WalkForwardResult

    Raises
    ------
    ValueError
        If ``len(X) != len(y)``, if ``X.index`` and ``y.index`` differ,
        if ``train_window``/``test_window``/``step`` are non-positive, or
        if no fold fits (``train_window + test_window > len(X)``).
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    if not X.index.equals(y.index):
        raise ValueError("X.index and y.index must be identical")
    if train_window < 1:
        raise ValueError(f"train_window must be >= 1, got {train_window}")
    if test_window < 1:
        raise ValueError(f"test_window must be >= 1, got {test_window}")
    if step is None:
        step = test_window
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    n = len(X)
    if train_window + test_window > n:
        raise ValueError(
            f"train_window ({train_window}) + test_window ({test_window}) "
            f"exceeds data length ({n}); cannot fit a single fold"
        )

    fold_metrics: list[dict[str, Any]] = []
    pred_index: list[Any] = []
    pred_values: list[float] = []

    fold = 0
    train_start = 0
    while True:
        if expanding:
            train_end = train_window + fold * step
            current_train_start = 0
        else:
            current_train_start = fold * step
            train_end = current_train_start + train_window

        test_start = train_end
        test_end = test_start + test_window
        if test_end > n:
            break

        X_train = X.iloc[current_train_start:train_end]
        y_train = y.iloc[current_train_start:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        model = fit_fn(X_train, y_train)
        raw = predict_fn(model, X_test)
        preds = np.asarray(raw, dtype=float).reshape(-1)
        if preds.shape[0] != len(X_test):
            raise ValueError(
                f"predict_fn returned {preds.shape[0]} values for "
                f"X_test of length {len(X_test)} (fold {fold})"
            )

        actual = y_test.to_numpy(dtype=float)
        fold_metrics.append(
            {
                "fold": fold,
                "r2": _r2(preds, actual),
                "mse": _mse(preds, actual),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "train_start": int(current_train_start),
                "train_end": int(train_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
            }
        )
        for idx, val in zip(X_test.index, preds, strict=False):
            pred_index.append(idx)
            pred_values.append(float(val))
        fold += 1
        train_start += step

    if not fold_metrics:
        # Shouldn't be reachable thanks to the upfront check, but kept as
        # a defensive guard against future refactors.
        raise ValueError("no folds were produced")

    # Drop duplicates keeping last (most recent overlapping prediction).
    predictions = pd.Series(pred_values, index=pred_index, name="prediction")
    if predictions.index.has_duplicates:
        predictions = predictions[~predictions.index.duplicated(keep="last")]
    predictions = predictions.sort_index()

    # Align actuals to the prediction index for overall metrics.
    actual_overall = y.reindex(predictions.index).to_numpy(dtype=float)
    pred_overall = predictions.to_numpy(dtype=float)
    summary = {
        "r2": _r2(pred_overall, actual_overall),
        "mse": _mse(pred_overall, actual_overall),
        "ic": _spearman_ic(pred_overall, actual_overall),
    }

    return WalkForwardResult(
        predictions=predictions,
        fold_metrics=fold_metrics,
        summary=summary,
        folds=len(fold_metrics),
    )
