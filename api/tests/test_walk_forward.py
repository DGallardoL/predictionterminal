"""Tests for :mod:`pfm.quant.walk_forward`.

Covers:
- Synthetic-DGP recovery (known signal + noise -> IC > 0.3)
- Null signal -> IC near zero
- Edge cases (train_window too large -> ValueError, 1 fold, step parameter)
- Expanding-mode train-size monotonicity
- Reproducibility under a fixed seed
- Predictions concatenation, fold metric structure, and result wiring
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from pfm.quant.walk_forward import WalkForwardResult, walk_forward

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _ols_fit(X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """Tiny in-test OLS with intercept; returns beta vector (k+1,)."""
    Xa = np.column_stack([np.ones(len(X_train)), X_train.to_numpy(dtype=float)])
    y = y_train.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    return beta


def _ols_predict(model: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    Xa = np.column_stack([np.ones(len(X_test)), X_test.to_numpy(dtype=float)])
    return Xa @ model


def _make_signal_dataset(
    n: int = 600,
    *,
    seed: int = 7,
    snr: float = 1.0,
    k: int = 2,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Generate y = X @ beta + noise with controlled SNR."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n, k))
    beta = np.array([0.8, -0.4][:k])
    if k > 2:
        beta = np.concatenate([beta, rng.standard_normal(k - 2)])
    signal = X @ beta
    noise = rng.standard_normal(n) / max(snr, 1e-9)
    y = signal + noise
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return (
        pd.DataFrame(X, index=idx, columns=[f"f{i}" for i in range(k)]),
        pd.Series(y, index=idx, name="y"),
        beta,
    )


# ---------------------------------------------------------------------------
# 1. Result type + happy path
# ---------------------------------------------------------------------------


def test_result_type_and_structure():
    X, y, _ = _make_signal_dataset(n=400, seed=1)
    res = walk_forward(
        _ols_fit,
        _ols_predict,
        X,
        y,
        train_window=200,
        test_window=20,
    )
    assert isinstance(res, WalkForwardResult)
    assert isinstance(res.predictions, pd.Series)
    assert isinstance(res.fold_metrics, list)
    assert res.folds == len(res.fold_metrics)
    # Result is frozen — attribute assignment should fail.
    with pytest.raises(Exception):  # FrozenInstanceError subclasses Exception
        res.folds = 99  # type: ignore[misc]


def test_known_signal_recovered_ic_positive():
    X, y, _ = _make_signal_dataset(n=600, seed=11, snr=1.5)
    res = walk_forward(_ols_fit, _ols_predict, X, y, train_window=252, test_window=21)
    assert res.summary["ic"] > 0.3
    assert res.summary["r2"] > 0.2


def test_null_signal_ic_near_zero():
    rng = np.random.default_rng(99)
    n = 600
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    X = pd.DataFrame(rng.standard_normal(size=(n, 2)), index=idx, columns=["a", "b"])
    y = pd.Series(rng.standard_normal(n), index=idx, name="y")
    res = walk_forward(_ols_fit, _ols_predict, X, y, train_window=252, test_window=21)
    assert abs(res.summary["ic"]) < 0.15  # well within sampling noise


# ---------------------------------------------------------------------------
# 2. Edge cases
# ---------------------------------------------------------------------------


def test_train_window_too_large_raises():
    X, y, _ = _make_signal_dataset(n=100, seed=2)
    with pytest.raises(ValueError, match="exceeds data length"):
        walk_forward(_ols_fit, _ols_predict, X, y, train_window=200, test_window=10)


def test_train_window_plus_test_window_exact_fit_one_fold():
    X, y, _ = _make_signal_dataset(n=110, seed=3)
    res = walk_forward(_ols_fit, _ols_predict, X, y, train_window=100, test_window=10)
    assert res.folds == 1
    fm = res.fold_metrics[0]
    assert fm["n_train"] == 100
    assert fm["n_test"] == 10
    assert len(res.predictions) == 10


def test_nonpositive_window_args_raise():
    X, y, _ = _make_signal_dataset(n=100, seed=4)
    with pytest.raises(ValueError, match="train_window must be"):
        walk_forward(_ols_fit, _ols_predict, X, y, train_window=0, test_window=10)
    with pytest.raises(ValueError, match="test_window must be"):
        walk_forward(_ols_fit, _ols_predict, X, y, train_window=50, test_window=0)
    with pytest.raises(ValueError, match="step must be"):
        walk_forward(_ols_fit, _ols_predict, X, y, train_window=50, test_window=10, step=0)


def test_xy_length_mismatch_raises():
    X, y, _ = _make_signal_dataset(n=100, seed=5)
    with pytest.raises(ValueError, match="length mismatch"):
        walk_forward(_ols_fit, _ols_predict, X.iloc[:-1], y, train_window=50, test_window=10)


def test_xy_index_mismatch_raises():
    X, y, _ = _make_signal_dataset(n=100, seed=6)
    y2 = y.copy()
    y2.index = pd.RangeIndex(start=1000, stop=1000 + len(y2))
    with pytest.raises(ValueError, match="index"):
        walk_forward(_ols_fit, _ols_predict, X, y2, train_window=50, test_window=10)


def test_wrong_types_raise():
    X, y, _ = _make_signal_dataset(n=100, seed=7)
    with pytest.raises(TypeError):
        walk_forward(_ols_fit, _ols_predict, X.to_numpy(), y, train_window=50, test_window=10)
    with pytest.raises(TypeError):
        walk_forward(_ols_fit, _ols_predict, X, y.to_numpy(), train_window=50, test_window=10)


# ---------------------------------------------------------------------------
# 3. Step parameter
# ---------------------------------------------------------------------------


def test_step_defaults_to_test_window_disjoint_folds():
    X, y, _ = _make_signal_dataset(n=500, seed=8)
    res = walk_forward(_ols_fit, _ols_predict, X, y, train_window=200, test_window=20)
    # With default step = test_window, OOS slices should be disjoint and
    # total prediction count should be folds * test_window.
    assert len(res.predictions) == res.folds * 20
    # Test windows back-to-back: fold k test_start = previous test_end.
    for prev, curr in zip(res.fold_metrics[:-1], res.fold_metrics[1:], strict=False):
        assert curr["test_start"] == prev["test_end"]


def test_step_smaller_than_test_window_overlaps_dedupe_keeps_last():
    X, y, _ = _make_signal_dataset(n=400, seed=9)
    res = walk_forward(_ols_fit, _ols_predict, X, y, train_window=200, test_window=40, step=20)
    # With overlap, dedup-keep-last yields strictly fewer rows than
    # sum(fold n_test).
    raw = sum(fm["n_test"] for fm in res.fold_metrics)
    assert len(res.predictions) < raw
    # Predictions index must be unique post-dedupe.
    assert res.predictions.index.is_unique


def test_step_larger_than_test_window_leaves_gaps():
    X, y, _ = _make_signal_dataset(n=500, seed=10)
    res = walk_forward(_ols_fit, _ols_predict, X, y, train_window=200, test_window=10, step=30)
    # Each fold contributes exactly test_window predictions; gaps mean
    # the total of test_starts increments by step (=30) each fold.
    for prev, curr in zip(res.fold_metrics[:-1], res.fold_metrics[1:], strict=False):
        assert curr["test_start"] - prev["test_start"] == 30


# ---------------------------------------------------------------------------
# 4. Expanding mode
# ---------------------------------------------------------------------------


def test_expanding_mode_train_size_monotonic_increasing():
    X, y, _ = _make_signal_dataset(n=600, seed=12)
    res = walk_forward(
        _ols_fit,
        _ols_predict,
        X,
        y,
        train_window=200,
        test_window=20,
        expanding=True,
    )
    sizes = [fm["n_train"] for fm in res.fold_metrics]
    assert sizes[0] == 200
    # Each subsequent fold has strictly more training data.
    assert all(a < b for a, b in itertools.pairwise(sizes))
    # Train always starts at 0 in expanding mode.
    assert all(fm["train_start"] == 0 for fm in res.fold_metrics)


def test_rolling_vs_expanding_differ_on_signal():
    X, y, _ = _make_signal_dataset(n=600, seed=13, snr=1.2)
    r_roll = walk_forward(
        _ols_fit, _ols_predict, X, y, train_window=200, test_window=20, expanding=False
    )
    r_exp = walk_forward(
        _ols_fit, _ols_predict, X, y, train_window=200, test_window=20, expanding=True
    )
    # Both should recover signal but produce different predictions in
    # later folds because the training set differs.
    assert r_roll.summary["ic"] > 0.2
    assert r_exp.summary["ic"] > 0.2
    # Rolling vs expanding share fold 0 (same train slice). From fold 1
    # onwards the OOS predictions should diverge.
    last_idx = r_roll.fold_metrics[-1]["test_end"]
    common = r_roll.predictions.index.intersection(r_exp.predictions.index)
    assert len(common) > 0  # sanity
    # Predictions on the SECOND fold onwards should differ at least somewhere.
    second_fold_start = r_roll.fold_metrics[1]["test_start"]
    tail_roll = r_roll.predictions.iloc[second_fold_start - r_roll.fold_metrics[0]["test_start"] :]
    tail_exp = r_exp.predictions.reindex(tail_roll.index)
    assert not np.allclose(tail_roll.to_numpy(), tail_exp.to_numpy(), atol=1e-12)
    assert last_idx <= len(X)


# ---------------------------------------------------------------------------
# 5. Fold metric structure
# ---------------------------------------------------------------------------


def test_fold_metric_keys_present():
    X, y, _ = _make_signal_dataset(n=400, seed=14)
    res = walk_forward(_ols_fit, _ols_predict, X, y, train_window=200, test_window=20)
    expected = {
        "fold",
        "r2",
        "mse",
        "n_train",
        "n_test",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    }
    for fm in res.fold_metrics:
        assert expected.issubset(fm.keys())
    # Folds are numbered 0..N-1 in order.
    assert [fm["fold"] for fm in res.fold_metrics] == list(range(res.folds))


def test_summary_keys_present_and_finite():
    X, y, _ = _make_signal_dataset(n=500, seed=15)
    res = walk_forward(_ols_fit, _ols_predict, X, y, train_window=200, test_window=20)
    for k in ("r2", "mse", "ic"):
        assert k in res.summary
        assert np.isfinite(res.summary[k])


def test_predict_fn_wrong_length_raises():
    X, y, _ = _make_signal_dataset(n=200, seed=16)

    def bad_predict(model, X_test):
        return np.zeros(len(X_test) - 1)

    with pytest.raises(ValueError, match="returned"):
        walk_forward(_ols_fit, bad_predict, X, y, train_window=100, test_window=20)


# ---------------------------------------------------------------------------
# 6. Reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility_same_inputs_same_outputs():
    X, y, _ = _make_signal_dataset(n=400, seed=17)
    a = walk_forward(_ols_fit, _ols_predict, X, y, train_window=200, test_window=20)
    b = walk_forward(_ols_fit, _ols_predict, X, y, train_window=200, test_window=20)
    pd.testing.assert_series_equal(a.predictions, b.predictions)
    assert a.fold_metrics == b.fold_metrics
    assert a.summary == b.summary
    assert a.folds == b.folds


# ---------------------------------------------------------------------------
# 7. Predictions alignment
# ---------------------------------------------------------------------------


def test_predictions_indexed_by_y_index_subset():
    X, y, _ = _make_signal_dataset(n=400, seed=18)
    res = walk_forward(_ols_fit, _ols_predict, X, y, train_window=200, test_window=20)
    # Every prediction index entry must be in y.index.
    assert res.predictions.index.isin(y.index).all()
    # Predictions index must be sorted.
    assert res.predictions.index.is_monotonic_increasing


def test_predict_fn_receives_correct_window_size():
    seen: list[int] = []

    def fit(X_train, y_train):
        return None

    def predict(model, X_test):
        seen.append(len(X_test))
        return np.zeros(len(X_test))

    X, y, _ = _make_signal_dataset(n=400, seed=19)
    walk_forward(fit, predict, X, y, train_window=150, test_window=25)
    assert seen and all(s == 25 for s in seen)
