"""Unit tests for the logit transform and ΔLogit utility."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pfm.model import delta_logit, logit_transform


class TestLogitTransform:
    def test_known_values(self) -> None:
        s = pd.Series([0.5, 0.25, 0.75])
        out = logit_transform(s)
        assert math.isclose(out.iloc[0], 0.0, abs_tol=1e-12)
        assert math.isclose(out.iloc[1], math.log(0.25 / 0.75), abs_tol=1e-12)
        assert math.isclose(out.iloc[2], math.log(0.75 / 0.25), abs_tol=1e-12)

    def test_clipping_at_zero(self) -> None:
        out = logit_transform(pd.Series([0.0]), epsilon=0.01)
        # logit(0.01) = log(0.01 / 0.99)
        assert math.isclose(out.iloc[0], math.log(0.01 / 0.99), abs_tol=1e-12)

    def test_clipping_at_one(self) -> None:
        out = logit_transform(pd.Series([1.0]), epsilon=0.01)
        assert math.isclose(out.iloc[0], math.log(0.99 / 0.01), abs_tol=1e-12)

    def test_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError):
            logit_transform(pd.Series([0.5]), epsilon=0.0)
        with pytest.raises(ValueError):
            logit_transform(pd.Series([0.5]), epsilon=0.5)

    def test_accepts_numpy_array(self) -> None:
        out = logit_transform(np.array([0.3, 0.7]))
        assert isinstance(out, pd.Series)
        assert len(out) == 2

    def test_preserves_index(self) -> None:
        idx = pd.date_range("2025-01-01", periods=3, freq="D")
        s = pd.Series([0.4, 0.5, 0.6], index=idx)
        out = logit_transform(s)
        assert (out.index == idx).all()


class TestDeltaLogit:
    def test_first_diff_length(self) -> None:
        s = pd.Series([0.4, 0.5, 0.6])
        out = delta_logit(s)
        assert len(out) == 3
        assert pd.isna(out.iloc[0])
        assert not pd.isna(out.iloc[1])

    def test_constant_series_zero_delta(self) -> None:
        s = pd.Series([0.5] * 5)
        out = delta_logit(s).dropna()
        assert (out == 0.0).all()

    def test_clipping_makes_delta_zero(self) -> None:
        # Both values fall below epsilon; both get clipped to epsilon ⇒ Δ=0.
        s = pd.Series([0.005, 0.002])
        out = delta_logit(s, epsilon=0.01).dropna()
        assert math.isclose(out.iloc[0], 0.0, abs_tol=1e-12)
