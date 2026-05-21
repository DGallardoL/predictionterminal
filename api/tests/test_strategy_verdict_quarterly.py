"""Tests for the 4-quarter Sharpe stability enforcer."""

from __future__ import annotations

import math

import pytest

from pfm.strategy_verdict import quarterly_stability_test


class TestQuarterlyStability:
    def test_empty_list_is_tentative(self) -> None:
        out = quarterly_stability_test([])
        assert out["n_quarters"] == 0
        assert out["n_positive"] == 0
        assert out["sign_flips"] == 0
        assert out["passes_4q_gold"] is False
        assert out["passes_4q_silver"] is False
        assert out["tier_recommendation"] == "C_TENTATIVE"

    def test_single_quarter_cannot_promote(self) -> None:
        out = quarterly_stability_test([2.5])
        assert out["n_quarters"] == 1
        assert out["n_positive"] == 1
        assert out["passes_4q_gold"] is False  # < 4 quarters
        assert out["passes_4q_silver"] is False  # < 4 quarters
        assert out["tier_recommendation"] == "C_TENTATIVE"

    def test_four_strong_positive_quarters_gold(self) -> None:
        out = quarterly_stability_test([1.2, 0.9, 1.5, 0.8], threshold=0.5)
        assert out["n_quarters"] == 4
        assert out["n_positive"] == 4
        assert out["sign_flips"] == 0
        assert out["passes_4q_gold"] is True
        assert out["passes_4q_silver"] is True
        assert out["tier_recommendation"] == "A_GOLD"

    def test_three_positive_one_below_threshold_silver(self) -> None:
        # Below threshold but above zero → no sign flip, but n_positive=3 → silver.
        out = quarterly_stability_test([1.0, 0.3, 0.9, 1.1], threshold=0.5)
        assert out["n_positive"] == 3
        assert out["sign_flips"] == 0
        assert out["passes_4q_gold"] is False
        assert out["passes_4q_silver"] is True
        assert out["tier_recommendation"] == "B_VALIDATED"

    def test_sign_flip_demotes_from_gold(self) -> None:
        # Four "strong" quarters but a sign flip in the middle → no gold.
        out = quarterly_stability_test([1.0, -0.6, 1.2, 0.9], threshold=0.5)
        # 3 quarters above threshold (-0.6 fails), so silver only.
        assert out["sign_flips"] == 2
        assert out["passes_4q_gold"] is False

    def test_all_negative(self) -> None:
        out = quarterly_stability_test([-1.0, -0.5, -1.2, -0.8], threshold=0.5)
        assert out["n_positive"] == 0
        assert out["sign_flips"] == 0  # all same sign
        assert out["passes_4q_gold"] is False
        assert out["passes_4q_silver"] is False
        assert out["tier_recommendation"] == "C_TENTATIVE"

    def test_alternating_signs_max_flips(self) -> None:
        out = quarterly_stability_test([1.0, -1.0, 1.0, -1.0], threshold=0.5)
        assert out["sign_flips"] == 3
        assert out["passes_4q_gold"] is False

    def test_threshold_inclusive_boundary(self) -> None:
        # Exactly at threshold should NOT count (strict >).
        out = quarterly_stability_test([0.5, 0.5, 0.5, 0.5], threshold=0.5)
        assert out["n_positive"] == 0
        assert out["tier_recommendation"] == "C_TENTATIVE"

    def test_nan_quarters_handled(self) -> None:
        # NaN should not count as positive and should not introduce sign flips.
        out = quarterly_stability_test([1.0, float("nan"), 1.2, 0.9, 0.8], threshold=0.5)
        assert out["n_quarters"] == 5
        assert out["n_positive"] == 4
        assert out["sign_flips"] == 0

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            quarterly_stability_test([1.0, 1.0, 1.0, 1.0], threshold=-0.1)

    def test_five_quarters_one_negative_silver(self) -> None:
        out = quarterly_stability_test([1.5, 1.2, -0.3, 0.9, 1.1], threshold=0.5)
        # 4 quarters above threshold but with a sign flip — silver, not gold.
        assert out["n_positive"] == 4
        assert out["sign_flips"] == 2
        assert out["passes_4q_gold"] is False
        assert out["passes_4q_silver"] is True
        assert out["tier_recommendation"] == "B_VALIDATED"

    def test_zero_quarters_against_zero_threshold(self) -> None:
        # threshold=0.0 → only strictly positive quarters count.
        out = quarterly_stability_test([0.0, 0.0, 0.0, 0.0], threshold=0.0)
        assert out["n_positive"] == 0

    def test_non_numeric_treated_as_nan(self) -> None:
        out = quarterly_stability_test([1.0, "oops", 1.2, 0.8])  # type: ignore[list-item]
        assert out["n_quarters"] == 4
        # Three numerics are above the default threshold of 0.5.
        assert out["n_positive"] == 3
        # The NaN entry breaks sign-flip detection across the gap → still 0.
        assert out["sign_flips"] == 0


class TestQuarterlyStabilityShape:
    def test_keys_present(self) -> None:
        out = quarterly_stability_test([1.0, 1.0, 1.0, 1.0])
        for k in (
            "n_quarters",
            "n_positive",
            "sign_flips",
            "passes_4q_gold",
            "passes_4q_silver",
            "tier_recommendation",
        ):
            assert k in out

    def test_tier_string_in_allowed_set(self) -> None:
        out = quarterly_stability_test([1.0, 1.0, 1.0, 1.0])
        assert out["tier_recommendation"] in {"A_GOLD", "B_VALIDATED", "C_TENTATIVE"}

    def test_no_nan_leak_in_n_positive(self) -> None:
        out = quarterly_stability_test([float("nan")] * 4)
        assert out["n_positive"] == 0
        assert not math.isnan(out["sign_flips"])
