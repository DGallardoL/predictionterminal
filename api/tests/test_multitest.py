"""Tests for pfm.multitest — BH-FDR, Bonferroni, and alpha-card tagging."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from pfm.multitest import (
    apply_multitest_to_alphas,
    benjamini_hochberg_fdr,
    bonferroni_correction,
)

# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    def test_empty_list(self) -> None:
        out = benjamini_hochberg_fdr([], alpha=0.05)
        assert out == {"rejected_idx": [], "q_values": [], "n_significant": 0}

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be"):
            benjamini_hochberg_fdr([0.01, 0.02], alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be"):
            benjamini_hochberg_fdr([0.01, 0.02], alpha=1.5)

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid probability"):
            benjamini_hochberg_fdr([0.5, 1.5], alpha=0.05)
        with pytest.raises(ValueError, match="not a valid probability"):
            benjamini_hochberg_fdr([0.5, float("nan")], alpha=0.05)

    def test_q_values_align_to_input_order(self) -> None:
        p = [0.50, 0.001, 0.20, 0.005]
        out = benjamini_hochberg_fdr(p, alpha=0.05)
        assert len(out["q_values"]) == 4
        # The input minimum (index 1, p=0.001) must own the smallest q-value.
        argmin_q = int(np.argmin(out["q_values"]))
        assert argmin_q == 1
        # The largest p must own the largest q-value.
        argmax_q = int(np.argmax(out["q_values"]))
        assert argmax_q == 0

    def test_all_significant(self) -> None:
        p = [0.001, 0.002, 0.003, 0.004, 0.005]
        out = benjamini_hochberg_fdr(p, alpha=0.05)
        assert out["n_significant"] == 5
        assert sorted(out["rejected_idx"]) == [0, 1, 2, 3, 4]

    def test_none_significant(self) -> None:
        p = [0.40, 0.60, 0.55, 0.80, 0.90]
        out = benjamini_hochberg_fdr(p, alpha=0.05)
        assert out["n_significant"] == 0
        assert out["rejected_idx"] == []

    def test_synthetic_10_real_90_null_recovers_close_to_10(self) -> None:
        """Headline test: 10 real signals + 90 nulls (uniform 0-1).

        BH-FDR at α=0.05 should recover most of the real signals while keeping
        false positives well below the FDR target.

        Note: with m=100 the BH threshold at rank k is α·k/m, so the smallest
        rank-1 p needs to clear 5e-4 and rank-10 needs to clear 5e-3.  Strong
        real signals (p < 0.005) recover all 10; weaker signals get partly
        absorbed because real-vs-null tails overlap.  We use uniform(0, 0.005)
        to get a deterministic recovery.
        """
        rng = np.random.default_rng(42)
        real = rng.uniform(0.0, 0.005, size=10).tolist()
        null = rng.uniform(0.0, 1.0, size=90).tolist()
        p = real + null
        out = benjamini_hochberg_fdr(p, alpha=0.05)
        # Recover at least 8 of the 10 real signals deterministically.
        n_real_rejected = sum(1 for i in out["rejected_idx"] if i < 10)
        assert n_real_rejected >= 8
        # Total rejected should be roughly the number of true positives — definitely
        # not blowing past 20 (which would mean FDR >> 0.05).
        assert out["n_significant"] <= 20

    def test_synthetic_weaker_signals_partial_recovery(self) -> None:
        """Sanity: with weaker signals (uniform 0-0.05), BH still controls FDR.

        Recovery is patchy by design; the contract is that we don't drown in
        false positives.
        """
        rng = np.random.default_rng(42)
        real = rng.uniform(0.0, 0.05, size=10).tolist()
        null = rng.uniform(0.0, 1.0, size=90).tolist()
        p = real + null
        out = benjamini_hochberg_fdr(p, alpha=0.05)
        # With m=100, alpha=0.05, the BH cutoff is too strict for weak signals.
        # The point of this test is FDR control: false-positive count stays ≤ 1.
        false_positives = sum(1 for i in out["rejected_idx"] if i >= 10)
        assert false_positives <= 1

    def test_q_values_monotone_in_sorted_p(self) -> None:
        rng = np.random.default_rng(0)
        p = sorted(rng.uniform(0, 1, 50).tolist())
        out = benjamini_hochberg_fdr(p, alpha=0.05)
        # q-values are parallel to input order; here input is already sorted
        # so q-values must be non-decreasing.
        qs = out["q_values"]
        for a, b in pairwise(qs):
            assert a <= b + 1e-12

    def test_single_p_value(self) -> None:
        out = benjamini_hochberg_fdr([0.01], alpha=0.05)
        assert out["n_significant"] == 1
        assert out["rejected_idx"] == [0]
        assert out["q_values"][0] == pytest.approx(0.01)
        out2 = benjamini_hochberg_fdr([0.50], alpha=0.05)
        assert out2["n_significant"] == 0


# ---------------------------------------------------------------------------
# Bonferroni
# ---------------------------------------------------------------------------


class TestBonferroni:
    def test_empty(self) -> None:
        out = bonferroni_correction([], alpha=0.05)
        assert out["n_significant"] == 0

    def test_only_smallest_passes(self) -> None:
        # m=10, threshold = 0.005; only 0.001 passes.
        p = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        out = bonferroni_correction(p, alpha=0.05)
        assert out["n_significant"] == 1
        assert out["rejected_idx"] == [0]
        assert out["adjusted_p_values"][0] == pytest.approx(0.01)

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha must be"):
            bonferroni_correction([0.01], alpha=-0.1)


# ---------------------------------------------------------------------------
# apply_multitest_to_alphas
# ---------------------------------------------------------------------------


class TestApplyMultitestToAlphas:
    def test_tags_each_card(self) -> None:
        alphas = [
            {"name": "a1", "perm_p": 0.001},
            {"name": "a2", "perm_p": 0.04},
            {"name": "a3", "perm_p": 0.40},
        ]
        out = apply_multitest_to_alphas(alphas)
        assert len(out) == 3
        for card in out:
            assert "bh_q_value" in card
            assert "passes_bh_q05" in card
            assert "passes_bh_q10" in card
        # Original input not mutated.
        assert "bh_q_value" not in alphas[0]

    def test_missing_p_field_skipped(self) -> None:
        alphas = [
            {"name": "good", "perm_p": 0.001},
            {"name": "no_p"},
            {"name": "none_p", "perm_p": None},
            {"name": "bad_p", "perm_p": "not-a-number"},
            {"name": "out_of_range", "perm_p": 1.5},
        ]
        out = apply_multitest_to_alphas(alphas)
        assert out[0]["passes_bh_q05"] is True
        for i in (1, 2, 3, 4):
            assert out[i]["bh_q_value"] is None
            assert out[i]["passes_bh_q05"] is False
            assert out[i]["passes_bh_q10"] is False

    def test_custom_p_field(self) -> None:
        alphas = [
            {"name": "a1", "deflated_p": 0.001},
            {"name": "a2", "deflated_p": 0.50},
        ]
        out = apply_multitest_to_alphas(alphas, p_field="deflated_p")
        assert out[0]["passes_bh_q05"] is True
        assert out[1]["passes_bh_q05"] is False

    def test_all_missing_returns_default_tags(self) -> None:
        alphas = [{"name": "x"}, {"name": "y", "perm_p": None}]
        out = apply_multitest_to_alphas(alphas)
        for card in out:
            assert card["bh_q_value"] is None
            assert card["passes_bh_q05"] is False
