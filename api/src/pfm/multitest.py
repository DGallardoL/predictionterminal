"""Multiple-testing corrections and the deflated Sharpe ratio.

Running 88+ alpha candidates through a permutation/bootstrap test yields a
collection of marginal p-values.  At α=0.05 we expect ≈ 4 false positives by
chance even if every alpha is null.  This module exposes the two standard
family-wise / false-discovery-rate corrections:

* :func:`benjamini_hochberg_fdr` — controls the expected proportion of
  false discoveries (FDR) at level ``α``.  Less conservative than Bonferroni
  and the de-facto standard in financial-strategy validation pipelines
  (Harvey, Liu & Zhu 2016, "...and the Cross-Section of Expected Returns",
  *Review of Financial Studies* 29:5–68).
* :func:`bonferroni_correction` — controls the family-wise error rate
  strictly; each test must clear ``α/m``.  Useful for small ``m`` or when a
  single false discovery is unacceptable.

A convenience wrapper :func:`apply_multitest_to_alphas` runs BH-FDR over a
list of alpha-card dicts (the rows of ``web/data/alpha_strategies.json``)
and tags each row with ``bh_q_value``, ``passes_bh_q05`` and
``passes_bh_q10``.

References
----------
Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery
    rate: a practical and powerful approach to multiple testing."
    *Journal of the Royal Statistical Society B* 57:289-300.
Harvey, C., Liu, Y., & Zhu, H. (2016). "...and the Cross-Section of
    Expected Returns." *Review of Financial Studies* 29:5-68.
"""

from __future__ import annotations

import math
from typing import Any

# Euler-Mascheroni constant, appears in the Bailey-Lopez de Prado (2014)
# expression for the expected maximum Sharpe under the null.
EULER_MASCHERONI: float = 0.57721566490153286


def benjamini_hochberg_fdr(
    p_values: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply the Benjamini-Hochberg FDR procedure.

    Sorts the ``m`` input p-values in ascending order, then finds the
    largest rank ``k`` for which ``p_(k) <= alpha * k / m``.  All hypotheses
    with rank ``<= k`` are rejected.  The accompanying q-values are the
    standard step-up adjustment ``q_i = min_{j>=i} ( m * p_(j) / j )``,
    re-mapped to the original index order.

    Args:
        p_values: List of marginal p-values in [0, 1].
        alpha: Target FDR level.  Default 0.05.

    Returns:
        ``{"rejected_idx": [...], "q_values": [...], "n_significant": int}``.
        ``rejected_idx`` is sorted ascending; ``q_values`` is parallel to the
        input ``p_values`` (same order, same length).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    m = len(p_values)
    if m == 0:
        return {"rejected_idx": [], "q_values": [], "n_significant": 0}
    for i, p in enumerate(p_values):
        if not (0.0 <= float(p) <= 1.0) or math.isnan(p):
            raise ValueError(f"p_values[{i}]={p} is not a valid probability")

    indexed: list[tuple[float, int]] = sorted(
        ((float(p), i) for i, p in enumerate(p_values)),
        key=lambda t: t[0],
    )

    # Step-up rejection set --------------------------------------------------
    largest_k = -1
    for rank, (p_sorted, _) in enumerate(indexed, start=1):
        if p_sorted <= alpha * rank / m:
            largest_k = rank
    rejected_sorted_positions = list(range(largest_k))  # ranks 1..k → positions 0..k-1
    rejected_idx = sorted(indexed[pos][1] for pos in rejected_sorted_positions)

    # q-values: standard BH step-up adjustment, monotone non-decreasing in p.
    q_sorted = [0.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        p_sorted_val = indexed[rank - 1][0]
        q_raw = p_sorted_val * m / rank
        running_min = min(running_min, q_raw)
        q_sorted[rank - 1] = running_min

    q_values: list[float] = [0.0] * m
    for sorted_pos, (_, original_idx) in enumerate(indexed):
        q_values[original_idx] = float(min(1.0, q_sorted[sorted_pos]))

    return {
        "rejected_idx": rejected_idx,
        "q_values": q_values,
        "n_significant": len(rejected_idx),
    }


def bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Family-wise error-rate control by Bonferroni.

    Reject hypothesis ``i`` iff ``p_i <= alpha / m``.  The "adjusted" p-value
    is ``min(1, m * p_i)`` and is reported parallel to the input ordering.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    m = len(p_values)
    if m == 0:
        return {"rejected_idx": [], "adjusted_p_values": [], "n_significant": 0}
    threshold = alpha / m
    adjusted: list[float] = []
    rejected: list[int] = []
    for i, p in enumerate(p_values):
        p_f = float(p)
        if not (0.0 <= p_f <= 1.0) or math.isnan(p_f):
            raise ValueError(f"p_values[{i}]={p} is not a valid probability")
        adjusted.append(min(1.0, p_f * m))
        if p_f <= threshold:
            rejected.append(i)
    return {
        "rejected_idx": rejected,
        "adjusted_p_values": adjusted,
        "n_significant": len(rejected),
    }


def apply_multitest_to_alphas(
    alphas: list[dict[str, Any]],
    p_field: str = "perm_p",
) -> list[dict[str, Any]]:
    """Run BH-FDR over a list of alpha-card dicts and attach q-value tags.

    The input shape mirrors ``web/data/alpha_strategies.json`` rows.  Cards
    that lack ``p_field`` (or carry ``None``) are skipped — they keep their
    original fields untouched and pick up ``bh_q_value=None`` plus
    ``passes_bh_q05=False`` and ``passes_bh_q10=False`` so downstream code
    can treat missing-p as "fails the gate".

    The returned list is a *copy* in the same order; the input is not mutated.
    """
    out: list[dict[str, Any]] = [dict(card) for card in alphas]
    pvals: list[float] = []
    positions: list[int] = []
    for i, card in enumerate(out):
        p = card.get(p_field)
        if p is None:
            continue
        try:
            p_f = float(p)
        except (TypeError, ValueError):
            continue
        if math.isnan(p_f) or not (0.0 <= p_f <= 1.0):
            continue
        pvals.append(p_f)
        positions.append(i)

    # Default tagging for every card; overwrite the ones that participated.
    for card in out:
        card["bh_q_value"] = None
        card["passes_bh_q05"] = False
        card["passes_bh_q10"] = False

    if not pvals:
        return out

    bh = benjamini_hochberg_fdr(pvals, alpha=0.05)
    qvals = bh["q_values"]
    for sub_i, original_pos in enumerate(positions):
        q = float(qvals[sub_i])
        out[original_pos]["bh_q_value"] = q
        out[original_pos]["passes_bh_q05"] = q <= 0.05
        out[original_pos]["passes_bh_q10"] = q <= 0.10

    return out


# ---------------------------------------------------------------------------
# Deflated Sharpe ratio — Bailey & Lopez de Prado (2014), full version
# ---------------------------------------------------------------------------


def deflated_sharpe_full(
    sharpe_observed: float,
    n_obs: int,
    *,
    n_trials: int = 100,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sd_of_trial_sharpes: float | None = None,
    annualisation: float = 252.0,
) -> dict[str, float]:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (DSR), full form.

    The DSR adjusts an observed Sharpe ratio for two distinct biases:

    1. **Selection bias** when ``n_trials`` strategies were searched
       before the reported one was kept. The expected maximum Sharpe
       under the null grows roughly like ``Phi^{-1}(1 - 1/N)`` and is
       sharpened in BLDP eq. (5) using a Mill's-ratio-style two-term
       expansion involving the Euler-Mascheroni constant ``gamma``:

       .. math::

           E[\\widehat{SR}_{max}]
           \\approx \\sigma_{SR} \\cdot
           \\Big( (1-\\gamma) \\cdot \\Phi^{-1}(1 - 1/N)
                  + \\gamma \\cdot \\Phi^{-1}(1 - 1/(N e)) \\Big)

       where ``sigma_{SR}`` is the cross-trial Sharpe dispersion. When
       ``sd_of_trial_sharpes`` is None we use ``sigma_{SR} = 1`` (the
       conventional simplification when only the trial *count* is known).

    2. **Non-Gaussian-returns variance correction** (Bailey & Lopez de
       Prado 2014 eq. (9)): the standard error of the Sharpe estimator
       inflates with negative skew and excess kurtosis. The Edgeworth-
       expansion-based finite-sample SE is

       .. math::

           SE(\\widehat{SR})
           = \\sqrt{ \\frac{1 - \\gamma_3 \\widehat{SR}
                           + \\frac{\\gamma_4 - 1}{4} \\widehat{SR}^{2}}{T - 1} }

       where ``gamma_3`` is skew, ``gamma_4`` is the *non-excess* fourth
       moment (kurtosis; 3 for Gaussian).

    The deflated Sharpe statistic is the studentised distance between the
    observed Sharpe and the expected null maximum, all on a *per-period*
    scale:

    .. math::

        DSR = \\Phi\\!\\left( \\frac{\\widehat{SR} - E[\\widehat{SR}_{max}]}
                                  {SE(\\widehat{SR})} \\right)

    The deflated p-value is ``1 - DSR``.

    Backward compatibility: with ``skew=0, kurtosis=3,
    sd_of_trial_sharpes=None`` this reduces to the same expression
    implemented in :func:`pfm.robust_validation.deflated_sharpe_ratio`.

    Args:
        sharpe_observed: annualised observed Sharpe (matches the existing
            convention in :mod:`pfm.robust_validation`).
        n_obs: number of return observations T.
        n_trials: number of strategies searched (data-mining budget).
        skew: third standardised moment of returns. 0 = symmetric.
        kurtosis: fourth standardised moment (Pearson, NOT excess).
            3 = Gaussian.
        sd_of_trial_sharpes: optional cross-trial Sharpe dispersion
            ``sigma_{SR}``. Defaults to 1 when unknown.
        annualisation: trading-period count for converting annualised
            Sharpe to per-period Sharpe (252 for daily by default).

    Returns:
        Dict with ``deflated_sharpe`` (per-period DSR-deflated Sharpe),
        ``deflated_p_value``, ``expected_max_sharpe_under_null`` (per
        period), ``sigma_se`` (Edgeworth SE), ``skew``, ``kurtosis``,
        and ``n_trials``.
    """
    from scipy.stats import norm

    if n_obs < 5:
        return {
            "deflated_sharpe": 0.0,
            "deflated_p_value": 1.0,
            "expected_max_sharpe_under_null": 0.0,
            "sigma_se": float("nan"),
            "skew": float(skew),
            "kurtosis": float(kurtosis),
            "n_trials": int(n_trials),
        }
    n_trials = max(n_trials, 1)

    sigma_sr = 1.0 if sd_of_trial_sharpes is None else float(sd_of_trial_sharpes)

    # Expected maximum Sharpe under the null (Mill's ratio expansion).
    if n_trials > 1:
        expected_max = sigma_sr * (
            (1.0 - EULER_MASCHERONI) * float(norm.ppf(1.0 - 1.0 / n_trials))
            + EULER_MASCHERONI * float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
        )
    else:
        expected_max = 0.0
    expected_max = max(expected_max, 0.0)

    # Per-period observed Sharpe.
    sr_per = float(sharpe_observed) / math.sqrt(max(annualisation, 1.0))

    # Edgeworth SE (BLDP eq. 9). ``kurtosis`` is the non-excess fourth
    # standardised moment (3 = Gaussian).
    inner = 1.0 - skew * sr_per + ((kurtosis - 1.0) / 4.0) * sr_per * sr_per
    inner = max(inner, 0.0)
    sigma_se = math.sqrt(inner / max(n_obs - 1, 1))
    if sigma_se <= 0:
        z_star = 0.0
    else:
        z_star = (sr_per - expected_max) / sigma_se
    deflated_p = float(1.0 - norm.cdf(z_star))

    return {
        "deflated_sharpe": float(sr_per - expected_max),
        "deflated_p_value": deflated_p,
        "expected_max_sharpe_under_null": float(expected_max),
        "sigma_se": float(sigma_se),
        "skew": float(skew),
        "kurtosis": float(kurtosis),
        "n_trials": int(n_trials),
    }


__all__ = [
    "EULER_MASCHERONI",
    "apply_multitest_to_alphas",
    "benjamini_hochberg_fdr",
    "bonferroni_correction",
    "deflated_sharpe_full",
]
