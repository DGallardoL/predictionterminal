"""Candidate pricing models for binary prediction markets.

Four models are provided, each conforming to the ``Pricer`` protocol:

* :class:`RiskNeutralLogit` — logistic re-pricing from ``market_price``,
  ``log(T)`` and a signed ``news_evidence`` term, calibrated by MLE.
* :class:`BlackScholesDigital` — digital-option valuation adapted to
  threshold-style markets (``BTC > K``).
* :class:`BrownianBridge` — closed-form Brownian-bridge survival
  probability for poll-style markets.
* :class:`BetaBinomialBayes` — Beta-Binomial posterior driven by signed
  news evidence; prior parameters fit by Method of Moments.

This module is *pure math* — no httpx, no live market calls. T82 wires
the models into the empirical-calibration harness.
"""

from __future__ import annotations

from pfm.pricing.binary_models import (
    BetaBinomialBayes,
    BlackScholesDigital,
    BrownianBridge,
    MarketState,
    Pricer,
    PricingResult,
    RiskNeutralLogit,
)

__all__ = [
    "BetaBinomialBayes",
    "BlackScholesDigital",
    "BrownianBridge",
    "MarketState",
    "Pricer",
    "PricingResult",
    "RiskNeutralLogit",
]
