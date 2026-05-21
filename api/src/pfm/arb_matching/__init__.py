"""Arb match-quality helpers.

This package houses utilities for vetting whether two prediction-market
listings (e.g. one on Polymarket, one on Kalshi) are about the *same*
event, before downstream consumers price them as an arbitrage pair.

Modules
-------
- ``date_extractor`` — robust resolution-window extraction from market
  titles/descriptions. See :func:`pfm.arb_matching.date_extractor.extract_resolution_window`.
"""

from pfm.arb_matching.date_extractor import (
    ResolutionWindow,
    extract_resolution_window,
    windows_overlap,
)

__all__ = [
    "ResolutionWindow",
    "extract_resolution_window",
    "windows_overlap",
]
