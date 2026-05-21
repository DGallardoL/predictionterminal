"""Shared literals, base classes, and small primitives reused across the API."""

from __future__ import annotations

from datetime import date as _date
from typing import Literal

from pydantic import BaseModel, Field

# ---- /fit -------------------------------------------------------------------

ReturnTypeLit = Literal["log", "simple"]
RegressionLit = Literal["ols", "hac", "ridge", "lasso", "quantile"]
AlignmentLit = Literal["strict", "ffill"]


# Stock/crypto ticker validation. Covers: alphabet, digits, dot (BRK.B),
# hyphen (BF-B, BTC-USD), caret (^GSPC). Rejects spaces, $, %, etc. — those
# trigger 4-second upstream fallthrough (yfinance/Tiingo/Stooq) before
# failing with a confusing 502; this regex catches them at Pydantic.
TICKER_PATTERN = r"^[A-Za-z0-9.\-^]{1,10}$"


class CustomFactor(BaseModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_\-]+$")
    slug: str = Field(min_length=1, max_length=200)
    name: str | None = None


# ---- /strategies/* ----------------------------------------------------------


class _StrategyPairBase(BaseModel):
    """Common fields for two-factor strategy requests."""

    start: _date
    end: _date
    epsilon: float = Field(default=0.01, ge=0.0, le=0.1)


# ---- /strategies/spot-vs-implied -------------------------------------------


GeometryLit = Literal["terminal", "one_touch_up", "one_touch_down"]


# ---- /strategies/fred-cointegration ----------------------------------------


FredSeriesLit = Literal["DFF", "DGS2", "DGS10", "CPIAUCSL", "UNRATE", "VIXCLS"]
FredTransformLit = Literal["raw", "diff", "log", "logit"]


# ---- /strategies/scan ------------------------------------------------------


ScanModeLit = Literal["implication", "conditional", "cointegration", "all"]


# ---- /health ----------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
