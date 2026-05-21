"""Schemas for /strategies/scan and the multi-mode scanner."""

from __future__ import annotations

from datetime import date as _date
from typing import Literal

from pydantic import BaseModel, Field

from pfm.schemas.common import (
    ScanModeLit,
)


class ScanRequest(BaseModel):
    mode: ScanModeLit = "all"
    theme: str | None = Field(default=None, description="Filter to one theme.")
    factor_ids: list[str] | None = Field(
        default=None,
        max_length=200,
        description="Explicit factor ids to scan; overrides theme when given.",
    )
    start: _date
    end: _date
    max_pairs: int = Field(default=500, ge=10, le=5000)
    n_obs_min: int = Field(default=30, ge=10, le=500)
    impl_tolerance: float = Field(default=0.02, ge=0.0, le=0.50)
    impl_n_violations_min: int = Field(default=5, ge=1, le=100)
    cond_beta_min: float = Field(default=0.30, ge=0.0, le=2.0)
    cond_r2_min: float = Field(default=0.10, ge=0.0, le=1.0)
    coint_adf_max_p: float = Field(default=0.05, gt=0.0, lt=0.5)
    coint_half_life_max: float = Field(default=60.0, ge=1.0, le=365.0)
    top_k_per_track: int = Field(default=25, ge=1, le=200)


class ScanHitOut(BaseModel):
    kind: Literal["implication", "conditional", "cointegration"]
    a_id: str
    b_id: str
    score: float
    n_obs: int
    summary: str
    n_violations: int = 0
    max_gap: float | None = None
    beta: float | None = None
    beta_ci_lo: float | None = None
    beta_ci_hi: float | None = None
    r_squared: float | None = None
    adf_pvalue: float | None = None
    half_life_days: float | None = None
    surprise: float | None = None


class ScanResponse(BaseModel):
    mode: ScanModeLit
    n_factors_scanned: int
    n_pairs_evaluated: int
    runtime_seconds: float
    implication: list[ScanHitOut]
    conditional: list[ScanHitOut]
    cointegration: list[ScanHitOut]
