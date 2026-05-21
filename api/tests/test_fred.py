"""Tests for ``pfm.sources.fred`` — auth-free FRED CSV fetcher."""

from __future__ import annotations

import time

import httpx
import pandas as pd
import pytest
import respx

from pfm.sources.fred import (
    FREDGRAPH_BASE,
    FredDataError,
    fetch_fred_series,
)

SAMPLE_CSV = """DATE,DFF
2025-09-01,5.32
2025-09-02,5.32
2025-09-03,.
2025-09-04,5.31
2025-09-05,5.31
"""


@respx.mock
def test_fetch_dff_basic() -> None:
    respx.get(FREDGRAPH_BASE).mock(return_value=httpx.Response(200, text=SAMPLE_CSV))
    s = fetch_fred_series(
        "DFF",
        start=pd.Timestamp("2025-09-01", tz="UTC"),
        end=pd.Timestamp("2025-09-05", tz="UTC"),
    )
    assert s.name == "DFF"
    # 5 days of values, with one missing (`.`) which forward-filled to 5.32.
    assert len(s) == 5
    assert s.iloc[0] == 5.32
    assert s.iloc[2] == 5.32  # ffill from prior bar
    assert s.iloc[-1] == 5.31


@respx.mock
def test_diff_transform() -> None:
    respx.get(FREDGRAPH_BASE).mock(return_value=httpx.Response(200, text=SAMPLE_CSV))
    s = fetch_fred_series(
        "DFF",
        start=pd.Timestamp("2025-09-01", tz="UTC"),
        end=pd.Timestamp("2025-09-05", tz="UTC"),
        transform="diff",
    )
    # diff drops first row → NaN there
    assert pd.isna(s.iloc[0])
    # diff[3] = 5.31 - 5.32 = -0.01 (the actual transition)
    assert pytest.approx(s.iloc[3], abs=1e-6) == -0.01
    # diff[-1] = 5.31 - 5.31 = 0
    assert pytest.approx(s.iloc[-1], abs=1e-6) == 0.0


@respx.mock
def test_log_transform() -> None:
    import numpy as np

    respx.get(FREDGRAPH_BASE).mock(return_value=httpx.Response(200, text=SAMPLE_CSV))
    s = fetch_fred_series(
        "DFF",
        start=pd.Timestamp("2025-09-01", tz="UTC"),
        end=pd.Timestamp("2025-09-05", tz="UTC"),
        transform="log",
    )
    assert pytest.approx(s.iloc[0], abs=1e-3) == np.log(5.32)


@respx.mock
def test_logit_rejects_unbounded() -> None:
    respx.get(FREDGRAPH_BASE).mock(return_value=httpx.Response(200, text=SAMPLE_CSV))
    with pytest.raises(FredDataError, match="logit transform requires"):
        fetch_fred_series(
            "DFF",
            start=pd.Timestamp("2025-09-01", tz="UTC"),
            end=pd.Timestamp("2025-09-05", tz="UTC"),
            transform="logit",
        )


@respx.mock
def test_429_retries(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    route = respx.get(FREDGRAPH_BASE).mock(
        side_effect=[
            httpx.Response(429, text="rate"),
            httpx.Response(429, text="rate"),
            httpx.Response(200, text=SAMPLE_CSV),
        ]
    )
    s = fetch_fred_series(
        "DFF",
        start=pd.Timestamp("2025-09-01", tz="UTC"),
        end=pd.Timestamp("2025-09-05", tz="UTC"),
        max_retries=5,
    )
    assert route.call_count == 3
    assert len(s) == 5


@respx.mock
def test_persistent_error_raises(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    respx.get(FREDGRAPH_BASE).mock(
        return_value=httpx.Response(500, text="server"),
    )
    with pytest.raises(FredDataError):
        fetch_fred_series(
            "DFF",
            start=pd.Timestamp("2025-09-01", tz="UTC"),
            end=pd.Timestamp("2025-09-05", tz="UTC"),
            max_retries=2,
        )
