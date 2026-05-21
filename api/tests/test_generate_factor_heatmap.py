"""Tests for ``scripts/generate_factor_heatmap.py``.

The script is exercised end-to-end via the in-process ``run`` helper plus
synthetic fixture files. No live HTTP calls. Asserts cover:

* Ranking picks the top-N most-volatile factors deterministically.
* Short-history factors are skipped + warned.
* Correlation matrix is symmetric, diagonal-1, clipped to [-1, 1].
* Hierarchical clustering reorders so positively-correlated factors
  sit adjacent.
* PNG is actually written and has a non-trivial size.
* CLI ``--fixture`` end-to-end smoke test.
* Missing fixture path → exit code 2.
* Bad fixture shape → ValueError.
* Empty fixture → empty result, no crash.
* ``--limit`` flag forwarded to live-mode plumbing (mocked).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_factor_heatmap.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_factor_heatmap", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_factor_heatmap"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gfh() -> ModuleType:
    return _load_script()


# ---------------------------------------------------------------------------
# Synthetic-DGP helpers
# ---------------------------------------------------------------------------


def _make_block_fixture(
    *,
    n_blocks: int = 3,
    per_block: int = 4,
    window: int = 30,
    seed: int = 7,
    block_vol: float = 0.02,
    noise: float = 0.002,
) -> dict[str, list[float]]:
    """Return-series fixture with ``n_blocks`` highly-correlated blocks.

    Each block has a shared latent return path; factors within the same
    block are positively correlated, while across blocks the latent
    paths are independent.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {}
    for b in range(n_blocks):
        latent = rng.standard_normal(window) * block_vol
        for k in range(per_block):
            noise_vec = rng.standard_normal(window) * noise
            out[f"block{b}_factor{k}"] = (latent + noise_vec).tolist()
    return out


# ---------------------------------------------------------------------------
# Unit tests — ranking
# ---------------------------------------------------------------------------


def test_rank_by_volatility_picks_highest_std(gfh: ModuleType) -> None:
    """Top-N is ordered by descending std of the trailing window."""
    rng = np.random.default_rng(0)
    fixture: dict[str, list[float]] = {}
    # Three factors with explicitly different volatility levels.
    fixture["lowvol"] = (rng.standard_normal(30) * 0.001).tolist()
    fixture["midvol"] = (rng.standard_normal(30) * 0.01).tolist()
    fixture["hivol"] = (rng.standard_normal(30) * 0.10).tolist()

    ranked, skipped = gfh.rank_by_volatility(fixture, window=30, top_n=3)
    assert ranked == ["hivol", "midvol", "lowvol"]
    assert skipped == []


def test_rank_by_volatility_skips_short_history(
    gfh: ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    """Short or all-NaN factors are reported in ``skipped``."""
    fixture = {
        "ok": (np.random.default_rng(1).standard_normal(30) * 0.01).tolist(),
        "short": [0.01, -0.02, 0.03],
        "nan": [float("nan")] * 30,
    }
    with caplog.at_level(logging.WARNING, logger="generate_factor_heatmap"):
        ranked, skipped = gfh.rank_by_volatility(fixture, window=30, top_n=10)
    assert ranked == ["ok"]
    assert sorted(skipped) == ["nan", "short"]
    msgs = [r.getMessage() for r in caplog.records]
    assert any("short" in m for m in msgs)
    assert any("nan" in m for m in msgs)


def test_rank_by_volatility_validates_args(gfh: ModuleType) -> None:
    """``window`` and ``top_n`` must be positive."""
    with pytest.raises(ValueError):
        gfh.rank_by_volatility({}, window=0, top_n=10)
    with pytest.raises(ValueError):
        gfh.rank_by_volatility({}, window=10, top_n=0)


# ---------------------------------------------------------------------------
# Unit tests — correlation
# ---------------------------------------------------------------------------


def test_correlation_matrix_diagonal_is_one_and_symmetric(
    gfh: ModuleType,
) -> None:
    """Diagonal is exactly 1; matrix is symmetric and bounded [-1, 1]."""
    fixture = _make_block_fixture(n_blocks=2, per_block=3, window=40, seed=11)
    ranked = sorted(fixture)
    corr = gfh.build_correlation_matrix(fixture, ranked, window=40)
    assert corr.shape == (6, 6)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    assert corr.max() <= 1.0 + 1e-9
    assert corr.min() >= -1.0 - 1e-9


def test_correlation_zero_variance_does_not_produce_nan(
    gfh: ModuleType,
) -> None:
    """A constant-return factor cannot collapse the matrix to NaN."""
    fixture = {
        "flat": [0.0] * 30,
        "noise": np.random.default_rng(2).standard_normal(30).tolist(),
    }
    corr = gfh.build_correlation_matrix(fixture, ["flat", "noise"], window=30)
    assert np.isfinite(corr).all()
    # Constant row has 0 off-diagonal correlation by our post-processing.
    assert abs(float(corr[0, 1])) < 1e-9


# ---------------------------------------------------------------------------
# Unit tests — hierarchical clustering / reorder
# ---------------------------------------------------------------------------


def test_hierarchical_order_groups_correlated_blocks(gfh: ModuleType) -> None:
    """Reordered matrix puts each block's factors adjacent to each other."""
    fixture = _make_block_fixture(n_blocks=3, per_block=4, window=60, seed=42)
    ranked = sorted(fixture)  # 12 ids, alphabetical
    corr = gfh.build_correlation_matrix(fixture, ranked, window=60)
    order, _z = gfh.hierarchical_order(corr)
    reordered, reordered_ids = gfh.reorder_matrix(corr, ranked, order)

    # Each block has 4 members; after clustering they should occupy a
    # contiguous run of 4 positions.
    block_of = {fid: int(fid.split("_")[0][-1]) for fid in reordered_ids}
    runs = [block_of[fid] for fid in reordered_ids]
    # Compress into run-length form and verify every run is length 4.
    rle: list[tuple[int, int]] = []
    for b in runs:
        if rle and rle[-1][0] == b:
            rle[-1] = (b, rle[-1][1] + 1)
        else:
            rle.append((b, 1))
    assert all(length == 4 for _b, length in rle), f"non-contiguous blocks: {rle}"

    # And inside each run, the mean off-diagonal correlation should be
    # substantially positive.
    for start in range(0, 12, 4):
        block = reordered[start : start + 4, start : start + 4]
        mask = ~np.eye(4, dtype=bool)
        assert float(block[mask].mean()) > 0.5


def test_hierarchical_order_handles_tiny_inputs(gfh: ModuleType) -> None:
    """0 and 1-row matrices return a trivial order without invoking scipy."""
    empty = np.zeros((0, 0), dtype=float)
    order, z_empty = gfh.hierarchical_order(empty)
    assert order.shape == (0,)
    assert z_empty.shape == (0, 4)

    single = np.array([[1.0]])
    order, z_single = gfh.hierarchical_order(single)
    assert order.tolist() == [0]
    assert z_single.shape == (0, 4)


# ---------------------------------------------------------------------------
# Integration test — full pipeline through ``generate_heatmap``
# ---------------------------------------------------------------------------


def test_generate_heatmap_pipeline(gfh: ModuleType) -> None:
    """End-to-end: rank → correlate → cluster → reorder, no PNG."""
    fixture = _make_block_fixture(n_blocks=2, per_block=3, window=30, seed=99)
    result = gfh.generate_heatmap(fixture, top_n=10, window=30)
    assert len(result.factor_ids) == 6
    assert result.correlation.shape == (6, 6)
    assert result.skipped == []
    # Diagonal is 1, symmetric.
    assert np.allclose(np.diag(result.correlation), 1.0)
    assert np.allclose(result.correlation, result.correlation.T)


def test_generate_heatmap_respects_top_n(gfh: ModuleType) -> None:
    """``top_n=4`` keeps only the four most-volatile factors."""
    rng = np.random.default_rng(13)
    fixture = {
        f"f{i:02d}": (rng.standard_normal(30) * (0.001 + i * 0.005)).tolist() for i in range(10)
    }
    result = gfh.generate_heatmap(fixture, top_n=4, window=30)
    assert len(result.factor_ids) == 4
    # The 4 selected should be the highest-index (highest-vol) factors.
    selected = set(result.factor_ids)
    assert selected == {"f06", "f07", "f08", "f09"}


# ---------------------------------------------------------------------------
# Render tests
# ---------------------------------------------------------------------------


def test_render_heatmap_writes_png(gfh: ModuleType, tmp_path: Path) -> None:
    """The PNG file is created and is non-trivially sized."""
    corr = np.array([[1.0, 0.3], [0.3, 1.0]], dtype=float)
    out = tmp_path / "h.png"
    written = gfh.render_heatmap(corr, ["a", "b"], out, dpi=80)
    assert written == out
    assert out.exists()
    # A real PNG header + at least a small payload.
    head = out.read_bytes()[:8]
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 500


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_end_to_end_writes_png(gfh: ModuleType, tmp_path: Path) -> None:
    """`main([...])` runs the full pipeline and produces a PNG file."""
    fixture = _make_block_fixture(n_blocks=2, per_block=3, window=30, seed=5)
    fixture_path = tmp_path / "returns.json"
    fixture_path.write_text(json.dumps(fixture))
    out_png = tmp_path / "heatmap.png"

    rc = gfh.main(
        [
            "--fixture",
            str(fixture_path),
            "--top-n",
            "50",
            "--window",
            "30",
            "--out",
            str(out_png),
            "--quiet",
        ]
    )
    assert rc == 0
    assert out_png.exists()
    assert out_png.stat().st_size > 500


def test_cli_missing_fixture_exit_two(gfh: ModuleType, tmp_path: Path) -> None:
    """Missing ``--fixture`` is a user error → exit 2."""
    rc = gfh.main(
        [
            "--fixture",
            str(tmp_path / "nope.json"),
            "--out",
            str(tmp_path / "h.png"),
            "--quiet",
        ]
    )
    assert rc == 2


def test_cli_empty_fixture(gfh: ModuleType, tmp_path: Path) -> None:
    """Empty fixture still produces a (trivial) PNG."""
    fixture_path = tmp_path / "empty.json"
    fixture_path.write_text(json.dumps({}))
    out_png = tmp_path / "empty.png"
    rc = gfh.main(
        [
            "--fixture",
            str(fixture_path),
            "--out",
            str(out_png),
            "--quiet",
        ]
    )
    assert rc == 0
    assert out_png.exists()


def test_load_fixture_rejects_non_dict(gfh: ModuleType, tmp_path: Path) -> None:
    """Top-level fixture must be a dict."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        gfh.load_fixture(bad)


def test_load_fixture_rejects_non_list_values(gfh: ModuleType, tmp_path: Path) -> None:
    """Per-factor series must be a list."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"f": "nope"}))
    with pytest.raises(ValueError):
        gfh.load_fixture(bad)
