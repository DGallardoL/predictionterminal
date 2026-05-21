"""Tests for ``scripts/factor_cluster_report.py``.

The script is exercised end-to-end via the in-process ``run`` helper plus
synthetic fixture files. No live HTTP calls. Asserts cover:

* Mocked factors yml with 20 factors → k=4 produces 4 clusters
* Empty input → empty output, no crash
* Missing return series → factor skipped + ``WARNING`` logged
* Reproducibility: same seed → identical cluster assignments
* KMeans ``n_init=10`` propagated from the CLI default
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

# Load the script as a module via importlib — it lives in ``scripts/`` and
# isn't a regular package member.
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "factor_cluster_report.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("factor_cluster_report", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so ``@dataclass`` (which resolves the owning
    # module via ``sys.modules[cls.__module__]``) finds the in-flight module.
    sys.modules["factor_cluster_report"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fcr() -> ModuleType:
    return _load_script()


# ---------------------------------------------------------------------------
# Synthetic-DGP helpers
# ---------------------------------------------------------------------------


def _make_four_block_fixture(
    *,
    n_per_block: int = 5,
    window: int = 30,
    seed: int = 7,
) -> dict[str, list[float]]:
    """Four well-separated clusters of correlated return series.

    Each block shares a strong latent signal so factors within a block
    z-score to nearly identical vectors. With k=4 KMeans should put each
    block in its own cluster.
    """
    rng = np.random.default_rng(seed)
    fixture: dict[str, list[float]] = {}
    block_signs: list[np.ndarray] = []
    for block in range(4):
        # Each block has a distinct "shape" — large step at a different point.
        latent = np.zeros(window)
        latent[block * (window // 4) : (block + 1) * (window // 4)] = 1.0
        latent = latent - latent.mean()  # de-mean so z-score is well defined
        block_signs.append(latent)
        for k in range(n_per_block):
            noise = rng.standard_normal(window) * 0.05
            series = latent + noise
            fixture[f"block{block}_factor{k}"] = series.tolist()
    return fixture


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_twenty_factors_four_clusters(tmp_path: Path, fcr: ModuleType) -> None:
    """20 factors in 4 well-separated blocks → exactly 4 clusters of size 5."""
    fixture = _make_four_block_fixture(n_per_block=5, window=30, seed=11)
    assert len(fixture) == 20

    fixture_path = tmp_path / "returns.json"
    fixture_path.write_text(json.dumps(fixture))

    out_path = tmp_path / "factor-clusters.json"
    rc = fcr.main(
        [
            "--fixture",
            str(fixture_path),
            "--k",
            "4",
            "--window",
            "30",
            "--seed",
            "42",
            "--out",
            str(out_path),
            "--quiet",
        ]
    )
    assert rc == 0

    report = json.loads(out_path.read_text())
    assert report["k"] == 4
    assert report["factor_count"] == 20
    assert report["window"] == 30
    assert len(report["clusters"]) == 4
    assert report["skipped"] == []

    sizes = sorted(c["size"] for c in report["clusters"])
    assert sizes == [5, 5, 5, 5]

    # Each block's 5 factors should end up in the same cluster.
    fid_to_cluster: dict[str, int] = {}
    for c in report["clusters"]:
        for fid in c["factors"]:
            fid_to_cluster[fid] = c["id"]
    for block in range(4):
        labels = {fid_to_cluster[f"block{block}_factor{k}"] for k in range(5)}
        assert len(labels) == 1, f"block {block} split across {labels}"


def test_empty_fixture_graceful(tmp_path: Path, fcr: ModuleType) -> None:
    """Empty input → empty cluster list, factor_count=0, no crash."""
    fixture_path = tmp_path / "empty.json"
    fixture_path.write_text(json.dumps({}))
    out_path = tmp_path / "out.json"

    rc = fcr.main(
        [
            "--fixture",
            str(fixture_path),
            "--k",
            "4",
            "--out",
            str(out_path),
            "--quiet",
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text())
    assert report["factor_count"] == 0
    assert report["clusters"] == []
    assert report["skipped"] == []


def test_short_series_skipped_with_warning(
    tmp_path: Path, fcr: ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    """A factor with <window finite obs is skipped and a warning is logged."""
    fixture = _make_four_block_fixture(n_per_block=2, window=30, seed=5)
    # Inject one short series — should be skipped.
    fixture["short_factor"] = [0.01, -0.01, 0.02]
    # And one all-NaN series — finite count = 0, should also be skipped.
    fixture["nan_factor"] = [float("nan")] * 30

    fixture_path = tmp_path / "returns.json"
    fixture_path.write_text(json.dumps(fixture))
    out_path = tmp_path / "out.json"

    with caplog.at_level(logging.WARNING, logger="factor_cluster_report"):
        rc = fcr.main(
            [
                "--fixture",
                str(fixture_path),
                "--k",
                "4",
                "--window",
                "30",
                "--out",
                str(out_path),
                "--quiet",
            ]
        )
    assert rc == 0

    report = json.loads(out_path.read_text())
    assert "short_factor" in report["skipped"]
    assert "nan_factor" in report["skipped"]
    # The non-skipped factors should populate the clusters.
    placed = sum(len(c["factors"]) for c in report["clusters"])
    assert placed == 8  # 4 blocks * 2 each
    # At least one warning per skipped factor.
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("short_factor" in m for m in msgs)
    assert any("nan_factor" in m for m in msgs)


def test_reproducibility_same_seed(tmp_path: Path, fcr: ModuleType) -> None:
    """Two runs with the same seed produce identical cluster assignments."""
    fixture = _make_four_block_fixture(n_per_block=5, window=30, seed=3)
    fixture_path = tmp_path / "returns.json"
    fixture_path.write_text(json.dumps(fixture))

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    args_template = [
        "--fixture",
        str(fixture_path),
        "--k",
        "4",
        "--window",
        "30",
        "--seed",
        "1234",
        "--quiet",
    ]
    assert fcr.main([*args_template, "--out", str(out_a)]) == 0
    assert fcr.main([*args_template, "--out", str(out_b)]) == 0

    report_a = json.loads(out_a.read_text())
    report_b = json.loads(out_b.read_text())

    # Strip the timestamp; everything else must be identical.
    report_a.pop("generated_at", None)
    report_b.pop("generated_at", None)
    assert report_a == report_b


def test_run_kmeans_n_init_default_is_ten(fcr: ModuleType) -> None:
    """Spec: ``n_init=10`` for KMeans stability."""
    assert fcr.DEFAULT_N_INIT == 10


def test_run_kmeans_clamps_k_to_n_rows(fcr: ModuleType) -> None:
    """Asking for more clusters than rows is clamped (not raised)."""
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((3, 30))
    labels, centroids = fcr.run_kmeans(matrix, k=10, seed=42)
    assert labels.shape == (3,)
    # 3 rows + k=10 → effective_k = 3
    assert centroids.shape == (3, 30)


def test_run_kmeans_empty_matrix(fcr: ModuleType) -> None:
    """Empty matrix returns empty arrays without raising."""
    matrix = np.zeros((0, 30), dtype=float)
    labels, centroids = fcr.run_kmeans(matrix, k=4, seed=42)
    assert labels.shape == (0,)
    assert centroids.shape == (0, 30)


def test_build_feature_matrix_zscores_rows(fcr: ModuleType) -> None:
    """Each surviving row is z-scored: mean≈0, std≈1 (or all-zero for flat)."""
    fixture = {
        "flat": [0.5] * 30,
        "linear": list(range(30)),
        "noise": np.random.default_rng(0).standard_normal(30).tolist(),
    }
    kept, matrix, skipped = fcr.build_feature_matrix(fixture, window=30)
    assert skipped == []
    assert sorted(kept) == kept  # deterministic ordering
    for row, fid in zip(matrix, kept, strict=True):
        if fid == "flat":
            # Constant series → zero row by design.
            assert np.allclose(row, 0.0)
        else:
            assert abs(float(row.mean())) < 1e-9
            assert abs(float(row.std()) - 1.0) < 1e-9


def test_uses_last_window_observations(fcr: ModuleType) -> None:
    """``window=30`` picks the trailing 30 entries from a longer series."""
    series = list(range(100))  # 100 obs
    kept, matrix, _ = fcr.build_feature_matrix({"f": series}, window=30)
    assert kept == ["f"]
    assert matrix.shape == (1, 30)
    # The trailing 30 of an arithmetic progression z-scores to the same
    # shape as the trailing 30 of the original (linearly increasing).
    assert matrix[0, 0] < matrix[0, -1]


def test_missing_fixture_returns_exit_code_two(tmp_path: Path, fcr: ModuleType) -> None:
    """Non-existent ``--fixture`` path is a user error (exit 2)."""
    rc = fcr.main(
        [
            "--fixture",
            str(tmp_path / "does_not_exist.json"),
            "--k",
            "4",
            "--out",
            str(tmp_path / "out.json"),
            "--quiet",
        ]
    )
    assert rc == 2


def test_load_fixture_rejects_non_dict(tmp_path: Path, fcr: ModuleType) -> None:
    """Fixture JSON must be a dict; a list is rejected with ValueError."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        fcr.load_fixture(bad)


def test_load_fixture_rejects_non_list_values(tmp_path: Path, fcr: ModuleType) -> None:
    """Each value must be a list of floats."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"f": "not a list"}))
    with pytest.raises(ValueError):
        fcr.load_fixture(bad)


def test_assemble_report_shape(fcr: ModuleType) -> None:
    """Report contains all required keys in the documented order/types."""
    factor_ids = ["a", "b", "c", "d"]
    labels = np.array([0, 0, 1, 1])
    centroids = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    report = fcr.assemble_report(
        factor_ids,
        labels,
        centroids,
        k=2,
        window=3,
        seed=42,
        factor_count=4,
        skipped=["skipped_one"],
        generated_at="2026-05-16T00:00:00Z",
    )
    assert set(report.keys()) == {
        "generated_at",
        "k",
        "factor_count",
        "window",
        "seed",
        "clusters",
        "skipped",
    }
    assert report["generated_at"] == "2026-05-16T00:00:00Z"
    assert report["skipped"] == ["skipped_one"]
    assert len(report["clusters"]) == 2
    for c in report["clusters"]:
        assert set(c.keys()) == {"id", "size", "factors", "centroid_summary"}
        assert "mean=" in c["centroid_summary"]
