"""Tests for ``api/scripts/sanitize_alpha_strategies.py``.

The script is invoked both directly via ``python3 api/scripts/...`` and
imported from these tests as a module to exercise its detectors and
end-to-end ``sanitize(path)`` flow.

We import via ``importlib`` because the script lives outside the
``pfm`` package's ``src/`` layout.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "api" / "scripts" / "sanitize_alpha_strategies.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("sanitize_alpha_strategies", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def san():
    return _load_script_module()


def _write_payload(tmp_path: Path, strategies: list[dict], **extra) -> Path:
    payload: dict = {"strategies": strategies}
    payload.update(extra)
    target = tmp_path / "alpha_strategies.json"
    target.write_text(json.dumps(payload, indent=2))
    return target


# ---------------------------------------------------------------------------
# Detector unit tests
# ---------------------------------------------------------------------------


class TestDetectFullSharpeZeroOosHigh:
    def test_zero_full_sharpe_and_high_oos_flags(self, san):
        assert san._detect_full_sharpe_zero_oos_high({"full_sharpe": 0.0, "oos_sharpe": 2.5})

    def test_none_full_sharpe_and_high_oos_flags(self, san):
        assert san._detect_full_sharpe_zero_oos_high({"full_sharpe": None, "oos_sharpe": 1.7})

    def test_oos_below_threshold_does_not_flag(self, san):
        assert not san._detect_full_sharpe_zero_oos_high({"full_sharpe": 0.0, "oos_sharpe": 0.5})

    def test_full_sharpe_present_and_nontrivial_does_not_flag(self, san):
        assert not san._detect_full_sharpe_zero_oos_high({"full_sharpe": 1.2, "oos_sharpe": 2.5})

    def test_missing_oos_does_not_flag(self, san):
        assert not san._detect_full_sharpe_zero_oos_high({"full_sharpe": 0.0})

    def test_garbage_oos_string_does_not_flag(self, san):
        assert not san._detect_full_sharpe_zero_oos_high({"full_sharpe": 0.0, "oos_sharpe": "n/a"})


class TestDetectLowNObs:
    def test_low_n_obs_with_full_sharpe_flags(self, san):
        assert san._detect_low_n_obs_with_sharpe(
            {"n_obs": 25, "full_sharpe": 1.2, "oos_sharpe": 1.2}
        )

    def test_low_n_obs_with_only_oos_flags(self, san):
        assert san._detect_low_n_obs_with_sharpe(
            {"n_obs": 5, "full_sharpe": None, "oos_sharpe": 3.0}
        )

    def test_low_n_obs_but_no_sharpe_does_not_flag(self, san):
        assert not san._detect_low_n_obs_with_sharpe(
            {"n_obs": 3, "full_sharpe": None, "oos_sharpe": None}
        )

    def test_low_n_obs_with_zero_sharpes_does_not_flag(self, san):
        assert not san._detect_low_n_obs_with_sharpe(
            {"n_obs": 10, "full_sharpe": 0.0, "oos_sharpe": 0.0}
        )

    def test_boundary_30_does_not_flag(self, san):
        assert not san._detect_low_n_obs_with_sharpe(
            {"n_obs": 30, "full_sharpe": 1.5, "oos_sharpe": 1.5}
        )

    def test_missing_n_obs_does_not_flag(self, san):
        assert not san._detect_low_n_obs_with_sharpe({"full_sharpe": 1.5, "oos_sharpe": 1.5})


class TestDetectHalfLife:
    def test_below_lower_bound_flags(self, san):
        assert san._detect_half_life_out_of_range({"half_life_days": 0.1})

    def test_above_upper_bound_flags(self, san):
        assert san._detect_half_life_out_of_range({"half_life_days": 500.0})

    def test_at_lower_bound_does_not_flag(self, san):
        assert not san._detect_half_life_out_of_range({"half_life_days": 0.5})

    def test_at_upper_bound_does_not_flag(self, san):
        assert not san._detect_half_life_out_of_range({"half_life_days": 365.0})

    def test_typical_value_does_not_flag(self, san):
        assert not san._detect_half_life_out_of_range({"half_life_days": 5.0})

    def test_missing_does_not_flag(self, san):
        assert not san._detect_half_life_out_of_range({})


class TestDetectSharpeDivergence:
    def test_large_positive_divergence_flags(self, san):
        assert san._detect_sharpe_divergence({"full_sharpe": 0.3, "oos_sharpe": 6.5})

    def test_large_negative_divergence_flags(self, san):
        assert san._detect_sharpe_divergence({"full_sharpe": 3.0, "oos_sharpe": -3.0})

    def test_small_divergence_does_not_flag(self, san):
        assert not san._detect_sharpe_divergence({"full_sharpe": 1.0, "oos_sharpe": 2.5})

    def test_missing_either_does_not_flag(self, san):
        assert not san._detect_sharpe_divergence({"full_sharpe": 1.0})
        assert not san._detect_sharpe_divergence({"oos_sharpe": 1.0})


# ---------------------------------------------------------------------------
# apply_warning + normalize tests
# ---------------------------------------------------------------------------


class TestApplyWarning:
    def test_apply_single_tag_to_empty(self, san):
        s: dict = {}
        san._apply_warning(s, "tag_a")
        assert s["data_quality_warning"] == "tag_a"

    def test_apply_merges_with_existing(self, san):
        s = {"data_quality_warning": "tag_a"}
        san._apply_warning(s, "tag_b")
        assert s["data_quality_warning"] == "tag_a;tag_b"

    def test_apply_is_idempotent(self, san):
        s = {"data_quality_warning": "tag_a"}
        san._apply_warning(s, "tag_a")
        assert s["data_quality_warning"] == "tag_a"

    def test_apply_sorts_tags_for_diff_stability(self, san):
        s: dict = {}
        san._apply_warning(s, "zeta")
        san._apply_warning(s, "alpha")
        assert s["data_quality_warning"] == "alpha;zeta"

    def test_normalize_legacy_bool_true(self, san):
        s = {"data_quality_warning": True}
        san._normalize_existing_warning(s)
        assert s["data_quality_warning"] == san.WARNING_FS_ZERO_OOS_HIGH

    def test_normalize_legacy_bool_false_removes(self, san):
        s = {"data_quality_warning": False}
        san._normalize_existing_warning(s)
        assert "data_quality_warning" not in s


# ---------------------------------------------------------------------------
# End-to-end sanitize() tests
# ---------------------------------------------------------------------------


class TestSanitizeEndToEnd:
    def test_basic_flow_demotes_and_tags(self, san, tmp_path):
        path = _write_payload(
            tmp_path,
            [
                {
                    "pair_id": "good_one",
                    "tier": "A_STRUCTURAL",
                    "full_sharpe": 2.5,
                    "oos_sharpe": 2.5,
                    "n_obs": 250,
                    "half_life_days": 3.0,
                },
                {
                    "pair_id": "zero_fs_high_oos",
                    "tier": "C_TENTATIVE",
                    "full_sharpe": 0.0,
                    "oos_sharpe": 4.2,
                    "n_obs": 100,
                    "half_life_days": 2.0,
                },
                {
                    "pair_id": "tiny_sample",
                    "tier": "C_TENTATIVE",
                    "full_sharpe": 3.0,
                    "oos_sharpe": 3.0,
                    "n_obs": 5,
                    "half_life_days": 4.0,
                },
                {
                    "pair_id": "instant_revert",
                    "tier": "B_VALIDATED",
                    "full_sharpe": 1.5,
                    "oos_sharpe": 1.5,
                    "n_obs": 200,
                    "half_life_days": 0.1,
                },
                {
                    "pair_id": "diverge",
                    "tier": "B_VALIDATED",
                    "full_sharpe": 0.5,
                    "oos_sharpe": 9.0,
                    "n_obs": 200,
                    "half_life_days": 4.0,
                },
            ],
        )
        summary = san.sanitize(path)

        payload = json.loads(path.read_text())
        strategies = {s["pair_id"]: s for s in payload["strategies"]}

        # good_one untouched
        assert strategies["good_one"]["tier"] == "A_STRUCTURAL"
        assert "data_quality_warning" not in strategies["good_one"]

        # zero_fs_high_oos demoted + tagged
        assert strategies["zero_fs_high_oos"]["tier"] == "D_RAW"
        assert (
            san.WARNING_FS_ZERO_OOS_HIGH in strategies["zero_fs_high_oos"]["data_quality_warning"]
        )

        # tiny_sample tagged
        assert strategies["tiny_sample"]["tier"] == "D_RAW"
        assert san.WARNING_LOW_N_OBS in strategies["tiny_sample"]["data_quality_warning"]

        # instant_revert tagged
        assert strategies["instant_revert"]["tier"] == "D_RAW"
        assert san.WARNING_HALF_LIFE in strategies["instant_revert"]["data_quality_warning"]

        # diverge tagged with both divergence AND fs-zero (oos>1, fs<0.01? No,
        # 0.5 > 0.01, so only divergence fires).
        assert strategies["diverge"]["tier"] == "D_RAW"
        assert san.WARNING_SHARPE_DIVERGENCE in strategies["diverge"]["data_quality_warning"]

        assert summary["_total_flagged_rows"] == 4
        assert summary["_demoted_to_d_raw"] == 4
        assert summary[san.WARNING_FS_ZERO_OOS_HIGH] == 1
        assert summary[san.WARNING_LOW_N_OBS] == 1
        assert summary[san.WARNING_HALF_LIFE] == 1
        assert summary[san.WARNING_SHARPE_DIVERGENCE] == 1

    def test_duplicate_pair_id_only_marks_after_first(self, san, tmp_path):
        path = _write_payload(
            tmp_path,
            [
                {
                    "pair_id": "twin",
                    "tier": "A_STRUCTURAL",
                    "full_sharpe": 2.0,
                    "oos_sharpe": 2.0,
                    "n_obs": 200,
                    "half_life_days": 3.0,
                },
                {
                    "pair_id": "twin",
                    "tier": "A_STRUCTURAL",
                    "full_sharpe": 2.0,
                    "oos_sharpe": 2.0,
                    "n_obs": 200,
                    "half_life_days": 3.0,
                },
                {
                    "pair_id": "twin",
                    "tier": "A_STRUCTURAL",
                    "full_sharpe": 2.0,
                    "oos_sharpe": 2.0,
                    "n_obs": 200,
                    "half_life_days": 3.0,
                },
            ],
        )
        summary = san.sanitize(path)
        payload = json.loads(path.read_text())
        rows = payload["strategies"]
        assert rows[0]["tier"] == "A_STRUCTURAL"
        assert "data_quality_warning" not in rows[0]
        assert rows[1]["tier"] == "D_RAW"
        assert rows[1]["data_quality_warning"] == san.WARNING_DUPLICATE_PAIR_ID
        assert rows[2]["tier"] == "D_RAW"
        assert summary[san.WARNING_DUPLICATE_PAIR_ID] == 2

    def test_legacy_bool_warning_is_normalized(self, san, tmp_path):
        path = _write_payload(
            tmp_path,
            [
                {
                    "pair_id": "legacy",
                    "tier": "D_RAW",
                    "full_sharpe": None,
                    "oos_sharpe": None,
                    "n_obs": 100,
                    "half_life_days": 2.0,
                    "data_quality_warning": True,
                }
            ],
        )
        san.sanitize(path)
        payload = json.loads(path.read_text())
        assert payload["strategies"][0]["data_quality_warning"] == san.WARNING_FS_ZERO_OOS_HIGH

    def test_multiple_tags_joined_sorted(self, san, tmp_path):
        # A row that triggers both half-life and full-sharpe-zero detectors.
        path = _write_payload(
            tmp_path,
            [
                {
                    "pair_id": "multi",
                    "tier": "C_TENTATIVE",
                    "full_sharpe": 0.0,
                    "oos_sharpe": 3.0,
                    "n_obs": 100,
                    "half_life_days": 0.1,
                }
            ],
        )
        san.sanitize(path)
        payload = json.loads(path.read_text())
        warning = payload["strategies"][0]["data_quality_warning"]
        parts = warning.split(";")
        assert parts == sorted(parts)
        assert san.WARNING_FS_ZERO_OOS_HIGH in parts
        assert san.WARNING_HALF_LIFE in parts

    def test_idempotent(self, san, tmp_path):
        path = _write_payload(
            tmp_path,
            [
                {
                    "pair_id": "x",
                    "tier": "C_TENTATIVE",
                    "full_sharpe": 0.0,
                    "oos_sharpe": 3.0,
                    "n_obs": 100,
                    "half_life_days": 0.1,
                }
            ],
        )
        san.sanitize(path)
        first = path.read_text()
        san.sanitize(path)
        second = path.read_text()
        assert first == second

    def test_unexpected_schema_raises(self, san, tmp_path):
        bad = tmp_path / "broken.json"
        bad.write_text(json.dumps({"strategies": "not a list"}))
        with pytest.raises(SystemExit):
            san.sanitize(bad)

    def test_main_with_explicit_path(self, san, tmp_path):
        path = _write_payload(
            tmp_path,
            [
                {
                    "pair_id": "x",
                    "tier": "A_STRUCTURAL",
                    "full_sharpe": 2.0,
                    "oos_sharpe": 2.0,
                    "n_obs": 200,
                    "half_life_days": 3.0,
                }
            ],
        )
        rc = san.main(["sanitize_alpha_strategies.py", str(path)])
        assert rc == 0

    def test_main_missing_file_raises(self, san, tmp_path):
        with pytest.raises(SystemExit):
            san.main(["sanitize_alpha_strategies.py", str(tmp_path / "nope.json")])
