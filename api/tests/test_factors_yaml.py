"""Tests for ``load_factors``."""

from __future__ import annotations

from pathlib import Path

import pytest

from pfm.factors import load_factors


def test_load_factors_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "f.yml"
    p.write_text(
        """
factors:
  - id: a
    name: A
    slug: slug-a
    source: polymarket
    description: desc a
"""
    )
    factors = load_factors(p)
    assert "a" in factors
    assert factors["a"].slug == "slug-a"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_factors(tmp_path / "nope.yml")


def test_duplicate_id_raises(tmp_path: Path) -> None:
    p = tmp_path / "f.yml"
    p.write_text(
        """
factors:
  - id: a
    name: A
    slug: x
    source: polymarket
    description: d
  - id: a
    name: A2
    slug: y
    source: polymarket
    description: d
"""
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_factors(p)


def test_missing_keys_raises(tmp_path: Path) -> None:
    p = tmp_path / "f.yml"
    p.write_text(
        """
factors:
  - id: a
    name: A
"""
    )
    with pytest.raises(ValueError, match="missing keys"):
        load_factors(p)
