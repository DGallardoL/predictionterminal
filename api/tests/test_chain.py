"""Tests for ``pfm.sources.chain.fetch_chained_history`` and the
extended ``FactorConfig`` / loader.

Strategy: mock the per-source fetchers via injectable parameters and feed
synthetic ``DataFrame`` outputs whose dates we control. We then assert
that the chain fetcher slices each segment to its proper active window
and concatenates correctly.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from pfm.factors import (
    CHAIN_SOURCE,
    ChainSegment,
    FactorConfig,
    load_factors,
)
from pfm.sources.chain import (
    _filter_segment_bars,
    _segment_window,
    fetch_chained_history,
    segments_signature,
)

# ───────────────────────────── helpers ────────────────────────────────


def _make_df(prices: dict[str, float]) -> pd.DataFrame:
    """Build a price DataFrame indexed by UTC dates."""
    idx = pd.to_datetime(list(prices.keys()), utc=True)
    return pd.DataFrame({"price": list(prices.values())}, index=idx)


def _segs(*tuples: tuple[str, str, str]) -> tuple[ChainSegment, ...]:
    """Shorthand: ``_segs(("polymarket", "x", "2026-02-12"), ...)``."""
    return tuple(ChainSegment(source=s, slug=sl, end=date.fromisoformat(e)) for s, sl, e in tuples)


def _stub_fetcher(per_slug: dict[str, pd.DataFrame]):
    """Build a stub that mimics ``fetch_polymarket_history`` /
    ``fetch_kalshi_history`` signatures and returns the pre-canned df
    for the requested slug."""

    def fetch_poly(client, *, slug, start, end):
        return per_slug.get(slug, pd.DataFrame()).copy()

    def fetch_kalshi(client, *, market_ticker, start, end):
        return per_slug.get(market_ticker, pd.DataFrame()).copy()

    return fetch_poly, fetch_kalshi


# ───────────────────────── ChainSegment validation ────────────────────


class TestChainSegment:
    def test_valid(self) -> None:
        s = ChainSegment(source="kalshi", slug="KXFOO-26MAY", end=date(2026, 5, 13))
        assert s.source == "kalshi"

    def test_bad_source(self) -> None:
        with pytest.raises(ValueError, match="source must be one of"):
            ChainSegment(source="garbage", slug="x", end=date(2026, 1, 1))

    def test_empty_slug(self) -> None:
        with pytest.raises(ValueError, match="slug must be non-empty"):
            ChainSegment(source="kalshi", slug="", end=date(2026, 1, 1))

    def test_bad_end_type(self) -> None:
        with pytest.raises(ValueError, match="must be a date"):
            ChainSegment(source="kalshi", slug="x", end="2026-01-01")  # type: ignore[arg-type]


# ─────────────────────── FactorConfig validation ──────────────────────


class TestFactorConfigChain:
    def test_chain_factor_valid(self) -> None:
        fc = FactorConfig(
            id="cpi",
            name="CPI",
            slug="cpi",
            source=CHAIN_SOURCE,
            description="d",
            segments=_segs(
                ("kalshi", "A", "2026-02-12"),
                ("kalshi", "B", "2026-03-12"),
            ),
        )
        assert fc.is_chained
        assert len(fc.segments) == 2

    def test_chain_factor_no_segments_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty segments"):
            FactorConfig(
                id="x",
                name="x",
                slug="x",
                source=CHAIN_SOURCE,
                description="d",
                segments=(),
            )

    def test_chain_factor_unsorted_rejected(self) -> None:
        with pytest.raises(ValueError, match="ascending end-date order"):
            FactorConfig(
                id="x",
                name="x",
                slug="x",
                source=CHAIN_SOURCE,
                description="d",
                segments=_segs(
                    ("kalshi", "A", "2026-03-12"),
                    ("kalshi", "B", "2026-02-12"),
                ),
            )

    def test_chain_factor_duplicate_ends_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique end dates"):
            FactorConfig(
                id="x",
                name="x",
                slug="x",
                source=CHAIN_SOURCE,
                description="d",
                segments=_segs(
                    ("kalshi", "A", "2026-02-12"),
                    ("kalshi", "B", "2026-02-12"),
                ),
            )

    def test_single_source_with_segments_rejected(self) -> None:
        with pytest.raises(ValueError, match="only source=chain may carry segments"):
            FactorConfig(
                id="x",
                name="x",
                slug="some-slug",
                source="polymarket",
                description="d",
                segments=_segs(("kalshi", "A", "2026-02-12")),
            )

    def test_unknown_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="source must be one of"):
            FactorConfig(
                id="x",
                name="x",
                slug="x",
                source="bogus",
                description="d",
            )


# ────────────────────────── _segment_window ───────────────────────────


class TestSegmentWindow:
    def test_first_segment_no_lower(self) -> None:
        segs = _segs(
            ("kalshi", "A", "2026-02-12"),
            ("kalshi", "B", "2026-03-12"),
        )
        lower, upper = _segment_window(segs, 0)
        assert lower is None
        assert upper == date(2026, 2, 12)

    def test_later_segment_lower_is_prev_end(self) -> None:
        segs = _segs(
            ("kalshi", "A", "2026-02-12"),
            ("kalshi", "B", "2026-03-12"),
            ("kalshi", "C", "2026-04-10"),
        )
        lower, upper = _segment_window(segs, 2)
        assert lower == date(2026, 3, 12)
        assert upper == date(2026, 4, 10)


# ────────────────────────── _filter_segment_bars ──────────────────────


class TestFilterSegmentBars:
    def test_empty_input(self) -> None:
        out = _filter_segment_bars(pd.DataFrame(), None, date(2026, 2, 12))
        assert out.empty

    def test_first_segment_inclusive_upper(self) -> None:
        df = _make_df(
            {
                "2026-01-15": 0.10,
                "2026-02-12": 0.50,  # exactly the upper bound — kept
                "2026-02-13": 0.55,  # past upper — dropped
            }
        )
        out = _filter_segment_bars(df, None, date(2026, 2, 12))
        assert list(out.index.strftime("%Y-%m-%d")) == ["2026-01-15", "2026-02-12"]

    def test_later_segment_exclusive_lower(self) -> None:
        df = _make_df(
            {
                "2026-02-12": 0.50,  # belongs to the previous segment — dropped
                "2026-02-13": 0.55,  # belongs to this segment — kept
                "2026-03-12": 0.60,  # upper bound — kept
                "2026-03-13": 0.62,  # past upper — dropped
            }
        )
        out = _filter_segment_bars(df, date(2026, 2, 12), date(2026, 3, 12))
        assert list(out.index.strftime("%Y-%m-%d")) == ["2026-02-13", "2026-03-12"]


# ──────────────────────── fetch_chained_history ───────────────────────


class TestFetchChainedHistory:
    def test_two_segment_concatenation(self) -> None:
        # Each segment returns its full history; the chain function
        # must filter each to its active window and concatenate.
        seg_a = _make_df(
            {
                "2026-01-10": 0.20,
                "2026-02-01": 0.30,
                "2026-02-12": 0.45,  # last bar of segment A's window
                "2026-02-25": 0.50,  # AFTER A.end — must be dropped from A
            }
        )
        seg_b = _make_df(
            {
                "2026-01-30": 0.10,  # before A.end — dropped (belongs to A's window)
                "2026-02-13": 0.42,  # first bar of segment B's window
                "2026-03-12": 0.48,
            }
        )
        fpoly, fkalshi = _stub_fetcher({"slug-A": seg_a, "slug-B": seg_b})
        segs = _segs(
            ("polymarket", "slug-A", "2026-02-12"),
            ("polymarket", "slug-B", "2026-03-12"),
        )
        out = fetch_chained_history(
            segs,
            poly="dummy",
            kalshi=None,
            start=pd.Timestamp("2026-01-01", tz="UTC"),
            end=pd.Timestamp("2026-04-01", tz="UTC"),
            polymarket_fetch=fpoly,
            kalshi_fetch=fkalshi,
        )
        # Expected dates: A's window then B's window, no overlap.
        expected = ["2026-01-10", "2026-02-01", "2026-02-12", "2026-02-13", "2026-03-12"]
        assert list(out.index.strftime("%Y-%m-%d")) == expected
        # Prices come from the right segments
        assert out.loc[pd.Timestamp("2026-02-12", tz="UTC"), "price"] == 0.45  # A
        assert out.loc[pd.Timestamp("2026-02-13", tz="UTC"), "price"] == 0.42  # B

    def test_mixed_sources(self) -> None:
        """One segment from Polymarket, next from Kalshi."""
        seg_poly = _make_df({"2026-01-10": 0.10, "2026-02-12": 0.40})
        seg_kalshi = _make_df({"2026-02-13": 0.42, "2026-03-12": 0.50})

        def fpoly(client, *, slug, start, end):
            assert slug == "poly-A"
            return seg_poly.copy()

        def fkalshi(client, *, market_ticker, start, end):
            assert market_ticker == "KXX-26MAR"
            return seg_kalshi.copy()

        segs = _segs(
            ("polymarket", "poly-A", "2026-02-12"),
            ("kalshi", "KXX-26MAR", "2026-03-12"),
        )
        out = fetch_chained_history(
            segs,
            poly="P",
            kalshi="K",
            start=pd.Timestamp("2026-01-01", tz="UTC"),
            end=pd.Timestamp("2026-04-01", tz="UTC"),
            polymarket_fetch=fpoly,
            kalshi_fetch=fkalshi,
        )
        assert len(out) == 4
        assert out.iloc[-1]["price"] == 0.50

    def test_empty_segment_yields_gap(self) -> None:
        """If one segment returns no bars, the chain should still succeed
        and contain only the other segments' bars."""
        seg_a = _make_df({"2026-01-10": 0.20, "2026-02-12": 0.45})
        seg_b = pd.DataFrame()  # empty
        seg_c = _make_df({"2026-03-13": 0.60, "2026-04-10": 0.70})
        fpoly, fkalshi = _stub_fetcher({"A": seg_a, "B": seg_b, "C": seg_c})
        segs = _segs(
            ("polymarket", "A", "2026-02-12"),
            ("polymarket", "B", "2026-03-12"),
            ("polymarket", "C", "2026-04-10"),
        )
        out = fetch_chained_history(
            segs,
            poly="P",
            kalshi=None,
            start=pd.Timestamp("2026-01-01", tz="UTC"),
            end=pd.Timestamp("2026-05-01", tz="UTC"),
            polymarket_fetch=fpoly,
            kalshi_fetch=fkalshi,
        )
        assert list(out.index.strftime("%Y-%m-%d")) == [
            "2026-01-10",
            "2026-02-12",
            "2026-03-13",
            "2026-04-10",
        ]

    def test_all_empty_returns_empty(self) -> None:
        fpoly, fkalshi = _stub_fetcher({"A": pd.DataFrame(), "B": pd.DataFrame()})
        segs = _segs(
            ("polymarket", "A", "2026-02-12"),
            ("polymarket", "B", "2026-03-12"),
        )
        out = fetch_chained_history(
            segs,
            poly="P",
            kalshi=None,
            start=pd.Timestamp("2026-01-01", tz="UTC"),
            end=pd.Timestamp("2026-04-01", tz="UTC"),
            polymarket_fetch=fpoly,
            kalshi_fetch=fkalshi,
        )
        assert out.empty

    def test_user_window_clamps_output(self) -> None:
        seg = _make_df(
            {
                "2026-01-10": 0.10,
                "2026-02-01": 0.20,  # inside user window
                "2026-02-12": 0.30,  # inside
            }
        )
        fpoly, fkalshi = _stub_fetcher({"A": seg})
        segs = _segs(("polymarket", "A", "2026-02-12"))
        out = fetch_chained_history(
            segs,
            poly="P",
            kalshi=None,
            start=pd.Timestamp("2026-01-25", tz="UTC"),  # cuts off Jan 10
            end=pd.Timestamp("2026-02-15", tz="UTC"),
            polymarket_fetch=fpoly,
            kalshi_fetch=fkalshi,
        )
        assert list(out.index.strftime("%Y-%m-%d")) == ["2026-02-01", "2026-02-12"]

    def test_missing_client_raises(self) -> None:
        segs = _segs(("kalshi", "K", "2026-02-12"))
        with pytest.raises(RuntimeError, match="needs Kalshi client"):
            fetch_chained_history(
                segs,
                poly="P",
                kalshi=None,
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-03-01", tz="UTC"),
            )

    def test_empty_segments_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one segment"):
            fetch_chained_history(
                (),
                poly=None,
                kalshi=None,
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-03-01", tz="UTC"),
            )

    def test_unsorted_segments_raises(self) -> None:
        # We construct the bad list directly (FactorConfig would reject it,
        # but the fetcher must defend itself if called by other code).
        segs = (
            ChainSegment(source="kalshi", slug="A", end=date(2026, 3, 12)),
            ChainSegment(source="kalshi", slug="B", end=date(2026, 2, 12)),
        )
        with pytest.raises(ValueError, match="strictly ascending"):
            fetch_chained_history(
                segs,
                poly=None,
                kalshi="K",
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-04-01", tz="UTC"),
            )


# ──────────────────────────── signature ───────────────────────────────


class TestDispatcherIntegration:
    """``_cached_factor_history`` must dispatch ``source=chain`` to
    ``fetch_chained_history`` and use a cache key that includes the segments
    signature so two distinct chains don't collide."""

    def test_chain_dispatch_concatenates_segments(self, monkeypatch) -> None:
        import pfm.main as main_mod
        from pfm.cache import NullCache
        from pfm.config import Settings
        from pfm.factors import CHAIN_SOURCE

        seg_a = _make_df({"2026-01-10": 0.20, "2026-02-12": 0.45})
        seg_b = _make_df({"2026-02-13": 0.42, "2026-03-12": 0.50})

        def fake_chain(segments, *, poly, kalshi, start, end):
            # Verify dispatcher passes both clients through.
            assert poly is not None
            assert len(segments) == 2
            return pd.concat([seg_a, seg_b])

        monkeypatch.setattr(main_mod, "fetch_chained_history", fake_chain)
        # Make sure we're not accidentally dispatching to the single-source paths.
        monkeypatch.setattr(
            main_mod,
            "fetch_factor_history",
            lambda *a, **k: pytest.fail("polymarket fetch called for chain"),
        )
        monkeypatch.setattr(
            main_mod,
            "fetch_kalshi_history",
            lambda *a, **k: pytest.fail("kalshi fetch called for chain"),
        )

        fc = FactorConfig(
            id="cpi_chain",
            name="cpi",
            slug="cpi_chain",
            source=CHAIN_SOURCE,
            description="d",
            segments=_segs(
                ("kalshi", "A", "2026-02-12"),
                ("kalshi", "B", "2026-03-12"),
            ),
        )

        cache = NullCache()
        settings = Settings(redis_url="memory://")  # NullCache anyway

        out = main_mod._cached_factor_history(
            fc,
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-04-01", tz="UTC"),
            poly="dummy_poly",
            cache=cache,
            settings=settings,
        )
        assert len(out) == 4
        assert out.iloc[-1]["price"] == 0.50

    def test_chain_cache_key_distinguishes_compositions(self, monkeypatch) -> None:
        """Two chains with same id but different segments must not collide."""
        import pfm.main as main_mod
        from pfm.cache import NullCache
        from pfm.config import Settings
        from pfm.factors import CHAIN_SOURCE

        calls: list[tuple] = []

        def fake_chain(segments, *, poly, kalshi, start, end):
            calls.append(tuple((s.source, s.slug, s.end) for s in segments))
            return _make_df({"2026-02-15": 0.30})

        monkeypatch.setattr(main_mod, "fetch_chained_history", fake_chain)

        cache = NullCache()
        settings = Settings(redis_url="memory://")

        fc1 = FactorConfig(
            id="x",
            name="x",
            slug="x",
            source=CHAIN_SOURCE,
            description="d",
            segments=_segs(("kalshi", "A", "2026-02-12")),
        )
        fc2 = FactorConfig(
            id="x",
            name="x",
            slug="x",
            source=CHAIN_SOURCE,
            description="d",
            segments=_segs(("kalshi", "B", "2026-02-12")),
        )

        for fc in (fc1, fc2):
            main_mod._cached_factor_history(
                fc,
                pd.Timestamp("2026-01-01", tz="UTC"),
                pd.Timestamp("2026-03-01", tz="UTC"),
                poly="dummy",
                cache=cache,
                settings=settings,
            )

        # Both chains were fetched (different segment compositions).
        assert len(calls) == 2
        assert calls[0] != calls[1]


class TestSegmentsSignature:
    def test_stable_for_same_segments(self) -> None:
        segs = _segs(
            ("kalshi", "A", "2026-02-12"),
            ("polymarket", "B-slug", "2026-03-12"),
        )
        assert segments_signature(segs) == segments_signature(segs)

    def test_distinguishes_different_lists(self) -> None:
        a = _segs(("kalshi", "A", "2026-02-12"))
        b = _segs(("kalshi", "B", "2026-02-12"))
        assert segments_signature(a) != segments_signature(b)

    def test_includes_source_slug_end(self) -> None:
        segs = _segs(
            ("kalshi", "A", "2026-02-12"),
            ("polymarket", "B", "2026-03-12"),
        )
        sig = segments_signature(segs)
        assert "kalshi|A|2026-02-12" in sig
        assert "polymarket|B|2026-03-12" in sig


# ─────────────────────────────── loader ───────────────────────────────


class TestLoader:
    def test_loads_chain_factor_from_yaml(self, tmp_path: Path) -> None:
        body = """\
factors:
  - id: cpi_yoy_3pct_chain
    name: "Next-month CPI YoY > 3% (chained)"
    slug: cpi_yoy_3pct_chain
    source: chain
    theme: macro
    description: |
        Chained next-month CPI YoY > 3% probability.
    segments:
      - source: kalshi
        slug: KXCPIYOY-26FEB-T3.0
        end: 2026-02-12
      - source: kalshi
        slug: KXCPIYOY-26MAR-T3.0
        end: 2026-03-12
"""
        f = tmp_path / "factors.yml"
        f.write_text(body, encoding="utf-8")
        out = load_factors(f)
        fc = out["cpi_yoy_3pct_chain"]
        assert fc.is_chained
        assert len(fc.segments) == 2
        assert fc.segments[0].slug == "KXCPIYOY-26FEB-T3.0"
        assert fc.segments[1].end == date(2026, 3, 12)

    def test_chain_without_segments_rejected_at_load(self, tmp_path: Path) -> None:
        body = """\
factors:
  - id: bad
    name: "x"
    slug: bad
    source: chain
    description: bad
"""
        f = tmp_path / "factors.yml"
        f.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match="segments must be a non-empty list"):
            load_factors(f)

    def test_chain_segment_missing_field_rejected(self, tmp_path: Path) -> None:
        body = """\
factors:
  - id: bad
    name: x
    slug: bad
    source: chain
    description: bad
    segments:
      - source: kalshi
        end: 2026-02-12
"""
        f = tmp_path / "factors.yml"
        f.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match=r"segment\[0\] missing keys"):
            load_factors(f)

    def test_existing_single_source_still_loads(self, tmp_path: Path) -> None:
        body = """\
factors:
  - id: foo
    name: Foo
    slug: foo-slug
    source: polymarket
    description: foo
"""
        f = tmp_path / "factors.yml"
        f.write_text(body, encoding="utf-8")
        out = load_factors(f)
        assert out["foo"].is_chained is False
        assert out["foo"].segments == ()
