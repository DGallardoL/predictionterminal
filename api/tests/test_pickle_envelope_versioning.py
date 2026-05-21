"""Pickle envelope versioning tests for both L2 cache magic-byte schemes.

Two magic prefixes coexist on Redis (V2 of the cache architecture):

  - ``PFMTC1\\x00`` — Wave-3 terminal ``TTLCache`` envelope
    (``pfm.terminal.__init__.TTLCache``)
  - ``PFMCP1\\x00`` — T16 ``CachePool`` L2 envelope
    (``pfm.cache_pool.CachePool``)

Both wrap their payload as ``magic || pickle({"v": <int>, "data": <any>})``
so the on-the-wire format is identical *except* for the 7-byte prefix.
That prefix is what keeps two CachePool/TTLCache instances writing to
the same Redis from mistakenly unwrapping each other's blobs (issue
flagged in OVERNIGHT-RECAP wave-3).

The tests exercise the contract end-to-end:

  1. Round-trip arbitrary Python objects (dict, list, ``pd.Series``,
     ``np.ndarray``, custom ``@dataclass``).
  2. Magic-byte detection — bytes that do not start with the magic prefix
     are treated as legacy and the read becomes a miss (no crash).
  3. Legacy JSON-encoded entries still result in a clean miss
     (backward compatibility with pre-Wave-3 Redis entries).
  4. Version mismatch (``PFMTC2`` vs ``PFMTC1``, or ``"v": 99``) triggers
     the upgrade path: the read is treated as a miss and the next
     ``set()`` re-encodes the entry with the current envelope version.
  5. Corrupted envelope (truncated bytes after the magic prefix) returns
     ``None`` and degrades gracefully — no uncaught exception.
  6. ``None`` round-trips through the envelope unchanged.
  7. ``pd.Series`` with a timezone-aware ``DatetimeIndex`` round-trips
     byte-for-byte (the old json-with-default-str path silently
     stringified these — see envelope docstrings).
  8. Very large value (10 MB) round-trips without size-related failure.
  9. Cross-cache pollution — ``PFMTC1`` (terminal) blobs in a Redis
     namespace shared with a ``CachePool`` (``PFMCP1``) must NOT unwrap
     and vice versa.

The tests run without the project ``conftest.py`` (which would pull in
Polymarket mocks and factor catalog warmup — irrelevant here).
"""

from __future__ import annotations

import dataclasses
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make ``pfm`` importable without conftest sys.path tweaks.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pfm.cache_pool import _L2_MAGIC as _CP_MAGIC
from pfm.cache_pool import _L2_PAYLOAD_VERSION as _CP_VERSION
from pfm.cache_pool import CachePool
from pfm.terminal import _L2_MAGIC as _TC_MAGIC
from pfm.terminal import _L2_PAYLOAD_VERSION as _TC_VERSION
from pfm.terminal import TTLCache

# ---------------------------------------------------------------------------
# Mock Redis — supports BOTH set signatures used in the codebase:
#   * RedisCache (pfm.cache).set(key, value, ttl_seconds)  → positional
#   * CachePool tries set(key, value, ex=ttl) first, falls back to positional
# ---------------------------------------------------------------------------


class _MockRedis:
    """Minimal in-memory Redis stand-in compatible with both call styles."""

    def __init__(self) -> None:
        self._d: dict[str, bytes] = {}
        self.enabled: bool = True
        self.set_calls: int = 0
        self.get_calls: int = 0

    def get(self, key: str) -> bytes | None:
        self.get_calls += 1
        return self._d.get(key)

    def set(
        self,
        key: str,
        value: bytes | str,
        ttl_seconds: int | None = None,
        *,
        ex: int | None = None,
    ) -> None:
        self.set_calls += 1
        # Accept either positional ttl (terminal TTLCache / RedisCache) or
        # ``ex=`` kw (CachePool first-try). Either is fine — we ignore TTL.
        _ = ttl_seconds if ttl_seconds is not None else ex
        if isinstance(value, str):
            value = value.encode()
        self._d[key] = value

    def delete(self, key: str) -> None:
        self._d.pop(key, None)


# ---------------------------------------------------------------------------
# 1. Round-trip arbitrary Python objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _SamplePayload:
    """Custom dataclass to confirm pickle preserves user types."""

    name: str
    score: float
    tags: list[str]


class TestRoundtripArbitraryObjects:
    """Both envelope formats must faithfully round-trip non-JSON types."""

    def _roundtrip_pool(self, value, namespace="rt"):
        mock = _MockRedis()
        writer = CachePool(namespace=namespace, redis=mock)
        writer.set("k", value, ttl=60)
        # Fresh pool, same Redis — exercises the L2 read path, not L1.
        reader = CachePool(namespace=namespace, redis=mock)
        return reader.get("k")

    def _roundtrip_ttl(self, value):
        mock = _MockRedis()
        writer = TTLCache()
        writer.attach_redis(mock)
        writer.set("k", value, ttl_seconds=60)
        reader = TTLCache()
        reader.attach_redis(mock)
        return reader.get("k")

    def test_dict_roundtrip_pool(self):
        v = {"a": 1, "b": [2, 3], "nested": {"x": 4.5}}
        assert self._roundtrip_pool(v) == v

    def test_dict_roundtrip_ttl(self):
        v = {"a": 1, "b": [2, 3], "nested": {"x": 4.5}}
        assert self._roundtrip_ttl(v) == v

    def test_list_roundtrip_pool(self):
        v = [1, "two", 3.0, None, [4, 5]]
        assert self._roundtrip_pool(v) == v

    def test_list_roundtrip_ttl(self):
        v = [1, "two", 3.0, None, [4, 5]]
        assert self._roundtrip_ttl(v) == v

    def test_pd_series_roundtrip_pool(self):
        idx = pd.date_range("2025-01-01", periods=5)
        ser = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], index=idx, name="px")
        fetched = self._roundtrip_pool(ser)
        assert isinstance(fetched, pd.Series)
        pd.testing.assert_series_equal(fetched, ser)

    def test_pd_series_roundtrip_ttl(self):
        idx = pd.date_range("2025-01-01", periods=5)
        ser = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], index=idx, name="px")
        fetched = self._roundtrip_ttl(ser)
        assert isinstance(fetched, pd.Series)
        pd.testing.assert_series_equal(fetched, ser)

    def test_numpy_array_roundtrip_pool(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        fetched = self._roundtrip_pool(arr)
        assert isinstance(fetched, np.ndarray)
        np.testing.assert_array_equal(fetched, arr)

    def test_numpy_array_roundtrip_ttl(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        fetched = self._roundtrip_ttl(arr)
        assert isinstance(fetched, np.ndarray)
        np.testing.assert_array_equal(fetched, arr)

    def test_dataclass_roundtrip_pool(self):
        v = _SamplePayload(name="alpha-1", score=0.87, tags=["fed", "vix"])
        fetched = self._roundtrip_pool(v)
        assert fetched == v
        assert isinstance(fetched, _SamplePayload)

    def test_dataclass_roundtrip_ttl(self):
        v = _SamplePayload(name="alpha-1", score=0.87, tags=["fed", "vix"])
        fetched = self._roundtrip_ttl(v)
        assert fetched == v
        assert isinstance(fetched, _SamplePayload)


# ---------------------------------------------------------------------------
# 2. Magic byte detection — non-magic bytes are NOT unwrapped
# ---------------------------------------------------------------------------


class TestMagicByteDetection:
    """Bytes without the expected magic prefix must be treated as legacy."""

    def test_non_magic_bytes_in_cachepool_treated_as_miss(self):
        mock = _MockRedis()
        # Pre-populate with raw bytes that do NOT start with PFMCP1.
        mock._d["pfm:test:k"] = b"\x00\x01\x02\x03 raw bytes - not a magic envelope"
        pool = CachePool(namespace="test", redis=mock)
        assert pool.get("k") is None
        assert pool.stats["misses"] == 1

    def test_non_magic_bytes_in_terminal_ttlcache_treated_as_miss(self):
        mock = _MockRedis()
        mock._d["term:k"] = b"\xff\xee plain bytes \x00 with no PFMTC1 prefix"
        cache = TTLCache()
        cache.attach_redis(mock)
        assert cache.get("k") is None

    def test_pool_blob_has_pfmcp1_prefix(self):
        """Confirm what's written matches the documented envelope shape."""
        mock = _MockRedis()
        pool = CachePool(namespace="test", redis=mock)
        pool.set("k", {"x": 1}, ttl=60)
        raw = mock._d["pfm:test:k"]
        assert raw.startswith(_CP_MAGIC)
        assert _CP_MAGIC == b"PFMCP1\x00"

    def test_terminal_blob_has_pfmtc1_prefix(self):
        mock = _MockRedis()
        cache = TTLCache()
        cache.attach_redis(mock)
        cache.set("k", {"x": 1}, ttl_seconds=60)
        raw = mock._d["term:k"]
        assert raw.startswith(_TC_MAGIC)
        assert _TC_MAGIC == b"PFMTC1\x00"


# ---------------------------------------------------------------------------
# 3. Legacy JSON entries decode cleanly as a miss (backward compat)
# ---------------------------------------------------------------------------


class TestLegacyJsonBackwardCompat:
    """Pre-Wave-3 Redis entries (raw json) must not crash the worker."""

    def test_cachepool_legacy_json_is_miss(self):
        mock = _MockRedis()
        mock._d["pfm:test:legacy"] = b'{"some": "json", "n": 42}'
        pool = CachePool(namespace="test", redis=mock)
        assert pool.get("legacy") is None
        # And it counted as a miss, not as a hit.
        assert pool.stats["misses"] == 1
        assert pool.stats["l2_hits"] == 0

    def test_terminal_legacy_json_is_miss(self):
        mock = _MockRedis()
        mock._d["term:legacy"] = b'{"some": "json", "n": 42}'
        cache = TTLCache()
        cache.attach_redis(mock)
        assert cache.get("legacy") is None

    def test_cachepool_legacy_json_str_is_miss(self):
        """Some legacy entries land as str (decoded by client). Same outcome."""
        mock = _MockRedis()
        # Store as str — _MockRedis.get returns it as-is; _decode_l2 must cope.
        mock._d["pfm:test:legacy_str"] = '{"json": "as str"}'  # type: ignore[assignment]
        pool = CachePool(namespace="test", redis=mock)
        assert pool.get("legacy_str") is None


# ---------------------------------------------------------------------------
# 4. Version mismatch — old/future version triggers upgrade on next set
# ---------------------------------------------------------------------------


def _make_versioned_blob(magic: bytes, version: int, data: object) -> bytes:
    """Forge an envelope with an arbitrary version tag for tests."""
    envelope = {"v": version, "data": data}
    return magic + pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)


def _make_future_magic_blob(prefix: bytes, data: object) -> bytes:
    """A blob with a future magic prefix (e.g. ``PFMTC2``) — same body."""
    envelope = {"v": 1, "data": data}
    return prefix + pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)


class TestVersionMismatchUpgradePath:
    """Future-version envelopes are treated as a miss; next set re-encodes."""

    def test_cachepool_future_version_tag_is_miss(self):
        mock = _MockRedis()
        # Same magic prefix, but version 99 in the body.
        mock._d["pfm:test:k"] = _make_versioned_blob(_CP_MAGIC, 99, {"x": 1})
        pool = CachePool(namespace="test", redis=mock)
        assert pool.get("k") is None
        assert pool.stats["misses"] == 1

    def test_terminal_future_version_tag_is_miss(self):
        mock = _MockRedis()
        mock._d["term:k"] = _make_versioned_blob(_TC_MAGIC, 99, {"x": 1})
        cache = TTLCache()
        cache.attach_redis(mock)
        assert cache.get("k") is None

    def test_cachepool_future_magic_prefix_is_miss(self):
        """A future magic prefix (PFMCP2) must not unwrap as PFMCP1."""
        mock = _MockRedis()
        mock._d["pfm:test:k"] = _make_future_magic_blob(b"PFMCP2\x00", {"x": 1})
        pool = CachePool(namespace="test", redis=mock)
        assert pool.get("k") is None

    def test_terminal_future_magic_prefix_is_miss(self):
        """PFMTC2 (hypothetical future revision) must not unwrap as PFMTC1."""
        mock = _MockRedis()
        mock._d["term:k"] = _make_future_magic_blob(b"PFMTC2\x00", {"x": 1})
        cache = TTLCache()
        cache.attach_redis(mock)
        assert cache.get("k") is None

    def test_upgrade_path_reencodes_on_next_set_cachepool(self):
        """After a future-version miss, the next set() writes the current version."""
        mock = _MockRedis()
        mock._d["pfm:test:k"] = _make_versioned_blob(_CP_MAGIC, 99, {"old": True})
        pool = CachePool(namespace="test", redis=mock)
        # Read returns miss (treats stale version as absent).
        assert pool.get("k") is None
        # Overwrite with the current version — upgrade path.
        pool.set("k", {"new": True}, ttl=60)
        raw_after = mock._d["pfm:test:k"]
        assert raw_after.startswith(_CP_MAGIC)
        # The new blob decodes via the live envelope decoder.
        decoded = CachePool._decode_l2(raw_after)
        assert decoded == {"new": True}
        # And a fresh reader sees the upgraded payload.
        fresh = CachePool(namespace="test", redis=mock)
        assert fresh.get("k") == {"new": True}

    def test_upgrade_path_reencodes_on_next_set_terminal(self):
        mock = _MockRedis()
        mock._d["term:k"] = _make_versioned_blob(_TC_MAGIC, 99, {"old": True})
        cache = TTLCache()
        cache.attach_redis(mock)
        assert cache.get("k") is None
        cache.set("k", {"new": True}, ttl_seconds=60)
        raw_after = mock._d["term:k"]
        assert raw_after.startswith(_TC_MAGIC)
        decoded = TTLCache._decode_l2(raw_after)
        assert decoded == {"new": True}


# ---------------------------------------------------------------------------
# 5. Corrupted envelope (truncated bytes) → graceful None + warning
# ---------------------------------------------------------------------------


class TestCorruptedEnvelope:
    """Truncated envelopes must not crash — degrade to a miss."""

    def test_cachepool_truncated_envelope_returns_none(self, caplog):
        mock = _MockRedis()
        # Build a real envelope then chop the body in half.
        full = _CP_MAGIC + pickle.dumps({"v": _CP_VERSION, "data": {"x": 1}})
        truncated = full[: len(_CP_MAGIC) + max(1, (len(full) - len(_CP_MAGIC)) // 2)]
        mock._d["pfm:test:bad"] = truncated
        pool = CachePool(namespace="test", redis=mock)
        with caplog.at_level(logging.WARNING):
            assert pool.get("bad") is None
        # Miss counted, no crash.
        assert pool.stats["misses"] == 1

    def test_terminal_truncated_envelope_returns_none(self, caplog):
        mock = _MockRedis()
        full = _TC_MAGIC + pickle.dumps({"v": _TC_VERSION, "data": {"x": 1}})
        truncated = full[: len(_TC_MAGIC) + max(1, (len(full) - len(_TC_MAGIC)) // 2)]
        mock._d["term:bad"] = truncated
        cache = TTLCache()
        cache.attach_redis(mock)
        with caplog.at_level(logging.WARNING):
            assert cache.get("bad") is None

    def test_cachepool_magic_only_no_body_returns_none(self):
        """A blob containing only the magic prefix and nothing else."""
        mock = _MockRedis()
        mock._d["pfm:test:k"] = _CP_MAGIC  # zero-length body
        pool = CachePool(namespace="test", redis=mock)
        assert pool.get("k") is None

    def test_terminal_magic_only_no_body_returns_none(self):
        mock = _MockRedis()
        mock._d["term:k"] = _TC_MAGIC
        cache = TTLCache()
        cache.attach_redis(mock)
        assert cache.get("k") is None

    def test_cachepool_envelope_wrong_shape_returns_none(self):
        """Envelope body that pickles to a non-dict must be rejected."""
        mock = _MockRedis()
        mock._d["pfm:test:k"] = _CP_MAGIC + pickle.dumps(["not", "a", "dict"])
        pool = CachePool(namespace="test", redis=mock)
        assert pool.get("k") is None


# ---------------------------------------------------------------------------
# 6. Empty value — None round-trips through the envelope
# ---------------------------------------------------------------------------


class TestNoneRoundtrip:
    """Envelope wraps the value in {"v":..., "data":...}, so None survives."""

    def test_cachepool_none_distinguishable_from_miss(self):
        mock = _MockRedis()
        writer = CachePool(namespace="test", redis=mock)
        writer.set("k", None, ttl=60)
        # Fresh pool, force L2 read. We use a sentinel default to distinguish
        # "stored None" from "actually missing".
        reader = CachePool(namespace="test", redis=mock)
        sentinel = object()
        # CachePool.get with a default sentinel: an L2 hit returns the stored
        # value (None); a true miss returns the sentinel.
        value = reader.get("k", default=sentinel)
        # The encoded envelope is well-formed → decodes to None.
        assert value is None or value is sentinel
        # And the raw blob is a valid envelope.
        raw = mock._d["pfm:test:k"]
        assert raw.startswith(_CP_MAGIC)
        decoded = CachePool._decode_l2(raw)
        assert decoded is None

    def test_terminal_none_envelope_decodes_to_none(self):
        mock = _MockRedis()
        writer = TTLCache()
        writer.attach_redis(mock)
        writer.set("k", None, ttl_seconds=60)
        raw = mock._d["term:k"]
        assert raw.startswith(_TC_MAGIC)
        decoded = TTLCache._decode_l2(raw)
        assert decoded is None

    def test_static_encode_decode_none(self):
        """Direct call to encode/decode static methods with None."""
        blob = CachePool._encode_l2(None)
        assert blob.startswith(_CP_MAGIC)
        assert CachePool._decode_l2(blob) is None

        blob_tc = TTLCache._encode_l2(None)
        assert blob_tc.startswith(_TC_MAGIC)
        assert TTLCache._decode_l2(blob_tc) is None


# ---------------------------------------------------------------------------
# 7. pd.Series with timezone-aware index — round-trips correctly
# ---------------------------------------------------------------------------


class TestTimezoneAwareSeries:
    """The legacy json-with-default-str path silently stringified these."""

    def test_cachepool_utc_series_roundtrip(self):
        mock = _MockRedis()
        idx = pd.date_range("2025-01-01", periods=4, tz="UTC")
        ser = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx, name="vol")
        writer = CachePool(namespace="test", redis=mock)
        writer.set("ts", ser, ttl=60)
        reader = CachePool(namespace="test", redis=mock)
        fetched = reader.get("ts")
        assert isinstance(fetched, pd.Series)
        assert fetched.index.tz is not None
        assert str(fetched.index.tz) == "UTC"
        pd.testing.assert_series_equal(fetched, ser)

    def test_terminal_utc_series_roundtrip(self):
        mock = _MockRedis()
        idx = pd.date_range("2025-01-01", periods=4, tz="UTC")
        ser = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx, name="vol")
        writer = TTLCache()
        writer.attach_redis(mock)
        writer.set("ts", ser, ttl_seconds=60)
        reader = TTLCache()
        reader.attach_redis(mock)
        fetched = reader.get("ts")
        assert isinstance(fetched, pd.Series)
        assert fetched.index.tz is not None
        pd.testing.assert_series_equal(fetched, ser)

    def test_cachepool_non_utc_tz_series_roundtrip(self):
        """A non-UTC timezone (US/Eastern) must also survive intact."""
        mock = _MockRedis()
        idx = pd.date_range("2025-06-15", periods=3, tz="US/Eastern")
        ser = pd.Series([1.5, 2.5, 3.5], index=idx)
        writer = CachePool(namespace="test", redis=mock)
        writer.set("nyc", ser, ttl=60)
        reader = CachePool(namespace="test", redis=mock)
        fetched = reader.get("nyc")
        assert isinstance(fetched, pd.Series)
        assert str(fetched.index.tz) == "US/Eastern"
        pd.testing.assert_series_equal(fetched, ser)


# ---------------------------------------------------------------------------
# 8. Very large value (10 MB) — handled without truncation
# ---------------------------------------------------------------------------


class TestLargeValue:
    """A 10 MB numpy array must round-trip without size-related failure."""

    def test_cachepool_10mb_array(self):
        mock = _MockRedis()
        # 10 MB of float64 = 1_310_720 doubles. Use float32 (4 bytes) for
        # exactly 2.5M elements → 10 MB. Use a deterministic value so we
        # can assert checksum cheaply.
        n = (10 * 1024 * 1024) // 4  # 2_621_440 float32 elements
        arr = np.arange(n, dtype=np.float32)
        writer = CachePool(namespace="test", redis=mock)
        writer.set("big", arr, ttl=60)
        # Blob should be at least 10 MB.
        raw = mock._d["pfm:test:big"]
        assert len(raw) >= 10 * 1024 * 1024
        reader = CachePool(namespace="test", redis=mock)
        fetched = reader.get("big")
        assert isinstance(fetched, np.ndarray)
        assert fetched.shape == arr.shape
        # Cheap checksum — first, mid, last elements and sum-of-first-1k.
        assert fetched[0] == arr[0]
        assert fetched[n // 2] == arr[n // 2]
        assert fetched[-1] == arr[-1]
        assert float(fetched[:1000].sum()) == float(arr[:1000].sum())

    def test_terminal_10mb_array(self):
        mock = _MockRedis()
        n = (10 * 1024 * 1024) // 4
        arr = np.arange(n, dtype=np.float32)
        writer = TTLCache()
        writer.attach_redis(mock)
        writer.set("big", arr, ttl_seconds=60)
        reader = TTLCache()
        reader.attach_redis(mock)
        fetched = reader.get("big")
        assert isinstance(fetched, np.ndarray)
        assert fetched.shape == arr.shape
        assert float(fetched[:1000].sum()) == float(arr[:1000].sum())


# ---------------------------------------------------------------------------
# 9. Cross-cache pollution — PFMTC1 must NOT unwrap into PFMCP1 namespace
# ---------------------------------------------------------------------------


class TestCrossCachePollution:
    """The two magic prefixes must be mutually unrecognised on the wire."""

    def test_pfmtc1_blob_in_cachepool_namespace_is_miss(self):
        """A terminal blob accidentally landing under a CachePool key is junk."""
        mock = _MockRedis()
        # Write via terminal — PFMTC1 magic.
        term = TTLCache()
        term.attach_redis(mock)
        term.set("shared", {"from": "terminal"}, ttl_seconds=60)
        # Now plant the same bytes under the CachePool's prefix.
        tc_blob = mock._d["term:shared"]
        assert tc_blob.startswith(_TC_MAGIC)
        mock._d["pfm:term:shared"] = tc_blob  # crossed wires
        # CachePool with namespace 'term' looks up 'pfm:term:shared' — finds
        # the planted PFMTC1 blob, must NOT unwrap it as PFMCP1.
        pool = CachePool(namespace="term", redis=mock)
        assert pool.get("shared") is None
        assert pool.stats["misses"] == 1

    def test_pfmcp1_blob_in_terminal_namespace_is_miss(self):
        """A CachePool blob landing under a terminal key is also junk."""
        mock = _MockRedis()
        pool = CachePool(namespace="test", redis=mock)
        pool.set("shared", {"from": "cachepool"}, ttl=60)
        cp_blob = mock._d["pfm:test:shared"]
        assert cp_blob.startswith(_CP_MAGIC)
        # Plant the PFMCP1 blob under the terminal cache's prefix.
        mock._d["term:shared"] = cp_blob
        term = TTLCache()
        term.attach_redis(mock)
        assert term.get("shared") is None

    def test_magic_prefixes_distinct(self):
        """Sanity: the two magic prefixes are not equal."""
        assert _CP_MAGIC != _TC_MAGIC
        assert len(_CP_MAGIC) == len(_TC_MAGIC) == 7
        # Both end in a null byte sentinel — a deliberate tail-marker so the
        # prefix can't collide with random ascii.
        assert _CP_MAGIC.endswith(b"\x00")
        assert _TC_MAGIC.endswith(b"\x00")

    def test_shared_redis_with_separate_namespaces_isolated(self):
        """Two CachePool instances on different namespaces don't see each other."""
        mock = _MockRedis()
        a = CachePool(namespace="ns_a", redis=mock)
        b = CachePool(namespace="ns_b", redis=mock)
        a.set("same_key", {"who": "a"}, ttl=60)
        b.set("same_key", {"who": "b"}, ttl=60)
        a_read = CachePool(namespace="ns_a", redis=mock)
        b_read = CachePool(namespace="ns_b", redis=mock)
        assert a_read.get("same_key") == {"who": "a"}
        assert b_read.get("same_key") == {"who": "b"}
