"""Tests for the Redis cache wrapper. The Redis client itself is mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import redis

from pfm.cache import NullCache, RedisCache


def test_null_cache_is_disabled() -> None:
    c = NullCache()
    assert c.enabled is False
    assert c.get("k") is None
    c.set("k", b"v", ttl_seconds=10)  # no exception


def test_redis_cache_disabled_when_ping_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = MagicMock()
    fake_client.ping.side_effect = redis.ConnectionError("nope")
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, *a, **k: fake_client))

    c = RedisCache("redis://nowhere:6379")
    assert c.enabled is False
    assert c.get("k") is None


def test_redis_cache_get_set_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = MagicMock()
    fake_client.ping.return_value = True
    fake_client.get.return_value = b"hello"
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, *a, **k: fake_client))

    c = RedisCache("redis://localhost:6379")
    assert c.enabled is True
    assert c.get("k") == b"hello"
    c.set("k", b"v", ttl_seconds=60)
    fake_client.set.assert_called_once_with("k", b"v", ex=60)


def test_redis_get_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = MagicMock()
    fake_client.ping.return_value = True
    fake_client.get.side_effect = redis.RedisError("boom")
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, *a, **k: fake_client))

    c = RedisCache("redis://localhost:6379")
    assert c.get("k") is None  # logged, but no exception
