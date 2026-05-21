"""SQLite-backed persistence for API keys + per-key request counters.

Two tables:

    CREATE TABLE api_keys (
        key                   TEXT PRIMARY KEY,
        user_id               TEXT NOT NULL,
        tier                  TEXT NOT NULL,
        created_at            REAL NOT NULL,
        last_used_at          REAL,
        enabled               INTEGER NOT NULL DEFAULT 1,
        rate_limit_per_minute INTEGER NOT NULL,
        daily_quota           INTEGER NOT NULL,
        expires_at            REAL                       -- NULL = never expires
    );
    CREATE INDEX idx_api_keys_user ON api_keys(user_id);

    CREATE TABLE request_counts (
        key      TEXT NOT NULL,            -- API key (or 'anon:<ip>')
        bucket   TEXT NOT NULL,            -- 'min:YYYYMMDDHHMM' | 'day:YYYYMMDD'
        endpoint TEXT NOT NULL,            -- request.url.path (or '*' for global)
        count    INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (key, bucket, endpoint)
    );
    CREATE INDEX idx_rc_bucket ON request_counts(bucket);

The ``request_counts`` table is the rate limiter's storage; we lean on the
``(key, bucket, endpoint)`` PK and ``ON CONFLICT … DO UPDATE`` to do
atomic increment-and-read in a single statement.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import UTC, datetime
from typing import Any

from pfm.auth.models import APIKey, Tier


def _bucket_minute(now: float | None = None) -> str:
    t = datetime.fromtimestamp(now or time.time(), tz=UTC)
    return f"min:{t:%Y%m%d%H%M}"


def _bucket_day(now: float | None = None) -> str:
    t = datetime.fromtimestamp(now or time.time(), tz=UTC)
    return f"day:{t:%Y%m%d}"


def _next_minute_reset(now: float | None = None) -> float:
    t = now or time.time()
    return float(int(t // 60) * 60 + 60)


def _next_day_reset(now: float | None = None) -> float:
    t = datetime.fromtimestamp(now or time.time(), tz=UTC)
    next_day = t.replace(hour=0, minute=0, second=0, microsecond=0)
    return next_day.timestamp() + 86_400


class APIKeyStore:
    """Thin sqlite3 wrapper. Same memory-DB trick as :class:`AlertStore`."""

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS api_keys (
        key                   TEXT PRIMARY KEY,
        user_id               TEXT NOT NULL,
        tier                  TEXT NOT NULL,
        created_at            REAL NOT NULL,
        last_used_at          REAL,
        enabled               INTEGER NOT NULL DEFAULT 1,
        rate_limit_per_minute INTEGER NOT NULL,
        daily_quota           INTEGER NOT NULL,
        expires_at            REAL
    );
    CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);

    CREATE TABLE IF NOT EXISTS request_counts (
        key      TEXT NOT NULL,
        bucket   TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        count    INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (key, bucket, endpoint)
    );
    CREATE INDEX IF NOT EXISTS idx_rc_bucket ON request_counts(bucket);

    CREATE TABLE IF NOT EXISTS demo_key_quota (
        client_ip TEXT NOT NULL,
        day       TEXT NOT NULL,
        count     INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (client_ip, day)
    );
    CREATE INDEX IF NOT EXISTS idx_demo_quota_day ON demo_key_quota(day);
    """

    def __init__(self, db_path: str = "/tmp/pfm_auth.db") -> None:
        self.db_path = db_path
        self._memory_conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        if db_path == ":memory:":
            self._memory_conn = sqlite3.connect(db_path, check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        self.init_schema()

    # ------------------------------------------------------------ low-level

    def _conn(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _close(self, c: sqlite3.Connection) -> None:
        if self._memory_conn is None:
            c.close()

    def init_schema(self) -> None:
        c = self._conn()
        try:
            c.executescript(self.SCHEMA_SQL)
            c.commit()
        finally:
            self._close(c)

    # ------------------------------------------------------------ keys

    def save_key(self, key: APIKey, expires_at: float | None = None) -> APIKey:
        c = self._conn()
        try:
            c.execute(
                """INSERT OR REPLACE INTO api_keys
                   (key, user_id, tier, created_at, last_used_at, enabled,
                    rate_limit_per_minute, daily_quota, expires_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    key.key,
                    key.user_id,
                    key.tier,
                    key.created_at.timestamp(),
                    key.last_used_at.timestamp() if key.last_used_at else None,
                    1 if key.enabled else 0,
                    int(key.rate_limit_per_minute),
                    int(key.daily_quota),
                    expires_at,
                ),
            )
            c.commit()
        finally:
            self._close(c)
        return key

    def get_key(self, key: str) -> APIKey | None:
        c = self._conn()
        try:
            row = c.execute("SELECT * FROM api_keys WHERE key=?", (key,)).fetchone()
        finally:
            self._close(c)
        if row is None:
            return None
        # Expired? Treat as gone.
        if row["expires_at"] is not None and row["expires_at"] < time.time():
            return None
        return APIKey(
            key=row["key"],
            user_id=row["user_id"],
            tier=row["tier"],
            created_at=datetime.fromtimestamp(row["created_at"], tz=UTC),
            last_used_at=(
                datetime.fromtimestamp(row["last_used_at"], tz=UTC)
                if row["last_used_at"] is not None
                else None
            ),
            enabled=bool(row["enabled"]),
            rate_limit_per_minute=int(row["rate_limit_per_minute"]),
            daily_quota=int(row["daily_quota"]),
        )

    def list_keys(self, user_id: str | None = None) -> list[APIKey]:
        sql = "SELECT * FROM api_keys"
        args: list[Any] = []
        if user_id is not None:
            sql += " WHERE user_id=?"
            args.append(user_id)
        sql += " ORDER BY created_at DESC"
        c = self._conn()
        try:
            rows = c.execute(sql, args).fetchall()
        finally:
            self._close(c)
        out: list[APIKey] = []
        now = time.time()
        for r in rows:
            if r["expires_at"] is not None and r["expires_at"] < now:
                continue
            out.append(
                APIKey(
                    key=r["key"],
                    user_id=r["user_id"],
                    tier=r["tier"],
                    created_at=datetime.fromtimestamp(r["created_at"], tz=UTC),
                    last_used_at=(
                        datetime.fromtimestamp(r["last_used_at"], tz=UTC)
                        if r["last_used_at"] is not None
                        else None
                    ),
                    enabled=bool(r["enabled"]),
                    rate_limit_per_minute=int(r["rate_limit_per_minute"]),
                    daily_quota=int(r["daily_quota"]),
                )
            )
        return out

    def revoke_key(self, key: str) -> bool:
        c = self._conn()
        try:
            cur = c.execute("UPDATE api_keys SET enabled=0 WHERE key=?", (key,))
            c.commit()
            return cur.rowcount > 0
        finally:
            self._close(c)

    def delete_key(self, key: str) -> bool:
        c = self._conn()
        try:
            cur = c.execute("DELETE FROM api_keys WHERE key=?", (key,))
            c.commit()
            return cur.rowcount > 0
        finally:
            self._close(c)

    def update_tier(self, key: str, tier: Tier) -> APIKey | None:
        from pfm.auth.models import TIER_DEFAULTS

        rpm, quota = TIER_DEFAULTS[tier]
        c = self._conn()
        try:
            c.execute(
                """UPDATE api_keys
                   SET tier=?, rate_limit_per_minute=?, daily_quota=?
                   WHERE key=?""",
                (tier, rpm, quota, key),
            )
            c.commit()
        finally:
            self._close(c)
        return self.get_key(key)

    def touch(self, key: str) -> None:
        """Mark ``last_used_at = now`` (best-effort)."""
        c = self._conn()
        try:
            c.execute(
                "UPDATE api_keys SET last_used_at=? WHERE key=?",
                (time.time(), key),
            )
            c.commit()
        finally:
            self._close(c)

    # ------------------------------------------------------------ counters

    def increment(
        self,
        key: str,
        endpoint: str = "*",
        now: float | None = None,
    ) -> tuple[int, int]:
        """Atomically bump the (minute, day) buckets for ``key``.

        Returns ``(this_minute, today)`` post-increment.
        """
        bm = _bucket_minute(now)
        bd = _bucket_day(now)
        with self._lock:
            c = self._conn()
            try:
                c.execute(
                    """INSERT INTO request_counts (key, bucket, endpoint, count)
                       VALUES (?,?,?,1)
                       ON CONFLICT(key, bucket, endpoint)
                       DO UPDATE SET count = count + 1""",
                    (key, bm, endpoint),
                )
                c.execute(
                    """INSERT INTO request_counts (key, bucket, endpoint, count)
                       VALUES (?,?,?,1)
                       ON CONFLICT(key, bucket, endpoint)
                       DO UPDATE SET count = count + 1""",
                    (key, bd, endpoint),
                )
                cur_min = c.execute(
                    "SELECT count FROM request_counts WHERE key=? AND bucket=? AND endpoint=?",
                    (key, bm, endpoint),
                ).fetchone()
                cur_day = c.execute(
                    "SELECT count FROM request_counts WHERE key=? AND bucket=? AND endpoint=?",
                    (key, bd, endpoint),
                ).fetchone()
                c.commit()
            finally:
                self._close(c)
        return int(cur_min["count"] if cur_min else 0), int(cur_day["count"] if cur_day else 0)

    def get_counts(
        self, key: str, endpoint: str = "*", now: float | None = None
    ) -> tuple[int, int]:
        """Read ``(this_minute, today)`` without incrementing."""
        bm = _bucket_minute(now)
        bd = _bucket_day(now)
        c = self._conn()
        try:
            r_min = c.execute(
                "SELECT count FROM request_counts WHERE key=? AND bucket=? AND endpoint=?",
                (key, bm, endpoint),
            ).fetchone()
            r_day = c.execute(
                "SELECT count FROM request_counts WHERE key=? AND bucket=? AND endpoint=?",
                (key, bd, endpoint),
            ).fetchone()
        finally:
            self._close(c)
        return int(r_min["count"] if r_min else 0), int(r_day["count"] if r_day else 0)

    def aggregate(self, now: float | None = None) -> dict[str, Any]:
        """Crude usage-dashboard query: today's totals, by tier, by endpoint."""
        bd = _bucket_day(now)
        c = self._conn()
        try:
            total = c.execute(
                "SELECT COALESCE(SUM(count),0) AS n FROM request_counts WHERE bucket=?",
                (bd,),
            ).fetchone()["n"]
            by_tier_rows = c.execute(
                """SELECT k.tier AS tier, COALESCE(SUM(rc.count),0) AS n
                   FROM request_counts rc
                   LEFT JOIN api_keys k ON rc.key = k.key
                   WHERE rc.bucket=?
                   GROUP BY k.tier""",
                (bd,),
            ).fetchall()
            by_endpoint_rows = c.execute(
                """SELECT endpoint, SUM(count) AS n
                   FROM request_counts
                   WHERE bucket=? AND endpoint != '*'
                   GROUP BY endpoint
                   ORDER BY n DESC LIMIT 10""",
                (bd,),
            ).fetchall()
            by_user_rows = c.execute(
                """SELECT k.user_id AS user_id, k.tier AS tier,
                          SUM(rc.count) AS n
                   FROM request_counts rc
                   LEFT JOIN api_keys k ON rc.key = k.key
                   WHERE rc.bucket=? AND k.user_id IS NOT NULL
                   GROUP BY k.user_id, k.tier
                   ORDER BY n DESC LIMIT 10""",
                (bd,),
            ).fetchall()
        finally:
            self._close(c)
        return {
            "total_requests_today": int(total or 0),
            "by_tier": {(r["tier"] or "anonymous"): int(r["n"]) for r in by_tier_rows},
            "top_endpoints": [
                {"endpoint": r["endpoint"], "count": int(r["n"])} for r in by_endpoint_rows
            ],
            "top_users": [
                {
                    "user_id": r["user_id"],
                    "tier": r["tier"] or "anonymous",
                    "count": int(r["n"]),
                }
                for r in by_user_rows
            ],
        }

    # ------------------------------------------------------------ demo quota

    def get_demo_quota_count(
        self, client_ip: str, day: str | None = None, now: float | None = None
    ) -> int:
        """Return today's demo-key issuance count for ``client_ip``."""
        d = day or _bucket_day(now).split(":", 1)[1]
        c = self._conn()
        try:
            row = c.execute(
                "SELECT count FROM demo_key_quota WHERE client_ip=? AND day=?",
                (client_ip, d),
            ).fetchone()
        finally:
            self._close(c)
        return int(row["count"] if row else 0)

    def increment_demo_quota(
        self, client_ip: str, day: str | None = None, now: float | None = None
    ) -> int:
        """Atomically bump the demo-key-issuance counter; returns new value."""
        d = day or _bucket_day(now).split(":", 1)[1]
        with self._lock:
            c = self._conn()
            try:
                c.execute(
                    """INSERT INTO demo_key_quota (client_ip, day, count)
                       VALUES (?,?,1)
                       ON CONFLICT(client_ip, day)
                       DO UPDATE SET count = count + 1""",
                    (client_ip, d),
                )
                row = c.execute(
                    "SELECT count FROM demo_key_quota WHERE client_ip=? AND day=?",
                    (client_ip, d),
                ).fetchone()
                c.commit()
            finally:
                self._close(c)
        return int(row["count"] if row else 0)

    # ------------------------------------------------------------ helpers

    @staticmethod
    def next_minute_reset(now: float | None = None) -> float:
        return _next_minute_reset(now)

    @staticmethod
    def next_day_reset(now: float | None = None) -> float:
        return _next_day_reset(now)


# Process-wide singleton ----------------------------------------------------

DEFAULT_DB_PATH = os.environ.get("PFM_AUTH_DB", "/tmp/pfm_auth.db")
_store_singleton: APIKeyStore | None = None
_singleton_lock = threading.Lock()


def get_api_key_store() -> APIKeyStore:
    """Return (and lazily create) the process-wide :class:`APIKeyStore`.

    Tests override this dependency with a ``:memory:`` store via
    ``app.dependency_overrides``.
    """
    global _store_singleton
    with _singleton_lock:
        if _store_singleton is None:
            _store_singleton = APIKeyStore(DEFAULT_DB_PATH)
    return _store_singleton


def _reset_singleton_for_tests() -> None:
    """Internal: clear the cached store (used by tests + lifespan reload)."""
    global _store_singleton
    with _singleton_lock:
        _store_singleton = None
