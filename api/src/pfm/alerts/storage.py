"""SQLite-backed persistence for alert rules + events.

Schema (also exposed via ``AlertStore.SCHEMA_SQL``):

    CREATE TABLE alert_rules (
        id            TEXT PRIMARY KEY,
        user_id       TEXT NOT NULL,
        kind          TEXT NOT NULL,
        name          TEXT NOT NULL,
        enabled       INTEGER NOT NULL DEFAULT 1,
        cooldown_seconds INTEGER NOT NULL DEFAULT 300,
        spec_json     TEXT NOT NULL,        -- full pydantic dump
        last_fired_at REAL,                 -- unix seconds, NULL until first fire
        last_state    TEXT,                 -- 'fired' | 'armed' (edge-trigger)
        created_at    REAL NOT NULL,
        updated_at    REAL NOT NULL
    );

    CREATE INDEX idx_rules_user ON alert_rules(user_id);

    CREATE TABLE alert_events (
        event_id    TEXT PRIMARY KEY,
        rule_id     TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        kind        TEXT NOT NULL,
        fired_at    REAL NOT NULL,
        payload_json TEXT NOT NULL,
        delivered_json TEXT NOT NULL DEFAULT '[]',
        acked       INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY(rule_id) REFERENCES alert_rules(id)
    );

    CREATE INDEX idx_events_user ON alert_events(user_id);
    CREATE INDEX idx_events_rule ON alert_events(rule_id);
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any

from pfm.alerts.schemas import AlertRule


class AlertStore:
    """Thin wrapper over sqlite3. Stateless connections per call.

    For ``:memory:`` we keep a single persistent connection on the instance
    so the schema survives across calls (default sqlite3 behaviour gives a
    fresh DB per connection).
    """

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS alert_rules (
        id            TEXT PRIMARY KEY,
        user_id       TEXT NOT NULL,
        kind          TEXT NOT NULL,
        name          TEXT NOT NULL,
        enabled       INTEGER NOT NULL DEFAULT 1,
        cooldown_seconds INTEGER NOT NULL DEFAULT 300,
        spec_json     TEXT NOT NULL,
        last_fired_at REAL,
        last_state    TEXT,
        created_at    REAL NOT NULL,
        updated_at    REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_rules_user ON alert_rules(user_id);

    CREATE TABLE IF NOT EXISTS alert_events (
        event_id    TEXT PRIMARY KEY,
        rule_id     TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        kind        TEXT NOT NULL,
        fired_at    REAL NOT NULL,
        payload_json TEXT NOT NULL,
        delivered_json TEXT NOT NULL DEFAULT '[]',
        acked       INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY(rule_id) REFERENCES alert_rules(id)
    );
    CREATE INDEX IF NOT EXISTS idx_events_user ON alert_events(user_id);
    CREATE INDEX IF NOT EXISTS idx_events_rule ON alert_events(rule_id);

    CREATE TABLE IF NOT EXISTS channel_throttle (
        channel_key       TEXT PRIMARY KEY,
        window_start      REAL NOT NULL,
        count             INTEGER NOT NULL DEFAULT 0,
        digest_buffer_json TEXT NOT NULL DEFAULT '[]',
        last_event_at     REAL NOT NULL DEFAULT 0
    );
    """

    def __init__(self, db_path: str = "/tmp/pfm_alerts.db") -> None:
        self.db_path = db_path
        # For in-memory DBs, keep a persistent connection so schema sticks.
        # FastAPI's TestClient runs request handlers on a different thread
        # than the one that constructed the store, so we disable the
        # same-thread check (sqlite3 itself serializes access at the C
        # level when the default threadsafety mode is in effect).
        self._memory_conn: sqlite3.Connection | None = None
        # RLock so mutating methods that call read-only helpers (e.g.
        # patch_rule -> get_rule, record_event -> get_rule) do not deadlock
        # if a future change ever extends the locked region into the helper.
        self._lock = threading.RLock()
        if db_path == ":memory:":
            self._memory_conn = sqlite3.connect(db_path, check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        self.init_schema()

    # ------------------------------------------------------------------ low-level

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
        with self._lock:
            c = self._conn()
            try:
                c.executescript(self.SCHEMA_SQL)
                c.commit()
            finally:
                self._close(c)

    # ------------------------------------------------------------------ rules

    def save_rule(self, rule: AlertRule) -> str:
        """Insert (or replace) a rule. Returns the assigned id."""
        rid = rule.id or f"rule_{uuid.uuid4().hex[:12]}"
        spec = rule.model_dump(mode="json")
        spec["id"] = rid
        now = time.time()
        with self._lock:
            c = self._conn()
            try:
                existing = c.execute(
                    "SELECT id, last_fired_at, last_state, created_at FROM alert_rules WHERE id=?",
                    (rid,),
                ).fetchone()
                if existing is None:
                    c.execute(
                        """INSERT INTO alert_rules
                           (id, user_id, kind, name, enabled, cooldown_seconds,
                            spec_json, last_fired_at, last_state, created_at, updated_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            rid,
                            rule.user_id,
                            rule.kind,
                            rule.name,
                            1 if rule.enabled else 0,
                            rule.cooldown_seconds,
                            json.dumps(spec),
                            None,
                            "armed",
                            now,
                            now,
                        ),
                    )
                else:
                    c.execute(
                        """UPDATE alert_rules SET
                              user_id=?, kind=?, name=?, enabled=?, cooldown_seconds=?,
                              spec_json=?, updated_at=?
                           WHERE id=?""",
                        (
                            rule.user_id,
                            rule.kind,
                            rule.name,
                            1 if rule.enabled else 0,
                            rule.cooldown_seconds,
                            json.dumps(spec),
                            now,
                            rid,
                        ),
                    )
                c.commit()
            finally:
                self._close(c)
        return rid

    def list_rules(self, user_id: str, enabled: bool | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM alert_rules WHERE user_id=?"
        args: list[Any] = [user_id]
        if enabled is not None:
            sql += " AND enabled=?"
            args.append(1 if enabled else 0)
        sql += " ORDER BY created_at DESC"
        c = self._conn()
        try:
            rows = c.execute(sql, args).fetchall()
        finally:
            self._close(c)
        return [self._row_to_dict(r) for r in rows]

    def get_rule(self, id: str) -> dict[str, Any] | None:
        c = self._conn()
        try:
            row = c.execute("SELECT * FROM alert_rules WHERE id=?", (id,)).fetchone()
        finally:
            self._close(c)
        return self._row_to_dict(row) if row else None

    def delete_rule(self, id: str) -> bool:
        with self._lock:
            c = self._conn()
            try:
                cur = c.execute("DELETE FROM alert_rules WHERE id=?", (id,))
                c.commit()
                return cur.rowcount > 0
            finally:
                self._close(c)

    def patch_rule(self, id: str, **fields: Any) -> dict[str, Any] | None:
        """Apply partial updates to a stored rule's flat columns + spec_json."""
        with self._lock:
            existing = self.get_rule(id)
            if existing is None:
                return None
            spec = json.loads(existing["spec_json"])
            for k, v in fields.items():
                if v is None:
                    continue
                if k == "channels":
                    spec["channels"] = [
                        cc.model_dump(mode="json") if hasattr(cc, "model_dump") else cc for cc in v
                    ]
                else:
                    spec[k] = v
            # Rewrite flat cols too.
            c = self._conn()
            try:
                c.execute(
                    """UPDATE alert_rules SET
                          name=?, enabled=?, cooldown_seconds=?, spec_json=?, updated_at=?
                       WHERE id=?""",
                    (
                        spec.get("name", existing["name"]),
                        1 if spec.get("enabled", bool(existing["enabled"])) else 0,
                        int(spec.get("cooldown_seconds", existing["cooldown_seconds"])),
                        json.dumps(spec),
                        time.time(),
                        id,
                    ),
                )
                c.commit()
            finally:
                self._close(c)
            return self.get_rule(id)

    def update_fire_state(self, rule_id: str, fired_at: float | None, state: str) -> None:
        with self._lock:
            c = self._conn()
            try:
                c.execute(
                    "UPDATE alert_rules SET last_fired_at=?, last_state=?, updated_at=? WHERE id=?",
                    (fired_at, state, time.time(), rule_id),
                )
                c.commit()
            finally:
                self._close(c)

    # ------------------------------------------------------------------ events

    def record_event(self, rule_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            rule = self.get_rule(rule_id)
            if rule is None:
                raise ValueError(f"unknown rule_id {rule_id}")
            eid = f"evt_{uuid.uuid4().hex[:12]}"
            now = time.time()
            c = self._conn()
            try:
                c.execute(
                    """INSERT INTO alert_events
                       (event_id, rule_id, user_id, kind, fired_at, payload_json, delivered_json, acked)
                       VALUES (?,?,?,?,?,?,?,0)""",
                    (
                        eid,
                        rule_id,
                        rule["user_id"],
                        rule["kind"],
                        now,
                        json.dumps(payload),
                        "[]",
                    ),
                )
                c.commit()
            finally:
                self._close(c)
        return {
            "event_id": eid,
            "rule_id": rule_id,
            "user_id": rule["user_id"],
            "kind": rule["kind"],
            "fired_at": now,
            "payload": payload,
            "delivered": [],
            "acked": False,
        }

    def attach_delivery(self, event_id: str, deliveries: list[dict[str, Any]]) -> None:
        with self._lock:
            c = self._conn()
            try:
                c.execute(
                    "UPDATE alert_events SET delivered_json=? WHERE event_id=?",
                    (json.dumps(deliveries), event_id),
                )
                c.commit()
            finally:
                self._close(c)

    def list_events(
        self, user_id: str, unack_only: bool = False, limit: int = 50
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM alert_events WHERE user_id=?"
        args: list[Any] = [user_id]
        if unack_only:
            sql += " AND acked=0"
        sql += " ORDER BY fired_at DESC LIMIT ?"
        args.append(int(limit))
        c = self._conn()
        try:
            rows = c.execute(sql, args).fetchall()
        finally:
            self._close(c)
        return [self._event_row_to_dict(r) for r in rows]

    def ack_event(self, event_id: str) -> bool:
        with self._lock:
            c = self._conn()
            try:
                cur = c.execute("UPDATE alert_events SET acked=1 WHERE event_id=?", (event_id,))
                c.commit()
                return cur.rowcount > 0
            finally:
                self._close(c)

    # ------------------------------------------------------------------ throttle

    def throttle_check_and_record(
        self,
        channel_key: str,
        event: dict[str, Any],
        *,
        max_per_minute: int = 10,
        target: str | None = None,
        now: float | None = None,
    ) -> tuple[bool, int]:
        """Token bucket per ``channel_key`` (1-minute window).

        Returns ``(allow, count_after)``. If ``allow`` is False, the event
        (along with its delivery ``target``) has been appended to the
        channel's digest buffer; caller should NOT deliver it. Count resets
        at minute boundaries.
        """
        t = now if now is not None else time.time()
        wrapped = {"event": event, "target": target or ""}
        with self._lock:
            c = self._conn()
            try:
                row = c.execute(
                    "SELECT window_start, count, digest_buffer_json "
                    "FROM channel_throttle WHERE channel_key=?",
                    (channel_key,),
                ).fetchone()
                if row is None:
                    c.execute(
                        """INSERT INTO channel_throttle
                           (channel_key, window_start, count,
                            digest_buffer_json, last_event_at)
                           VALUES (?,?,?,?,?)""",
                        (channel_key, t, 1, "[]", t),
                    )
                    c.commit()
                    return True, 1
                window_start = float(row["window_start"])
                count = int(row["count"])
                if (t - window_start) >= 60.0:
                    # New window — reset counter (preserve buffer if any).
                    c.execute(
                        """UPDATE channel_throttle SET window_start=?,
                              count=1, last_event_at=? WHERE channel_key=?""",
                        (t, t, channel_key),
                    )
                    c.commit()
                    return True, 1
                if count < max_per_minute:
                    c.execute(
                        """UPDATE channel_throttle SET count=count+1,
                              last_event_at=? WHERE channel_key=?""",
                        (t, channel_key),
                    )
                    c.commit()
                    return True, count + 1
                # Bucket full → buffer the event for digest.
                buf = json.loads(row["digest_buffer_json"] or "[]")
                buf.append(wrapped)
                c.execute(
                    """UPDATE channel_throttle
                       SET digest_buffer_json=?, last_event_at=?
                       WHERE channel_key=?""",
                    (json.dumps(buf, default=str), t, channel_key),
                )
                c.commit()
                return False, count
            finally:
                self._close(c)

    def flush_pending_digests(
        self,
        *,
        quiet_seconds: float = 60.0,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Drain digest buffers for channels quiet for >= ``quiet_seconds``.

        Returns one ``{channel_key, count, events, summary}`` dict per
        flushed channel. Caller is responsible for actually dispatching the
        digest event(s) through the relevant channel(s).
        """
        t = now if now is not None else time.time()
        flushed: list[dict[str, Any]] = []
        with self._lock:
            c = self._conn()
            try:
                rows = c.execute(
                    """SELECT channel_key, last_event_at, digest_buffer_json
                       FROM channel_throttle
                       WHERE digest_buffer_json != '[]'"""
                ).fetchall()
                for r in rows:
                    last = float(r["last_event_at"])
                    if (t - last) < quiet_seconds:
                        continue
                    buf = json.loads(r["digest_buffer_json"] or "[]")
                    if not buf:
                        continue
                    flushed.append(
                        {
                            "channel_key": r["channel_key"],
                            "count": len(buf),
                            "events": buf,
                            "summary": (
                                f"{len(buf)} alerts buffered while throttled on {r['channel_key']}"
                            ),
                        }
                    )
                    c.execute(
                        """UPDATE channel_throttle
                           SET digest_buffer_json='[]', count=0,
                               window_start=?, last_event_at=?
                           WHERE channel_key=?""",
                        (t, t, r["channel_key"]),
                    )
                c.commit()
            finally:
                self._close(c)
        return flushed

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        d["enabled"] = bool(d["enabled"])
        return d

    @staticmethod
    def _event_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        return {
            "event_id": d["event_id"],
            "rule_id": d["rule_id"],
            "user_id": d["user_id"],
            "kind": d["kind"],
            "fired_at": d["fired_at"],
            "payload": json.loads(d["payload_json"]),
            "delivered": json.loads(d["delivered_json"]),
            "acked": bool(d["acked"]),
        }
