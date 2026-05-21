"""Rule evaluation engine.

Stateless evaluators that, given a rule + a context dict, return
``(fired: bool, payload: dict)``. The orchestration layer
(``evaluate_all``) consults the store for cooldown / hysteresis state and
records events as appropriate.

The ``ctx`` parameter is shaped like::

    {
        "snapshot": {slug: float},                       # current price (0..1)
        "history": {slug: list[(ts, price)]},            # ordered ascending
        "volume_history": {slug: list[(ts, vol)]},
        "now": float,                                    # unix seconds
    }

Missing context keys cause a rule to silently not-fire; the engine never
raises on data gaps.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

from pfm.alerts.schemas import (
    PriceChangePctRule,
    PriceCrossRule,
    VolumeSpikeRule,
    ZScorePairRule,
)
from pfm.alerts.storage import AlertStore

logger = logging.getLogger("pfm.alerts.engine")

WINDOW_SECONDS = {"1h": 3600, "4h": 14400, "24h": 86400}


# ---------------------------------------------------------------- per-rule evals


def _eval_price_cross(
    rule: PriceCrossRule, ctx: dict[str, Any], last_state: str | None
) -> tuple[bool, dict[str, Any]]:
    snap = ctx.get("snapshot", {})
    if rule.slug not in snap:
        return False, {}
    price = float(snap[rule.slug])
    crossed = price > rule.threshold if rule.op == ">" else price < rule.threshold
    # Edge-trigger: only fire on the *transition* from armed → fired.
    if crossed and last_state != "fired":
        return True, {
            "rule_kind": "price_cross",
            "rule_name": rule.name,
            "slug": rule.slug,
            "op": rule.op,
            "threshold": rule.threshold,
            "price": price,
            "message": f"{rule.slug} {rule.op} {rule.threshold} (price={price:.4f})",
        }
    return False, {}


def _re_arm_price_cross(rule: PriceCrossRule, ctx: dict[str, Any]) -> bool:
    """Check hysteresis condition: returns True when we should reset to
    'armed' after a previous fire."""
    snap = ctx.get("snapshot", {})
    if rule.slug not in snap:
        return False
    price = float(snap[rule.slug])
    if rule.op == ">":
        return price < (rule.threshold - rule.hysteresis)
    return price > (rule.threshold + rule.hysteresis)


def _eval_price_change_pct(
    rule: PriceChangePctRule, ctx: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    hist = ctx.get("history", {}).get(rule.slug, [])
    if len(hist) < 2:
        return False, {}
    now_ts, now_price = hist[-1]
    window = WINDOW_SECONDS[rule.window]
    cutoff = now_ts - window
    # Find latest sample at or before cutoff.
    base_price = None
    for ts, p in reversed(hist):
        if ts <= cutoff:
            base_price = p
            break
    if base_price is None or base_price == 0:
        return False, {}
    change = (now_price - base_price) / base_price
    if abs(change) >= rule.pct_abs:
        return True, {
            "rule_kind": "price_change_pct",
            "rule_name": rule.name,
            "slug": rule.slug,
            "window": rule.window,
            "pct": change,
            "from_price": base_price,
            "to_price": now_price,
        }
    return False, {}


def _eval_zscore_pair(rule: ZScorePairRule, ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    ha = ctx.get("history", {}).get(rule.slug_a, [])
    hb = ctx.get("history", {}).get(rule.slug_b, [])
    if len(ha) < rule.window or len(hb) < rule.window:
        return False, {}
    a = [p for _, p in ha[-rule.window :]]
    b = [p for _, p in hb[-rule.window :]]
    spread = [ai - rule.beta * bi for ai, bi in zip(a, b, strict=False)]
    n = len(spread)
    mean = sum(spread) / n
    var = sum((s - mean) ** 2 for s in spread) / max(n - 1, 1)
    sd = math.sqrt(var)
    if sd == 0:
        return False, {}
    z = (spread[-1] - mean) / sd
    if abs(z) >= rule.z_threshold:
        return True, {
            "rule_kind": "zscore_pair",
            "rule_name": rule.name,
            "slug_a": rule.slug_a,
            "slug_b": rule.slug_b,
            "z": z,
            "threshold": rule.z_threshold,
            "spread": spread[-1],
        }
    return False, {}


def _eval_volume_spike(rule: VolumeSpikeRule, ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    hist = ctx.get("volume_history", {}).get(rule.slug, [])
    if len(hist) < rule.lookback_days + 1:
        return False, {}
    today = hist[-1][1]
    lookback = [v for _, v in hist[-(rule.lookback_days + 1) : -1]]
    n = len(lookback)
    mean = sum(lookback) / n
    var = sum((v - mean) ** 2 for v in lookback) / max(n - 1, 1)
    sd = math.sqrt(var)
    if sd == 0:
        return False, {}
    sigma = (today - mean) / sd
    if sigma >= rule.n_sigma:
        return True, {
            "rule_kind": "volume_spike",
            "rule_name": rule.name,
            "slug": rule.slug,
            "today_volume": today,
            "mean_volume": mean,
            "sigma": sigma,
        }
    return False, {}


# ---------------------------------------------------------------- public surface


def _hydrate_rule(spec: dict[str, Any]):
    """Re-construct a typed rule from the stored spec_json."""
    kind = spec["kind"]
    cls = {
        "price_cross": PriceCrossRule,
        "price_change_pct": PriceChangePctRule,
        "zscore_pair": ZScorePairRule,
        "volume_spike": VolumeSpikeRule,
    }[kind]
    return cls.model_validate(spec)


def evaluate_rule(rule_row: dict[str, Any], ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Evaluate one stored rule. Returns ``(fired, payload)``.

    ``rule_row`` is a row dict from ``AlertStore`` (so it has ``spec_json``,
    ``last_fired_at``, ``last_state``, ``cooldown_seconds``).
    """
    if not rule_row.get("enabled", True):
        return False, {}
    spec = json.loads(rule_row["spec_json"])
    rule = _hydrate_rule(spec)
    now = ctx.get("now", time.time())

    # Cooldown: don't fire if we recently fired.
    last_fired = rule_row.get("last_fired_at")
    if last_fired is not None:
        if (now - float(last_fired)) < rule.cooldown_seconds:
            return False, {}

    if isinstance(rule, PriceCrossRule):
        return _eval_price_cross(rule, ctx, rule_row.get("last_state"))
    if isinstance(rule, PriceChangePctRule):
        return _eval_price_change_pct(rule, ctx)
    if isinstance(rule, ZScorePairRule):
        return _eval_zscore_pair(rule, ctx)
    if isinstance(rule, VolumeSpikeRule):
        return _eval_volume_spike(rule, ctx)
    return False, {}


def evaluate_all(
    store: AlertStore, ctx: dict[str, Any], user_id: str | None = None
) -> list[dict[str, Any]]:
    """Evaluate every enabled rule (optionally filtered by user_id) and
    record events for those that fire. Returns the list of fired event dicts.
    """
    rules: list[dict[str, Any]]
    if user_id:
        rules = store.list_rules(user_id, enabled=True)
    else:
        # Enumerate across all users — small scale POC, full table scan is fine.
        c = store._conn()
        try:
            rows = c.execute("SELECT * FROM alert_rules WHERE enabled=1").fetchall()
        finally:
            store._close(c)
        rules = [store._row_to_dict(r) for r in rows if r]

    fired: list[dict[str, Any]] = []
    now = ctx.get("now", time.time())
    for r in rules:
        try:
            ok, payload = evaluate_rule(r, ctx)
        except Exception as e:
            logger.exception("evaluate_rule failed for %s: %s", r.get("id"), e)
            continue
        if ok:
            event = store.record_event(r["id"], payload)
            store.update_fire_state(r["id"], now, "fired")
            fired.append(event)
        # Re-arm hysteresis-based rules when condition cleared.
        elif r.get("kind") == "price_cross" and r.get("last_state") == "fired":
            spec = json.loads(r["spec_json"])
            rule = _hydrate_rule(spec)
            if isinstance(rule, PriceCrossRule) and _re_arm_price_cross(rule, ctx):
                store.update_fire_state(r["id"], r.get("last_fired_at"), "armed")
    return fired
