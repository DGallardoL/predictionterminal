"""FastAPI router for the multi-channel alert engine.

Endpoints (tagged ``alerts``):
    POST   /alerts                 → create rule
    GET    /alerts?user_id=...     → list rules for user
    GET    /alerts/{id}            → rule detail
    PATCH  /alerts/{id}            → partial update
    DELETE /alerts/{id}            → delete rule
    POST   /alerts/{id}/test       → fan out a synthetic event (dry-run)
    GET    /alerts/events          → list events (?user_id=&unack=1)
    POST   /alerts/events/{event_id}/ack → ack an event

The store is wired via dependency injection so tests can substitute a
``:memory:`` store. ``main.py`` should override
``get_alert_store`` if it needs a different path.
"""

from __future__ import annotations

import json
import os
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from pfm.alerts.channels import DEFAULT_REGISTRY, fanout
from pfm.alerts.schemas import AlertRule, AlertRulePatch
from pfm.alerts.storage import AlertStore
from pfm.auth.dependencies import require_api_key
from pfm.auth.models import APIKey

DEFAULT_DB_PATH = os.environ.get("PFM_ALERTS_DB", "/tmp/pfm_alerts.db")

_store_singleton: AlertStore | None = None


def get_alert_store() -> AlertStore:
    """FastAPI dependency: lazy-init a process-wide AlertStore."""
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = AlertStore(DEFAULT_DB_PATH)
    return _store_singleton


router = APIRouter(prefix="/alerts", tags=["alerts"])


def _enforce_owner(key: APIKey, owner_user_id: str) -> None:
    """Reject cross-user access. The system key (auth disabled / dev) bypasses.

    When auth is enabled, a real ``APIKey`` may only see/mutate rules and events
    that belong to its own ``user_id``. The synthetic ``system`` key — used in
    dev and tests when auth is off — is allowed through so existing flows keep
    working.
    """
    if key.user_id == "system":
        return
    if key.user_id != owner_user_id:
        raise HTTPException(status_code=403, detail="not your rule/event")


# ---------------------------------------------------------------- rules


@router.post("", summary="Create a new alert rule")
async def create_rule(
    rule: Annotated[AlertRule, Body(..., discriminator="kind")],
    store: AlertStore = Depends(get_alert_store),
    key: APIKey = Depends(require_api_key),
) -> dict:
    _enforce_owner(key, rule.user_id)
    rid = store.save_rule(rule)
    return store.get_rule(rid) or {}


@router.get("", summary="List alert rules for a user")
async def list_rules(
    user_id: str = Query(..., min_length=1),
    enabled: bool | None = None,
    store: AlertStore = Depends(get_alert_store),
    key: APIKey = Depends(require_api_key),
) -> list[dict]:
    _enforce_owner(key, user_id)
    return store.list_rules(user_id, enabled=enabled)


@router.get("/events", summary="List alert events")
async def list_events(
    user_id: str = Query(..., min_length=1),
    unack: int = 0,
    limit: int = 50,
    store: AlertStore = Depends(get_alert_store),
    key: APIKey = Depends(require_api_key),
) -> list[dict]:
    _enforce_owner(key, user_id)
    return store.list_events(user_id, unack_only=bool(unack), limit=limit)


@router.post("/events/{event_id}/ack", summary="Acknowledge an event")
async def ack_event(
    event_id: str,
    store: AlertStore = Depends(get_alert_store),
    key: APIKey = Depends(require_api_key),
) -> dict:
    # Best-effort cross-user check: peek at the event's owner before acking.
    if key.user_id != "system":
        owner = None
        try:
            evs = store.list_events(key.user_id, limit=10_000)
            owner = next((e for e in evs if e.get("event_id") == event_id), None)
        except Exception:
            owner = None
        if owner is None:
            # Either it doesn't belong to this key or no such event — 404 either way.
            raise HTTPException(status_code=404, detail="event not found")
    ok = store.ack_event(event_id)
    if not ok:
        raise HTTPException(status_code=404, detail="event not found")
    return {"event_id": event_id, "acked": True}


@router.get("/{id}", summary="Get an alert rule by id")
async def get_rule(
    id: str,
    store: AlertStore = Depends(get_alert_store),
    key: APIKey = Depends(require_api_key),
) -> dict:
    rule = store.get_rule(id)
    if rule is None:
        raise HTTPException(status_code=404, detail="rule not found")
    _enforce_owner(key, rule.get("user_id", ""))
    return rule


@router.patch("/{id}", summary="Partial-update an alert rule")
async def patch_rule(
    id: str,
    patch: AlertRulePatch,
    store: AlertStore = Depends(get_alert_store),
    key: APIKey = Depends(require_api_key),
) -> dict:
    existing = store.get_rule(id)
    if existing is None:
        raise HTTPException(status_code=404, detail="rule not found")
    _enforce_owner(key, existing.get("user_id", ""))
    fields = patch.model_dump(exclude_unset=True)
    updated = store.patch_rule(id, **fields)
    if updated is None:
        raise HTTPException(status_code=404, detail="rule not found")
    return updated


@router.delete("/{id}", summary="Delete an alert rule")
async def delete_rule(
    id: str,
    store: AlertStore = Depends(get_alert_store),
    key: APIKey = Depends(require_api_key),
) -> dict:
    existing = store.get_rule(id)
    if existing is None:
        raise HTTPException(status_code=404, detail="rule not found")
    _enforce_owner(key, existing.get("user_id", ""))
    ok = store.delete_rule(id)
    if not ok:
        raise HTTPException(status_code=404, detail="rule not found")
    return {"id": id, "deleted": True}


@router.post("/{id}/test", summary="Dry-run dispatch to the rule's channels")
async def test_rule(
    id: str,
    store: AlertStore = Depends(get_alert_store),
    key: APIKey = Depends(require_api_key),
) -> dict:
    rule = store.get_rule(id)
    if rule is None:
        raise HTTPException(status_code=404, detail="rule not found")
    _enforce_owner(key, rule.get("user_id", ""))
    spec = json.loads(rule["spec_json"])
    # Synthetic test event; UI/integration sees this exact shape on real fires.
    event = {
        "event_id": f"test_{id}",
        "rule_id": id,
        "user_id": rule["user_id"],
        "kind": rule["kind"],
        "fired_at": rule.get("updated_at") or 0.0,
        "payload": {
            "rule_name": rule["name"],
            "rule_kind": rule["kind"],
            "test": True,
            "message": f"[TEST] {rule['name']}",
        },
        "delivered": [],
        "acked": False,
    }
    deliveries = await fanout(event, spec.get("channels", []), DEFAULT_REGISTRY)
    return {"event": event, "deliveries": deliveries, "dry_run": True}
