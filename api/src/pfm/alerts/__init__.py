"""Alert engine multi-channel.

Persistent rules + delivery to in-app, webhook, Slack, Discord. SQLite-backed
storage with cooldown / hysteresis / edge-trigger semantics.

Public surface:
    - schemas: AlertRule (discriminated union), ChannelRef
    - storage: AlertStore (SQLite CRUD)
    - channels: Channel protocol + concrete implementations
    - engine: evaluate_rule, evaluate_all
    - router: FastAPI router (mount via app.include_router)
"""

from __future__ import annotations

from pfm.alerts.schemas import (
    AlertRule,
    ChannelRef,
    PriceChangePctRule,
    PriceCrossRule,
    VolumeSpikeRule,
    ZScorePairRule,
)
from pfm.alerts.storage import AlertStore

__all__ = [
    "AlertRule",
    "AlertStore",
    "ChannelRef",
    "PriceChangePctRule",
    "PriceCrossRule",
    "VolumeSpikeRule",
    "ZScorePairRule",
]
