"""Pydantic schemas for alert rules.

Uses Pydantic v2 discriminated union on the ``kind`` field so the API can
accept any of the 4 rule types in a single endpoint. Each rule shares a base
(id, user_id, name, channels, cooldown, enabled) and adds kind-specific
parameters.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ChannelRef(BaseModel):
    """Reference to a delivery channel.

    Args:
        type: One of inapp / webhook / slack / discord.
        target: Channel-specific destination (URL for webhook/slack/discord,
            user_id for inapp).
        enabled: If False, this channel is skipped during fan-out.
    """

    type: Literal["inapp", "email", "webhook", "slack", "discord", "telegram"]
    target: str
    enabled: bool = True


class _Base(BaseModel):
    id: str | None = None
    user_id: str
    name: str
    cooldown_seconds: int = Field(default=300, ge=0)
    channels: list[ChannelRef] = Field(default_factory=list)
    enabled: bool = True


class PriceCrossRule(_Base):
    """Fires when contract price crosses a threshold (with hysteresis).

    Hysteresis prevents flap: once fired on ">", the rule must dip below
    threshold - hysteresis before it can re-arm.
    """

    kind: Literal["price_cross"] = "price_cross"
    slug: str
    op: Literal[">", "<"]
    threshold: float = Field(ge=0, le=1)
    hysteresis: float = Field(default=0.01, ge=0, le=1)


class PriceChangePctRule(_Base):
    """Fires when |Δprice| / price over a rolling window exceeds pct_abs."""

    kind: Literal["price_change_pct"] = "price_change_pct"
    slug: str
    window: Literal["1h", "4h", "24h"]
    pct_abs: float = Field(ge=0)


class ZScorePairRule(_Base):
    """Fires when z-score of (price_a - beta*price_b) exceeds |z_threshold|."""

    kind: Literal["zscore_pair"] = "zscore_pair"
    slug_a: str
    slug_b: str
    beta: float = 1.0
    window: int = Field(default=30, ge=2)
    z_threshold: float = Field(default=2.0, ge=0)


class VolumeSpikeRule(_Base):
    """Fires when today's volume is more than n_sigma above lookback mean."""

    kind: Literal["volume_spike"] = "volume_spike"
    slug: str
    lookback_days: int = Field(default=7, ge=2)
    n_sigma: float = Field(default=2.0, ge=0)


AlertRule = Annotated[
    PriceCrossRule | PriceChangePctRule | ZScorePairRule | VolumeSpikeRule,
    Field(discriminator="kind"),
]


class AlertRuleEnvelope(BaseModel):
    """Wrapper used by FastAPI request body so discriminated union resolves."""

    rule: AlertRule


class AlertEvent(BaseModel):
    """Materialized event row returned by the API."""

    event_id: str
    rule_id: str
    user_id: str
    kind: str
    fired_at: float
    payload: dict
    delivered: list[dict] = Field(default_factory=list)
    acked: bool = False


class AlertRulePatch(BaseModel):
    """Partial update body for PATCH /alerts/{id}.

    All fields optional; only provided fields are applied. We keep this
    intentionally narrow — to mutate the discriminator-bearing fields, the
    client should DELETE + POST.
    """

    name: str | None = None
    cooldown_seconds: int | None = Field(default=None, ge=0)
    channels: list[ChannelRef] | None = None
    enabled: bool | None = None
