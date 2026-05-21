"""Pydantic models for API key auth + tier metadata.

Tiers (in ascending order of privilege):

- ``free``       — anonymous + signed-up free users. Throttle = 30/min, 1k/day.
- ``pro``        — paid Pro tier. 300/min, 10k/day.
- ``quant``      — paid Quant tier. 3000/min, 100k/day.
- ``enterprise`` — bespoke contracts. 30000/min, effectively unlimited daily.

The numbers live in :data:`TIER_DEFAULTS` so middleware, tests, and the docs
all read the same source of truth. Keys produced by the router default to
these limits; ops can override per-key via the storage layer if a customer
needs custom throughput.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Tier = Literal["free", "pro", "quant", "enterprise"]

TIER_ORDER: tuple[Tier, ...] = ("free", "pro", "quant", "enterprise")

# (rate_limit_per_minute, daily_quota). ``daily_quota=0`` means unlimited.
TIER_DEFAULTS: dict[Tier, tuple[int, int]] = {
    "free": (30, 1_000),
    "pro": (300, 10_000),
    "quant": (3_000, 100_000),
    "enterprise": (30_000, 0),
}

# Anonymous (no header) requests use a stricter sub-free bucket so the
# Free tier still has a reason to sign up.
ANON_RATE_PER_MIN = 10
ANON_DAILY_QUOTA = 100


class APIKey(BaseModel):
    """A single API key + its tier metadata.

    The opaque ``key`` is what the client sends in ``Authorization: Bearer ...``.
    Format: ``sk_pfm_<32-char-urlsafe>``. We never re-display the secret after
    creation; only the prefix is safe to show in dashboards.
    """

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., description="Opaque secret. Format sk_pfm_<random>.")
    user_id: str = Field(..., min_length=1, max_length=120)
    tier: Tier = "free"
    created_at: datetime
    last_used_at: datetime | None = None
    enabled: bool = True
    rate_limit_per_minute: int = Field(default=30, ge=0, le=1_000_000)
    daily_quota: int = Field(default=1_000, ge=0)

    @classmethod
    def new(cls, user_id: str, tier: Tier = "free") -> APIKey:
        """Mint a fresh key with tier-default throttles."""
        rpm, quota = TIER_DEFAULTS[tier]
        return cls(
            key=f"sk_pfm_{secrets.token_urlsafe(24)}",
            user_id=user_id,
            tier=tier,
            created_at=datetime.now(UTC),
            rate_limit_per_minute=rpm,
            daily_quota=quota,
        )

    def masked(self) -> str:
        """Display-safe form: ``sk_pfm_abcd…wxyz``."""
        if len(self.key) < 14:
            return self.key
        return f"{self.key[:11]}…{self.key[-4:]}"


class APIKeyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(..., min_length=1, max_length=120)
    tier: Tier = "free"


class APIKeyPublic(BaseModel):
    """Sanitised view used in list/usage endpoints."""

    model_config = ConfigDict(extra="forbid")
    key_masked: str
    user_id: str
    tier: Tier
    created_at: datetime
    last_used_at: datetime | None = None
    enabled: bool = True
    rate_limit_per_minute: int
    daily_quota: int


class UsageStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str
    tier: Tier
    requests_this_minute: int
    requests_today: int
    rate_limit_per_minute: int
    daily_quota: int  # 0 = unlimited
    daily_remaining: int  # -1 = unlimited
    minute_remaining: int


def tier_at_least(actual: Tier, required: Tier) -> bool:
    """True iff ``actual`` is ranked at or above ``required`` in TIER_ORDER."""
    return TIER_ORDER.index(actual) >= TIER_ORDER.index(required)
