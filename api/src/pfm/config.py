"""Application settings loaded from env via pydantic-settings."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"
    polymarket_clob_url: str = "https://clob.polymarket.com"

    redis_url: str = "redis://redis:6379/0"
    cache_ttl_seconds: int = Field(default=3600, ge=0)

    factors_file: Path = Path(__file__).resolve().parent / "factors.yml"

    log_level: str = "INFO"
    request_timeout_seconds: float = 30.0
    default_epsilon: float = Field(default=0.01, gt=0.0, lt=0.5)

    # CORS configuration. Comma-separated origins or "*" for any origin.
    # In production, "*" emits a stderr warning (see `Settings.production`).
    cors_origins: str = "*"

    # Optional alerting / monitoring hooks (read by terminal_alerts and
    # observability layers). Defaults are dry-run-friendly.
    sentry_dsn: str = ""
    slack_webhook_url: str = ""

    @property
    def production(self) -> bool:
        """Truthy when the deployment marks itself as production.

        Driven by the `ENV` environment variable (e.g. `ENV=production` on
        Render / Fly.io). Independent of pydantic so it can be checked
        without re-instantiating Settings.
        """
        return os.environ.get("ENV", "").lower() == "production"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Cached accessor — also the FastAPI dependency."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _warn_if_unsafe_production(_settings)
    return _settings


def _warn_if_unsafe_production(settings: Settings) -> None:
    """Emit stderr warnings for production deployments with unsafe defaults.

    Does NOT raise; the service must still start (e.g. for first-boot smoke
    tests). Operators should treat warnings as deploy-blockers in CI.
    """
    if not settings.production:
        return
    if settings.cors_origins.strip() in ("*", ""):
        logger.warning(
            "CORS_ORIGINS is '*' in production. Set it to your "
            "domain(s), e.g. CORS_ORIGINS='https://your-app.fly.dev'."
        )
    if settings.redis_url.startswith("redis://redis:"):
        logger.warning(
            "REDIS_URL points at the docker-compose service name "
            "'redis' but ENV=production. Confirm the prod Redis URL is set."
        )
