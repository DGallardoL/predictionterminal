"""Configure structlog with JSON formatter for production-readiness.

The default behaviour is JSON-line output to stdout, which plays well with
Docker/Fly/Render log shippers and downstream sinks (Loki, Datadog, ELK).
For local development set ``LOG_FORMAT=text`` to get coloured console output
via ``structlog.dev.ConsoleRenderer``.

Usage:

    from pfm.logging_setup import configure_logging, get_logger

    configure_logging()           # call once, very early at startup
    log = get_logger(__name__)
    log.info("startup", n_factors=1090)

The function is idempotent: calling it twice is safe (structlog's own state
is replaced; stdlib ``logging.basicConfig`` no-ops after the first call
unless ``force=True`` is passed).
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging() -> None:
    """Configure structlog + stdlib logging according to env vars.

    Environment variables read:

    - ``LOG_FORMAT``: ``"json"`` (default) or ``"text"``. JSON renders one
      JSON object per line; text uses structlog's coloured ConsoleRenderer.
    - ``LOG_LEVEL``: any level name accepted by stdlib logging
      (``DEBUG``/``INFO``/``WARNING``/``ERROR``/``CRITICAL``). Defaults to
      ``INFO``.
    """
    json_logs = os.environ.get("LOG_FORMAT", "json").lower() == "json"
    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.CallsiteParameterAdder(
            parameters={
                structlog.processors.CallsiteParameter.MODULE,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging through stdout at the same level so libraries that
    # call ``logging.getLogger(__name__).info(...)`` show up in the same
    # stream. ``force=True`` lets us re-run this in tests.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )


def get_logger(name: str | None = None):
    """Return a bound structlog logger; pass ``__name__`` from call sites."""
    return structlog.get_logger(name)
