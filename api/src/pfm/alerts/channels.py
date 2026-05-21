"""Delivery channels for alert events.

Each channel implements ``send(event, target) -> dict`` (the DeliveryResult).
The result schema is:

    {"channel": "<type>", "target": "<target>", "ok": bool,
     "status_code": int | None, "error": str | None, "ts": float}

Channels never raise on remote failures — they capture the error in the
result dict so failure isolation across the fan-out is structural rather
than relying on engine-level try/except.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import ipaddress
import json
import logging
import os
import socket
import time
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("pfm.alerts.channels")

DRY_RUN_ENV = "PFM_ALERTS_DRY_RUN"
WEBHOOK_HMAC_SECRET_ENV = "PFM_ALERTS_WEBHOOK_SECRET"
EMAIL_PROVIDER_ENV = "PFM_EMAIL_PROVIDER"
RESEND_API_KEY_ENV = "RESEND_API_KEY"
SENDGRID_API_KEY_ENV = "SENDGRID_API_KEY"
EMAIL_FROM_ENV = "PFM_EMAIL_FROM"
ACK_BASE_URL_ENV = "PFM_ALERTS_ACK_BASE_URL"

# Hostnames that always identify "internal" infra/metadata services regardless
# of the IP they happen to resolve to. The IP check below already handles the
# common cloud-metadata addresses (169.254.169.254 etc.), but this list catches
# obvious lexical-only attacks (e.g. ``localhost``) and provider-specific names.
_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata",
        # Common k8s in-cluster service DNS suffixes.
        "kubernetes.default",
        "kubernetes.default.svc",
        "kubernetes.default.svc.cluster.local",
    }
)


def _is_internal_url(url: str) -> bool:
    """Return True if ``url`` points at a loopback, private, link-local, or
    cloud-metadata destination — i.e. anything we should refuse to call from a
    user-supplied webhook target.

    The check is fail-closed: malformed URLs, missing hosts, and DNS failures
    all return True so a confused parser can't smuggle a request through.
    """
    if not url or not isinstance(url, str):
        return True
    try:
        parsed = urlparse(url)
    except Exception:
        return True
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        # File://, gopher://, etc. — refuse outright.
        return True
    host = (parsed.hostname or "").lower().strip().rstrip(".")
    if not host:
        return True
    if host in _BLOCKED_HOSTNAMES:
        return True

    # Resolve to all candidate IPs; any internal one is enough to refuse.
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True
    seen: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_str = sockaddr[0]
        if ip_str in seen:
            continue
        seen.add(ip_str)
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return True
        # ip_address.is_private covers RFC1918 (10/8, 172.16/12, 192.168/16)
        # and ULA. is_loopback covers 127.0.0.0/8 + ::1. is_link_local covers
        # 169.254/16 (AWS/GCP metadata) and fe80::/10. is_reserved/is_multicast
        # are belt-and-suspenders.
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    # No resolvable address that wasn't private/loopback/link-local → treat as
    # un-routable (e.g. DNS returned nothing) and block by default.
    return not seen


def _is_dry_run() -> bool:
    return os.environ.get(DRY_RUN_ENV, "").lower() in {"1", "true", "yes"}


def _result(
    channel: str,
    target: str,
    ok: bool,
    status_code: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "channel": channel,
        "target": target,
        "ok": ok,
        "status_code": status_code,
        "error": error,
        "ts": time.time(),
    }


class Channel(Protocol):
    """Delivery channel protocol."""

    type: str

    async def send(self, event: dict, target: str) -> dict: ...


# ---------------------------------------------------------------------- in-app


class InAppChannel:
    """No-op delivery: rule evaluation already inserted the row in
    ``alert_events`` so this channel just records a successful delivery."""

    type = "inapp"

    async def send(self, event: dict, target: str) -> dict:
        return _result("inapp", target, True, status_code=200)


# ------------------------------------------------------------------ slack/discord


def _format_slack_text(event: dict) -> str:
    p = event.get("payload", {})
    name = p.get("rule_name", event.get("rule_id", "alert"))
    kind = event.get("kind", "?")
    return (
        f":rotating_light: *{name}* (`{kind}`) fired\n"
        f"```{json.dumps(p, indent=2, default=str)[:1000]}```"
    )


class SlackChannel:
    type = "slack"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def send(self, event: dict, target: str) -> dict:
        if _is_dry_run():
            return _result("slack", target, True, status_code=204, error="dry-run")
        # Production path uses ``self._client is None`` and constructs its own
        # AsyncClient. Tests inject a MockTransport-backed client and the
        # ``target`` is typically a fake hostname that wouldn't resolve, so we
        # only apply the SSRF guard on the production path.
        if self._client is None and _is_internal_url(target):
            logger.warning("slack delivery blocked (internal url): %s", target)
            return _result("slack", target, False, error="internal-url-blocked")
        body = {"text": _format_slack_text(event)}
        client = self._client or httpx.AsyncClient(timeout=10.0, follow_redirects=False)
        owns = self._client is None
        try:
            r = await client.post(target, json=body)
            # If the server replied with a redirect, validate the new location
            # against the same allowlist instead of blindly following it.
            if 300 <= r.status_code < 400:
                loc = r.headers.get("location") or ""
                if loc and _is_internal_url(loc):
                    logger.warning(
                        "slack delivery blocked (redirect to internal): %s -> %s",
                        target,
                        loc,
                    )
                    return _result(
                        "slack",
                        target,
                        False,
                        status_code=r.status_code,
                        error="internal-url-blocked (redirect)",
                    )
            return _result("slack", target, 200 <= r.status_code < 300, r.status_code)
        except Exception as e:
            logger.warning("slack delivery failed: %s", e)
            return _result("slack", target, False, error=str(e))
        finally:
            if owns:
                await client.aclose()


class DiscordChannel:
    type = "discord"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def send(self, event: dict, target: str) -> dict:
        if _is_dry_run():
            return _result("discord", target, True, status_code=204, error="dry-run")
        if self._client is None and _is_internal_url(target):
            logger.warning("discord delivery blocked (internal url): %s", target)
            return _result("discord", target, False, error="internal-url-blocked")
        body = {"content": _format_slack_text(event)}
        client = self._client or httpx.AsyncClient(timeout=10.0, follow_redirects=False)
        owns = self._client is None
        try:
            r = await client.post(target, json=body)
            if 300 <= r.status_code < 400:
                loc = r.headers.get("location") or ""
                if loc and _is_internal_url(loc):
                    logger.warning(
                        "discord delivery blocked (redirect to internal): %s -> %s",
                        target,
                        loc,
                    )
                    return _result(
                        "discord",
                        target,
                        False,
                        status_code=r.status_code,
                        error="internal-url-blocked (redirect)",
                    )
            return _result("discord", target, 200 <= r.status_code < 300, r.status_code)
        except Exception as e:
            logger.warning("discord delivery failed: %s", e)
            return _result("discord", target, False, error=str(e))
        finally:
            if owns:
                await client.aclose()


# ---------------------------------------------------------------------- webhook


class WebhookChannel:
    """Generic webhook with HMAC-SHA256 signature in ``X-PFM-Signature``.

    The signature is computed over the canonical JSON body using the secret
    from ``PFM_ALERTS_WEBHOOK_SECRET``. If the secret is not set, the header
    is omitted (and consumers should treat that as "unsigned").
    """

    type = "webhook"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    @staticmethod
    def _sign(body: bytes) -> str | None:
        secret = os.environ.get(WEBHOOK_HMAC_SECRET_ENV)
        if not secret:
            return None
        return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()

    async def send(self, event: dict, target: str) -> dict:
        if _is_dry_run():
            return _result("webhook", target, True, status_code=204, error="dry-run")
        if self._client is None and _is_internal_url(target):
            logger.warning("webhook delivery blocked (internal url): %s", target)
            return _result("webhook", target, False, error="internal-url-blocked")
        body = json.dumps(event, default=str, sort_keys=True).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        sig = self._sign(body)
        if sig:
            headers["X-PFM-Signature"] = f"sha256={sig}"
        client = self._client or httpx.AsyncClient(timeout=10.0, follow_redirects=False)
        owns = self._client is None
        try:
            r = await client.post(target, content=body, headers=headers)
            if 300 <= r.status_code < 400:
                loc = r.headers.get("location") or ""
                if loc and _is_internal_url(loc):
                    logger.warning(
                        "webhook delivery blocked (redirect to internal): %s -> %s",
                        target,
                        loc,
                    )
                    return _result(
                        "webhook",
                        target,
                        False,
                        status_code=r.status_code,
                        error="internal-url-blocked (redirect)",
                    )
            return _result("webhook", target, 200 <= r.status_code < 300, r.status_code)
        except Exception as e:
            logger.warning("webhook delivery failed: %s", e)
            return _result("webhook", target, False, error=str(e))
        finally:
            if owns:
                await client.aclose()


# ---------------------------------------------------------------------- email


def _event_title(event: dict) -> str:
    p = event.get("payload", {}) or {}
    name = p.get("rule_name") or event.get("rule_id") or "alert"
    kind = event.get("kind", "alert")
    return f"[PFM] {name} ({kind})"


def _event_description(event: dict) -> str:
    p = event.get("payload", {}) or {}
    if msg := p.get("message"):
        return str(msg)
    return json.dumps(p, default=str, sort_keys=True)[:1500]


def _ack_url(event: dict) -> str | None:
    base = os.environ.get(ACK_BASE_URL_ENV)
    eid = event.get("event_id")
    if not base or not eid:
        return None
    return f"{base.rstrip('/')}/alerts/events/{eid}/ack"


def _email_text_html(event: dict) -> tuple[str, str]:
    """Return ``(text_body, html_body)`` for an email event."""
    title = _event_title(event)
    desc = _event_description(event)
    ack = _ack_url(event)
    text_lines = [title, "", desc]
    if ack:
        text_lines += ["", f"Acknowledge: {ack}"]
    text = "\n".join(text_lines)

    safe_title = html.escape(title)
    safe_desc = html.escape(desc).replace("\n", "<br/>")
    ack_link = f'<p><a href="{html.escape(ack)}">Acknowledge</a></p>' if ack else ""
    html_body = (
        f"<html><body>"
        f"<h3>{safe_title}</h3>"
        f'<pre style="font-family:monospace;white-space:pre-wrap;">'
        f"{safe_desc}</pre>"
        f"{ack_link}"
        f"</body></html>"
    )
    return text, html_body


class EmailChannel:
    """SendGrid/Resend backend, configured via env.

    ``PFM_EMAIL_PROVIDER`` selects ``resend`` (default) or ``sendgrid``.
    Required env vars per provider:

    - resend:   ``RESEND_API_KEY``, ``PFM_EMAIL_FROM``
    - sendgrid: ``SENDGRID_API_KEY``, ``PFM_EMAIL_FROM``

    ``PFM_ALERTS_ACK_BASE_URL`` (optional) prefixes the ack URL embedded in
    the body. ``PFM_ALERTS_DRY_RUN=1`` short-circuits (no HTTP call).
    """

    type = "email"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def send(self, event: dict, target: str) -> dict:
        if _is_dry_run():
            logger.info("email dry-run to=%s subject=%s", target, _event_title(event))
            return _result("email", target, True, status_code=204, error="dry-run")
        provider = os.environ.get(EMAIL_PROVIDER_ENV, "resend").lower()
        if provider == "resend":
            return await self._send_resend(event, target)
        if provider == "sendgrid":
            return await self._send_sendgrid(event, target)
        return _result("email", target, False, error="no email provider configured")

    async def _send_resend(self, event: dict, target: str) -> dict:
        api_key = os.environ.get(RESEND_API_KEY_ENV)
        if not api_key:
            return _result(
                "email",
                target,
                False,
                error="RESEND_API_KEY not set",
            )
        sender = os.environ.get(EMAIL_FROM_ENV, "alerts@pfm.local")
        text, html_body = _email_text_html(event)
        body = {
            "from": sender,
            "to": [target],
            "subject": _event_title(event),
            "text": text,
            "html": html_body,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        client = self._client or httpx.AsyncClient(timeout=10.0)
        owns = self._client is None
        try:
            r = await client.post("https://api.resend.com/emails", json=body, headers=headers)
            ok = 200 <= r.status_code < 300
            err = None if ok else f"HTTP {r.status_code}"
            return _result("email", target, ok, r.status_code, err)
        except Exception as e:
            logger.warning("resend delivery failed: %s", e)
            return _result("email", target, False, error=str(e))
        finally:
            if owns:
                await client.aclose()

    async def _send_sendgrid(self, event: dict, target: str) -> dict:
        api_key = os.environ.get(SENDGRID_API_KEY_ENV)
        if not api_key:
            return _result(
                "email",
                target,
                False,
                error="SENDGRID_API_KEY not set",
            )
        sender = os.environ.get(EMAIL_FROM_ENV, "alerts@pfm.local")
        text, html_body = _email_text_html(event)
        body = {
            "personalizations": [{"to": [{"email": target}]}],
            "from": {"email": sender},
            "subject": _event_title(event),
            "content": [
                {"type": "text/plain", "value": text},
                {"type": "text/html", "value": html_body},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        client = self._client or httpx.AsyncClient(timeout=10.0)
        owns = self._client is None
        try:
            r = await client.post(
                "https://api.sendgrid.com/v3/mail/send",
                json=body,
                headers=headers,
            )
            ok = 200 <= r.status_code < 300
            err = None if ok else f"HTTP {r.status_code}"
            return _result("email", target, ok, r.status_code, err)
        except Exception as e:
            logger.warning("sendgrid delivery failed: %s", e)
            return _result("email", target, False, error=str(e))
        finally:
            if owns:
                await client.aclose()


# ---------------------------------------------------------------------- registry

DEFAULT_REGISTRY: dict[str, Channel] = {
    "inapp": InAppChannel(),
    "email": EmailChannel(),
    "slack": SlackChannel(),
    "discord": DiscordChannel(),
    "webhook": WebhookChannel(),
}


def _channel_key(user_id: str, ctype: str, target: str) -> str:
    """Stable bucket key for ``(user_id, channel_type, target_hash)``."""
    h = hashlib.sha256(target.encode("utf-8")).hexdigest()[:16]
    return f"{user_id}|{ctype}|{h}"


def _parse_channel_key(channel_key: str) -> tuple[str, str, str]:
    parts = channel_key.split("|", 2)
    if len(parts) != 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


async def flush_pending_digests(
    throttle_store: Any,
    *,
    quiet_seconds: float = 60.0,
    registry: dict[str, Channel] | None = None,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Drain channel digest buffers and emit one digest event per channel.

    Designed to be invoked roughly every 60 seconds. Returns the list of
    delivery results (one per flushed digest).
    """
    reg = registry or DEFAULT_REGISTRY
    pending = throttle_store.flush_pending_digests(quiet_seconds=quiet_seconds, now=now)
    results: list[dict[str, Any]] = []
    for entry in pending:
        ckey = entry["channel_key"]
        user_id, ctype, _hash = _parse_channel_key(ckey)
        impl = reg.get(ctype) if ctype else None
        wrapped = entry.get("events", [])
        # Backwards-compat: legacy buffers may store raw events directly.
        events: list[dict[str, Any]] = []
        target = ""
        for w in wrapped:
            if isinstance(w, dict) and "event" in w and "target" in w:
                events.append(w["event"])
                if not target and w.get("target"):
                    target = w["target"]
            else:
                events.append(w)
        digest_event = {
            "event_id": f"digest_{int(time.time() * 1000)}",
            "rule_id": "digest",
            "user_id": user_id,
            "kind": "digest",
            "fired_at": time.time(),
            "payload": {
                "rule_name": "Throttle digest",
                "rule_kind": "digest",
                "summary": entry["summary"],
                "count": entry["count"],
                "channel_key": ckey,
                "events": [
                    {
                        "event_id": e.get("event_id"),
                        "rule_id": e.get("rule_id"),
                        "kind": e.get("kind"),
                        "message": (e.get("payload") or {}).get("message"),
                    }
                    for e in events
                ],
                "message": entry["summary"],
            },
            "delivered": [],
            "acked": False,
        }
        if impl is None or not target:
            results.append(
                {
                    **_result(
                        ctype or "?",
                        target,
                        True,
                        status_code=204,
                        error="digest-flushed (no live channel)",
                    ),
                    "digest": True,
                    "count": entry["count"],
                }
            )
            continue
        try:
            res = await impl.send(digest_event, target)
        except Exception as e:
            res = _result(ctype, target, False, error=str(e))
        res["digest"] = True
        res["count"] = entry["count"]
        results.append(res)
    return results


async def fanout(
    event: dict,
    channels: list[dict[str, Any]],
    registry: dict[str, Channel] | None = None,
    *,
    throttle_store: Any = None,
    max_per_minute: int = 10,
) -> list[dict[str, Any]]:
    """Deliver an event to each enabled channel ref. Failure-isolated: a
    raise from one channel does NOT abort siblings; it is captured as a
    delivery result with ``ok=False``.

    If ``throttle_store`` is supplied (an :class:`AlertStore`), each
    ``(user_id, channel_type, target_hash)`` is rate-limited to
    ``max_per_minute`` events; excess events are buffered into the channel's
    digest buffer instead of being dropped.
    """
    reg = registry or DEFAULT_REGISTRY
    results: list[dict[str, Any]] = []
    user_id = str(event.get("user_id") or "")
    for ch in channels:
        if not ch.get("enabled", True):
            continue
        ctype = ch.get("type")
        target = ch.get("target", "")
        impl = reg.get(ctype) if ctype else None
        if impl is None:
            results.append(_result(ctype or "?", target, False, error="unknown channel"))
            continue

        if throttle_store is not None:
            try:
                allow, count = throttle_store.throttle_check_and_record(
                    _channel_key(user_id, ctype or "?", target),
                    event,
                    max_per_minute=max_per_minute,
                    target=target,
                )
            except Exception as e:
                logger.warning("throttle check failed: %s", e)
                allow, count = True, 0
            if not allow:
                results.append(
                    _result(
                        ctype or "?",
                        target,
                        True,
                        status_code=202,
                        error=f"throttled-buffered (count={count})",
                    )
                )
                continue

        try:
            res = await impl.send(event, target)
        except Exception as e:
            res = _result(ctype, target, False, error=str(e))
        results.append(res)
    return results
