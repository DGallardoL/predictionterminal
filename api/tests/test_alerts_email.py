"""Tests for the EmailChannel (Resend + SendGrid backends + dry-run)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from pfm.alerts.channels import EmailChannel


def _event() -> dict:
    return {
        "event_id": "evt_abc",
        "rule_id": "rule_1",
        "user_id": "u1",
        "kind": "price_cross",
        "fired_at": 1000.0,
        "payload": {
            "rule_name": "AAPL crossing",
            "rule_kind": "price_cross",
            "message": "AAPL > 0.5",
        },
        "delivered": [],
        "acked": False,
    }


# ---------------------------------------------------------------- Resend


@respx.mock
def test_email_resend_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFM_EMAIL_PROVIDER", "resend")
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("PFM_EMAIL_FROM", "alerts@pfm.test")
    monkeypatch.delenv("PFM_ALERTS_DRY_RUN", raising=False)

    route = respx.post("https://api.resend.com/emails").mock(
        return_value=httpx.Response(200, json={"id": "re_evt_123"})
    )

    res = asyncio.run(EmailChannel().send(_event(), "user@example.com"))
    assert route.called
    assert res["ok"] is True
    assert res["channel"] == "email"
    assert res["status_code"] == 200

    # Inspect the request body sent to Resend.
    sent = route.calls[0].request
    body = sent.read().decode()
    assert "user@example.com" in body
    assert "AAPL crossing" in body
    assert sent.headers["authorization"] == "Bearer re_test_key"


def test_email_resend_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFM_EMAIL_PROVIDER", "resend")
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.delenv("PFM_ALERTS_DRY_RUN", raising=False)

    res = asyncio.run(EmailChannel().send(_event(), "user@example.com"))
    assert res["ok"] is False
    assert res["channel"] == "email"
    assert "RESEND_API_KEY" in (res["error"] or "")


@respx.mock
def test_email_resend_4xx_marks_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFM_EMAIL_PROVIDER", "resend")
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.delenv("PFM_ALERTS_DRY_RUN", raising=False)
    respx.post("https://api.resend.com/emails").mock(
        return_value=httpx.Response(401, json={"error": "bad key"})
    )
    res = asyncio.run(EmailChannel().send(_event(), "user@example.com"))
    assert res["ok"] is False
    assert res["status_code"] == 401
    assert "401" in (res["error"] or "")


# ---------------------------------------------------------------- SendGrid


@respx.mock
def test_email_sendgrid_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFM_EMAIL_PROVIDER", "sendgrid")
    monkeypatch.setenv("SENDGRID_API_KEY", "SG.test_key")
    monkeypatch.setenv("PFM_EMAIL_FROM", "alerts@pfm.test")
    monkeypatch.delenv("PFM_ALERTS_DRY_RUN", raising=False)

    route = respx.post("https://api.sendgrid.com/v3/mail/send").mock(
        return_value=httpx.Response(202, json={})
    )

    res = asyncio.run(EmailChannel().send(_event(), "user@example.com"))
    assert route.called
    assert res["ok"] is True
    assert res["status_code"] == 202

    sent = route.calls[0].request
    body = sent.read().decode()
    assert "user@example.com" in body
    assert "AAPL crossing" in body
    assert sent.headers["authorization"] == "Bearer SG.test_key"


def test_email_sendgrid_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFM_EMAIL_PROVIDER", "sendgrid")
    monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
    monkeypatch.delenv("PFM_ALERTS_DRY_RUN", raising=False)

    res = asyncio.run(EmailChannel().send(_event(), "user@example.com"))
    assert res["ok"] is False
    assert "SENDGRID_API_KEY" in (res["error"] or "")


# ---------------------------------------------------------------- Dry-run


def test_email_dry_run_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFM_ALERTS_DRY_RUN", "1")
    monkeypatch.setenv("PFM_EMAIL_PROVIDER", "resend")
    monkeypatch.delenv("RESEND_API_KEY", raising=False)

    res = asyncio.run(EmailChannel().send(_event(), "user@example.com"))
    assert res["ok"] is True
    assert res["error"] == "dry-run"


def test_email_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PFM_EMAIL_PROVIDER", "mailgun")
    monkeypatch.delenv("PFM_ALERTS_DRY_RUN", raising=False)

    res = asyncio.run(EmailChannel().send(_event(), "user@example.com"))
    assert res["ok"] is False
    assert "no email provider configured" in (res["error"] or "")


# ---------------------------------------------------------------- ack URL


@respx.mock
def test_email_includes_ack_url_when_base_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PFM_EMAIL_PROVIDER", "resend")
    monkeypatch.setenv("RESEND_API_KEY", "rk")
    monkeypatch.setenv("PFM_ALERTS_ACK_BASE_URL", "https://api.pfm.test")
    monkeypatch.delenv("PFM_ALERTS_DRY_RUN", raising=False)

    route = respx.post("https://api.resend.com/emails").mock(
        return_value=httpx.Response(200, json={"id": "x"})
    )
    asyncio.run(EmailChannel().send(_event(), "to@example.com"))
    body = route.calls[0].request.read().decode()
    assert "/alerts/events/evt_abc/ack" in body
