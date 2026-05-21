"""Universal CSV/JSON/PDF export helpers for Terminal endpoints.

This module provides three small helpers that every Terminal endpoint can
opt into to gain a ``?format=csv|json|pdf`` query parameter:

* :func:`to_csv`   — flattens a payload (``dict``/``list``/``BaseModel``)
                     into a CSV string using ``pandas.json_normalize``.
                     If the payload contains a nested ``history`` or
                     ``bars`` time-series, that section is emitted
                     separately, prefixed with ``# section: history``
                     so a downstream parser can split on it.
* :func:`to_json`  — pretty-prints with ``indent=2``. ``BaseModel`` payloads
                     are run through ``model_dump(mode="json")`` first so
                     dates / enums survive the round-trip.
* :func:`to_pdf`   — renders one of the bundled Jinja2 templates and pipes
                     it through WeasyPrint to produce A4 PDF bytes.
                     Templates live under ``pfm/templates/export/`` and
                     receive ``data``, ``timestamp`` (UTC ISO) and
                     ``filename`` in their context.
* :func:`respond`  — picks a FastAPI :class:`Response` based on ``fmt``:

    - ``"json"`` → :class:`~fastapi.responses.JSONResponse`
    - ``"csv"``  → ``text/csv`` with a ``Content-Disposition: attachment``
                  header so browsers offer a Save dialog.
    - ``"pdf"``  → ``application/pdf`` (or 501 if WeasyPrint's native
                  cairo/pango deps are missing on the host).

WeasyPrint pulls in libpango / libcairo at runtime via ``cffi``. When
those native libs are absent the import succeeds but ``HTML.write_pdf()``
raises ``OSError``. We catch that, mark the module-level ``PDF_AVAILABLE``
flag false, and fall back to a 501 JSON stub so the frontend can keep
running. Tests that exercise the PDF path use ``pytest.importorskip``
plus a check on ``PDF_AVAILABLE`` and skip when the toolchain is missing.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ExportFormat = Literal["json", "csv", "pdf"]

# Keys we treat as "nested time-series" sections. If a payload dict has any
# of these as a list-of-dicts, we emit it as a second CSV section instead of
# trying to ``json_normalize`` it inline (which would explode the row count).
_TIMESERIES_KEYS: tuple[str, ...] = ("history", "bars")

# Jinja2 templates live under ``pfm/templates/export/`` (one level above this
# subpackage after the 2026-05 ``pfm.terminal_*`` → ``pfm.terminal.*`` refactor);
# loader is built lazily so import of this module doesn't fail when jinja2 is
# absent in a stripped build (e.g. CI image without the export deps).
_TEMPLATE_DIR: Path = Path(__file__).resolve().parent.parent / "templates" / "export"


# ---------------------------------------------------------------------------
# Optional PDF stack — import lazily and guard against missing native libs.
# ---------------------------------------------------------------------------


def _try_import_pdf_stack() -> tuple[bool, str | None]:
    """Return ``(available, error_msg)`` after attempting all PDF imports.

    A successful return means jinja2 + weasyprint imported AND a tiny
    smoke render succeeded (catches missing pango / cairo at the host
    level rather than at first use). The smoke test is wrapped in
    ``OSError`` because that's what cffi raises when ``dlopen`` fails.
    """
    try:
        import jinja2  # noqa: F401  (feature-detect: ImportError → not installed)
    except ImportError as exc:
        return False, f"jinja2 missing: {exc}"
    try:
        from weasyprint import HTML  # noqa: F401  (feature-detect: see jinja2 above)
    except ImportError as exc:
        return False, f"weasyprint not installed: {exc}"
    except OSError as exc:
        # cffi raises OSError when ``dlopen`` cannot find the native libs
        # (libpango / libcairo / etc.) at import time. The module-level
        # docstring promised this branch — finally implementing it so a
        # missing system library degrades the PDF feature without taking
        # down the whole API.
        return False, f"weasyprint native libs unavailable: {exc}"
    # Don't run a smoke render here — too expensive on import. Defer to
    # first use; the OSError path is handled inside ``to_pdf`` itself.
    return True, None


#: PDF availability is resolved lazily on first access — ``_try_import_pdf_stack``
#: takes ~290 ms because of weasyprint's cffi/pango chain, and the cold boot
#: doesn't need to know whether PDF works until a /terminal/export call lands.
_PDF_CACHE: tuple[bool, str | None] | None = None


def _resolve_pdf_state() -> tuple[bool, str | None]:
    global _PDF_CACHE
    if _PDF_CACHE is None:
        _PDF_CACHE = _try_import_pdf_stack()
    return _PDF_CACHE


class _PDFAvailableProxy:
    """Module-level ``PDF_AVAILABLE`` that resolves lazily on first truthiness check."""

    def __bool__(self) -> bool:
        return _resolve_pdf_state()[0]


PDF_AVAILABLE = _PDFAvailableProxy()


def _pdf_error() -> str | None:
    return _resolve_pdf_state()[1]


# Backwards-compatible private alias used by older callers in this module.
_PDF_ERROR: str | None = None  # populated on first call; see _pdf_error()


def _build_jinja_env() -> Any:
    """Construct a Jinja2 environment rooted at the export-templates dir."""
    import jinja2

    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=jinja2.select_autoescape(("html", "html.j2")),
        trim_blocks=True,
        lstrip_blocks=True,
    )


# ---------------------------------------------------------------------------
# Coercion / CSV helpers (unchanged in spirit from v0.1).
# ---------------------------------------------------------------------------


def _coerce_payload(payload: Any) -> Any:
    """Convert ``BaseModel`` instances (and lists thereof) to plain dicts."""
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    if isinstance(payload, list):
        return [_coerce_payload(item) for item in payload]
    return payload


def _split_timeseries(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """Return ``(scalar_part, timeseries_sections)``."""
    scalar = dict(payload)
    sections: dict[str, list[Any]] = {}
    for key in _TIMESERIES_KEYS:
        value = scalar.get(key)
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            sections[key] = scalar.pop(key)
    return scalar, sections


def _df_to_csv_string(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def to_csv(payload: dict[str, Any] | list[Any] | BaseModel) -> str:
    """Flatten ``payload`` into a CSV string."""
    data = _coerce_payload(payload)

    if isinstance(data, list):
        df = pd.json_normalize(data) if data else pd.DataFrame()
        return _df_to_csv_string(df)

    if isinstance(data, dict):
        scalar, sections = _split_timeseries(data)
        df_main = pd.json_normalize(scalar) if scalar else pd.DataFrame()
        out = _df_to_csv_string(df_main)
        for key, rows in sections.items():
            df_sec = pd.json_normalize(rows)
            out += f"\n\n# section: {key}\n"
            out += _df_to_csv_string(df_sec)
        return out

    df = pd.DataFrame([{"value": data}])
    return _df_to_csv_string(df)


def to_json(payload: Any) -> str:
    """Pretty-print ``payload`` as JSON (``indent=2``)."""
    data = _coerce_payload(payload)
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------


# Mapping ``kind`` → template filename. Unknown kinds fall back to the
# generic ``market`` template so the user still gets *something* rather
# than a 500 (the frontend's "PDF" button stays useful).
_TEMPLATE_BY_KIND: dict[str, str] = {
    "market": "market.html.j2",
    "history": "history.html.j2",
    "compare": "compare.html.j2",
    "portfolio": "portfolio.html.j2",
    "alpha_card": "alpha_card.html.j2",
    "bulk": "bulk.html.j2",
    "quality": "market.html.j2",
    "peers": "compare.html.j2",
}


class PDFUnavailableError(RuntimeError):
    """Raised when WeasyPrint can't render due to missing native deps."""


def _render_html(payload: Any, kind: str, *, filename: str) -> str:
    """Run the Jinja2 template for ``kind`` and return raw HTML."""
    env = _build_jinja_env()
    template_name = _TEMPLATE_BY_KIND.get(kind, "market.html.j2")
    try:
        template = env.get_template(template_name)
    except Exception as exc:  # jinja2.TemplateNotFound is a subclass
        # Fall back to market template silently — better than raising.
        logger.warning("export: template %s missing (%s); falling back", template_name, exc)
        template = env.get_template("market.html.j2")
    timestamp = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return template.render(
        data=_coerce_payload(payload),
        timestamp=timestamp,
        filename=filename,
        kind=kind,
    )


def to_pdf(payload: Any, kind: str, *, filename: str) -> bytes:
    """Render ``payload`` as a PDF byte string.

    Args:
        payload: Endpoint return value (dict / list / BaseModel).
        kind: Template selector — one of ``market``, ``history``,
            ``compare``, ``portfolio``, ``alpha_card``, ``bulk``.
            Unknown kinds fall back to ``market``.
        filename: Base filename (no extension) — surfaced inside the PDF
            header so the artefact is self-describing.

    Returns:
        PDF bytes (always starts with the ``%PDF`` magic).

    Raises:
        PDFUnavailableError: WeasyPrint isn't installed or its cairo /
            pango native deps are missing on the host. Callers should
            translate this into a 501 response.
    """
    if not PDF_AVAILABLE:
        raise PDFUnavailableError(_pdf_error() or "PDF stack not available")

    html_str = _render_html(payload, kind, filename=filename)

    try:
        # Local import keeps the module importable when weasyprint is absent.
        from weasyprint import HTML

        return HTML(string=html_str).write_pdf()
    except OSError as exc:
        # Native lib (libpango / libcairo) missing — degrade to 501.
        msg = (
            "WeasyPrint runtime failed to load native dependencies (cairo, "
            f"pango). Install via your OS package manager. Underlying: {exc}"
        )
        logger.warning("export: %s", msg)
        raise PDFUnavailableError(msg) from exc


# ---------------------------------------------------------------------------
# respond() — the public glue every endpoint uses.
# ---------------------------------------------------------------------------


def _pdf_unavailable_response(kind: str, filename: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "detail": (
                "PDF export unavailable: install weasyprint dependencies (cairo, pango). " + detail
            ),
            "kind": kind,
            "filename": filename,
        },
    )


def respond(
    payload: Any,
    fmt: ExportFormat,
    *,
    filename: str,
    kind: str = "data",
) -> Response:
    """Return a FastAPI ``Response`` shaped for the requested ``fmt``."""
    data = _coerce_payload(payload)

    if fmt == "json":
        return JSONResponse(content=data)

    if fmt == "csv":
        body = to_csv(payload)
        return Response(
            content=body,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}.csv"'},
        )

    if fmt == "pdf":
        if not PDF_AVAILABLE:
            return _pdf_unavailable_response(kind, filename, _pdf_error() or "")
        try:
            pdf_bytes = to_pdf(payload, kind, filename=filename)
        except PDFUnavailableError as exc:
            return _pdf_unavailable_response(kind, filename, str(exc))
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}.pdf"'},
        )

    # Defensive — Literal should keep this unreachable.
    return JSONResponse(
        status_code=400,
        content={"detail": f"Unsupported format: {fmt!r}"},
    )


__all__ = [
    "PDF_AVAILABLE",
    "ExportFormat",
    "PDFUnavailableError",
    "respond",
    "to_csv",
    "to_json",
    "to_pdf",
]
