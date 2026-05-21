"""``POST /portfolio/import`` — parse a user-supplied CSV of holdings.

The endpoint accepts either a ``multipart/form-data`` upload (field
``file``) or a raw ``text/csv`` body. The CSV must have a header row
with the columns ``ticker``, ``shares``, and optionally ``cost_basis``
(case-insensitive). Up to 200 data rows are accepted; anything beyond
that returns ``413 Payload Too Large``.

A successful parse stores the parsed portfolio in
``request.app.state.portfolios`` keyed by an opaque handle (a
date-prefixed slug). The store is a FIFO ``OrderedDict`` capped at 100
handles — the oldest entry is evicted when the 101st is inserted. This
keeps memory bounded without requiring a backing database for the POC.

Downstream consumers (e.g. ``pfm.terminal_portfolio_sim`` or
``pfm.portfolio_optimizer_router``) can retrieve the
:class:`Portfolio` object via ``app.state.portfolios[handle]`` and use
it as the source of truth for tickers / weights / cost basis.

Path note (DEVIATION from original task)
----------------------------------------
The task originally requested ``pfm/portfolio/__init__.py`` +
``pfm/portfolio/import_router.py``. We pivoted to a single flat module
because ``pfm/portfolio.py`` already exists at the package root and
exports ``vol_targeted_combiner`` (consumed by
``pfm/strategies_router.py``). Creating a ``pfm/portfolio/`` package
would shadow that module on import and break the strategies pipeline.
The flat-module path mirrors siblings such as
``pfm.portfolio_optimizer_router``.

Integration note
----------------
``api/src/pfm/main.py`` is currently claimed by
``metrics-audit-endpoint-1778985000`` for its ``routes`` section. This
router is therefore left **standalone**. The next agent that owns
``main.py:routes`` should mount it with::

    from pfm.portfolio_import_router import router as _portfolio_router
    app.include_router(_portfolio_router)

near the other ``app.include_router(...)`` calls at the bottom of
``main.py``.

Validation rules
----------------
* ``ticker`` — uppercased and matched against ``^[A-Z]{1,5}$``.
* ``shares`` — parsed as ``float``; must be strictly positive. Fractional
  shares are accepted (Robinhood-style).
* ``cost_basis`` — optional; if present must be a non-negative ``float``.
  Missing cells trigger a per-row warning but do not fail the import.
* Empty body / no data rows after the header → ``400``.
* Malformed CSV → ``400`` with ``detail`` reporting the failing row
  number (1-indexed including the header) and the offending column.
"""

from __future__ import annotations

import csv
import io
import re
import secrets
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

router = APIRouter(tags=["portfolio"])


# --- public limits ---------------------------------------------------------

MAX_ROWS = 200
"""Hard cap on data rows accepted by ``POST /portfolio/import``."""

MAX_HANDLES = 100
"""FIFO capacity of the per-app ``app.state.portfolios`` store."""

MAX_BODY_BYTES = 256 * 1024
"""Hard cap on raw upload size (256 KiB). 200 rows of CSV stays well below."""

_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


# --- pydantic schemas ------------------------------------------------------


class PortfolioRow(BaseModel):
    """One parsed holding line — equity ticker, share count, optional cost."""

    ticker: str = Field(..., examples=["NVDA"])
    shares: float = Field(..., gt=0.0, examples=[100.0])
    cost_basis: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Total cost basis in USD for the position (not per-share). "
            "Optional; when missing the downstream simulator falls back "
            "to today's market value."
        ),
        examples=[12450.50],
    )


class ImportResponse(BaseModel):
    """Result of a successful ``POST /portfolio/import`` call."""

    handle: str = Field(
        ...,
        description="Opaque handle to retrieve the portfolio downstream.",
        examples=["pf_2026-05-16_a3f5d1"],
    )
    row_count: int = Field(..., ge=1, le=MAX_ROWS)
    tickers: list[str] = Field(..., description="Tickers in input order.")
    total_cost_basis: float = Field(
        ...,
        ge=0.0,
        description=(
            "Sum of provided cost_basis values. Rows without a cost_basis "
            "contribute 0 to this total — see warnings."
        ),
    )
    warnings: list[str] = Field(default_factory=list)


# --- internal dataclass kept on app.state ----------------------------------


@dataclass
class Portfolio:
    """Server-side handle payload — what downstream endpoints read."""

    handle: str
    rows: list[PortfolioRow]
    created_at: datetime
    warnings: list[str] = field(default_factory=list)

    @property
    def tickers(self) -> list[str]:
        return [r.ticker for r in self.rows]

    @property
    def total_cost_basis(self) -> float:
        return float(sum((r.cost_basis or 0.0) for r in self.rows))


# --- store helpers ---------------------------------------------------------


def _get_store(request: Request) -> OrderedDict[str, Portfolio]:
    """Lazily initialise ``app.state.portfolios`` as an OrderedDict."""
    store = getattr(request.app.state, "portfolios", None)
    if not isinstance(store, OrderedDict):
        store = OrderedDict()
        request.app.state.portfolios = store
    return store


def _insert_with_fifo(
    store: OrderedDict[str, Portfolio], handle: str, portfolio: Portfolio
) -> None:
    """Insert, evicting the oldest handle if at capacity."""
    store[handle] = portfolio
    while len(store) > MAX_HANDLES:
        store.popitem(last=False)


def _make_handle() -> str:
    today = datetime.now(UTC).date().isoformat()
    suffix = secrets.token_hex(3)  # 6 hex chars
    return f"pf_{today}_{suffix}"


# --- CSV parsing -----------------------------------------------------------


def _parse_csv(text: str) -> tuple[list[PortfolioRow], list[str]]:
    """Parse a CSV body. Returns (rows, warnings) or raises HTTPException."""

    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="empty CSV body",
        )

    # csv.reader copes with \r\n, \n, and quoted fields. We do header
    # detection manually so we can lowercase column names.
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="empty CSV body",
        ) from None

    header_norm = [h.strip().lower() for h in header]
    required = ("ticker", "shares")
    missing = [c for c in required if c not in header_norm]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"missing required column(s): {', '.join(missing)}",
        )

    idx_ticker = header_norm.index("ticker")
    idx_shares = header_norm.index("shares")
    idx_cost = header_norm.index("cost_basis") if "cost_basis" in header_norm else -1

    rows: list[PortfolioRow] = []
    warnings: list[str] = []
    seen_tickers: set[str] = set()

    for line_no, raw in enumerate(reader, start=2):  # header is row 1
        # Tolerate blank lines mid-file
        if not raw or all(c.strip() == "" for c in raw):
            continue

        if len(rows) >= MAX_ROWS:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"too many rows; max {MAX_ROWS}",
            )

        def _cell(i: int, row: list[str] = raw) -> str:
            return row[i].strip() if 0 <= i < len(row) else ""

        ticker_raw = _cell(idx_ticker).upper()
        shares_raw = _cell(idx_shares)
        cost_raw = _cell(idx_cost) if idx_cost >= 0 else ""

        if not ticker_raw:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"row {line_no}: missing column 'ticker'",
            )
        if not _TICKER_RE.match(ticker_raw):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"row {line_no}: column 'ticker' invalid "
                    f"(got {ticker_raw!r}; must be 1-5 uppercase letters)"
                ),
            )

        if not shares_raw:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"row {line_no}: missing column 'shares'",
            )
        try:
            shares_val = float(shares_raw)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(f"row {line_no}: column 'shares' is not a number (got {shares_raw!r})"),
            ) from None
        if not (shares_val > 0.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(f"row {line_no}: column 'shares' must be positive (got {shares_val})"),
            )

        cost_val: float | None
        if cost_raw == "":
            cost_val = None
            warnings.append(f"row {line_no}: missing cost_basis, assumed market value")
        else:
            try:
                cost_val = float(cost_raw)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"row {line_no}: column 'cost_basis' is not a number (got {cost_raw!r})"
                    ),
                ) from None
            if cost_val < 0.0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"row {line_no}: column 'cost_basis' must be non-negative (got {cost_val})"
                    ),
                )

        if ticker_raw in seen_tickers:
            warnings.append(f"row {line_no}: duplicate ticker {ticker_raw} (keeping both rows)")
        seen_tickers.add(ticker_raw)

        rows.append(
            PortfolioRow(
                ticker=ticker_raw,
                shares=shares_val,
                cost_basis=cost_val,
            )
        )

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="no data rows after header",
        )

    return rows, warnings


async def _read_body(request: Request) -> str:
    """Pull raw body, multipart payload, or fail with 400."""
    content_type = (request.headers.get("content-type") or "").lower()

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not isinstance(upload, UploadFile):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="multipart request missing 'file' field",
            )
        raw = await upload.read()
    else:
        raw = await request.body()

    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="empty CSV body",
        )
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"payload too large; max {MAX_BODY_BYTES} bytes",
        )

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV body is not valid UTF-8",
        ) from None


# --- endpoint --------------------------------------------------------------


@router.post(
    "/portfolio/import",
    response_model=ImportResponse,
    summary="Import a user-supplied CSV of equity holdings.",
    description=(
        "Parse a CSV with header `ticker, shares, cost_basis` (cost_basis "
        "optional). Returns an opaque handle that can be passed to "
        "downstream Terminal endpoints — notably the portfolio simulator "
        "— without re-uploading the file. Accepts either multipart "
        "upload (field name `file`) or a raw `text/csv` body."
    ),
)
async def import_portfolio(request: Request) -> ImportResponse:
    text = await _read_body(request)
    rows, warnings = _parse_csv(text)

    handle = _make_handle()
    portfolio = Portfolio(
        handle=handle,
        rows=rows,
        created_at=datetime.now(UTC),
        warnings=list(warnings),
    )

    store = _get_store(request)
    _insert_with_fifo(store, handle, portfolio)

    return ImportResponse(
        handle=handle,
        row_count=len(rows),
        tickers=portfolio.tickers,
        total_cost_basis=portfolio.total_cost_basis,
        warnings=warnings,
    )


# --- public helpers (for downstream consumers) -----------------------------


def get_portfolio(app_state: Any, handle: str) -> Portfolio:
    """Look up a stored portfolio or raise ``HTTPException(404)``.

    Downstream modules can do::

        from pfm.portfolio_import_router import get_portfolio
        pf = get_portfolio(request.app.state, handle)
        tickers = pf.tickers
    """
    store = getattr(app_state, "portfolios", None)
    if not isinstance(store, OrderedDict) or handle not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown portfolio handle {handle!r}",
        )
    return store[handle]
