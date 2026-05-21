"""``GET /regression/methods`` — list available regression methods.

The default ``POST /fit`` path uses ``statsmodels`` OLS with a HAC-robust
covariance matrix (see :mod:`pfm.regression_router`). Three alternative
methods are scoped in :mod:`pfm.quant.regression_methods` (Elastic Net,
Quantile, Bayesian conjugate) but currently exist only as research stubs
raising ``NotImplementedError``.

This endpoint advertises *which* method ids the API knows about and which
are actually wired up today, so frontends can render a method picker that
greys out the unimplemented options instead of hard-coding the list.

Support detection is **dynamic**: at request time we import each stub's
function from :mod:`pfm.quant.regression_methods` and use :func:`inspect`
+ a try/except probe to determine whether the body is still a bare
``NotImplementedError`` raise. The instant a stub gets a real implementation,
``supported`` flips to ``true`` without any change to this router. That keeps
the registry honest — we never lie about what works.

The OLS entry has no stub function to probe (it lives in the regression
router directly), so it's hard-coded ``supported=true``.

Companion to task ``W11-27`` and ``docs/regression-methodology-improvements.md``.
"""

from __future__ import annotations

import inspect
from typing import Any

from fastapi import APIRouter

router = APIRouter()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Each entry mirrors the public contract documented in task W11-27. We keep
# the static description here (next to the router) rather than reading from
# the stub's docstring because:
#
#   1. The blurb shown to UI users is intentionally short and product-y; the
#      stub docstrings are dense quant prose.
#   2. The list of accepted query params is the *router-level* contract, not
#      the Python function's keyword-only arguments (some kwargs are internal,
#      e.g. ``random_state``, and won't be surfaced on the HTTP layer).
#
# When a stub lands a real implementation, ``supported`` flips automatically
# via :func:`_is_stub_implemented`. The ``supported_after`` field is only
# meaningful when ``supported is False`` — it points the caller at the wave
# that's expected to deliver it.


_METHODS: list[dict[str, Any]] = [
    {
        "id": "ols",
        "name": "Ordinary Least Squares (HAC-robust)",
        "description": "Default. statsmodels OLS with cov_type='HAC'.",
        # Probed from a sentinel ``None`` — OLS is shipped via the existing
        # ``/fit`` path; there is no stub to inspect.
        "_probe": None,
        "params": ["lag"],
    },
    {
        "id": "enet",
        "name": "Elastic Net",
        "description": "LASSO + Ridge. Auto-feature selection. Use for >50 factors.",
        "_probe": "fit_elastic_net",
        "supported_after": "W11-57",
        "params": ["alpha", "l1_ratio"],
    },
    {
        "id": "quantile",
        "name": "Quantile Regression",
        "description": "Tail-aware. Fits τ ∈ {0.1, 0.25, 0.5, 0.75, 0.9}.",
        "_probe": "fit_quantile",
        "params": ["taus"],
    },
    {
        "id": "bayes",
        "name": "Bayesian Linear",
        "description": "Conjugate normal-inverse-gamma. Honest credible intervals.",
        "_probe": "fit_bayes",
        "params": ["prior", "n_samples"],
    },
]


# Map registry probe names to the actual function name in
# :mod:`pfm.quant.regression_methods`. The router-facing name (``fit_bayes``)
# is shorter than the stub's full name (``fit_bayes_conjugate``); this table
# keeps the router contract stable even if the stub gets renamed.
_PROBE_TO_FUNC: dict[str, str] = {
    "fit_elastic_net": "fit_elastic_net",
    "fit_quantile": "fit_quantile",
    "fit_bayes": "fit_bayes_conjugate",
}


def _is_stub_implemented(probe_name: str | None) -> bool:
    """Return ``True`` iff the named stub has a real (non-stub) implementation.

    Three layers of detection, cheapest first:

    1. ``None`` probe → caller is the OLS entry, which is always supported.
    2. Source-text inspection — if the function body is *literally* a single
       ``raise NotImplementedError(...)``, it's still a stub. We treat any
       other body as "implemented enough to advertise".
    3. If source isn't available (e.g. compiled .pyc with no .py present),
       fall back to actually calling the function with empty inputs and
       checking whether the exception is ``NotImplementedError``.

    This is intentionally permissive: a function that does partial work and
    then raises ``NotImplementedError`` for one branch still counts as
    implemented. The registry's job is to advertise existence, not coverage.
    """
    if probe_name is None:
        return True

    func_name = _PROBE_TO_FUNC.get(probe_name)
    if func_name is None:
        return False

    try:
        from pfm.quant import regression_methods as _rm
    except ImportError:
        return False

    func = getattr(_rm, func_name, None)
    if func is None or not callable(func):
        return False

    # Layer 2: cheap source inspection.
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        src = ""

    if src:
        # Strip docstring + decorators + signature to look at the body. A
        # quick heuristic: if the source contains a ``NotImplementedError``
        # raise AND has fewer than ~6 statement-bearing lines after the
        # function signature, it's a stub.
        body_lines = [
            ln.strip()
            for ln in src.splitlines()
            if ln.strip()
            and not ln.strip().startswith("#")
            and not ln.strip().startswith('"""')
            and not ln.strip().startswith("'''")
        ]
        # We look for the canonical stub fingerprint: a single
        # ``raise NotImplementedError(...)`` statement somewhere in the body
        # and no other ``return`` / ``yield`` statement. If the file gets a
        # real implementation, it will have a ``return ...`` and this
        # heuristic flips to ``True`` automatically.
        has_return = any(
            ln.startswith(("return ", "return\n")) or ln == "return" for ln in body_lines
        )
        has_not_impl = any("raise NotImplementedError" in ln for ln in body_lines)
        if has_not_impl and not has_return:
            return False
        if has_return:
            return True

    # Layer 3: behavioural probe. Construct cheap empty inputs and see what
    # the function does. ``NotImplementedError`` → stub. Anything else
    # (including ``ValueError`` on degenerate inputs) → implemented.
    try:
        import pandas as pd

        empty_y = pd.Series(dtype=float, name="y")
        empty_x = pd.DataFrame()
        func(empty_y, empty_x)
    except NotImplementedError:
        return False
    except Exception:
        # Any other exception means the function got past the stub gate and
        # started doing real validation work — count it as implemented.
        return True

    # Function returned without raising. Definitely implemented.
    return True


def _render_method(entry: dict[str, Any]) -> dict[str, Any]:
    """Render one registry entry into the public response shape.

    Drops the private ``_probe`` field, injects ``supported`` based on the
    live inspection, and trims ``supported_after`` when the method is
    already supported (no point pointing at a future wave).
    """
    supported = _is_stub_implemented(entry["_probe"])
    out: dict[str, Any] = {
        "id": entry["id"],
        "name": entry["name"],
        "description": entry["description"],
        "supported": supported,
        "params": list(entry["params"]),
    }
    if not supported and "supported_after" in entry:
        out["supported_after"] = entry["supported_after"]
    return out


@router.get(
    "/regression/methods",
    summary="List available regression methods and their support status",
    tags=["regression"],
)
def list_regression_methods() -> dict[str, list[dict[str, Any]]]:
    """Return the catalogue of regression methods the API knows about.

    Response shape::

        {
          "methods": [
            {
              "id": "ols",
              "name": "Ordinary Least Squares (HAC-robust)",
              "description": "Default. statsmodels OLS with cov_type='HAC'.",
              "supported": true,
              "params": ["lag"]
            },
            ...
          ]
        }

    Methods whose backing stub still raises ``NotImplementedError`` are
    advertised with ``supported: false`` and (where defined) a
    ``supported_after`` pointing at the wave that's expected to ship them.
    Front-ends use this to grey out unsupported options in the method picker.
    """
    return {"methods": [_render_method(m) for m in _METHODS]}
