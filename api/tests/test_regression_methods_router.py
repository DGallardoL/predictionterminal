"""Tests for ``GET /regression/methods`` (task W11-27).

The router advertises the catalogue of regression methods the API knows
about. The contract under test:

* HTTP shape: ``{"methods": [...]}`` with one entry per known method.
* Static identity: ``ols``, ``enet``, ``quantile``, ``bayes`` are always
  present with the descriptions documented in task W11-27.
* Dynamic support flag: ``supported: true`` only for methods whose backing
  implementation in :mod:`pfm.quant.regression_methods` is not a bare
  ``NotImplementedError`` stub. We verify this by introspecting the stub
  module at test time — not by hard-coding "today only OLS is supported",
  which would silently rot the moment a stub gets implemented.
* ``supported_after`` only appears for unsupported methods that declare it.

The tests run against a standalone ``FastAPI`` app that mounts only the
router under test. That keeps them fast (~milliseconds), independent of
``pfm.main`` lifespan, and immune to the global gunicorn restart hazard
called out in the multi-session protocol.
"""

from __future__ import annotations

import importlib
import inspect

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.quant import regression_methods as _rm_stubs
from pfm.quant.regression_methods_router import (
    _METHODS,
    _PROBE_TO_FUNC,
    _is_stub_implemented,
    router,
)

REQUIRED_FIELDS = {"id", "name", "description", "supported", "params"}
EXPECTED_IDS = ("ols", "enet", "quantile", "bayes")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """Isolated FastAPI app with only the methods router mounted.

    Avoids importing ``pfm.main`` so the test is decoupled from the
    long-running lifespan + 271-endpoint surface area.
    """
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# HTTP-shape tests
# ---------------------------------------------------------------------------


def test_get_methods_returns_200(client: TestClient) -> None:
    """Endpoint is reachable and returns the documented top-level shape."""
    resp = client.get("/regression/methods")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)
    assert "methods" in body
    assert isinstance(body["methods"], list)
    assert len(body["methods"]) == 4


def test_each_method_has_required_fields(client: TestClient) -> None:
    """Every entry exposes id/name/description/supported/params."""
    body = client.get("/regression/methods").json()
    for entry in body["methods"]:
        missing = REQUIRED_FIELDS - set(entry.keys())
        assert not missing, f"method {entry.get('id')!r} missing fields {missing}"
        assert isinstance(entry["id"], str) and entry["id"]
        assert isinstance(entry["name"], str) and entry["name"]
        assert isinstance(entry["description"], str) and entry["description"]
        assert isinstance(entry["supported"], bool)
        assert isinstance(entry["params"], list)
        for p in entry["params"]:
            assert isinstance(p, str) and p


def test_method_ids_exact_set(client: TestClient) -> None:
    """The four documented ids are present, in the documented order."""
    body = client.get("/regression/methods").json()
    ids = tuple(m["id"] for m in body["methods"])
    assert ids == EXPECTED_IDS


def test_descriptions_match_task_contract(client: TestClient) -> None:
    """Pin the user-visible blurbs from task W11-27.

    These strings ship into the frontend method picker; changing them is a
    UX-visible contract change and should require an intentional test edit.
    """
    body = client.get("/regression/methods").json()
    by_id = {m["id"]: m for m in body["methods"]}
    assert by_id["ols"]["description"] == ("Default. statsmodels OLS with cov_type='HAC'.")
    assert by_id["enet"]["description"] == (
        "LASSO + Ridge. Auto-feature selection. Use for >50 factors."
    )
    # Greek tau (U+03C4) and "element of" symbol (U+2208) must survive JSON.
    assert "Tail-aware" in by_id["quantile"]["description"]
    assert "τ" in by_id["quantile"]["description"]
    assert by_id["bayes"]["description"] == (
        "Conjugate normal-inverse-gamma. Honest credible intervals."
    )


# ---------------------------------------------------------------------------
# Support-flag tests (the dynamic part of the contract)
# ---------------------------------------------------------------------------


def test_ols_is_supported(client: TestClient) -> None:
    """OLS always reports ``supported: true`` — it ships via the regression router."""
    body = client.get("/regression/methods").json()
    ols = next(m for m in body["methods"] if m["id"] == "ols")
    assert ols["supported"] is True
    # OLS has no future-wave promise to make.
    assert "supported_after" not in ols


def test_unimplemented_stubs_marked_unsupported(client: TestClient) -> None:
    """Reflect the actual ``NotImplementedError`` status of each stub.

    For every non-OLS method we look up the backing function in
    :mod:`pfm.quant.regression_methods` and inspect whether its body is the
    canonical stub fingerprint (single ``raise NotImplementedError``). The
    router's ``supported`` flag must match that determination.

    This is the test that auto-flips when a stub lands a real implementation
    — there is no list of names to keep in sync.
    """
    body = client.get("/regression/methods").json()
    by_id = {m["id"]: m for m in body["methods"]}

    method_to_func = {
        "enet": "fit_elastic_net",
        "quantile": "fit_quantile",
        "bayes": "fit_bayes_conjugate",
    }

    for method_id, func_name in method_to_func.items():
        func = getattr(_rm_stubs, func_name)
        src = inspect.getsource(func)
        is_stub = (
            "raise NotImplementedError" in src
            and "    return " not in src
            and "\treturn " not in src
        )
        expected_supported = not is_stub
        actual_supported = by_id[method_id]["supported"]
        assert actual_supported is expected_supported, (
            f"method {method_id!r}: router says supported={actual_supported} "
            f"but stub-inspection says supported={expected_supported}"
        )


def test_supported_after_only_on_unsupported(client: TestClient) -> None:
    """``supported_after`` is meaningful only when ``supported is False``.

    Per the task contract, ``enet`` carries ``supported_after: "W11-57"``.
    Once supported, the field must be dropped to avoid stale promises.
    """
    body = client.get("/regression/methods").json()
    for entry in body["methods"]:
        if entry.get("supported_after") is not None and "supported_after" in entry:
            assert entry["supported"] is False, (
                f"{entry['id']!r} carries supported_after while supported=True"
            )

    by_id = {m["id"]: m for m in body["methods"]}
    if not by_id["enet"]["supported"]:
        assert by_id["enet"].get("supported_after") == "W11-57"


def test_params_match_task_contract(client: TestClient) -> None:
    """Each method advertises its documented query-parameter names."""
    body = client.get("/regression/methods").json()
    by_id = {m["id"]: m for m in body["methods"]}
    assert by_id["ols"]["params"] == ["lag"]
    assert by_id["enet"]["params"] == ["alpha", "l1_ratio"]
    assert by_id["quantile"]["params"] == ["taus"]
    assert by_id["bayes"]["params"] == ["prior", "n_samples"]


# ---------------------------------------------------------------------------
# Internal-helper tests (kept small; cover branches the HTTP layer can't see)
# ---------------------------------------------------------------------------


def test_is_stub_implemented_none_probe_returns_true() -> None:
    """The OLS sentinel (``_probe = None``) is treated as always supported."""
    assert _is_stub_implemented(None) is True


def test_is_stub_implemented_unknown_probe_returns_false() -> None:
    """A probe name not in the lookup table cannot be supported."""
    assert _is_stub_implemented("fit_nonexistent_method") is False


def test_is_stub_implemented_flips_when_stub_gets_real_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a stub is replaced with a real-looking body, support flips to True.

    Simulates the post-W11-57 world by swapping ``fit_elastic_net`` for a
    function whose source contains a ``return`` statement. The detector
    must observe the change and report ``supported is True`` without any
    other code edit.
    """
    rm_mod = importlib.import_module("pfm.quant.regression_methods")

    def fake_fit_elastic_net(y, x, **kwargs):
        """Pretend-implemented elastic net."""
        return {"coefficients": {}, "intercept": 0.0}

    monkeypatch.setattr(rm_mod, "fit_elastic_net", fake_fit_elastic_net)
    assert _is_stub_implemented("fit_elastic_net") is True


def test_probe_table_covers_all_unsupported_methods() -> None:
    """Every non-OLS registry entry maps to a real function in the stubs module.

    Catches typos in ``_PROBE_TO_FUNC`` that would silently make a method
    appear unsupported even after its stub lands a real implementation.
    """
    for entry in _METHODS:
        probe = entry["_probe"]
        if probe is None:
            continue
        assert probe in _PROBE_TO_FUNC, (
            f"registry probe {probe!r} for id={entry['id']!r} not in _PROBE_TO_FUNC"
        )
        func_name = _PROBE_TO_FUNC[probe]
        assert hasattr(_rm_stubs, func_name), (
            f"_PROBE_TO_FUNC points {probe!r} → {func_name!r} but the function "
            "doesn't exist in pfm.quant.regression_methods"
        )
