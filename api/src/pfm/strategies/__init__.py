"""``pfm.strategies`` — strategies package.

Historical note: this package replaces the former ``pfm/strategies.py``
single-file module. To keep its public surface (`implication_test`,
`conditional_regression`, `frechet_bounds`, `ImplicationResult`,
`ConditionalRegressionResult`, `FrechetBoundsResult`) importable from
``pfm.strategies`` (consumed by ``pfm.scanner`` and the test suite at
``tests/test_strategies.py``), this ``__init__`` loads the legacy
``pfm/strategies.py`` file by path and re-exports every name in its
``__all__``.

This lets us add submodules under ``pfm.strategies/<name>.py`` — such as
``binary_pricing_alpha`` — without touching the legacy file (which is owned
by a different coordination scope).

If, in the future, the legacy ``pfm/strategies.py`` is removed or moved,
this loader silently no-ops; in that case the legacy symbols simply won't
be exposed via ``pfm.strategies`` and importers should be updated to the
new home.
"""

from __future__ import annotations

import importlib.util as _iu
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
# Legacy file path: ``pfm/strategies.py`` (sibling of this package directory).
_LEGACY_PATH = _os.path.normpath(_os.path.join(_HERE, "..", "strategies.py"))


def _load_legacy() -> None:
    """Load ``pfm/strategies.py`` by path and re-export its public symbols."""
    if not _os.path.exists(_LEGACY_PATH):  # pragma: no cover - defensive
        return
    legacy_name = f"{__name__}._legacy"
    if legacy_name in _sys.modules:
        legacy = _sys.modules[legacy_name]
    else:
        spec = _iu.spec_from_file_location(legacy_name, _LEGACY_PATH)
        if spec is None or spec.loader is None:  # pragma: no cover
            return
        legacy = _iu.module_from_spec(spec)
        _sys.modules[legacy_name] = legacy
        spec.loader.exec_module(legacy)
    names = getattr(legacy, "__all__", None)
    if not names:
        names = [n for n in vars(legacy) if not n.startswith("_")]
    for n in names:
        globals()[n] = getattr(legacy, n)


_load_legacy()
