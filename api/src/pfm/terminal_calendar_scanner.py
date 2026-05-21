"""Compat shim — module moved to ``pfm.terminal.calendar_scanner`` in 2026-05 refactor."""

from __future__ import annotations

import sys as _sys

from pfm.terminal import calendar_scanner as _new

# Alias this legacy module to the new location so attribute access (and
# monkeypatch.setattr) operates on the real module, not a stale namespace.
_sys.modules[__name__] = _new
