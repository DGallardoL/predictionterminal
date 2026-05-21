"""Sanitize ``web/data/alpha_strategies.json``.

Some upstream pipelines (notably ``scripts/backfill_ah_sweeps.py``) default
``full_sharpe`` to ``0.0`` when the spread sample is too short or has zero
variance instead of returning ``None`` / NaN. The resulting JSON ends up with
rows where ``full_sharpe == 0`` but ``oos_sharpe`` is implausibly large — which
shows up as "noise alphas" if a user sorts by OOS Sharpe or filters to the
``D — Raw`` tier in the α Hub.

This script flags several anomaly classes and demotes the offending rows to
``D_RAW`` so the front-end can rely on an "anything in A/B/C is at least
plausible" invariant.

Anomaly classes (each emits its own ``data_quality_warning`` tag):

1. ``full_sharpe_zero_but_oos_high`` — ``full_sharpe is None`` or
   ``|full_sharpe| < ε`` AND ``oos_sharpe > 1`` (the original signal: empty
   in-sample but implausible OOS).
2. ``low_n_obs_with_sharpe`` — ``n_obs < 30`` but a non-zero Sharpe is still
   reported. 30 daily observations is the smallest sample where a Sharpe is
   even loosely meaningful; below that, the metric is noise.
3. ``half_life_out_of_range`` — ``half_life_days < 0.5`` (sub-12h mean
   reversion — degenerate, almost always a numerical artifact) or
   ``half_life_days > 365`` (the spread is not mean-reverting on any
   tradeable horizon).
4. ``sharpe_divergence`` — ``|oos_sharpe - full_sharpe| > 5`` (regime change
   or data alignment bug; either way, not deployable as-is).
5. ``duplicate_pair_id`` — applied to every occurrence after the FIRST
   instance of a repeated ``pair_id``. The first instance is left alone so
   we don't lose data; downstream consumers can dedupe by keeping rows
   without this warning.

When a row matches multiple classes the warnings are joined with ``;`` in
sorted order so the tag remains stable across runs (important for diff
review).

Run from the repo root::

    python3 api/scripts/sanitize_alpha_strategies.py

or with an explicit path::

    python3 api/scripts/sanitize_alpha_strategies.py /path/to/alpha_strategies.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Anomaly tags & thresholds — exported for tests.
# ---------------------------------------------------------------------------

WARNING_FS_ZERO_OOS_HIGH = "full_sharpe_zero_but_oos_high"
WARNING_LOW_N_OBS = "low_n_obs_with_sharpe"
WARNING_HALF_LIFE = "half_life_out_of_range"
WARNING_SHARPE_DIVERGENCE = "sharpe_divergence"
WARNING_DUPLICATE_PAIR_ID = "duplicate_pair_id"

# Legacy alias kept for callers that imported the original constant.
WARNING_TAG = WARNING_FS_ZERO_OOS_HIGH

ZERO_TOL = 0.01
OOS_FLAG_THRESHOLD = 1.0

MIN_N_OBS = 30
HALF_LIFE_MIN_DAYS = 0.5
HALF_LIFE_MAX_DAYS = 365.0
SHARPE_DIVERGENCE_THRESHOLD = 5.0


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _detect_full_sharpe_zero_oos_high(strategy: dict) -> bool:
    """Original anomaly: empty/near-zero in-sample Sharpe but loud OOS Sharpe."""
    oos = _as_float(strategy.get("oos_sharpe"))
    if oos is None or oos <= OOS_FLAG_THRESHOLD:
        return False
    fs_raw = strategy.get("full_sharpe")
    if fs_raw is None:
        return True
    fs = _as_float(fs_raw)
    if fs is None:
        return True
    return abs(fs) < ZERO_TOL


def _detect_low_n_obs_with_sharpe(strategy: dict) -> bool:
    n = _as_int(strategy.get("n_obs"))
    if n is None or n >= MIN_N_OBS:
        return False
    for key in ("full_sharpe", "oos_sharpe"):
        v = _as_float(strategy.get(key))
        if v is not None and abs(v) > ZERO_TOL:
            return True
    return False


def _detect_half_life_out_of_range(strategy: dict) -> bool:
    hl = _as_float(strategy.get("half_life_days"))
    if hl is None:
        return False
    return hl < HALF_LIFE_MIN_DAYS or hl > HALF_LIFE_MAX_DAYS


def _detect_sharpe_divergence(strategy: dict) -> bool:
    fs = _as_float(strategy.get("full_sharpe"))
    oos = _as_float(strategy.get("oos_sharpe"))
    if fs is None or oos is None:
        return False
    return abs(oos - fs) > SHARPE_DIVERGENCE_THRESHOLD


# Backwards-compatible alias for the legacy single-anomaly detector.
def _is_anomalous(strategy: dict) -> bool:
    return _detect_full_sharpe_zero_oos_high(strategy)


def _normalize_existing_warning(strategy: dict) -> None:
    """Coerce legacy boolean ``data_quality_warning: true`` into the original
    tag string so subsequent diffs are clean.

    Older pipeline versions wrote a raw boolean instead of a tag. Tests rely
    on the warning field being a tag string (or absent), so normalize on
    every pass.
    """
    existing = strategy.get("data_quality_warning")
    if existing is True:
        strategy["data_quality_warning"] = WARNING_FS_ZERO_OOS_HIGH
    elif existing is False:
        # `false` is meaningless — drop the key entirely.
        strategy.pop("data_quality_warning", None)


def _apply_warning(strategy: dict, tag: str) -> None:
    """Merge ``tag`` into ``strategy['data_quality_warning']``.

    Multiple tags are joined with ``;`` in sorted order so the resulting
    string is stable across runs (helps with diff review).
    """
    _normalize_existing_warning(strategy)
    existing = strategy.get("data_quality_warning")
    tags: set[str] = set()
    if isinstance(existing, str) and existing:
        tags.update(part.strip() for part in existing.split(";") if part.strip())
    tags.add(tag)
    strategy["data_quality_warning"] = ";".join(sorted(tags))


def sanitize(path: Path) -> dict[str, int]:
    """Run the sanitisation pass in-place and return a summary dict.

    Returns a mapping of ``warning_tag -> count_flagged`` plus the
    sentinel keys ``_total_flagged`` and ``_demoted_to_d_raw``.
    """
    payload = json.loads(path.read_text())
    strategies = payload.get("strategies", [])
    if not isinstance(strategies, list):
        raise SystemExit(f"unexpected schema: 'strategies' is {type(strategies).__name__}")

    # Pre-pass: collect pair_id occurrences so we can mark every duplicate
    # AFTER the first instance.
    pair_id_seen: set[str] = set()

    counts_by_tag: Counter[str] = Counter()
    flagged_by_old_tier: Counter[str] = Counter()
    n_demoted = 0

    for s in strategies:
        if not isinstance(s, dict):
            continue

        # Normalize legacy boolean warning fields on every row, even if no
        # new anomaly fires this pass.
        _normalize_existing_warning(s)

        local_tags: list[str] = []

        if _detect_full_sharpe_zero_oos_high(s):
            local_tags.append(WARNING_FS_ZERO_OOS_HIGH)
        if _detect_low_n_obs_with_sharpe(s):
            local_tags.append(WARNING_LOW_N_OBS)
        if _detect_half_life_out_of_range(s):
            local_tags.append(WARNING_HALF_LIFE)
        if _detect_sharpe_divergence(s):
            local_tags.append(WARNING_SHARPE_DIVERGENCE)

        pair_id = s.get("pair_id")
        if isinstance(pair_id, str) and pair_id:
            if pair_id in pair_id_seen:
                local_tags.append(WARNING_DUPLICATE_PAIR_ID)
            else:
                pair_id_seen.add(pair_id)

        if not local_tags:
            continue

        for tag in local_tags:
            counts_by_tag[tag] += 1
            _apply_warning(s, tag)

        old_tier = s.get("tier", "UNKNOWN")
        flagged_by_old_tier[old_tier] += 1
        if old_tier != "D_RAW":
            s["tier"] = "D_RAW"
            n_demoted += 1

    payload["data_quality_sanitized_at"] = payload.get("data_quality_sanitized_at")
    payload["data_quality_sanitized_pass"] = "sanitize_alpha_strategies.py"

    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    total_flagged_rows = sum(flagged_by_old_tier.values())
    print(f"sanitize_alpha_strategies: scanned {len(strategies)} strategies")
    print(f"  rows flagged (any anomaly): {total_flagged_rows}")
    for tag, count in sorted(counts_by_tag.items()):
        print(f"    {tag}: {count}")
    if flagged_by_old_tier:
        print("  flagged by old tier:")
        for tier, count in sorted(flagged_by_old_tier.items()):
            print(f"    {tier}: {count}")
    print(f"  demoted to D_RAW (non-D_RAW originally): {n_demoted}")
    print(f"  wrote: {path}")

    summary: dict[str, int] = dict(counts_by_tag)
    summary["_total_flagged_rows"] = total_flagged_rows
    summary["_demoted_to_d_raw"] = n_demoted
    return summary


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        target = Path(argv[1]).expanduser().resolve()
    else:
        target = Path(__file__).resolve().parents[2] / "web" / "data" / "alpha_strategies.json"
    if not target.exists():
        raise SystemExit(f"file not found: {target}")
    sanitize(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
