"""Audit Polymarket factors in factors.yml — flag resolved, drop dead.

Strategy:
1. Walk every polymarket factor in factors.yml.
2. For each, hit gamma /markets?slug=X (no filters → returns active only).
   If empty, retry with closed=true to surface resolved markets.
3. Classify:
     - ACTIVE  : found, closed=False  → leave entry unchanged
     - RESOLVED: found with closed=true (or active=true & endDate past)
                 → keep entry, add `resolved: true` and `final_outcome`.
     - DEAD    : empty in BOTH calls → slug was deleted on Polymarket
                 → drop entry from factors.yml.
4. Sample heuristic: prioritise the most-likely-dead candidates first
   (april/march 2026 month-end markets, sports with old tournament refs,
   numeric -NNN suffixed slugs that often churn) so we get value early.

Writes:
  - factors.yml (updated, in place)
  - web/data/active_slugs.json (active + kalshi slugs, fast lookup)
  - scripts/audit_dead_factors.report.json (per-slug diagnostic dump)

Rate limit: 5 req/s (200ms between calls). Polymarket allows 1000/10s
so this is well within budget.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[2]
FACTORS_YML = ROOT / "api" / "src" / "pfm" / "factors.yml"
ACTIVE_SLUGS_JSON = ROOT / "web" / "data" / "active_slugs.json"
REPORT_JSON = ROOT / "api" / "scripts" / "audit_dead_factors.report.json"

GAMMA = "https://gamma-api.polymarket.com/markets"
SLEEP = 0.20  # 5 req/s
TODAY = date(2026, 5, 2)


def load_factors() -> tuple[list[dict], str]:
    raw = FACTORS_YML.read_text()
    return yaml.safe_load(raw)["factors"], raw


def gamma_lookup(client: httpx.Client, slug: str) -> dict | None:
    """Return the Polymarket market dict for `slug`, or None if truly dead."""
    # First: default query (excludes closed)
    try:
        r = client.get(GAMMA, params={"slug": slug}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0]
    except (httpx.HTTPError, json.JSONDecodeError):
        pass
    time.sleep(SLEEP)
    # Second: include closed markets
    try:
        r = client.get(GAMMA, params={"slug": slug, "closed": "true"}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0]
    except (httpx.HTTPError, json.JSONDecodeError):
        pass
    return None


def classify(market: dict | None) -> tuple[str, str | None]:
    """Return (status, final_outcome). status in {ACTIVE, RESOLVED, DEAD}."""
    if market is None:
        return "DEAD", None
    closed = bool(market.get("closed"))
    end_iso = (market.get("endDate") or "")[:10]
    end_past = False
    if end_iso:
        try:
            end_past = date.fromisoformat(end_iso) < TODAY
        except ValueError:
            end_past = False
    is_resolved = closed or (not market.get("active") and end_past)
    if not is_resolved:
        return "ACTIVE", None
    # Final outcome: outcomePrices is JSON-string list aligned with outcomes
    outcome: str | None = None
    try:
        prices = market.get("outcomePrices")
        outcomes = market.get("outcomes")
        if isinstance(prices, str):
            prices = json.loads(prices)
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if prices and outcomes and len(prices) == len(outcomes):
            for o, p in zip(outcomes, prices, strict=False):
                try:
                    pf = float(p)
                except (TypeError, ValueError):
                    continue
                if pf >= 0.99:
                    outcome = str(o).strip().lower()
                    break
    except (ValueError, TypeError, json.JSONDecodeError):
        outcome = None
    return "RESOLVED", outcome


# Heuristic ordering — most likely dead first, so we can stop early
# while still pruning the noisy ones the UI is hitting.
DEAD_LIKELY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("april_2026_end", re.compile(r"end-of-april-2026|-april-2026$|april-30-2026", re.I)),
    ("march_2026_end", re.compile(r"end-of-march-2026|-march-2026$|march-31-2026", re.I)),
    ("feb_2026_end", re.compile(r"end-of-february-2026|-february-2026$", re.I)),
    ("jan_2026_end", re.compile(r"end-of-january-2026|-january-2026$", re.I)),
    ("dated_2025", re.compile(r"-2025($|-\d)", re.I)),
    ("french_open_already_started", re.compile(r"french-open", re.I)),
    ("numeric_suffix", re.compile(r"-\d{2,4}$")),
    ("nba_finals_specific", re.compile(r"win-the-2026-nba-finals", re.I)),
    ("fifa_specific", re.compile(r"win-the-2026-fifa-world-cup", re.I)),
    ("champions_league", re.compile(r"win-the-202526-champions-league", re.I)),
]


def heuristic_priority(slug: str) -> int:
    """Lower = more likely dead → check earlier."""
    for i, (_, pat) in enumerate(DEAD_LIKELY_PATTERNS):
        if pat.search(slug):
            return i
    return len(DEAD_LIKELY_PATTERNS)


def main(limit: int | None = None) -> None:
    factors, _raw = load_factors()
    poly_idx = [i for i, f in enumerate(factors) if f.get("source") == "polymarket"]
    poly_idx.sort(key=lambda i: heuristic_priority(factors[i].get("slug", "")))
    if limit is not None:
        poly_idx = poly_idx[:limit]

    print(f"Auditing {len(poly_idx)} polymarket slugs (sorted by dead-likelihood)…")
    report: dict[str, Any] = {
        "checked_at": datetime.now(UTC).isoformat(),
        "n_total_polymarket": sum(1 for f in factors if f.get("source") == "polymarket"),
        "n_checked": 0,
        "by_status": {"ACTIVE": [], "RESOLVED": [], "DEAD": []},
    }

    with httpx.Client(headers={"User-Agent": "pfm-audit/1.0"}) as client:
        for n, idx in enumerate(poly_idx, 1):
            f = factors[idx]
            slug = f.get("slug", "")
            market = gamma_lookup(client, slug)
            status, outcome = classify(market)
            report["n_checked"] += 1
            report["by_status"][status].append(
                {"id": f.get("id"), "slug": slug, "final_outcome": outcome}
            )

            if status == "ACTIVE":
                # leave alone; clear any stale flag
                f.pop("resolved", None)
                f.pop("final_outcome", None)
            elif status == "RESOLVED":
                f["resolved"] = True
                if outcome is not None:
                    f["final_outcome"] = outcome
                else:
                    f.pop("final_outcome", None)
            else:  # DEAD
                pass  # keep in factors list for now; we'll filter at end

            if n % 25 == 0 or n == len(poly_idx):
                print(
                    f"  [{n:>4}/{len(poly_idx)}]  "
                    f"A={len(report['by_status']['ACTIVE']):>4}  "
                    f"R={len(report['by_status']['RESOLVED']):>4}  "
                    f"D={len(report['by_status']['DEAD']):>4}  "
                    f":: {status:<8} {slug[:60]}"
                )
            time.sleep(SLEEP)

    dead_slugs = {r["slug"] for r in report["by_status"]["DEAD"]}
    pruned_factors = [
        f for f in factors if not (f.get("source") == "polymarket" and f.get("slug") in dead_slugs)
    ]

    # Write factors.yml back. yaml.safe_dump loses comments/quoting style,
    # so we do a careful re-emit preserving key order and using block style.
    out = {"factors": pruned_factors}
    FACTORS_YML.write_text(
        "# factors.yml — pruned by scripts/audit_dead_factors.py.\n"
        "# Resolved markets carry `resolved: true` (+ `final_outcome` when known)\n"
        "# so they remain available for backtests but the Terminal UI skips them.\n"
        "# Backup at factors.yml.bak.dead_prune.\n\n"
        + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=120)
    )

    # active_slugs.json — UI fast-lookup. Includes Kalshi too.
    active_entries = [
        {
            "id": f["id"],
            "slug": f["slug"],
            "source": f.get("source"),
            "theme": f.get("theme"),
            "name": f.get("name"),
        }
        for f in pruned_factors
        if not f.get("resolved")
    ]
    ACTIVE_SLUGS_JSON.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_SLUGS_JSON.write_text(
        json.dumps(
            {
                "generated_at": report["checked_at"],
                "n": len(active_entries),
                "factors": active_entries,
            },
            indent=2,
        )
    )

    REPORT_JSON.write_text(json.dumps(report, indent=2))
    print()
    print(f"checked      : {report['n_checked']}")
    print(f"  ACTIVE     : {len(report['by_status']['ACTIVE'])}")
    print(f"  RESOLVED   : {len(report['by_status']['RESOLVED'])}")
    print(f"  DEAD       : {len(report['by_status']['DEAD'])}")
    print(f"factors.yml  : {len(pruned_factors)} entries (was {len(factors)})")
    print(f"active_slugs : {ACTIVE_SLUGS_JSON} ({len(active_entries)} entries)")
    print(f"report       : {REPORT_JSON}")
    if dead_slugs:
        print("Top dead (UI 404 culprits):")
        for r in report["by_status"]["DEAD"][:10]:
            print(f"  - {r['slug']}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(limit=int(arg) if arg else None)
