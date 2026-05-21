"""Periodic arb-discovery loop.

Re-runs ``auto_discover.match_events`` every ``INTERVAL`` seconds with a
freshness bias — recently-listed markets (created in the last
``FRESH_WINDOW_DAYS`` days) get priority because the audit-confirmed
sweet spot for cross-venue arbitrage is the first ~2 weeks of a market's
life, when pricing on the two venues hasn't converged yet.

What it does each cycle:
  1. Pull Kalshi + Polymarket events (newest_first=True).
  2. Score every fresh K candidate against the full Poly pool.
  3. Merge HIGH/MED matches into the live ``markets_config.json`` so the
     running engine picks them up on its next ``_reload_configs`` tick.
  4. Skip duplicates by ``kalshi_ticker``/``poly_slug``.
  5. Write a per-run report to ``discovery_runs/<ts>.json``.

Run with:
    cd arbstuff && ../api/.venv/bin/python auto_discover_loop.py

Or as a background daemon:
    nohup ../api/.venv/bin/python auto_discover_loop.py > /tmp/disc-loop.log 2>&1 &

Tunable via env:
    PFM_DISCOVERY_INTERVAL_S   default 1800 (30 min)
    PFM_DISCOVERY_FRESH_DAYS   default 14
    PFM_DISCOVERY_MIN_TIER     default "MED" — accept HIGH/MED, skip LOW
    PFM_DISCOVERY_DRY_RUN      default 0 — when 1, never writes config
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make auto_discover importable from same dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_discover as ad

INTERVAL_S = int(os.environ.get("PFM_DISCOVERY_INTERVAL_S", "1800"))
FRESH_WINDOW_DAYS = int(os.environ.get("PFM_DISCOVERY_FRESH_DAYS", "14"))
# Ultra-fresh window — markets created in this window get a separate pass
# with a more permissive tier floor, because that's where the most arb
# pricing divergence lives (the venues haven't priced-converged yet).
ULTRA_FRESH_HOURS = int(os.environ.get("PFM_DISCOVERY_ULTRA_FRESH_HOURS", "48"))
# MED default: user prefers coverage over precision — MED has ~17% FP rate but
# the FPs surface in the UI with their confidence badge so they're easy to
# spot. Set PFM_DISCOVERY_MIN_TIER=HIGH to restore the audit-conservative
# behaviour.
MIN_TIER = os.environ.get("PFM_DISCOVERY_MIN_TIER", "MED").upper()
DRY_RUN = os.environ.get("PFM_DISCOVERY_DRY_RUN", "0") == "1"
# When 1 (default), discoveries land directly in markets_config_discovered.json
# so the engine picks them up automatically and they render under the
# "Discovered" tab in the UI. When 0, discoveries land in
# pending_discoveries.json for manual accept/reject in the UI.
AUTO_APPLY = os.environ.get("PFM_DISCOVERY_AUTO_APPLY", "1") == "1"
CONFIG_FILE = Path(__file__).parent / "markets_config.json"
DISCOVERED_FILE = Path(__file__).parent / "markets_config_discovered.json"
PENDING_FILE = Path(__file__).parent / "pending_discoveries.json"
REJECTED_FILE = Path(__file__).parent / "rejected_discoveries.json"
REPORTS_DIR = Path(__file__).parent / "discovery_runs"
REPORTS_DIR.mkdir(exist_ok=True)

TIER_ORDER = {"LOW": 0, "MED": 1, "HIGH": 2}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _age_hours(event: dict) -> float | None:
    """Hours since the event was listed, or None if unknown."""
    raw = (event.get("created_at") or event.get("createdAt")
           or event.get("creation_time") or event.get("startDate")
           or event.get("start_date") or "")
    if not raw:
        return None
    dt = ad._parse_date(raw)
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    # Use timezone-naive UTC "now" to match how _parse_date strips tzinfo above.
    from datetime import timezone as _tz
    now = datetime.now(_tz.utc).replace(tzinfo=None)
    return (now - dt).total_seconds() / 3600.0


def _is_fresh(event: dict, max_age_days: int) -> bool:
    """True when the event's created_at is within the freshness window."""
    age_h = _age_hours(event)
    if age_h is None:
        return True
    return age_h / 24.0 <= max_age_days


def _is_ultra_fresh(event: dict) -> bool:
    """True when the event is within ULTRA_FRESH_HOURS — gets MED tier
    auto-accept because the venues haven't price-converged yet and this is
    where the cross-venue spread is structurally widest."""
    age_h = _age_hours(event)
    if age_h is None:
        return False
    return age_h <= ULTRA_FRESH_HOURS


def _load_existing() -> tuple[dict, set, set]:
    """Return (config, existing_kalshi_tickers, existing_poly_slugs)."""
    try:
        cfg = json.loads(CONFIG_FILE.read_text())
    except FileNotFoundError:
        cfg = {"poll_interval": 240, "threshold": 0.94, "min_alert_profit": 1.0, "events": []}
    tickers = {e.get("kalshi_ticker") for e in cfg.get("events", []) if e.get("kalshi_ticker")}
    slugs = {e.get("poly_slug") for e in cfg.get("events", []) if e.get("poly_slug")}
    return cfg, tickers, slugs


def _load_pending() -> list:
    """Return the list of pending discoveries (or empty list)."""
    try:
        return json.loads(PENDING_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _load_pending_keys() -> set:
    """Return set of (k_ticker, p_slug) tuples already in pending file."""
    return {(p.get("kalshi_event_ticker"), p.get("poly_slug")) for p in _load_pending()}


def _load_rejected_tickers() -> set:
    """Return set of K event tickers the user has rejected — never re-queue."""
    try:
        rej = json.loads(REJECTED_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return set()
    return {r.get("kalshi_event_ticker") for r in rej if r.get("kalshi_event_ticker")}


def _append_pending(matches: list) -> None:
    """Append new matches to the pending queue, dedup'd by (ticker, slug)."""
    existing = _load_pending()
    seen = {(p.get("kalshi_event_ticker"), p.get("poly_slug")) for p in existing}
    for m in matches:
        key = (m.get("k_event_ticker"), m.get("p_slug"))
        if key in seen:
            continue
        seen.add(key)
        existing.append(_build_event_row(m))
    PENDING_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False))


def _build_event_row(match: dict) -> dict:
    """Shape a match into the engine's config-row format."""
    return {
        "name": match.get("k_title", ""),
        "kalshi_event_ticker": match.get("k_event_ticker", ""),
        "kalshi_ticker": match.get("k_event_ticker", ""),
        "poly_slug": match.get("p_slug", ""),
        "category": match.get("k_category", ""),
        "confidence": match.get("confidence", "MED"),
        "event_score": match.get("event_score", 0.0),
        "discovered_at": _now_iso(),
        "outcomes": match.get("outcome_matches", []),
    }


def cycle() -> dict:
    """One discovery cycle. Returns a per-run report dict."""
    t0 = time.time()
    cfg, existing_tickers, existing_slugs = _load_existing()

    print(f"[{_now_iso()}] fetching Kalshi + Poly events…", flush=True)
    k_events = ad.fetch_kalshi_events()
    p_events = ad.fetch_poly_events(newest_first=True)

    # Freshness filter — prioritise markets listed in the last N days.
    k_all = list(k_events)
    p_all = list(p_events)
    k_fresh = [ke for ke in k_events if _is_fresh(ke, FRESH_WINDOW_DAYS)]
    p_fresh = [pe for pe in p_events if _is_fresh(pe, FRESH_WINDOW_DAYS)]
    print(f"  Kalshi: {len(k_fresh)}/{len(k_all)} fresh (≤{FRESH_WINDOW_DAYS}d)", flush=True)
    print(f"  Poly:   {len(p_fresh)}/{len(p_all)} fresh", flush=True)

    # Three passes:
    #  (a) fresh K × ALL P — fully unconstrained match for new Kalshi events.
    #  (b) ALL K × fresh P — catches recently-listed Poly markets whose
    #      Kalshi twin already existed.
    #  (c) ultra-fresh K × ultra-fresh P — markets ≤48h old, accepted with
    #      MED tier floor (vs HIGH on (a)+(b)) because that's where the
    #      cross-venue spread is structurally widest.
    print("  Pass 1: fresh-K × all-P", flush=True)
    m1 = ad.match_events(k_fresh, p_all, min_score=0.65,
                         existing_poly_slugs=existing_slugs)
    print("  Pass 2: all-K × fresh-P", flush=True)
    fresh_slugs = {pe.get("slug") for pe in p_fresh}
    m2 = ad.match_events(k_all, p_fresh, min_score=0.65,
                         existing_poly_slugs=existing_slugs - fresh_slugs)

    # Pass (c): ultra-fresh both sides, MED accepted.
    k_ultra = [ke for ke in k_all if _is_ultra_fresh(ke)]
    p_ultra = [pe for pe in p_all if _is_ultra_fresh(pe)]
    print(f"  Pass 3: ultra-fresh ({len(k_ultra)} K × {len(p_ultra)} P, ≤{ULTRA_FRESH_HOURS}h)", flush=True)
    m3 = []
    if k_ultra and p_ultra:
        m3 = ad.match_events(k_ultra, p_ultra, min_score=0.65,
                             existing_poly_slugs=set())

    seen = set()
    merged = []
    ultra_keys = set()
    for m in m3:
        key = (m.get("k_event_ticker"), m.get("p_slug"))
        if key in seen or not all(key):
            continue
        ultra_keys.add(key)
        seen.add(key)
        merged.append(m)
    for m in m1 + m2:
        key = (m.get("k_event_ticker"), m.get("p_slug"))
        if key in seen or not all(key):
            continue
        seen.add(key)
        merged.append(m)

    # Apply tier floor. Ultra-fresh edges get a MED-tier floor regardless of
    # the global MIN_TIER (intentional bias toward newly-listed markets).
    floor = TIER_ORDER.get(MIN_TIER, 1)
    accepted = []
    for m in merged:
        key = (m.get("k_event_ticker"), m.get("p_slug"))
        local_floor = TIER_ORDER["MED"] if key in ultra_keys else floor
        if TIER_ORDER.get(m.get("confidence", "LOW"), 0) >= local_floor:
            accepted.append(m)
    skipped_low = len(merged) - len(accepted)

    # Drop matches whose K or P already lives in config (true "new" only).
    truly_new = [m for m in accepted
                 if m.get("k_event_ticker") not in existing_tickers
                 and m.get("p_slug") not in existing_slugs]

    # Persist truly_new entries.
    #   - AUTO_APPLY=1 → write directly to live config (legacy behaviour).
    #   - AUTO_APPLY=0 (default) → append to pending_discoveries.json for
    #     manual review via the UI. The user accepts/rejects each in the
    #     "Pending Discoveries" section under Strategies → Arb.
    rejected_tickers = _load_rejected_tickers()
    pending_set = _load_pending_keys()
    pending_new = [m for m in truly_new
                   if m.get("k_event_ticker") not in rejected_tickers
                   and (m.get("k_event_ticker"), m.get("p_slug")) not in pending_set]

    if pending_new and not DRY_RUN:
        if AUTO_APPLY:
            # Land in markets_config_discovered.json so engine picks up via
            # PFM_ARB_INCLUDE_SECONDARY=1 (default) and rows show up under
            # the "Discovered" tab with their confidence badge. Keep main
            # config curated (HIGH/MED audited).
            try:
                disc_cfg = json.loads(DISCOVERED_FILE.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                disc_cfg = {"events": []}
            if isinstance(disc_cfg, list):
                disc_cfg = {"events": disc_cfg}
            existing_disc_tickers = {
                e.get("kalshi_ticker") for e in disc_cfg.get("events", []) if e.get("kalshi_ticker")
            }
            new_rows = [_build_event_row(m) for m in pending_new
                        if m.get("k_event_ticker") not in existing_disc_tickers]
            disc_cfg.setdefault("events", []).extend(new_rows)
            DISCOVERED_FILE.write_text(json.dumps(disc_cfg, indent=2, ensure_ascii=False))
            print(f"  AUTO-APPLY: wrote {len(new_rows)} new events to {DISCOVERED_FILE.name}", flush=True)
        else:
            _append_pending(pending_new)
            print(f"  Queued {len(pending_new)} discoveries for manual review in {PENDING_FILE.name}", flush=True)
    elif DRY_RUN:
        print(f"  [DRY-RUN] {len(pending_new)} new events would be queued/applied", flush=True)

    elapsed = time.time() - t0
    by_tier = {t: sum(1 for m in truly_new if m.get("confidence") == t)
               for t in ("HIGH", "MED", "LOW")}
    report = {
        "ts": _now_iso(),
        "elapsed_s": round(elapsed, 1),
        "k_total": len(k_all), "k_fresh": len(k_fresh),
        "p_total": len(p_all), "p_fresh": len(p_fresh),
        "matches_seen": len(merged),
        "accepted_above_floor": len(accepted),
        "skipped_below_floor": skipped_low,
        "already_in_config": len(accepted) - len(truly_new),
        "truly_new_added": len(truly_new),
        "by_tier": by_tier,
        "config_size": len(cfg.get("events", [])),
    }
    out = REPORTS_DIR / f"{report['ts'].replace(':','-')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"  cycle done in {elapsed:.1f}s — {len(truly_new)} added "
          f"({by_tier['HIGH']}H/{by_tier['MED']}M/{by_tier['LOW']}L)", flush=True)
    return report


def main():
    print(f"=== auto_discover_loop ===", flush=True)
    print(f"  interval={INTERVAL_S}s  fresh_window={FRESH_WINDOW_DAYS}d  "
          f"min_tier={MIN_TIER}  dry_run={DRY_RUN}", flush=True)
    while True:
        try:
            cycle()
        except Exception as exc:
            print(f"[{_now_iso()}] cycle failed: {type(exc).__name__}: {exc}",
                  flush=True)
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
