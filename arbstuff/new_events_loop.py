"""Fast-poll loop dedicated to newly-listed markets.

Runs every PFM_NEW_EVENTS_INTERVAL_S (default 300 = 5 min). Unlike the main
auto_discover_loop (30 min cadence, ≤14d window), this loop only looks at
markets created in the **last 24 hours** on either venue and pushes any
match to pending_discoveries.json with a LOW tier floor.

Rationale: the first 24-72h of a new market is when MM-seed mispricing is
maximal — the seed liquidity sits unmoved while real flow hasn't arrived.
Catching those pairings within minutes (not 30 min) means more of the seed
size is still on the book.

Run alongside auto_discover_loop.py:
    nohup ../api/.venv/bin/python new_events_loop.py > /tmp/new-events.log 2>&1 &
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_discover as ad
import auto_discover_loop as adl

INTERVAL_S = int(os.environ.get("PFM_NEW_EVENTS_INTERVAL_S", "300"))
NEW_WINDOW_HOURS = int(os.environ.get("PFM_NEW_EVENTS_HOURS", "24"))
# MIN_SCORE much lower than the main loop's 0.65 — MM-seed arbs work on
# mispricing not title match, so we want to surface candidates that might
# normally fall below the bi-encoder threshold.
MIN_SCORE = float(os.environ.get("PFM_NEW_EVENTS_MIN_SCORE", "0.55"))
TIER_FLOOR = os.environ.get("PFM_NEW_EVENTS_TIER", "LOW").upper()
PENDING_FILE = Path(__file__).parent / "pending_discoveries.json"


def _is_new(event: dict) -> bool:
    age_h = adl._age_hours(event)
    if age_h is None:
        return False
    return age_h <= NEW_WINDOW_HOURS


def cycle() -> dict:
    t0 = time.time()
    print(f"[{adl._now_iso()}] fetching for new-events scan (last {NEW_WINDOW_HOURS}h)…", flush=True)
    k_events = ad.fetch_kalshi_events()
    p_events = ad.fetch_poly_events(newest_first=True)

    k_new = [ke for ke in k_events if _is_new(ke)]
    p_new = [pe for pe in p_events if _is_new(pe)]
    print(f"  K new: {len(k_new)}  P new: {len(p_new)}", flush=True)

    if not k_new and not p_new:
        return {"ts": adl._now_iso(), "added": 0, "k_new": 0, "p_new": 0}

    # Match new-on-both-sides first (most likely MM-seed pairs), then
    # cross-match each side against the full opposite pool.
    all_matches = []
    if k_new and p_new:
        all_matches.extend(ad.match_events(k_new, p_new, min_score=MIN_SCORE,
                                           existing_poly_slugs=set()))
    if k_new:
        all_matches.extend(ad.match_events(k_new, p_events, min_score=MIN_SCORE,
                                           existing_poly_slugs=set()))
    if p_new:
        all_matches.extend(ad.match_events(k_events, p_new, min_score=MIN_SCORE,
                                           existing_poly_slugs=set()))

    # Dedup by (k_ticker, p_slug).
    seen, deduped = set(), []
    for m in all_matches:
        key = (m.get("k_event_ticker"), m.get("p_slug"))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        deduped.append(m)

    # Filter by tier floor. LOW means accept all confidence levels.
    floor = adl.TIER_ORDER.get(TIER_FLOOR, 0)
    accepted = [m for m in deduped
                if adl.TIER_ORDER.get(m.get("confidence", "LOW"), 0) >= floor]

    # Skip rejected + already-in-config + already-pending.
    rejected = adl._load_rejected_tickers()
    pending_keys = adl._load_pending_keys()
    in_config_tickers = {e.get("kalshi_event_ticker") for e in
                         json.loads(adl.CONFIG_FILE.read_text()).get("events", [])}
    in_config_slugs = {e.get("poly_slug") for e in
                       json.loads(adl.CONFIG_FILE.read_text()).get("events", [])}
    novel = [m for m in accepted
             if m.get("k_event_ticker") not in rejected
             and (m.get("k_event_ticker"), m.get("p_slug")) not in pending_keys
             and m.get("k_event_ticker") not in in_config_tickers
             and m.get("p_slug") not in in_config_slugs]

    if novel:
        # Tag each as MM-seed candidate so the UI shows a badge.
        for m in novel:
            m["source"] = "new-events-fast"
        adl._append_pending(novel)
        print(f"  Queued {len(novel)} new-events matches in {PENDING_FILE.name}", flush=True)

    elapsed = time.time() - t0
    return {
        "ts": adl._now_iso(),
        "elapsed_s": round(elapsed, 1),
        "k_new": len(k_new), "p_new": len(p_new),
        "matches_seen": len(deduped),
        "added": len(novel),
    }


def main():
    print(f"=== new_events_loop ===", flush=True)
    print(f"  interval={INTERVAL_S}s  window={NEW_WINDOW_HOURS}h  "
          f"min_score={MIN_SCORE}  tier_floor={TIER_FLOOR}", flush=True)
    while True:
        try:
            report = cycle()
            print(f"  cycle: {report.get('added', 0)} added "
                  f"(k_new={report.get('k_new', 0)} p_new={report.get('p_new', 0)})",
                  flush=True)
        except Exception as exc:
            print(f"[{adl._now_iso()}] cycle failed: {type(exc).__name__}: {exc}",
                  flush=True)
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
