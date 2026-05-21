"""
politics_discover.py — Discover ALL US political markets across all 50 states
and match them with high precision using politics_matcher.

Workflow:
  1. Fetch all open Kalshi events
  2. Fetch all active Polymarket events
  3. Parse each into a PoliticalRace (states, offices, districts)
  4. For each Kalshi political race, find Poly races with identical structure
  5. Build outcome mappings (party-based for general, candidate-based for primaries)
  6. Write results to markets_config_politics.json

Usage:
    python politics_discover.py                     # discover, write config
    python politics_discover.py --dry-run           # don't write, just print
    python politics_discover.py --state OH          # only specific state
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

from politics_matcher import (
    parse_political_title, PoliticalRace, STATES,
    build_party_mapping, build_candidate_mapping,
    RACE_GENERAL, RACE_PRIMARY, RACE_SPECIAL, RACE_RUNOFF,
)

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA_BASE = "https://gamma-api.polymarket.com"
OUTPUT_FILE = "markets_config_politics.json"


# ════════════════════════════════════════════════════════════════════════════
# Fetchers
# ════════════════════════════════════════════════════════════════════════════

def _kalshi_headers():
    """Build authenticated headers for Kalshi (re-sign per call)."""
    key_id = os.getenv("KALSHI_API_KEY_ID")
    if not key_id:
        return {}
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        import base64
        pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
        if not pk_path:
            return {}
        with open(pk_path, "rb") as f:
            pem = serialization.load_pem_private_key(f.read(), password=None)
        ts = str(int(time.time() * 1000))
        msg = f"{ts}GET/trade-api/v2/events"
        sig = pem.sign(msg.encode(),
                       padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                   salt_length=padding.PSS.DIGEST_LENGTH),
                       hashes.SHA256())
        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }
    except Exception as e:
        print(f"  [warn] Kalshi auth failed: {e}", flush=True)
        return {}


def fetch_kalshi_events() -> list:
    """Fetch all open Kalshi events with nested markets."""
    print("[Kalshi] Fetching all open events...", flush=True)
    events, cursor = [], ""
    while True:
        params = {"status": "open", "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        try:
            resp = requests.get(f"{KALSHI_BASE}/events",
                                params=params, headers=_kalshi_headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("events", [])
            events += batch
            cursor = data.get("cursor", "")
            print(f"  ...{len(events)} so far", flush=True)
            if not cursor or not batch:
                break
            time.sleep(0.3)
        except Exception as e:
            print(f"  [error] {e}", flush=True)
            break
    print(f"[Kalshi] Total: {len(events)} events", flush=True)
    return events


def fetch_poly_events() -> list:
    """Fetch all active Polymarket events with markets."""
    print("[Poly] Fetching all active events...", flush=True)
    events, offset, limit = [], 0, 500
    while True:
        try:
            resp = requests.get(f"{GAMMA_BASE}/events",
                                params={"active": "true", "closed": "false",
                                        "limit": limit, "offset": offset,
                                        "order": "endDate", "ascending": "true"},
                                timeout=30)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            events += batch
            offset += len(batch)
            print(f"  ...{len(events)} so far", flush=True)
            if len(batch) < limit:
                break
            time.sleep(0.2)
        except Exception as e:
            print(f"  [error] {e}", flush=True)
            break
    print(f"[Poly] Total: {len(events)} events", flush=True)
    return events


# ════════════════════════════════════════════════════════════════════════════
# Outcome extraction (per-platform)
# ════════════════════════════════════════════════════════════════════════════

def extract_kalshi_outcomes(event: dict) -> list[dict]:
    """Return [{suffix, name, ticker}] for each Kalshi sub-market."""
    out = []
    for m in event.get("markets", []):
        ticker = m.get("ticker", "")
        suffix = ticker.split("-")[-1].lower() if "-" in ticker else ticker.lower()
        name = m.get("yes_sub_title") or m.get("subtitle") or m.get("title", "")
        if name:
            out.append({"suffix": suffix, "name": name, "ticker": ticker})
    return out


def extract_poly_outcomes(event: dict) -> list[dict]:
    """Return [{name}] for each Polymarket market in the event."""
    out = []
    for m in event.get("markets", []):
        name = m.get("groupItemTitle") or m.get("outcome") or m.get("question", "")
        if name:
            out.append({"name": name})
    return out


# ════════════════════════════════════════════════════════════════════════════
# Date helpers
# ════════════════════════════════════════════════════════════════════════════

def _parse_iso(iso: str) -> Optional[datetime]:
    if not iso:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(iso[:19].replace("Z", ""), fmt.replace("Z", "")).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def event_year(event: dict, source: str = "kalshi") -> Optional[int]:
    """Extract election year from event metadata. Used to disambiguate cycles."""
    if source == "kalshi":
        end = event.get("close_date") or event.get("expected_expiration_date")
    else:
        end = event.get("endDate")
    dt = _parse_iso(end)
    if dt:
        return dt.year
    return None


# ════════════════════════════════════════════════════════════════════════════
# Main matcher
# ════════════════════════════════════════════════════════════════════════════

def parse_events(events: list, source: str) -> list[tuple]:
    """Parse a list of events. Returns [(race, event, year), ...] for political ones."""
    parsed = []
    for ev in events:
        title = ev.get("title", "")
        race = parse_political_title(title)
        if not race:
            continue
        # Pull year from end date if not in title
        if not race.year:
            race.year = event_year(ev, source)
        parsed.append((race, ev, race.year))
    return parsed


def match_political_events(k_parsed, p_parsed, state_filter=None):
    """Match Kalshi political events with Polymarket equivalents."""
    matches = []
    used_p = set()

    # Index Poly by (state, office, district, race_type)
    p_index = defaultdict(list)
    for race, ev, _ in p_parsed:
        if state_filter and race.state != state_filter:
            continue
        key = (race.state, race.office, race.district, race.race_type)
        p_index[key].append((race, ev))

    for k_race, k_ev, _ in k_parsed:
        if state_filter and k_race.state != state_filter:
            continue
        key = (k_race.state, k_race.office, k_race.district, k_race.race_type)
        candidates = p_index.get(key, [])
        if not candidates:
            continue

        best = None
        best_reason = ""
        for p_race, p_ev in candidates:
            slug = p_ev.get("slug", "")
            if slug in used_p:
                continue
            ok, reason = k_race.is_compatible_with(p_race)
            if not ok:
                continue
            # Prefer matching party (for primaries)
            if k_race.race_type == RACE_PRIMARY and k_race.party and p_race.party:
                if k_race.party != p_race.party:
                    continue
            best = (p_race, p_ev)
            best_reason = reason
            break

        if not best:
            continue
        p_race, p_ev = best
        used_p.add(p_ev.get("slug", ""))

        # Build outcome mapping
        k_outs = extract_kalshi_outcomes(k_ev)
        p_outs = extract_poly_outcomes(p_ev)

        if k_race.race_type == RACE_PRIMARY:
            mapping = build_candidate_mapping(k_outs, p_outs)
        else:
            mapping = build_party_mapping(k_outs, p_outs)

        matches.append({
            "k_event_ticker": k_ev.get("event_ticker", ""),
            "k_title": k_ev.get("title", ""),
            "p_slug": p_ev.get("slug", ""),
            "p_title": p_ev.get("title", ""),
            "state": k_race.state,
            "office": k_race.office,
            "district": k_race.district,
            "party": k_race.party,
            "race_type": k_race.race_type,
            "year": k_race.year,
            "k_market_count": len(k_outs),
            "p_market_count": len(p_outs),
            "all_k_outcomes": k_outs,
            "all_p_outcomes": p_outs,
            "mapping": mapping,
            "k_end_date": k_ev.get("close_date") or k_ev.get("expected_expiration_date"),
            "p_end_date": p_ev.get("endDate"),
        })

    return matches


# ════════════════════════════════════════════════════════════════════════════
# Output
# ════════════════════════════════════════════════════════════════════════════

def write_config(matches: list, output_file: str = OUTPUT_FILE):
    """Write the matched events to a config file (compatible with arb_engine)."""
    events = []
    for m in matches:
        if not m["mapping"]:
            continue  # skip if no outcome mapping found
        # Friendly name
        parts = [STATES.get(m["state"], m["state"])]
        if m["district"] is not None:
            parts.append(f"District {m['district']}")
        parts.append(m["office"])
        if m["race_type"] != RACE_GENERAL:
            parts.append(f"({m['race_type']})")
        if m["party"]:
            parts.append(f"[{m['party']}]")
        if m["year"]:
            parts.append(str(m["year"]))
        name = " ".join(parts)

        events.append({
            "name": name,
            "kalshi_ticker": m["k_event_ticker"],
            "poly_slug": m["p_slug"],
            "mapping": m["mapping"],
        })

    cfg = {
        "poll_interval": 8,
        "threshold": 0.94,
        "min_alert_profit": 1.0,
        "events": events,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Wrote {len(events)} events to {output_file}")


def print_summary(matches: list):
    """Print a summary of matches by state."""
    by_state = defaultdict(list)
    for m in matches:
        by_state[m["state"]].append(m)

    print(f"\n{'=' * 70}")
    print(f"  POLITICAL MATCHES SUMMARY  ({len(matches)} total)")
    print(f"{'=' * 70}")

    # Sort states alphabetically
    for state in sorted(by_state.keys()):
        items = by_state[state]
        state_name = STATES.get(state, state)
        print(f"\n  {state} ({state_name}) — {len(items)} matches")
        for m in items[:8]:  # show first 8
            office = m["office"]
            dist = f"-{m['district']:02d}" if m["district"] is not None else ""
            party = f" [{m['party']}]" if m["party"] else ""
            yr = f" {m['year']}" if m["year"] else ""
            mp = len(m["mapping"])
            tot = m["k_market_count"]
            print(f"    {office}{dist}{party}{yr} | {m['race_type']} | mapping {mp}/{tot}")
        if len(items) > 8:
            print(f"    ...and {len(items) - 8} more")

    # Stats by office
    by_office = defaultdict(int)
    by_type = defaultdict(int)
    for m in matches:
        by_office[m["office"]] += 1
        by_type[m["race_type"]] += 1
    print(f"\n  By office: {dict(by_office)}")
    print(f"  By type:   {dict(by_type)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state", help="Filter to a specific 2-letter state code")
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    state_filter = args.state.upper() if args.state else None

    k_events = fetch_kalshi_events()
    p_events = fetch_poly_events()

    print(f"\n[Parse] Extracting political races...", flush=True)
    k_parsed = parse_events(k_events, source="kalshi")
    p_parsed = parse_events(p_events, source="poly")
    print(f"[Parse] Kalshi: {len(k_parsed)} political | Poly: {len(p_parsed)} political")

    matches = match_political_events(k_parsed, p_parsed, state_filter=state_filter)
    print_summary(matches)

    if not args.dry_run:
        write_config(matches, args.output)
    else:
        print(f"\n[DRY RUN] Would write {len(matches)} events to {args.output}")


if __name__ == "__main__":
    main()
