"""
auto_discover.py -- Automatically match Kalshi events to Polymarket events.

Usage:
    python auto_discover.py                     # discover, write to dummy JSON
    python auto_discover.py --review            # interactive review: accept/reject + map outcomes
    python auto_discover.py --min-score 0.6     # lower match threshold (default 0.65)
    python auto_discover.py --new-only          # only show events NOT in current config
    python auto_discover.py --write-real        # overwrite real markets_config.json (backs up first)

Workflow:
    1. Run without flags -> writes markets_config_discovered.json (safe)
    2. Run with --review  -> interactively accept/reject new matches
       Accepted matches are saved to reviewed_matches.json (persistent cache)
       Rejected matches are also cached so you don't see them again
    3. Run with --write-real -> merges accepted matches into real config
"""

import json
import re
import sys
import time
import os
from collections import defaultdict
import requests
from difflib import SequenceMatcher
from dotenv import load_dotenv

# Fix Windows console encoding for unicode characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

KALSHI_BASE   = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA_BASE    = "https://gamma-api.polymarket.com"
CONFIG_FILE   = "markets_config.json"
OUTPUT_FILE   = "markets_config_discovered.json"
REVIEW_FILE   = "reviewed_matches.json"   # persistent accept/reject cache
FULL_MATCHES  = "discovered_matches_full.json"  # rich data for review app

# Stop words that don't help matching
STOP_WORDS = frozenset([
    "the", "a", "an", "in", "on", "at", "to", "of", "for", "by", "is", "it",
    "be", "will", "or", "and", "this", "that", "with", "from", "as", "are",
    "was", "has", "have", "do", "does", "not", "no", "yes", "what", "who",
    "which", "when", "where", "how", "more", "than", "before", "after",
])

# Generic template words that shouldn't distinguish events.
# NOTE: house/senate/governor/etc. are NOT here — they're handled by DISTINGUISHING_PAIRS
# (a hard-reject mechanism), so they should still contribute to scoring.
GENERIC_WORDS = frozenset([
    "winner", "election", "nominee", "party", "first", "half",
    "will", "2025", "2026", "2027", "2028", "division", "conference",
    "champion", "championship", "round", "match", "game",
])

# Sport-specific keywords -- events should match within the same sport
SPORT_TAGS = {
    "nfl": "football", "nfc": "football", "afc": "football",
    "nba": "basketball", "wnba": "basketball",
    "mlb": "baseball",
    "nhl": "hockey",
    "mls": "soccer",
    "premier": "soccer", "liga": "soccer", "bundesliga": "soccer",
    "serie": "soccer", "ligue": "soccer", "champions": "soccer",
    "kbo": "baseball", "npb": "baseball",
    "f1": "racing", "nascar": "racing",
    "ufc": "mma", "pfl": "mma",
    "atp": "tennis", "wta": "tennis",
    "pga": "golf",
}

# Hard-reject pairs: if both titles contain one half of any pair, REJECT the match
# (not just LOW confidence). These are mutually exclusive concepts.
DISTINGUISHING_PAIRS = [
    # Government bodies
    ("house", "senate"),
    ("house", "governor"),
    ("senate", "governor"),
    ("mayor", "governor"),
    ("president", "governor"),
    ("president", "mayor"),
    ("congress", "senate"),
    # Political stages
    ("primary", "general"),
    ("primary", "runoff"),
    ("nomination", "general"),
    # Parties (within primary races)
    ("democratic", "republican"),
    ("democrat", "republican"),
    # Award categories
    ("actor", "actress"),
    ("director", "screenplay"),
    ("song", "album"),
    ("song", "record"),
    ("album", "record"),
    ("supporting", "lead"),
    # Game stages
    ("semifinal", "final"),
    ("quarterfinal", "semifinal"),
    # Other
    ("male", "female"),
    ("men", "women"),
    ("boys", "girls"),
]


def _check_distinguishing(title_a: str, title_b: str) -> str | None:
    """If both titles contain opposite halves of a distinguishing pair, return the conflict."""
    a_low = title_a.lower()
    b_low = title_b.lower()
    for w1, w2 in DISTINGUISHING_PAIRS:
        # Word boundary check (avoid 'house' matching 'household')
        a_has_1 = re.search(rf'\b{w1}\b', a_low) is not None
        a_has_2 = re.search(rf'\b{w2}\b', a_low) is not None
        b_has_1 = re.search(rf'\b{w1}\b', b_low) is not None
        b_has_2 = re.search(rf'\b{w2}\b', b_low) is not None
        # A has word1 but not word2, B has word2 but not word1 → conflict
        if a_has_1 and not a_has_2 and b_has_2 and not b_has_1:
            return f"{w1}/{w2}"
        if a_has_2 and not a_has_1 and b_has_1 and not b_has_2:
            return f"{w2}/{w1}"
    return None


# Pairs of words/names that look similar but mean different things.
# If both titles each contain one half of a confusable pair, force LOW confidence.
CONFUSABLES = [
    ("austria", "australia"),
    ("australian", "austrian"),
    ("austin", "austria"),
    ("iran", "iraq"),
    ("iraqi", "iranian"),
    ("niger", "nigeria"),
    ("nigerian", "nigerien"),
    ("columbia", "colombia"),
    ("colombian", "columbian"),
    ("guinea", "papua"),
    ("sweden", "switzerland"),
    ("swedish", "swiss"),
    ("pakistan", "palestine"),
    ("slovaki", "sloveni"),    # slovakia/slovenia partial
    ("dominican", "dominica"),
    ("gambia", "zambia"),
    ("mali", "malawi"),
    ("mauritani", "mauritius"),
    ("lakers", "Lakers"),      # team names that overlap
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str) -> set:
    """Extract meaningful word tokens from a string."""
    words = set(_norm(s).split())
    return words - STOP_WORDS

def _clean(s: str) -> str:
    """No spaces at all (matches match_markets in bot)."""
    return s.lower().replace(" ", "").replace(".", "").replace("-", "")

def _sim(a_norm: str, b_norm: str) -> float:
    """Similarity on already-normalized strings."""
    return SequenceMatcher(None, a_norm, b_norm).ratio()

def _detect_sport(text: str) -> str | None:
    """Detect sport category from text. Returns sport name or None."""
    text_low = text.lower()
    for keyword, sport in SPORT_TAGS.items():
        if keyword in text_low:
            return sport
    return None

def _check_confusable(title_a: str, title_b: str) -> str | None:
    """Check if two titles contain a confusable pair. Returns the pair as string or None."""
    a_low = title_a.lower()
    b_low = title_b.lower()
    for w1, w2 in CONFUSABLES:
        # a has w1 but not w2, and b has w2 but not w1 -> confusable
        if (w1 in a_low and w2 not in a_low and w2 in b_low and w1 not in b_low):
            return f"{w1}/{w2}"
        if (w2 in a_low and w1 not in a_low and w1 in b_low and w2 not in b_low):
            return f"{w1}/{w2}"
    return None

def _short_date(iso: str | None) -> str:
    """Parse ISO date to short format like 'Apr 15' or 'N/A'."""
    if not iso:
        return "N/A"
    try:
        # Handle various formats
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
            try:
                from datetime import datetime
                dt = datetime.strptime(iso[:19].replace("Z", ""), fmt.replace("Z", ""))
                return dt.strftime("%b %d")
            except ValueError:
                continue
        return iso[:10]
    except Exception:
        return "N/A"


def _parse_date(iso: str | None):
    """Parse ISO date to datetime object. Returns None if invalid."""
    if not iso:
        return None
    try:
        from datetime import datetime
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
            try:
                return datetime.strptime(iso[:19].replace("Z", ""), fmt.replace("Z", ""))
            except ValueError:
                continue
    except Exception:
        pass
    return None


# Regex for extracting numeric tokens like "07", "17", "2026", etc.
_NUM_RE = re.compile(r'\b\d{1,4}\b')

def _numbers_in(text: str) -> set:
    """Extract numeric tokens from text (excluding common years 2024-2030)."""
    if not text:
        return set()
    nums = set(_NUM_RE.findall(text.lower()))
    # Drop common year tokens — they're not distinguishing
    nums = {n for n in nums if n not in ("2024", "2025", "2026", "2027", "2028", "2029", "2030")}
    return nums


def _numbers_compatible(text_a: str, text_b: str) -> bool:
    """Check if numbers in two texts are compatible.

    True if either text has no numbers, OR they share at least one number
    AND don't have conflicting numbers in the same context.

    e.g. 'PA-07' vs 'PA-17' -> False (different numbers)
         'House race 2026' vs 'House election' -> True (only year, ignored)
         'Best of 7' vs 'Best of 7' -> True (same number)
    """
    nums_a = _numbers_in(text_a)
    nums_b = _numbers_in(text_b)
    # If neither has meaningful numbers, OK
    if not nums_a and not nums_b:
        return True
    # If only one side has numbers, allow (one might be more specific)
    if not nums_a or not nums_b:
        return True
    # Both have numbers — they must share at least one
    return bool(nums_a & nums_b)


# ---------------------------------------------------------------------------
# Review cache
# ---------------------------------------------------------------------------

def load_review_cache() -> dict:
    """Load persistent accept/reject decisions."""
    try:
        with open(REVIEW_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"accepted": {}, "rejected": {}}

def save_review_cache(cache: dict):
    with open(REVIEW_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Kalshi -- fetch all open events with markets
# ---------------------------------------------------------------------------

def _build_kalshi_headers():
    """Build fresh auth headers for Kalshi (re-sign each time for pagination)."""
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
        sig = pem.sign(msg.encode(), padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }, pem
    except Exception as e:
        print(f"  [warn] Could not build Kalshi auth headers: {e}")
        return {}, None


def fetch_kalshi_events() -> list:
    print("[Kalshi] Fetching all open events...", flush=True)
    events, cursor = [], ""

    key_id = os.getenv("KALSHI_API_KEY_ID")
    pem = None

    # Load key once
    if key_id:
        try:
            from cryptography.hazmat.primitives import serialization
            pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
            if pk_path:
                with open(pk_path, "rb") as f:
                    pem = serialization.load_pem_private_key(f.read(), password=None)
        except Exception as e:
            print(f"  [warn] Could not load Kalshi key: {e}")

    while True:
        # Re-sign each page request with fresh timestamp
        headers = {}
        if key_id and pem:
            try:
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import padding
                import base64
                ts = str(int(time.time() * 1000))
                msg = f"{ts}GET/trade-api/v2/events"
                sig = pem.sign(msg.encode(), padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
                headers = {
                    "KALSHI-ACCESS-KEY": key_id,
                    "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
                    "KALSHI-ACCESS-TIMESTAMP": ts,
                }
            except Exception as e:
                print(f"  [warn] Kalshi signing failed: {e}")

        params = {
            "status": "open",
            "with_nested_markets": "true",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{KALSHI_BASE}/events", params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data    = resp.json()
        batch   = data.get("events", [])
        events += batch
        cursor  = data.get("cursor", "")
        print(f"  fetched {len(events)} Kalshi events so far...", flush=True)
        if not cursor or not batch:
            break
        time.sleep(0.3)

    print(f"[Kalshi] Total open events: {len(events)}")
    return events


# ---------------------------------------------------------------------------
# Polymarket -- fetch all active events with markets (newest first)
# ---------------------------------------------------------------------------

def fetch_poly_events(newest_first: bool = True) -> list:
    print("[Poly] Fetching all active events...", flush=True)
    events, offset = [], 0
    limit = 100
    # Order by startDate descending to get newest events first
    order = "startDate" if newest_first else "volume24hr"
    while True:
        resp = requests.get(
            f"{GAMMA_BASE}/events",
            params={"active": "true", "closed": "false",
                    "order": order, "ascending": "false",
                    "limit": limit, "offset": offset},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        events += batch
        offset += len(batch)
        print(f"  fetched {len(events)} Poly events so far...", flush=True)
        if len(batch) < limit:
            break
        time.sleep(0.2)

    print(f"[Poly] Total active events: {len(events)}")
    return events


# ---------------------------------------------------------------------------
# Fast matching with inverted word index
# ---------------------------------------------------------------------------

def build_poly_index(p_events: list):
    """Build an inverted index: word -> set of poly event indices."""
    index = defaultdict(set)
    norm_titles = []

    for i, pe in enumerate(p_events):
        title = pe.get("title", "")
        nt = _norm(title)
        norm_titles.append(nt)
        for word in _tokens(title):
            if len(word) >= 2:
                index[word].add(i)

    return index, norm_titles


def find_candidates(k_title: str, index: dict, max_candidates: int = 80) -> set:
    """Use inverted index to find poly events sharing words with k_title."""
    words = _tokens(k_title)
    scores = defaultdict(int)
    for w in words:
        if len(w) < 2:
            continue
        for idx in index.get(w, []):
            scores[idx] += 1

    if not scores:
        return set()

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidates = set()
    for idx, count in ranked[:max_candidates]:
        if count >= 2 or len(candidates) < 20:
            candidates.add(idx)
    return candidates


def match_outcomes(k_markets: list, p_markets: list) -> tuple:
    """Match Kalshi markets to Polymarket markets within a paired event.

    Returns: (matched_list, unmatched_kalshi, unmatched_poly)
    """
    matches = []
    used_poly = set()

    p_names = []
    p_cleans = []
    p_norms = []
    for pm in p_markets:
        name = pm.get("groupItemTitle") or pm.get("outcome") or pm.get("question", "")
        p_names.append(name)
        p_cleans.append(_clean(name))
        p_norms.append(_norm(name))

    for km in k_markets:
        k_ticker = km.get("ticker", "")
        k_suffix = k_ticker.split("-")[-1].lower() if "-" in k_ticker else k_ticker.lower()

        k_outcome = km.get("yes_sub_title") or km.get("subtitle") or km.get("title", "")
        if not k_outcome:
            continue

        k_cl = _clean(k_outcome)
        if "company" in k_cl or "other" in k_cl:
            continue

        best_idx = None
        best_score = 0.0
        k_norm = _norm(k_outcome)

        for j, pm in enumerate(p_markets):
            if j in used_poly or not p_names[j]:
                continue

            # Exact clean match = instant win
            if k_cl == p_cleans[j]:
                best_idx = j
                best_score = 1.0
                break

            score = _sim(k_norm, p_norms[j])
            if score > best_score:
                best_score = score
                best_idx = j

        if best_idx is not None and best_score >= 0.55:
            used_poly.add(best_idx)
            matches.append({
                "k_ticker":    k_ticker,
                "k_suffix":    k_suffix,
                "k_outcome":   k_outcome,
                "p_outcome":   p_names[best_idx],
                "p_index":     best_idx,
                "score":       round(best_score, 3),
            })

    # Collect unmatched items for display
    matched_k_tickers = {m["k_ticker"] for m in matches}
    matched_p_indices = {m["p_index"] for m in matches}

    unmatched_k = []
    for km in k_markets:
        t = km.get("ticker", "")
        name = km.get("yes_sub_title") or km.get("subtitle") or km.get("title", "")
        if t not in matched_k_tickers and name:
            unmatched_k.append({"ticker": t, "name": name})

    unmatched_p = []
    for j, pm in enumerate(p_markets):
        if j not in matched_p_indices and p_names[j]:
            unmatched_p.append({"index": j, "name": p_names[j]})

    return matches, unmatched_k, unmatched_p


def match_events(k_events: list, p_events: list, min_score: float,
                 existing_poly_slugs: set = None) -> list:
    """
    Fast event matching with:
    - Inverted word index for speed
    - Sport-category cross-check
    - Confusable word detection
    - Significant-word overlap for mid-confidence matches
    - Outcome-level quality gate
    - No duplicate Poly events (including existing config)
    """
    print("  Building Poly word index...", flush=True)
    index, p_norms = build_poly_index(p_events)
    print(f"  Index built: {len(index)} unique words across {len(p_events)} events", flush=True)

    results = []
    used_poly_slugs = set(existing_poly_slugs or [])
    checked = 0
    t0 = time.time()

    for ke in k_events:
        k_title = ke.get("title", "") or ke.get("sub_title", "")
        if not k_title:
            continue

        checked += 1
        if checked % 500 == 0:
            elapsed = time.time() - t0
            print(f"  matched {checked}/{len(k_events)} Kalshi events ({elapsed:.1f}s)...", flush=True)

        k_sport = _detect_sport(k_title)

        # SKIP SPORTS ENTIRELY: draws, ties, and live odds make matching unreliable.
        # Politics/awards/elections/movies are much safer to auto-discover.
        if k_sport:
            continue

        # Also skip Kalshi events tagged as sports by category
        k_cat = (ke.get("category") or "").lower()
        if any(s in k_cat for s in ("sport", "athletic", "football", "basketball", "baseball",
                                     "hockey", "soccer", "tennis", "golf", "racing", "ufc", "mma")):
            continue

        # Step 1: fast candidate filter via word overlap
        candidates = find_candidates(k_title, index)
        if not candidates:
            continue

        # Pre-compute Kalshi metadata for matching
        k_desc = ke.get("sub_title") or ""
        k_combined = f"{k_title} {k_desc}".strip()
        k_end = ke.get("close_date") or ke.get("expected_expiration_date") or ke.get("settlement_date")
        k_end_dt = _parse_date(k_end)
        k_numbers = _numbers_in(k_combined)

        # Step 2: fuzzy match only against candidates
        k_norm = _norm(k_title)
        k_words = _tokens(k_title)
        best_idx, best_score = None, 0.0
        confusable_flag = None

        for idx in candidates:
            p_slug = p_events[idx].get("slug", "")
            if p_slug in used_poly_slugs:
                continue

            p_title = p_events[idx].get("title", "")

            # Skip Poly sports events too (we already excluded Kalshi sports above)
            p_sport = _detect_sport(p_title)
            if p_sport:
                continue

            # ── DATE PROXIMITY CHECK ──
            # If both events have known end dates, they should be within ±60 days.
            # Otherwise we'd match "2026 election" with "2027 election" easily.
            p_end = p_events[idx].get("endDate") or p_events[idx].get("end_date_iso")
            p_end_dt = _parse_date(p_end)
            if k_end_dt and p_end_dt:
                delta_days = abs((k_end_dt - p_end_dt).days)
                if delta_days > 60:
                    continue

            # ── NUMERIC TOKEN CHECK ──
            # PA-07 must NOT match PA-17. Extract numbers from titles+descriptions
            # and verify they're compatible (share at least one if both have any).
            p_desc = p_events[idx].get("description") or ""
            p_combined = f"{p_title} {p_desc}".strip()
            p_numbers = _numbers_in(p_combined)
            # Only enforce when BOTH titles have numbers (descriptions are noisier)
            k_title_nums = _numbers_in(k_title)
            p_title_nums = _numbers_in(p_title)
            if k_title_nums and p_title_nums:
                if not (k_title_nums & p_title_nums):
                    continue

            # ── DISTINGUISHING-WORDS CHECK ──
            # House vs Senate, Primary vs General, Actor vs Actress, etc.
            # If titles are on opposite sides of a distinguishing pair, hard-reject.
            if _check_distinguishing(k_title, p_title):
                continue

            score = _sim(k_norm, p_norms[idx])

            # For mid-confidence matches (< 0.85), verify key content words overlap
            if 0.85 > score >= min_score:
                p_words = _tokens(p_title)
                k_sig = {w for w in k_words if len(w) >= 4 and w not in GENERIC_WORDS}
                p_sig = {w for w in p_words if len(w) >= 4 and w not in GENERIC_WORDS}
                if k_sig and p_sig:
                    overlap = k_sig & p_sig
                    min_set = min(len(k_sig), len(p_sig))
                    if min_set > 0 and len(overlap) / min_set < 0.5:
                        continue

            # Description boost: if descriptions share significant words, bump score
            if k_desc and p_desc and 0.65 <= score < 0.85:
                k_desc_sig = {w for w in _tokens(k_desc) if len(w) >= 5 and w not in GENERIC_WORDS}
                p_desc_sig = {w for w in _tokens(p_desc) if len(w) >= 5 and w not in GENERIC_WORDS}
                if k_desc_sig and p_desc_sig:
                    desc_overlap = k_desc_sig & p_desc_sig
                    if len(desc_overlap) >= 2:
                        score = min(1.0, score + 0.05)  # modest boost

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score < min_score or best_idx is None:
            continue

        pe = p_events[best_idx]
        k_markets = ke.get("markets", [])
        p_markets = pe.get("markets", [])
        outcome_matches, unmatched_k, unmatched_p = match_outcomes(k_markets, p_markets)

        if not outcome_matches:
            continue

        # Quality gate: average outcome score must be reasonable
        avg_out_score = sum(om["score"] for om in outcome_matches) / len(outcome_matches)
        high_quality = sum(1 for om in outcome_matches if om["score"] >= 0.8)

        if avg_out_score < 0.6 and high_quality < 2:
            if not (len(outcome_matches) == 1 and outcome_matches[0]["score"] >= 0.6):
                continue

        p_slug = pe.get("slug", "")
        used_poly_slugs.add(p_slug)

        # Check for confusable words
        p_title = pe.get("title", "")
        confusable_flag = _check_confusable(k_title, p_title)

        # Outcome count ratio -- flag big mismatches
        k_count = len(k_markets)
        p_count = len(p_markets)
        count_ratio = min(k_count, p_count) / max(k_count, p_count) if max(k_count, p_count) > 0 else 1.0

        # Confidence tier
        if confusable_flag:
            confidence = "LOW"  # Force LOW when confusable detected
        elif best_score >= 0.90 and avg_out_score >= 0.85 and count_ratio >= 0.5:
            confidence = "HIGH"
        elif best_score >= 0.80 and avg_out_score >= 0.70:
            confidence = "MED"
        else:
            confidence = "LOW"

        # Extract rich metadata
        k_end = ke.get("close_date") or ke.get("expected_expiration_date") or ke.get("settlement_date")
        p_end = pe.get("endDate") or pe.get("end_date_iso")
        k_desc = ke.get("sub_title") or ke.get("title", "")
        p_desc = pe.get("description") or pe.get("title", "")
        k_category = ke.get("category", "") or ke.get("series_ticker", "")

        # Build full outcome lists for manual mapping in review app
        all_k_outcomes = []
        for km in k_markets:
            t = km.get("ticker", "")
            suffix = t.split("-")[-1].lower() if "-" in t else t.lower()
            name = km.get("yes_sub_title") or km.get("subtitle") or km.get("title", "")
            if name:
                all_k_outcomes.append({"suffix": suffix, "name": name, "ticker": t})
        all_p_outcomes = []
        for pm in p_markets:
            name = pm.get("groupItemTitle") or pm.get("outcome") or pm.get("question", "")
            if name:
                all_p_outcomes.append({"name": name})

        results.append({
            "k_event_ticker": ke.get("event_ticker", ""),
            "k_title":        k_title,
            "k_description":  k_desc[:200],
            "k_end_date":     k_end,
            "k_category":     k_category,
            "k_market_count": k_count,
            "p_slug":         p_slug,
            "p_title":        p_title,
            "p_description":  (p_desc or "")[:200],
            "p_end_date":     p_end,
            "p_market_count": p_count,
            "event_score":    round(best_score, 3),
            "avg_outcome_score": round(avg_out_score, 3),
            "count_ratio":    round(count_ratio, 2),
            "confidence":     confidence,
            "confusable":     confusable_flag,
            "outcome_matches": outcome_matches,
            "unmatched_k":    unmatched_k,
            "unmatched_p":    unmatched_p,
            "all_k_outcomes": all_k_outcomes,
            "all_p_outcomes": all_p_outcomes,
        })

    elapsed = time.time() - t0
    print(f"  Matching done in {elapsed:.1f}s", flush=True)
    results.sort(key=lambda x: x["event_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Build config in the EXACT format versionfinparparal.py expects
# ---------------------------------------------------------------------------

def build_config(matches: list, existing_config: dict, new_only: bool = False,
                 review_cache: dict = None):
    """Produce markets_config. New events first."""
    existing_tickers = {e.get("kalshi_ticker") for e in existing_config.get("events", [])}
    accepted = (review_cache or {}).get("accepted", {})

    new_events = []
    existing_events = []

    for m in matches:
        # Use mapping from accepted cache if available (may have manual edits)
        cached = accepted.get(m["k_event_ticker"], {})
        if cached.get("mapping"):
            mapping = cached["mapping"]
        else:
            mapping = {}
            for om in m["outcome_matches"]:
                mapping[om["k_suffix"]] = om["p_outcome"]

        entry = {
            "name":          m["k_title"],
            "kalshi_ticker": m["k_event_ticker"],
            "poly_slug":     m["p_slug"],
            "mapping":       mapping,
        }

        if m["k_event_ticker"] in existing_tickers:
            if not new_only:
                existing_events.append(entry)
        else:
            if review_cache is not None:
                if m["k_event_ticker"] in accepted:
                    new_events.append(entry)
            else:
                new_events.append(entry)

    all_events = new_events + existing_events

    if not new_only:
        auto_tickers = {e["kalshi_ticker"] for e in all_events}
        for e in existing_config.get("events", []):
            if e.get("kalshi_ticker") not in auto_tickers:
                all_events.append(e)

    config = {
        "poll_interval":   existing_config.get("poll_interval", 8),
        "threshold":       existing_config.get("threshold", 0.94),
        "min_alert_profit": existing_config.get("min_alert_profit", 1.0),
        "events":          all_events,
    }
    return config, len(new_events)


# ---------------------------------------------------------------------------
# Interactive review (enhanced)
# ---------------------------------------------------------------------------

def _print_match_detail(m: dict, num: int = 0):
    """Print rich detail for one match."""
    conf = m["confidence"]
    warn = f"  !! CONFUSABLE: {m['confusable']}" if m.get("confusable") else ""
    count_warn = ""
    if m["count_ratio"] < 0.3:
        count_warn = f"  !! OUTCOME COUNT MISMATCH: K={m['k_market_count']} vs P={m['p_market_count']}"

    prefix = f"#{num} " if num else ""
    print(f"\n  {prefix}[{conf}] score={m['event_score']:.2f}  outcomes={m['avg_outcome_score']:.2f}  ratio={m['count_ratio']:.0%}{warn}{count_warn}")
    print(f"  Kalshi:  {m['k_title'][:70]}")
    print(f"  Poly:    {m['p_title'][:70]}")
    print(f"  K desc:  {m.get('k_description', '')[:80]}")
    print(f"  P desc:  {m.get('p_description', '')[:80]}")
    print(f"  Ends:    K={_short_date(m.get('k_end_date'))}  P={_short_date(m.get('p_end_date'))}")
    print(f"  Markets: K={m['k_market_count']}  P={m['p_market_count']}  Matched={len(m['outcome_matches'])}")

    if m.get("k_category"):
        print(f"  Category: {m['k_category']}")

    # Show matched outcomes
    oms = m["outcome_matches"]
    if oms:
        print(f"  --- Matched outcomes ---")
        for i, om in enumerate(oms[:10]):
            flag = " " if om["score"] >= 0.8 else "?"
            print(f"    {i+1:>2}. {flag} {om['score']:.2f}  K: {om['k_outcome'][:28]:<28} -> P: {om['p_outcome'][:30]}")
        if len(oms) > 10:
            print(f"    ... and {len(oms)-10} more")

    # Show unmatched
    uk = m.get("unmatched_k", [])
    up = m.get("unmatched_p", [])
    if uk:
        print(f"  --- Unmatched Kalshi ({len(uk)}) ---")
        for u in uk[:5]:
            print(f"       K: {u['name'][:40]}")
        if len(uk) > 5:
            print(f"       ... and {len(uk)-5} more")
    if up:
        print(f"  --- Unmatched Poly ({len(up)}) ---")
        for u in up[:5]:
            print(f"       P: {u['name'][:40]}")
        if len(up) > 5:
            print(f"       ... and {len(up)-5} more")


def _manual_map_outcomes(m: dict) -> dict:
    """Interactive manual outcome mapping. Returns mapping dict {k_suffix: p_outcome}."""
    k_markets_raw = m.get("_k_markets", [])
    p_markets_raw = m.get("_p_markets", [])

    # Build lists to display
    k_items = []
    for km in k_markets_raw:
        t = km.get("ticker", "")
        suffix = t.split("-")[-1].lower() if "-" in t else t.lower()
        name = km.get("yes_sub_title") or km.get("subtitle") or km.get("title", "")
        if name:
            k_items.append({"suffix": suffix, "name": name, "ticker": t})

    p_items = []
    for pm in p_markets_raw:
        name = pm.get("groupItemTitle") or pm.get("outcome") or pm.get("question", "")
        if name:
            p_items.append({"name": name})

    if not k_items or not p_items:
        print("    No outcomes to map.")
        return {}

    print(f"\n  === MANUAL OUTCOME MAPPING ===")
    print(f"  Kalshi outcomes ({len(k_items)}):")
    for i, ki in enumerate(k_items):
        print(f"    K{i+1:>2}. [{ki['suffix']:<12}] {ki['name'][:50]}")
    print(f"  Polymarket outcomes ({len(p_items)}):")
    for j, pi in enumerate(p_items):
        print(f"    P{j+1:>2}. {pi['name'][:50]}")

    print(f"\n  Type mappings as 'K#-P#' (e.g. '1-3' maps K1 to P3)")
    print(f"  Type 'auto' to keep auto-matched, 'done' to finish, 'skip' to cancel")

    mapping = {}
    # Pre-populate with auto matches
    for om in m.get("outcome_matches", []):
        mapping[om["k_suffix"]] = om["p_outcome"]

    while True:
        resp = input("  map> ").strip().lower()
        if resp in ("done", "d"):
            break
        elif resp in ("skip", "s"):
            return {}
        elif resp == "auto":
            print(f"    Keeping {len(mapping)} auto-matched pairs")
            break
        elif resp == "clear":
            mapping = {}
            print("    Cleared all mappings")
        elif resp == "show":
            print(f"    Current mapping ({len(mapping)}):")
            for k, v in mapping.items():
                print(f"      {k} -> {v}")
        else:
            # Parse "K#-P#" or just "#-#"
            parts = resp.replace("k", "").replace("p", "").split("-")
            if len(parts) == 2:
                try:
                    ki = int(parts[0]) - 1
                    pi = int(parts[1]) - 1
                    if 0 <= ki < len(k_items) and 0 <= pi < len(p_items):
                        mapping[k_items[ki]["suffix"]] = p_items[pi]["name"]
                        print(f"    Mapped: {k_items[ki]['name'][:30]} -> {p_items[pi]['name'][:30]}")
                    else:
                        print(f"    Invalid indices (K: 1-{len(k_items)}, P: 1-{len(p_items)})")
                except ValueError:
                    print("    Format: #-# (e.g. 1-3)")
            else:
                print("    Format: #-# | auto | done | skip | show | clear")

    return mapping


def interactive_review(matches: list, existing_tickers: set, review_cache: dict,
                       k_events_by_ticker: dict = None, p_events_by_slug: dict = None) -> dict:
    """
    Walk through new matches with rich display.
    User can accept, reject, skip, or manually map outcomes.
    """
    accepted = review_cache.get("accepted", {})
    rejected = review_cache.get("rejected", {})

    new_matches = [m for m in matches if m["k_event_ticker"] not in existing_tickers]

    # Split by confidence
    high = [m for m in new_matches if m["confidence"] == "HIGH" and m["k_event_ticker"] not in accepted and m["k_event_ticker"] not in rejected]
    med  = [m for m in new_matches if m["confidence"] == "MED"  and m["k_event_ticker"] not in accepted and m["k_event_ticker"] not in rejected]
    low  = [m for m in new_matches if m["confidence"] == "LOW"  and m["k_event_ticker"] not in accepted and m["k_event_ticker"] not in rejected]

    already = sum(1 for m in new_matches if m["k_event_ticker"] in accepted or m["k_event_ticker"] in rejected)
    confusable_count = sum(1 for m in new_matches if m.get("confusable"))

    print(f"\n{'='*70}")
    print(f"INTERACTIVE REVIEW")
    print(f"  {len(high)} HIGH confidence (auto-accept, press 'n' to reject)")
    print(f"  {len(med)} MED confidence (review recommended)")
    print(f"  {len(low)} LOW confidence (likely need manual check)")
    print(f"  {already} already reviewed (cached)")
    if confusable_count:
        print(f"  !! {confusable_count} matches flagged as CONFUSABLE")
    print(f"{'='*70}")
    print(f"Commands: y=accept  n=reject  m=manual-map  s=skip  q=quit  a=accept-all-HIGH")
    print()

    # Attach raw markets for manual mapping
    for m in new_matches:
        if k_events_by_ticker and p_events_by_slug:
            ke = k_events_by_ticker.get(m["k_event_ticker"], {})
            pe = p_events_by_slug.get(m["p_slug"], {})
            m["_k_markets"] = ke.get("markets", [])
            m["_p_markets"] = pe.get("markets", [])

    # HIGH confidence
    if high:
        print(f"--- HIGH confidence ({len(high)} matches) ---")
        resp = input(f"  Auto-accept all {len(high)} HIGH confidence matches? [Y/n/review]: ").strip().lower()
        if resp in ("", "y", "yes"):
            for m in high:
                mapping = {}
                for om in m["outcome_matches"]:
                    mapping[om["k_suffix"]] = om["p_outcome"]
                accepted[m["k_event_ticker"]] = {
                    "k_title": m["k_title"], "p_slug": m["p_slug"],
                    "p_title": m["p_title"], "score": m["event_score"],
                    "mapping": mapping,
                }
            print(f"  Accepted {len(high)} HIGH matches")
        elif resp == "review":
            for m in high:
                if not _review_one(m, accepted, rejected):
                    break

    # MED
    if med:
        print(f"\n--- MED confidence ({len(med)} matches) ---")
        for m in med:
            if not _review_one(m, accepted, rejected):
                break

    # LOW
    if low:
        print(f"\n--- LOW confidence ({len(low)} matches) ---")
        resp = input(f"  Review {len(low)} LOW confidence matches? [y/N]: ").strip().lower()
        if resp in ("y", "yes"):
            for m in low:
                if not _review_one(m, accepted, rejected):
                    break

    review_cache["accepted"] = accepted
    review_cache["rejected"] = rejected
    save_review_cache(review_cache)

    print(f"\nReview saved: {len(accepted)} accepted, {len(rejected)} rejected -> {REVIEW_FILE}")
    return review_cache


def _review_one(m: dict, accepted: dict, rejected: dict) -> bool:
    """Show one match with rich detail and get user decision. Returns False to quit."""
    kt = m["k_event_ticker"]
    if kt in accepted or kt in rejected:
        return True

    _print_match_detail(m)

    while True:
        resp = input("  [y/n/m/s/q] > ").strip().lower()
        if resp in ("y", "yes", ""):
            mapping = {}
            for om in m["outcome_matches"]:
                mapping[om["k_suffix"]] = om["p_outcome"]
            accepted[kt] = {
                "k_title": m["k_title"], "p_slug": m["p_slug"],
                "p_title": m["p_title"], "score": m["event_score"],
                "mapping": mapping,
            }
            print("  -> ACCEPTED")
            return True
        elif resp in ("n", "no"):
            rejected[kt] = {
                "k_title": m["k_title"], "p_slug": m["p_slug"],
                "reason": "manual_reject",
            }
            print("  -> REJECTED")
            return True
        elif resp in ("m", "map", "manual"):
            manual_mapping = _manual_map_outcomes(m)
            if manual_mapping:
                accepted[kt] = {
                    "k_title": m["k_title"], "p_slug": m["p_slug"],
                    "p_title": m["p_title"], "score": m["event_score"],
                    "mapping": manual_mapping,
                }
                print(f"  -> ACCEPTED with {len(manual_mapping)} manual mappings")
                return True
            else:
                print("  (manual mapping cancelled, pick again)")
        elif resp in ("s", "skip"):
            return True
        elif resp in ("q", "quit"):
            return False
        else:
            print("  (y=accept, n=reject, m=manual-map, s=skip, q=quit)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args       = sys.argv[1:]
    write_real = "--write-real" in args
    new_only   = "--new-only" in args
    do_review  = "--review" in args
    min_score  = 0.65
    for i, a in enumerate(args):
        if a == "--min-score" and i + 1 < len(args):
            min_score = float(args[i + 1])

    k_events = fetch_kalshi_events()
    p_events = fetch_poly_events(newest_first=True)

    # Load existing config
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {"poll_interval": 8, "threshold": 0.94, "min_alert_profit": 1.0, "events": []}

    # Collect existing poly slugs to prevent duplicate matching
    existing_poly_slugs = {e.get("poly_slug") for e in existing.get("events", []) if e.get("poly_slug")}
    existing_tickers = {e.get("kalshi_ticker") for e in existing.get("events", [])}

    print(f"\nMatching with min_score={min_score}...")
    matches = match_events(k_events, p_events, min_score=min_score,
                          existing_poly_slugs=existing_poly_slugs)
    print(f"Found {len(matches)} matched event pairs\n")

    # Save rich match data for review app
    with open(FULL_MATCHES, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kalshi_count": len(k_events),
            "poly_count": len(p_events),
            "existing_tickers": list(existing_tickers),
            "matches": matches,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved rich match data to {FULL_MATCHES}")

    # Print summary
    new_matches = [m for m in matches if m["k_event_ticker"] not in existing_tickers]
    old_matches = [m for m in matches if m["k_event_ticker"] in existing_tickers]

    high = sum(1 for m in new_matches if m["confidence"] == "HIGH")
    med  = sum(1 for m in new_matches if m["confidence"] == "MED")
    low  = sum(1 for m in new_matches if m["confidence"] == "LOW")
    confusable = sum(1 for m in new_matches if m.get("confusable"))

    print(f"  NEW: {len(new_matches)} events ({high} HIGH, {med} MED, {low} LOW confidence)")
    if confusable:
        print(f"  !! {confusable} flagged as CONFUSABLE (forced LOW)")
    print(f"  EXISTING: {len(old_matches)} already in config")
    print()

    # Table header
    print(f"{'':>3} {'SCORE':>5}  {'CONF':<4} {'R':>4} {'K#':>3}{'P#':>3} {'KALSHI TITLE':<35} {'POLY TITLE':<35}  {'ENDS':>12}  OUT")
    print("-" * 155)

    for m in new_matches:
        n_out = len(m["outcome_matches"])
        conf = m["confidence"]
        warn = "*" if m.get("confusable") else " "
        ratio = f"{m['count_ratio']:.0%}" if m["count_ratio"] < 0.5 else ""
        ends = f"K:{_short_date(m.get('k_end_date'))} P:{_short_date(m.get('p_end_date'))}"
        print(f" {warn} {m['event_score']:.2f}  {conf:<4} {ratio:>4} {m['k_market_count']:>3}{m['p_market_count']:>3} {m['k_title'][:35]:<35} {m['p_title'][:35]:<35}  {ends:>12}  {n_out:>3}")
        # Show confusable warning
        if m.get("confusable"):
            print(f"        !! CONFUSABLE: {m['confusable']}")
        # Show outcome details for MED/LOW matches
        if conf in ("MED", "LOW"):
            for om in m["outcome_matches"][:3]:
                flag = " " if om["score"] >= 0.8 else "?"
                print(f"        {flag} {om['score']:.2f}  K: {om['k_suffix']:<12} -> P: {om['p_outcome'][:35]}")
            if len(m["outcome_matches"]) > 3:
                print(f"          ... and {len(m['outcome_matches'])-3} more")

    print(f"\n=== {len(new_matches)} NEW ({high} HIGH, {med} MED, {low} LOW) | {len(old_matches)} existing ===\n")

    # Build lookup dicts for interactive review (raw market data for manual mapping)
    k_events_by_ticker = {ke.get("event_ticker", ""): ke for ke in k_events}
    p_events_by_slug = {pe.get("slug", ""): pe for pe in p_events}

    # Interactive review mode
    review_cache = load_review_cache() if (do_review or write_real) else None
    if do_review:
        review_cache = interactive_review(
            matches, existing_tickers, review_cache,
            k_events_by_ticker=k_events_by_ticker,
            p_events_by_slug=p_events_by_slug,
        )

    # Build output config
    config, n_new = build_config(matches, existing, new_only=new_only,
                                 review_cache=review_cache)
    out_json = json.dumps(config, indent=2, ensure_ascii=False)

    if write_real:
        if review_cache and not review_cache.get("accepted"):
            print("No accepted matches in review cache. Run --review first.")
            return
        import shutil
        try:
            shutil.copy(CONFIG_FILE, CONFIG_FILE + ".bak")
            print(f"Backed up existing config to {CONFIG_FILE}.bak")
        except FileNotFoundError:
            pass
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(out_json)
        print(f"Wrote {len(config['events'])} events to {CONFIG_FILE} ({n_new} new)")
    else:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(out_json)
        print(f"Wrote {len(config['events'])} events to {OUTPUT_FILE} ({n_new} new)")
        if not do_review:
            print(f"  -> Run with --review to accept/reject matches interactively")
        print(f"  -> Or run with --write-real to apply accepted matches to real config")


if __name__ == "__main__":
    main()
