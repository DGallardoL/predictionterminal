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
    ("gubernatorial", "senate"),
    ("gubernatorial", "presidential"),
    ("parliamentary", "presidential"),
    ("parliament", "senate"),
    ("attorney", "governor"),
    # House vs vice-president are different races even though both elected.
    ("house", "vp"),
    ("house", "vice"),
    ("senate", "vp"),
    ("senate", "vice"),
    ("delegate", "nominee"),
    ("delegate", "winner"),
    # Political stages
    ("primary", "general"),
    ("primary", "runoff"),
    ("nomination", "general"),
    # Parties (within primary races)
    ("democratic", "republican"),
    ("democrat", "republican"),
    # Resolution metric (margin vs winner is a different question)
    ("margin", "winner"),
    ("margin", "advance"),
    ("endorse", "winner"),
    # Nth-place is NOT winner. Audit (2026-05-18) flagged 6 MED FPs of this
    # pattern: Alabama/Georgia/Oregon/Florida primary 2nd-place vs winner.
    ("2nd place", "winner"),
    ("second place", "winner"),
    ("3rd place", "winner"),
    ("third place", "winner"),
    ("2nd place", "1st place"),
    ("second place", "first place"),
    # Nominee != runoff (one is the primary outcome, the other is a second
    # round). Closes the Texas-AG nominee/runoff FP.
    ("nominee", "runoff"),
    ("nominee", "2nd place"),
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
    # Topic mismatches (debt vs Fed, abolish vs end)
    ("debt", "fed"),
    # Audit 2026-05-18: vote-percent / turnout / counties-won are NOT "winner".
    ("vote percent", "turnout"),
    ("percent", "turnout"),
    ("counties", "winner"),
    ("counties", "first place"),
    ("counties", "advancing"),
    ("turnout", "winner"),
    # Trailer / release / launch are different resolution metrics.
    ("trailer", "released"),
    ("trailer", "release"),
    ("trailer", "launch"),
    # Topic-nonsense guards caught in audit.
    ("impeached", "coin"),
    ("hardware", "gpt"),
    ("hardware", "model"),
    # Action-verb antonyms (audit 2026-05-19). Opposite directions of the
    # same event can't be the same question — they're inverse markets.
    ("hike", "cut"),
    ("hike", "fail"),
    ("ipo", "bankruptcy"),
    ("ipo", "acquired"),
    ("launch", "shutdown"),
    ("launch", "discontinue"),
    ("announce", "withdraw"),
    ("announce", "drop out"),
    ("acquire", "ipo"),
    ("fire", "promote"),
    ("hire", "fire"),
    ("hire", "leave"),
    ("expand", "shrink"),
    ("rise", "fall"),
    ("up", "down"),
]


_PRIMARY_WORDS = re.compile(r"\b(primary|nominee|nomination|caucus|runoff)\b", re.IGNORECASE)

# Month tokens — used to reject "X in May" vs "X before July" pairings.
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b", re.IGNORECASE,
)

# Year tokens. Standalone 4-digit years 2024-2035. Reject pairs that name
# disjoint years explicitly (2028 vs 2032, 2026 vs 2027). Audit 2026-05-19
# flagged KX2028DRUN→announce-before-2027 as 12 FPs from one bad pair.
_YEAR_RE = re.compile(r"\b(202[4-9]|203[0-5])\b")

# Office tokens at title level — broader than ENTITY_OFFICES (which is
# multi-word). Catches "mayor" vs "senate" cross-office FPs caught in audit:
# KXANNARBORMAYORD (Ann Arbor mayor) → delaware-democratic-senate-primary.
_OFFICE_TITLE_TOKENS = (
    "mayor", "mayoral",
    "senate", "senator", "senatorial",
    "house",
    "governor", "gubernatorial",
    "president", "presidential",
    "attorney general", "ag",
    "secretary of state", "sos",
    "treasurer",
    "comptroller",
    "judge", "justice",
)

def _check_office_mismatch(a: str, b: str) -> str | None:
    """Reject when titles mention different elected offices.

    More aggressive than DISTINGUISHING_PAIRS — looks for ANY office on each
    side and rejects when the sets are disjoint. The mayor/senate/governor
    confusion (Ann Arbor mayor mapped to Delaware Senate) accounted for
    multiple top-40 FPs.
    """
    if not a or not b:
        return None
    al = a.lower(); bl = b.lower()
    a_offices = {t for t in _OFFICE_TITLE_TOKENS if t in al}
    b_offices = {t for t in _OFFICE_TITLE_TOKENS if t in bl}
    if a_offices and b_offices:
        # Treat "governor"/"gubernatorial" as the same office; same for
        # senate/senator/senatorial, mayor/mayoral, president/presidential.
        norm = {
            "gubernatorial": "governor",
            "senator": "senate", "senatorial": "senate",
            "mayoral": "mayor",
            "presidential": "president",
        }
        a_norm = {norm.get(t, t) for t in a_offices}
        b_norm = {norm.get(t, t) for t in b_offices}
        if a_norm.isdisjoint(b_norm):
            return f"office:{sorted(a_norm)}/{sorted(b_norm)}"
    return None


def _check_year_window(a: str, b: str) -> str | None:
    """Reject pairs whose titles name explicitly disjoint years.

    Catches "Will X happen in 2028?" vs "Will X happen before 2027?".
    When only one side names a year, accept (the other side is likely
    relative). Year 2026 (current) is ignored — too many markets default
    to "this year" implicitly.
    """
    a_years = {int(m) for m in _YEAR_RE.findall(a or "") if m != "2026"}
    b_years = {int(m) for m in _YEAR_RE.findall(b or "") if m != "2026"}
    if a_years and b_years and a_years.isdisjoint(b_years):
        return f"year-window:{sorted(a_years)}/{sorted(b_years)}"
    return None

def _check_month_window(a: str, b: str) -> str | None:
    """Reject pairs that explicitly name disjoint month windows in titles.

    "What will Trump say in May?" vs "What will Trump say before July?" share
    the noun/subject pattern but resolve over different months — they're
    sibling markets, not the same event.
    """
    a_months = {m.group(1).lower() for m in _MONTH_RE.finditer(a or "")}
    b_months = {m.group(1).lower() for m in _MONTH_RE.finditer(b or "")}
    if a_months and b_months and a_months.isdisjoint(b_months):
        return f"month-window:{sorted(a_months)}/{sorted(b_months)}"
    return None


_WHAT_NOUN_RE = re.compile(r"^what\s+(\w+)\s+will\s+(\w+)", re.IGNORECASE)

def _check_what_noun(a: str, b: str) -> str | None:
    """"What <noun> will <subject>" — reject when the noun differs.

    Catches "What nicknames will Trump say…" vs "What animals will Trump
    say…" — same subject, but the *subject of the question* (nicknames vs
    animals) makes them different markets.
    """
    ma = _WHAT_NOUN_RE.match((a or "").strip())
    mb = _WHAT_NOUN_RE.match((b or "").strip())
    if ma and mb and ma.group(2).lower() == mb.group(2).lower():
        if ma.group(1).lower() != mb.group(1).lower():
            return f"what-noun:{ma.group(1)}/{mb.group(1)}"
    return None


_EARNINGS_RE = re.compile(
    r"what will ([\w&.\-]+(?:\s+[\w&.\-]+)?) say during", re.IGNORECASE
)

def _check_earnings_subject(a: str, b: str) -> str | None:
    """Reject "What will X say during earnings" pairs where X differs.

    "What will Home Depot say during their next earnings call?" must not
    match "What will NVIDIA say during their next earnings call?" — same
    template, different reporting company.
    """
    ma = _EARNINGS_RE.search(a or "")
    mb = _EARNINGS_RE.search(b or "")
    if ma and mb and ma.group(1).lower() != mb.group(1).lower():
        return f"earnings-subject:{ma.group(1)}/{mb.group(1)}"
    return None


# Strip "(Show Name)" / "(Season N)" parenthetical suffixes from outcome
# strings so anime-award TPs lift from LOW to MED (the audit found 5+ pairs
# with event_score ~0.88 but avg_outcome_score ~0.6 purely because Poly
# appends "(SHOW)" and the fuzzy matcher penalises the mismatch).
_PAREN_SUFFIX_RE = re.compile(r"\s*\([^)]+\)\s*$")

def _strip_paren_suffix(s: str) -> str:
    if not s:
        return ""
    return _PAREN_SUFFIX_RE.sub("", s).strip()


# Date-key consistency. Kalshi outcome suffixes often encode a target date
# like "26jul01" or "jul01"; if the matched Poly outcome explicitly names a
# different month, the mapping is internally inconsistent and is dropped.
_KALSHI_DATE_RE = re.compile(r"\b(\d{0,2})?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(\d{2})?\b", re.IGNORECASE)
_FULL_MONTH_RE = _MONTH_RE  # re-use the title-month regex

_MONTH_ABBR = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "may": "may", "jun": "june", "jul": "july", "aug": "august",
    "sep": "september", "oct": "october", "nov": "november", "dec": "december",
}

def _date_keys_consistent(k_suffix: str, p_outcome: str) -> bool:
    """True when K suffix's month (if any) matches a month in P outcome.

    Returns True when there's no month info on either side (most common —
    candidate names, not date suffixes), so we don't accidentally reject
    person/team outcomes.
    """
    if not k_suffix or not p_outcome:
        return True
    k_match = _KALSHI_DATE_RE.search(k_suffix.lower())
    if not k_match:
        return True
    k_month = _MONTH_ABBR.get(k_match.group(2).lower())
    if not k_month:
        return True
    p_months = {m.group(1).lower() for m in _FULL_MONTH_RE.finditer(p_outcome or "")}
    if not p_months:
        # Poly side has no explicit month — let it through (the K date is
        # informational, not contradicted).
        return True
    return k_month in p_months

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

    # Asymmetric primary-vs-general check.
    # Polymarket labels general elections as "<Office> Election Winner" (no
    # "primary"), and primaries as "<Office> Primary Winner". Kalshi often
    # omits both — "Wyoming Senate winner?" is the general. So when exactly
    # ONE title says "primary"/"nominee"/"caucus" and the other says
    # "election" with no primary-tier word, they resolve differently.
    a_prim = _PRIMARY_WORDS.search(a_low) is not None
    b_prim = _PRIMARY_WORDS.search(b_low) is not None
    if a_prim != b_prim:
        a_elec = "election" in a_low
        b_elec = "election" in b_low
        if (a_prim and not b_prim and b_elec) or (b_prim and not a_prim and a_elec):
            return "primary/election"

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

def _strip_accents(s: str) -> str:
    """Fold é→e, ñ→n, etc. so 'Série A' matches the 'serie' sport keyword."""
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def _detect_sport(text: str) -> str | None:
    """Detect sport category from text. Returns sport name or None."""
    text_low = _strip_accents(text.lower())
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
# Polarity detection (hard-gate)
# ---------------------------------------------------------------------------

# Phrases that indicate negative polarity (the question asks whether X does NOT happen).
# We only trigger when one title is negative and the other is not — same polarity OK.
_NEG_RE = re.compile(
    r"\b(?:not|won'?t|will\s+not|fails?\s+to|doesn'?t|don'?t|"
    r"never|no\s+(?:longer|more)|fail\s+to|miss(?:es)?|"
    r"avoid(?:s|ed)?|skip(?:s|ped)?)\b",
    re.IGNORECASE,
)

def _polarity(text: str) -> str:
    """Return 'neg' if title carries negation polarity, 'pos' otherwise."""
    if not text:
        return "pos"
    return "neg" if _NEG_RE.search(text) else "pos"


# ---------------------------------------------------------------------------
# Entity extraction (lightweight NER via regex over fixed lists)
# ---------------------------------------------------------------------------

# People who appear frequently in prediction markets. Lowercase, used as
# word-boundary regex. Distinctive last names only — "Trump" not "Donald".
ENTITY_PEOPLE = frozenset([
    "trump", "biden", "harris", "vance", "newsom", "desantis", "pence",
    "obama", "clinton", "sanders", "warren", "buttigieg", "ramaswamy",
    "haley", "christie", "kennedy", "stein", "west", "rfk",
    "putin", "zelensky", "zelenskyy", "xi", "modi", "netanyahu", "starmer",
    "macron", "merkel", "scholz", "milei", "lula", "maduro",
    "musk", "bezos", "zuckerberg", "buffett", "powell", "yellen",
    "altman", "huang", "cook", "pichai", "nadella",
])

# Public-company tickers commonly referenced. Treat as case-sensitive (uppercase).
ENTITY_TICKERS = frozenset([
    "AAPL", "MSFT", "GOOG", "GOOGL", "META", "AMZN", "NVDA", "TSLA",
    "NFLX", "AMD", "INTC", "ORCL", "CRM", "ADBE", "PYPL", "SQ",
    "BRK", "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA",
    "WMT", "TGT", "COST", "HD", "LOW", "MCD", "SBUX",
    "XOM", "CVX", "BP", "COP",
    "PFE", "MRNA", "JNJ", "LLY", "UNH",
    "BTC", "ETH", "SOL", "DOGE", "XRP", "ADA",
])

# Case-insensitive asset aliases. Each maps to a canonical entity token —
# "ripple" and "xrp" become the same entity for compatibility checks. Indices
# and crypto cover the FP cases like "S&P close" vs "Bitcoin price".
ASSET_ALIASES = {
    "bitcoin": "btc", "btc": "btc",
    "ethereum": "eth", "ether": "eth",
    "solana": "sol",
    "ripple": "xrp",
    "cardano": "ada",
    "dogecoin": "doge",
    "s&p": "spx", "s&p 500": "spx", "sp500": "spx", "spx": "spx",
    "nasdaq": "ndx", "ndx": "ndx",
    "dow jones": "djia", "dow": "djia", "djia": "djia",
    "russell": "rut", "russell 2000": "rut",
    "vix": "vix",
}
_ASSET_RE = re.compile(
    r"\b(" + "|".join(sorted(ASSET_ALIASES.keys(), key=len, reverse=True))
        .replace(".", r"\.") + r")\b",
    re.IGNORECASE,
)

# US state abbreviations as a strict district-code prefix ("MD-06", "WV-01")
# only. Bare 2-letter codes are too ambiguous to use as standalone entities.
US_STATE_ABBR = frozenset([
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC",
])
_DISTRICT_RE = re.compile(r"\b(" + "|".join(US_STATE_ABBR) + r")-\d{1,3}\b")

# Reverse-map state abbrev → full state name (used by district code check
# to verify the non-district side mentions the state by full name).
_ABBR_TO_FULL = {
    "AL":"alabama","AK":"alaska","AZ":"arizona","AR":"arkansas",
    "CA":"california","CO":"colorado","CT":"connecticut",
    "DE":"delaware","FL":"florida","GA":"georgia","HI":"hawaii",
    "ID":"idaho","IL":"illinois","IN":"indiana","IA":"iowa",
    "KS":"kansas","KY":"kentucky","LA":"louisiana","ME":"maine",
    "MD":"maryland","MA":"massachusetts","MI":"michigan",
    "MN":"minnesota","MS":"mississippi","MO":"missouri",
    "MT":"montana","NE":"nebraska","NV":"nevada",
    "NH":"new hampshire","NJ":"new jersey","NM":"new mexico",
    "NY":"new york","NC":"north carolina","ND":"north dakota",
    "OH":"ohio","OK":"oklahoma","OR":"oregon","PA":"pennsylvania",
    "RI":"rhode island","SC":"south carolina","SD":"south dakota",
    "TN":"tennessee","TX":"texas","UT":"utah","VT":"vermont",
    "VA":"virginia","WA":"washington","WV":"west virginia",
    "WI":"wisconsin","WY":"wyoming","DC":"district of columbia",
}

def _district_codes(text: str) -> frozenset:
    """Extract district codes like 'MD-06', 'WV-01' as frozenset of state abbrevs."""
    if not text:
        return frozenset()
    return frozenset(m for m in _DISTRICT_RE.findall(text))

# Sports teams / countries that often appear in event titles.
# (sports events are skipped already, but team names sometimes appear in
# politics/awards by accident.)
ENTITY_COUNTRIES = frozenset([
    "ukraine", "russia", "israel", "palestine", "gaza", "iran", "iraq",
    "china", "taiwan", "japan", "korea", "india", "pakistan",
    "venezuela", "argentina", "brazil", "mexico", "colombia",
    "france", "germany", "italy", "spain", "poland", "turkey",
    "uk", "britain", "england", "scotland",
    "canada", "australia",
    "syria", "lebanon", "yemen", "saudi", "qatar", "uae",
    "nigeria", "egypt", "ethiopia", "kenya",
])

# US states by full name. State abbreviations (e.g. "PA", "CA") collide with
# too many other tokens (ca = "California" but also "circa") so we only match
# full names — district codes like "PA-07" are caught by the numeric check.
ENTITY_STATES = frozenset([
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "ohio", "oklahoma", "oregon", "pennsylvania", "tennessee", "texas",
    "utah", "vermont", "virginia", "washington", "wisconsin", "wyoming",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "south carolina", "south dakota",
    "west virginia", "rhode island",
    # Brazilian states (most populated / commonly traded)
    "rio de janeiro", "são paulo", "sao paulo", "minas gerais", "bahia",
    "ceará", "ceara", "paraná", "parana", "pernambuco", "santa catarina",
    "rio grande do sul", "goiás", "goias", "amazonas", "espírito santo",
    "espirito santo", "pará", "para", "maranhão", "maranhao",
    # Canadian provinces
    "ontario", "quebec", "québec", "alberta", "saskatchewan", "manitoba",
    "british columbia", "nova scotia", "new brunswick", "newfoundland",
    "prince edward island", "yukon", "northwest territories", "nunavut",
    # Mexican states (high-traffic)
    "mexico city", "ciudad de méxico", "jalisco", "nuevo león", "nuevo leon",
    "veracruz", "puebla", "guanajuato", "chihuahua", "michoacán", "michoacan",
    "oaxaca", "sonora", "sinaloa", "tamaulipas",
    # German Länder (commonly referenced)
    "bavaria", "bayern", "saxony", "sachsen", "hesse", "hessen",
    "thuringia", "thüringen", "brandenburg", "hamburg", "berlin",
    "north rhine-westphalia", "nordrhein-westfalen", "baden-württemberg",
    "baden-wurttemberg", "lower saxony", "niedersachsen",
    "rhineland-palatinate", "schleswig-holstein",
    # Indian states (most populous)
    "uttar pradesh", "maharashtra", "bihar", "west bengal", "madhya pradesh",
    "tamil nadu", "rajasthan", "karnataka", "gujarat", "andhra pradesh",
    "odisha", "telangana", "kerala", "jharkhand", "assam", "punjab",
    "haryana", "chhattisgarh", "uttarakhand", "himachal pradesh",
    # Australian states / UK regions
    "new south wales", "victoria", "queensland", "western australia",
    "south australia", "tasmania", "northern territory",
    "wales", "northern ireland",
])

# Multi-word offices treated as distinct entities — without this, "Texas
# Lieutenant Governor" matches "Texas Governor" because only "texas" + "governor"
# are in the entity sets. Order matters: longer phrases must match first.
ENTITY_OFFICES = frozenset([
    "lieutenant governor", "attorney general", "secretary of state",
    "secretary of defense", "secretary of treasury", "secretary of labor",
    "speaker of the house", "speaker of the senate",
    "majority leader", "minority leader", "whip",
    "prime minister", "deputy prime minister",
    "chief justice", "supreme court justice",
    "fed chair", "fed chairman", "fed chairwoman", "federal reserve chair",
    "vice president", "vice-president",
    "chancellor",
])

_PERSON_RE = re.compile(r"\b(" + "|".join(sorted(ENTITY_PEOPLE)) + r")\b", re.IGNORECASE)
_COUNTRY_RE = re.compile(r"\b(" + "|".join(sorted(ENTITY_COUNTRIES)) + r")\b", re.IGNORECASE)
_TICKER_RE = re.compile(r"\b(" + "|".join(sorted(ENTITY_TICKERS)) + r")\b")
# States sorted longest-first so "new york" matches before "york" (not present
# but defensive against multi-word states).
_STATE_RE = re.compile(
    r"\b(" + "|".join(sorted(ENTITY_STATES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_OFFICE_RE = re.compile(
    r"\b(" + "|".join(sorted(ENTITY_OFFICES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

def _entities(text: str) -> frozenset:
    """Return frozenset of canonical entities (lowercased) — people, countries,
    states/subdivisions, multi-word offices, plus tickers (uppercase)."""
    if not text:
        return frozenset()
    out = set()
    # Strip multi-word offices first so their words don't fragment into
    # individual entity hits (e.g. "lieutenant governor" → {lieutenant governor},
    # not {governor}).
    text_low = text.lower()
    consumed = ""
    last = 0
    for m in _OFFICE_RE.finditer(text_low):
        out.add(m.group(0).lower())
        consumed += text_low[last:m.start()] + " " * (m.end() - m.start())
        last = m.end()
    consumed += text_low[last:]

    out.update(m.lower() for m in _PERSON_RE.findall(consumed))
    out.update(m.lower() for m in _COUNTRY_RE.findall(consumed))
    out.update(m.lower() for m in _STATE_RE.findall(consumed))
    out.update(_TICKER_RE.findall(text))  # tickers case-sensitive on original
    # Asset aliases — canonicalize "bitcoin" and "btc" both to "btc" so they
    # match as the same entity.
    for m in _ASSET_RE.findall(text):
        out.add(ASSET_ALIASES[m.lower()])
    return frozenset(out)


def _entities_compatible(text_a: str, text_b: str) -> bool:
    """Hard-gate: if both texts have entities and don't overlap, reject."""
    ea = _entities(text_a)
    eb = _entities(text_b)
    if not ea or not eb:
        return True  # one side has no entities → leave to other gates
    return bool(ea & eb)


# ---------------------------------------------------------------------------
# Outcome set extraction (for Jaccard tiebreaker)
# ---------------------------------------------------------------------------

def _kalshi_outcome_set(ke: dict) -> frozenset:
    """Normalized set of outcome labels from a Kalshi event's markets."""
    out = set()
    for m in ke.get("markets", []) or []:
        name = m.get("yes_sub_title") or m.get("subtitle") or m.get("title", "")
        if name:
            out.add(_clean(name))
    return frozenset(out)


def _poly_outcome_set(pe: dict) -> frozenset:
    """Normalized set of outcome labels from a Polymarket event's markets."""
    out = set()
    for m in pe.get("markets", []) or []:
        name = m.get("groupItemTitle") or m.get("outcome") or m.get("question", "")
        if name:
            out.add(_clean(name))
    return frozenset(out)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Kalshi rules aggregation
# ---------------------------------------------------------------------------

def _kalshi_rules_text(ke: dict) -> str:
    """Concatenate rules_primary across all markets of a Kalshi event."""
    parts = []
    for m in ke.get("markets", []) or []:
        for f in ("rules_primary", "rules_secondary"):
            v = m.get(f)
            if v:
                parts.append(str(v))
    # Dedup-ish: keep first 1500 chars total
    return " ".join(parts)[:1500]


# ---------------------------------------------------------------------------
# Sentence-embedding similarity (lazy-loaded)
# ---------------------------------------------------------------------------

_EMB_MODEL = None
_EMB_DISABLE = os.getenv("PFM_DISCOVER_NO_EMBED") == "1"


def _emb_model():
    global _EMB_MODEL
    if _EMB_MODEL is None and not _EMB_DISABLE:
        try:
            from sentence_transformers import SentenceTransformer
            _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            print(f"  [warn] embeddings disabled ({e}); falling back to lexical _sim")
            return None
    return _EMB_MODEL


def _embed(texts: list[str]):
    """Batch-encode a list of texts. Returns (N, 384) numpy array, or None."""
    m = _emb_model()
    if m is None or not texts:
        return None
    import numpy as np
    return np.asarray(m.encode(texts, batch_size=64, show_progress_bar=False,
                               normalize_embeddings=True))


def _cosine(a, b) -> float:
    """Cosine of two pre-normalized vectors."""
    import numpy as np
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Cross-encoder re-ranking (lazy-loaded)
# ---------------------------------------------------------------------------
# Bi-encoders embed each side independently — fast (<1ms batched) but coarse.
# Cross-encoders jointly attend over (a, b) pairs and produce much more
# accurate match scores at the cost of ~50ms per pair. We only run the
# cross-encoder on borderline edges (score in the uncertainty band) to keep
# total runtime bounded while still upgrading true paraphrases from MED→HIGH
# and demoting weak edges currently riding above the threshold.

_CROSSENC_MODEL = None
_CROSSENC_DISABLE = os.getenv("PFM_DISCOVER_NO_CROSSENC") == "1"


def _cross_model():
    global _CROSSENC_MODEL
    if _CROSSENC_MODEL is None and not _CROSSENC_DISABLE:
        try:
            from sentence_transformers import CrossEncoder
            _CROSSENC_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        except Exception as e:
            print(f"  [warn] cross-encoder disabled ({e}); skipping re-rank")
            return None
    return _CROSSENC_MODEL


def _cross_score(pairs: list[tuple[str, str]]) -> list[float] | None:
    """Batch-score a list of (a, b) pairs with the cross-encoder.

    Returns a list of floats in [0, 1] (sigmoid-mapped from raw logits),
    or None if the model is disabled / failed to load. ms-marco-MiniLM-L6-v2
    emits raw logits roughly in [-10, 10]; sigmoid is the standard remap.
    """
    m = _cross_model()
    if m is None or not pairs:
        return None
    import math
    try:
        raw = m.predict(pairs, batch_size=32, show_progress_bar=False)
    except Exception as e:
        print(f"  [warn] cross-encoder predict failed ({e}); skipping re-rank")
        return None
    # Sigmoid → [0, 1]. Clip to avoid math.exp overflow on extreme logits.
    out: list[float] = []
    for x in raw:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            xf = 0.0
        if xf > 30.0:
            out.append(1.0)
        elif xf < -30.0:
            out.append(0.0)
        else:
            out.append(1.0 / (1.0 + math.exp(-xf)))
    return out


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
    """Build an inverted index: word -> set of poly event indices.

    Also indexes words from the description so paraphrases catch as candidates
    even when titles don't share lexical tokens. Capped per event.
    """
    index = defaultdict(set)
    norm_titles = []

    for i, pe in enumerate(p_events):
        title = pe.get("title", "")
        nt = _norm(title)
        norm_titles.append(nt)
        for word in _tokens(title):
            if len(word) >= 2:
                index[word].add(i)
        # Pull a few descriptive words too — bounded so popular generic words
        # in long descriptions don't blow up candidate sets.
        desc = pe.get("description") or ""
        if desc:
            desc_words = list(_tokens(desc))
            for w in desc_words[:30]:
                if len(w) >= 4 and w not in GENERIC_WORDS:
                    index[w].add(i)

    return index, norm_titles


def find_candidates(k_title: str, index: dict, max_candidates: int = 80) -> set:
    """Use inverted index to find poly events sharing words with k_title.

    Also matches prefixes ≥4 chars: 'governorship' candidates everything that
    indexes 'governor', so morphological variants don't drop relevant polys.
    """
    words = _tokens(k_title)
    scores = defaultdict(int)
    for w in words:
        if len(w) < 2:
            continue
        for idx in index.get(w, []):
            scores[idx] += 1
        # Prefix match: 'governorship' → also pick up 'governor' index
        if len(w) >= 6:
            prefix = w[:max(4, len(w) - 4)]  # try shortening to find a stem
            for indexed_word in index:
                if indexed_word != w and (
                    indexed_word.startswith(prefix) or w.startswith(indexed_word)
                ) and len(indexed_word) >= 4:
                    for idx in index[indexed_word]:
                        scores[idx] += 1
                        break  # only count once per prefix hit

    if not scores:
        return set()

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidates = set()
    # Loosened: 50 single-hit candidates instead of 20 so short K titles
    # like "Who will win the governorship in Arkansas?" still see the obvious
    # Poly match.
    for idx, count in ranked[:max_candidates]:
        if count >= 2 or len(candidates) < 50:
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
        # Strip parenthetical suffixes like "Naomi (Frieren)" → "Naomi" before
        # fuzzy matching. Polymarket appends the show name to anime-award
        # candidates which artificially deflates the outcome similarity vs
        # Kalshi's bare-name convention.
        name_clean = _strip_paren_suffix(name)
        p_cleans.append(_clean(name_clean))
        p_norms.append(_norm(name_clean))

    for km in k_markets:
        k_ticker = km.get("ticker", "")
        k_suffix = k_ticker.split("-")[-1].lower() if "-" in k_ticker else k_ticker.lower()

        k_outcome = km.get("yes_sub_title") or km.get("subtitle") or km.get("title", "")
        if not k_outcome:
            continue
        # Strip "(Show Name)" etc. on Kalshi side too, in case the platform
        # ever adopts the convention. Today it's a no-op for Kalshi outcomes.
        k_outcome_clean = _strip_paren_suffix(k_outcome)

        k_cl = _clean(k_outcome_clean)
        if "company" in k_cl or "other" in k_cl:
            continue

        best_idx = None
        best_score = 0.0
        k_norm = _norm(k_outcome_clean)

        for j, pm in enumerate(p_markets):
            if j in used_poly or not p_names[j]:
                continue

            # Exact clean match = instant win
            if k_cl == p_cleans[j]:
                best_idx = j
                best_score = 1.0
                break

            # Country/place mismatch guard. Audit 2026-05-18 found
            # 'uae'→'United States' and 'sara'→'Australia' pairings in the
            # Trump-Putin-meet event — outcome FPs caused by fuzzy collisions
            # on short ambiguous suffixes. When BOTH outcome names contain
            # country entities (per ENTITY_COUNTRIES list) and they don't
            # share at least one, hard-reject this pair before fuzzy scoring.
            k_country_ents = _entities(k_outcome_clean) & ENTITY_COUNTRIES
            p_country_ents = _entities(p_names[j]) & ENTITY_COUNTRIES
            if k_country_ents and p_country_ents and not (k_country_ents & p_country_ents):
                continue

            # ACRONYM guard. Audit 2026-05-19 found 'djt → Donald Brodie' —
            # the K suffix 'djt' is Donald J Trump's initials, but fuzzy
            # matched to 'Donald Brodie' because both start with "Don". When
            # K suffix is short (≤4 chars, all letters) and the P outcome is
            # a multi-word name, require the suffix letters to be a prefix
            # of, OR initials of, the P outcome words.
            ks = k_suffix.lower() if k_suffix else ""
            if 2 <= len(ks) <= 4 and ks.isalpha():
                p_words = p_names[j].split()
                if len(p_words) >= 2:
                    # Compute initials of P (first letter of each word)
                    p_initials = "".join(w[0].lower() for w in p_words if w)
                    # Accept if K-suffix == P-initials (or its prefix/suffix
                    # subset) OR K-suffix is contained in first word.
                    first_word = p_words[0].lower()
                    if (ks != p_initials
                        and ks not in p_initials
                        and p_initials[:len(ks)] != ks
                        and not first_word.startswith(ks)):
                        continue

            score = _sim(k_norm, p_norms[j])
            if score > best_score:
                best_score = score
                best_idx = j

        if best_idx is not None and best_score >= 0.55:
            # Date-key consistency check. Kalshi outcome suffixes often encode
            # a target date like "26jul01" (YY + month-abbr + DD). When the
            # matched Poly outcome explicitly names a different month, this
            # is an outcome-FP (audit caught `'26jul01'→'February 14, 2026'`
            # on the Claude-5 release market — internal contradiction).
            if not _date_keys_consistent(k_suffix, p_names[best_idx]):
                continue
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
    - Sentence-embedding similarity (replaces lexical _sim as primary score)
    - Polarity + entity (NER) hard-gates
    - rules_primary (Kalshi) folded into matching text
    - Outcome-set hard-gate (≥40% of smaller side must pair ≥0.6)
    - Score-sorted greedy global assignment (avoids first-event stealing)
    """
    print("  Building Poly word index...", flush=True)
    index, p_norms = build_poly_index(p_events)
    print(f"  Index built: {len(index)} unique words across {len(p_events)} events", flush=True)

    # Precompute embeddings for poly events (title + clipped description).
    # Done once for the whole p_events list, not per Kalshi iteration.
    p_embeds = None
    p_emb_texts = []
    if not _EMB_DISABLE:
        for pe in p_events:
            t = (pe.get("title") or "") + ". " + ((pe.get("description") or "")[:200])
            p_emb_texts.append(t.strip())
        t_emb = time.time()
        p_embeds = _embed(p_emb_texts)
        if p_embeds is not None:
            print(f"  Embedded {len(p_emb_texts)} Poly events in {time.time()-t_emb:.1f}s", flush=True)

    # Collected edges (k_idx, p_idx, score, ...meta). After collection we sort
    # by score desc and greedily assign — each Kalshi and each Poly event used
    # at most once. This avoids the "first Kalshi event steals best Poly"
    # problem of the old per-k greedy.
    edges: list[dict] = []
    used_poly_slugs = set(existing_poly_slugs or [])
    checked = 0
    t0 = time.time()

    # Precompute Kalshi embed inputs (title + rules_primary)
    k_emb_texts = []
    for ke in k_events:
        title = ke.get("title", "") or ke.get("sub_title", "")
        rules = _kalshi_rules_text(ke)
        k_emb_texts.append((title + ". " + rules[:200]).strip())
    k_embeds = _embed(k_emb_texts) if (p_embeds is not None) else None
    if k_embeds is not None:
        print(f"  Embedded {len(k_embeds)} Kalshi events", flush=True)

    # Embedding-based candidate filter. Compute the full K×P cosine matrix
    # once and per-K take the top-30 nearest Polys. This augments the lexical
    # candidate set so paraphrase pairs with no word overlap
    # ("Anthropic release Claude 5" ↔ "Claude 5 released by…?") still see
    # each other as candidates. Cheap: one matmul ≈ 3s for 7k×10k normalized.
    emb_candidates_per_k = None
    if k_embeds is not None and p_embeds is not None:
        import numpy as np
        t_sim = time.time()
        sim_matrix = k_embeds @ p_embeds.T  # cosine, since both normalized
        # For each K row, pick top 30 P indices.
        top_n = 30
        emb_candidates_per_k = np.argpartition(-sim_matrix, top_n, axis=1)[:, :top_n]
        print(f"  K×P cosine matrix ({sim_matrix.shape}) and top-{top_n} candidates in {time.time()-t_sim:.1f}s", flush=True)

    # Precompute outcome label sets for Jaccard tiebreaker.
    k_outcome_sets = [_kalshi_outcome_set(ke) for ke in k_events]
    p_outcome_sets = [_poly_outcome_set(pe) for pe in p_events]

    for k_idx, ke in enumerate(k_events):
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

        # Step 1: candidate filter — lexical word overlap UNION embedding-NN.
        # Embedding-NN catches paraphrases the lexical index misses.
        candidates = find_candidates(k_title, index)
        if emb_candidates_per_k is not None:
            candidates = candidates | set(int(i) for i in emb_candidates_per_k[k_idx])
        if not candidates:
            continue

        # Pre-compute Kalshi metadata for matching
        k_desc = ke.get("sub_title") or ""
        k_rules = _kalshi_rules_text(ke)
        k_combined = f"{k_title} {k_desc} {k_rules}".strip()
        k_end = ke.get("close_date") or ke.get("expected_expiration_date") or ke.get("settlement_date")
        k_end_dt = _parse_date(k_end)
        k_polarity = _polarity(k_title)  # title only — descriptions have boilerplate
        k_entities = _entities(k_title)  # title only — rules list candidate names that drown the signal

        # Step 2: collect candidate edges that pass hard-gates. Final
        # assignment is global score-sorted greedy (after this loop).
        k_norm = _norm(k_title)
        k_words = _tokens(k_title)

        for idx in candidates:
            p_slug = p_events[idx].get("slug", "")
            if p_slug in used_poly_slugs:
                continue

            pe = p_events[idx]
            p_title = pe.get("title", "")
            p_desc = pe.get("description") or ""
            p_combined = f"{p_title} {p_desc}".strip()

            # Skip Poly sports events too
            if _detect_sport(p_title):
                continue

            # DATE PROXIMITY (±60d). Compute delta for later score nudge.
            p_end = pe.get("endDate") or pe.get("end_date_iso")
            p_end_dt = _parse_date(p_end)
            date_delta_days = None
            if k_end_dt and p_end_dt:
                date_delta_days = abs((k_end_dt - p_end_dt).days)
                if date_delta_days > 60:
                    continue

            # NUMERIC TOKEN
            k_title_nums = _numbers_in(k_title)
            p_title_nums = _numbers_in(p_title)
            if k_title_nums and p_title_nums and not (k_title_nums & p_title_nums):
                continue

            # DISTRICT CODE — "MD-06" must share the state with the other
            # side. Bidirectional: K with district vs P without (and vice
            # versa) only matches when the side lacking the district has the
            # corresponding full state name in its title.
            k_districts = _district_codes(k_title)
            p_districts = _district_codes(p_title)
            if k_districts and p_districts and not (k_districts & p_districts):
                continue
            if k_districts != p_districts:
                # Asymmetric: one side has district code, the other doesn't.
                # Require the full state name to appear in the other side.
                only = k_districts or p_districts
                other_title = (p_title if k_districts else k_title).lower()
                full_names = {_ABBR_TO_FULL.get(d, "") for d in only}
                if not any(name and name in other_title for name in full_names):
                    continue

            # DISTINGUISHING WORDS
            if _check_distinguishing(k_title, p_title):
                continue

            # MONTH-WINDOW guard: "before July" vs "in May" are sibling
            # markets, not the same event.
            if _check_month_window(k_title, p_title):
                continue

            # YEAR-WINDOW guard: "2028 D nomination" vs "announce before 2027"
            # — different election cycles, can't be the same event.
            if _check_year_window(k_title, p_title):
                continue
            # Also gate vs rules text on Kalshi side: many K markets put the
            # year only in rules_primary (KX2028DRUN's title omits 2028).
            if _check_year_window(k_rules, p_title):
                continue

            # OFFICE-MISMATCH guard: mayor vs senate, governor vs house, etc.
            if _check_office_mismatch(k_title, p_title):
                continue

            # "What <noun> will X" topic-noun guard: nicknames vs animals.
            if _check_what_noun(k_title, p_title):
                continue

            # Earnings-call subject (Home Depot vs NVIDIA).
            if _check_earnings_subject(k_title, p_title):
                continue

            # POLARITY (new hard-gate). Apply ONLY to titles, not descriptions —
            # Polymarket disclaimers like "independents will not be encompassed
            # by the Democrat or Republican categories" contain incidental
            # negations that don't change the question's polarity.
            if k_polarity != _polarity(p_title):
                continue

            # ENTITY OVERLAP (new hard-gate). Title-only — descriptions contain
            # candidate name dumps and explanatory text that introduce spurious
            # entities and cause false rejections.
            if not _entities_compatible(k_title, p_title):
                continue

            # PRIMARY SCORE: blend embedding (semantic) + lexical (exact-text)
            # Lexical acts as a tiebreaker so near-identical titles
            # ("Maryland Governor winner?" vs "Maryland Governor Election Winner")
            # beat semantically-close-but-different siblings competing for the
            # same Poly slot.
            lex_score = _sim(k_norm, p_norms[idx])
            if k_embeds is not None and p_embeds is not None:
                emb_score = _cosine(k_embeds[k_idx], p_embeds[idx])
                emb_score = max(0.0, (emb_score + 1.0) / 2.0)  # cosine→[0,1]
                score = 0.7 * emb_score + 0.3 * lex_score
            else:
                score = lex_score

            # Mid-confidence band: require significant content overlap
            if 0.85 > score >= min_score:
                p_words = _tokens(p_title)
                k_sig = {w for w in k_words if len(w) >= 4 and w not in GENERIC_WORDS}
                p_sig = {w for w in p_words if len(w) >= 4 and w not in GENERIC_WORDS}
                if k_sig and p_sig:
                    overlap = k_sig & p_sig
                    min_set = min(len(k_sig), len(p_sig))
                    if min_set > 0 and len(overlap) / min_set < 0.4:
                        continue

            # Description boost (lexical signal in addition to embedding)
            if k_desc and p_desc and 0.65 <= score < 0.85:
                k_desc_sig = {w for w in _tokens(k_desc) if len(w) >= 5 and w not in GENERIC_WORDS}
                p_desc_sig = {w for w in _tokens(p_desc) if len(w) >= 5 and w not in GENERIC_WORDS}
                if k_desc_sig and p_desc_sig and len(k_desc_sig & p_desc_sig) >= 2:
                    score = min(1.0, score + 0.05)

            # Gate FIRST on raw score — bonus shouldn't rescue weak matches.
            if score < min_score:
                continue

            # Time-aware nudge AFTER gate. Resolution dates within a week
            # strongly suggest the same event; 30-60d apart suggests sibling
            # cycles (primary vs general). Small (±0.02) so it tie-breaks
            # without reordering the bulk of the greedy assignment ranking.
            if date_delta_days is not None:
                if date_delta_days <= 7:
                    score = min(1.0, score + 0.02)
                elif date_delta_days >= 30:
                    score = max(0.0, score - 0.02)

            # Outcome-set Jaccard tiebreaker. Pairs whose markets list the
            # same candidates/answers are obviously the same event even when
            # titles drift; weight is small (max +0.03) so it only resolves
            # ties without reshaping the ranking.
            outc_jaccard = _jaccard(k_outcome_sets[k_idx], p_outcome_sets[idx])
            if outc_jaccard > 0:
                score = min(1.0, score + 0.03 * outc_jaccard)

            edges.append({
                "k_idx": k_idx, "p_idx": idx, "score": score,
                "ke": ke, "pe": pe, "k_title": k_title,
                "k_rules": k_rules, "p_desc": p_desc, "p_title": p_title,
            })

    # ── Cross-encoder re-rank on borderline edges ──
    # Bi-encoder cosine + lexical is fast but coarse near the decision
    # boundary. We pick edges whose blended score falls in the uncertainty
    # band [0.68, 0.92] and re-score (k_title + k_rules, p_title + p_desc)
    # jointly with a cross-encoder, then blend:
    #     final = 0.5 * old_score + 0.5 * crossenc_sigmoid
    # This typically pushes true paraphrases up into HIGH territory and
    # drags weak FP-risk edges below the MED→HIGH threshold. The pool is
    # capped at 200 so worst-case re-rank latency is ~10s (50ms × 200).
    if not _CROSSENC_DISABLE:
        borderline_idxs = [
            i for i, e in enumerate(edges) if 0.68 <= e["score"] <= 0.92
        ]
        # Cap: stratified sample around the two tier boundaries (MED↔LOW at
        # ~0.76 and MED↔HIGH at ~0.88). Re-ranking has highest leverage near
        # those boundaries since the bi-encoder's score and the true match
        # quality disagree more often at the edges than in the middle.
        if len(borderline_idxs) > 200:
            def boundary_dist(i: int) -> float:
                s = edges[i]["score"]
                return min(abs(s - 0.76), abs(s - 0.88))
            borderline_idxs.sort(key=boundary_dist)
            borderline_idxs = borderline_idxs[:200]
        if borderline_idxs:
            t_ce = time.time()
            pairs = []
            for i in borderline_idxs:
                e = edges[i]
                k_text = (e["k_title"] + " " + (e["k_rules"] or "")[:100]).strip()
                p_text = (e["p_title"] + " " + (e["p_desc"] or "")[:100]).strip()
                pairs.append((k_text, p_text))
            ce_scores = _cross_score(pairs)
            if ce_scores is not None:
                for i, ce in zip(borderline_idxs, ce_scores):
                    old = edges[i]["score"]
                    final = 0.5 * old + 0.5 * float(ce)
                    edges[i]["score_pre_crossenc"] = old
                    edges[i]["crossenc"] = float(ce)
                    edges[i]["score"] = min(1.0, max(0.0, final))
                print(
                    f"  Cross-encoder re-ranked {len(borderline_idxs)} borderline edges "
                    f"in {time.time()-t_ce:.1f}s",
                    flush=True,
                )

    # ── Score-sorted greedy global assignment ──
    # Each Kalshi and each Poly event used at most once. This avoids the
    # old greedy's first-event-steals-best-Poly pattern.
    edges.sort(key=lambda e: e["score"], reverse=True)
    print(f"  Collected {len(edges)} candidate edges; assigning...", flush=True)
    results = []
    used_k = set()
    used_p_idx = set()

    for e in edges:
        if e["k_idx"] in used_k or e["p_idx"] in used_p_idx:
            continue
        ke = e["ke"]; pe = e["pe"]
        p_slug = pe.get("slug", "")
        if p_slug in used_poly_slugs:
            continue

        k_markets = ke.get("markets", []) or []
        p_markets = pe.get("markets", []) or []
        outcome_matches, unmatched_k, unmatched_p = match_outcomes(k_markets, p_markets)

        if not outcome_matches:
            continue

        avg_out_score = sum(om["score"] for om in outcome_matches) / len(outcome_matches)
        high_quality = sum(1 for om in outcome_matches if om["score"] >= 0.8)

        # OUTCOME-SET HARD-GATE: require ≥30% of smaller side to pair ≥0.6.
        # Tighter than the legacy avg-score gate but looser than 40% which
        # killed legit matches where Kalshi+Poly name outcomes very differently
        # ("CA-14 special election" vs "CA-14 Special Election Winner?").
        min_side = min(len(k_markets), len(p_markets)) or 1
        good_matches = sum(1 for om in outcome_matches if om["score"] >= 0.6)
        if good_matches / min_side < 0.3 and high_quality < 2:
            if not (len(outcome_matches) == 1 and outcome_matches[0]["score"] >= 0.6):
                continue

        # Legacy avg-score floor
        if avg_out_score < 0.55 and high_quality < 1:
            continue

        # Audit 2026-05-19 fix #3: SINGLE-OUTCOME minimum. Events with one
        # mapping entry slip 0.71 fuzzy matches through (Ann Arbor mayor →
        # Delaware Senate). Force LOW unless K+P outcome names share ≥2
        # content words.
        k_title = e["k_title"]
        p_title = pe.get("title", "")
        if len(outcome_matches) == 1:
            om = outcome_matches[0]
            ko = om.get("k_outcome", "") or ""
            po = om.get("p_outcome", "") or ""
            k_words = {w for w in _tokens(ko) if len(w) >= 4}
            p_words = {w for w in _tokens(po) if len(w) >= 4}
            shared = k_words & p_words
            if len(shared) < 2:
                # Demote to LOW (don't fully skip — let the LOW tier capture
                # it so we can audit). The downstream HIGH-only engine
                # filter will silently drop it.
                pass  # falls through; confidence tier below will hit LOW.

        # Audit fix #4: CITY-LIST guard. When outcomes match cleanly (mean
        # ≥0.85, ≥3 matches) but event titles diverge (_sim <0.7), the
        # outcomes are too compatible to discriminate — both sides ask about
        # cities/dates, hiding scope divergence (WAYMO 2026 vs by-June-30).
        title_sim_check = _sim(_norm(k_title), _norm(p_title))
        if (len(outcome_matches) >= 3 and avg_out_score >= 0.85
                and title_sim_check < 0.70):
            # Force LOW so it doesn't pollute MED/HIGH.
            force_low = True
        else:
            force_low = False

        used_k.add(e["k_idx"])
        used_p_idx.add(e["p_idx"])
        used_poly_slugs.add(p_slug)

        best_score = e["score"]
        confusable_flag = _check_confusable(k_title, p_title)
        k_count = len(k_markets); p_count = len(p_markets)
        count_ratio = min(k_count, p_count) / max(k_count, p_count) if max(k_count, p_count) > 0 else 1.0

        # Thresholds calibrated for blended score (0.7*emb + 0.3*lex).
        # Embedding paraphrase matches sit in [0.78, 0.88] for clear cases;
        # near-identical text matches reach [0.92, 1.00].
        single_out_weak = (len(outcome_matches) == 1 and
                           len({w for w in _tokens(outcome_matches[0].get("k_outcome", "") or "")
                                if len(w) >= 4} &
                               {w for w in _tokens(outcome_matches[0].get("p_outcome", "") or "")
                                if len(w) >= 4}) < 2)
        if force_low or confusable_flag or single_out_weak:
            confidence = "LOW"
        elif best_score >= 0.88 and avg_out_score >= 0.80 and count_ratio >= 0.5:
            confidence = "HIGH"
        elif best_score >= 0.76 and avg_out_score >= 0.65:
            confidence = "MED"
        else:
            confidence = "LOW"

        k_end = ke.get("close_date") or ke.get("expected_expiration_date") or ke.get("settlement_date")
        p_end = pe.get("endDate") or pe.get("end_date_iso")
        k_desc = ke.get("sub_title") or ke.get("title", "")
        p_desc = pe.get("description") or pe.get("title", "")
        k_category = ke.get("category", "") or ke.get("series_ticker", "")

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
    print(f"  Matching done in {elapsed:.1f}s ({len(edges)} edges -> {len(results)} assignments)", flush=True)
    results.sort(key=lambda x: x["event_score"], reverse=True)

    # Coverage tracking: top unmatched Poly events by liquidity. These are
    # markets where some arb might exist but we failed to pair to Kalshi —
    # the bigger the unmatched volume, the more $ we're leaving on the table.
    try:
        assigned_p_idx = used_p_idx
        unmatched = []
        for i, pe in enumerate(p_events):
            if i in assigned_p_idx:
                continue
            slug = pe.get("slug", "")
            if slug in used_poly_slugs:
                continue
            if _detect_sport(pe.get("title") or ""):
                continue
            vol = pe.get("volume24hr") or pe.get("volume") or 0
            try:
                vol = float(vol)
            except (TypeError, ValueError):
                vol = 0
            unmatched.append({
                "slug": slug,
                "title": pe.get("title", ""),
                "end_date": pe.get("endDate") or pe.get("end_date_iso"),
                "volume_24h": vol,
                "market_count": len(pe.get("markets", []) or []),
            })
        unmatched.sort(key=lambda x: x["volume_24h"], reverse=True)
        out_path = "unmatched_high_volume.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_unmatched": len(unmatched),
                "top": unmatched[:50],
            }, f, indent=2, ensure_ascii=False)
        print(f"  Coverage: {len(unmatched)} unmatched Polys; top 50 saved to {out_path}", flush=True)
    except Exception as e:
        print(f"  [warn] coverage tracking failed: {e}", flush=True)

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
    # Default 0.60 (was 0.65) — user prefers more coverage across categories
    # (entertainment, sci-tech, economics, crypto, climate, mentions) over
    # election-dominated precision. FPs are tagged with their confidence
    # badge in the UI so they're easy to filter.
    min_score  = 0.60
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
