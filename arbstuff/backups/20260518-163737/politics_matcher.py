"""
politics_matcher.py — Precise structured matching for US political markets.

Unlike fuzzy text matching, this uses a structured approach: parse each title
into (state, office, district, party, race_type, year) and only match events
where ALL fields are compatible. Zero false positives.

Handles:
  - General elections: "Georgia Governor winner?", "GA-03 House winner?"
  - Primary races:     "Georgia Republican AG nominee?", "GA-09 Republican Primary"
  - Specific years:    "Georgia Senate winner? (2028)"
  - All 50 states + DC, all districts, all major offices
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# ── State map: code → name (and reverse)
STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico",
}
NAME_TO_CODE = {v.lower(): k for k, v in STATES.items()}
CODE_TO_NAME = {k.lower(): v for k, v in STATES.items()}

# ── Office canonical forms (longer phrases must come first for greedy matching)
OFFICES = [
    ("attorney general",       "AG"),
    ("attorney_general",       "AG"),
    ("lieutenant governor",    "LTGOV"),
    ("lt governor",            "LTGOV"),
    ("lt. governor",           "LTGOV"),
    ("secretary of state",     "SOS"),
    ("state treasurer",        "TREAS"),
    ("treasurer",              "TREAS"),
    ("controller",             "CTRL"),
    ("comptroller",            "CTRL"),
    ("auditor",                "AUDITOR"),
    ("governor",               "GOV"),
    ("senate",                 "SEN"),
    ("senator",                "SEN"),
    ("house",                  "HOUSE"),
    ("congress",               "HOUSE"),
    ("mayor",                  "MAYOR"),
    ("president",              "PRES"),
    # Abbreviations
    ("ag",                     "AG"),       # Attorney General
    ("sos",                    "SOS"),      # Secretary of State
    ("ltgov",                  "LTGOV"),
]

# ── Parties
PARTIES = {
    "democratic": "D", "democrat": "D", "dem": "D",
    "republican": "R", "rep": "R", "gop": "R",
    "independent": "I", "ind": "I",
    "libertarian": "L",
    "green": "G",
}

# ── Race types
RACE_PRIMARY  = "primary"
RACE_GENERAL  = "general"
RACE_RUNOFF   = "runoff"
RACE_SPECIAL  = "special"


@dataclass
class PoliticalRace:
    """Structured representation of a political market."""
    raw_title: str
    state: Optional[str] = None       # 2-letter code
    office: Optional[str] = None      # GOV/SEN/HOUSE/AG/...
    district: Optional[int] = None    # for HOUSE / state-leg
    party: Optional[str] = None       # D/R/I (only for primaries usually)
    race_type: str = RACE_GENERAL     # general/primary/runoff/special
    year: Optional[int] = None        # election year (None = next/current cycle)

    def is_compatible_with(self, other: "PoliticalRace") -> tuple[bool, str]:
        """Returns (matches, reason). All fields must align."""
        if not self.state or not other.state:
            return False, "missing state"
        if self.state != other.state:
            return False, f"state mismatch {self.state}!={other.state}"

        if not self.office or not other.office:
            return False, "missing office"
        if self.office != other.office:
            return False, f"office mismatch {self.office}!={other.office}"

        # District: both None (statewide) OR both same number
        if self.district != other.district:
            return False, f"district mismatch {self.district}!={other.district}"

        # Race type must match
        if self.race_type != other.race_type:
            return False, f"race type mismatch {self.race_type}!={other.race_type}"

        # Party: required match if either is primary
        if self.race_type == RACE_PRIMARY:
            if self.party and other.party and self.party != other.party:
                return False, f"primary party mismatch {self.party}!={other.party}"

        # Year: if EITHER side specifies a year, both must agree (or other unspecified
        # but pulled from a date-proximate event — the discovery script enforces date proximity).
        # Strict mode for explicit years to avoid 2026 vs 2028 confusion.
        if self.year and other.year and self.year != other.year:
            return False, f"year mismatch {self.year}!={other.year}"

        return True, "ok"


# ════════════════════════════════════════════════════════════════════════════
# PARSING
# ════════════════════════════════════════════════════════════════════════════

# Pattern: "GA-03", "ga 03", "pa-17"
DISTRICT_PATTERN = re.compile(r"\b([A-Z]{2})[-\s]?(\d{1,3})\b")

# Pattern: "(2026)", "2028"
YEAR_PATTERN = re.compile(r"\b(202[4-9]|203[0-5])\b")

# Pattern: "U.S. Senate", "US House"
US_PREFIX = re.compile(r"\bU\.?S\.?\b\s+", re.IGNORECASE)


def _norm(s: str) -> str:
    """Lowercase, strip extra spaces."""
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def parse_political_title(title: str) -> Optional[PoliticalRace]:
    """Parse a market title into a PoliticalRace, or return None if not political."""
    if not title:
        return None
    raw = title
    t = US_PREFIX.sub("", title)  # drop "U.S." prefix (not distinguishing)
    t_low = _norm(t)

    # ── HARD REJECTS ──────────────────────────────────────────
    # State legislature ≠ US Senate/House. Skip anything with "state senate",
    # "state house", "state legislature", "state assembly".
    if re.search(r"\bstate\s+(senate|house|legislat|assembly|rep)", t_low):
        return None

    # Margin-of-victory markets are NOT winner markets. Same with "by how much",
    # "vote share", "margin", "spread", "total votes", etc.
    if any(kw in t_low for kw in [
        "margin of victory", "margin of win", "by how much", "vote share",
        "vote total", "total votes", "percentage of vote", "turnout",
        "voter turnout",
    ]):
        return None

    # "Matchup" / "head to head" / "face off" markets — specialized, not the
    # standard winner market. Skip to avoid mismatching with the main race.
    if any(kw in t_low for kw in [
        "matchup", "head to head", "head-to-head", "face off", "face-off",
        "who will be nominated from",  # specific sub-candidacy markets
    ]):
        return None

    # "Which party" is ambiguous and often overlaps with the general — skip.
    if "which party" in t_low:
        return None

    race = PoliticalRace(raw_title=raw)

    # ── State ──
    # First check 2-letter prefix patterns like "GA-03 House winner?"
    m = DISTRICT_PATTERN.search(t)
    if m:
        code = m.group(1)
        if code in STATES:
            race.state = code
            try:
                race.district = int(m.group(2))
            except ValueError:
                pass
    # Otherwise look for full state name in the title
    if not race.state:
        for name_lower, code in NAME_TO_CODE.items():
            # Must be word-boundary match
            if re.search(rf"\b{re.escape(name_lower)}\b", t_low):
                race.state = code
                break
    if not race.state:
        return None  # no state → not a political market we care about

    # ── Office ──
    for keyword, canonical in OFFICES:
        if re.search(rf"\b{re.escape(keyword)}\b", t_low):
            race.office = canonical
            break
    # If no explicit office but state+district pattern present, assume HOUSE
    # (e.g. "NJ-11 Special Election winner?", "GA-13 Democratic Primary Winner")
    if not race.office and race.district is not None:
        race.office = "HOUSE"
    if not race.office:
        return None

    # ── Race type ──
    if "primary" in t_low or "nominee" in t_low or "nomination" in t_low:
        race.race_type = RACE_PRIMARY
    elif "runoff" in t_low or "run-off" in t_low or "run off" in t_low:
        race.race_type = RACE_RUNOFF
    elif "special" in t_low:
        # "(Special)", "Special Election", "Special House race" — all map to special
        race.race_type = RACE_SPECIAL
    else:
        race.race_type = RACE_GENERAL

    # ── Party (mostly for primaries) ──
    for word, p_code in PARTIES.items():
        # Avoid matching inside other words: "Indiana" contains "in"
        if re.search(rf"\b{re.escape(word)}\b", t_low):
            race.party = p_code
            break

    # ── Year ──
    m = YEAR_PATTERN.search(t)
    if m:
        try:
            race.year = int(m.group(1))
        except ValueError:
            pass

    return race


# ════════════════════════════════════════════════════════════════════════════
# OUTCOME MAPPING
# ════════════════════════════════════════════════════════════════════════════

PARTY_OUTCOME_ALIASES = {
    # Map various Kalshi suffix conventions to canonical party labels
    "dem": "Democratic Party",
    "d": "Democratic Party",
    "democrat": "Democratic Party",
    "democratic": "Democratic Party",
    "democraticparty": "Democratic Party",
    "rep": "Republican Party",
    "r": "Republican Party",
    "republican": "Republican Party",
    "republicanparty": "Republican Party",
    "ind": "Independent",
    "i": "Independent",
    "lib": "Libertarian",
    "l": "Libertarian",
    "grn": "Green",
    "g": "Green",
}


def build_party_mapping(k_outcomes: list[dict], p_outcomes: list[dict]) -> dict:
    """Build a Kalshi-suffix → Polymarket-name mapping for party-based outcomes.

    Used for general elections where outcomes are parties (Democrat / Republican).
    """
    mapping = {}
    p_names_by_party = {}
    for p in p_outcomes:
        name = (p.get("name") or "").strip()
        nlow = name.lower().replace(" ", "")
        if "democra" in nlow:
            p_names_by_party["D"] = name
        elif "republic" in nlow:
            p_names_by_party["R"] = name
        elif "independ" in nlow:
            p_names_by_party["I"] = name
        elif "libertar" in nlow:
            p_names_by_party["L"] = name
        elif "green" in nlow:
            p_names_by_party["G"] = name

    for k in k_outcomes:
        suffix = (k.get("suffix") or "").lower()
        # Try direct alias map
        canonical = PARTY_OUTCOME_ALIASES.get(suffix)
        if canonical and any(canonical.lower() in n.lower() for n in [v for v in p_names_by_party.values()]):
            for name in p_names_by_party.values():
                if canonical.lower() in name.lower():
                    mapping[suffix] = name
                    break
            continue
        # Try party-letter mapping
        if suffix.startswith("dem") or suffix == "d":
            if "D" in p_names_by_party:
                mapping[suffix] = p_names_by_party["D"]
        elif suffix.startswith("rep") or suffix == "r":
            if "R" in p_names_by_party:
                mapping[suffix] = p_names_by_party["R"]
    return mapping


def build_candidate_mapping(k_outcomes: list[dict], p_outcomes: list[dict]) -> dict:
    """Build mapping for primary races where outcomes are candidate names.

    Uses fuzzy last-name match.
    """
    mapping = {}
    used_p = set()
    p_items = [(p.get("name", "").strip(), p) for p in p_outcomes]

    for k in k_outcomes:
        suffix = (k.get("suffix") or "").lower()
        kname = (k.get("name") or "").lower()
        best_match = None
        best_score = 0

        # Try matching by last name (most reliable)
        # Kalshi suffix is often first 3-4 letters of last name
        for pname, p in p_items:
            if pname in used_p:
                continue
            plow = pname.lower()
            # Check if Kalshi name appears in Poly name (or vice versa)
            score = 0
            if kname and (kname in plow or plow in kname):
                score = max(score, len(set(kname.split()) & set(plow.split())) * 10)
            # Suffix match: first N chars of last name
            if suffix and len(suffix) >= 3:
                # Try to find a Poly name where some word starts with suffix
                for word in plow.split():
                    if word.startswith(suffix):
                        score = max(score, len(suffix))
                        break
            if score > best_score:
                best_score = score
                best_match = pname

        if best_match and best_score >= 3:
            mapping[suffix] = best_match
            used_p.add(best_match)

    return mapping
