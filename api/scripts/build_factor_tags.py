"""Build /web/data/factor_tags.json from factors.yml.

Derives fine-grained tags via regex/keyword on each factor's id, name, slug,
description, theme, and embedded `vol_1mo` / `dte` hints.

Output schema (per factor id):
    {
      "asset_class":       str,         # equity|crypto|fx|commodity|rate|election|sport|geopolitical|...
      "sub_asset":         str | null,  # BTC, ETH, AAPL, fed-funds, NFL, F1, ...
      "geo":               list[str],   # ["US", "Russia", ...]
      "event_type":        str,         # binary|range|ladder|multi-outcome
      "resolution_period": str | null,  # 2026-Q1 | 2027-H1 | 2028 | ...
      "liquidity_tier":    str,         # A | B | C | unknown
      "full_tags":         list[str]
    }

Run from repo root:
    python api/scripts/build_factor_tags.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FACTORS_YML = ROOT / "api" / "src" / "pfm" / "factors.yml"
OUT = ROOT / "web" / "data" / "factor_tags.json"


# --------------------------------------------------------------------------- #
# Keyword tables
# --------------------------------------------------------------------------- #

CRYPTO_TICKERS = {
    "BTC": [r"\bbtc\b", r"bitcoin"],
    "ETH": [r"\beth\b", r"ethereum"],
    "SOL": [r"\bsol\b", r"solana"],
    "XRP": [r"\bxrp\b", r"ripple"],
    "DOGE": [r"\bdoge\b", r"dogecoin"],
    "ADA": [r"\bada\b", r"cardano"],
    "AVAX": [r"\bavax\b", r"avalanche"],
    "BNB": [r"\bbnb\b", r"binance.coin"],
    "LTC": [r"\bltc\b", r"litecoin"],
    "LINK": [r"\blink\b", r"chainlink"],
    "MATIC": [r"\bmatic\b", r"polygon"],
    "DOT": [r"\bpolkadot\b"],
    "PEPE": [r"\bpepe\b"],
    "SHIB": [r"\bshib\b", r"shiba"],
    "TRX": [r"\btron\b"],
}

EQUITY_TICKERS = [
    # Mega-cap tech / mag7
    "AAPL",
    "MSFT",
    "GOOGL",
    "GOOG",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    # Other liquid names commonly traded as PM/Kalshi factors
    "AMD",
    "INTC",
    "AVGO",
    "ORCL",
    "CRM",
    "NFLX",
    "DIS",
    "BA",
    "JPM",
    "GS",
    "BRK",
    "WMT",
    "COST",
    "HD",
    "PEP",
    "KO",
    "XOM",
    "CVX",
    "PFE",
    "JNJ",
    "UNH",
    "V",
    "MA",
    "PYPL",
    "SQ",
    "COIN",
    "MSTR",
    "PLTR",
    "SMCI",
    "ARM",
    "MU",
    "QCOM",
    "TSM",
    "ASML",
    "BABA",
    "NIO",
    "RIVN",
    "LCID",
    "F",
    "GM",
    "UBER",
    "LYFT",
    "ABNB",
    "DASH",
    "SHOP",
    "SPOT",
    "ROKU",
    "SNAP",
    "PINS",
    "TWLO",
    "ZM",
    "DOCU",
    "OKTA",
    "CRWD",
    "DDOG",
    "NET",
    "SNOW",
    "MDB",
]

# Companies referred to by name (not ticker) — map to ticker
COMPANY_NAMES = {
    "AAPL": [r"\bapple\b"],
    "MSFT": [r"\bmicrosoft\b"],
    "GOOGL": [r"\bgoogle\b", r"\balphabet\b"],
    "AMZN": [r"\bamazon\b"],
    "META": [r"\bmeta\b", r"\bfacebook\b", r"\binstagram\b"],
    "NVDA": [r"\bnvidia\b"],
    "TSLA": [r"\btesla\b"],
    "NFLX": [r"\bnetflix\b"],
    "DIS": [r"\bdisney\b"],
    "BA": [r"\bboeing\b"],
    "COIN": [r"\bcoinbase\b"],
    "MSTR": [r"\bmicrostrategy\b", r"\bstrategy inc\b"],
    "OPENAI": [r"\bopenai\b"],
    "ANTHROPIC": [r"\banthropic\b"],
    "XAI": [r"\bxai\b", r"\bgrok\b"],
    "SPACEX": [r"\bspacex\b", r"\bstarship\b"],
    "TSM": [r"\btsmc\b", r"taiwan semi"],
}

GEO_KEYWORDS = {
    "US": [
        r"\bus\b",
        r"\bunited states\b",
        r"\bamerica\b",
        r"\bfederal reserve\b",
        r"\bfed\b",
        r"\btrump\b",
        r"\bbiden\b",
        r"\bharris\b",
        r"\bnyc\b",
        r"\bcalifornia\b",
        r"\btexas\b",
        r"\bcongress\b",
        r"\bsenate\b",
    ],
    "EU": [
        r"\beuro\b",
        r"\beurope\b",
        r"\bfrance\b",
        r"\bgermany\b",
        r"\bspain\b",
        r"\bitaly\b",
        r"\bnetherlands\b",
        r"\bpoland\b",
        r"\bnato\b",
        r"\bukraine\b",
        r"\bmacron\b",
        r"\bmerz\b",
    ],
    "UK": [
        r"\buk\b",
        r"\bbritain\b",
        r"\bbritish\b",
        r"\benglan?d\b",
        r"\bstarmer\b",
        r"\bbank of england\b",
    ],
    "China": [
        r"\bchina\b",
        r"\bchinese\b",
        r"\bxi jinping\b",
        r"\bbeijing\b",
        r"\btaiwan\b",
        r"\bhong kong\b",
    ],
    "Russia": [r"\brussia\b", r"\brussian\b", r"\bputin\b", r"\bmoscow\b"],
    "Iran": [r"\biran\b", r"\biranian\b", r"\btehran\b"],
    "Israel": [r"\bisrael\b", r"\bnetanyahu\b", r"\bgaza\b", r"\bhamas\b", r"\bhezbollah\b"],
    "Japan": [r"\bjapan\b", r"\bjapanese\b", r"\byen\b", r"\bboj\b"],
    "India": [r"\bindia\b", r"\bmodi\b"],
    "Korea": [r"\bkorea\b", r"\bnorth korean?\b", r"\bsouth korean?\b", r"\bkim jong un\b"],
    "Mexico": [r"\bmexico\b", r"\bmexican\b"],
    "Canada": [r"\bcanad(a|ian)\b", r"\bcarney\b", r"\btrudeau\b"],
    "Brazil": [r"\bbrazil\b", r"\blula\b"],
    "Argentina": [r"\bargentin(a|e)\b", r"\bmilei\b"],
    "Venezuela": [r"\bvenezuela\b", r"\bmaduro\b"],
    "SaudiArabia": [r"\bsaudi\b", r"\bopec\+?\b"],
    "Turkey": [r"\bturkey\b", r"\bturkish\b", r"\berdogan\b"],
    "Africa": [r"\bafrica\b", r"\bnigeria\b", r"\bsudan\b", r"\bethiopia\b"],
}

# Sport leagues / vehicles
SPORT_KEYWORDS = {
    "NFL": [r"\bnfl\b", r"\bsuper bowl\b", r"\bafc\b", r"\bnfc\b"],
    "NBA": [r"\bnba\b", r"\bnba.finals?\b", r"\blakers\b", r"\bceltics\b"],
    "MLB": [r"\bmlb\b", r"\bworld series\b"],
    "NHL": [r"\bnhl\b", r"\bstanley cup\b"],
    "F1": [r"\bf1\b", r"formula 1", r"formula one", r"\bgrand prix\b"],
    "Soccer": [
        r"\bpremier league\b",
        r"\bla liga\b",
        r"\bchampions league\b",
        r"\bfifa\b",
        r"\bworld cup\b",
        r"\buefa\b",
        r"\bmls\b",
    ],
    "Tennis": [
        r"\btennis\b",
        r"\bwimbledon\b",
        r"\bus open\b",
        r"\baustralian open\b",
        r"\broland.garros\b",
        r"\bfrench open\b",
    ],
    "Golf": [r"\bgolf\b", r"\bmasters\b", r"\bpga\b", r"\bliv golf\b"],
    "Boxing": [r"\bboxing\b", r"\bheavyweight\b"],
    "MMA": [r"\bufc\b", r"\bmma\b"],
    "Esports": [r"\besports\b", r"\bleague of legends\b", r"\bvalorant\b", r"\bdota\b"],
    "Olympics": [r"\bolympics?\b"],
    "Cricket": [r"\bcricket\b", r"\bipl\b"],
    "ChampionshipChess": [r"\bchess\b"],
}

COMMODITY_KEYWORDS = {
    "oil": [r"\boil\b", r"\bcrude\b", r"\bbrent\b", r"\bwti\b"],
    "gas": [r"\bnat.?gas\b", r"\bnatural gas\b", r"\bhenry hub\b"],
    "gold": [r"\bgold\b", r"\bxau\b"],
    "silver": [r"\bsilver\b", r"\bxag\b"],
    "copper": [r"\bcopper\b"],
    "uranium": [r"\buranium\b"],
    "wheat": [r"\bwheat\b"],
    "corn": [r"\bcorn\b"],
    "lithium": [r"\blithium\b"],
}

RATE_KEYWORDS = [
    r"\bfed\b",
    r"\bfomc\b",
    r"\bfederal reserve\b",
    r"\brate cut\b",
    r"\brate hike\b",
    r"\bbasis points\b",
    r"\bbps\b",
    r"\bcpi\b",
    r"\binflation\b",
    r"\bunemployment\b",
    r"\bjobs report\b",
    r"\bnfp\b",
    r"\btreasur(y|ies)\b",
    r"\byield\b",
    r"\bgdp\b",
    r"\bpowell\b",
    r"\bnonfarm\b",
    r"\brecession\b",
]

ELECTION_KEYWORDS = [
    r"\belection\b",
    r"\bwin the\b.*\bnomination\b",
    r"\bgovernor\b",
    r"\bsenator\b",
    r"\bpresident\b",
    r"\bpresidential\b",
    r"\bmidterm\b",
    r"\bprimary\b",
    r"\bnominee\b",
    r"\bcaucus\b",
    r"\bmayor\b",
    r"\bcandidate\b",
]

GEOPOL_KEYWORDS = [
    r"\bwar\b",
    r"\bceasefire\b",
    r"\binvasion\b",
    r"\bmilitary\b",
    r"\bsanctions?\b",
    r"\btariff\b",
    r"\bpeace deal\b",
    r"\bnuclear\b",
    r"\bcoup\b",
    r"\bregime\b",
    r"\btreaty\b",
    r"\bdiploma(t|cy)\b",
    r"\bborder\b",
]

EVENT_TYPE_HINTS = {
    "range": [r"\bbetween\b.*\band\b", r"\brange\b", r"\bclose between\b"],
    "ladder": [
        r"\babove\b",
        r"\bbelow\b",
        r"\bover\b",
        r"\bunder\b",
        r"\bat least\b",
        r"\bmore than\b",
        r"\bfewer than\b",
        r"\bexceed\b",
        r"\bgreater than\b",
        r"\bless than\b",
    ],
    "multi-outcome": [
        r"\bwhich\b.*\bwill\b",
        r"\bwho will\b",
        r"\bwinner\b",
        r"\bwhich team\b",
        r"\bwhich candidate\b",
        r"\bchampion\b",
        r"\bnominee\b",
        r"\bbest picture\b",
        r"\boscars?\b",
    ],
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _hay(f: dict) -> str:
    """Lowercased haystack of all searchable text for one factor."""
    parts = [
        str(f.get("id") or ""),
        str(f.get("name") or ""),
        str(f.get("slug") or ""),
        str(f.get("description") or ""),
        str(f.get("theme") or ""),
    ]
    return " ".join(parts).lower()


def _any(patterns: list[str], hay: str) -> bool:
    return any(re.search(p, hay) for p in patterns)


def detect_geo(hay: str) -> list[str]:
    out = []
    for geo, pats in GEO_KEYWORDS.items():
        if _any(pats, hay):
            out.append(geo)
    return out


def detect_sub_asset(hay: str, asset_class: str) -> str | None:
    if asset_class == "crypto":
        for tk, pats in CRYPTO_TICKERS.items():
            if _any(pats, hay):
                return tk
    if asset_class == "equity":
        # Direct ticker match (uppercase boundary in original text)
        for tk in EQUITY_TICKERS:
            if re.search(rf"\b{tk.lower()}\b", hay):
                return tk
        for tk, pats in COMPANY_NAMES.items():
            if _any(pats, hay):
                return tk
    if asset_class == "rate":
        if re.search(r"\bcpi\b|\binflation\b", hay):
            return "cpi"
        if re.search(r"\bunemploy|nfp\b|jobs report|nonfarm", hay):
            return "jobs"
        if re.search(r"\bgdp\b", hay):
            return "gdp"
        if re.search(r"\bfed|fomc|rate cut|rate hike|basis points|bps|powell", hay):
            return "fed-funds"
        if re.search(r"\btreasur|yield", hay):
            return "ust-yield"
    if asset_class == "commodity":
        for cm, pats in COMMODITY_KEYWORDS.items():
            if _any(pats, hay):
                return cm
    if asset_class == "sport":
        for lg, pats in SPORT_KEYWORDS.items():
            if _any(pats, hay):
                return lg
    if asset_class == "fx":
        for cur in ["yen", "euro", "pound", "yuan", "peso", "ruble", "lira", "real"]:
            if cur in hay:
                return cur
    return None


def detect_asset_class(f: dict, hay: str) -> str:
    theme = (f.get("theme") or "").lower()

    # Strong theme-based mapping first
    if theme == "crypto":
        return "crypto"
    if theme == "equity":
        return "equity"
    if theme == "sports":
        return "sport"
    if theme == "macro":
        # macro is mostly rates / inflation; if it mentions FX explicitly route there
        if re.search(r"\b(usd|eur|jpy|gbp|cny|peso|ruble|forex|fx)\b", hay):
            return "fx"
        return "rate"
    if theme == "commodities":
        return "commodity"
    if theme == "energy":
        if re.search(r"\boil\b|\bcrude\b|\bnatural gas\b|\bopec\b", hay):
            return "commodity"
        return "commodity"
    if theme == "politics":
        if _any(ELECTION_KEYWORDS, hay):
            return "election"
        return "election"
    if theme == "geopolitics":
        return "geopolitical"
    if theme == "chips":
        return "equity"
    if theme == "ai":
        return "equity"

    # Theme didn't decide — content-based fallback
    for pats in CRYPTO_TICKERS.values():
        if _any(pats, hay):
            return "crypto"
    if _any(RATE_KEYWORDS, hay):
        return "rate"
    if _any(ELECTION_KEYWORDS, hay):
        return "election"
    if _any(GEOPOL_KEYWORDS, hay):
        return "geopolitical"
    for pats in SPORT_KEYWORDS.values():
        if _any(pats, hay):
            return "sport"
    for pats in COMMODITY_KEYWORDS.values():
        if _any(pats, hay):
            return "commodity"
    for tk in EQUITY_TICKERS:
        if re.search(rf"\b{tk.lower()}\b", hay):
            return "equity"
    return "other"


def detect_event_type(hay: str) -> str:
    # multi-outcome wins over ladder wins over range wins over binary
    if _any(EVENT_TYPE_HINTS["multi-outcome"], hay):
        return "multi-outcome"
    if _any(EVENT_TYPE_HINTS["range"], hay):
        return "range"
    if _any(EVENT_TYPE_HINTS["ladder"], hay):
        return "ladder"
    return "binary"


# Resolution period detection ------------------------------------------------- #

MONTH_TO_QUARTER = {
    1: "Q1",
    2: "Q1",
    3: "Q1",
    4: "Q2",
    5: "Q2",
    6: "Q2",
    7: "Q3",
    8: "Q3",
    9: "Q3",
    10: "Q4",
    11: "Q4",
    12: "Q4",
}

MONTH_NAMES = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def detect_resolution_period(hay: str) -> str | None:
    # Look for explicit YYYY-MM in slug-like text
    m = re.search(r"\b(20\d{2})-(\d{2})-\d{2}\b", hay)
    if m:
        year, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return f"{year}-{MONTH_TO_QUARTER[mo]}"

    # Month name + year
    for name, mo in MONTH_NAMES.items():
        m2 = re.search(rf"\b{name}\b[^\d]{{0,12}}\b(20\d{{2}})\b", hay)
        if m2:
            year = int(m2.group(1))
            return f"{year}-{MONTH_TO_QUARTER[mo]}"

    # Quarter mention: Q1 2026 / 2026 Q1
    m3 = re.search(r"\b(q[1-4])\s*(20\d{2})\b", hay)
    if m3:
        return f"{m3.group(2)}-{m3.group(1).upper()}"
    m4 = re.search(r"\b(20\d{2})\s*(q[1-4])\b", hay)
    if m4:
        return f"{m4.group(1)}-{m4.group(2).upper()}"

    # H1/H2 mention
    m5 = re.search(r"\b(h[12])\s*(20\d{2})\b", hay)
    if m5:
        return f"{m5.group(2)}-{m5.group(1).upper()}"
    m6 = re.search(r"\b(20\d{2})\s*(h[12])\b", hay)
    if m6:
        return f"{m6.group(1)}-{m6.group(2).upper()}"

    # "by end of YYYY" / "in YYYY"
    m7 = re.search(r"\b(?:end of|in|by|during|for)\s+(20\d{2})\b", hay)
    if m7:
        return m7.group(1)

    # Fallback: first plausible year mention
    m8 = re.search(r"\b(20[2-3]\d)\b", hay)
    if m8:
        return m8.group(1)

    return None


def detect_volume(desc: str) -> float | None:
    """Pull `vol_1mo=NNN,NNN` (or vol_24h) out of a description string."""
    if not desc:
        return None
    m = re.search(r"vol_1mo\s*=\s*([0-9,\.]+)", desc)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None
    m = re.search(r"vol_24h(?:r)?\s*=\s*([0-9,\.]+)", desc)
    if m:
        try:
            # 24h vol ~ 1/30 of monthly; scale up for tiering consistency
            return float(m.group(1).replace(",", "")) * 30
        except ValueError:
            return None
    m = re.search(r"volume\s*[:=]\s*\$?([0-9,\.]+)", desc, re.I)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def liquidity_tier(vol: float | None) -> str:
    if vol is None:
        return "unknown"
    if vol > 100_000:
        return "A"
    if vol > 10_000:
        return "B"
    return "C"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def tag_factor(f: dict) -> dict:
    hay = _hay(f)
    asset_class = detect_asset_class(f, hay)
    sub_asset = detect_sub_asset(hay, asset_class)
    geo = detect_geo(hay)
    event_type = detect_event_type(hay)
    resolution_period = detect_resolution_period(hay)
    vol = detect_volume(f.get("description") or "")
    tier = liquidity_tier(vol)

    full_tags = [
        f"asset:{asset_class}",
        f"event:{event_type}",
        f"liq:{tier}",
        f"src:{f.get('source', 'unknown')}",
        f"theme:{f.get('theme', 'unknown')}",
    ]
    if sub_asset:
        full_tags.append(f"sub:{sub_asset}")
    if resolution_period:
        full_tags.append(f"res:{resolution_period}")
    for g in geo:
        full_tags.append(f"geo:{g}")
    if f.get("resolved"):
        full_tags.append("status:resolved")

    return {
        "asset_class": asset_class,
        "sub_asset": sub_asset,
        "geo": geo,
        "event_type": event_type,
        "resolution_period": resolution_period,
        "liquidity_tier": tier,
        "full_tags": full_tags,
    }


def main() -> None:
    with FACTORS_YML.open() as fh:
        data = yaml.safe_load(fh)
    factors = data["factors"]

    out: dict[str, dict] = {}
    for f in factors:
        fid = f.get("id")
        if not fid:
            continue
        out[fid] = tag_factor(f)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    # ---- report -----------------------------------------------------------
    n = len(out)
    tag_counter: Counter[str] = Counter()
    tier_counter: Counter[str] = Counter()
    for entry in out.values():
        tag_counter.update(entry["full_tags"])
        tier_counter[entry["liquidity_tier"]] += 1

    print(f"N factors tagged: {n}")
    print("Top 10 tags:")
    for tag, c in tag_counter.most_common(10):
        print(f"  {tag:<28} {c}")
    print("Liquidity tier distribution:")
    for tier in ["A", "B", "C", "unknown"]:
        print(f"  {tier}: {tier_counter.get(tier, 0)}")
    sample_id = next(iter(out))
    print(f"Sample entry [{sample_id}]:")
    print(json.dumps(out[sample_id], indent=2))


if __name__ == "__main__":
    main()
