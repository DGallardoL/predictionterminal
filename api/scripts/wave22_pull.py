"""Wave 22 Polymarket factor pull.

Strong tilt toward under-represented themes (science, health, space, weather,
climate, business, art, religion, food).

Pull angles (logged per-source):
  1. Search-based with diverse keywords
  2. By tag (tag_slug=science|health|space|climate)
  3. Featured / curated (gamma /markets?featured=true)
  4. CLOB-volume sorted (volume24hrClob)

Filters: age ≥30d, dte ∈ [21,800], vol_1mo ≥ 1000, dedupe vs existing slugs.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import yaml

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

EXISTING_YAML = Path("/Users/damiangallardoloya/Desktop/proyectofuentes/api/src/pfm/factors.yml")

THEME_KEYWORDS: dict[str, list[str]] = {
    "macro": [
        "fed",
        "cpi",
        "inflation",
        "recession",
        "rate cut",
        "unemployment",
        "gdp",
        "treasury",
        "powell",
        "yield",
        "jobs report",
        "payroll",
        "ppi",
        "pce",
    ],
    "crypto": [
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "solana",
        "sol",
        " crypto",
        "doge",
        "xrp",
        "stablecoin",
        "tether",
        "usdc",
        "altcoin",
        "binance",
        "coinbase",
        "ripple",
        "memecoin",
    ],
    "ai": [
        "openai",
        "chatgpt",
        "gpt",
        "claude",
        "anthropic",
        "gemini",
        "llm",
        "agi",
        "deepmind",
        "grok",
        "deepseek",
        "mistral",
        "llama",
    ],
    "chips": ["nvidia", "tsmc", "amd", "intel", "chip", "semiconductor", "lithography", "asml"],
    "business": [
        "ipo",
        "merger",
        "acquisition",
        "bankrupt",
        "buyout",
        "ceo ",
        " ceo",
        "layoff",
        "patent",
        "antitrust",
        " ftc",
        "elon musk",
        "musk post",
        "tweets in",
    ],
    "equity": [
        "stock",
        "nasdaq",
        "s&p",
        "spx",
        "nyse",
        "buyback",
        "dividend",
        "tesla",
        "apple",
        "microsoft",
        "google",
        "amazon",
        "meta",
        "berkshire",
    ],
    "energy": [
        "oil",
        "gas",
        "wti",
        "brent",
        "opec",
        " energy",
        "saudi",
        "petroleum",
        "diesel",
        "lng",
        "pipeline",
    ],
    "commodities": [
        "gold",
        "silver",
        "copper",
        "wheat",
        "corn",
        "soybean",
        "platinum",
        "uranium",
        "lithium",
        "palladium",
    ],
    "geopolitics": [
        "russia",
        "ukraine",
        "putin",
        "zelensky",
        "china",
        "taiwan",
        "iran",
        "israel",
        "gaza",
        "nato",
        " war ",
        "ceasefire",
        "hostage",
        "north korea",
        "venezuela",
        "syria",
        "lebanon",
        "houthi",
        "hamas",
        "hezbollah",
    ],
    "politics": [
        "trump",
        "biden",
        "harris",
        "vance",
        "election",
        "primary",
        "congress",
        "senate",
        "house ",
        "governor",
        "president",
        "pardon",
        "impeach",
        " vote ",
        "supreme court",
        "scotus",
        "cabinet",
        "nominee",
        "republican",
        "democrat",
        "leader out",
        "prime minister",
        " pm ",
        "macron",
        "abbas",
        "netanyahu",
        "orbán",
        "orban",
        "petro",
        "bolojan",
        "rodriguez",
        "rayner",
        "phillipson",
        "initiative",
        "referendum",
        "parti ",
        "quebec general",
        "british columbia",
        "house seat",
        "win the most seats",
        "parliamentary election",
        "local elections",
        "republican party",
        "democratic party",
        "reform party",
        "rebuilding korea",
        "moderate party",
        "korean local",
    ],
    "sports": [
        "nba",
        "nfl",
        "mlb",
        "nhl",
        "ncaa",
        "fifa",
        "uefa",
        "premier league",
        "champions league",
        "tennis",
        "atp",
        "wta",
        "golf",
        "pga",
        "masters",
        "boxing",
        "ufc",
        "f1",
        "formula",
        "playoff",
        "super bowl",
        "world cup",
        "stanley cup",
        "world series",
        "olympic",
        "rugby",
        "ballon d'or",
        "cy young",
        "mvp",
        "mls cup",
        "al central",
        "nl central",
        "al east",
        "al west",
        "nl east",
        "nl west",
        "french open",
        "us open",
        "australian open",
        "wimbledon",
        "cup final",
        "champions",
        "heisman",
        "memorial trophy",
        "hart trophy",
        "norris trophy",
        "vezina",
        "art ross",
    ],
    "climate": [
        "climate",
        "warming",
        "co2",
        "carbon",
        "emissions",
        "temperature record",
        "global temperature",
    ],
    "weather": [
        "hurricane",
        "storm",
        "tornado",
        "snow",
        "rain",
        "weather",
        "drought",
        "flood",
        "el nino",
        "la nina",
        "atlantic season",
    ],
    "space": [
        "spacex",
        "starship",
        "nasa",
        "rocket",
        "satellite",
        " mars ",
        "moon landing",
        "asteroid",
        "spacewalk",
        "artemis",
        " iss ",
        "blue origin",
        "rocket launch",
        "spacecraft",
        "space station",
        "telescope",
        "space mission",
    ],
    "health": [
        "fda",
        "vaccine",
        " drug ",
        "pharma",
        "ozempic",
        "wegovy",
        "cancer",
        "alzheimer",
        "covid",
        " flu ",
        "h5n1",
        "bird flu",
        "outbreak",
        "measles",
        "ebola",
        "pandemic",
        "clinical trial",
        "approval",
    ],
    "science": [
        "discovery",
        "nobel",
        "experiment",
        "physics",
        "biology",
        "chemistry",
        "research",
        "scientist",
        "laboratory",
        "fusion",
        "quantum",
    ],
    "art": [
        "museum",
        "painting",
        "art gallery",
        "art auction",
        "sculpture",
        "sotheby",
        "christie's",
        "biennale",
        "venice biennale",
    ],
    "religion": [
        "pope",
        "vatican",
        "catholic",
        "christian",
        "muslim",
        "jewish",
        "religion",
        "religious",
        "rabbi",
        "imam",
        "bishop",
        "cardinal",
    ],
    "food": [
        "recipe",
        " food ",
        "restaurant",
        "michelin",
        "chef",
        "burger",
        "mcdonald",
        "starbucks",
        "pepsi",
        "coca cola",
        "wine",
        "beer",
    ],
    "pop_culture": [
        "taylor swift",
        "drake",
        "kanye",
        "kardashian",
        "movie",
        "oscar",
        "grammy",
        "emmy",
        "golden globe",
        "box office",
        "billboard",
        "spotify",
        "tiktok",
        "youtube",
        "netflix",
        "disney",
        "marvel",
        "album",
        "song",
        "playboi carti",
        "euphoria",
        "season ",
        "opening weekend",
        "die in",
        "reality",
    ],
    "legal": [
        "lawsuit",
        "indict",
        "guilty",
        "verdict",
        "trial",
        "court ruling",
        "convict",
        "sentenced",
        "doj",
        "appeal",
    ],
    "other": [],
}

# Order matters; check more specific themes first
THEME_ORDER = [
    "chips",
    "ai",
    "crypto",
    "macro",
    "energy",
    "commodities",
    "geopolitics",
    "space",
    "health",
    "science",
    "weather",
    "climate",
    "religion",
    "food",
    "sports",
    "politics",
    "art",
    "business",
    "legal",
    "pop_culture",
    "equity",
]

UNDER_REPRESENTED = {
    "science",
    "health",
    "space",
    "weather",
    "climate",
    "business",
    "art",
    "religion",
    "food",
}

# Search keywords (Angle 1)
SEARCH_KEYWORDS = [
    # Original under-rep targeted
    "discovery",
    "vaccine",
    "fda",
    "clinical",
    "earthquake",
    "hurricane",
    "nasa",
    "spacex launch",
    "moon landing",
    "mars",
    "olympic",
    "nobel",
    "patent",
    "merger",
    "acquisition",
    "ipo",
    "bankrupt",
    "drug approval",
    "cancer",
    "alzheimer",
    "asteroid",
    "starship",
    "climate",
    "temperature",
    "drought",
    "flood",
    "wildfire",
    "tornado",
    "snowfall",
    "pope",
    "vatican",
    "michelin",
    "restaurant",
    "auction",
    "museum",
    "gallery",
    "buyout",
    "layoff",
    "earnings",
    # Additional broad pulls to increase candidate pool
    "release",
    "reach",
    "win",
    "before 2027",
    "by december",
    "approval",
    "complete",
    "miss",
    "exceed",
    "above",
    "below",
    "hit ",
    "passed",
    "elected",
    "defeat",
    "next",
    "between",
    "quarter",
    "annual",
    "2026",
    "2027",
    "league",
    "championship",
    # More under-rep
    "spacecraft",
    "space station",
    "telescope",
    "mission",
    "ai model",
    "best model",
    "release date",
    "global temperature",
    "co2",
    "warming",
    "economy",
    "recession risk",
    "growth",
    "summit",
    "meeting",
    "treaty",
    "agreement",
]

# Tags to try on tag_slug endpoint (Angle 2)
TAG_SLUGS = [
    "science",
    "health",
    "space",
    "climate",
    "business",
    "weather",
    "tech",
    "world",
    "entertainment",
]


def classify_theme(question: str, slug: str, tags: list[str] | None = None) -> str:
    text = f"{question} {slug}".lower()
    if tags:
        text = text + " " + " ".join(t.lower() for t in tags)
    for theme in THEME_ORDER:
        for kw in THEME_KEYWORDS[theme]:
            if kw in text:
                return theme
    return "other"


def slug_to_id(slug: str, prefix: str = "p22") -> str:
    s = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    s = re.sub(r"_+", "_", s)
    if len(s) > 60:
        s = s[:60].rstrip("_")
    return f"{prefix}_{s}"


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def collect_candidates() -> tuple[dict[str, dict], dict[str, int]]:
    """Pull from multiple gamma angles. Returns (slug→market, source_counts)."""
    candidates: dict[str, dict] = {}
    source_counts: dict[str, int] = {}

    with httpx.Client(timeout=30.0) as client:
        # Angle 1: Search-based with diverse keywords
        print("[1] Search-based keyword pulls ...", file=sys.stderr)
        n_before = len(candidates)
        for kw in SEARCH_KEYWORDS:
            try:
                r = client.get(
                    f"{GAMMA}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": 100,
                        "order": "volumeNum",
                        "ascending": "false",
                        "q": kw,
                    },
                )
                if r.status_code != 200:
                    continue
                ms = r.json()
                if not isinstance(ms, list):
                    continue
                added = 0
                for m in ms:
                    sl = m.get("slug")
                    if sl and sl not in candidates:
                        candidates[sl] = m
                        added += 1
                if added:
                    print(f"  q='{kw}': +{added}", file=sys.stderr)
                time.sleep(0.1)
            except Exception as e:
                print(f"  q='{kw}' err: {e}", file=sys.stderr)
        source_counts["search_keyword"] = len(candidates) - n_before
        print(f"  TOTAL +{source_counts['search_keyword']}", file=sys.stderr)

        # Angle 2: By tag_slug
        print("[2] By tag_slug ...", file=sys.stderr)
        n_before = len(candidates)
        for tag in TAG_SLUGS:
            for params in [
                {
                    "tag_slug": tag,
                    "active": "true",
                    "closed": "false",
                    "limit": 200,
                    "order": "volumeNum",
                    "ascending": "false",
                },
                {
                    "tag": tag,
                    "active": "true",
                    "closed": "false",
                    "limit": 200,
                    "order": "volumeNum",
                    "ascending": "false",
                },
            ]:
                try:
                    r = client.get(f"{GAMMA}/markets", params=params)
                    if r.status_code != 200:
                        continue
                    ms = r.json()
                    if not isinstance(ms, list) or not ms:
                        continue
                    added = 0
                    for m in ms:
                        sl = m.get("slug")
                        if sl and sl not in candidates:
                            candidates[sl] = m
                            added += 1
                    if added:
                        print(
                            f"  tag={tag} via {next(iter(params.keys()))}: +{added}",
                            file=sys.stderr,
                        )
                        break
                except Exception as e:
                    print(f"  tag={tag} err: {e}", file=sys.stderr)
            time.sleep(0.1)
        source_counts["tag_slug"] = len(candidates) - n_before
        print(f"  TOTAL +{source_counts['tag_slug']}", file=sys.stderr)

        # Angle 3: Featured / curated
        print("[3] Featured / curated ...", file=sys.stderr)
        n_before = len(candidates)
        try:
            r = client.get(
                f"{GAMMA}/markets",
                params={
                    "featured": "true",
                    "active": "true",
                    "closed": "false",
                    "limit": 200,
                },
            )
            if r.status_code == 200:
                ms = r.json()
                if isinstance(ms, list):
                    for m in ms:
                        sl = m.get("slug")
                        if sl and sl not in candidates:
                            candidates[sl] = m
        except Exception as e:
            print(f"  featured err: {e}", file=sys.stderr)
        source_counts["featured"] = len(candidates) - n_before
        print(f"  +{source_counts['featured']}", file=sys.stderr)

        # Angle 4: CLOB-volume sorted (volume24hrClob)
        print("[4] CLOB-volume sorted (volume24hrClob) ...", file=sys.stderr)
        n_before = len(candidates)
        for offset in [0, 500, 1000, 1500, 2000, 2500]:
            try:
                r = client.get(
                    f"{GAMMA}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": 500,
                        "offset": offset,
                        "order": "volume24hrClob",
                        "ascending": "false",
                    },
                )
                if r.status_code != 200:
                    continue
                ms = r.json()
                if not isinstance(ms, list) or not ms:
                    break
                for m in ms:
                    sl = m.get("slug")
                    if sl and sl not in candidates:
                        candidates[sl] = m
                time.sleep(0.2)
            except Exception as e:
                print(f"  offset={offset} err: {e}", file=sys.stderr)
        source_counts["clob_volume"] = len(candidates) - n_before
        print(f"  TOTAL +{source_counts['clob_volume']}", file=sys.stderr)

    print(f"TOTAL pulled: {len(candidates)}", file=sys.stderr)
    print(f"Source counts: {source_counts}", file=sys.stderr)
    return candidates, source_counts


def fetch_clob_history(client: httpx.Client, market_token: str) -> list[dict]:
    try:
        r = client.get(
            f"{CLOB}/prices-history",
            params={"market": market_token, "fidelity": 1440, "interval": "all"},
        )
        if r.status_code != 200:
            return []
        return r.json().get("history", [])
    except Exception:
        return []


def select_factors(
    candidates: dict[str, dict],
    existing_slugs: set[str],
    target_max: int = 200,
) -> list[dict]:
    now = datetime.now(UTC)
    age_min = timedelta(days=30)
    dte_min = timedelta(days=21)
    dte_max = timedelta(days=800)
    vol_1mo_min = 1000.0  # lowered from 1500

    pre_filter: list[tuple[float, dict]] = []

    skipped = {"dup": 0, "age": 0, "dte": 0, "vol": 0, "no_tokens": 0, "closed": 0}

    for slug, m in candidates.items():
        if slug in existing_slugs:
            skipped["dup"] += 1
            continue
        if m.get("closed") or m.get("resolved"):
            skipped["closed"] += 1
            continue

        start_dt = parse_iso(m.get("startDate") or m.get("createdAt"))
        end_dt = parse_iso(m.get("endDate"))
        if not start_dt or not end_dt:
            skipped["age"] += 1
            continue
        age = now - start_dt
        if age < age_min:
            skipped["age"] += 1
            continue
        dte = end_dt - now
        if dte < dte_min or dte > dte_max:
            skipped["dte"] += 1
            continue
        try:
            vol_1mo = float(m.get("volume1mo") or m.get("volume1moClob") or 0)
        except (TypeError, ValueError):
            vol_1mo = 0.0
        if vol_1mo < vol_1mo_min:
            try:
                vol_total = float(m.get("volumeNum") or m.get("volume") or 0)
            except (TypeError, ValueError):
                vol_total = 0.0
            age_days = max(age.days, 1)
            implied = vol_total / age_days * 30.0
            if implied < vol_1mo_min:
                skipped["vol"] += 1
                continue
            vol_1mo = implied

        ct_raw = m.get("clobTokenIds")
        if not ct_raw:
            skipped["no_tokens"] += 1
            continue
        try:
            tokens = json.loads(ct_raw) if isinstance(ct_raw, str) else ct_raw
        except Exception:
            skipped["no_tokens"] += 1
            continue
        if not tokens:
            skipped["no_tokens"] += 1
            continue

        question = m.get("question") or ""
        tags = m.get("tags") or []
        tag_names = []
        if isinstance(tags, list):
            for t in tags:
                if isinstance(t, dict):
                    tag_names.append(t.get("label") or t.get("slug") or "")
                else:
                    tag_names.append(str(t))
        theme = classify_theme(question, slug, tag_names)
        # Heavy bonus for under-rep themes (3x)
        bonus = 3.0 if theme in UNDER_REPRESENTED else 1.0
        score = vol_1mo * bonus

        m["_theme"] = theme
        m["_token"] = tokens[0]
        m["_vol_1mo"] = vol_1mo
        m["_age_days"] = age.days
        m["_dte_days"] = dte.days
        pre_filter.append((score, m))

    print(f"After basic filters: {len(pre_filter)}", file=sys.stderr)
    print(f"Skipped breakdown: {skipped}", file=sys.stderr)

    pre_filter.sort(key=lambda x: -x[0])

    selected: list[dict] = []
    seen_questions: set[str] = set()
    probe_limit = target_max * 4

    # Theme caps biased toward under-rep
    max_per_theme: dict[str, int] = {
        # Cap heavily-represented themes
        "politics": 15,
        "sports": 18,
        "equity": 12,
        "crypto": 12,
        "macro": 10,
        "geopolitics": 20,
        "ai": 12,
        "chips": 8,
        "energy": 12,
        "commodities": 10,
        "pop_culture": 15,
        "legal": 8,
        # Allow more under-rep
        "climate": 30,
        "weather": 30,
        "space": 35,
        "health": 35,
        "science": 30,
        "business": 30,
        "art": 25,
        "religion": 20,
        "food": 25,
        "other": 25,
    }
    theme_count: dict[str, int] = {}

    print(f"Probing CLOB for history (top {probe_limit}) ...", file=sys.stderr)
    with httpx.Client(timeout=15.0) as client:
        for i, (_score, m) in enumerate(pre_filter[:probe_limit]):
            if len(selected) >= target_max:
                break
            theme = m["_theme"]
            if theme_count.get(theme, 0) >= max_per_theme.get(theme, 15):
                continue
            q = (m.get("question") or "").lower().strip()
            qkey = q[:60]
            if qkey in seen_questions:
                continue

            hist = fetch_clob_history(client, m["_token"])
            if len(hist) < 14:
                continue

            seen_questions.add(qkey)
            theme_count[theme] = theme_count.get(theme, 0) + 1
            m["_bars"] = len(hist)
            selected.append(m)
            if len(selected) % 20 == 0:
                print(f"  selected {len(selected)} / probed {i + 1}", file=sys.stderr)
            time.sleep(0.05)

    print(f"Selected: {len(selected)}", file=sys.stderr)
    print(
        f"Theme distribution: {sorted(theme_count.items(), key=lambda x: -x[1])}", file=sys.stderr
    )
    return selected


def render_yaml_block(selected: list[dict]) -> str:
    lines = ["", "# === WAVE 22 ==="]
    lines.append("# Pulled via search-keywords, tag_slug, featured, and volume24hrClob.")
    lines.append("# Strong tilt to under-rep themes (science, health, space, weather,")
    lines.append("# climate, business, art, religion, food). See scripts/wave22_pull.py.")
    used_ids: set[str] = set()
    for m in selected:
        slug = m["slug"]
        fid = slug_to_id(slug, prefix="p22")
        base = fid
        n = 2
        while fid in used_ids:
            fid = f"{base}_{n}"
            n += 1
        used_ids.add(fid)
        question = (m.get("question") or "").replace("'", "''")
        theme = m["_theme"]
        vol = int(m["_vol_1mo"])
        bars = m.get("_bars", 0)
        dte = m["_dte_days"]
        age = m["_age_days"]
        desc = (
            f"{question} "
            f"(theme={theme}, vol_1mo={vol:,}, bars={bars}, "
            f"age={age}d, dte={dte}d) [wave22]"
        )
        name = (m.get("question") or slug)[:140]
        name_esc = name.replace("'", "''")
        desc_esc = desc.replace("'", "''")

        lines.append(f"- id: {fid}")
        lines.append(f"  name: '{name_esc}'")
        lines.append(f"  slug: {slug}")
        lines.append("  source: polymarket")
        lines.append(f"  theme: {theme}")
        lines.append(f"  description: '{desc_esc}'")
    return "\n".join(lines) + "\n"


def main() -> None:
    print("Loading existing factors ...", file=sys.stderr)
    with EXISTING_YAML.open() as fh:
        existing = yaml.safe_load(fh) or {}
    existing_slugs = {
        f["slug"] for f in existing.get("factors", []) if f.get("source") == "polymarket"
    }
    print(f"Existing polymarket slugs: {len(existing_slugs)}", file=sys.stderr)

    candidates, source_counts = collect_candidates()
    selected = select_factors(candidates, existing_slugs, target_max=200)

    block = render_yaml_block(selected)
    out_path = Path("/tmp/wave22_block.yml")
    out_path.write_text(block)
    print(f"Wrote {len(selected)} factors to {out_path}", file=sys.stderr)

    theme_counts: dict[str, int] = {}
    for m in selected:
        theme_counts[m["_theme"]] = theme_counts.get(m["_theme"], 0) + 1

    # Top 10 by vol_1mo
    top10 = sorted(selected, key=lambda x: -x["_vol_1mo"])[:10]

    summary = {
        "source_counts": source_counts,
        "total_pulled": len(candidates),
        "selected": len(selected),
        "theme_counts": dict(sorted(theme_counts.items(), key=lambda x: -x[1])),
        "top10_by_vol_1mo": [
            {
                "slug": m["slug"],
                "question": m.get("question"),
                "theme": m["_theme"],
                "vol_1mo": int(m["_vol_1mo"]),
            }
            for m in top10
        ],
    }
    Path("/tmp/wave22_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
