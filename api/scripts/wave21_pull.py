"""Wave 21 Polymarket factor pull.

Pulls candidates via several angles:
  1. Liquidity-sorted (4 pages × 500)
  2. Recently-launched-with-traction (createdAt>14d, vol24h>50k)
  3. Per-tag deep dives on under-represented themes
  4. Long-tail micro-prob (lastTradePrice asc)
  5. Series mass-pull → child markets

Filters: age ≥30d, dte ∈ [21,800], vol_1mo ≥ 1500, dedupe vs existing slugs.
Targets +100-200 factors. Bias toward under-represented themes.
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
        "rate",
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
        "crypto",
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
    "equity": [
        "earnings",
        "stock",
        "ipo",
        "nasdaq",
        "s&p",
        "spx",
        "nyse",
        "buyback",
        "dividend",
        "merger",
        "acquisition",
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
        "energy",
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
        "war",
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
        "house",
        "governor",
        "president",
        "pardon",
        "impeach",
        "vote",
        "supreme court",
        "scotus",
        "cabinet",
        "nominee",
        "republican",
        "democrat",
        "gop",
        "doge",
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
        "open",
        "boxing",
        "ufc",
        "f1",
        "formula",
        "match",
        "game",
        "playoff",
        "super bowl",
        "world cup",
        "stanley cup",
        "world series",
        "olympics",
        "rugby",
    ],
    "climate": ["climate", "warming", "co2", "carbon", "emissions", "temperature record"],
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
        "atlantic",
    ],
    "space": [
        "spacex",
        "starship",
        "nasa",
        "rocket",
        "satellite",
        "mars",
        "moon",
        "asteroid",
        "spacewalk",
        "artemis",
        "iss",
        "blue origin",
    ],
    "health": [
        "fda",
        "vaccine",
        "drug",
        "pharma",
        "ozempic",
        "weight",
        "cancer",
        "alzheimer",
        "covid",
        "flu",
        "h5n1",
        "bird flu",
        "outbreak",
        "measles",
        "ebola",
        "pandemic",
        "wegovy",
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

UNDER_REPRESENTED = {
    "science",
    "world",
    "health",
    "space",
    "weather",
    "entertainment",
    "business",
    "tech",
    "religion",
    "food",
    "art",
}


def classify_theme(question: str, slug: str, tags: list[str] | None = None) -> str:
    text = f"{question} {slug}".lower()
    if tags:
        text = text + " " + " ".join(t.lower() for t in tags)
    # Order matters; check more specific themes first
    order = [
        "chips",
        "ai",
        "crypto",
        "macro",
        "energy",
        "commodities",
        "geopolitics",
        "space",
        "health",
        "weather",
        "climate",
        "legal",
        "pop_culture",
        "sports",
        "politics",
        "equity",
    ]
    for theme in order:
        for kw in THEME_KEYWORDS[theme]:
            if kw in text:
                return theme
    return "other"


def slug_to_id(slug: str, prefix: str = "p21") -> str:
    """Convert slug to a snake_case factor id."""
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


def collect_polymarket_candidates() -> dict[str, dict]:
    """Pull from multiple gamma angles. Returns slug → market dict."""
    candidates: dict[str, dict] = {}

    with httpx.Client(timeout=30.0) as client:
        # Angle 1: Liquidity-sorted (4 pages × 500)
        print("[1] Liquidity-sorted ...", file=sys.stderr)
        n_before = len(candidates)
        for page in range(4):
            try:
                r = client.get(
                    f"{GAMMA}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": 500,
                        "offset": page * 500,
                        "order": "liquidityNum",
                        "ascending": "false",
                    },
                )
                r.raise_for_status()
                ms = r.json()
                if not ms:
                    break
                for m in ms:
                    sl = m.get("slug")
                    if sl and sl not in candidates:
                        candidates[sl] = m
                time.sleep(0.2)
            except Exception as e:
                print(f"  page {page} err: {e}", file=sys.stderr)
        print(f"  +{len(candidates) - n_before}", file=sys.stderr)

        # Angle 2: Recent-with-traction. Use createdAt filter via volume24hr ordering
        print("[2] Recent-with-traction (volume24hr desc) ...", file=sys.stderr)
        n_before = len(candidates)
        for page in range(3):
            try:
                r = client.get(
                    f"{GAMMA}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": 500,
                        "offset": page * 500,
                        "order": "volume24hr",
                        "ascending": "false",
                    },
                )
                r.raise_for_status()
                ms = r.json()
                if not ms:
                    break
                for m in ms:
                    sl = m.get("slug")
                    if sl and sl not in candidates:
                        candidates[sl] = m
                time.sleep(0.2)
            except Exception as e:
                print(f"  page {page} err: {e}", file=sys.stderr)
        print(f"  +{len(candidates) - n_before}", file=sys.stderr)

        # Angle 3: per-tag deep dives - try tag_slug/tags filter on gamma
        # Try tag id-based pull: /markets?tag={tag}
        print("[3] Per-tag pulls ...", file=sys.stderr)
        n_before = len(candidates)
        for tag in sorted(UNDER_REPRESENTED):
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
                    if not ms or not isinstance(ms, list):
                        continue
                    added = 0
                    for m in ms:
                        sl = m.get("slug")
                        if sl and sl not in candidates:
                            candidates[sl] = m
                            added += 1
                    if added:
                        print(
                            f"  tag={tag} param={next(iter(params.keys()))}: +{added}",
                            file=sys.stderr,
                        )
                        break
                except Exception as e:
                    print(f"  tag={tag} err: {e}", file=sys.stderr)
            time.sleep(0.15)
        print(f"  +{len(candidates) - n_before}", file=sys.stderr)

        # Angle 4: long-tail micro-prob (lastTradePrice asc). Don't pull too many.
        print("[4] Long-tail micro-prob ...", file=sys.stderr)
        n_before = len(candidates)
        for page in range(2):
            try:
                r = client.get(
                    f"{GAMMA}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": 500,
                        "offset": page * 500,
                        "order": "lastTradePrice",
                        "ascending": "true",
                    },
                )
                r.raise_for_status()
                ms = r.json()
                if not ms:
                    break
                for m in ms:
                    sl = m.get("slug")
                    if sl and sl not in candidates:
                        candidates[sl] = m
                time.sleep(0.2)
            except Exception as e:
                print(f"  page {page} err: {e}", file=sys.stderr)
        print(f"  +{len(candidates) - n_before}", file=sys.stderr)

        # Angle 5: series mass-pull
        print("[5] Series mass-pull ...", file=sys.stderr)
        n_before = len(candidates)
        try:
            r = client.get(
                f"{GAMMA}/series",
                params={"active": "true", "limit": 200, "order": "volumeNum", "ascending": "false"},
            )
            if r.status_code == 200:
                series_list = r.json() if isinstance(r.json(), list) else r.json().get("data", [])
                series_ids = []
                for s in series_list[:200]:
                    sid = s.get("id") or s.get("slug")
                    if sid:
                        series_ids.append(sid)
                print(f"  series count: {len(series_ids)}", file=sys.stderr)
                # For each series try to find child markets
                for sid in series_ids[:80]:
                    for params in [
                        {"series_id": sid, "active": "true", "closed": "false", "limit": 100},
                        {"series": sid, "active": "true", "closed": "false", "limit": 100},
                    ]:
                        try:
                            r = client.get(f"{GAMMA}/markets", params=params)
                            if r.status_code != 200:
                                continue
                            ms = r.json()
                            if not ms or not isinstance(ms, list):
                                continue
                            added_any = False
                            for m in ms:
                                sl = m.get("slug")
                                if sl and sl not in candidates:
                                    candidates[sl] = m
                                    added_any = True
                            if added_any:
                                break
                        except Exception:
                            continue
        except Exception as e:
            print(f"  series err: {e}", file=sys.stderr)
        print(f"  +{len(candidates) - n_before}", file=sys.stderr)

    print(f"TOTAL pulled: {len(candidates)}", file=sys.stderr)
    return candidates


def fetch_clob_history(client: httpx.Client, market_token: str) -> list[dict]:
    """Fetch daily price history (fidelity=1440) for a CLOB token."""
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
    target_min: int = 100,
) -> list[dict]:
    """Apply filters and return ordered list of selected market dicts."""
    now = datetime.now(UTC)
    age_min = timedelta(days=30)
    dte_min = timedelta(days=21)
    dte_max = timedelta(days=800)
    vol_1mo_min = 1500.0

    pre_filter: list[tuple[float, dict]] = []  # (priority_score, market)

    # Theme distribution targets (existing): under-rep themes get bonus
    under_rep_themes = {
        "science",
        "world",
        "health",
        "space",
        "weather",
        "business",
        "climate",
        "energy",
        "commodities",
        "pop_culture",
        "chips",
    }

    skipped = {
        "dup": 0,
        "no_slug": 0,
        "age": 0,
        "dte": 0,
        "vol": 0,
        "no_tokens": 0,
        "closed": 0,
        "resolved": 0,
    }

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
            # Try other vol fields
            try:
                vol_total = float(m.get("volumeNum") or m.get("volume") or 0)
            except (TypeError, ValueError):
                vol_total = 0.0
            # Fallback: use total volume / age_days * 30 as proxy
            age_days = max(age.days, 1)
            implied = vol_total / age_days * 30.0
            if implied < vol_1mo_min:
                skipped["vol"] += 1
                continue
            vol_1mo = implied

        # Need clobTokenIds
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

        # Score: vol_1mo * theme_bonus
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
        bonus = 2.0 if theme in under_rep_themes else 1.0
        score = vol_1mo * bonus

        m["_theme"] = theme
        m["_token"] = tokens[0]
        m["_vol_1mo"] = vol_1mo
        m["_age_days"] = age.days
        m["_dte_days"] = dte.days
        pre_filter.append((score, m))

    print(f"After basic filters: {len(pre_filter)}", file=sys.stderr)
    print(f"Skipped breakdown: {skipped}", file=sys.stderr)

    # Sort by score desc
    pre_filter.sort(key=lambda x: -x[0])

    # Now verify history exists with a CLOB call. Cap probe attempts to avoid
    # hammering — probe top ~target_max*2.
    selected: list[dict] = []
    seen_questions: set[str] = set()  # near-dup detection by question text
    probe_limit = target_max * 3

    # Theme caps to bias balance
    max_per_theme: dict[str, int] = {
        "politics": 25,
        "sports": 25,
        "equity": 25,
        "crypto": 20,
        "macro": 15,
        "geopolitics": 25,
        "ai": 15,
        "chips": 12,
        "energy": 15,
        "commodities": 15,
        "climate": 12,
        "weather": 15,
        "space": 18,
        "health": 18,
        "pop_culture": 15,
        "legal": 12,
        "other": 30,
    }
    theme_count: dict[str, int] = {}

    print(f"Probing CLOB for history (top {probe_limit}) ...", file=sys.stderr)
    with httpx.Client(timeout=15.0) as client:
        for i, (_score, m) in enumerate(pre_filter[:probe_limit]):
            if len(selected) >= target_max:
                break
            theme = m["_theme"]
            if theme_count.get(theme, 0) >= max_per_theme.get(theme, 20):
                continue
            q = (m.get("question") or "").lower().strip()
            # Near-dup by first 60 chars of question
            qkey = q[:60]
            if qkey in seen_questions:
                continue

            hist = fetch_clob_history(client, m["_token"])
            if len(hist) < 14:  # need at least ~2 weeks of bars
                continue

            seen_questions.add(qkey)
            theme_count[theme] = theme_count.get(theme, 0) + 1
            m["_bars"] = len(hist)
            selected.append(m)
            if len(selected) % 10 == 0:
                print(f"  selected {len(selected)} / probed {i + 1}", file=sys.stderr)
            time.sleep(0.05)

    print(f"Selected: {len(selected)}", file=sys.stderr)
    print(
        f"Theme distribution: {sorted(theme_count.items(), key=lambda x: -x[1])}", file=sys.stderr
    )
    return selected


def render_yaml_block(selected: list[dict]) -> str:
    lines = ["", "# === WAVE 21 ==="]
    lines.append("# Pulled via liquidity-sorted, recent-traction, per-tag deep dives,")
    lines.append("# long-tail micro-prob, and series mass-pull. See scripts/wave21_pull.py.")
    used_ids: set[str] = set()
    for m in selected:
        slug = m["slug"]
        fid = slug_to_id(slug, prefix="p21")
        # Ensure unique id
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
        # Build short description
        desc = (
            f"{question} "
            f"(theme={theme}, vol_1mo={vol:,}, bars={bars}, "
            f"age={age}d, dte={dte}d) [wave21]"
        )
        # Truncate name to ~120 chars
        name = (m.get("question") or slug)[:140]
        # Escape single quotes for YAML single-quoted style
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

    candidates = collect_polymarket_candidates()
    selected = select_factors(candidates, existing_slugs, target_max=200, target_min=100)

    block = render_yaml_block(selected)
    out_path = Path("/tmp/wave21_block.yml")
    out_path.write_text(block)
    print(f"Wrote {len(selected)} factors to {out_path}", file=sys.stderr)

    # Also dump theme distribution
    theme_counts: dict[str, int] = {}
    for m in selected:
        theme_counts[m["_theme"]] = theme_counts.get(m["_theme"], 0) + 1
    print(f"FINAL theme dist: {sorted(theme_counts.items(), key=lambda x: -x[1])}")
    print(f"FINAL total candidates pulled: {len(candidates)}")
    print(f"FINAL selected: {len(selected)}")


if __name__ == "__main__":
    main()
