"""Build Roll's (1984) effective-spread cache for every Polymarket leg in the
arb engine, and emit a blacklist of "fake arb" pairs where Roll exceeds the
displayed cross-venue gap.

Theory: Roll JF 1984. Under bid-ask bounce, ``Cov(Δp_t, Δp_{t-1}) = -s²/4``
so ``s_Roll = 2·√(-Cov)``. When Cov ≥ 0 the asset is trending and Roll is
undefined (we tag those separately).

Run:    python build_roll_filter.py
Writes: roll_spread_cache.json (per-slug Roll values + blacklist).

The arb_engine.py reads roll_spread_cache.json on each scan and drops
candidates whose `arb_key` is in `blacklist_arb_keys`. This catches the
bid-ask-bounce-illusion arbs that the simple |kalshi − poly| filter doesn't.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ARBSTUFF = HERE.parent
CACHE_PATH = ARBSTUFF / "roll_spread_cache.json"
STATE_PATH = ARBSTUFF / "dashboard_state.json"

UA = "Mozilla/5.0 pfm-roll-research"


def _fetch_json(url: str, timeout: int = 15) -> dict | list | None:
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (HTTPError, OSError, json.JSONDecodeError):
        return None


def _poly_token_ids_for_slug(slug: str) -> list[str]:
    """Return Polymarket clobTokenIds for a slug (list, length 2 typically)."""
    data = _fetch_json(f"https://gamma-api.polymarket.com/markets?slug={slug}")
    if not isinstance(data, list) or not data:
        return []
    raw = data[0].get("clobTokenIds")
    if not raw:
        return []
    try:
        return json.loads(raw) if isinstance(raw, str) else list(raw)
    except json.JSONDecodeError:
        return []


def _fetch_history(token_id: str) -> list[float]:
    """Pull HOURLY midpoint history (last month) for a Polymarket token.

    Wave-6 finding 2026-05-19: the previous ``fidelity=1440&interval=max`` pull
    returned 117-152 days of daily prices. Old regimes (resolved or stale)
    blacklisted markets that are CURRENTLY clean — 2 of 4 black listed pairs
    (SpaceX→MS, Viking Therapeutics) were false positives under that method.
    Switching to ``fidelity=60&interval=1m`` (hourly bars, ~1 month back)
    keeps Beshear/Emanuel-style structurally-bouncy markets flagged while
    un-blacklisting markets whose noisy regime is historical only.
    """
    url = (
        f"https://clob.polymarket.com/prices-history"
        f"?market={token_id}&fidelity=60&interval=1m"
    )
    data = _fetch_json(url)
    if not isinstance(data, dict):
        return []
    return [float(p["p"]) for p in data.get("history", []) if p.get("p") is not None]


def roll_spread(prices: list[float]) -> tuple[float | None, bool, int]:
    """Compute Roll's (1984) implicit spread from a price series.

    Returns:
        (spread, trending, n_diffs). ``spread`` is None when prices are
        trending (cov ≥ 0). ``trending`` is True in that case.
    """
    if len(prices) < 31:
        return None, False, 0
    dp = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    n = len(dp)
    mean_dp = sum(dp) / n
    cov1 = sum((dp[i] - mean_dp) * (dp[i + 1] - mean_dp) for i in range(n - 1)) / (n - 1)
    if cov1 >= 0:
        return None, True, n
    return 2.0 * math.sqrt(-cov1), False, n


def build_cache(verbose: bool = True) -> dict:
    """Walk the live dashboard_state opportunities + compute Roll per leg."""
    if not STATE_PATH.exists():
        print(f"dashboard_state.json missing at {STATE_PATH}", file=sys.stderr)
        sys.exit(1)
    state = json.loads(STATE_PATH.read_text())
    opps = state.get("opportunities", [])

    cache: dict[str, dict] = {}
    blacklist_keys: list[str] = []
    seen_tokens: dict[str, dict] = {}
    rated = trending = too_short = bad = 0

    for op in opps:
        token = op.get("poly_token_id") or ""
        slug = op.get("poly_slug") or ""
        if not token:
            continue
        if token in seen_tokens:
            ev = seen_tokens[token]
        else:
            prices = _fetch_history(token)
            time.sleep(0.05)
            roll_s, trend, n = roll_spread(prices)
            ev = {"slug": slug, "token": token, "n_diffs": n,
                  "roll_spread": roll_s, "trending": trend}
            seen_tokens[token] = ev
            cache[token] = ev
            if roll_s is None:
                if trend: trending += 1
                else: too_short += 1
            else:
                rated += 1

        displayed = float(op.get("spread") or 0.0)
        roll_s = ev["roll_spread"]
        ratio = (roll_s / displayed) if (roll_s and displayed > 0) else None
        op_record = {"displayed": displayed, "ratio": ratio,
                     "name": op.get("name", "")[:80]}
        ev.setdefault("opps", []).append(op_record)
        # Blacklist when Roll > displayed (fake arb — within-venue noise
        # swamps the cross-venue mispricing).
        if roll_s is not None and displayed > 0 and roll_s > displayed:
            bad += 1
            arb_key = op.get("arb_key")
            if arb_key and arb_key not in blacklist_keys:
                blacklist_keys.append(arb_key)
            if verbose:
                print(f"  BLACKLIST {ratio:.2f}x  Roll={roll_s:.3f}  "
                      f"displayed={displayed:.3f}  {op.get('name','')[:70]}")

    summary = {
        "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tokens": len(seen_tokens),
        "rated": rated,
        "trending": trending,
        "too_short": too_short,
        "blacklisted_keys": len(blacklist_keys),
    }

    payload = {
        "summary": summary,
        "per_token": cache,
        "blacklist_arb_keys": blacklist_keys,
    }
    CACHE_PATH.write_text(json.dumps(payload, indent=2))

    if verbose:
        print(f"\nSummary: {summary}")
        print(f"Wrote: {CACHE_PATH}")
    return payload


if __name__ == "__main__":
    build_cache(verbose=True)
