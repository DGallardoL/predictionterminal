"""Live probe against Polymarket. NOT used in CI — manual exploration only."""

from __future__ import annotations

import sys

import httpx

sys.path.insert(0, "src")
from pfm.sources.polymarket import PolymarketClient, fetch_factor_history


def find_high_volume_markets(min_volume: float = 100_000.0, limit: int = 200) -> list[dict]:
    with httpx.Client(timeout=20) as c:
        r = c.get(
            "https://gamma-api.polymarket.com/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": limit,
                "order": "volumeNum",
                "ascending": "false",
            },
        )
        r.raise_for_status()
        markets = r.json()

    out = []
    for m in markets:
        try:
            v = float(m.get("volume") or 0)
        except (TypeError, ValueError):
            v = 0.0
        if v >= min_volume:
            out.append(
                {
                    "volume": v,
                    "slug": m.get("slug", ""),
                    "question": (m.get("question") or "?")[:80],
                    "end": (m.get("endDate") or "")[:10],
                    "active": m.get("active"),
                    "closed": m.get("closed"),
                }
            )
    out.sort(key=lambda r: -r["volume"])
    return out


def main() -> None:
    print("=" * 80)
    print("STEP 1 — discover high-volume active markets")
    print("=" * 80)
    big = find_high_volume_markets(min_volume=200_000)
    print(f"Found {len(big)} markets with volume >= $200k")
    for m in big[:12]:
        print(f"  ${m['volume']:>14,.0f}  end={m['end']}  {m['slug'][:55]}")
        print(f'                 "{m["question"]}"')

    if not big:
        print("\nNo high-volume markets — try lower threshold or check API")
        return

    print()
    print("=" * 80)
    print("STEP 2 — pull metadata + price history for top 3 via OUR client code")
    print("=" * 80)
    with PolymarketClient(
        gamma_url="https://gamma-api.polymarket.com",
        clob_url="https://clob.polymarket.com",
    ) as poly:
        for m in big[:3]:
            slug = m["slug"]
            print(f"\n--- {slug} ---")
            print(f'    "{m["question"]}"  (volume ${m["volume"]:,.0f})')
            try:
                meta = poly.get_market_metadata(slug)
                print(f"    yes_token_id = {meta.yes_token_id[:30]}…")
                print(f"    no_token_id  = {meta.no_token_id[:30]}…")
            except Exception as e:
                print(f"    META ERROR: {e}")
                continue

            try:
                df = fetch_factor_history(poly, slug)
                if df.empty:
                    print("    HISTORY EMPTY — market may have no daily prints yet")
                else:
                    print(
                        f"    history: {len(df)} daily bars from {df.index.min().date()} to {df.index.max().date()}"
                    )
                    print(f"    first 5: {df['price'].head(5).round(3).tolist()}")
                    print(f"    last  5: {df['price'].tail(5).round(3).tolist()}")
                    print(f"    range:   min={df['price'].min():.3f}  max={df['price'].max():.3f}")
                    # quick Δlogit smell test
                    from pfm.model import delta_logit

                    dl = delta_logit(df["price"]).dropna()
                    print(
                        f"    Δlogit:  n={len(dl)}  mean={dl.mean():+.4f}  std={dl.std():.4f}  "
                        f"min={dl.min():+.3f}  max={dl.max():+.3f}"
                    )
            except Exception as e:
                print(f"    HISTORY ERROR: {e!r}")


if __name__ == "__main__":
    main()
