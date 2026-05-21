import requests
import json

# --- Configuration ---
# You can change these or input them when running
DEFAULT_KALSHI_TICKER = "kxbeninparliament-26jan11"
DEFAULT_POLY_SLUG = "benin-parliamentary-election-winner"

class KalshiClient:
    def __init__(self):
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
    
    def get_event_markets(self, event_ticker):
        url = f"{self.base_url}/events/{event_ticker}"
        params = {"with_nested_markets": "true"}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            event = data.get('event', {})
            markets = []
            for m in event.get('markets', []):
                markets.append({
                    "ticker": m.get('ticker'),
                    "subtitle": m.get('subtitle'),
                    "title": m.get('title')
                })
            return markets
        except Exception as e:
            print(f"Error fetching Kalshi: {e}")
            return []

class PolymarketClient:
    def __init__(self):
        self.gamma_url = "https://gamma-api.polymarket.com/events"

    def get_event_markets(self, slug):
        params = {"slug": slug}
        try:
            # 1. Try fetching as Event
            response = requests.get(self.gamma_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # 2. If empty, try fetching as Market (to find parent event)
            if not data:
                market_url = "https://gamma-api.polymarket.com/markets"
                market_res = requests.get(market_url, params={"slug": slug})
                if market_res.status_code == 200:
                    market_data = market_res.json()
                    if market_data and isinstance(market_data, list) and len(market_data) > 0:
                        # Found the market! Extract parent event slug
                        parent_event = market_data[0].get('events', [{}])[0]
                        event_slug = parent_event.get('slug')
                        if event_slug:
                            print(f"  [Info] Input appears to be a Market Slug. Auto-switching to Event Slug: {event_slug}")
                            return self.get_event_markets(event_slug)
            
            if not data:
                print(f"  [Warning] 0 markets found for slug: {slug}")
                print(f"  (Tip: Ensure you are using the EVENT slug, not a specific MARKET slug.)")
                return []
                
            event = data[0]
            markets = []
            for m in event.get('markets', []):
                name = m.get('groupItemTitle', m.get('question'))
                markets.append({
                    "name": name,
                    "condition_id": m.get('conditionId')
                })
            return markets
        except Exception as e:
            print(f"Error fetching Polymarket: {e}")
            return []

def main():
    import sys
    print("--- Market Match Helper ---")
    
    if len(sys.argv) > 1:
        k_ticker = sys.argv[1].upper()
        p_slug = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_POLY_SLUG
    else:
        k_ticker = input(f"Enter Kalshi Event Ticker (default: {DEFAULT_KALSHI_TICKER}): ").strip().upper() or DEFAULT_KALSHI_TICKER
        p_slug = input(f"Enter Polymarket Slug (default: {DEFAULT_POLY_SLUG}): ").strip() or DEFAULT_POLY_SLUG
    
    print(f"\nFetching data for:\nKalshi: {k_ticker}\nPoly: {p_slug}\n")
    
    kalshi = KalshiClient()
    poly = PolymarketClient()
    
    k_markets = kalshi.get_event_markets(k_ticker)
    p_markets = poly.get_event_markets(p_slug)
    
    print(f"{'='*30}")
    print(f"KALSHI MARKETS ({len(k_markets)})")
    print(f"{'='*30}")
    print(f"{'SUFFIX':<10} | {'FULL TICKER':<30} | {'TITLE'}")
    print("-" * 60)
    
    for m in k_markets:
        ticker = m['ticker']
        suffix = ticker.split('-')[-1].lower()
        print(f"{suffix:<10} | {ticker:<30} | {m['subtitle']}")

    print(f"\n{'='*30}")
    print(f"POLYMARKET MARKETS ({len(p_markets)})")
    print(f"{'='*30}")
    print(f"{'NAME'}")
    print("-" * 30)
    
    for m in p_markets:
        print(f"{m['name']}")
        
    print(f"\n{'='*30}")
    print("SUGGESTED MAPPING ENTRIES")
    print("Use these to update the 'mapping' dict in predictionarbitrage.py")
    print(f"{'='*30}")
    
    # Simple suggestion logic
    for m in k_markets:
        suffix = m['ticker'].split('-')[-1].lower()
        print(f'"{suffix}": "???",  # Match with one of the Polymarket names above')

    print(f"\n{'='*30}")
    print("EXCEL COPY-PASTE LISTS")
    print(f"{'='*30}")
    
    print("--- Kalshi Suffixes (Copy this column) ---")
    for m in k_markets:
        print(m['ticker'].split('-')[-1].lower())
        
    print("\n--- Polymarket Names (Copy this column) ---")
    for m in p_markets:
        print(m['name'])

if __name__ == "__main__":
    main()
