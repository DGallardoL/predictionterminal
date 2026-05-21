"""
bench_latency.py - WebSocket vs REST latency benchmark for Polymarket & Kalshi.

Measures:
  1. Time-to-first-price: how fast each channel delivers the first usable price
  2. Ongoing update latency: WS message inter-arrival vs REST poll round-trip
  3. REST round-trip time per request

Runs for ~30 seconds per platform, then prints a summary table in ms.

Self-contained: loads .env and handles Kalshi auth without importing versionfinpar.
"""

import asyncio
import base64
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import requests
import websockets

# ============================================================================
# .env loader (standalone - no dotenv dependency needed)
# ============================================================================

def load_env_file(path: str = ".env"):
    """Minimal .env parser: KEY=VALUE lines, ignores comments and export prefix."""
    env_path = Path(__file__).parent / path
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)

load_env_file()


# ============================================================================
# Kalshi request signing (standalone)
# ============================================================================

def sign_kalshi_request(timestamp: str, method: str, path: str) -> str | None:
    """RSA-PSS sign a Kalshi API request. Returns base64 signature or None."""
    pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if not pk_path:
        return None
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError:
        print("  [!] cryptography package not installed -- Kalshi WS auth disabled")
        return None
    try:
        with open(pk_path, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)
        msg = f"{timestamp}{method}{path}"
        sig = private_key.sign(
            msg.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")
    except Exception as e:
        print(f"  [!] Kalshi signing error: {e}")
        return None


# ============================================================================
# CONFIG
# ============================================================================
BENCH_DURATION_S = 30        # how long to collect data per platform
REST_POLL_INTERVAL_S = 0.5   # REST poll every 500 ms
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
POLY_CLOB_BASE = "https://clob.polymarket.com"
POLY_GAMMA_BASE = "https://gamma-api.polymarket.com"


# ============================================================================
# HELPERS
# ============================================================================

def ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def summarize(label: str, latencies: list):
    """Print min/median/mean/p95/max for a list of latencies in seconds."""
    if not latencies:
        print(f"  {label}: NO DATA")
        return
    ms_vals = [v * 1000 for v in latencies]
    print(f"  {label}:")
    print(f"    count  = {len(ms_vals)}")
    print(f"    min    = {min(ms_vals):.2f} ms")
    print(f"    median = {statistics.median(ms_vals):.2f} ms")
    print(f"    mean   = {statistics.mean(ms_vals):.2f} ms")
    if len(ms_vals) >= 20:
        p95 = sorted(ms_vals)[int(len(ms_vals) * 0.95)]
        print(f"    p95    = {p95:.2f} ms")
    print(f"    max    = {max(ms_vals):.2f} ms")


# ============================================================================
# STEP 1: Discover live markets
# ============================================================================

def discover_kalshi_market() -> str | None:
    """Find an active, high-volume Kalshi market ticker via public REST."""
    print("[Kalshi] Discovering an active market...")
    try:
        # Fetch open markets sorted by volume (descending) to find actively traded ones
        resp = requests.get(
            f"{KALSHI_BASE}/markets",
            params={"limit": 50, "status": "open"},
            timeout=10,
        )
        resp.raise_for_status()
        markets = resp.json().get("markets", [])
        # Sort by volume descending to pick the most liquid market
        markets.sort(key=lambda m: int(m.get("volume", 0) or 0), reverse=True)
        for m in markets:
            ticker = m.get("ticker")
            vol = m.get("volume", 0)
            if ticker and m.get("status") == "open" and int(vol or 0) > 0:
                print(f"  -> Using Kalshi ticker: {ticker}  (vol={vol}, {m.get('title','')[:50]})")
                return ticker
        if markets:
            ticker = markets[0].get("ticker")
            print(f"  -> Fallback Kalshi ticker: {ticker}")
            return ticker
    except Exception as e:
        print(f"  [!] Kalshi discovery error: {e}")
    return None


def discover_poly_token() -> tuple:
    """Find an active, high-volume Polymarket token_id. Returns (token_id, question)."""
    print("[Polymarket] Discovering an active market...")
    try:
        # Sort by 24h volume descending to pick the most liquid market
        resp = requests.get(
            f"{POLY_GAMMA_BASE}/markets",
            params={
                "active": "true", "closed": "false",
                "limit": 50, "order": "volume24hr", "ascending": "false",
            },
            timeout=10,
        )
        resp.raise_for_status()
        markets = resp.json()
        # Filter for markets with actual best bid/ask (book has liquidity)
        for m in markets:
            token_ids_raw = m.get("clobTokenIds", "[]")
            try:
                token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
            except json.JSONDecodeError:
                continue
            if not token_ids or len(token_ids) == 0:
                continue
            # Prefer markets with a best bid and ask (active book)
            best_bid = m.get("bestBid")
            best_ask = m.get("bestAsk")
            vol = m.get("volume24hr", 0)
            if best_bid and best_ask and float(best_bid) > 0 and float(best_ask) < 1:
                tid = token_ids[0]  # YES token
                question = m.get("question", "?")
                print(f"  -> Using Poly token: {tid[:24]}...  (vol24h={vol}, {question[:50]})")
                return tid, question
        # Fallback: any token
        for m in markets:
            token_ids_raw = m.get("clobTokenIds", "[]")
            try:
                token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
            except json.JSONDecodeError:
                continue
            if token_ids and len(token_ids) > 0:
                tid = token_ids[0]
                question = m.get("question", "?")
                print(f"  -> Fallback Poly token: {tid[:24]}...  ({question[:60]})")
                return tid, question
    except Exception as e:
        print(f"  [!] Polymarket discovery error: {e}")
    return None, None


# ============================================================================
# POLYMARKET BENCHMARK
# ============================================================================

class PolyBench:
    def __init__(self, token_id: str):
        self.token_id = token_id
        self.ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self.rest_url = f"{POLY_CLOB_BASE}/book"

        self.ws_first_price_t = None
        self.rest_first_price_t = None
        self.t0 = None

        self.ws_msg_times = []
        self.rest_latencies = []

        self._stop = threading.Event()

    # -- WebSocket leg --

    async def _ws_listen(self):
        try:
            async with websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                await ws.send(json.dumps({
                    "type": "market",
                    "assets_ids": [self.token_id],
                    "custom_feature_enabled": True,
                }))

                while not self._stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    t_recv = time.perf_counter()
                    try:
                        data = json.loads(raw)
                        items = data if isinstance(data, list) else [data]
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            evt = item.get("event_type", "")
                            if evt in ("book", "price_change", "best_bid_ask"):
                                self.ws_msg_times.append(t_recv)
                                if self.ws_first_price_t is None:
                                    self.ws_first_price_t = t_recv
                    except (json.JSONDecodeError, TypeError):
                        pass
        except Exception as e:
            print(f"  [WS-Poly] Error: {e}")

    def _ws_thread(self):
        if sys.platform == "win32":
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except (DeprecationWarning, AttributeError):
                pass
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ws_listen())

    # -- REST leg --

    def _rest_thread(self):
        session = requests.Session()
        while not self._stop.is_set():
            t_start = time.perf_counter()
            try:
                resp = session.get(self.rest_url, params={"token_id": self.token_id}, timeout=5)
                t_end = time.perf_counter()
                if resp.status_code == 200:
                    data = resp.json()
                    has_price = bool(data.get("bids") or data.get("asks"))
                    rtt = t_end - t_start
                    self.rest_latencies.append(rtt)
                    if has_price and self.rest_first_price_t is None:
                        self.rest_first_price_t = t_end
            except Exception as e:
                print(f"  [REST-Poly] Error: {e}")
            elapsed = time.perf_counter() - t_start
            sleep_for = max(0, REST_POLL_INTERVAL_S - elapsed)
            if sleep_for > 0 and not self._stop.is_set():
                self._stop.wait(sleep_for)

    # -- Run --

    def run(self):
        print_section("Polymarket Benchmark")
        print(f"  Token: {self.token_id[:30]}...")
        print(f"  Duration: {BENCH_DURATION_S}s | REST poll interval: {REST_POLL_INTERVAL_S}s")

        self.t0 = time.perf_counter()

        ws_t = threading.Thread(target=self._ws_thread, daemon=True)
        rest_t = threading.Thread(target=self._rest_thread, daemon=True)

        ws_t.start()
        rest_t.start()

        self._stop.wait(BENCH_DURATION_S)
        self._stop.set()

        ws_t.join(timeout=5)
        rest_t.join(timeout=5)

        self._report()

    def _report(self):
        print(f"\n  --- Results (Polymarket) ---")

        if self.ws_first_price_t is not None:
            print(f"  WS  time-to-first-price: {ms(self.ws_first_price_t - self.t0)} ms")
        else:
            print(f"  WS  time-to-first-price: NO DATA (no WS messages received)")

        if self.rest_first_price_t is not None:
            print(f"  REST time-to-first-price: {ms(self.rest_first_price_t - self.t0)} ms")
        else:
            print(f"  REST time-to-first-price: NO DATA")

        if len(self.ws_msg_times) >= 2:
            intervals = [
                self.ws_msg_times[i] - self.ws_msg_times[i - 1]
                for i in range(1, len(self.ws_msg_times))
            ]
            summarize("WS message inter-arrival", intervals)
        else:
            print(f"  WS messages received: {len(self.ws_msg_times)} (not enough for inter-arrival stats)")

        summarize("REST round-trip time", self.rest_latencies)

        print(f"\n  WS total messages: {len(self.ws_msg_times)}")
        print(f"  REST total polls:  {len(self.rest_latencies)}")


# ============================================================================
# KALSHI BENCHMARK
# ============================================================================

class KalshiBench:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.ws_url = "wss://api.elections.kalshi.com/trade-api/ws/v2"
        self.rest_url = f"{KALSHI_BASE}/markets/{ticker}"

        self.ws_first_price_t = None
        self.rest_first_price_t = None
        self.t0 = None

        self.ws_msg_times = []       # list of (perf_counter, is_our_ticker)
        self.rest_latencies = []

        self._stop = threading.Event()
        self._ws_connected = False

    # -- Auth headers --

    def _make_ws_headers(self):
        key_id = os.getenv("KALSHI_API_KEY_ID")
        if not key_id:
            print("  [WS-Kalshi] KALSHI_API_KEY_ID not set -- WS auth will fail")
            return {}
        timestamp = str(int(time.time() * 1000))
        sig = sign_kalshi_request(timestamp, "GET", "/trade-api/ws/v2")
        if not sig:
            print("  [WS-Kalshi] Could not sign request -- check KALSHI_PRIVATE_KEY_PATH")
            return {}
        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    # -- WebSocket leg --

    async def _ws_listen(self):
        headers = self._make_ws_headers()
        if not headers:
            print("  [WS-Kalshi] Skipping WS benchmark (no credentials)")
            return
        try:
            async with websockets.connect(
                self.ws_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                self._ws_connected = True
                print("  [WS-Kalshi] Connected!")

                # Subscribe to ticker channel (all markets -- firehose)
                await ws.send(json.dumps({
                    "id": 1,
                    "cmd": "subscribe",
                    "params": {"channels": ["ticker"]}
                }))

                while not self._stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    t_recv = time.perf_counter()
                    try:
                        data = json.loads(raw)
                        msg_type = data.get("type")
                        msg = data.get("msg", {})
                        if msg_type == "ticker":
                            mt = msg.get("market_ticker", "")
                            self.ws_msg_times.append((t_recv, mt == self.ticker))
                            if self.ws_first_price_t is None and mt == self.ticker:
                                self.ws_first_price_t = t_recv
                    except (json.JSONDecodeError, TypeError):
                        pass
        except Exception as e:
            print(f"  [WS-Kalshi] Error: {e}")

    def _ws_thread(self):
        if sys.platform == "win32":
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except (DeprecationWarning, AttributeError):
                pass
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ws_listen())

    # -- REST leg --

    def _rest_thread(self):
        session = requests.Session()
        while not self._stop.is_set():
            t_start = time.perf_counter()
            try:
                resp = session.get(self.rest_url, timeout=5)
                t_end = time.perf_counter()
                if resp.status_code == 200:
                    data = resp.json()
                    market = data.get("market", {})
                    has_price = (
                        market.get("yes_ask_dollars") is not None
                        or market.get("yes_bid_dollars") is not None
                    )
                    rtt = t_end - t_start
                    self.rest_latencies.append(rtt)
                    if has_price and self.rest_first_price_t is None:
                        self.rest_first_price_t = t_end
                elif resp.status_code == 429:
                    print(f"  [REST-Kalshi] Rate limited, backing off...")
                    self._stop.wait(2)
            except Exception as e:
                print(f"  [REST-Kalshi] Error: {e}")
            elapsed = time.perf_counter() - t_start
            sleep_for = max(0, REST_POLL_INTERVAL_S - elapsed)
            if sleep_for > 0 and not self._stop.is_set():
                self._stop.wait(sleep_for)

    # -- Run --

    def run(self):
        print_section("Kalshi Benchmark")
        print(f"  Ticker: {self.ticker}")
        print(f"  Duration: {BENCH_DURATION_S}s | REST poll interval: {REST_POLL_INTERVAL_S}s")

        self.t0 = time.perf_counter()

        ws_t = threading.Thread(target=self._ws_thread, daemon=True)
        rest_t = threading.Thread(target=self._rest_thread, daemon=True)

        ws_t.start()
        rest_t.start()

        self._stop.wait(BENCH_DURATION_S)
        self._stop.set()

        ws_t.join(timeout=5)
        rest_t.join(timeout=5)

        self._report()

    def _report(self):
        print(f"\n  --- Results (Kalshi) ---")

        if self.ws_first_price_t is not None:
            print(f"  WS  time-to-first-price (this ticker): {ms(self.ws_first_price_t - self.t0)} ms")
        else:
            reason = "no credentials" if not self._ws_connected else "ticker may not have traded"
            print(f"  WS  time-to-first-price (this ticker): NO DATA ({reason})")

        # Also show time-to-first for any ticker on the firehose
        all_ws = [t for t, _ in self.ws_msg_times]
        if all_ws:
            print(f"  WS  time-to-first-price (any ticker):  {ms(all_ws[0] - self.t0)} ms")

        if self.rest_first_price_t is not None:
            print(f"  REST time-to-first-price: {ms(self.rest_first_price_t - self.t0)} ms")
        else:
            print(f"  REST time-to-first-price: NO DATA")

        our_ws_times = [t for t, is_ours in self.ws_msg_times if is_ours]
        total_ws = len(all_ws)
        our_ws = len(our_ws_times)

        print(f"\n  WS ticker messages (all markets, firehose): {total_ws}")
        print(f"  WS ticker messages (this market only):      {our_ws}")

        if len(all_ws) >= 2:
            intervals = [all_ws[i] - all_ws[i - 1] for i in range(1, len(all_ws))]
            summarize("WS firehose inter-arrival (all tickers)", intervals)

        if len(our_ws_times) >= 2:
            intervals = [our_ws_times[i] - our_ws_times[i - 1] for i in range(1, len(our_ws_times))]
            summarize("WS inter-arrival (this ticker only)", intervals)

        summarize("REST round-trip time", self.rest_latencies)

        print(f"\n  REST total polls: {len(self.rest_latencies)}")


# ============================================================================
# FINAL COMPARISON
# ============================================================================

def print_comparison(poly: PolyBench, kalshi: KalshiBench):
    print_section("COMPARISON SUMMARY")

    print("\n  Time-to-first-price (ms):")
    print(f"  {'Platform':<14} {'WebSocket':>12} {'REST':>12} {'Winner':>10}")
    print(f"  {'-'*50}")

    for name, bench in [("Polymarket", poly), ("Kalshi", kalshi)]:
        ws_ttfp = ms(bench.ws_first_price_t - bench.t0) if bench.ws_first_price_t else None
        rest_ttfp = ms(bench.rest_first_price_t - bench.t0) if bench.rest_first_price_t else None

        ws_str = f"{ws_ttfp:.1f}" if ws_ttfp is not None else "N/A"
        rest_str = f"{rest_ttfp:.1f}" if rest_ttfp is not None else "N/A"

        if ws_ttfp is not None and rest_ttfp is not None:
            winner = "WS" if ws_ttfp < rest_ttfp else "REST"
        else:
            winner = "?"

        print(f"  {name:<14} {ws_str:>12} {rest_str:>12} {winner:>10}")

    print("\n  Ongoing latency (ms, median):")
    print(f"  {'Platform':<14} {'WS inter-arr':>14} {'REST RTT':>12} {'WS faster?':>12}")
    print(f"  {'-'*54}")

    for name, bench in [("Polymarket", poly), ("Kalshi", kalshi)]:
        if name == "Polymarket":
            ws_times = bench.ws_msg_times
            if len(ws_times) >= 2:
                intervals = [ws_times[i] - ws_times[i - 1] for i in range(1, len(ws_times))]
                ws_med = statistics.median(intervals) * 1000
            else:
                ws_med = None
        else:
            all_ws_times = [t for t, _ in bench.ws_msg_times]
            if len(all_ws_times) >= 2:
                intervals = [all_ws_times[i] - all_ws_times[i - 1] for i in range(1, len(all_ws_times))]
                ws_med = statistics.median(intervals) * 1000
            else:
                ws_med = None

        if bench.rest_latencies:
            rest_med = statistics.median(bench.rest_latencies) * 1000
        else:
            rest_med = None

        ws_str = f"{ws_med:.1f}" if ws_med is not None else "N/A"
        rest_str = f"{rest_med:.1f}" if rest_med is not None else "N/A"

        if ws_med is not None and rest_med is not None:
            faster = "YES" if ws_med < rest_med else "NO"
        else:
            faster = "?"

        print(f"  {name:<14} {ws_str:>14} {rest_str:>12} {faster:>12}")

    print(f"\n  Note: WS inter-arrival measures how often NEW data arrives (push).")
    print(f"  REST RTT measures the round-trip time per poll request.")
    print(f"  For arb detection, WS wins when events are active -- it delivers")
    print(f"  updates as they happen vs waiting for the next poll cycle.")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  WebSocket vs REST Latency Benchmark")
    print("  Polymarket & Kalshi")
    print("=" * 60)

    # Show auth status
    key_id = os.getenv("KALSHI_API_KEY_ID")
    pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    print(f"\n  Kalshi auth: API key {'SET' if key_id else 'NOT SET'}, "
          f"PK path {'SET' if pk_path else 'NOT SET'}")

    # Discover markets
    kalshi_ticker = discover_kalshi_market()
    poly_token, poly_question = discover_poly_token()

    if not kalshi_ticker:
        print("\n[!] Could not find a Kalshi market. Skipping Kalshi benchmark.")
    if not poly_token:
        print("\n[!] Could not find a Polymarket token. Skipping Polymarket benchmark.")

    if not kalshi_ticker and not poly_token:
        print("\n[FATAL] No markets found on either platform. Exiting.")
        sys.exit(1)

    # Run benchmarks
    poly_bench = None
    kalshi_bench = None

    if poly_token:
        poly_bench = PolyBench(poly_token)
        poly_bench.run()

    if kalshi_ticker:
        kalshi_bench = KalshiBench(kalshi_ticker)
        kalshi_bench.run()

    # Comparison
    if poly_bench and kalshi_bench:
        print_comparison(poly_bench, kalshi_bench)
    elif poly_bench:
        print_section("Results (Polymarket only)")
        print("  Kalshi benchmark was skipped.")
    elif kalshi_bench:
        print_section("Results (Kalshi only)")
        print("  Polymarket benchmark was skipped.")


if __name__ == "__main__":
    main()
