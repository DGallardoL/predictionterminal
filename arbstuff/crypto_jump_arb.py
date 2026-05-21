"""
crypto_jump_arb.py — Latency arbitrage on Polymarket "Up or Down" 5-min markets.

Strategy:
  1. Stream BTC/ETH/SOL/XRP/DOGE spot prices from Binance WS (ms latency)
  2. Detect price jumps (>$X in <Y ms)
  3. Check if Polymarket "Up or Down" markets for the same asset
     haven't priced in the jump yet (orderbook still shows old probability)
  4. Execute on Poly before the gap closes (~1-2 second window)

Key insight: these markets resolve on Chainlink, but Chainlink follows real BTC
prices. Binance WS leads Chainlink by ~1-3 seconds, and Poly's UI/CLOB leads
Chainlink by even less. If we detect a jump faster than Poly's orderbook adjusts,
we capture the spread.

Usage:
    python crypto_jump_arb.py                    # dry-run (no orders)
    python crypto_jump_arb.py --live             # REAL orders
    python crypto_jump_arb.py --max-trade 5      # max $5 per trade
    python crypto_jump_arb.py --jump-threshold 25  # $25 jump in 1s
    python crypto_jump_arb.py --assets BTC,ETH   # only these assets

Log: crypto_jump_log.jsonl (every signal + outcome)
"""

import asyncio
import json
import logging
import os
import sys
import time
import threading
import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Optional

import requests
from dotenv import load_dotenv

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY
    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False

# Reuse the battle-tested Poly WebSocket client from arb_engine
try:
    from arb_engine import PolyWSClient
    HAS_POLY_WS = True
except ImportError:
    HAS_POLY_WS = False

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────
log = logging.getLogger("jump")
log.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
                                   datefmt="%H:%M:%S"))
log.addHandler(_h)

LOG_FILE = "crypto_jump_log.jsonl"

# ── Config ───────────────────────────────────────────────────────
ASSET_MAP = {
    "BTC": "btcusdt",
    "ETH": "ethusdt",
    "SOL": "solusdt",
    "XRP": "xrpusdt",
    "DOGE": "dogeusdt",
}
SLUG_PREFIX = {  # how Poly names these markets
    "BTC": "btc-updown-5m",
    "ETH": "eth-updown-5m",
    "SOL": "sol-updown-5m",
    "XRP": "xrp-updown-5m",
    "DOGE": "doge-updown-5m",
}

POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"
BINANCE_WS = "wss://stream.binance.com:9443/stream"


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class PricePoint:
    ts: float       # epoch seconds
    price: float


@dataclass
class ShortMarket:
    """A single Polymarket 'Up or Down 5min' market."""
    asset: str           # "BTC"
    event_slug: str      # "btc-updown-5m-1776663300"
    target_price: float  # price to beat
    end_ts: float        # epoch seconds when market closes
    up_token_id: str
    down_token_id: str
    up_ask: float = 0.0  # cached best ask for UP
    down_ask: float = 0.0
    last_book_update: float = 0.0


@dataclass
class TradeRecord:
    ts: str
    asset: str
    side: str            # "up" | "down"
    target: float
    binance_price: float
    delta: float
    seconds_left: float
    poly_ask: float
    edge: float
    volume: int
    trade_value: float
    test_mode: bool
    order_id: str = ""
    error: str = ""


# ============================================================================
# BINANCE WS CLIENT
# ============================================================================

class BinanceWSClient:
    """Streams trade prices for the configured assets."""

    def __init__(self, assets: list[str]):
        self.symbols = [ASSET_MAP[a] for a in assets if a in ASSET_MAP]
        self.assets = [a for a in assets if a in ASSET_MAP]
        self._prices: dict[str, deque] = {a: deque(maxlen=200) for a in self.assets}  # last 200 ticks
        self._latest: dict[str, float] = {}
        self._lock = threading.RLock()
        self._running = False
        self._thread = None
        self._loop = None
        self._connected = threading.Event()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="BinanceWS")
        self._thread.start()
        if not self._connected.wait(timeout=10):
            log.warning("[Binance] Connection timeout")

    def stop(self):
        self._running = False

    def _run(self):
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self):
        backoff = 1
        while self._running:
            try:
                await self._connect()
                backoff = 1
            except Exception as e:
                self._connected.clear()
                log.warning("[Binance] Disconnected: %s. Retry %ds", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _connect(self):
        # Subscribe to trade stream (every individual trade — fastest)
        streams = "/".join(f"{s}@trade" for s in self.symbols)
        url = f"{BINANCE_WS}?streams={streams}"
        async with websockets.connect(url, ping_interval=20) as ws:
            self._connected.set()
            log.info("[Binance] Connected: %s", self.assets)
            async for msg in ws:
                self._handle(json.loads(msg))

    def _handle(self, msg):
        data = msg.get("data", {})
        sym = data.get("s", "").lower()
        price = data.get("p")
        ts = data.get("T")  # trade time in ms
        if not sym or not price or not ts:
            return
        # Reverse map
        for asset, s in ASSET_MAP.items():
            if s == sym:
                px = float(price)
                ts_sec = ts / 1000.0
                with self._lock:
                    self._prices[asset].append(PricePoint(ts_sec, px))
                    self._latest[asset] = px
                return

    def get_latest(self, asset: str) -> Optional[float]:
        with self._lock:
            return self._latest.get(asset)

    def get_history(self, asset: str) -> list[PricePoint]:
        """Return copy of last N ticks."""
        with self._lock:
            return list(self._prices.get(asset, []))

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()


# ============================================================================
# POLY SHORT-MARKET DISCOVERY
# ============================================================================

class ShortMarketWatcher:
    """Discovers active Up-or-Down markets for the configured assets."""

    def __init__(self, assets: list[str]):
        self.assets = assets
        self.markets: dict[str, ShortMarket] = {}  # event_slug -> ShortMarket
        self.session = requests.Session()

    def discover(self) -> int:
        """Find currently-active markets via Gamma. Returns count added.

        Strategy: fetch active events ordered by endDate ascending, then filter by
        slug prefix. Paginates to catch markets up to ~6 hours out.
        """
        added = 0
        seen_slugs = set()
        try:
            # Fetch one big batch (up to 1000 active events)
            for offset in (0, 500):
                resp = self.session.get(
                    f"{POLY_GAMMA}/events",
                    params={
                        "active": "true", "closed": "false",
                        "limit": 500, "offset": offset,
                        "order": "endDate", "ascending": "true",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                events = resp.json()
                if not events:
                    break
                for ev in events:
                    slug = ev.get("slug", "")
                    if slug in seen_slugs:
                        continue
                    seen_slugs.add(slug)
                    # Match any of our asset prefixes
                    for asset in self.assets:
                        prefix = SLUG_PREFIX[asset]
                        if slug.startswith(prefix):
                            if slug not in self.markets:
                                market = self._parse_event(asset, ev)
                                if market:
                                    self.markets[slug] = market
                                    added += 1
                            break
        except Exception as e:
            log.warning("[Watcher] Discovery error: %s", e)
        return added

    def _parse_event(self, asset: str, ev: dict) -> Optional[ShortMarket]:
        """Extract end_ts, token IDs, and ask prices from a Gamma event.

        We don't try to fetch the target price — we use jump-vs-ask logic instead.
        Target is set to 0 as a placeholder; it's populated later from Binance at start_ts.
        """
        try:
            markets = ev.get("markets", [])
            if not markets:
                return None
            m = markets[0]
            if m.get("closed") or not m.get("active", True):
                return None

            # End time
            end_iso = ev.get("endDate") or m.get("endDate")
            if not end_iso:
                return None
            end_ts = self._parse_iso(end_iso)
            if end_ts < time.time():
                return None

            # Tokens
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            outcomes = json.loads(m.get("outcomes", '["Up", "Down"]'))
            if len(token_ids) < 2 or len(outcomes) < 2:
                return None
            up_token = down_token = None
            for i, o in enumerate(outcomes):
                ol = o.lower()
                if "up" in ol and i < len(token_ids):
                    up_token = token_ids[i]
                elif "down" in ol and i < len(token_ids):
                    down_token = token_ids[i]
            if not up_token or not down_token:
                return None

            # Initial ask prices from Gamma (might be stale — WS/REST update later)
            best_ask = m.get("bestAsk")
            best_bid = m.get("bestBid")
            outcome_prices = []
            try:
                outcome_prices = json.loads(m.get("outcomePrices", "[]"))
            except Exception:
                pass

            up_ask = 0.0
            down_ask = 0.0
            if outcome_prices and len(outcome_prices) >= 2:
                # outcomePrices is [Up_mid, Down_mid]
                try:
                    up_ask = float(outcome_prices[0])
                    down_ask = float(outcome_prices[1])
                except (ValueError, TypeError):
                    pass
            # bestAsk is for the primary outcome (Up)
            if best_ask is not None:
                try:
                    up_ask = float(best_ask)
                except (ValueError, TypeError):
                    pass
            # Down side: we can derive from 1-best_bid (Up's bid) or outcomePrices
            # Better: need separate fetches from CLOB /book

            return ShortMarket(
                asset=asset,
                event_slug=ev["slug"],
                target_price=0.0,  # filled later (we don't require it)
                end_ts=end_ts,
                up_token_id=up_token,
                down_token_id=down_token,
                up_ask=up_ask,
                down_ask=down_ask,
            )
        except Exception as e:
            log.debug("[Watcher] Parse failed for %s: %s", ev.get("slug", "?"), e)
            return None

    def _parse_iso(self, iso: str) -> float:
        """ISO datetime -> epoch seconds."""
        try:
            from datetime import datetime, timezone
            for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S"):
                try:
                    dt = datetime.strptime(iso[:26], fmt[:len(iso[:26])])
                    return dt.replace(tzinfo=timezone.utc).timestamp()
                except ValueError:
                    continue
        except Exception:
            pass
        return 0.0

    def fetch_orderbook(self, market: ShortMarket):
        """Update up_ask and down_ask for a market via REST."""
        try:
            for token, side in [(market.up_token_id, "up"), (market.down_token_id, "down")]:
                resp = self.session.get(f"{POLY_CLOB}/book",
                                        params={"token_id": token}, timeout=3)
                if resp.status_code != 200:
                    continue
                book = resp.json()
                asks = book.get("asks", [])
                if asks:
                    # Sort ascending (lowest ask = best ask)
                    best = min(float(a["price"]) for a in asks if float(a.get("size", 0)) > 0)
                    if side == "up":
                        market.up_ask = best
                    else:
                        market.down_ask = best
            market.last_book_update = time.time()
        except Exception as e:
            log.debug("[Watcher] Book fetch failed for %s: %s", market.event_slug, e)

    def cleanup_expired(self):
        """Remove markets that have ended."""
        now = time.time()
        expired = [s for s, m in self.markets.items() if m.end_ts < now]
        for s in expired:
            del self.markets[s]
        return len(expired)

    def get_markets_for_asset(self, asset: str, max_seconds_left: float = 60) -> list[ShortMarket]:
        """Return active markets for an asset closing soon."""
        now = time.time()
        return [m for m in self.markets.values()
                if m.asset == asset and 0 < (m.end_ts - now) <= max_seconds_left]


# ============================================================================
# JUMP DETECTION
# ============================================================================

@dataclass
class Jump:
    asset: str
    direction: str  # "up" | "down"
    magnitude: float  # absolute price delta
    from_price: float
    to_price: float
    duration_ms: float
    ts: float


def detect_jump(history: list[PricePoint], window_ms: float = 1500,
                threshold: float = 25.0) -> Optional[Jump]:
    """Detect if there was a price jump >= threshold within the last window_ms.

    Returns Jump if found, else None. Uses the most recent price as endpoint.
    """
    if len(history) < 2:
        return None
    now_pt = history[-1]
    cutoff = now_pt.ts - window_ms / 1000.0
    # Find oldest point still inside window
    oldest = None
    for p in history:
        if p.ts >= cutoff:
            oldest = p
            break
    if oldest is None or oldest is now_pt:
        return None
    delta = now_pt.price - oldest.price
    if abs(delta) < threshold:
        return None
    return Jump(
        asset="",  # filled by caller
        direction="up" if delta > 0 else "down",
        magnitude=abs(delta),
        from_price=oldest.price,
        to_price=now_pt.price,
        duration_ms=(now_pt.ts - oldest.ts) * 1000,
        ts=now_pt.ts,
    )


# ============================================================================
# EXECUTION
# ============================================================================

# Cached Poly client (singleton)
_clob_client = None
_clob_lock = threading.Lock()


def get_clob_client():
    global _clob_client
    if _clob_client is not None:
        return _clob_client
    with _clob_lock:
        if _clob_client is not None:
            return _clob_client
        client = ClobClient(
            POLY_CLOB,
            key=os.getenv("POLY_PRIVATE_KEY"),
            chain_id=137,
            signature_type=int(os.getenv("POLY_SIGNATURE_TYPE", "1")),
            funder=os.getenv("POLYMARKET_PROXY_ADDRESS"),
        )
        client.set_api_creds(client.create_or_derive_api_creds())
        _clob_client = client
        return _clob_client


def execute_poly_order(token_id: str, ask_price: float, max_spend_usd: float,
                       slippage: float = 0.03, test_mode: bool = True) -> tuple[bool, str, int, str]:
    """Execute a FOK order on Poly.

    Returns (success, order_id, volume, error)
    """
    if ask_price <= 0 or ask_price >= 1:
        return False, "", 0, f"Invalid ask {ask_price}"
    limit_price = float(Decimal(str(ask_price + slippage)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN))
    if limit_price >= 0.99:
        limit_price = 0.99
    volume = int(max_spend_usd / limit_price)
    if volume < 5:  # Polymarket min is usually 5 contracts
        return False, "", 0, f"Volume too low ({volume})"

    if test_mode:
        log.info("[TEST] BUY %d @ $%.4f (limit) token=%s…", volume, limit_price, token_id[:16])
        return True, "TEST_ORDER", volume, ""

    if not HAS_CLOB:
        return False, "", 0, "py-clob-client not installed"

    try:
        client = get_clob_client()
        order = client.create_order(OrderArgs(
            token_id=token_id, price=limit_price, size=float(volume), side=BUY
        ))
        resp = client.post_order(order, OrderType.FOK)
        if resp.get("success"):
            return True, resp.get("orderID", ""), volume, ""
        return False, "", 0, resp.get("errorMsg", "Unknown")
    except Exception as e:
        return False, "", 0, str(e)


def append_log(record: TradeRecord):
    """Append trade record to JSONL log."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.__dict__) + "\n")
    except Exception as e:
        log.warning("Log write failed: %s", e)


# ============================================================================
# MAIN ENGINE
# ============================================================================

class JumpArbEngine:
    def __init__(self, assets: list[str], jump_threshold: float = 25.0,
                 jump_window_ms: float = 1500, max_seconds_left: float = 30,
                 min_seconds_left: float = 3, min_edge: float = 0.10,
                 max_trade_usd: float = 5.0, test_mode: bool = True,
                 cooldown_per_market_s: float = 10.0):
        self.assets = assets
        self.jump_threshold = jump_threshold
        self.jump_window_ms = jump_window_ms
        self.max_seconds_left = max_seconds_left
        self.min_seconds_left = min_seconds_left
        self.min_edge = min_edge
        self.max_trade_usd = max_trade_usd
        self.test_mode = test_mode
        self.cooldown_per_market = cooldown_per_market_s

        self.binance = BinanceWSClient(assets)
        self.watcher = ShortMarketWatcher(assets)
        self.poly_ws: Optional[object] = None
        self._subscribed_tokens: set[str] = set()
        self._last_trade_per_market: dict[str, float] = {}
        self._running = False

    def run(self):
        log.info("=" * 60)
        log.info("Crypto Jump Arb Engine | %s | Assets: %s",
                 "TEST" if self.test_mode else "LIVE", self.assets)
        log.info("  Jump threshold: $%.2f in %dms", self.jump_threshold, self.jump_window_ms)
        log.info("  Window: %d-%ds before close | Min edge: %.0f¢",
                 self.min_seconds_left, self.max_seconds_left, self.min_edge * 100)
        log.info("  Max trade: $%.2f", self.max_trade_usd)
        log.info("=" * 60)

        # Initial market discovery
        added = self.watcher.discover()
        log.info("[Init] Discovered %d markets", added)

        # Start Poly WS — real-time orderbook updates for all tokens
        if HAS_POLY_WS:
            self.poly_ws = PolyWSClient()
            initial_tokens = self._collect_tokens()
            self.poly_ws.start(initial_tokens=initial_tokens)
            self._subscribed_tokens.update(initial_tokens)
            log.info("[Init] Poly WS started with %d tokens", len(initial_tokens))
        else:
            log.warning("[Init] Poly WS not available — using REST polling only")

        self.binance.start()
        time.sleep(3)
        if not self.binance.is_connected:
            log.error("Binance WS failed to connect")
            return

        self._running = True
        last_discover = time.time()
        last_book_refresh = 0
        loop_count = 0

        try:
            while self._running:
                now = time.time()
                loop_count += 1

                # Re-discover every 2 minutes (markets created continuously)
                if now - last_discover > 120:
                    n_added = self.watcher.discover()
                    n_removed = self.watcher.cleanup_expired()
                    if n_added or n_removed:
                        log.info("[Discover] +%d -%d (total %d)",
                                 n_added, n_removed, len(self.watcher.markets))
                    # Subscribe new tokens on WS
                    if self.poly_ws and n_added > 0:
                        new_tokens = self._collect_tokens() - self._subscribed_tokens
                        if new_tokens:
                            self.poly_ws.subscribe_tokens(new_tokens)
                            self._subscribed_tokens.update(new_tokens)
                            log.info("[WS-P] Subscribed %d new tokens", len(new_tokens))
                    last_discover = now

                # Update asks from WS (every 500ms) for closing-soon markets
                # Fallback to REST only if WS not available or token not yet cached
                if now - last_book_refresh > 0.5:
                    for asset in self.assets:
                        for m in self.watcher.get_markets_for_asset(asset, self.max_seconds_left):
                            self._refresh_asks(m)
                    last_book_refresh = now

                # Check each asset for jumps
                for asset in self.assets:
                    history = self.binance.get_history(asset)
                    if not history:
                        continue
                    jump = detect_jump(history, self.jump_window_ms, self.jump_threshold)
                    if not jump:
                        continue
                    jump.asset = asset
                    self._evaluate_jump(jump)

                # Status log every ~30s
                if loop_count % 300 == 0:
                    self._log_status()

                time.sleep(0.1)  # 10Hz check loop

        except KeyboardInterrupt:
            log.info("Stopped by user")
        finally:
            self.binance.stop()
            if self.poly_ws:
                self.poly_ws.stop()

    def _evaluate_jump(self, jump: Jump):
        """A jump was detected — check if any markets haven't priced it in yet.

        Logic (simplified — no need for target price):
        - Jump magnitude tells us about momentum relative to BTC volatility
        - Compare to time remaining: $25 jump in 500ms with 10s left -> very strong signal
        - If Poly's ask for the jumping side is still in a "cheap" range,
          the market hasn't priced the jump yet -> trade it

        The model: expected value of the jump persisting is proportional to
        magnitude / sqrt(seconds_left * std_per_sec). If this z-score is high
        AND Poly's implied prob is < some threshold, fire.
        """
        import math
        STD_PER_SEC = {"BTC": 4.0, "ETH": 0.3, "SOL": 0.05, "XRP": 0.002, "DOGE": 0.0003}
        std_sec = STD_PER_SEC.get(jump.asset, 1.0)

        markets = self.watcher.get_markets_for_asset(jump.asset, self.max_seconds_left)
        for m in markets:
            seconds_left = m.end_ts - time.time()
            if seconds_left < self.min_seconds_left:
                continue

            last = self._last_trade_per_market.get(m.event_slug, 0)
            if time.time() - last < self.cooldown_per_market:
                continue

            # z-score of the jump vs time-scaled volatility
            vol_scale = std_sec * math.sqrt(max(seconds_left, 1))
            z = jump.magnitude / vol_scale if vol_scale > 0 else 0
            # Estimated P(direction holds) via normal CDF of z
            true_prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))

            # Pick side
            if jump.direction == "up":
                side = "up"; ask = m.up_ask; token = m.up_token_id
            else:
                side = "down"; ask = m.down_ask; token = m.down_token_id

            if ask <= 0 or ask >= 0.95:
                continue  # no data or already priced

            edge = true_prob - ask
            if edge < self.min_edge:
                continue

            log.info(
                "[SIGNAL] %s jump %s $%.2f->$%.2f (%.0fms, z=%.1f) | left=%.1fs | "
                "%s_ask=%.2f true_prob=%.2f edge=%.2f",
                jump.asset, jump.direction, jump.from_price, jump.to_price,
                jump.duration_ms, z, seconds_left,
                side, ask, true_prob, edge,
            )

            ok, oid, vol, err = execute_poly_order(
                token, ask, self.max_trade_usd, test_mode=self.test_mode
            )

            record = TradeRecord(
                ts=datetime.now().isoformat(timespec="seconds"),
                asset=jump.asset, side=side,
                target=m.target_price, binance_price=jump.to_price,
                delta=jump.magnitude * (1 if jump.direction == "up" else -1),
                seconds_left=seconds_left,
                poly_ask=ask, edge=edge, volume=vol,
                trade_value=vol * ask, test_mode=self.test_mode,
                order_id=oid, error=err,
            )
            append_log(record)
            self._last_trade_per_market[m.event_slug] = time.time()

            if ok:
                log.info("  -> ORDER OK: vol=%d order_id=%s", vol, oid)
            else:
                log.warning("  -> ORDER FAILED: %s", err)

    def _collect_tokens(self) -> set[str]:
        """Return set of all up/down token IDs across active markets."""
        tokens = set()
        for m in self.watcher.markets.values():
            if m.up_token_id:
                tokens.add(m.up_token_id)
            if m.down_token_id:
                tokens.add(m.down_token_id)
        return tokens

    def _refresh_asks(self, market: ShortMarket):
        """Update up_ask and down_ask — prefer WS, fall back to REST."""
        got_up = got_down = False

        if self.poly_ws:
            up = self.poly_ws.get_best_price(market.up_token_id)
            if up and up.get("best_ask", 0) > 0:
                market.up_ask = up["best_ask"]
                got_up = True
            down = self.poly_ws.get_best_price(market.down_token_id)
            if down and down.get("best_ask", 0) > 0:
                market.down_ask = down["best_ask"]
                got_down = True

        # If WS doesn't have data yet, fall back to REST for this market only
        if not (got_up and got_down):
            # Rate-limit: only REST-fetch at most once every 3s per market
            if time.time() - market.last_book_update > 3:
                self.watcher.fetch_orderbook(market)
                market.last_book_update = time.time()

    def _log_status(self):
        latest = {a: self.binance.get_latest(a) for a in self.assets}
        n_active = len(self.watcher.markets)
        n_closing = sum(
            1 for m in self.watcher.markets.values()
            if 0 < (m.end_ts - time.time()) <= self.max_seconds_left
        )
        prices = " ".join(f"{a}=${p:.2f}" if p else f"{a}=?" for a, p in latest.items())
        ws_status = ""
        if self.poly_ws:
            ws_status = f" | Poly-WS: {'OK' if self.poly_ws.is_connected else 'OFF'} ({len(self._subscribed_tokens)} tokens)"
        log.info("[Status] %d markets (%d closing <%ds) | %s%s",
                 n_active, n_closing, int(self.max_seconds_left), prices, ws_status)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Crypto jump arb on Polymarket short markets")
    parser.add_argument("--live", action="store_true", help="Place REAL orders (default: dry-run)")
    parser.add_argument("--max-trade", type=float, default=5.0, help="Max USD per trade (default 5)")
    parser.add_argument("--jump-threshold", type=float, default=25.0,
                        help="Min jump magnitude in BTC USD-equivalent (default 25)")
    parser.add_argument("--jump-window", type=float, default=1500,
                        help="Jump detection window in ms (default 1500)")
    parser.add_argument("--max-seconds", type=float, default=120,
                        help="Only trade markets closing within X seconds (default 120)")
    parser.add_argument("--min-seconds", type=float, default=3,
                        help="Don't trade if < X seconds left (default 3)")
    parser.add_argument("--min-edge", type=float, default=0.10,
                        help="Min edge to trade (default 0.10 = 10¢)")
    parser.add_argument("--assets", default="BTC", help="Comma-separated assets (default BTC)")
    parser.add_argument("--cooldown", type=float, default=10,
                        help="Cooldown per market in seconds (default 10)")
    args = parser.parse_args()

    if not HAS_WEBSOCKETS:
        log.error("websockets not installed. Run: pip install websockets")
        return

    test_mode = not args.live
    if not test_mode:
        log.warning("=" * 60)
        log.warning("  LIVE MODE — REAL MONEY")
        log.warning("=" * 60)
        time.sleep(3)

    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
    # Validate
    bad = [a for a in assets if a not in ASSET_MAP]
    if bad:
        log.error("Unknown assets: %s. Supported: %s", bad, list(ASSET_MAP.keys()))
        return

    # Per-asset jump threshold scaling (BTC=25 -> ETH=2 -> SOL=0.5 -> XRP=0.01 -> DOGE=0.001)
    # User passed --jump-threshold for BTC; scale others by typical price ratio
    asset_thresholds = {"BTC": 1.0, "ETH": 0.08, "SOL": 0.012, "XRP": 0.0003, "DOGE": 0.00004}

    engine = JumpArbEngine(
        assets=assets,
        jump_threshold=args.jump_threshold,  # this is the BTC number; per-asset scale could be added
        jump_window_ms=args.jump_window,
        max_seconds_left=args.max_seconds,
        min_seconds_left=args.min_seconds,
        min_edge=args.min_edge,
        max_trade_usd=args.max_trade,
        test_mode=test_mode,
        cooldown_per_market_s=args.cooldown,
    )
    engine.run()


if __name__ == "__main__":
    main()
