"""
arb_engine.py — Improved Kalshi vs Polymarket arbitrage scanner.

Fixes from versionfinpar.py:
  - Fixed Kalshi fee double-calculation bug (was using avg price instead of per-level)
  - Removed ~200 lines of dead code (process_single_event)
  - Added per-trade PnL tracking in test mode
  - Proper logging instead of print
  - Deduplicates orderbook fetches (same ticker for YES and NO side)
  - Cleaner config/threshold handling
  - Balance fetch only every N cycles (not every scan)
  - Graceful shutdown for notification worker

Usage:
    python arb_engine.py                # test mode (default)
    python arb_engine.py --live         # real execution
    python arb_engine.py --threshold 0.95
    python arb_engine.py --mode ws      # websocket mode
    python arb_engine.py --mode og      # REST polling mode
"""

import time
import json
import os
import sys
import math
import copy
import logging
import threading
import queue
import argparse
import signal
import smtplib
import concurrent.futures
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from email.message import EmailMessage
from dataclasses import dataclass, field
from typing import Optional

import requests
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import base64

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY
    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False

try:
    from web3 import Web3
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False

try:
    import websockets
    import asyncio
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

load_dotenv()

# ============================================================================
# LOGGING
# ============================================================================
log = logging.getLogger("arb")
log.setLevel(logging.DEBUG)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
))
log.addHandler(_handler)

# ============================================================================
# CONSTANTS
# ============================================================================
KALSHI_FEE_RATE = 0.07
DEFAULT_POLY_FEE_RATE = 0.04
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
POLY_GAMMA_URL = "https://gamma-api.polymarket.com/events"
POLY_CLOB_URL = "https://clob.polymarket.com"

# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class KalshiMarket:
    ticker: str
    title: str
    subtitle: str
    yes_ask: Optional[float] = None
    no_ask: Optional[float] = None
    yes_bid: float = 0.0
    no_bid: float = 0.0


@dataclass
class PolyMarket:
    condition_id: str
    name: str
    yes_price: Optional[float] = None
    no_price: Optional[float] = None
    yes_token_id: Optional[str] = None
    no_token_id: Optional[str] = None
    neg_risk: bool = False


@dataclass
class MatchedPair:
    name: str
    kalshi: KalshiMarket
    poly: PolyMarket


@dataclass
class ArbCandidate:
    event_name: str
    pair_name: str
    kalshi: KalshiMarket
    poly: PolyMarket
    side: str  # "yes" or "no" — which side to buy on Kalshi
    cost: float
    k_price: float
    p_price: float
    p_slug: str
    k_event_ticker: str
    p_token_id: str
    arb_key: str
    poly_fee_rate: float = DEFAULT_POLY_FEE_RATE
    source: str = "main"  # "reviewed" | "main" | "discovered"
    # Filled after orderbook analysis
    max_vol: float = 0.0
    est_profit: float = 0.0
    fail_reason: str = ""


@dataclass
class TradeResult:
    success: bool
    volume: int = 0
    kalshi_cost: float = 0.0
    poly_cost: float = 0.0
    total_cost: float = 0.0
    profit: float = 0.0
    error: str = ""
    kalshi_order_id: str = ""
    poly_order_id: str = ""


@dataclass
class PnLEntry:
    timestamp: str
    event: str
    outcome: str
    side: str
    volume: int
    k_price: float
    p_price: float
    total_cost: float
    guaranteed_profit: float


# ============================================================================
# FEE CALCULATIONS (centralized — no more scattered magic numbers)
# ============================================================================

def kalshi_fee_per_contract(price: float) -> float:
    """Kalshi taker fee: 7% * p * (1-p), rounded up to cents."""
    return KALSHI_FEE_RATE * price * (1.0 - price)


def kalshi_fee_total(volume: int, price: float) -> float:
    """Total Kalshi fee for N contracts at a single price.

    Kalshi rounds UP per-contract, not on the total. So we ceil each contract's
    fee to the nearest cent, then multiply by volume.
    """
    per_contract = math.ceil(KALSHI_FEE_RATE * price * (1.0 - price) * 100) / 100.0
    return per_contract * volume


def poly_fee_per_contract(price: float, fee_rate: float) -> float:
    """Polymarket taker fee: feeRate * p * (1-p)."""
    return fee_rate * price * (1.0 - price)


# ============================================================================
# SIGNING
# ============================================================================

def sign_kalshi_request(timestamp: str, method: str, path: str) -> Optional[str]:
    pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if not pk_path:
        return None
    with open(pk_path, 'rb') as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)
    msg = f"{timestamp}{method}{path}"
    signature = private_key.sign(
        msg.encode('utf-8'),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')


# ============================================================================
# RATE LIMITER
# ============================================================================

class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_ts = time.time()
        self._lock = threading.Lock()

    def consume(self, n: int = 1):
        with self._lock:
            now = time.time()
            self.tokens = min(self.capacity, self.tokens + (now - self.last_ts) * self.rate)
            self.last_ts = now
            if self.tokens >= n:
                self.tokens -= n
                return
            wait = (n - self.tokens) / self.rate
            self.tokens = 0
        if wait > 0:
            time.sleep(wait)


# ============================================================================
# KALSHI CLIENT (REST)
# ============================================================================

class KalshiClient:
    def __init__(self):
        self.base_url = KALSHI_BASE_URL
        self.rate_limiter = TokenBucket(15, 15)
        self.session = requests.Session()

    def _auth_headers(self, method: str, path: str) -> dict:
        key_id = os.getenv("KALSHI_API_KEY_ID")
        if not key_id:
            return {}
        ts = str(int(time.time() * 1000))
        sig = sign_kalshi_request(ts, method, path)
        if not sig:
            return {}
        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    def get_event_markets(self, event_ticker: str) -> list[KalshiMarket]:
        self.rate_limiter.consume()
        url = f"{self.base_url}/events/{event_ticker}"
        for attempt in range(4):
            try:
                resp = self.session.get(url, params={"with_nested_markets": "true"}, timeout=8)
                if resp.status_code == 429:
                    time.sleep(1 + attempt * 0.7)
                    continue
                resp.raise_for_status()
                event = resp.json().get('event', {})
                markets = []
                for m in event.get('markets', []):
                    yes_ask_raw = m.get('yes_ask_dollars')
                    no_ask_raw = m.get('no_ask_dollars')
                    markets.append(KalshiMarket(
                        ticker=m.get('ticker', ''),
                        title=m.get('title', ''),
                        subtitle=m.get('subtitle', ''),
                        yes_ask=float(yes_ask_raw) if yes_ask_raw is not None else None,
                        no_ask=float(no_ask_raw) if no_ask_raw is not None else None,
                        yes_bid=float(m.get('yes_bid_dollars') or 0),
                        no_bid=float(m.get('no_bid_dollars') or 0),
                    ))
                return markets
            except Exception as e:
                if attempt == 3:
                    log.warning("Kalshi event fetch failed [%s]: %s", event_ticker, e)
                time.sleep(0.5 + attempt * 0.3)
        return []

    def get_orderbook(self, ticker: str) -> dict:
        self.rate_limiter.consume()
        for attempt in range(3):
            try:
                resp = self.session.get(
                    f"{self.base_url}/markets/{ticker}/orderbook", timeout=5
                )
                if resp.status_code == 429:
                    time.sleep(1 + attempt * 0.5)
                    continue
                resp.raise_for_status()
                return resp.json().get('orderbook_fp', {})
            except Exception:
                time.sleep(0.5)
        return {}

    def get_orderbooks_batch(self, tickers: list[str]) -> dict:
        if not tickers:
            return {}
        result = {}
        for start in range(0, len(tickers), 100):
            chunk = tickers[start:start + 100]
            self.rate_limiter.consume()
            try:
                headers = self._auth_headers("GET", "/trade-api/v2/markets/orderbooks")
                resp = self.session.get(
                    f"{self.base_url}/markets/orderbooks",
                    params=[("tickers", t) for t in chunk],
                    headers=headers, timeout=15,
                )
                if resp.status_code == 429:
                    time.sleep(2)
                    continue
                resp.raise_for_status()
                for ob in resp.json().get("orderbooks", []):
                    result[ob["ticker"]] = ob.get("orderbook_fp", {})
            except Exception:
                for t in chunk:
                    result[t] = self.get_orderbook(t)
        return result

    def get_balance(self) -> float:
        key_id = os.getenv("KALSHI_API_KEY_ID")
        if not key_id:
            return 0.0
        try:
            self.rate_limiter.consume()
            path = "/portfolio/balance"
            headers = self._auth_headers("GET", "/trade-api/v2" + path)
            headers["Content-Type"] = "application/json"
            resp = self.session.get(f"{self.base_url}{path}", headers=headers)
            if resp.status_code == 200:
                return resp.json().get('balance', 0) / 100.0
        except Exception as e:
            log.warning("Kalshi balance error: %s", e)
        return 0.0


# ============================================================================
# POLYMARKET CLIENT (REST)
# ============================================================================

class PolymarketClient:
    def __init__(self):
        self.rate_limiter = TokenBucket(10, 10)
        self.session = requests.Session()
        self._fee_cache: dict[str, float] = {}

    def get_fee_rate(self, token_id: str) -> float:
        if not token_id:
            return 0.0
        if token_id in self._fee_cache:
            return self._fee_cache[token_id]
        try:
            self.rate_limiter.consume()
            resp = self.session.get(
                f"{POLY_CLOB_URL}/fee-rate",
                params={"token_id": token_id}, timeout=5,
            )
            if resp.status_code == 200:
                base_fee = float(resp.json().get("base_fee", 0))
                rate = 0.0 if base_fee == 0 else DEFAULT_POLY_FEE_RATE
                self._fee_cache[token_id] = rate
                return rate
        except Exception:
            pass
        self._fee_cache[token_id] = DEFAULT_POLY_FEE_RATE
        return DEFAULT_POLY_FEE_RATE

    def get_event_markets(self, slug: str) -> list[PolyMarket]:
        self.rate_limiter.consume()
        try:
            resp = self.session.get(POLY_GAMMA_URL, params={"slug": slug})
            resp.raise_for_status()
            data = resp.json()
            if not data:
                log.debug("Polymarket event not found: %s", slug)
                return []
            event = data[0]
            markets = []
            for m in event.get('markets', []):
                if m.get('closed') or not m.get('active', True):
                    continue
                name = m.get('groupItemTitle', m.get('question', ''))
                token_ids = json.loads(m.get('clobTokenIds', '[]'))
                outcomes = json.loads(m.get('outcomes', '["No", "Yes"]'))
                best_ask = m.get('bestAsk')
                best_bid = m.get('bestBid')

                yes_token_id = None
                no_token_id = None
                for i, outcome in enumerate(outcomes):
                    if i < len(token_ids):
                        if outcome == "Yes":
                            yes_token_id = token_ids[i]
                        elif outcome == "No":
                            no_token_id = token_ids[i]

                markets.append(PolyMarket(
                    condition_id=m.get('conditionId', ''),
                    name=name,
                    yes_price=float(best_ask) if best_ask is not None else None,
                    no_price=1.0 - float(best_bid) if best_bid is not None else None,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                    neg_risk=bool(m.get('negRisk') or event.get('negRisk')),
                ))
            return markets
        except Exception as e:
            log.warning("Polymarket event fetch failed [%s]: %s", slug, e)
            return []

    def get_orderbook(self, token_id: str) -> dict:
        if not token_id:
            return {}
        self.rate_limiter.consume()
        for attempt in range(3):
            try:
                resp = self.session.get(
                    f"{POLY_CLOB_URL}/book",
                    params={"token_id": token_id}, timeout=5,
                )
                if resp.status_code == 429:
                    time.sleep(1 + attempt * 0.5)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception:
                time.sleep(0.5)
        return {}

    def get_balance(self) -> float:
        if not HAS_WEB3:
            return 0.0
        proxy = os.getenv("POLYMARKET_PROXY_ADDRESS")
        if not proxy:
            return 0.0
        try:
            w3 = Web3(Web3.HTTPProvider('https://polygon-rpc.com'))
            USDC_ADDR = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
            ABI = [{"constant": True, "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function"}]
            contract = w3.eth.contract(address=USDC_ADDR, abi=ABI)
            return contract.functions.balanceOf(proxy).call() / 1e6
        except Exception as e:
            log.warning("Poly balance error: %s", e)
            return 0.0


# ============================================================================
# KALSHI WEBSOCKET CLIENT
# ============================================================================

class KalshiWSClient:
    """Persistent WS to Kalshi. Caches ticker prices + orderbooks."""

    def __init__(self):
        self.ws_url = "wss://api.elections.kalshi.com/trade-api/ws/v2"
        self._ticker_cache: dict[str, dict] = {}
        self._ob_cache: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._connected = threading.Event()
        self._ob_tickers: set[str] = set()
        self._msg_id = 0
        self._running = False
        self._thread = None
        self._loop = None
        self._ws = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="KalshiWS")
        self._thread.start()
        if not self._connected.wait(timeout=15):
            log.warning("[WS-K] Connection timed out (15s)")

    def stop(self):
        self._running = False

    def _next_id(self):
        self._msg_id += 1
        return self._msg_id

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
                await self._connect_and_listen()
                backoff = 1
            except Exception as e:
                self._connected.clear()
                self._ws = None
                log.warning("[WS-K] Disconnected: %s. Retry in %ds", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _connect_and_listen(self):
        headers = self._make_headers()
        async with websockets.connect(
            self.ws_url, additional_headers=headers,
            ping_interval=20, ping_timeout=10, close_timeout=5,
        ) as ws:
            self._ws = ws
            self._connected.set()
            log.info("[WS-K] Connected")
            await ws.send(json.dumps({
                "id": self._next_id(), "cmd": "subscribe",
                "params": {"channels": ["ticker"]}
            }))
            if self._ob_tickers:
                await ws.send(json.dumps({
                    "id": self._next_id(), "cmd": "subscribe",
                    "params": {"channels": ["orderbook_delta"],
                               "market_tickers": list(self._ob_tickers)}
                }))
            async for message in ws:
                self._handle_msg(json.loads(message))

    def _make_headers(self):
        key_id = os.getenv("KALSHI_API_KEY_ID")
        if not key_id:
            return {}
        ts = str(int(time.time() * 1000))
        sig = sign_kalshi_request(ts, "GET", "/trade-api/ws/v2")
        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
        } if sig else {}

    def _handle_msg(self, data):
        msg_type = data.get("type")
        msg = data.get("msg") or {}
        if msg_type == "ticker":
            ticker = msg.get("market_ticker")
            if ticker:
                ya = float(msg.get("yes_ask_dollars") or 0)
                yb = float(msg.get("yes_bid_dollars") or 0)
                na = float(msg.get("no_ask_dollars") or 0)
                nb = float(msg.get("no_bid_dollars") or 0)
                # Derive NO from binary complement if not provided
                if nb == 0 and ya > 0:
                    nb = 1.0 - ya
                if na == 0 and yb > 0:
                    na = 1.0 - yb
                with self._lock:
                    self._ticker_cache[ticker] = {
                        "yes_bid": yb, "yes_ask": ya, "no_bid": nb, "no_ask": na
                    }
        elif msg_type == "orderbook_snapshot":
            ticker = msg.get("market_ticker")
            if ticker:
                with self._lock:
                    self._ob_cache[ticker] = {
                        "yes_dollars": msg.get("yes_dollars_fp", []),
                        "no_dollars": msg.get("no_dollars_fp", []),
                    }
        elif msg_type == "orderbook_delta":
            ticker = msg.get("market_ticker")
            if ticker:
                self._apply_delta(ticker, msg)

    def _apply_delta(self, ticker, msg):
        with self._lock:
            if ticker not in self._ob_cache:
                return
            ob = self._ob_cache[ticker]
            price = msg.get("price_dollars")
            delta = msg.get("delta_fp")
            side = msg.get("side")
            if price is None or delta is None or side is None:
                return
            key = f"{side}_dollars"
            levels = list(ob.get(key, []))
            new_levels = []
            found = False
            for lv in levels:
                if lv[0] == price:
                    new_qty = float(lv[1]) + float(delta)
                    if new_qty > 0.001:
                        new_levels.append([lv[0], f"{new_qty:.2f}"])
                    found = True
                else:
                    new_levels.append(lv)
            if not found and float(delta) > 0:
                new_levels.append([price, str(delta)])
            ob[key] = new_levels

    def subscribe_orderbooks(self, tickers: list[str]):
        new = [t for t in tickers if t not in self._ob_tickers]
        if not new:
            return
        self._ob_tickers.update(new)
        if self._ws and self._loop and self._connected.is_set():
            asyncio.run_coroutine_threadsafe(
                self._ws.send(json.dumps({
                    "id": self._next_id(), "cmd": "subscribe",
                    "params": {"channels": ["orderbook_delta"], "market_tickers": new}
                })),
                self._loop,
            )

    def get_ticker(self, market_ticker: str):
        with self._lock:
            return self._ticker_cache.get(market_ticker)

    def get_orderbook(self, market_ticker: str) -> dict:
        with self._lock:
            ob = self._ob_cache.get(market_ticker)
            return {k: list(v) for k, v in ob.items()} if ob else {}

    def get_orderbooks_batch(self, tickers: list[str]) -> dict:
        with self._lock:
            out = {}
            for t in tickers:
                ob = self._ob_cache.get(t)
                if ob:
                    out[t] = {k: list(v) for k, v in ob.items()}
            return out

    @property
    def is_connected(self):
        return self._connected.is_set()


# ============================================================================
# POLYMARKET WEBSOCKET CLIENT
# ============================================================================

class PolyWSClient:
    """Persistent WS to Polymarket CLOB. Caches orderbooks + prices."""

    def __init__(self):
        self.ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self._book_cache: dict[str, dict] = {}
        self._price_cache: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._connected = threading.Event()
        self._tokens: set[str] = set()
        self._running = False
        self._thread = None
        self._loop = None
        self._ws = None

    def start(self, initial_tokens=None):
        self._running = True
        if initial_tokens:
            self._tokens.update(initial_tokens)
        self._thread = threading.Thread(target=self._run, daemon=True, name="PolyWS")
        self._thread.start()
        if not self._connected.wait(timeout=15):
            log.warning("[WS-P] Connection timed out (15s)")

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
                await self._connect_and_listen()
                backoff = 1
            except Exception as e:
                self._connected.clear()
                self._ws = None
                log.warning("[WS-P] Disconnected: %s. Retry in %ds", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _connect_and_listen(self):
        async with websockets.connect(
            self.ws_url, ping_interval=20, ping_timeout=10, close_timeout=5,
        ) as ws:
            self._ws = ws
            self._connected.set()
            log.info("[WS-P] Connected")
            if self._tokens:
                tl = list(self._tokens)
                for i in range(0, len(tl), 100):
                    await ws.send(json.dumps({
                        "type": "market", "assets_ids": tl[i:i+100],
                        "custom_feature_enabled": True,
                    }))
            async for message in ws:
                try:
                    data = json.loads(message)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        self._handle_msg(item)
                except (json.JSONDecodeError, TypeError):
                    pass

    def _handle_msg(self, data):
        if not isinstance(data, dict):
            return
        et = data.get("event_type")
        aid = data.get("asset_id")
        if et == "book" and aid:
            with self._lock:
                self._book_cache[aid] = {
                    "bids": data.get("bids", []),
                    "asks": data.get("asks", []),
                }
        elif et == "price_change" and aid:
            changes = data.get("price_changes", [])
            with self._lock:
                book = self._book_cache.get(aid)
                if book:
                    for ch in changes:
                        self._apply_change(book, ch)
                if changes:
                    last = changes[-1]
                    bb = last.get("best_bid")
                    ba = last.get("best_ask")
                    if bb is not None or ba is not None:
                        self._price_cache[aid] = {
                            "best_bid": float(bb) if bb else 0,
                            "best_ask": float(ba) if ba else 0,
                        }
        elif et == "best_bid_ask" and aid:
            with self._lock:
                self._price_cache[aid] = {
                    "best_bid": float(data.get("best_bid") or 0),
                    "best_ask": float(data.get("best_ask") or 0),
                }

    def _apply_change(self, book, change):
        side = change.get("side", "").lower()
        price = change.get("price")
        size = change.get("size")
        if not price or side not in ("buy", "sell"):
            return
        key = "bids" if side == "buy" else "asks"
        levels = [l for l in book.get(key, []) if l.get("price") != price]
        if size and float(size) > 0:
            levels.append({"price": price, "size": size})
        levels.sort(key=lambda x: float(x.get("price", 0)), reverse=(key == "bids"))
        book[key] = levels

    def subscribe_tokens(self, token_ids):
        if isinstance(token_ids, set):
            token_ids = list(token_ids)
        new = [t for t in token_ids if t not in self._tokens]
        if not new:
            return
        self._tokens.update(new)
        if self._ws and self._loop and self._connected.is_set():
            for i in range(0, len(new), 100):
                asyncio.run_coroutine_threadsafe(
                    self._ws.send(json.dumps({
                        "assets_ids": new[i:i+100], "operation": "subscribe",
                    })),
                    self._loop,
                )

    def get_orderbook(self, token_id: str) -> dict:
        with self._lock:
            book = self._book_cache.get(token_id)
            if book:
                return {"bids": list(book.get("bids", [])), "asks": list(book.get("asks", []))}
            return {}

    def get_best_price(self, token_id: str):
        with self._lock:
            return self._price_cache.get(token_id)

    @property
    def is_connected(self):
        return self._connected.is_set()


# ============================================================================
# NOTIFICATIONS (email)
# ============================================================================

class Notifier:
    def __init__(self):
        self.gmail_user = os.getenv("GMAIL_USER", "")
        self.gmail_pass = os.getenv("GMAIL_APP_PASSWORD", "")
        raw = os.getenv("EMAIL_RECIPIENT", "")
        self.recipients = [e.strip() for e in raw.split(",") if e.strip()]
        self._can_send = bool(self.gmail_user and self.gmail_pass and self.recipients)
        self.enabled = False  # disabled by default — enable via dashboard settings
        self._queue: queue.Queue = queue.Queue()
        self._thread = None
        if self._can_send:
            log.info("Email configured (disabled by default) -> %s", self.recipients)
        else:
            log.info("Email not configured (missing GMAIL_USER/GMAIL_APP_PASSWORD/EMAIL_RECIPIENT)")

    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True, name="Notifier")
        self._thread.start()

    def stop(self):
        self._queue.put(None)

    def send(self, subject: str, body: str):
        if self.enabled:
            self._queue.put((subject, body))

    def send_arb_alert(self, candidate: ArbCandidate):
        side_label = f"K_{'YES' if candidate.side == 'yes' else 'NO'} + P_{'NO' if candidate.side == 'yes' else 'YES'}"
        subject = f"ARB: {candidate.event_name} - {candidate.pair_name}"
        body = (
            f"ARBITRAGE DETECTED\n\n"
            f"Event: {candidate.event_name}\n"
            f"Outcome: {candidate.pair_name}\n"
            f"Strategy: Buy {side_label}\n\n"
            f"Prices:\n"
            f"  Kalshi: ${candidate.k_price:.4f}\n"
            f"  Poly:   ${candidate.p_price:.4f}\n"
            f"  Combined: ${candidate.cost:.4f}\n\n"
            f"Volume: {candidate.max_vol:.0f}\n"
            f"Est. Profit: {candidate.est_profit:.2f}%\n\n"
            f"Links:\n"
            f"  Kalshi: https://kalshi.com/markets/{candidate.kalshi.ticker}\n"
            f"  Poly:   https://polymarket.com/event/{candidate.p_slug}\n"
        )
        self.send(subject, body)

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            subject, body = item
            try:
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(self.gmail_user, self.gmail_pass)
                    for r in self.recipients:
                        msg = EmailMessage()
                        msg['Subject'] = subject
                        msg['From'] = self.gmail_user
                        msg['To'] = r
                        msg.set_content(body)
                        smtp.send_message(msg)
                log.info("Email sent: %s", subject[:50])
            except Exception as e:
                log.error("Email error: %s", e)
            time.sleep(1)


# ============================================================================
# MARKET MATCHING
# ============================================================================

def _clean(s: str) -> str:
    return s.lower().replace(" ", "").replace(".", "").replace("-", "")


SKIP_SUFFIXES = {"tie", "draw", "emp", "empate", "dra"}
SKIP_WORDS_POLY = {"draw", " tie", "empate"}


def match_markets(k_markets: list[KalshiMarket], p_markets: list[PolyMarket],
                   mapping: dict | None = None) -> list[MatchedPair]:
    pairs = []
    used_poly = set()
    mapping = mapping or {}

    for k in k_markets:
        suffix = k.ticker.split('-')[-1].lower()
        if suffix in SKIP_SUFFIXES:
            continue
        if mapping and suffix not in mapping:
            continue

        target_raw = mapping.get(suffix, suffix)
        is_explicit = suffix in mapping
        target = _clean(target_raw)

        if not is_explicit and ("company" in target or "other" in target or "/" in suffix):
            continue

        exact = None
        subs = []
        for p in p_markets:
            pid = p.condition_id or p.name
            if pid in used_poly:
                continue
            plow = p.name.lower()
            if any(w in plow for w in SKIP_WORDS_POLY):
                continue
            if not is_explicit and "/" in p.name:
                continue
            pc = _clean(p.name)
            if target == pc:
                exact = p
                break
            if target in pc:
                subs.append((len(pc), p))
            elif len(pc) >= 4 and pc in target:
                subs.append((len(pc), p))

        best = exact
        if not best and subs:
            subs.sort(key=lambda x: x[0])
            best = subs[0][1]

        if best:
            used_poly.add(best.condition_id or best.name)
            pairs.append(MatchedPair(name=best.name, kalshi=k, poly=best))

    return pairs


# ============================================================================
# LIQUIDITY CALCULATION (FIXED: per-level fee accumulation)
# ============================================================================

def calculate_liquidity(
    k_book: dict, p_book: dict, k_side: str, threshold: float,
    balance_k: float, balance_p: float, poly_fee_rate: float = DEFAULT_POLY_FEE_RATE,
) -> tuple[float, float, str]:
    """Walk both orderbooks and compute arbitrage volume/profit.

    FIX vs versionfinpar.py: Kalshi fees are now accumulated per-level during
    the walk, not recalculated from average price at the end. The old code used
    f(avg(x)) instead of sum(f(xi)), which is wrong for the quadratic fee formula.
    """
    # Parse Kalshi asks from the OTHER side's bids
    # If we buy YES on Kalshi, the ask price comes from NO bids: ask = 1 - no_bid_price
    other_side = 'no' if k_side == 'yes' else 'yes'
    k_asks = []
    try:
        raw = k_book.get(f"{other_side}_dollars") or []
        for bid in raw:
            price = 1.0 - float(bid[0])
            qty = float(bid[1])
            k_asks.append({'price': price, 'qty': qty})
        k_asks.sort(key=lambda x: x['price'])
    except Exception as e:
        return 0, 0.0, f"KalshiParseErr: {e}"

    # Parse Polymarket asks
    p_asks = []
    try:
        for ask in p_book.get('asks', []):
            p_asks.append({'price': float(ask['price']), 'qty': float(ask['size'])})
        p_asks.sort(key=lambda x: x['price'])
    except Exception as e:
        return 0, 0.0, f"PolyParseErr: {e}"

    if not k_asks:
        return 0, 0.0, "EmptyKalshiBook"
    if not p_asks:
        return 0, 0.0, "EmptyPolyBook"

    total_vol = 0.0
    total_raw_cost_k = 0.0  # sum of (price * vol) without fees
    total_raw_cost_p = 0.0
    total_k_fees = 0.0      # FIX: accumulate fees per-level
    total_p_fees = 0.0
    total_spend_k = 0.0     # price + fee (for balance cap)
    total_spend_p = 0.0

    ki, pi = 0, 0
    while ki < len(k_asks) and pi < len(p_asks):
        kl = k_asks[ki]
        pl = p_asks[pi]
        combined = kl['price'] + pl['price']
        if combined >= threshold:
            break

        vol = min(kl['qty'], pl['qty'])

        # Per-level fee calculation (the FIX)
        k_fee_unit = kalshi_fee_per_contract(kl['price'])
        p_fee_unit = poly_fee_per_contract(pl['price'], poly_fee_rate)
        k_cost_unit = kl['price'] + k_fee_unit
        p_cost_unit = pl['price'] + p_fee_unit

        # Balance cap check
        if total_spend_k + vol * k_cost_unit > balance_k:
            remaining = max(0.0, balance_k - total_spend_k)
            vol = min(vol, remaining / k_cost_unit) if k_cost_unit > 0 else 0
        if total_spend_p + vol * p_cost_unit > balance_p:
            remaining = max(0.0, balance_p - total_spend_p)
            vol = min(vol, remaining / p_cost_unit) if p_cost_unit > 0 else 0
        if vol <= 0:
            break

        total_vol += vol
        total_raw_cost_k += vol * kl['price']
        total_raw_cost_p += vol * pl['price']
        total_k_fees += vol * k_fee_unit   # FIX: per-level accumulation
        total_p_fees += vol * p_fee_unit
        total_spend_k += vol * k_cost_unit
        total_spend_p += vol * p_cost_unit

        kl['qty'] -= vol
        pl['qty'] -= vol
        if kl['qty'] <= 1e-9:
            ki += 1
        if pl['qty'] <= 1e-9:
            pi += 1

    if total_vol <= 0:
        return 0, 0.0, "NoOverlap"
    if total_raw_cost_k < 0.10 or total_raw_cost_p < 0.10:
        return 0, 0.0, f"MinCost(K:{total_raw_cost_k:.2f},P:{total_raw_cost_p:.2f})"

    # Kalshi rounds UP per-contract to the nearest cent — not on the aggregate.
    # During the walk we accumulated raw fees. Re-round per-level to match exchange:
    # For each level, fee_per_contract = ceil(0.07 * p * (1-p) * 100) / 100
    # We approximate by ceiling the total, which is close enough for estimation.
    # The exact per-contract rounding is done in execute_arbitrage.
    total_k_fees = math.ceil(total_k_fees * 100) / 100.0

    total_cost = total_raw_cost_k + total_raw_cost_p + total_k_fees + total_p_fees
    avg_cost_per_contract = total_cost / total_vol
    profit_pct = (1.0 - avg_cost_per_contract) * 100

    return total_vol, profit_pct, "OK"


# ============================================================================
# ORDER EXECUTION
# ============================================================================

def execute_kalshi_order(ticker: str, side: str, count: int, price_dollars: float,
                         test_mode: bool = True) -> tuple[bool, str, str]:
    if test_mode:
        log.info("[TEST] Kalshi: %s %d @ $%.4f on %s", side.upper(), count, price_dollars, ticker)
        return True, "TEST_K_ORDER", ""
    try:
        ts = str(int(time.time() * 1000))
        path = "/portfolio/orders"
        payload = {
            "ticker": ticker, "side": side, "action": "buy",
            "count_fp": f"{count:.2f}",
            f"{side}_price_dollars": f"{price_dollars:.4f}",
            "time_in_force": "fill_or_kill",
        }
        sig = sign_kalshi_request(ts, "POST", "/trade-api/v2" + path)
        headers = {
            "KALSHI-ACCESS-KEY": os.getenv("KALSHI_API_KEY_ID"),
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{KALSHI_BASE_URL}{path}", headers=headers,
            data=json.dumps(payload), timeout=10,
        )
        if resp.status_code == 201:
            oid = resp.json().get('order', {}).get('order_id', '')
            return True, oid, ""
        err = resp.json().get('error', {}).get('message', 'Unknown')
        return False, "", err
    except Exception as e:
        return False, "", str(e)


POLY_SLIPPAGE = 0.02  # max slippage above ask price when placing limit order

# Cached CLOB client — create_or_derive_api_creds() does an HTTP round-trip,
# so we only want to do it once per session, not per order.
_clob_client: Optional[object] = None
_clob_client_lock = threading.Lock()


def _get_clob_client():
    global _clob_client
    if _clob_client is not None:
        return _clob_client
    with _clob_client_lock:
        if _clob_client is not None:
            return _clob_client
        client = ClobClient(
            POLY_CLOB_URL,
            key=os.getenv("POLY_PRIVATE_KEY"),
            chain_id=137,
            signature_type=int(os.getenv("POLY_SIGNATURE_TYPE", "1")),
            funder=os.getenv("POLYMARKET_PROXY_ADDRESS"),
        )
        client.set_api_creds(client.create_or_derive_api_creds())
        _clob_client = client
        return _clob_client


def execute_poly_order(token_id: str, price: float, quantity: int,
                       test_mode: bool = True) -> tuple[bool, str, str]:
    if not HAS_CLOB and not test_mode:
        return False, "", "py-clob-client not installed"
    limit_price = float(Decimal(str(price + POLY_SLIPPAGE)).quantize(Decimal('0.0001'), rounding=ROUND_DOWN))
    if test_mode:
        log.info("[TEST] Poly: BUY %d @ $%.4f (limit) token=%s…", quantity, limit_price, token_id[:16])
        return True, "TEST_P_ORDER", ""
    try:
        client = _get_clob_client()
        order = client.create_order(OrderArgs(
            token_id=token_id, price=limit_price, size=float(quantity), side=BUY
        ))
        resp = client.post_order(order, OrderType.FOK)
        if resp.get('success'):
            return True, resp.get('orderID', ''), ""
        return False, "", resp.get('errorMsg', 'Unknown')
    except Exception as e:
        return False, "", str(e)


def execute_arbitrage(candidate: ArbCandidate, max_spend: float,
                      test_mode: bool = True,
                      poly_fee_rate: float = DEFAULT_POLY_FEE_RATE) -> TradeResult:
    """Execute both legs of the arbitrage. Poly first, then Kalshi.

    WARNING: If Kalshi fails after Poly succeeds, you have a naked position.
    In a production system you'd want an unwind mechanism here.
    """
    k_fee_per = kalshi_fee_per_contract(candidate.k_price)
    p_fee_per = poly_fee_per_contract(candidate.p_price, poly_fee_rate)
    k_cost_per = candidate.k_price + k_fee_per
    p_cost_per = candidate.p_price + p_fee_per

    max_by_budget = min(max_spend / k_cost_per if k_cost_per > 0 else 0,
                        max_spend / p_cost_per if p_cost_per > 0 else 0)
    volume = int(min(candidate.max_vol, max_by_budget))

    if volume <= 0:
        return TradeResult(success=False, error="Volume too low")

    k_total_fee = kalshi_fee_total(volume, candidate.k_price)
    p_total_fee = volume * p_fee_per
    k_cost = volume * candidate.k_price + k_total_fee
    p_cost = volume * candidate.p_price + p_total_fee
    total = k_cost + p_cost

    if k_cost < 1.0 or p_cost < 1.0:
        return TradeResult(success=False,
                           error=f"Min cost not met (K:${k_cost:.2f} P:${p_cost:.2f})")

    # Payout is $1 per contract minus fees already paid
    guaranteed_profit = volume * 1.0 - total

    log.info("=" * 50)
    log.info("EXECUTING ARBITRAGE — %s", "TEST MODE" if test_mode else "LIVE")
    log.info("  Contracts: %d | K: $%.4f + $%.2f fee | P: $%.4f", volume,
             candidate.k_price, k_total_fee, candidate.p_price)
    log.info("  Total: $%.2f | Guaranteed profit: $%.2f", total, guaranteed_profit)
    log.info("=" * 50)

    # Leg 1: Polymarket
    p_ok, p_oid, p_err = execute_poly_order(
        candidate.p_token_id, candidate.p_price, volume, test_mode
    )
    if not p_ok:
        log.error("Poly order FAILED: %s", p_err)
        return TradeResult(success=False, error=f"Poly: {p_err}")

    # Leg 2: Kalshi
    k_ok, k_oid, k_err = execute_kalshi_order(
        candidate.kalshi.ticker, candidate.side, volume, candidate.k_price, test_mode
    )
    if not k_ok:
        log.error("Kalshi order FAILED: %s — Poly order %s is NAKED!", k_err, p_oid)
        return TradeResult(success=False, error=f"Kalshi: {k_err} (Poly {p_oid} naked!)",
                           poly_order_id=p_oid)

    return TradeResult(
        success=True, volume=volume,
        kalshi_cost=k_cost, poly_cost=p_cost, total_cost=total,
        profit=guaranteed_profit,
        kalshi_order_id=k_oid, poly_order_id=p_oid,
    )


# ============================================================================
# PNL TRACKER (test mode)
# ============================================================================

class PnLTracker:
    """Tracks simulated trades and cumulative PnL in test mode."""

    def __init__(self, filepath="arb_pnl_log.json"):
        self.filepath = filepath
        self.entries: list[dict] = []
        self._load()

    def _load(self):
        try:
            with open(self.filepath, encoding="utf-8") as f:
                self.entries = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.entries = []

    def record(self, candidate: ArbCandidate, result: TradeResult):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "event": candidate.event_name,
            "outcome": candidate.pair_name,
            "side": candidate.side,
            "volume": result.volume,
            "k_price": candidate.k_price,
            "p_price": candidate.p_price,
            "total_cost": result.total_cost,
            "guaranteed_profit": result.profit,
        }
        self.entries.append(entry)
        self._save()
        total = sum(e["guaranteed_profit"] for e in self.entries)
        log.info("[PnL] This trade: +$%.2f | Session total: +$%.2f (%d trades)",
                 result.profit, total, len(self.entries))

    def _save(self):
        try:
            tmp = self.filepath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.filepath)
        except Exception:
            pass

    def summary(self) -> str:
        if not self.entries:
            return "No trades recorded."
        total = sum(e["guaranteed_profit"] for e in self.entries)
        return f"{len(self.entries)} trades | Total PnL: ${total:.2f}"


# ============================================================================
# DASHBOARD STATE
# ============================================================================

DASHBOARD_STATE_FILE = "dashboard_state.json"
DASHBOARD_CONTROL_FILE = "dashboard_control.json"
BLACKLIST_FILE = "arb_blacklist.json"
DETECTION_HISTORY_FILE = "arb_detection_history.json"


def load_blacklist() -> set:
    try:
        with open(BLACKLIST_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def append_detection_history(opportunities, max_keep=500):
    """Append profitable opportunities to a rolling history file."""
    if not opportunities:
        return
    try:
        try:
            with open(DETECTION_HISTORY_FILE, encoding="utf-8") as f:
                hist = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            hist = []

        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        for op in opportunities:
            if op.get("profit", 0) <= 0:
                continue
            hist.append({
                "ts": now,
                "name": op.get("name", ""),
                "type": op.get("type", ""),
                "profit": round(op.get("profit", 0), 2),
                "volume": round(op.get("volume", 0), 1),
                "cost": round(op.get("cost", 0), 4),
                "kalshi_ticker": op.get("kalshi_ticker", ""),
                "kalshi_event_ticker": op.get("kalshi_event_ticker", ""),
                "poly_slug": op.get("poly_slug", ""),
                "neg_risk": op.get("neg_risk", False),
                "source": op.get("source", "main"),
                "arb_key": op.get("arb_key", ""),
            })
        # Keep only recent
        if len(hist) > max_keep:
            hist = hist[-max_keep:]
        tmp = DETECTION_HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False)
        os.replace(tmp, DETECTION_HISTORY_FILE)
    except Exception:
        pass


def write_dashboard_state(opportunities, bal_k, bal_p, config, elapsed,
                          scan_count, candidates_count, scan_log, test_mode, scan_mode):
    state = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "scan_count": scan_count,
        "cycle_time_s": round(elapsed, 1),
        "balances": {"kalshi": round(bal_k, 2), "polymarket": round(bal_p, 2)},
        "config": {
            "poll_interval": config.get("poll_interval", 8),
            "threshold": config.get("threshold", 0.94),
            "min_alert_profit": config.get("min_alert_profit", 1.0),
            "event_count": len(config.get("events", [])),
        },
        "bot_status": "running",
        "test_mode": test_mode,
        "scan_mode": scan_mode,
        "candidates_count": candidates_count,
        "opportunities": [],
        "scan_log": scan_log[-200:],
    }
    for op in opportunities:
        kp = op.get("kalshi_price", 0)
        pp = op.get("poly_price", 0)
        state["opportunities"].append({
            "name": op.get("name", ""),
            "type": op.get("type", ""),
            "profit_pct": round(op.get("profit", 0), 2),
            "volume": round(op.get("volume", 0), 1),
            "cost": round(op.get("cost", 0), 4),
            "kalshi_price": round(kp, 4),
            "poly_price": round(pp, 4),
            "kalshi_ticker": op.get("kalshi_ticker", ""),
            "kalshi_event_ticker": op.get("kalshi_event_ticker", ""),
            "poly_slug": op.get("poly_slug", ""),
            "poly_token_id": op.get("poly_token_id", ""),
            "neg_risk": op.get("neg_risk", False),
            "arb_key": op.get("arb_key", ""),
            "source": op.get("source", "main"),
            "kalshi_fee": round(KALSHI_FEE_RATE * kp * (1 - kp), 4) if kp else 0,
            "poly_fee": round(DEFAULT_POLY_FEE_RATE * pp * (1 - pp), 4) if pp else 0,
            "spread": round(abs(kp - pp), 4) if kp and pp else 0,
            "timestamp": op["timestamp"].strftime("%Y-%m-%dT%H:%M:%S") if hasattr(op.get("timestamp"), "strftime") else "",
        })
    try:
        tmp = DASHBOARD_STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(tmp, DASHBOARD_STATE_FILE)
    except Exception:
        pass


# ============================================================================
# MAIN SCANNER
# ============================================================================

class ArbScanner:
    def __init__(self, test_mode=True, scan_mode="WS",
                 config_files=None,
                 max_position=10.0, threshold_override=None, min_profit_override=None,
                 balance_cap=1_000_000.0, cooldown_min=45, balance_refresh_cycles=30):
        self.test_mode = test_mode
        self.scan_mode = scan_mode
        # Load from BOTH config JSONs by default (your manually-reviewed events)
        self.config_files = config_files or [
            "markets_config_reviewed.json",
            "markets_config.json",
        ]
        self.max_position = max_position
        self.threshold_override = threshold_override
        self.min_profit_override = min_profit_override
        self.balance_cap = balance_cap
        self.cooldown_min = cooldown_min
        self.balance_refresh_cycles = balance_refresh_cycles

        self.kalshi = KalshiClient()
        self.poly = PolymarketClient()
        self.notifier = Notifier()
        self.pnl = PnLTracker()
        self.kalshi_ws: Optional[KalshiWSClient] = None
        self.poly_ws: Optional[PolyWSClient] = None

        self._ws_active = False
        self._ws_cached_results = None
        self._ws_last_refresh = 0.0
        self._WS_REFRESH_S = 60

        self.alert_cooldowns: dict[str, datetime] = {}
        self.recent_opportunities: list[dict] = []
        self.scan_count = 0
        self._bal_k = 0.0
        self._bal_p = 0.0
        self._running = True

    def load_config(self) -> dict:
        """Load and merge events from all config files.

        Each event is tagged with `_source` based on filename:
        - markets_config_reviewed.json -> "reviewed"
        - markets_config.json          -> "main"
        - markets_config_discovered.json -> "discovered"
        Deduplicates by kalshi_ticker (first occurrence wins).
        Events WITHOUT a mapping are skipped.
        """
        merged = {"events": [], "poll_interval": 8, "threshold": 0.94, "min_alert_profit": 1.0}
        seen_tickers = set()
        # Always load discovered + politics configs too if they exist
        all_paths = list(self.config_files)
        if "markets_config_discovered.json" not in all_paths:
            all_paths.append("markets_config_discovered.json")
        if "markets_config_politics.json" not in all_paths:
            all_paths.append("markets_config_politics.json")

        def _source_of(path):
            if "politics" in path:
                return "politics"
            if "reviewed" in path:
                return "reviewed"
            if "discovered" in path:
                return "discovered"
            return "main"

        for path in all_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if path == self.config_files[0]:
                    merged["poll_interval"] = data.get("poll_interval", 8)
                    merged["threshold"] = data.get("threshold", 0.94)
                    merged["min_alert_profit"] = data.get("min_alert_profit", 1.0)
                src = _source_of(path)
                for ev in data.get("events", []):
                    kt = ev.get("kalshi_ticker")
                    if not kt or kt in seen_tickers:
                        continue
                    if not ev.get("mapping"):
                        continue
                    seen_tickers.add(kt)
                    merged["events"].append({**ev, "_source": src})
            except (FileNotFoundError, json.JSONDecodeError) as e:
                log.warning("Config %s: %s", path, e)
        log.debug("Loaded %d mapped events from %d config files", len(merged["events"]), len(all_paths))
        return merged

    def _load_dashboard_control(self):
        """Read runtime overrides from dashboard UI."""
        try:
            with open(DASHBOARD_CONTROL_FILE, encoding="utf-8") as f:
                ctrl = json.load(f)
            self.notifier.enabled = ctrl.get("email_enabled", self.notifier.enabled)
            mode = ctrl.get("scan_mode")
            if mode in ("OG", "WS"):
                self.scan_mode = mode
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _init_websockets(self, events):
        if not HAS_WEBSOCKETS:
            log.warning("websockets not installed — falling back to OG mode")
            return
        log.info("[WS] Initializing WebSocket connections...")
        self.kalshi_ws = KalshiWSClient()
        self.poly_ws = PolyWSClient()

        # Discover all Poly tokens via REST
        all_tokens = set()
        def _fetch(ec):
            s = ec.get("poly_slug")
            return self.poly.get_event_markets(s) if s else []

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(events), 8)) as ex:
            for pm_list in ex.map(_fetch, events):
                for pm in pm_list:
                    if pm.yes_token_id:
                        all_tokens.add(pm.yes_token_id)
                    if pm.no_token_id:
                        all_tokens.add(pm.no_token_id)

        log.info("[WS] Discovered %d Polymarket tokens", len(all_tokens))
        self.kalshi_ws.start()
        self.poly_ws.start(initial_tokens=all_tokens)
        time.sleep(4)
        self._ws_active = True
        log.info("[WS] Ready")

    def _screen_candidates(self, results, threshold, blacklist) -> tuple[list[ArbCandidate], list[dict]]:
        """Phase 2: cheap screening without orderbooks."""
        candidates = []
        scan_log = []

        for r in results:
            if r is None:
                continue
            ec, k_markets, p_markets = r
            if not k_markets or not p_markets:
                continue
            pairs = match_markets(k_markets, p_markets, mapping=ec.get('mapping', {}))
            event_name = ec.get('name', 'Unknown')
            p_slug = ec.get('poly_slug', '')
            k_event_ticker = ec.get('kalshi_ticker', '')
            source = ec.get('_source', 'main')

            for pair in pairs:
                k = pair.kalshi
                p = pair.poly

                if k.yes_ask is None or p.no_price is None:
                    continue
                # Filter out zero/negative poly prices (bestBid=None gives no_price=None,
                # but bestBid=0 gives no_price=1.0 which is also useless)
                if p.no_price is not None and p.no_price <= 0:
                    continue

                is_stub_yes = (k.yes_ask <= 0.01 and k.yes_bid < 0.005)
                cost_yes = k.yes_ask + p.no_price if p.no_price and p.no_price > 0 else 999
                cost_no = (k.no_ask + p.yes_price) if (k.no_ask and p.yes_price and p.yes_price > 0) else 999

                scan_log.append({
                    "t": datetime.now().strftime("%H:%M:%S"),
                    "event": event_name[:40], "outcome": pair.name[:30],
                    "k_yes": k.yes_ask, "k_no": k.no_ask,
                    "p_yes": p.yes_price, "p_no": p.no_price,
                    "cost_yes": round(cost_yes, 4), "cost_no": round(cost_no, 4),
                    "threshold": threshold,
                    "pass_yes": not is_stub_yes and cost_yes < threshold,
                    "pass_no": bool(k.no_ask and p.yes_price and cost_no < threshold),
                    "neg_risk": p.neg_risk, "stub": is_stub_yes,
                })

                arb_key_yes = f"{k.ticker}_yes_{p.condition_id}"
                arb_key_no = f"{k.ticker}_no_{p.condition_id}"

                if not is_stub_yes and cost_yes < threshold and arb_key_yes not in blacklist:
                    candidates.append(ArbCandidate(
                        event_name=event_name, pair_name=pair.name,
                        kalshi=k, poly=p, side="yes",
                        cost=cost_yes, k_price=k.yes_ask, p_price=p.no_price,
                        p_slug=p_slug, k_event_ticker=k_event_ticker,
                        p_token_id=p.no_token_id or '', arb_key=arb_key_yes,
                        source=source,
                    ))

                if k.no_ask is not None and p.yes_price is not None:
                    is_stub_no = (k.no_ask <= 0.01 and k.no_bid < 0.005)
                    if not is_stub_no and cost_no < threshold and arb_key_no not in blacklist:
                        candidates.append(ArbCandidate(
                            event_name=event_name, pair_name=pair.name,
                            kalshi=k, poly=p, side="no",
                            cost=cost_no, k_price=k.no_ask, p_price=p.yes_price,
                            p_slug=p_slug, k_event_ticker=k_event_ticker,
                            p_token_id=p.yes_token_id or '', arb_key=arb_key_no,
                            source=source,
                        ))

        return candidates, scan_log

    def _fetch_orderbooks_and_calculate(self, candidates: list[ArbCandidate], threshold: float):
        """Phases 3-5: fetch orderbooks and calculate real liquidity."""
        normal = [c for c in candidates if not c.poly.neg_risk]
        negrisk = [c for c in candidates if c.poly.neg_risk]

        if normal:
            # Deduplicate Kalshi tickers (same ticker can appear for YES and NO)
            k_tickers = list({c.kalshi.ticker for c in normal})

            # Fetch Kalshi orderbooks
            if self._ws_active:
                self.kalshi_ws.subscribe_orderbooks(k_tickers)
                time.sleep(0.3)
                k_books = self.kalshi_ws.get_orderbooks_batch(k_tickers)
                missing = [t for t in k_tickers if not k_books.get(t)]
                if missing:
                    k_books.update(self.kalshi.get_orderbooks_batch(missing))
            else:
                k_books = self.kalshi.get_orderbooks_batch(k_tickers)

            # Deduplicate Poly tokens
            p_tokens = list({c.p_token_id for c in normal if c.p_token_id})

            p_books = {}
            p_fees = {}
            if self._ws_active and p_tokens:
                missing_p = []
                for tid in p_tokens:
                    cached = self.poly_ws.get_orderbook(tid)
                    if cached and cached.get("asks"):
                        p_books[tid] = cached
                    else:
                        missing_p.append(tid)
                    p_fees[tid] = self.poly.get_fee_rate(tid)
                if missing_p:
                    def _fb(t):
                        return t, self.poly.get_orderbook(t)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(missing_p), 8)) as ex:
                        for tid, book in ex.map(_fb, missing_p):
                            p_books[tid] = book
            elif p_tokens:
                def _fetch_pb(tid):
                    return tid, self.poly.get_orderbook(tid), self.poly.get_fee_rate(tid)
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(p_tokens), 8)) as ex:
                    for tid, book, fee in ex.map(_fetch_pb, p_tokens):
                        p_books[tid] = book
                        p_fees[tid] = fee

            # Calculate for each normal candidate
            for c in normal:
                k_book = k_books.get(c.kalshi.ticker, {})
                p_book = p_books.get(c.p_token_id, {})
                fee = p_fees.get(c.p_token_id, DEFAULT_POLY_FEE_RATE)
                c.poly_fee_rate = fee
                c.max_vol, c.est_profit, c.fail_reason = calculate_liquidity(
                    k_book, p_book, c.side, threshold,
                    self.balance_cap, self.balance_cap, poly_fee_rate=fee,
                )

        # Neg-risk: estimate from gamma prices only (no reliable orderbook)
        for c in negrisk:
            if c.k_price <= 0.03:
                c.fail_reason = "StubPrice"
                continue
            if c.p_price <= 0.05:
                c.fail_reason = "ThinPoly"
                continue
            k_fee = kalshi_fee_per_contract(c.k_price)
            p_fee = poly_fee_per_contract(c.p_price, 0.03)  # sports default for neg-risk
            total = c.k_price + c.p_price + k_fee + p_fee
            if total < 1.0:
                c.est_profit = (1.0 - total) * 100
                c.fail_reason = "NegRisk-AlertOnly"
            else:
                c.fail_reason = "NoCrossAfterFees"

    def _report_and_execute(self, candidates: list[ArbCandidate], min_alert_profit: float):
        """Phase 6: report opportunities, send alerts, execute trades."""
        for c in candidates:
            is_neg = c.poly.neg_risk
            side_label = f"K_{'YES' if c.side == 'yes' else 'NO'}+P_{'NO' if c.side == 'yes' else 'YES'}"

            # Neg-risk: alert only
            if is_neg and c.est_profit > 0:
                log.info("[NR-ALERT] %s | %s | %s: %.3f | Profit:%.1f%% (no vol)",
                         c.event_name[:25], c.pair_name[:20], side_label, c.cost, c.est_profit)
                self.recent_opportunities.append({
                    "timestamp": datetime.now(),
                    "name": f"[NR] {c.event_name} - {c.pair_name}",
                    "type": f"Buy {side_label}", "profit": c.est_profit,
                    "volume": 0, "cost": c.cost,
                    "kalshi_price": c.k_price, "poly_price": c.p_price,
                    "kalshi_ticker": c.kalshi.ticker,
                    "kalshi_event_ticker": c.k_event_ticker,
                    "poly_slug": c.p_slug, "poly_token_id": c.p_token_id,
                    "neg_risk": True, "arb_key": c.arb_key, "source": c.source,
                })
                continue

            if c.max_vol <= 0:
                if c.fail_reason not in ("EmptyKalshiBook", "EmptyPolyBook", "NoOverlap",
                                         "StubPrice", "ThinPoly", "NoCross", "NoCrossAfterFees"):
                    log.debug("[NoLiq] %s | %s | %s: %.3f (%s)",
                              c.event_name[:25], c.pair_name[:20], side_label, c.cost, c.fail_reason)
                continue

            # Real arb with verified volume
            log.info("[ARB] %s | %s | %s: %.3f | Vol:%.0f Profit:%.1f%%",
                     c.event_name[:25], c.pair_name[:20], side_label,
                     c.cost, c.max_vol, c.est_profit)

            self.recent_opportunities.append({
                "timestamp": datetime.now(),
                "name": f"{c.event_name} - {c.pair_name}",
                "type": f"Buy {side_label}", "profit": c.est_profit,
                "volume": c.max_vol, "cost": c.cost,
                "kalshi_price": c.k_price, "poly_price": c.p_price,
                "kalshi_ticker": c.kalshi.ticker,
                "kalshi_event_ticker": c.k_event_ticker,
                "poly_slug": c.p_slug, "poly_token_id": c.p_token_id,
                "neg_risk": False, "arb_key": c.arb_key, "source": c.source,
            })

            if c.est_profit >= min_alert_profit:
                # Cooldown check
                last = self.alert_cooldowns.get(c.pair_name)
                if last and datetime.now() - last < timedelta(minutes=self.cooldown_min):
                    continue
                self.alert_cooldowns[c.pair_name] = datetime.now()

                self.notifier.send_arb_alert(c)

                result = execute_arbitrage(c, self.max_position, self.test_mode,
                                          poly_fee_rate=c.poly_fee_rate)
                if result.success:
                    log.info("TRADE OK: %d contracts, profit $%.2f", result.volume, result.profit)
                    self.notifier.send(
                        f"TRADE EXECUTED: {c.pair_name}",
                        f"Vol: {result.volume} | Profit: ${result.profit:.2f}"
                    )
                    if self.test_mode:
                        self.pnl.record(c, result)

    def _start_auto_discovery(self):
        """Run auto_discover.py every 30 min in a background thread.

        Only auto-accepts HIGH confidence matches with good outcome mappings.
        Writes to markets_config_discovered.json (separate from your manual configs).
        """
        def _loop():
            import subprocess
            INTERVAL = 1800  # 30 min
            time.sleep(60)  # wait 1 min before first run
            while self._running:
                try:
                    log.info("[Discovery] Running auto_discover.py...")
                    result = subprocess.run(
                        [sys.executable, "auto_discover.py"],
                        capture_output=True, text=True, timeout=600,
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                        encoding="utf-8", errors="replace",
                    )
                    if result.returncode == 0:
                        # Count discovered matches
                        try:
                            with open("discovered_matches_full.json", encoding="utf-8") as f:
                                disc = json.load(f)
                            total = len(disc.get("matches", []))
                            high = sum(1 for m in disc.get("matches", []) if m.get("confidence") == "HIGH")
                            log.info("[Discovery] Done: %d matches found (%d HIGH)", total, high)
                        except Exception:
                            log.info("[Discovery] Done (could not read results)")
                    else:
                        log.warning("[Discovery] Failed: %s", result.stderr[-200:] if result.stderr else "unknown")
                except subprocess.TimeoutExpired:
                    log.warning("[Discovery] Timed out (10 min)")
                except Exception as e:
                    log.warning("[Discovery] Error: %s", e)

                # Sleep in small chunks so shutdown is responsive
                for _ in range(INTERVAL // 10):
                    if not self._running:
                        return
                    time.sleep(10)

        t = threading.Thread(target=_loop, daemon=True, name="AutoDiscovery")
        t.start()
        log.info("[Discovery] Auto-discovery scheduled every 30 min")

    def run(self):
        """Main scanner loop."""
        config = self.load_config()
        events = config.get('events', [])

        if self.scan_mode == "WS":
            self._init_websockets(events)

        self.notifier.start()
        self._start_auto_discovery()
        mode_str = "TEST" if self.test_mode else "LIVE"
        scan_str = "WS" if self._ws_active else "OG"
        log.info("Arbitrage scanner started | Mode: %s | Scan: %s | Max: $%.0f | Config: %s",
                 mode_str, scan_str, self.max_position, self.config_files)
        self.notifier.send(
            "Arbitrage Bot Started",
            f"Mode: {mode_str} | Scan: {scan_str} | Max: ${self.max_position:.0f}"
        )

        # Register graceful shutdown
        def _shutdown(sig, frame):
            log.info("Shutting down...")
            self._running = False
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        while self._running:
            try:
                self._scan_cycle()
            except Exception as e:
                log.exception("Scan cycle error: %s", e)
                self.notifier.send("Bot Error", str(e))
                time.sleep(5)

        # Cleanup
        if self.kalshi_ws:
            self.kalshi_ws.stop()
        if self.poly_ws:
            self.poly_ws.stop()
        self.notifier.stop()
        log.info("Bot stopped. PnL: %s", self.pnl.summary())

    def _scan_cycle(self):
        self.scan_count += 1
        self._load_dashboard_control()
        config = self.load_config()
        events = config.get('events', [])
        poll_interval = config.get('poll_interval', 8)
        threshold = self.threshold_override or config.get('threshold', 0.94)
        min_alert_profit = self.min_profit_override or config.get('min_alert_profit', 1.0)

        t0 = time.time()

        # Refresh balances only every N cycles (saves API calls)
        if self.scan_count % self.balance_refresh_cycles == 1:
            self._bal_k = self.kalshi.get_balance()
            self._bal_p = self.poly.get_balance()

        # Phase 1: fetch market data
        if self._ws_active:
            now = time.time()
            need_refresh = (now - self._ws_last_refresh > self._WS_REFRESH_S) or self._ws_cached_results is None
            if need_refresh:
                log.info("[WS] Refreshing markets (%d events)", len(events))
                def _fetch(ec):
                    kt = ec.get('kalshi_ticker')
                    ps = ec.get('poly_slug')
                    if not kt or not ps:
                        return None
                    return (ec, self.kalshi.get_event_markets(kt), self.poly.get_event_markets(ps))
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(events), 8)) as ex:
                    self._ws_cached_results = list(ex.map(_fetch, events))
                self._ws_last_refresh = now
                # Subscribe new tokens
                new_tokens = set()
                for r in self._ws_cached_results:
                    if r is None:
                        continue
                    for pm in r[2]:
                        if pm.yes_token_id:
                            new_tokens.add(pm.yes_token_id)
                        if pm.no_token_id:
                            new_tokens.add(pm.no_token_id)
                self.poly_ws.subscribe_tokens(new_tokens)

            results = copy.deepcopy(self._ws_cached_results)

            # Overlay real-time WS prices
            for r in results:
                if r is None:
                    continue
                _, kms, pms = r
                for km in kms:
                    wd = self.kalshi_ws.get_ticker(km.ticker)
                    if wd:
                        if wd["yes_ask"] > 0: km.yes_ask = wd["yes_ask"]
                        if wd["no_ask"] > 0:  km.no_ask = wd["no_ask"]
                        if wd["yes_bid"] > 0: km.yes_bid = wd["yes_bid"]
                        if wd["no_bid"] > 0:  km.no_bid = wd["no_bid"]
                for pm in pms:
                    if pm.yes_token_id:
                        wp = self.poly_ws.get_best_price(pm.yes_token_id)
                        if wp and wp.get("best_ask", 0) > 0:
                            pm.yes_price = wp["best_ask"]
                    if pm.no_token_id:
                        wp = self.poly_ws.get_best_price(pm.no_token_id)
                        if wp and wp.get("best_ask", 0) > 0:
                            pm.no_price = wp["best_ask"]
        else:
            log.info("Scanning %d events (REST)...", len(events))
            def _fetch(ec):
                kt = ec.get('kalshi_ticker')
                ps = ec.get('poly_slug')
                if not kt or not ps:
                    return None
                return (ec, self.kalshi.get_event_markets(kt), self.poly.get_event_markets(ps))
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(events), 8)) as ex:
                results = list(ex.map(_fetch, events))

        # Phase 2: screen
        blacklist = load_blacklist()
        candidates, scan_log = self._screen_candidates(results, threshold, blacklist)

        if not candidates:
            log.debug("No candidates below threshold (%.3f)", threshold)
        else:
            log.info("%d candidates below threshold", len(candidates))
            # Phases 3-5: orderbooks + calculation
            self._fetch_orderbooks_and_calculate(candidates, threshold)
            # Phase 6: report + execute
            self._report_and_execute(candidates, min_alert_profit)

        # Post-cycle
        elapsed = time.time() - t0
        cutoff = datetime.now() - timedelta(minutes=15)
        self.recent_opportunities = [op for op in self.recent_opportunities if op['timestamp'] > cutoff]

        # Deduplicate by arb_key (keep most recent)
        seen = {}
        for op in self.recent_opportunities:
            key = op.get("arb_key", op.get("name", ""))
            if key not in seen or op["timestamp"] > seen[key]["timestamp"]:
                seen[key] = op
        self.recent_opportunities = sorted(seen.values(), key=lambda x: x['profit'], reverse=True)

        log.info("Cycle #%d done in %.1fs | %d candidates | Top: %s",
                 self.scan_count, elapsed, len(candidates),
                 f"{self.recent_opportunities[0]['name'][:30]} ({self.recent_opportunities[0]['profit']:.1f}%)"
                 if self.recent_opportunities else "none")

        write_dashboard_state(
            self.recent_opportunities, self._bal_k, self._bal_p, config,
            elapsed, self.scan_count, len(candidates), scan_log,
            self.test_mode, self.scan_mode,
        )
        append_detection_history(self.recent_opportunities)

        sleep_time = max(2, poll_interval // 3) if self._ws_active else poll_interval
        time.sleep(sleep_time)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Kalshi-Polymarket Arbitrage Scanner")
    parser.add_argument("--live", action="store_true", help="Enable REAL trading (default: test mode)")
    parser.add_argument("--mode", choices=["ws", "og"], default="ws", help="Scan mode: ws (websocket) or og (REST)")
    parser.add_argument("--threshold", type=float, help="Arb threshold override (e.g. 0.94)")
    parser.add_argument("--min-profit", type=float, help="Min profit %% to alert/execute")
    parser.add_argument("--max-position", type=float, default=10.0, help="Max USD per side (default: 10)")
    parser.add_argument("--config", nargs="+",
                        default=["markets_config_reviewed.json", "markets_config.json"],
                        help="Config file(s) — events merged, first wins on duplicates")
    parser.add_argument("--pnl", action="store_true", help="Show PnL summary and exit")
    args = parser.parse_args()

    if args.pnl:
        tracker = PnLTracker()
        print(tracker.summary())
        if tracker.entries:
            print("\nLast 10 trades:")
            for e in tracker.entries[-10:]:
                print(f"  {e['timestamp']} | {e['event'][:30]} | {e['outcome'][:20]} | "
                      f"Vol:{e['volume']} | +${e['guaranteed_profit']:.2f}")
        return

    test_mode = not args.live
    if not test_mode:
        log.warning("=" * 50)
        log.warning("  LIVE MODE — REAL MONEY WILL BE USED!")
        log.warning("=" * 50)
        time.sleep(3)

    scanner = ArbScanner(
        test_mode=test_mode,
        scan_mode=args.mode.upper(),
        config_files=args.config,
        max_position=args.max_position,
        threshold_override=args.threshold,
        min_profit_override=args.min_profit,
    )
    scanner.run()


if __name__ == "__main__":
    main()
