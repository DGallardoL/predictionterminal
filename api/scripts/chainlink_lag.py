"""Measure lag between Binance spot BTC and Chainlink BTC/USD onchain feed.

Polymarket BTC up/down 5m / 15m markets resolve via Chainlink BTC/USD. The on-chain
aggregator (0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c, 8 decimals) updates roughly
every heartbeat (~1 hour) OR on a 0.5% deviation. The off-chain Data Streams feed at
data.chain.link updates ~1s. We poll both, alongside Binance bookTicker, to estimate
how much Binance leads Chainlink.

Output: /tmp/chainlink_lag.json
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import statistics
import time
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import websockets

# ---- Constants -------------------------------------------------------------

CHAINLINK_AGG = "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c"  # BTC/USD on Ethereum
LATEST_ROUND_DATA_SELECTOR = "0xfeaf968c"  # latestRoundData()
ETH_RPCS = [
    "https://eth.llamarpc.com",
    "https://ethereum-rpc.publicnode.com",
    "https://rpc.ankr.com/eth",
]
CHAINLINK_OFFCHAIN_URL = "https://data.chain.link/api/streams/v1/feeds/btc-usd"
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
RUN_SECONDS = 240
ONCHAIN_POLL_INTERVAL = 5.0  # seconds between RPC polls
OFFCHAIN_POLL_INTERVAL = 1.0  # seconds between off-chain HTTP polls


@dataclass
class Sample:
    ts_ms: int
    source: str
    price: float


@dataclass
class State:
    samples: list[Sample] = field(default_factory=list)
    onchain_round_seen: set[int] = field(default_factory=set)
    offchain_last_price: float | None = None


# ---- Binance ---------------------------------------------------------------


async def binance_ws_consumer(state: State, deadline: float) -> None:
    """Subscribe to Binance bookTicker, store mid prices."""
    while time.time() < deadline:
        try:
            async with websockets.connect(BINANCE_WS, ping_interval=20) as ws:
                while time.time() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    msg = json.loads(raw)
                    bid = float(msg["b"])
                    ask = float(msg["a"])
                    mid = (bid + ask) / 2.0
                    state.samples.append(
                        Sample(ts_ms=int(time.time() * 1000), source="binance", price=mid)
                    )
        except Exception as exc:
            print(f"[binance] reconnecting after error: {exc}")
            await asyncio.sleep(1)


# ---- Chainlink onchain via JSON-RPC ----------------------------------------


def _decode_latest_round_data(hex_result: str) -> dict[str, Any]:
    """Decode the ABI-encoded tuple returned by latestRoundData()."""
    # Strip 0x; expect 5 * 32 bytes = 320 hex chars
    h = hex_result[2:] if hex_result.startswith("0x") else hex_result
    if len(h) < 64 * 5:
        raise ValueError(f"short response: {hex_result}")

    def word(i: int) -> int:
        return int(h[i * 64 : (i + 1) * 64], 16)

    def word_signed(i: int) -> int:
        u = word(i)
        if u >= 1 << 255:
            u -= 1 << 256
        return u

    round_id = word(0)
    answer = word_signed(1)
    started_at = word(2)
    updated_at = word(3)
    answered_in_round = word(4)
    return {
        "round_id": round_id,
        "answer": answer,
        "started_at": started_at,
        "updated_at": updated_at,
        "answered_in_round": answered_in_round,
    }


async def _rpc_call(client: httpx.AsyncClient, rpc: str) -> dict[str, Any] | None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [
            {"to": CHAINLINK_AGG, "data": LATEST_ROUND_DATA_SELECTOR},
            "latest",
        ],
    }
    try:
        r = await client.post(rpc, json=payload, timeout=8)
        r.raise_for_status()
        data = r.json()
        if "result" not in data:
            return None
        return _decode_latest_round_data(data["result"])
    except Exception as exc:
        print(f"[onchain] {rpc} error: {exc}")
        return None


async def chainlink_onchain_poller(state: State, deadline: float) -> None:
    async with httpx.AsyncClient() as client:
        rpc_idx = 0
        while time.time() < deadline:
            rpc = ETH_RPCS[rpc_idx % len(ETH_RPCS)]
            decoded = await _rpc_call(client, rpc)
            if decoded is None:
                rpc_idx += 1
                await asyncio.sleep(1)
                continue
            price = decoded["answer"] / 1e8
            now_ms = int(time.time() * 1000)
            updated_at_ms = decoded["updated_at"] * 1000
            round_id = decoded["round_id"]
            # Always record the polled value
            state.samples.append(Sample(ts_ms=now_ms, source="chainlink_onchain_poll", price=price))
            # Record only NEW rounds (when updated_at advances) — that's the actual feed update
            if round_id not in state.onchain_round_seen:
                state.onchain_round_seen.add(round_id)
                state.samples.append(
                    Sample(
                        ts_ms=updated_at_ms,
                        source="chainlink_onchain_update",
                        price=price,
                    )
                )
                print(
                    f"[onchain] new round {round_id} price=${price:,.2f} "
                    f"updated_at={updated_at_ms} (lag-from-now={now_ms - updated_at_ms}ms)"
                )
            await asyncio.sleep(ONCHAIN_POLL_INTERVAL)


# ---- Chainlink off-chain (Data Streams dashboard) --------------------------


async def chainlink_offchain_poller(state: State, deadline: float) -> None:
    """Poll the public Chainlink data dashboard. Best-effort — endpoint may 403.

    The dashboard uses an internal API; we try a couple of likely endpoints. If
    they don't work, the script still produces useful onchain numbers.
    """
    candidates = [
        "https://data.chain.link/api/streams/v1/feeds/btc-usd",
        "https://reference-data-directory.vercel.app/feeds-mainnet.json",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 chainlink-lag-probe",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(headers=headers) as client:
        # Find a working endpoint once
        working = None
        for url in candidates:
            try:
                r = await client.get(url, timeout=8)
                if r.status_code == 200:
                    working = url
                    print(f"[offchain] using {url}")
                    break
                else:
                    print(f"[offchain] {url} -> {r.status_code}")
            except Exception as exc:
                print(f"[offchain] {url} error: {exc}")
        if working is None:
            print("[offchain] no working endpoint, skipping off-chain stream")
            return

        while time.time() < deadline:
            try:
                r = await client.get(working, timeout=8)
                if r.status_code != 200:
                    await asyncio.sleep(2)
                    continue
                data = r.json()
                # Try several shapes
                price = None
                if isinstance(data, dict):
                    for k in ("price", "answer", "value", "latestPrice"):
                        if k in data:
                            with contextlib.suppress(ValueError, TypeError):
                                price = float(data[k])
                if price is not None and price != state.offchain_last_price:
                    state.offchain_last_price = price
                    state.samples.append(
                        Sample(
                            ts_ms=int(time.time() * 1000),
                            source="chainlink_offchain",
                            price=price,
                        )
                    )
            except Exception as exc:
                print(f"[offchain] poll error: {exc}")
            await asyncio.sleep(OFFCHAIN_POLL_INTERVAL)


# ---- Lag estimation --------------------------------------------------------


def estimate_lag(state: State) -> dict[str, Any]:
    """For each chainlink update, find the Binance sample closest in price BEFORE
    the chainlink timestamp. The time delta is the lead time of Binance.

    Method: sliding nearest-price match. For each onchain update event, scan
    Binance samples in the prior 120s window and find the one whose price is
    closest to the new chainlink price. Report (chainlink_ts - binance_ts) ms.
    """
    binance = sorted([s for s in state.samples if s.source == "binance"], key=lambda s: s.ts_ms)
    onchain_updates = sorted(
        [s for s in state.samples if s.source == "chainlink_onchain_update"],
        key=lambda s: s.ts_ms,
    )
    offchain = sorted(
        [s for s in state.samples if s.source == "chainlink_offchain"],
        key=lambda s: s.ts_ms,
    )

    def lead_for_event(event: Sample, max_window_ms: int = 120_000) -> int | None:
        # Find Binance sample in window with closest price to event.price
        lo = event.ts_ms - max_window_ms
        candidates = [b for b in binance if lo <= b.ts_ms <= event.ts_ms]
        if not candidates:
            return None
        best = min(candidates, key=lambda b: abs(b.price - event.price))
        return event.ts_ms - best.ts_ms

    onchain_leads = [lead_for_event(e) for e in onchain_updates if lead_for_event(e) is not None]
    offchain_leads = [lead_for_event(e) for e in offchain if lead_for_event(e) is not None]

    # Onchain heartbeat / update interval
    onchain_intervals_s = []
    for a, b in pairwise(onchain_updates):
        onchain_intervals_s.append((b.ts_ms - a.ts_ms) / 1000.0)

    def _summary(xs: list[int]) -> dict[str, Any]:
        if not xs:
            return {"n": 0}
        xs_sorted = sorted(xs)
        return {
            "n": len(xs_sorted),
            "median_ms": statistics.median(xs_sorted),
            "mean_ms": statistics.mean(xs_sorted),
            "p95_ms": xs_sorted[int(0.95 * (len(xs_sorted) - 1))],
            "min_ms": xs_sorted[0],
            "max_ms": xs_sorted[-1],
        }

    return {
        "n_binance_updates": len(binance),
        "n_chainlink_onchain_updates": len(onchain_updates),
        "n_chainlink_offchain_updates": len(offchain),
        "mean_chainlink_update_interval_s": (
            statistics.mean(onchain_intervals_s) if onchain_intervals_s else None
        ),
        "onchain_update_intervals_s": onchain_intervals_s,
        "lag_estimates_binance_to_onchain_ms": onchain_leads,
        "lag_estimates_binance_to_offchain_ms": offchain_leads,
        "summary": {
            "binance_to_onchain": _summary(onchain_leads),
            "binance_to_offchain": _summary(offchain_leads),
        },
    }


# ---- Main ------------------------------------------------------------------


async def main() -> None:
    state = State()
    deadline = time.time() + RUN_SECONDS
    print(f"Running for {RUN_SECONDS}s. Aggregator={CHAINLINK_AGG}")

    tasks = [
        asyncio.create_task(binance_ws_consumer(state, deadline)),
        asyncio.create_task(chainlink_onchain_poller(state, deadline)),
        asyncio.create_task(chainlink_offchain_poller(state, deadline)),
    ]
    try:
        await asyncio.wait(tasks, timeout=RUN_SECONDS + 10)
    finally:
        for t in tasks:
            t.cancel()

    result = estimate_lag(state)
    out_path = "/tmp/chainlink_lag.json"
    with Path(out_path).open("w") as f:  # noqa: ASYNC230
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print(json.dumps(result["summary"], indent=2))
    print(
        f"binance_samples={result['n_binance_updates']} "
        f"onchain_updates={result['n_chainlink_onchain_updates']} "
        f"offchain_updates={result['n_chainlink_offchain_updates']}"
    )


if __name__ == "__main__":
    asyncio.run(main())
