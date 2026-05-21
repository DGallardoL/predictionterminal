"""Sidecar that refreshes microstructure caches on a loop.

  - VPIN_bulk on Binance BTCUSDT every 5 min (rebuilds vpin_status.json)
  - Roll spread on all active arb pairs every 30 min (rebuilds
    roll_spread_cache.json + arb_engine blacklist).

The arb engine reads both files at scan time, so the engine doesn't need to
do these fetches itself.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_vpin_status as vpin_mod
import build_roll_filter as roll_mod

VPIN_INTERVAL_S = int(os.environ.get("PFM_VPIN_INTERVAL_S", "300"))   # 5 min
ROLL_INTERVAL_S = int(os.environ.get("PFM_ROLL_INTERVAL_S", "1800"))  # 30 min


def main() -> None:
    last_roll = 0.0
    while True:
        t = time.time()
        try:
            vpin_mod.main()
        except Exception as e:
            print(f"[sidecar] vpin error: {e}", flush=True)
        if t - last_roll >= ROLL_INTERVAL_S:
            try:
                roll_mod.build_cache(verbose=False)
                last_roll = t
                print("[sidecar] roll cache refreshed", flush=True)
            except Exception as e:
                print(f"[sidecar] roll error: {e}", flush=True)
        time.sleep(VPIN_INTERVAL_S)


if __name__ == "__main__":
    main()
