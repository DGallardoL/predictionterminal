"""
A/B harness for the auto_discover matcher.

Workflow:
    python harness.py snapshot         # fetch Kalshi+Poly once, save to /tmp/arb-snapshot/
    python harness.py run --tag old    # run match_events on snapshot, save as matches_old.json
    python harness.py run --tag new    # run again after changes, save as matches_new.json
    python harness.py diff             # compare the two

The snapshot is the slow part (~60s). Once frozen we can iterate on the matcher and
compare apples-to-apples.
"""

import json
import sys
import time
from pathlib import Path

import auto_discover as ad

SNAP_DIR = Path("/tmp/arb-snapshot")


def snapshot():
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching Kalshi events...", flush=True)
    k = ad.fetch_kalshi_events()
    print("Fetching Poly events...", flush=True)
    p = ad.fetch_poly_events(newest_first=True)
    (SNAP_DIR / "k_events.json").write_text(json.dumps(k))
    (SNAP_DIR / "p_events.json").write_text(json.dumps(p))
    print(f"Snapshot saved: {len(k)} Kalshi, {len(p)} Poly -> {SNAP_DIR}")


def load_snapshot():
    k = json.loads((SNAP_DIR / "k_events.json").read_text())
    p = json.loads((SNAP_DIR / "p_events.json").read_text())
    return k, p


def run(tag: str, min_score: float = 0.65):
    k, p = load_snapshot()
    print(f"Loaded snapshot: {len(k)} Kalshi, {len(p)} Poly")
    t0 = time.time()
    matches = ad.match_events(k, p, min_score=min_score, existing_poly_slugs=set())
    elapsed = time.time() - t0
    out = SNAP_DIR / f"matches_{tag}.json"
    out.write_text(json.dumps({
        "tag": tag,
        "elapsed_s": elapsed,
        "min_score": min_score,
        "matches": matches,
    }, indent=2))
    high = sum(1 for m in matches if m.get("confidence") == "HIGH")
    med = sum(1 for m in matches if m.get("confidence") == "MED")
    low = sum(1 for m in matches if m.get("confidence") == "LOW")
    print(f"[{tag}] {len(matches)} matches in {elapsed:.1f}s | HIGH={high} MED={med} LOW={low}")
    print(f"        -> {out}")


def diff():
    old_data = json.loads((SNAP_DIR / "matches_old.json").read_text())
    new_data = json.loads((SNAP_DIR / "matches_new.json").read_text())
    old = old_data["matches"]
    new = new_data["matches"]

    def key(m):
        return (m["k_event_ticker"], m.get("p_slug", ""))

    old_keys = {key(m): m for m in old}
    new_keys = {key(m): m for m in new}

    removed = [old_keys[k] for k in old_keys.keys() - new_keys.keys()]
    added   = [new_keys[k] for k in new_keys.keys() - old_keys.keys()]
    both    = old_keys.keys() & new_keys.keys()
    conf_changed = [
        (old_keys[k], new_keys[k]) for k in both
        if old_keys[k].get("confidence") != new_keys[k].get("confidence")
    ]

    print(f"\n=== A/B diff ===")
    print(f"OLD: {len(old):>4}  ({old_data['elapsed_s']:.1f}s)  "
          f"H={sum(1 for m in old if m.get('confidence')=='HIGH')} "
          f"M={sum(1 for m in old if m.get('confidence')=='MED')} "
          f"L={sum(1 for m in old if m.get('confidence')=='LOW')}")
    print(f"NEW: {len(new):>4}  ({new_data['elapsed_s']:.1f}s)  "
          f"H={sum(1 for m in new if m.get('confidence')=='HIGH')} "
          f"M={sum(1 for m in new if m.get('confidence')=='MED')} "
          f"L={sum(1 for m in new if m.get('confidence')=='LOW')}")
    print(f"REMOVED by new: {len(removed):>3}   (likely fixed FPs)")
    print(f"ADDED   by new: {len(added):>3}   (new recall — verify)")
    print(f"CONFIDENCE changed: {len(conf_changed):>3}")

    def show(label, items, n=10):
        print(f"\n--- {label} (sample {min(n,len(items))} of {len(items)}) ---")
        for m in items[:n]:
            if isinstance(m, tuple):
                a, b = m
                print(f"  {a['k_title'][:50]:50}  |  {a['p_title'][:50]:50}")
                print(f"    OLD conf={a.get('confidence')} score={a.get('event_score'):.2f}  "
                      f"->  NEW conf={b.get('confidence')} score={b.get('event_score'):.2f}")
            else:
                print(f"  [{m.get('confidence'):<4}] {m['k_title'][:50]:50}  |  {m['p_title'][:50]:50}  "
                      f"score={m.get('event_score'):.2f}")

    show("REMOVED by new", removed, 15)
    show("ADDED by new", added, 15)
    show("CONFIDENCE changed", conf_changed, 10)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "snapshot":
        snapshot()
    elif cmd == "run":
        tag = "new"
        for i, a in enumerate(sys.argv):
            if a == "--tag" and i + 1 < len(sys.argv):
                tag = sys.argv[i + 1]
        run(tag)
    elif cmd == "diff":
        diff()
    else:
        print(__doc__)
