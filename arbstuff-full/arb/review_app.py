"""
review_app.py  -  API backend for the React review UI.

    python review_app.py              # API on :5000
    cd review-ui && npm run dev       # React on :5173 (separate terminal)
"""
import sys, json, os, subprocess, traceback

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time as _time
from flask import Flask, jsonify, request, make_response, Response

FULL_MATCHES = "discovered_matches_full.json"
REVIEW_FILE  = "reviewed_matches.json"
CONFIG_FILE  = "markets_config.json"
EXPORT_FILE  = "markets_config_reviewed.json"
DASH_STATE   = "dashboard_state.json"
DASH_CONTROL = "dashboard_control.json"
BLACKLIST    = "arb_blacklist.json"

def _load(p, d=None):
    try:
        with open(p, encoding="utf-8") as f: return json.load(f)
    except Exception: return d

def _save(p, d):
    with open(p, "w", encoding="utf-8") as f: json.dump(d, f, indent=2, ensure_ascii=False)

app = Flask(__name__)

try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

@app.route("/api/data")
def api_data():
    md = _load(FULL_MATCHES, {})
    rv = _load(REVIEW_FILE, {"accepted": {}, "rejected": {}})
    ex = _load(EXPORT_FILE, {"events": []})
    cf = _load(CONFIG_FILE, {"events": []})
    return jsonify(
        matches=md.get("matches", []),
        meta={"generated_at": md.get("generated_at", ""), "kalshi_count": md.get("kalshi_count", 0), "poly_count": md.get("poly_count", 0)},
        review=rv,
        exported=[e.get("kalshi_ticker") for e in ex.get("events", [])],
        in_config=[e.get("kalshi_ticker") for e in cf.get("events", [])],
        existing=md.get("existing_tickers", []),
    )

@app.route("/api/accept", methods=["POST", "OPTIONS"])
def api_accept():
    if request.method == "OPTIONS":
        return make_response("", 204)
    try:
        b = request.get_json(force=True)
        t = b["ticker"]
        rv = _load(REVIEW_FILE, {"accepted": {}, "rejected": {}})
        rv.setdefault("accepted", {})[t] = {
            "k_title": b["k_title"], "p_slug": b["p_slug"],
            "p_title": b["p_title"], "score": b["score"],
            "mapping": b.get("mapping", {}),
            "accepted_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        rv.get("rejected", {}).pop(t, None)
        _save(REVIEW_FILE, rv)
        return jsonify(ok=True)
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500


@app.route("/api/recent-accepts")
def api_recent_accepts():
    """Return list of accepted matches sorted by accepted_at desc."""
    rv = _load(REVIEW_FILE, {"accepted": {}, "rejected": {}})
    items = []
    for ticker, info in rv.get("accepted", {}).items():
        items.append({
            "ticker": ticker,
            "k_title": info.get("k_title", ""),
            "p_slug": info.get("p_slug", ""),
            "p_title": info.get("p_title", ""),
            "mapping_count": len(info.get("mapping", {})),
            "accepted_at": info.get("accepted_at", ""),
        })
    # Sort by timestamp desc; entries without timestamp go last
    items.sort(key=lambda x: x.get("accepted_at") or "", reverse=True)
    return jsonify(items=items, total=len(items))

@app.route("/api/reject", methods=["POST", "OPTIONS"])
def api_reject():
    if request.method == "OPTIONS":
        return make_response("", 204)
    try:
        b = request.get_json(force=True)
        t = b["ticker"]
        rv = _load(REVIEW_FILE, {"accepted": {}, "rejected": {}})
        rv.setdefault("rejected", {})[t] = {"k_title": b["k_title"], "p_slug": b["p_slug"]}
        rv.get("accepted", {}).pop(t, None)
        _save(REVIEW_FILE, rv)
        return jsonify(ok=True)
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500

@app.route("/api/reset", methods=["POST", "OPTIONS"])
def api_reset():
    if request.method == "OPTIONS":
        return make_response("", 204)
    try:
        t = request.get_json(force=True)["ticker"]
        rv = _load(REVIEW_FILE, {"accepted": {}, "rejected": {}})
        rv.get("accepted", {}).pop(t, None)
        rv.get("rejected", {}).pop(t, None)
        _save(REVIEW_FILE, rv)
        return jsonify(ok=True)
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500

@app.route("/api/export", methods=["POST", "OPTIONS"])
def api_export():
    if request.method == "OPTIONS":
        return make_response("", 204)
    try:
        rv = _load(REVIEW_FILE, {"accepted": {}, "rejected": {}})
        ex = _load(EXPORT_FILE, {"poll_interval": 8, "threshold": 0.94, "min_alert_profit": 1.0, "events": []})
        have = {e.get("kalshi_ticker") for e in ex.get("events", [])}
        added = 0
        for t, info in rv.get("accepted", {}).items():
            if t not in have:
                ex["events"].append({"name": info["k_title"], "kalshi_ticker": t, "poly_slug": info["p_slug"], "mapping": info.get("mapping", {})})
                added += 1
        _save(EXPORT_FILE, ex)
        return jsonify(ok=True, added=added, total=len(ex["events"]))
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500

@app.route("/api/discover", methods=["POST", "OPTIONS"])
def api_discover():
    if request.method == "OPTIONS":
        return make_response("", 204)
    try:
        result = subprocess.run(
            [sys.executable, "auto_discover.py"],
            capture_output=True, text=True, timeout=600,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            encoding="utf-8", errors="replace",
        )
        return jsonify(ok=result.returncode == 0, stdout=result.stdout[-2000:], stderr=result.stderr[-1000:])
    except Exception as exc:
        return jsonify(ok=False, stderr=str(exc))

# ── Dashboard endpoints ──────────────────────────────────────────

@app.route("/api/dashboard/state")
def dashboard_state():
    return jsonify(_load(DASH_STATE, {"bot_status": "offline", "opportunities": []}))

@app.route("/api/dashboard/stream")
def dashboard_stream():
    """SSE endpoint — streams dashboard state every 2s."""
    def generate():
        while True:
            state = _load(DASH_STATE, {"bot_status": "offline", "opportunities": []})
            yield f"data: {json.dumps(state, ensure_ascii=False)}\n\n"
            _time.sleep(2)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/dashboard/settings", methods=["POST", "OPTIONS"])
def dashboard_settings():
    if request.method == "OPTIONS":
        return make_response("", 204)
    try:
        b = request.get_json(force=True)
        ctrl = _load(DASH_CONTROL, {})
        if "email_enabled" in b:
            ctrl["email_enabled"] = bool(b["email_enabled"])
        if "threshold" in b:
            ctrl["threshold"] = float(b["threshold"])
        if "min_alert_profit" in b:
            ctrl["min_alert_profit"] = float(b["min_alert_profit"])
        if b.get("scan_mode") in ("OG", "WS"):
            ctrl["scan_mode"] = b["scan_mode"]
        _save(DASH_CONTROL, ctrl)
        return jsonify(ok=True)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500

@app.route("/api/dashboard/orderbook")
def dashboard_orderbook():
    """On-demand orderbook fetch for the detail panel."""
    k_ticker = request.args.get("kalshi_ticker", "")
    p_token = request.args.get("poly_token", "")
    result = {"kalshi": {}, "poly": {}}
    try:
        if k_ticker:
            from arb_engine import KalshiClient
            kc = KalshiClient()
            result["kalshi"] = kc.get_orderbook(k_ticker)
        if p_token:
            from arb_engine import PolymarketClient
            pc = PolymarketClient()
            result["poly"] = pc.get_orderbook(p_token)
    except Exception as exc:
        result["error"] = str(exc)
    return jsonify(result)


@app.route("/api/dashboard/blacklist", methods=["POST", "OPTIONS"])
def dashboard_blacklist():
    """Add an arb_key to the blacklist so it never appears again."""
    if request.method == "OPTIONS":
        return make_response("", 204)
    try:
        b = request.get_json(force=True)
        arb_key = b.get("arb_key", "")
        if not arb_key:
            return jsonify(ok=False, error="No arb_key"), 400
        bl = _load(BLACKLIST, [])
        if arb_key not in bl:
            bl.append(arb_key)
            _save(BLACKLIST, bl)
        return jsonify(ok=True, blacklisted=len(bl))
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500


@app.route("/api/dashboard/clear-blacklist", methods=["POST", "OPTIONS"])
def dashboard_clear_blacklist():
    if request.method == "OPTIONS":
        return make_response("", 204)
    _save(BLACKLIST, [])
    return jsonify(ok=True)


PNL_LOG = "arb_pnl_log.json"
DETECTION_HISTORY = "arb_detection_history.json"

@app.route("/api/dashboard/pnl")
def dashboard_pnl():
    trades = _load(PNL_LOG, [])
    total = sum(t.get("guaranteed_profit", 0) for t in trades)
    return jsonify(trades=trades, total_pnl=round(total, 2), count=len(trades))


@app.route("/api/dashboard/detection-history")
def dashboard_detection_history():
    """Return the rolling detection history (max 500 entries)."""
    hist = _load(DETECTION_HISTORY, [])
    # Reverse so newest first
    return jsonify(items=list(reversed(hist)), count=len(hist))


@app.route("/api/dashboard/config-stats")
def dashboard_config_stats():
    """Return stats about loaded config files."""
    reviewed = _load(EXPORT_FILE, {"events": []})
    main = _load(CONFIG_FILE, {"events": []})
    r_mapped = sum(1 for e in reviewed.get("events", []) if e.get("mapping"))
    m_mapped = sum(1 for e in main.get("events", []) if e.get("mapping"))
    return jsonify(
        reviewed={"total": len(reviewed.get("events", [])), "mapped": r_mapped},
        main={"total": len(main.get("events", [])), "mapped": m_mapped},
        combined_mapped=r_mapped + m_mapped,
    )


DISCOVERED_CONFIG = "markets_config_discovered.json"

POLITICS_CONFIG = "markets_config_politics.json"

@app.route("/api/config-events")
def config_events():
    """Return merged event list from ALL config files (with source tag)."""
    reviewed = _load(EXPORT_FILE, {"events": []})
    main = _load(CONFIG_FILE, {"events": []})
    discovered = _load(DISCOVERED_CONFIG, {"events": []})
    politics = _load(POLITICS_CONFIG, {"events": []})
    seen = set()
    events = []
    for e in reviewed.get("events", []):
        kt = e.get("kalshi_ticker")
        if kt and e.get("mapping"):
            seen.add(kt)
            events.append({**e, "source": "reviewed"})
    for e in main.get("events", []):
        kt = e.get("kalshi_ticker")
        if kt and kt not in seen and e.get("mapping"):
            seen.add(kt)
            events.append({**e, "source": "main"})
    for e in politics.get("events", []):
        kt = e.get("kalshi_ticker")
        if kt and kt not in seen and e.get("mapping"):
            seen.add(kt)
            events.append({**e, "source": "politics"})
    for e in discovered.get("events", []):
        kt = e.get("kalshi_ticker")
        if kt and kt not in seen and e.get("mapping"):
            seen.add(kt)
            events.append({**e, "source": "discovered"})
    return jsonify(events=events)


@app.route("/api/politics/events")
def politics_events():
    """Return detailed breakdown of political matches with state/office/etc."""
    cfg = _load(POLITICS_CONFIG, {"events": []})
    events = cfg.get("events", [])

    # Parse names like "Texas District 33 HOUSE (primary) [D] 2026"
    import re
    parsed = []
    for ev in events:
        name = ev.get("name", "")
        state = None
        office = None
        district = None
        race_type = "general"
        party = None
        year = None

        for code in ("AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"):
            pass
        # Simpler: parse by keywords
        for kw in ("HOUSE","SEN","GOV","LTGOV","AG","SOS","TREAS","PRES","MAYOR"):
            if f" {kw}" in f" {name}" or name.startswith(kw):
                office = kw
                break
        m = re.search(r"District (\d+)", name)
        if m:
            district = int(m.group(1))
        if "(primary)" in name.lower():
            race_type = "primary"
        elif "(special)" in name.lower():
            race_type = "special"
        elif "(runoff)" in name.lower():
            race_type = "runoff"
        m = re.search(r"\[([DRIL])\]", name)
        if m:
            party = m.group(1)
        m = re.search(r"\b(20\d{2})\b", name)
        if m:
            year = int(m.group(1))
        # State: first word is state name
        state_name = name.split()[0] if name else ""
        STATE_NAMES = {"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","Wisconsin":"WI","Wyoming":"WY"}
        # Try combining first two words for multi-word states
        first_two = " ".join(name.split()[:2]) if name else ""
        state = STATE_NAMES.get(state_name) or STATE_NAMES.get(first_two)
        # Fallback: scan for all state names
        if not state:
            for sn, code in STATE_NAMES.items():
                if name.startswith(sn):
                    state = code
                    break

        parsed.append({
            "name": name,
            "kalshi_ticker": ev.get("kalshi_ticker"),
            "poly_slug": ev.get("poly_slug"),
            "mapping_count": len(ev.get("mapping", {})),
            "state": state,
            "office": office,
            "district": district,
            "race_type": race_type,
            "party": party,
            "year": year,
        })
    # Stats
    by_state = {}
    by_office = {}
    by_type = {}
    for e in parsed:
        if e["state"]:
            by_state[e["state"]] = by_state.get(e["state"], 0) + 1
        if e["office"]:
            by_office[e["office"]] = by_office.get(e["office"], 0) + 1
        by_type[e["race_type"]] = by_type.get(e["race_type"], 0) + 1

    return jsonify(
        events=parsed,
        total=len(parsed),
        stats={"by_state": by_state, "by_office": by_office, "by_type": by_type},
    )


@app.route("/api/politics/run", methods=["POST", "OPTIONS"])
def politics_run():
    """Trigger a fresh political discovery."""
    if request.method == "OPTIONS":
        return make_response("", 204)
    try:
        result = subprocess.run(
            [sys.executable, "politics_discover.py"],
            capture_output=True, text=True, timeout=600,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            encoding="utf-8", errors="replace",
        )
        return jsonify(ok=result.returncode == 0,
                       stdout=result.stdout[-3000:],
                       stderr=result.stderr[-1000:])
    except Exception as exc:
        return jsonify(ok=False, error=str(exc))


@app.route("/api/discovery/status")
def discovery_status():
    """Return status of last discovery run."""
    disc = _load(FULL_MATCHES, {})
    matches = disc.get("matches", [])
    high = [m for m in matches if m.get("confidence") == "HIGH"]
    med = [m for m in matches if m.get("confidence") == "MED"]
    low = [m for m in matches if m.get("confidence") == "LOW"]
    return jsonify(
        generated_at=disc.get("generated_at", ""),
        total=len(matches),
        high=len(high),
        med=len(med),
        low=len(low),
        kalshi_count=disc.get("kalshi_count", 0),
        poly_count=disc.get("poly_count", 0),
    )


@app.route("/api/discovery/run", methods=["POST", "OPTIONS"])
def discovery_run():
    """Trigger a manual discovery run."""
    if request.method == "OPTIONS":
        return make_response("", 204)
    try:
        result = subprocess.run(
            [sys.executable, "auto_discover.py"],
            capture_output=True, text=True, timeout=600,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            encoding="utf-8", errors="replace",
        )
        return jsonify(ok=result.returncode == 0,
                       stdout=result.stdout[-2000:],
                       stderr=result.stderr[-1000:])
    except Exception as exc:
        return jsonify(ok=False, error=str(exc))



# ── Sports arb endpoints ────────────────────────────────────────

SPORTS_STATE = "sports_tracking_state.json"
_sports_thread = None
_sports_stop   = False

@app.route("/api/sports/discover")
def sports_discover():
    """Run fast discovery, return proposed pairs."""
    try:
        from sports_discover import find_live_sports_pairs
        pairs = find_live_sports_pairs(resolve=False)
        # Check Kalshi liquidity for top pairs
        import requests as _req
        enriched = []
        for p in sorted(pairs, key=lambda x: x["event_score"], reverse=True)[:30]:
            et = p["kalshi_event_ticker"]
            try:
                r = _req.get(f"https://api.elections.kalshi.com/trade-api/v2/markets",
                    params={"event_ticker": et, "limit": 5}, timeout=5)
                mkts = r.json().get("markets", [])
                good = [m for m in mkts if float(m.get("yes_ask_dollars") or 0) > 0]
                p["k_markets"] = [{
                    "ticker": m["ticker"],
                    "title": m.get("title","")[:50],
                    "yes_bid": float(m.get("yes_bid_dollars") or 0),
                    "yes_ask": float(m.get("yes_ask_dollars") or 0),
                    "volume": float(m.get("volume_fp") or 0),
                } for m in good[:4]]
                p["k_liquid"] = len(good) >= 1
            except Exception:
                p["k_markets"] = []
                p["k_liquid"] = False
            enriched.append(p)
        return jsonify(ok=True, pairs=enriched, count=len(enriched))
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500


@app.route("/api/sports/track-manual", methods=["POST", "OPTIONS"])
def sports_track_manual():
    """
    Start tracking from a Kalshi event ticker + Polymarket slug.
    Resolves markets and tokens automatically.
    """
    global _sports_thread, _sports_stop
    if request.method == "OPTIONS":
        return make_response("", 204)
    try:
        import requests as _req
        b = request.get_json(force=True)
        k_event = b["kalshi_event_ticker"]
        p_slug  = b["poly_slug"]
        print(f"[SPORTS] Resolving K:{k_event} P:{p_slug}")

        # ── Resolve Kalshi: get the two team markets ──
        print(f"[SPORTS] Fetching Kalshi markets for {k_event}...")
        r = _req.get(f"https://api.elections.kalshi.com/trade-api/v2/markets",
            params={"event_ticker": k_event, "limit": 10}, timeout=(4, 6))
        k_mkts = r.json().get("markets", [])
        print(f"[SPORTS] Kalshi returned {len(k_mkts)} markets (status={r.status_code})")
        if len(k_mkts) < 2:
            # Also try without status filter in case markets are closed/settled
            all_statuses = r.json()
            print(f"[SPORTS] Raw response keys: {list(all_statuses.keys()) if isinstance(all_statuses, dict) else type(all_statuses)}")
            return jsonify(ok=False, error=f"Kalshi event '{k_event}' returned {len(k_mkts)} market(s). Check ticker spelling (e.g. KXNCAABBGAME not KXNCAAMBGAME)."), 400

        k_a = k_mkts[0]["ticker"]
        k_b = k_mkts[1]["ticker"]
        k_title = k_mkts[0].get("title", k_event)
        print(f"[SPORTS] K resolved: {k_a} / {k_b}")

        # ── Resolve Polymarket: get moneyline tokens from slug ──
        print(f"[SPORTS] Fetching Poly slug={p_slug}...")
        r2 = _req.get(f"https://gamma-api.polymarket.com/events?slug={p_slug}", timeout=(4, 6))
        p_evs = r2.json()
        print(f"[SPORTS] Poly returned {len(p_evs) if isinstance(p_evs, list) else 'obj'}")
        if not p_evs:
            return jsonify(ok=False, error=f"Polymarket: slug '{p_slug}' not found"), 400
        p_ev = p_evs[0] if isinstance(p_evs, list) else p_evs
        p_title = p_ev.get("title", p_slug)

        # Find the best market (prefer moneyline over handicap/total)
        best_pm = None
        for pm in p_ev.get("markets", []):
            q = pm.get("question", "")
            if "O/U" not in q and "Total" not in q and "Handicap" not in q and "Odd" not in q:
                best_pm = pm
                break
        if not best_pm:
            best_pm = p_ev.get("markets", [{}])[0]  # fallback to first

        clob_raw = best_pm.get("clobTokenIds", "[]")
        clob_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else clob_raw
        outcomes_raw = best_pm.get("outcomes", "[]")
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw

        if len(clob_ids) < 2:
            return jsonify(ok=False, error="Polymarket: no tokens found for this market"), 400

        p_a = clob_ids[0]
        p_b = clob_ids[1]
        out_a = outcomes[0] if len(outcomes) >= 1 else "A"
        out_b = outcomes[1] if len(outcomes) >= 2 else "B"
        title = f"{k_title} | {p_title}"

        # ── Stop previous + start new ──
        _sports_stop = True
        if _sports_thread and _sports_thread.is_alive():
            _sports_thread.join(timeout=3)
        _sports_stop = False

        import threading
        _sports_thread = threading.Thread(
            target=_run_sports_tracker,
            args=(k_a, k_b, p_a, p_b, out_a, out_b, title),
            daemon=True,
        )
        _sports_thread.start()

        return jsonify(ok=True, title=title,
            resolve_info=f"K: {k_a} / {k_b} | P: {out_a} / {out_b} | {best_pm.get('question','')[:40]}")
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500


@app.route("/api/sports/track", methods=["POST", "OPTIONS"])
def sports_track():
    """Start tracking with explicit tickers (legacy)."""
    global _sports_thread, _sports_stop
    if request.method == "OPTIONS":
        return make_response("", 204)
    try:
        b = request.get_json(force=True)
        k_a = b["kalshi_a"]
        k_b = b["kalshi_b"]
        p_a = b["poly_token_a"]
        p_b = b["poly_token_b"]
        out_a = b.get("outcome_a", "A")
        out_b = b.get("outcome_b", "B")
        title = b.get("title", "Sports Arb")

        _sports_stop = True
        if _sports_thread and _sports_thread.is_alive():
            _sports_thread.join(timeout=3)
        _sports_stop = False

        import threading
        _sports_thread = threading.Thread(
            target=_run_sports_tracker,
            args=(k_a, k_b, p_a, p_b, out_a, out_b, title),
            daemon=True,
        )
        _sports_thread.start()
        return jsonify(ok=True, msg="Tracking started")
    except Exception as exc:
        traceback.print_exc()
        return jsonify(ok=False, error=str(exc)), 500


@app.route("/api/sports/stop", methods=["POST", "OPTIONS"])
def sports_stop():
    global _sports_stop
    if request.method == "OPTIONS":
        return make_response("", 204)
    _sports_stop = True
    _save(SPORTS_STATE, {"status": "stopped"})
    return jsonify(ok=True)


@app.route("/api/sports/status")
def sports_status():
    return jsonify(_load(SPORTS_STATE, {"status": "idle"}))


def _run_sports_tracker(k_a, k_b, p_a, p_b, out_a, out_b, title):
    """Background thread: poll both sides, check arb, write state."""
    global _sports_stop
    import requests as _req
    KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
    CLOB_BASE   = "https://clob.polymarket.com"

    # Extract short Kalshi market names from ticker suffix
    k_a_name = k_a.split("-")[-1] if "-" in k_a else "A"
    k_b_name = k_b.split("-")[-1] if "-" in k_b else "B"

    state = {
        "status": "running", "title": title,
        "out_a": out_a, "out_b": out_b,
        "k_a_ticker": k_a, "k_b_ticker": k_b,
        "k_a_name": k_a_name, "k_b_name": k_b_name,
        "k_a_ask": 0, "k_b_ask": 0, "k_a_bid": 0, "k_b_bid": 0,
        "p_a_ask": 0, "p_b_ask": 0, "p_a_bid": 0, "p_b_bid": 0,
        "best_combined": 0, "gap": 0, "comb_1": 0, "comb_2": 0,
        "dir_1": "", "dir_2": "",
        "checks": 0, "arbs_found": 0, "trades": [],
        "total_pnl": 0, "snapshots": [],
        "poll_ms": 0, "poll_ts": "",
        "last_arb_ts": "", "last_arb_profit": 0,
    }
    distA = distB = 99  # initialize for scope

    session_k = _req.Session()
    session_p = _req.Session()
    last_trade_ts = 0

    while not _sports_stop:
        t0 = _time.time()
        state["checks"] += 1

        # Poll Kalshi
        for ticker, prefix in [(k_a, "k_a"), (k_b, "k_b")]:
            try:
                r = session_k.get(f"{KALSHI_BASE}/markets/{ticker}", timeout=5)
                if r.ok:
                    m = r.json().get("market", {})
                    state[f"{prefix}_ask"] = float(m.get("yes_ask_dollars") or 0)
                    state[f"{prefix}_bid"] = float(m.get("yes_bid_dollars") or 0)
            except Exception:
                pass

        # Poll Polymarket — use /price endpoint (not /book, which is raw CLOB)
        # Sports markets: /price gives the real executable price via cross-outcome math
        for token, prefix in [(p_a, "p_a"), (p_b, "p_b")]:
            try:
                # Buy price = what you pay to acquire the token (= ask)
                rp = session_p.get(f"{CLOB_BASE}/price",
                    params={"token_id": token, "side": "buy"}, timeout=5)
                if rp.ok:
                    buy_price = float(rp.json().get("price", 0))
                    state[f"{prefix}_ask"] = buy_price
                # Midpoint for bid estimate
                rm = session_p.get(f"{CLOB_BASE}/midpoint",
                    params={"token_id": token}, timeout=5)
                if rm.ok:
                    mid = float(rm.json().get("mid", 0))
                    state[f"{prefix}_bid"] = mid
            except Exception:
                pass

        # Arb check — cross-team pairing:
        # An arb = buy Team_X on Kalshi + buy Team_Y on Poly (opposite teams)
        # We need to detect which pairing is cross-team vs same-team.
        # Cross-team: both combos ≈ 1.0. Same-team: one ≈ 0.1, other ≈ 1.9.
        # Try both pairings, pick the one where values are closer to 1.0.
        distA = distB = 99
        if state["k_a_ask"] and state["k_b_ask"] and state["p_a_ask"] and state["p_b_ask"]:
            # Pairing A: k_a+p_a and k_b+p_b
            combAa = state["k_a_ask"] + state["p_a_ask"]
            combAb = state["k_b_ask"] + state["p_b_ask"]
            # Pairing B: k_a+p_b and k_b+p_a
            combBa = state["k_a_ask"] + state["p_b_ask"]
            combBb = state["k_b_ask"] + state["p_a_ask"]

            # The correct pairing has both values close to 1.0
            distA = abs(combAa - 1.0) + abs(combAb - 1.0)
            distB = abs(combBa - 1.0) + abs(combBb - 1.0)

            if distA < distB:
                comb1, comb2 = combAa, combAb
                dir1_label = f"K_{state.get('k_a_name','A')}+P_{out_a}"
                dir2_label = f"K_{state.get('k_b_name','B')}+P_{out_b}"
            else:
                comb1, comb2 = combBa, combBb
                dir1_label = f"K_{state.get('k_a_name','A')}+P_{out_b}"
                dir2_label = f"K_{state.get('k_b_name','B')}+P_{out_a}"
        else:
            comb1 = comb2 = 99
            dir1_label = dir2_label = ""

        best = min(comb1, comb2)
        state["best_combined"] = round(best, 4)
        state["gap"] = round(1.0 - best, 4)
        state["comb_1"] = round(comb1, 4)
        state["comb_2"] = round(comb2, 4)
        state["dir_1"] = dir1_label
        state["dir_2"] = dir2_label

        # Timing
        poll_ms = round((_time.time() - t0) * 1000)
        state["poll_ms"] = poll_ms
        state["poll_ts"] = _time.strftime("%H:%M:%S.") + f"{int(_time.time()*1000)%1000:03d}"

        # Simulate trade if arb
        if best < 1.0 and (_time.time() - last_trade_ts) > 30:
            k_price = state["k_a_ask"] if comb1 < comb2 else state["k_b_ask"]
            p_price = state["p_a_ask"] if (distA < distB and comb1 < comb2) or (distA >= distB and comb2 <= comb1) else state["p_b_ask"]
            k_fee = 0.07 * k_price * (1 - k_price)
            p_fee = 0.02 * p_price
            net = 1.0 - best - k_fee - p_fee
            if net > 0:
                contracts = int(50.0 / best)
                profit = round(contracts * net, 4)
                direction = dir1_label if comb1 < comb2 else dir2_label
                trade = {
                    "ts": _time.strftime("%H:%M:%S"),
                    "direction": direction,
                    "combined": round(best, 4),
                    "k_price": round(k_price, 4),
                    "p_price": round(p_price, 4),
                    "net_per_c": round(net, 4),
                    "contracts": contracts,
                    "profit": profit,
                    "k_fee": round(k_fee, 4),
                    "p_fee": round(p_fee, 4),
                }
                state["trades"].append(trade)
                state["arbs_found"] += 1
                state["total_pnl"] = round(sum(t["profit"] for t in state["trades"]), 4)
                state["last_arb_ts"] = _time.strftime("%H:%M:%S")
                state["last_arb_profit"] = profit
                last_trade_ts = _time.time()

        # Record snapshot (keep last 60)
        state["snapshots"].append({
            "ts": _time.strftime("%H:%M:%S"),
            "k_a": state["k_a_ask"], "k_b": state["k_b_ask"],
            "p_a": state["p_a_ask"], "p_b": state["p_b_ask"],
            "best": state["best_combined"], "gap": state["gap"],
        })
        if len(state["snapshots"]) > 60:
            state["snapshots"] = state["snapshots"][-60:]

        state["last_update"] = _time.strftime("%H:%M:%S")
        _save(SPORTS_STATE, state)

        elapsed = _time.time() - t0
        _time.sleep(max(0, 1.5 - elapsed))  # ~1.5s poll cycle

    state["status"] = "stopped"
    _save(SPORTS_STATE, state)


if __name__ == "__main__":
    port = 5060
    for i, a in enumerate(sys.argv):
        if a == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    print(f"  API -> http://localhost:{port}")
    print(f"  Run the React UI:  cd review-ui && npm run dev")
    app.run(port=port, debug=False)
