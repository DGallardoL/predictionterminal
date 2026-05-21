#!/bin/bash
# Usage: scripts/monitor.sh [api_url=http://localhost:8000] [interval=10]
# Pings /health every interval; logs status to /tmp/pfm-monitor.log.
# Sends alert (echo to stderr) when transitioning UP->DOWN or DOWN->UP,
# specifically when DOWN streak exceeds 3 consecutive checks.
#
# Status checks per tick:
#   * GET  /health         -> assert JSON {"status":"ok"}
#   * GET  /health/deep    -> log per-source latencies
#   * GET  /metrics/audit  -> grep err_rate, alert if > 5%
#   * ps   gunicorn pfm    -> worker process count
#   * redis-cli ping       -> only if REDIS_URL is set
#
# Output: stamped lines to stdout + /tmp/pfm-monitor.log.
# Alerts go to stderr (so callers can pipe them separately).
#
# Dependencies: curl, awk, grep, ps. jq optional (graceful fallback to grep).

set -u

API_URL="${1:-http://localhost:8000}"
INTERVAL="${2:-10}"
LOG_FILE="${PFM_MONITOR_LOG:-/tmp/pfm-monitor.log}"
ALERT_THRESHOLD="${PFM_MONITOR_ALERT_THRESHOLD:-3}"
ERR_RATE_THRESHOLD="${PFM_MONITOR_ERR_THRESHOLD:-5.0}"
CURL_TIMEOUT="${PFM_MONITOR_CURL_TIMEOUT:-5}"

# Globals populated by fetch(); initialised here so `set -u` is happy.
LAST_HTTP_CODE="000"
LAST_LATENCY_MS="0"

# --- helpers ---------------------------------------------------------------

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

have() { command -v "$1" >/dev/null 2>&1; }

log_line() {
    local line
    line="[$(stamp)] $*"
    echo "$line"
    echo "$line" >>"$LOG_FILE"
}

alert() {
    local line
    line="[$(stamp)] ALERT $*"
    echo "$line" >&2
    echo "$line" >>"$LOG_FILE"
}

# Fetch a URL with timeout. Echoes body on success; empty string on failure.
# Sets global LAST_HTTP_CODE and LAST_LATENCY_MS.
fetch() {
    local url="$1"
    local body code time_total
    local tmp
    tmp="$(mktemp -t pfm-monitor.XXXXXX)"
    # %{http_code} and %{time_total} ; write body to tmp
    local fmt='%{http_code} %{time_total}'
    local meta
    meta=$(curl -sS -o "$tmp" -w "$fmt" --max-time "$CURL_TIMEOUT" "$url" 2>/dev/null || echo "000 0")
    code=$(awk '{print $1}' <<<"$meta")
    time_total=$(awk '{print $2}' <<<"$meta")
    LAST_HTTP_CODE="$code"
    # convert to ms (integer)
    LAST_LATENCY_MS=$(awk -v t="$time_total" 'BEGIN{printf "%d", t*1000}')
    body=$(cat "$tmp")
    rm -f "$tmp"
    echo "$body"
}

# Extract a JSON scalar by key. Uses jq when available, falls back to grep.
json_get() {
    local body="$1" key="$2"
    if have jq; then
        echo "$body" | jq -r --arg k "$key" '..|objects| select(has($k)) | .[$k]' 2>/dev/null | head -n1
    else
        # naive: "key":"value"  or  "key": number
        echo "$body" | grep -o "\"$key\"[[:space:]]*:[[:space:]]*[^,}]*" | head -n1 \
            | sed -E "s/\"$key\"[[:space:]]*:[[:space:]]*//; s/^\"//; s/\"$//"
    fi
}

# --- per-tick checks -------------------------------------------------------

check_health() {
    local body status
    body=$(fetch "$API_URL/health")
    if [[ "$LAST_HTTP_CODE" != "200" || -z "$body" ]]; then
        log_line "health    http=${LAST_HTTP_CODE} latency=${LAST_LATENCY_MS}ms body_empty"
        return 1
    fi
    status=$(json_get "$body" "status")
    log_line "health    http=200 latency=${LAST_LATENCY_MS}ms status=${status:-?}"
    [[ "$status" == "ok" ]]
}

check_deep() {
    local body
    body=$(fetch "$API_URL/health/deep")
    if [[ "$LAST_HTTP_CODE" != "200" ]]; then
        log_line "deep      http=${LAST_HTTP_CODE} latency=${LAST_LATENCY_MS}ms (skipped)"
        return 0
    fi
    if have jq; then
        # log "source=latency_ms" per per-source entry where possible
        local pairs
        pairs=$(echo "$body" | jq -r '
            (.sources // .checks // .components // {}) |
            to_entries[] |
            "\(.key)=\((.value.latency_ms // .value.latency // .value.elapsed_ms // "?"))ms"
        ' 2>/dev/null | tr '\n' ' ')
        log_line "deep      http=200 latency=${LAST_LATENCY_MS}ms ${pairs:-no_sources_field}"
    else
        log_line "deep      http=200 latency=${LAST_LATENCY_MS}ms (install jq for per-source detail)"
    fi
}

check_audit() {
    local body err_rate
    body=$(fetch "$API_URL/metrics/audit")
    if [[ "$LAST_HTTP_CODE" != "200" ]]; then
        log_line "audit     http=${LAST_HTTP_CODE} latency=${LAST_LATENCY_MS}ms (skipped)"
        return 0
    fi
    # find err_rate (accept percent or fraction). Grep all matches, take max.
    err_rate=$(echo "$body" | grep -oE '"err_rate"[[:space:]]*:[[:space:]]*[0-9.]+' \
                | grep -oE '[0-9.]+' | sort -g | tail -n1)
    if [[ -z "$err_rate" ]]; then
        log_line "audit     http=200 latency=${LAST_LATENCY_MS}ms err_rate=n/a"
        return 0
    fi
    # if value < 1 assume fraction (0.07 -> 7%), else assume percent
    local pct
    pct=$(awk -v v="$err_rate" 'BEGIN{ if(v<1) printf "%.2f", v*100; else printf "%.2f", v }')
    log_line "audit     http=200 latency=${LAST_LATENCY_MS}ms err_rate=${pct}%"
    awk -v p="$pct" -v t="$ERR_RATE_THRESHOLD" 'BEGIN{ exit !(p>t) }' \
        && alert "err_rate=${pct}% exceeds threshold ${ERR_RATE_THRESHOLD}%"
    return 0
}

check_workers() {
    local count
    count=$(ps -A -o command= 2>/dev/null | grep -E 'gunicorn.*pfm\.main|pfm\.main:app' \
            | grep -v grep | wc -l | tr -d ' ')
    log_line "workers   gunicorn_pfm_count=${count}"
}

check_redis() {
    [[ -z "${REDIS_URL:-}" ]] && return 0
    if ! have redis-cli; then
        log_line "redis     skipped (redis-cli not installed)"
        return 0
    fi
    local reply
    reply=$(redis-cli -u "$REDIS_URL" --no-auth-warning ping 2>/dev/null || echo "FAIL")
    log_line "redis     url_set ping=${reply}"
}

# --- main loop -------------------------------------------------------------

mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
: >>"$LOG_FILE" || { echo "cannot write $LOG_FILE" >&2; exit 1; }

log_line "monitor   start api=${API_URL} interval=${INTERVAL}s log=${LOG_FILE} alert_threshold=${ALERT_THRESHOLD} err_threshold=${ERR_RATE_THRESHOLD}%"

# track UP/DOWN streak. state is empty until first check.
state=""           # "UP" or "DOWN"
down_streak=0
up_streak=0

trap 'log_line "monitor   stop (signal)"; exit 0' INT TERM

while true; do
    if check_health; then
        next="UP"
        up_streak=$((up_streak+1))
        down_streak=0
    else
        next="DOWN"
        down_streak=$((down_streak+1))
        up_streak=0
    fi

    # transition alerts
    if [[ -z "$state" ]]; then
        state="$next"
        log_line "state     initial=${state}"
    elif [[ "$state" != "$next" ]]; then
        if [[ "$next" == "DOWN" && "$down_streak" -ge "$ALERT_THRESHOLD" ]]; then
            alert "transition UP->DOWN (down_streak=${down_streak})"
            state="DOWN"
        elif [[ "$next" == "UP" && "$state" == "DOWN" ]]; then
            alert "transition DOWN->UP (recovered after $((down_streak)) ticks; up_streak=${up_streak})"
            state="UP"
        fi
    elif [[ "$state" == "DOWN" && "$down_streak" -eq "$ALERT_THRESHOLD" ]]; then
        # first time we cross the threshold while already trending DOWN from start
        alert "sustained DOWN (down_streak=${down_streak})"
    fi

    # auxiliary checks only when API reachable to avoid noisy logs
    if [[ "$next" == "UP" ]]; then
        check_deep
        check_audit
    fi
    check_workers
    check_redis

    sleep "$INTERVAL"
done
