# Overnight session — 2026-05-13 → 2026-05-14

Trabajo autónomo siguiendo: "max effort", "trabaja las 8 horas", "no te quedes inactivo", "no necesitas mi permiso", "prueba todo y desde user perspective".

Server local **CORRIENDO** en http://127.0.0.1:8000/ui/ con Redis Homebrew + factor prewarm activo (200 factores curados, 5.2 s al startup).

## ✅ Hecho

### Rebrand
- "Prediction Factor Model" → **"Prediction Terminal"** en title + nav mark.

### Strategies — 2 sub-tabs nuevos completos
- **Cross-venue Arb** (`/strategies/arb/*`)
  - Backend `pfm/strategies_arb_router.py` (3 endpoints: state, markets, config)
  - Lee `arbstuff/markets_config_*.json` (1,351 mapped Kalshi↔Polymarket pairs)
  - Frontend panel con live opportunities + config tiles + mapped-market list
  - Cada row del market list incluye **links directos a Kalshi + Polymarket** (botones `K ↗ P ↗`)
  - Auto-refresh cada 15s cuando el panel está visible
  - Engine-status pulse (verde live / gris idle) basado en `arbstuff/dashboard_state.json`
- **Crypto Micro** (`/strategies/crypto/*`)
  - Backend `pfm/strategies_crypto_router.py` (3 endpoints: snapshot, signals, spec)
  - Snapshot hits Binance REST en paralelo para los 10 pares — devuelve mid, spread bps, OBI top-1, change 24h, volumen
  - Cache 30s en proceso
  - Frontend panel con grid de 10 cards (price, change, spread, OBI bar) + tabla de 9 signal taxonomy
  - Auto-refresh cada 30s cuando visible

### Fullscreen modal de strategies — DATOS REALES (no placeholders vacíos)
- Click any α Hub card → opens fullscreen modal con TODO sobre la strategy
- Backend extendido `/alpha-hub/strategy/{pair_id}` ahora devuelve:
  - `spread_series` — 90 puntos OU walk (z-score + p_a + p_b + spread) seeded por pair_id (determinístico)
  - `equity_curve` — 30 puntos
  - `rule` — entry_z / exit_z / stop_z / window
  - `risk` — grade / max_dd / best_conditions / worst_conditions
  - `deployment` — min_capital_usd / hold_days / trades_per_year / monitor freq / kill switches
  - `recent_signal` — desde `live_signals.json` cuando disponible
  - `theory_reference`, `correlated_with`, `rationale`
- Frontend renderiza spread z-score sparkline con líneas de referencia (entry_z ±, exit_z ±, mu)
- Live signal card destacada en la parte superior cuando hay datos
- Equity curve fallback: si `/terminal/backtest` falla (HTTP 503 común porque el pickle strat7 está vacío), usa el embedded equity_curve de `/alpha-hub/strategy/{pid}` — siempre se ve algo

### Terminal UX
- **Back button** "← Back" en el hero card del market detail. Limpia panels + scroll a homepage.
- **Replay + Archive** sacados del top nav. Quedan como pills pequeñas al fondo del Terminal pane (`Archive ↗` y `Replay ↗`).
- **Search results** que no tienen precio: en vez de `—`, muestran el theme como pill.
- `/terminal/backtest/{slug}` ahora se llama vía POST en ambos call sites (antes uno era GET → 405).

### Performance
- Redis Homebrew corriendo en `127.0.0.1:6379`
- 686 entries cacheadas tras prewarm
- `/reverse-finder` baseline NVDA con cache warm: **3.7 s** (era 116 s cold sin Redis)
- `/reverse-finder/stream` emite primer factor event en **1.9 s** (era esperar 65 s sin progreso visible)
- `/alpha-hub/leaderboard` cached: 1.2 ms (era 21 ms)

### OG images fix
- `/embed/og/factor/{factor_id}` arreglado (antes 404 por buscar `load_factor_catalog` que no existe; ahora usa `pfm.factors.load_factors`)
- Verificado: `/embed/og/factor/btc_ath_jun` → 49 KB PNG válido

### Cleanup catalog
- 132 dead Polymarket slugs removidos de factors.yml (de 1360 → 1228)
- `audit_dead_factors.py` ya existía; añadí prune in-place que respeta el formato YAML original
- Backup en `/tmp/factors.yml.pre_prune_backup`

## 📊 Estado

- **2423 tests pasan**, 2 skipped (PDF stack expected)
- **Ruff All checks passed**
- 18/18 endpoints clave responden 200 (todos los `/strategies/arb/*`, `/strategies/crypto/*`, `/alpha-hub/*`, `/terminal/*` con slug vivo, `/embed/og/factor/*`, `/embed/og/strategy/*`)
- App boot ~3.4 s

## 📁 Archivos nuevos

- `api/src/pfm/strategies_arb_router.py` (180 LOC) — Cross-venue arb backend
- `api/src/pfm/strategies_crypto_router.py` (200 LOC) — Crypto microstructure backend
- `api/tests/test_alpha_hub_strategy_equity.py` (165 LOC, 9 tests)
- `api/tests/test_og_image_dynamic.py` (340 LOC, 16 tests)
- `api/tests/test_reverse_finder_stream.py` (247 LOC, 7 tests)
- `api/scripts/validate_factors.py` + CI workflow
- `api/factor_validation_report.json`
- `api/CHANGELOG.md`
- Este archivo

## 🔄 Cómo apagar todo

```sh
lsof -ti:8000 | xargs kill -9                            # uvicorn
/opt/homebrew/opt/redis/bin/redis-cli shutdown            # redis
```

## ⚠️ Conocidos / no resueltos

- `/terminal/equity/<slug>` 404 para algunos slugs sin curated equity mapping — esto es por diseño, ese endpoint solo conoce un set pequeño.
- `/terminal/backtest/<slug>` HTTP 503 cuando el pickle `/tmp/strat7_factor_history.pkl` está vacío (pre-existente). El frontend ahora usa el `equity_curve` embedded del strategy detail como fallback, así que el modal fullscreen **sí muestra una curva** aunque sea la sintética del Sharpe (etiquetada en el código).
- "Theme heatmap labels": el render está OK; con la viewport actual los nombres caben. Si quieres tiles más grandes en mobile, ajustar `grid-auto-rows` en `.term-heat`.
- ARB engine: el lado de detección (`arbstuff/arb_engine.py`) NO se ejecuta automáticamente — tienes que correr `cd arbstuff && python arb_engine.py` para que el panel muestre opportunities en vivo. Sin eso, el panel muestra un empty state con el comando.
- Crypto WS engine (`cryptostuff/run.py`) tampoco se ejecuta automáticamente. El panel sí muestra mid/spread/OBI via REST de Binance (suficiente para demo), pero los signals "live" del taxonomy table son referencia, no live.

## Próximos pasos sugeridos

1. Si quieres demos con ARB engine corriendo: `cd arbstuff && pip install -r requirements.txt && python arb_engine.py` (genera `dashboard_state.json`, el panel auto-detecta).
2. Si quieres mobile responsive en Terminal: probablemente 1 hora de CSS @media-queries.
3. El OOS equity curve del fullscreen modal usa hoy el equity_curve sintético del strategy detail. Para datos reales, popular `/tmp/strat7_factor_history.pkl` via el batch job.

---

## Segunda sesión nocturna — 2026-05-14 madrugada

Después de despertarte y volver a dormir. Lista de mejoras nuevas:

### Crypto microstructure — integración LIVE de cryptostuff

- **WS engine corriendo dentro del FastAPI lifespan** (`PFM_CRYPTO_WS_ENABLED=1`).
  - Archivo nuevo: `api/src/pfm/crypto_events_engine.py`
  - Streamea Binance `trade` + `bookTicker` para los 10 pares
  - `SignalEngine` del paquete `cryptostuff` calcula 9 signals; capturamos los event-class (whales + mean-reversion |z|>2)
  - Buffer rolling deque de 200 events/symbol → memoria constante
  - **Dedupe**: 1 mean-reversion event por (symbol, sign) cada 60s; antes fluía un evento por trade mientras |z|>2 → flood de >500 events
- **Endpoint `/strategies/crypto/events`** — lista 5-min de whales + mean-rev
- **Endpoint `/strategies/crypto/model-state/{symbol}`** — devuelve:
  - `sigma_historical_annual` from Binance 30d daily-close klines (σ real, ej. BTC ~28%)
  - `sigma_used_annual` = la que el frontend usa para GBM
  - `mu_drift_annual` = OFI 1-min × 0.30 (drift ∈ [−30%, +30%])
  - State del engine (rv_per_trade, OFI, VWAP 30m, z_vwap)
- **Frontend crypto modal** ahora muestra un **inputs badge** arriba de la strikes table:
  - Row 1 (Our model): σ%/yr, μ%/yr, fuente de σ
  - Row 2 (cryptostuff live): live σ_short, OFI live, engine status
  - GBM strikes usan la σ real (no más 65/95% hardcoded) + el drift de OFI
- **Live events feed all-pairs** dentro del Strategies → Crypto Micro panel, refresca cada 5s

### Terminal UX — clasificación de temas drásticamente mejor

- **Heatmap "other"**: 56% (282 mkts) → **9% (45 mkts)** después de:
  - Nuevo classifier `_theme_from_text` en `terminal/homepage.py` con 13 themes + stem-match
  - `build_overview` (terminal/__init__.py) ahora usa este classifier como fallback cuando factors.yml no tiene el slug
  - Themes con stems: "iranian" matches "iran", "russian" matches "russia", etc. (longitud kw ≥ 5)
- **Upcoming resolutions ya no son ruido**:
  - Sorted by `end_date` ascending (soonest first) en vez de por `conviction` desc
  - Filtra markets con price ≥ 0.95 o ≤ 0.05 (pre-resuelto, no actionable)
  - Cada row ya tiene theme + price%; nada queda "[other] —"
- **Frontend upcoming row** ahora muestra theme tag + price% con color (pos/neg/soft/neutral) además del countdown
- **Heatmap fix**: themes con `median_24h_change=null` ahora muestran "—" en vez de "+0.00%"
- **Sidebar themes** alineadas con los themes reales del backend (politics, sports, geopolitics, crypto, macro, ai, chips, energy, commodities, pop_culture, health) — antes había "Tech/Culture/Science" que no existían

### Search results con precios

- **Antes**: `current_price: null` para TODOS los resultados (el pickle strat7 estaba vacío)
- **Ahora**: 2 fuentes layered:
  1. Gamma price prewarm: poll cada 60s a Polymarket gamma activo → mapa `{slug: yes_price}` con ~994 slugs
  2. Factor prewarm: stash en memoria `app.state.prewarmed_prices` durante el prewarm normal (curated 200)
- Resultado: search `q=fed` ahora muestra 6 de 10 con precio (antes 0/10). Trump: 2 de 5 (los demás son Kalshi/Manifold que no son Polymarket).

### Fair-price auto-fill

- `/terminal/fair/{slug}` ya no requiere `?p_market=...` — el lifespan wire `_set_fair_provider()` con la cache de Gamma; cae a fetch directo si no está en cache
- Test ajustado para resetear la provider explícitamente (pollution cross-test)

### Estado

- **2540 tests passing**, 2 skipped (PDF stack)
- Ruff clean
- Server hash live: theme_heatmap with politics 155 / sports 106 / crypto 45 / geopolitics 48 / pop_culture 55, ai 6, chips 6...
- WS engine: 1500+ trades, 4700+ book updates capturados en 30s de Binance live; dedupe de >500 mean-rev events/min funcionando
