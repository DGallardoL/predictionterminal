# arbstuff

Motor de detección de arbitrajes entre **Kalshi** y **Polymarket** para terminales de prediction markets. Headless, sin UI propia: escribe el estado a `dashboard_state.json` para que un frontend externo (Bloomberg-style) lo consuma.

> **Solo detección.** Esta carpeta no ejecuta órdenes reales — funciona enteramente con endpoints públicos. El modo `TEST` simula la pierna del trade en logs pero **no toca ninguna cuenta**. No subas tus credenciales a este repo.

---

## Quick start

```bash
pip install -r requirements.txt
python arb_engine.py
```

Eso es todo. Sin `.env`, sin keys. El engine usa los orderbooks públicos de Kalshi y Polymarket y empieza a loguear arbs detectados.

```
[INFO] Email not configured (missing GMAIL_USER/...)
[INFO] Arbitrage scanner started | Mode: TEST | Scan: OG | Max: $10
[INFO] [ARB] Boeing | K_YES+P_NO: 0.930 | Vol:160 Profit:5.0%
[INFO] [ARB] Olivia Rodrigo | K_YES+P_NO: 0.389 | Vol:430 Profit:51.2%
[INFO] [NR-ALERT] Texas SEN | Paxton 9%+ | Profit:11.5% (no vol)
```

- `[ARB]` — arbitraje con volumen real en ambos lados (ejecutable).
- `[NR-ALERT]` — *no rest* (no liquidez del otro lado), solo señal informativa.
- `K_YES+P_NO` o `K_NO+P_YES` — la pierna corresponde a comprar YES en Kalshi + NO en Polymarket (o viceversa) cuando suman <1.

---

## ¿Qué está pasando?

Para cada par mapeado Kalshi↔Polymarket, el engine:

1. Fetchea ambos orderbooks (REST cada ciclo, o WS en tiempo real con `--mode ws`).
2. Calcula el **costo total** de las dos piernas complementarias, incluyendo:
   - Fee de Kalshi por nivel del book (1.75% sobre `min(price, 1-price)`).
   - Slippage real walking the book hasta el `MAX_POSITION`.
   - Fee de Polymarket (0% en el spot, gas implícito).
3. Si `K_price + P_price < 1.0 - threshold`, emite alerta con profit garantizado.
4. Persiste el estado en `dashboard_state.json` para el frontend.

Además **descubre pares nuevos cada 30 min** vía `auto_discover.py` — busca eventos abiertos de Kalshi, los matchea fuzzy contra Polymarket por título/fecha/outcomes, y los agrega a `markets_config_discovered.json`. El loop principal los pickea automático en el siguiente ciclo.

---

## Estructura

```
arbstuff/
├── arb_engine.py              # motor principal (loop, scoring, alertas, output)
├── review_app.py              # Flask SSE bridge — sirve /api/* al frontend
├── auto_discover.py           # descubrimiento genérico Kalshi↔Polymarket
├── politics_discover.py       # discovery especializado para elecciones
├── politics_matcher.py        # heurísticas de matcheo político
├── crypto_jump_arb.py         # vertical separada: arb de jumps en cripto
├── bench_latency.py           # benchmark de latencia REST vs WS
├── helper.py                  # utilidades comunes (logging, retries)
├── merge_markets.py           # mergea configs después de discovery
├── verify_config.py           # validador de configs
├── inspect_poly.py            # debug snippet para el gamma-api de Poly
│
├── markets_config.json              # mappings activos (auto-cargado)
├── markets_config_reviewed.json     # mappings revisados a mano (alta confianza)
├── markets_config_politics.json     # mappings políticos (alta confianza)
├── markets_config_discovered.json   # mappings auto-descubiertos (revisar)
├── discovered_matches_full.json     # output crudo de auto_discover
├── reviewed_matches.json            # subset validado
├── arb_blacklist.json               # tickers a ignorar
├── dashboard_control.json           # toggles runtime (threshold, scan_mode)
│
├── .env.example               # template (TODOS los valores son opcionales)
└── requirements.txt
```

---

## Backend Flask (`review_app.py`) — el puente con el frontend

El motor de detección por sí solo solo escribe `dashboard_state.json`. El frontend ([arbstuff-ui](https://github.com/DGallardoL/arbstuff-ui)) lo consume vía HTTP/SSE. `review_app.py` es ese puente:

```bash
python review_app.py    # arranca en :5000 (no :5060 — ajustá vite si difiere)
```

Rutas que expone (todas bajo `/api`):

| Ruta | Método | Para qué |
|---|---|---|
| `/dashboard/stream` | GET (SSE) | push del `dashboard_state.json` cada 2s |
| `/dashboard/state` | GET | snapshot del estado actual |
| `/dashboard/orderbook` | GET | orderbook live de un par (`?kalshi_ticker=…&poly_token=…`) |
| `/dashboard/pnl` | GET | log de PnL simulado |
| `/dashboard/detection-history` | GET | historial de detecciones |
| `/dashboard/config-stats` | GET | conteos de mappings (reviewed/main/combined) |
| `/dashboard/settings` | POST | actualiza threshold, min_alert_profit, scan_mode |
| `/dashboard/blacklist` | POST/DELETE | agregar/limpiar arbs bloqueados |
| `/config-events` | GET | lista cruzada Kalshi↔Polymarket de eventos mapeados |
| `/politics/events` + `/politics/run` | GET/POST | datos + trigger de `politics_discover.py` |
| `/discover` + `/discovery/{status,run}` | POST/GET | trigger de `auto_discover.py` y status |
| `/data` + `/accept` + `/reject` + `/reset` + `/export` + `/recent-accepts` | varios | endpoints de la página `/review` (validación manual de matches) |

> Las rutas `/api/sports/*` en `review_app.py` son dead-code (este repo no incluye la vertical sports). La UI no las llama.

Flujo end-to-end:

```
arb_engine.py  ──escribe──>  dashboard_state.json
                                    │
                                    │ (lectura cada 2s)
                                    ▼
                            review_app.py  (Flask)
                                    │
                                    │ SSE /api/dashboard/stream
                                    ▼
                            arbstuff-ui  (React, vite proxy :5060→:5000)
```

### Output que consume el frontend

`dashboard_state.json` se regenera cada ciclo (~60-90s) con esta forma:

```json
{
  "timestamp": "2026-05-13T23:32:21",
  "cycle": 1,
  "candidates": [
    {
      "event": "California GOV (primary)",
      "side": "Xavier Becerra",
      "direction": "K_YES+P_NO",
      "k_price": 0.18,
      "p_price": 0.32,
      "total": 0.50,
      "profit_pct": 28.3,
      "volume": 300,
      "k_ticker": "KXCAGOV2ND-26JUN02-2-XBEC",
      "p_token": "1108557541743907..."
    }
  ]
}
```

Tu terminal Bloomberg lee este archivo (o lo subscribe vía `fs.watch`) y renderiza la tabla.

---

## CLI flags

```bash
python arb_engine.py [opciones]

  --mode {og,ws}            REST polling (default) o WebSocket en tiempo real
  --threshold 0.95          umbral de suma de precios (default 0.94)
  --min-profit 1.0          mínimo profit absoluto USD para alertar
  --max-position 10         tamaño hipotético para calcular slippage
  --config FILE [FILE...]   qué archivos de mapping cargar
  --pnl                     imprime el PnL log y sale (modo simulado)
  --live                    [REQUIERE CREDENCIALES] — no usar en este repo
```

---

## Configuración opcional (alertas externas)

`arb/.env.example` lista todas las env vars que el motor lee pero ninguna es necesaria para detectar:

| Variable | Para qué |
|---|---|
| `GMAIL_USER` / `GMAIL_APP_PASSWORD` / `EMAIL_RECIPIENT` | enviar alertas por email |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | enviar alertas a Telegram |
| `TWILIO_*` | alertas WhatsApp vía Twilio |
| `KALSHI_API_KEY_ID` / `KALSHI_PRIVATE_KEY_PATH` | fetch de balance (no necesario para detección) |
| `POLY_*` | fetch de balance (no necesario para detección) |

Si decidís usar alguna, copiá `.env.example → .env` y rellená. **Nunca commitees `.env`** — el `.gitignore` ya lo bloquea.

---

## Auto-discovery: cómo crecen los mappings

```bash
python auto_discover.py
```

- Lista todos los eventos abiertos de Kalshi (status=open, paginado).
- Lista todos los eventos activos de Polymarket (gamma-api).
- Hace fuzzy match por título normalizado + fecha + outcomes.
- Genera candidatos con score de confianza.
- Escribe `discovered_matches_full.json` (todo) y `markets_config_discovered.json` (mappings listos para usar).

Corre standalone si querés un sweep manual, o en background — el motor principal lo schedulea cada 30 min.

Para política específicamente, `politics_discover.py` usa heurísticas extra (nombres de candidatos, distritos, primarias vs general) y produce `markets_config_politics.json`.

---

## Verticales

- **Politics** — `politics_matcher.py` + `politics_discover.py`. Maneja primarias, generales, gobernador, senador, casa.
- **Crypto jumps** — `crypto_jump_arb.py` standalone. Detecta cuando un mercado de "BTC arriba de $X antes de fecha Y" en Kalshi y Polymarket se desincronizan tras un movimiento brusco.
- **Sports, music, entertainment** — el matcher genérico de `auto_discover.py` cubre estos sin código especializado (porque los outcomes son típicamente binarios o pocas opciones).

---

## Notas técnicas

- **Fees**: el cálculo de fee de Kalshi es **por nivel del book**, no sobre el precio promedio. Esto es importante — usar el promedio (como hacía `versionfinpar.py`) sobreestimaba edge en books inclinados. Fix en `arb_engine.py:1040+`.
- **Dedup de fetches**: el mismo ticker se usa para evaluar YES y NO, así que el orderbook se fetchea una sola vez por ciclo.
- **Rate limits**: Kalshi tira 429 si pegás `/events` muy seguido. El discovery lo respeta con backoff exponencial. Si ves 429 ocasionales en el log, es normal.
- **PnL tracking simulado**: en modo TEST, el motor lleva un libro virtual de los trades que *hubiera* hecho. No es ejecución real — solo benchmark del scoring.

---

## Licencia / uso

Privado. Para integración con la terminal de prediction markets propia.
