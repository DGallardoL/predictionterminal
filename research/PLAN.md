# Prediction Factor Model — POC Plan

**Autor:** Damian Gallardo · **Fecha:** 2026-04-23 · **Status:** Plan de POC, listo para Claude Code.

---

## 1. Objetivo del proyecto

Construir un **Proof of Concept** de un servicio web (API + frontend mínimo) que estime **factor models** de retornos de acciones, donde los factores son **cambios en una transformación (logit) de probabilidades de eventos de prediction markets** (Polymarket, y opcionalmente Kalshi en una fase posterior).

**Modelo central:**

$$
r_{j,t} = \alpha_j + \sum_{i=1}^{K} \beta_{j,i} \cdot \Delta \text{logit}(p_{i,t}) + \varepsilon_{j,t}
$$

donde:
- $r_{j,t}$ = retorno log diario de la acción $j$ en $t$
- $p_{i,t}$ = probabilidad implícita del evento $i$ al cierre de $t$ (mid price del contrato YES de Polymarket, en [0,1])
- $\Delta \text{logit}(p_{i,t}) = \log\frac{p_{i,t}}{1-p_{i,t}} - \log\frac{p_{i,t-1}}{1-p_{i,t-1}}$
- $\beta_{j,i}$ = sensibilidad del retorno de $j$ a información sobre el evento $i$
- Standard errors: **HAC (heteroskedasticity- and autocorrelation-consistent)** con `lag = floor(4*(T/100)^(2/9))`

**Interpretación práctica:** `β_{NVDA, recession} = -2.3` significa que cuando la probabilidad implícita de recesión sube 1 "unidad de logit" (ej: de 10% a ~24%, o de 50% a ~73%), NVDA tiende a bajar 2.3% ese día.

---

## 2. Requisitos del curso (pizarrón)

El entregable debe cumplir:

1. **GitHub** con repo público y commits incrementales
2. **CI/CD automatizado** — GitHub Actions (lint + test + build)
3. **Documentación en `.md`** con los "quantas" explicados (matemática del modelo)
4. **ADRs** (Architecture Decision Records) de las decisiones técnicas
5. **OpenAPI** auto-generado (FastAPI lo da gratis)
6. **Infra documentada** (Docker-compose)
7. **README general** con instrucciones claras y descripción
8. **Todo corre en Docker directamente** con `docker-compose up`
9. **Demo/presentación final de 15 min**
10. **Tiene que estar mejor que 3** (el profe no aceptará proyectos flojos)

**Objetivo de calidad (informal):** que el proyecto **se vea profesional**, con arquitectura limpia, docs genuinos y un core quant real aunque pequeño.

---

## 3. Alcance del POC (scope MVP)

**Lo que SÍ entra en el POC:**

- **1 fuente de datos de probabilidades:** Polymarket CLOB API pública (sin auth)
- **3–5 factores hardcodeados** en `factors.yml` (contratos activos con >6 meses de historia y volumen decente)
- **Retornos de acciones vía `yfinance`** (cualquier ticker que el usuario pida)
- **Un endpoint de fit** que corre OLS con HAC y regresa β, t-stats, R², diagnóstico
- **Un endpoint de attribution** que descompone un retorno observado
- **Frontend de una sola página** (HTML estático + Plotly) con form y resultados
- **Redis cache** con TTL 1h para evitar hammer a Polymarket
- **Docker-compose** con 3 servicios: api, web, redis
- **CI pipeline** con ruff + pytest + docker build
- **Docs completas:** README, quantas, 6 ADRs, OpenAPI auto
- **Tests con mocks** (no hacer llamadas reales a Polymarket en CI)

**Lo que NO entra en el POC** (se deja para iteración posterior):

- Kalshi (solo Polymarket en POC)
- Autenticación / usuarios / guardar fits
- Base de datos persistente (Postgres)
- Frontend sofisticado (React, etc.)
- Multiples tickers cross-sectional (solo 1 a la vez)
- Rolling betas o time-varying parameters
- Backtesting de estrategia sobre residuos
- Detección automática de multicolinealidad con pruning
- Lag structure (asumimos contemporaneidad)

---

## 4. Arquitectura

```
┌──────────────┐         ┌─────────────────┐         ┌──────────────────┐
│              │         │                 │  HTTP   │ Polymarket CLOB  │
│   Browser    │◀───────▶│   FastAPI       │────────▶│ clob.polymarket  │
│  (web:8080)  │  HTTP   │   (api:8000)    │         │ .com             │
│              │         │                 │         │                  │
└──────────────┘         └────────┬────────┘         └──────────────────┘
                                  │
                                  │                  ┌──────────────────┐
                                  │  HTTP            │ Polymarket Gamma │
                                  ├─────────────────▶│ gamma-api        │
                                  │                  │ .polymarket.com  │
                                  │                  └──────────────────┘
                                  │
                                  │                  ┌──────────────────┐
                                  │  yfinance        │  Yahoo Finance   │
                                  ├─────────────────▶│  (implicit)      │
                                  │                  └──────────────────┘
                                  │
                                  │  redis protocol  ┌──────────────────┐
                                  └─────────────────▶│  Redis           │
                                                     │  (redis:6379)    │
                                                     │  cache, TTL=1h   │
                                                     └──────────────────┘
```

**Tres contenedores:**
1. **api** — FastAPI en Python 3.12, puerto 8000, core quant
2. **web** — nginx sirviendo HTML + JS estático, puerto 8080
3. **redis** — redis:7-alpine, puerto interno 6379, cache

---

## 5. API de Polymarket (investigación confirmada)

### 5.1 Gamma API (descubrir mercados)

**Base URL:** `https://gamma-api.polymarket.com`

**Endpoint para obtener metadata de mercado por slug:**
```
GET https://gamma-api.polymarket.com/markets?slug={slug}
```

Respuesta relevante (campo clave):
- `clobTokenIds`: string JSON con array de 2 token IDs — `[yes_token_id, no_token_id]`
- `startDate`, `endDate`, `closed`, `active`
- `volumeClob`, `liquidityClob`

**El slug sale de la URL del frontend de Polymarket:** `https://polymarket.com/event/fed-decision-in-october` → slug = `fed-decision-in-october`.

### 5.2 CLOB API (histórico de precios)

**Base URL:** `https://clob.polymarket.com`

**Endpoint de histórico:**
```
GET https://clob.polymarket.com/prices-history
    ?market={YES_TOKEN_ID}
    &interval=max          # o 1m, 1w, 1d, 6h, 1h
    &fidelity=1440         # minutos entre puntos; 1440 = daily
    &startTs={unix_sec}    # opcional
    &endTs={unix_sec}      # opcional
```

**Sin autenticación requerida.**

**Rate limit (confirmado):** 1000 requests / 10s para `/prices-history`. Nuestro uso será mucho menor.

**Response:**
```json
{
  "history": [
    {"t": 1706745600, "p": 0.42},
    {"t": 1706832000, "p": 0.45},
    ...
  ]
}
```

### 5.3 ⚠️ Trampa conocida: mercados resueltos

**De GitHub issue #216 del py-clob-client:** Para mercados ya **resueltos**, el endpoint solo retorna datos con `fidelity >= 720` minutos (12h). Con fidelity < 720 devuelve array vacío.

**Implicación para el POC:** usamos `fidelity=1440` (diario) siempre. Esto funciona tanto para mercados activos como resueltos.

### 5.4 Ejemplo de flujo end-to-end

```python
import requests

# Paso 1: Gamma API — obtener clobTokenIds desde slug
r = requests.get(
    "https://gamma-api.polymarket.com/markets",
    params={"slug": "fed-decision-in-october"}
)
market = r.json()[0]
token_ids = json.loads(market["clobTokenIds"])  # viene como string JSON
yes_token_id = token_ids[0]  # YES outcome

# Paso 2: CLOB API — obtener histórico del token YES
r2 = requests.get(
    "https://clob.polymarket.com/prices-history",
    params={
        "market": yes_token_id,
        "interval": "max",
        "fidelity": 1440  # daily
    }
)
hist = r2.json()["history"]
# [{"t": 1706745600, "p": 0.42}, ...]
```

---

## 6. Selección de factores (sugerencia inicial)

Criterios de selección:
- Mercados con **≥6 meses** de historia continua
- **Volumen total ≥ $500K** (proxy de liquidez)
- **Spread mediano ≤ 3 cents**
- Ideal: mercado **activo** (no resuelto), así datos fluyen incluso mientras se corre el proyecto

**Candidatos iniciales** (a validar en el momento de armar `factors.yml`):

| Factor ID | Descripción | Relevancia a equities |
|---|---|---|
| `fed_cuts_ge_2_2026` | Fed ≥2 cortes en 2026 | Rates-sensitive sectors, duration |
| `us_recession_2026` | Recesión US en 2026 | Cyclicals, financials |
| `trump_approval_h1_2026` | Trump approval >X% final Q2 | Policy-sensitive names |
| `spx_ath_by_year_end` | S&P 500 ATH antes de fin de año | Directional sentiment |
| `btc_above_100k_year_end` | BTC > $100K fin de año | Crypto-exposed equities |

**Nota:** durante la construcción del proyecto, Damian debe ir a `polymarket.com`, elegir mercados que cumplan los criterios, copiar los slugs, y actualizar `factors.yml`. El código no depende de qué mercados sean — es dato de configuración.

---

## 7. Estructura del repositorio

```
prediction-factor-model/
├── README.md
├── PLAN.md                          # este archivo (mover al research/ en el repo final)
├── LICENSE
├── .gitignore
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
│   ├── quants.md                    # matemática del modelo: logit, OLS, HAC
│   ├── architecture.md              # diagramas y flujo
│   ├── adrs/
│   │   ├── 0001-use-fastapi.md
│   │   ├── 0002-logit-transform.md
│   │   ├── 0003-hac-newey-west.md
│   │   ├── 0004-redis-cache-ttl.md
│   │   ├── 0005-no-persistence-poc.md
│   │   ├── 0006-timezone-alignment.md
│   │   └── 0007-daily-fidelity.md
│   └── openapi.json                 # generado post-deploy, commiteado
├── api/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── requirements.txt             # lock file para docker
│   ├── src/pfm/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI app, endpoints
│   │   ├── config.py                # settings (Pydantic BaseSettings)
│   │   ├── factors.yml              # config de factores
│   │   ├── model.py                 # OLS con HAC, logit transform
│   │   ├── attribution.py           # descomposición por factor
│   │   ├── cache.py                 # Redis wrapper
│   │   ├── sources/
│   │   │   ├── __init__.py
│   │   │   ├── polymarket.py        # Gamma + CLOB clients
│   │   │   └── equity.py            # yfinance wrapper
│   │   └── schemas.py               # Pydantic models para request/response
│   └── tests/
│       ├── conftest.py
│       ├── test_model.py            # unit tests del OLS con datos sintéticos
│       ├── test_logit.py            # unit tests de la transformación
│       ├── test_polymarket.py       # tests con responses mockeadas
│       └── test_endpoints.py        # integration con TestClient
└── web/
    ├── Dockerfile
    ├── nginx.conf
    └── index.html                   # form + Plotly
```

---

## 8. Endpoints de la API

### 8.1 `GET /health`
Healthcheck para docker y CI.
```json
{"status": "ok", "version": "0.1.0"}
```

### 8.2 `GET /factors`
Lista factores disponibles desde `factors.yml`.
```json
{
  "factors": [
    {
      "id": "fed_cuts_ge_2_2026",
      "name": "Fed ≥2 cuts in 2026",
      "slug": "fed-decision-in-october",
      "source": "polymarket",
      "description": "..."
    },
    ...
  ]
}
```

### 8.3 `POST /fit`
Body:
```json
{
  "ticker": "NVDA",
  "factors": ["fed_cuts_ge_2_2026", "us_recession_2026"],
  "start": "2025-10-01",
  "end": "2026-04-01"
}
```

Response:
```json
{
  "ticker": "NVDA",
  "n_obs": 124,
  "start": "2025-10-01",
  "end": "2026-04-01",
  "model": {
    "alpha": 0.0012,
    "r_squared": 0.184,
    "r_squared_adj": 0.170,
    "f_stat": 13.4,
    "f_pvalue": 3.2e-6,
    "residual_std": 0.019
  },
  "factors": [
    {
      "id": "fed_cuts_ge_2_2026",
      "beta": 0.0082,
      "std_err": 0.0034,
      "t_stat": 2.41,
      "p_value": 0.017,
      "ci_low": 0.0015,
      "ci_high": 0.0149
    },
    {
      "id": "us_recession_2026",
      "beta": -0.023,
      "std_err": 0.009,
      "t_stat": -2.55,
      "p_value": 0.012,
      "ci_low": -0.041,
      "ci_high": -0.005
    }
  ],
  "diagnostics": {
    "vif": {"fed_cuts_ge_2_2026": 1.12, "us_recession_2026": 1.12},
    "durbin_watson": 1.94,
    "hac_lag": 3
  }
}
```

### 8.4 `POST /attribution`
Body:
```json
{
  "ticker": "NVDA",
  "factors": ["fed_cuts_ge_2_2026", "us_recession_2026"],
  "start": "2025-10-01",
  "end": "2026-04-01",
  "date": "2026-03-15"
}
```

Response:
```json
{
  "date": "2026-03-15",
  "observed_return": -0.034,
  "predicted_return": -0.028,
  "residual": -0.006,
  "contributions": [
    {"id": "alpha", "contribution": 0.0012},
    {"id": "fed_cuts_ge_2_2026", "delta_logit": -0.15, "beta": 0.0082, "contribution": -0.00123},
    {"id": "us_recession_2026", "delta_logit": 1.2, "beta": -0.023, "contribution": -0.0276}
  ]
}
```

---

## 9. Matemática (resumen para docs/quants.md)

### 9.1 Transformación logit
$$
\text{logit}(p) = \log\frac{p}{1-p}
$$

Justificación:
- Mapea [0,1] → ℝ → estabiliza varianza
- Un cambio de +1 en logit siempre representa la misma cantidad de "información"
- Evita no-linealidades cerca de 0 y 1

**Clipping:** para evitar `log(0)` o `log(∞)`, clipear $p \in [\epsilon, 1-\epsilon]$ con $\epsilon = 0.01$.

### 9.2 HAC standard errors
Autocorrelación esperada en errores (retornos financieros tienen momentum/reversal).
Bandwidth con automatic bandwidth selection: `lag = floor(4*(T/100)^(2/9))`.

### 9.3 OLS setup
$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

con $\mathbf{X}$ incluyendo constante y los $\Delta \text{logit}(p_{i,t})$ como columnas.

### 9.4 Alineación temporal (IMPORTANTE)
- Polymarket: precio del mercado al cierre UTC de día $t$
- Acciones: close price al 4pm ET (20:00 UTC aprox.)
- **Decisión:** tomar ambos al final del día UTC, usar retorno $t$ vs $\Delta \text{logit}$ de $t$ contemporáneamente. Esto hace que sea un modelo "correlacional" más que causal, pero es honesto y está documentado en ADR-0006.

---

## 10. Plan de trabajo (orden sugerido para Claude Code)

Claude Code recibirá este folder. El orden de construcción recomendado:

### Fase 1 — Scaffold (1 hora)
1. Crear estructura de carpetas
2. `pyproject.toml` con deps: `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `statsmodels`, `pandas`, `numpy`, `yfinance`, `redis`, `httpx`, `pyyaml`, `pytest`, `ruff`
3. `Dockerfile` del API (multi-stage, Python 3.12-slim)
4. `Dockerfile` del web (nginx:alpine)
5. `docker-compose.yml` con 3 servicios

### Fase 2 — Core quant (2 horas)
6. `model.py`: funciones `logit_transform`, `delta_logit`, `fit_ols_hac`, `compute_diagnostics`. Probar con datos sintéticos primero.
7. `tests/test_model.py`: DGP conocido (simular betas, verificar recovery)
8. `attribution.py`: descomposición por factor

### Fase 3 — Data sources (2 horas)
9. `sources/polymarket.py`:
   - `get_market_metadata(slug)` → llama Gamma API
   - `get_price_history(token_id, start, end)` → llama CLOB, retorna DataFrame con columna `date, price`
10. `sources/equity.py`:
    - `get_returns(ticker, start, end)` → yfinance, log returns
11. `cache.py`: wrapper de Redis con fallback a "no cache" si redis no está disponible
12. `tests/test_polymarket.py`: mock con `respx` o `httpx_mock`

### Fase 4 — API (2 horas)
13. `schemas.py`: Pydantic models para request/response
14. `main.py`: FastAPI app con los 4 endpoints, CORS habilitado, lifespan para redis
15. `config.py`: settings con `POLYMARKET_GAMMA_URL`, `POLYMARKET_CLOB_URL`, `REDIS_URL`, `CACHE_TTL_SECONDS`
16. `factors.yml` con 3–5 factores iniciales (Damian llenará slugs reales)
17. `tests/test_endpoints.py` con `TestClient`

### Fase 5 — Frontend (1 hora)
18. `web/index.html`: form con ticker input, checkboxes de factores (fetcheados del /factors), date range, botón Fit. Al submit, hace POST a /fit y muestra tabla de coeficientes + bar chart de contribuciones (Plotly CDN).
19. `web/nginx.conf`: proxy_pass de `/api/*` al contenedor api.

### Fase 6 — Docs (2 horas)
20. `README.md`: overview, quickstart (`docker-compose up`), example curl, link a docs
21. `docs/quants.md`: derivación completa del modelo, logit, HAC
22. `docs/architecture.md`: diagrama + descripción de cada servicio
23. Los 7 ADRs (plantilla MADR: status, context, decision, consequences)
24. `docs/openapi.json`: generar con `curl http://localhost:8000/openapi.json > docs/openapi.json` después de levantar el api

### Fase 7 — CI/CD (1 hora)
25. `.github/workflows/ci.yml`: 
    - Job `lint`: ruff check
    - Job `test`: pytest con coverage
    - Job `build`: `docker-compose build`
    - Todo debe pasar antes de merge a main

### Fase 8 — Polish (1 hora)
26. `.gitignore`, `LICENSE` (MIT), banner en README, badges de CI
27. Smoke test manual: `docker-compose up`, abrir `localhost:8080`, correr un fit
28. Preparar slides de demo (no hacerlas en esta fase, pero asegurar que la narrativa funciona)

**Tiempo total estimado:** ~12 horas concentradas. Realista en 2–3 días de trabajo.

---

## 11. Guión de demo (15 min)

1. **[1 min] Contexto.** Problema: ¿cuánto del retorno de una acción es explicable por información de prediction markets?
2. **[2 min] Modelo.** Escribir ecuación en slide. Explicar logit. Explicar HAC.
3. **[3 min] Arquitectura.** Diagrama de 3 contenedores. Mostrar `docker-compose up` arrancando en vivo.
4. **[4 min] Demo en vivo.** Abrir web, elegir NVDA, seleccionar 2–3 factores, correr fit. Explicar el output: β positivo significativo en factor X, β negativo en factor Y. Correr attribution para una fecha con evento conocido.
5. **[2 min] OpenAPI.** Mostrar Swagger UI en `localhost:8000/docs`. Mostrar que todo está documentado.
6. **[2 min] Ingeniería.** Mostrar GitHub con CI verde, estructura de ADRs, quants.md.
7. **[1 min] Limitaciones y siguientes pasos.** Honest list: no hay causalidad, contemporaneidad, pocos factores. Próximo paso: Kalshi, cross-sectional, rolling betas.

---

## 12. Riesgos conocidos y mitigaciones

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Polymarket cambia su API | Baja | Endpoints documentados oficiales; versión del cliente fijada |
| Rate limit 1000/10s insuficiente | Muy baja | Cache Redis 1h; uso real será <<10 req/fit |
| Clipping de probabilidades causa bias | Media | ADR explícito; default ε=0.01; exponer en config |
| Multicolinealidad entre factores | Alta | Reportar VIF; avisar si VIF>5; no auto-dropear en POC |
| Endpoint con mercados resueltos falla sub-daily | Conocido | Siempre usar fidelity=1440 |
| Timezones mal alineados | Media | ADR explícito; normalizar a cierre UTC 00:00 |
| yfinance rate-limited | Baja | Cache local; usar agentes estándar |

---

## 13. Referencias y recursos

- Polymarket API docs: https://docs.polymarket.com/api-reference/introduction
- Prices-history endpoint: https://docs.polymarket.com/api-reference/markets/get-prices-history
- Rate limits: https://docs.polymarket.com/api-reference/rate-limits
- Gamma API (markets by slug): https://docs.polymarket.com/api-reference/markets/get-market-by-slug
- Fetching markets guide: https://docs.polymarket.com/market-data/fetching-markets
- py-clob-client (referencia de Python): https://github.com/Polymarket/py-clob-client
- Conocida trampa de fidelity en resueltos: https://github.com/Polymarket/py-clob-client/issues/216
- HAC covariance with automatic bandwidth selection
- statsmodels OLS con cov_type='HAC': https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html
