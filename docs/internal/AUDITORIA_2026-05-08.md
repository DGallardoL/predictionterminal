# Auditoría completa — `proyectofuentes`
**Fecha:** 2026-05-08 · **Auditores:** 24 sub-agentes paralelos · **Objetivo:** "Yahoo Finance de prediction markets + Strats Hub"

---

## ⚡ ESTADO POST-IMPLEMENTACIÓN (3 olas, 30 agentes paralelos)

**Wave 1 (10 agentes)**: 832 tests · 123 endpoints
**Wave 2 (10 agentes)**: 961 tests · 151 endpoints
**Wave 3 (10 agentes)**: **1251 tests pasando · 194 endpoints · 0 failures** ← actual

### Wave-3 additions (2026-05-08, sesión 3)

| Feature | Endpoints | Tests |
|---|---|---|
| Manifold Markets + PredictIt + Multi-venue search | 5 | 28 |
| Stock alt sources (Tiingo + Stooq) + delisted handling + sources/health | 3 | 26 |
| BLS jobless claims + 14 new FRED series + macro calendar + overlay | 6 | 60 |
| Earnings Whisper from PM + Vol Surface PM + Counterfactual Backtest | 6 | 36 |
| Whale Mirror Portfolio + Smart Money Divergence + Auto-Hedge Bot | 7 | 45 |
| Property-based tests (hypothesis) + Golden file regression (9 snapshots) | — | 26 |
| Mypy strict + py.typed + ruff hardening (PL/PERF/PTH/ASYNC) | — | — |
| Frontend wave-3: dark theme, notifications, shortcuts cheatsheet, watchlist bulk, freshness badges, SSE multiplex client, print stylesheet | — | — |
| Sentiment NER tagger + entity-factor map | 4 | 27 |
| Auth + rate limiting + tier gates (Free/Pro/Quant/Enterprise) | 6 | 42 |
| CHANGELOG + 2 new ADRs (factor curation + frontend stack) + DEMO_SCRIPT.md + PRODUCTION_CHECKLIST.md + USER_GUIDE update | — | — |
| **conftest.py fixture** que limpia caches volátiles entre tests (eliminó 5 flaky tests) | — | — |

### Implementado en esta sesión

| # | Feature | Status | Endpoints / archivos |
|---|---|---|---|
| 1 | Embargo walk-forward (Lopez de Prado) | ✅ | `pfm/advanced.py` param `embargo_size` |
| 2 | BH-FDR multitest | ✅ | `pfm/multitest.py` + `POST /quant/multitest/bh` |
| 3 | 4-quarter Sharpe enforcer | ✅ | `pfm/strategy_verdict.py:quarterly_stability_test` + `POST /quant/quarterly-stability` |
| 4 | Reverse Factor Finder | ✅ | `POST /reverse-finder` (killer feature, demo en 800ms) |
| 5 | Prediction-Driven Alpha Scanner | ✅ | `POST /alpha/prediction-driven` |
| 6 | Alpha Graveyard público | ✅ | 6 entries + 6 death-certificates + `GET /alpha-hub/graveyard` |
| 7 | Comparison tool side-by-side | ✅ | `GET /terminal/compare?slugs=a,b,c[,d]` con corr matrix + pairs trade z-score |
| 8 | Export universal CSV/JSON | ✅ | `?format=csv\|json` en `/terminal/market` + `POST /terminal/export/bulk` |
| 9 | Portfolio Optimizer (HRP+MV+min-var+ERC+EW) | ✅ | `POST /strategies/optimize` con efficient frontier + MC drawdown |
| 10 | Alert engine multi-canal | ✅ | SQLite + Slack/Discord/Webhook(HMAC)/InApp + 8 endpoints `/alerts/*` |
| 11 | Decay tracking | ✅ | `GET /alpha/decay` + `GET /alpha/{id}/rolling-sharpe` |
| 12 | Calendar unificado | ✅ | `GET /terminal/calendar` (resolution+earnings+macro mix) |
| 13 | Cache utils refactor (DRY) | ✅ | `pfm/cache_utils.py` + 4 módulos refactorizados |
| 14 | CORS restringido + security headers | ✅ | env `CORS_ORIGINS`, X-Frame, HSTS, Referrer-Policy |
| 15 | nginx gzip + rate limit + cache headers | ✅ | `web/nginx.conf` |
| 16 | Multi-worker uvicorn | ✅ | `--workers 4` en Dockerfile |
| 17 | Redis persistence | ✅ | volumen + `appendonly yes` |
| 18 | `.env.example` | ✅ | template completo |
| 19 | Prometheus `/metrics` | ✅ | `pfm/observability.py` |
| 20 | `/health/detail` mejorado | ✅ | redis ping, uptime, git_sha |
| 21 | CI: pip-audit + Codecov + timeouts | ✅ | `.github/workflows/ci.yml` |
| 22 | pre-commit hooks | ✅ | `.pre-commit-config.yaml` |
| 23 | Cleanup: 7 .bak files + .gitignore | ✅ | -2.3 MB |
| 24 | Frontend: Cmd-K modal | ✅ | `Cmd+K` global, autocomplete, recents en localStorage |
| 25 | Frontend: deep-linking URL state | ✅ | `?mode=...&market=...&compare=...` shareable |
| 26 | Frontend: Share button | ✅ | copy-to-clipboard + toast |
| 27 | Frontend: Alpha Graveyard tab | ✅ | tab visible con tabla pública |
| 28 | Frontend: disclaimer footer permanente | ✅ | "Not investment advice" + timestamp UTC |

### Pendiente (P2 — sprints futuros)

- News Causal Chain pipeline (#28)
- Resolution P&L Tree (#29)
- Macro Event Playbook (#30)
- Embed widgets + OG images (#31)
- Replay Mode (#32)
- Auto-generated Alpha Lab (#33)
- SSE multiplex refactor (sigue funcionando con polling)
- Split de `main.py` 4053 → routers separados (cosmético, todo funciona)
- Async refactor de loops N+1 a Polymarket
- Quote page detallada por contrato (UI dedicada)
- Watchlist sticky con sparklines (existe básico)


---

## 0. Veredicto en una línea

**Ingeniería sólida** (1090 factores, 61 endpoints, 469 tests, ruff limpio, 7 ADRs, math correcta) **pero a 4 sprints de ser un producto serio**: faltan UX patterns must-have (deep-linking, Cmd-K, watchlist sticky, comparison, export, alerts multi-canal), 5 fallos cuant críticos (embargo walk-forward, BH-FDR, 4Q enforcer, decay tracking, deflated-Sharpe en tier-up), CORS abierto, Redis ephemeral, gzip nginx ausente, y el moat real ("PM ↔ equity") está sin explotar comercialmente.

---

## 1. Estado por área (resumen de cada agente)

| Área | Score | Hallazgo más crítico |
|---|---|---|
| Backend API | 6/10 | CORS `["*"]`, 17 endpoints sin `response_model`, /fit & rank son sync con loops N+1 a Polymarket |
| Quant rigor | 5/10 | HAC ✓, logit ✓, VIF ✓ — pero **sin embargo walk-forward, sin BH-FDR, sin 4Q enforcer automático, sin decay tracking** |
| α Hub / Strategies | 6/10 | 88 cards sin `net_sharpe_after_tc`, `hit_rate`, `max_dd`, `decay_indicator`, `regime_flags` estructurados; graveyard sólo en docs, no público |
| Terminal modules | 7/10 | 27 endpoints sólidos pero `_cache_get/_set` reinventado en 8 módulos; sin screener avanzado, sin compare side-by-side, sin export |
| Frontend UX | 7/10 | Design system Bloomberg-tier ✓, pero **sin deep-linking, sin Cmd-K, sin copy-to-clipboard, sin export, mobile roto <768px, sin glossary** |
| Data sources | 7/10 | Caching Redis ✓, UTC ✓ — pero sin validación min-obs en load, sin handling de delisted stocks, sin fallback yfinance, política sesgada (35% del catálogo) |
| Tests / CI | 7/10 | 469 tests ✓, coverage gate selectivo (sólo 6 files), **sin golden files, sin mypy, sin property-based, sin pre-commit, sin perf tests** |
| Docs / ADRs | 8/10 | 7 ADRs genuinos ✓, USER_GUIDE ✓, alpha-reports v17/v18 ✓ — pero README sin curls reales, sin screenshots, sin Mermaid |
| Performance | 5/10 | Sync `httpx.Client` dentro de async (bloquea event loop), TTL monolítico 1h para todo, **sin gzip nginx, 1 worker uvicorn**, `.iterrows()` en hot paths |
| Real-time | 5/10 | SSE existe pero **polling sync 2s**, sin multiplex, sin heartbeat explícito, alerts solo browser-notif (sin email/Slack/webhook) |
| Seguridad | 6/10 | No secrets hardcoded ✓, Docker non-root ✓ — pero **CORS abierto, sin rate limit, sin CSP/HSTS, sin pip-audit en CI** |
| Code quality | 7/10 | ruff limpio, type hints modernos — pero `main.py` 4022 líneas, `schemas.py` 1796, `fit_endpoint` 295 líneas, 8 módulos con caché reinventada, 15 `except Exception` bare, 16 tests sin assertions |

---

## 2. Top 30 mejoras priorizadas

### **P0 — Bloquean grading o demo (semana 1-2)**

| # | Mejora | Tiempo | Archivo principal |
|---|---|---|---|
| 1 | Cerrar CORS a domain específico + headers `X-Frame-Options`/`HSTS`/CSP | 30m | `pfm/main.py:293`, `web/nginx.conf` |
| 2 | Gzip nginx + multi-worker uvicorn (`--workers 4`) | 15m | `web/nginx.conf:5`, `api/Dockerfile:47` |
| 3 | Implementar **embargo walk-forward** (Lopez de Prado) en `walk_forward_backtest` | 2h | `pfm/advanced.py:196-202` |
| 4 | Implementar **BH-FDR sobre los 88 pares** (no existe en código) | 3h | new `pfm/multitest.py` + `strategy_verdict.py` |
| 5 | **4-quarter Sharpe enforcer automático** — exigir Sharpe>threshold en 4Q antes de tier-up | 2h | `pfm/strategy_verdict.py:105` |
| 6 | Convertir `httpx.Client` sync → `httpx.AsyncClient` en endpoints async (ahora bloquea event loop) | 2h | `pfm/terminal_live_stream.py`, `pfm/main.py:267` |
| 7 | Paralelizar loops N+1 a Polymarket con `asyncio.gather` (`/fit`, `/factors/rank`, `/factors/discover`) | 3h | `pfm/main.py:529-569,3692-3707` |
| 8 | **Deep-linking URL state** (`?market=X&compare=Y,Z&tab=...`) — desbloquea sharing | 1h | `web/index.html` |
| 9 | **Cmd-K search global** con autocomplete sobre 1090 factores | 3h | `web/index.html` + `/terminal/search-index` |
| 10 | Agregar `response_model` a 17 endpoints terminal sin schema declarado | 2h | varios `pfm/terminal_*.py` |
| 11 | Restringir Redis con persistence + volumen (`appendonly yes`) | 10m | `docker-compose.yml`, nuevo `redis.conf` |
| 12 | Crear `.env.example` + remover hardcodes en `docker-compose.yml` | 10m | nuevo `.env.example` |
| 13 | README: curls reales + screenshots + diagrama Mermaid + TOC | 2h | `README.md` |

### **P1 — High-value features (semana 3-6)**

| # | Mejora | Tiempo | Notas |
|---|---|---|---|
| 14 | **Quote page detallada** por contrato (52w-range, holders, sparkline grande, peers, news) — endpoint `/terminal/quote/{slug}` | 1d | reemplaza hero actual |
| 15 | **Watchlist sticky panel** con sparklines, sortable, alerts inline — `POST /terminal/watchlist/snapshot` | 1d | reusa SSE existente |
| 16 | **Comparison tool side-by-side** N≤4 contratos — `GET /terminal/compare?slugs=a,b,c` con corr matrix + pairs trade z-score | 1d | nuevo `pfm/terminal_compare.py` |
| 17 | **Export CSV/PDF** universal con `?format=csv\|pdf\|png\|json` query param + WeasyPrint para PDF | 1d | nuevo `pfm/terminal_export.py` |
| 18 | **Alert engine multi-canal** (price-cross, vol-spike, z-score, news-volume, signal-flip, decay) con email + Slack + Discord + webhook | 2-3d | nuevo `pfm/alerts/` (SQLite + asyncio task) |
| 19 | **SSE multiplexado** — un stream por user que dispatchea book/tape/tick/alert/news con heartbeat 10s + reconnect backoff | 2d | refactor `pfm/terminal_live_stream.py` |
| 20 | **Top movers homepage con sparklines** + theme heatmap interactivo + tabs (Gainers/Losers/Active/New/Resolving) | 6h | extender `/terminal/overview` |
| 21 | **Calendario unificado** (resoluciones + earnings + macro FRED) — `GET /terminal/calendar?kinds=resolution,earnings,macro` | 1d | nuevo `pfm/terminal_calendar_unified.py` |
| 22 | **Decay tracking** — rolling Sharpe 30d vs full, alarma si <50% baseline → demote auto a C_TENTATIVE | 4h | new `pfm/decay_monitor.py` |
| 23 | Agregar campos críticos al schema de alpha cards: `net_sharpe_after_tc`, `hit_rate`, `max_dd`, `decay_indicator`, `regime_flags`, `capacity_usd`, `quarters_positive` | 4h | `web/data/alpha_strategies.json` + `pfm/alpha_hunter.py` |
| 24 | **Alpha Graveyard público** (`/web/data/alpha_graveyard.json` + endpoint `/alpha-hub/graveyard`) — visible en UI con causa de muerte | 4h | nuevo |
| 25 | **Portfolio Optimizer** (HRP default + Markowitz + risk-parity + Black-Litterman) — `POST /strategies/optimize` con efficient frontier + MC drawdown | 2d | nuevo `pfm/portfolio_optimizer.py` |

### **P2 — Killer differentiators (sprint 5-8)**

| # | Mejora | Esfuerzo | Por qué killer |
|---|---|---|---|
| 26 | **Reverse Factor Finder** — input ticker → top-5 PM markets que mejor explican return (decomp. ΔR²) | S | mind-blow demo en 800ms |
| 27 | **Prediction-Driven Alpha Scanner** — input PM market → basket equity con expected move por β·Δlogit | M | flagship feature, define el producto |
| 28 | **News → Market → Stock causal chain** — pipeline noticia → Δprob PM → Δprice equity con cadena renderizada | M | screenshot viral en Twitter |
| 29 | **Resolution P&L Tree** — para cada posición abierta, árbol de "if YES then +X% / if NO then -Y%" | S | imposible sin factor exposures |
| 30 | **Macro Event Playbook** — pre-FOMC: tabla "PM-implied vs IV-implied prob" con divergence-trade auto-suggested | L | pre-event edge único |
| 31 | **Embed widgets** (Twitter cards + iframe) con `og:image` matplotlib server-rendered → growth viral | M | crece sin spend |
| 32 | **Replay Mode** — rebobinar a fecha pasada (ej. 2024-11-04 election night), operar "como si fuera ese día" | L | educational + memorable |
| 33 | **Auto-generated Alpha Lab** — sistema corre `factor_model_pro` + `triple_barrier` + `walk_forward` sobre random subsets, surfacea las que pasan verdict | L | producto auto-creciente |
| 34 | **PM-VIX composite** — risk-on/off index derivado de odds de tail-risk markets | S | señal propia única |
| 35 | **Cross-venue arb scanner** — Kalshi vs Polymarket spread ≥2% sostenido 30min con tradeable size | M | money-maker |

### **P2 — Tech debt (cuando haya tiempo)**

| # | Mejora | Tiempo |
|---|---|---|
| 36 | Centralizar `_cache_get/_set` en `pfm/cache_utils.py` — 8 módulos terminal_* lo reinventan (~200 líneas duplicadas) | 2h |
| 37 | Dividir `main.py` (4022 líneas) en `pfm/routers/{health,factors,analysis,strategies}.py` | 6h |
| 38 | Dividir `schemas.py` (1796 líneas) en `pfm/schemas/{base,strategies,terminal}.py` | 3h |
| 39 | Refactorizar `fit_endpoint` (295 líneas) → `_validate_inputs/_prepare_data/_fit_model/_format_response` | 4h |
| 40 | Reemplazar `global` keyword por `@functools.lru_cache` en 8 módulos | 2h |
| 41 | Especificar excepciones en 15 `except Exception:` bare | 1.5h |
| 42 | Pre-commit hooks (ruff + ruff-format + EOF + trailing-whitespace) | 30m |
| 43 | Mypy strict en CI + `py.typed` marker | 2h |
| 44 | Golden-file regression tests para los 27 endpoints Terminal | 4h |
| 45 | Property-based testing (hypothesis) para logit/delta_logit/HAC | 3h |
| 46 | Codecov integration en CI (en vez de artifact silencioso) | 30m |
| 47 | `.github/dependabot.yml` weekly | 10m |
| 48 | /metrics Prometheus endpoint + structlog JSON formatter | 4h |
| 49 | Limpiar 7 `factors.yml.bak.*` (2.3 MB) y agregar `*.bak*` a `.gitignore` | 5m |
| 50 | Eliminar 16 tests sin `assert` (falsa cobertura) | 1h |

---

## 3. Roadmap 8 sprints (cada uno = 1 semana)

### **Sprint 1 — Foundation hardening (P0)**
- CORS + nginx headers + gzip + workers (#1, #2)
- async + paralelización Polymarket (#6, #7)
- response_model en 17 endpoints (#10)
- .env.example + Redis volume (#11, #12)
- README curls + Mermaid (#13)
**Deliverable**: deployable a Fly.io/Render con seguridad mínima y latencia 5-10x menor.

### **Sprint 2 — Quant rigor (P0)**
- Embargo walk-forward (#3)
- BH-FDR sobre 88 pares (#4)
- 4Q enforcer automático (#5)
- Decay tracking (#22)
- Schema fields nuevos en alpha cards (#23)
**Deliverable**: el verdict-engine es defendible ante un profesor PhD.

### **Sprint 3 — Yahoo Finance parity (P0+P1)**
- Deep-linking (#8)
- Cmd-K search (#9)
- Quote page detallada (#14)
- Watchlist sticky con sparklines (#15)
- Top movers + heatmap homepage (#20)
**Deliverable**: navegación profesional, retención clara.

### **Sprint 4 — Compare + Export + Calendar (P1)**
- Comparison tool N≤4 (#16)
- Export CSV/PDF universal (#17)
- Calendar unificado (#21)
- Alpha Graveyard público (#24)
**Deliverable**: feature-parity con Yahoo Finance + honestidad intelectual.

### **Sprint 5 — Real-time + Alerts (P1)**
- SSE multiplex (#19)
- Alert engine multi-canal (#18)
- Portfolio Optimizer (#25)
**Deliverable**: data hub real-time + alerts profesionales.

### **Sprint 6 — Define the wedge (P2 killer)**
- Reverse Factor Finder (#26)
- Prediction-Driven Alpha Scanner (#27)
**Deliverable**: 2 features que ningún competidor tiene → diferencial defendible.

### **Sprint 7 — Visualize the moat (P2 killer)**
- News Causal Chain (#28)
- Resolution P&L Tree (#29)
- Macro Event Playbook (#30)
**Deliverable**: producto cuenta la historia "PM ↔ equity" visualmente.

### **Sprint 8 — Growth + Defensibility (P2 killer)**
- Embed widgets (#31)
- Replay Mode (#32)
- Auto-generated Alpha Lab (#33)
- Onboarding tour (3 steps)
- Demo narrative pulido
**Deliverable**: producto se vende solo + plan de monetización ejecutable.

---

## 4. Killer differentiators (TOP-5 ranking impacto/esfuerzo)

| Rank | Feature | Score | Por qué |
|---|---|---|---|
| 1 | **Reverse Factor Finder** | 9.0 | S effort, todo existe, demo en 800ms |
| 2 | **Prediction-Driven Alpha Scanner** | 7.0 | M effort, define el pitch en 30s |
| 3 | **News Causal Chain** | 6.5 | M effort, viraliza en Twitter |
| 4 | **Resolution P&L Tree** | 7.0 | S effort, mind-blow visual |
| 5 | **Macro Event Playbook** | 6.0 | L effort pero high ceiling para FOMC days |

**Recomendación firme**: empezar por **#1 Reverse Factor Finder** esta misma semana. Es S de esfuerzo, reusa endpoints existentes (`/factors/best`), y produce el momento "wow" en cualquier demo.

---

## 5. Plan monetización (resumen)

```
Free        Pro $29/mo       Quant $99/mo        Enterprise $499+
─────────   ──────────       ─────────────       ───────────────
1 watchlist ∞ watchlists     + Portfolio         + Priority SLA
5 alerts    ∞ alerts           optimizer         + Dedicated factors
3 factors   1090 factors     + Walk-forward      + White-label embed
delayed 1h  Real-time          backtester        + SSO + audit
α Hub R/O   Webhook alerts   + Custom factors    + Onboarding call
            API 1k/day       + API 10k/day
```

**Path al primer dólar**: 100 free signups (HN post + Twitter thread + Reddit r/algotrading + embed widgets virales) → 5 Pro = $145/mo. Stack billing: Stripe + PostHog self-hosted.

**Disclaimer obligatorio**: footer permanente *"Educational tool. Not investment advice."* + ADR-0008 sobre regulatory posture.

---

## 6. Demo 15-min narrative (ejecutiva)

| Min | Acción | Wow moment |
|---|---|---|
| 0-1 | Hook: *"¿Y si las prediction markets predijeran retornos de stocks?"* | — |
| 1-3 | Terminal: orderbook live + top movers + news tape con factor-tags | live tick refresh ● |
| 3-6 | α Hub: 4 verdes (deploy) + 84 grises (graveyard, honestidad) | caveat-box visible "only ~6 names liquid" |
| 6-10 | Regression NVDA con 5 factores AI-race → R², HAC, VIF, Sharpe | factor cambia a verde-fuerte cuando t-stat>2 |
| 10-13 | Killer: "Fed-cut-Dec subió 5pp → modelo dice NVDA +2.1% → alpha BUY spread" | toast "⚡ Fed-cut crossed 75% threshold — 2 watchlisted alphas re-armed" |
| 13-15 | Q&A bait: robustness (4Q + BH-FDR), graveyard, plan futuro | repo abierto |

**Pre-flight**: caché caliente, watchlist con 2 markets en movimiento, fallback `PFM_DEMO_MODE=cassette` con respx.

---

## 7. Acción inmediata (este fin de semana)

Si tuvieras que pickear 5 cosas para hacer en 48h:

1. **Cerrar CORS** (10 min) — `pfm/main.py:293`
2. **Embargo walk-forward** (2h) — `pfm/advanced.py:196`
3. **Reverse Factor Finder endpoint + UI** (4h) — single feature killer
4. **Deep-linking URL state** (1h) — desbloquea sharing
5. **Alpha Graveyard público** (3h) — moat de honestidad intelectual

Total ~10 horas. Después de esto el proyecto tiene seguridad mínima + rigor cuant + un differentiator visible.

---

## 8. Anti-recomendaciones (qué NO hacer)

- ❌ NO migrar a React. Plain HTML + Plotly funciona y CLAUDE.md lo dice explícitamente.
- ❌ NO redeployar anti-alphas (recession-odds, crypto-ETF, senate-vol, oil-conflict, favorites-bias).
- ❌ NO añadir features fuera de scope sin escribir nota en `docs/future-work.md` primero.
- ❌ NO romper backward compat en `alpha_strategies.json` — usar aliases en deserializer.
- ❌ NO subir a registry público de Docker images con secrets en env.
- ❌ NO claim "deployable" sin 4Q stability + BH-FDR + cost sensitivity verde.
- ❌ NO promete "investment advice" en marketing — disclaimer educativo.

---

*Generado por 24 sub-agentes paralelos, 2 olas de auditoría exhaustiva. Todos los hallazgos llevan file:line en los reportes originales.*
