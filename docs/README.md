# Documentation Index

Documentación del proyecto **Prediction Terminal**. Esta carpeta cubre las dos
patas del entregable:

- **Cómo está construido (lo técnico / de cómputo)** — arquitectura, decisiones
  de ingeniería (ADRs), caché, streams, operación. → secciones 2 y 3.
- **La matemática del modelo (finanzas)** — logit, OLS, HAC, attribution.
  → sección 5, en [`quants.md`](quants.md).

> Para correr el proyecto: [`README.md`](../README.md) de la raíz
> (`docker-compose up`). Presentación del equipo:
> [`PRESENTACION_EQUIPO.md`](PRESENTACION_EQUIPO.md).

---

## 1. Empieza aquí

| Documento | Qué contiene |
|---|---|
| [`USER_GUIDE.md`](USER_GUIDE.md) | Recorrido por los tres modos (Regression / Strategies / Terminal). |
| [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | Guión de la demo de 15 minutos. |
| [`PRESENTACION_EQUIPO.md`](PRESENTACION_EQUIPO.md) | Explicación del proyecto para el equipo / presentación. |

## 2. Cómo funciona — explicación técnica (de cómputo)

Esta es la documentación de ingeniería: cómo está armado el sistema y por qué.

| Documento | Qué contiene |
|---|---|
| **[`architecture.md`](architecture.md)** | **Documento principal.** Arquitectura de 3 contenedores, frontend, backend, tiers de caché, coordinación multi-sesión, data sources, pipeline de cómputo, arb matching, sentiment NLP, streams en tiempo real (SSE), observabilidad y operación — con un apéndice de "por qué cada decisión". |
| [`API_REFERENCE.md`](API_REFERENCE.md) | Referencia de endpoints (complementa el OpenAPI auto-generado en `/openapi.json`). |
| [`CACHE.md`](CACHE.md) | Estrategia de caché: Redis, TTLs, prewarm, single-flight anti-stampede. |
| [`PERFORMANCE.md`](PERFORMANCE.md) | Presupuestos de rendimiento y optimizaciones. |
| [`sse_inventory.md`](sse_inventory.md) | Inventario de streams Server-Sent Events. |

## 3. Decisiones de arquitectura (ADRs)

18 ADRs en [`adrs/`](adrs/) — cada uno documenta una decisión técnica y su porqué:

| # | Decisión |
|---|---|
| [0001](adrs/0001-use-fastapi.md) | Usar FastAPI |
| [0002](adrs/0002-logit-transform.md) | Transformación logit de probabilidades |
| [0003](adrs/0003-hac-newey-west.md) | Errores estándar HAC (Newey-West) |
| [0004](adrs/0004-redis-cache-ttl.md) | Caché Redis con TTL |
| [0005](adrs/0005-no-persistence-poc.md) | Sin base de datos persistente en el POC |
| [0006](adrs/0006-timezone-alignment.md) | Alineación de timezones a UTC |
| [0007](adrs/0007-daily-fidelity.md) | `fidelity=1440` (daily) para histórico |
| [0008](adrs/0008-factor-universe-curation.md) | Curación del universo de factores |
| [0009](adrs/0009-frontend-vanilla-html.md) | Frontend vanilla HTML + Plotly |
| [0010](adrs/0010-multi-session-coordination.md) | Coordinación multi-sesión |
| [0011](adrs/0011-cache-tiering.md) | Tiering de caché |
| [0012](adrs/0012-arb-match-quality.md) | Calidad de matching de arbitraje |
| [0013](adrs/0013-anti-alpha-rule.md) | Regla anti-alpha (robustez 4-quarter) |
| [0014](adrs/0014-cache-stampede-singleflight.md) | Anti-stampede con single-flight |
| [0015](adrs/0015-rate-limit-retry.md) | Rate-limit y reintentos |
| [0016](adrs/0016-pickle-versioning.md) | Versionado de pickles en caché |
| [0017](adrs/0017-sse-vs-websocket.md) | SSE vs WebSocket para tiempo real |
| [0018](adrs/0018-frontend-bundle-strategy.md) | Estrategia de bundle del frontend |

## 4. Despliegue y operación

| Documento | Qué contiene |
|---|---|
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Cómo desplegar (Docker, Fly.io). |
| [`RUNBOOK.md`](RUNBOOK.md) | Runbook operativo. |
| [`PRODUCTION_CHECKLIST.md`](PRODUCTION_CHECKLIST.md), [`PRODUCTION_NOTES.md`](PRODUCTION_NOTES.md) | Checklist y notas de producción. |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Solución de problemas. |
| [`SECURITY.md`](SECURITY.md) | Notas de seguridad. |
| [`DEPS_AUDIT.md`](DEPS_AUDIT.md) | Auditoría de dependencias. |

## 5. La matemática del modelo (finanzas)

| Documento | Qué contiene |
|---|---|
| **[`quants.md`](quants.md)** | Logit de probabilidades, clipping, OLS, errores estándar HAC, diagnósticos, attribution, alineación temporal, y qué es / qué no es el modelo. Incluye el modelo crypto 5-min y la implied PDF. |
| [`quant_rigor_advanced.md`](quant_rigor_advanced.md) | Rigor estadístico (deflated Sharpe, reality checks). |
| [`regression-methodology-improvements.md`](regression-methodology-improvements.md), [`regression-cookbook.md`](regression-cookbook.md) | Metodología y recetas de regresión. |
| [`garch_asymmetric_theory.md`](garch_asymmetric_theory.md) | GARCH asimétrico (volatilidad). |
| [`multi_event_theory.md`](multi_event_theory.md), [`event_on_event_theory.md`](event_on_event_theory.md), [`advanced_event_theory.md`](advanced_event_theory.md) | Modelos multi-evento y evento-sobre-evento. |
| [`binary-pricing-results.md`](binary-pricing-results.md) | Pricing de contratos binarios. |

## 6. Estrategias y alphas

| Documento | Qué contiene |
|---|---|
| [`strategies.md`](strategies.md) | Catálogo de estrategias. |
| [`STRATEGY_LIFECYCLE.md`](STRATEGY_LIFECYCLE.md) | Ciclo de vida (propuesta → robustez → deploy). |
| [`alpha-report-v22.md`](alpha-report-v22.md) | **Reporte de alphas vigente.** Historial: [v18](alpha-report-v18.md) · [v19](alpha-report-v19.md) · [v20](alpha-report-v20.md) · [v21](alpha-report-v21.md). |
| [`robustness-lab-report.md`](robustness-lab-report.md) | Laboratorio de robustez (4-quarter, deflated Sharpe). |
| [`factor-curation-guide.md`](factor-curation-guide.md), [`factor-catalog-gaps.md`](factor-catalog-gaps.md) | Curación del catálogo de factores. |

## 7. Notas de investigación y futuro

Exploraciones (no necesariamente deployables):
[`markov-switching-research-note.md`](markov-switching-research-note.md) ·
[`cross-sectional-sweep.md`](cross-sectional-sweep.md) ·
[`exotic-regressions-report.md`](exotic-regressions-report.md) ·
[`regression_audit_findings.md`](regression_audit_findings.md) ·
[`multi_source_factors.md`](multi_source_factors.md) ·
[`pm_vix_slug_management.md`](pm_vix_slug_management.md) ·
[`vol-event-validation.md`](vol-event-validation.md) ·
[`vol-pm-iv-validation.md`](vol-pm-iv-validation.md) ·
[`vol-strategies-findings.md`](vol-strategies-findings.md) ·
[`future-work.md`](future-work.md)
