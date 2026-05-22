# Documentation Index

Documentación del proyecto **Prediction Terminal**. Esta carpeta cubre las dos
patas del entregable:

- **Cómo está construido (lo técnico / de cómputo)** — arquitectura, decisiones
  de ingeniería (ADRs), caché, streams, operación. → secciones 2, 3 y 4.
- **La matemática del modelo (finanzas)** — logit, OLS, HAC, attribution.
  → sección 5, en [`quant/quants.md`](quant/quants.md).

> Para correr el proyecto: [`README.md`](../README.md) de la raíz
> (`docker-compose up`). Presentación del equipo:
> [`guides/PRESENTACION_EQUIPO.md`](guides/PRESENTACION_EQUIPO.md).

## Estructura de carpetas

| Carpeta | Contenido |
|---|---|
| [`guides/`](guides/) | Material de arranque / demo / presentación. |
| [`architecture/`](architecture/) | Arquitectura técnica, referencia de API, caché, performance, SSE. |
| [`adrs/`](adrs/) | 18 Architecture Decision Records. |
| [`operations/`](operations/) | Despliegue, runbook, producción, seguridad, troubleshooting. |
| [`quant/`](quant/) | La matemática del modelo (logit, OLS, HAC, GARCH, teoría de eventos). |
| [`strategies/`](strategies/) | Catálogo de estrategias, ciclo de vida, robustez, curación de factores. |
| [`research-notes/`](research-notes/) | Exploraciones e investigación (no necesariamente deployable). |
| [`audits/`](audits/) | Auditorías de código, performance y lanzamiento. |
| [`alpha-reports/`](alpha-reports/) | Serie versionada de reportes de alphas (histórica). |
| [`regen-history/`](regen-history/) | Historial de regeneraciones de tiers de alpha. |
| [`graveyard/`](graveyard/) | Anti-alphas retiradas (no redesplegar). |
| [`internal/`](internal/) | Notas internas de sesión / handover. |

> **Nota:** algunos documentos se leen en tiempo de ejecución por el backend o
> el frontend y por eso permanecen en la raíz de `docs/` (no muévas sin
> actualizar el código): [`USER_GUIDE.md`](USER_GUIDE.md) (fetch del frontend),
> [`regression-methodology-improvements.md`](regression-methodology-improvements.md)
> y [`alpha-report-v22.md`](alpha-report-v22.md) (`citations_router`),
> [`binary-pricing-results.md`](binary-pricing-results.md) (gate de deploy de
> `binary_pricing_alpha`), y [`future-work.md`](future-work.md) (convención de
> CLAUDE.md).

---

## 1. Empieza aquí

| Documento | Qué contiene |
|---|---|
| [`USER_GUIDE.md`](USER_GUIDE.md) | Recorrido por los tres modos (Regression / Strategies / Terminal). |
| [`guides/DEMO_SCRIPT.md`](guides/DEMO_SCRIPT.md) | Guión de la demo de 15 minutos. |
| [`guides/PRESENTACION_EQUIPO.md`](guides/PRESENTACION_EQUIPO.md) | Explicación del proyecto para el equipo / presentación. |

## 2. Cómo funciona — explicación técnica (de cómputo)

Esta es la documentación de ingeniería: cómo está armado el sistema y por qué.

| Documento | Qué contiene |
|---|---|
| **[`architecture/architecture.md`](architecture/architecture.md)** | **Documento principal.** Arquitectura de 3 contenedores, frontend, backend, tiers de caché, coordinación multi-sesión, data sources, pipeline de cómputo, arb matching, sentiment NLP, streams en tiempo real (SSE), observabilidad y operación — con un apéndice de "por qué cada decisión". |
| [`architecture/API_REFERENCE.md`](architecture/API_REFERENCE.md) | Referencia de endpoints (complementa el OpenAPI auto-generado en `/openapi.json`). |
| [`architecture/CACHE.md`](architecture/CACHE.md) | Estrategia de caché: Redis, TTLs, prewarm, single-flight anti-stampede. |
| [`architecture/PERFORMANCE.md`](architecture/PERFORMANCE.md) | Presupuestos de rendimiento y optimizaciones. |
| [`architecture/sse_inventory.md`](architecture/sse_inventory.md) | Inventario de streams Server-Sent Events. |
| [`architecture/DEVELOPMENT.md`](architecture/DEVELOPMENT.md) | Guía de desarrollo local. |

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

## 4. Despliegue, operación y auditorías

| Documento | Qué contiene |
|---|---|
| [`operations/DEPLOYMENT.md`](operations/DEPLOYMENT.md) | Cómo desplegar (Docker, Fly.io). |
| [`operations/RUNBOOK.md`](operations/RUNBOOK.md) | Runbook operativo. |
| [`operations/PRODUCTION_CHECKLIST.md`](operations/PRODUCTION_CHECKLIST.md), [`operations/PRODUCTION_NOTES.md`](operations/PRODUCTION_NOTES.md) | Checklist y notas de producción. |
| [`operations/TROUBLESHOOTING.md`](operations/TROUBLESHOOTING.md) | Solución de problemas. |
| [`operations/SECURITY.md`](operations/SECURITY.md) | Notas de seguridad. |
| [`operations/DEPS_AUDIT.md`](operations/DEPS_AUDIT.md) | Auditoría de dependencias. |
| [`audits/CODE_QUALITY_AUDIT.md`](audits/CODE_QUALITY_AUDIT.md), [`audits/PERFORMANCE_AUDIT.md`](audits/PERFORMANCE_AUDIT.md) | Auditorías de calidad de código y rendimiento. |
| [`audits/LAUNCH_AUDIT.md`](audits/LAUNCH_AUDIT.md), [`audits/LAUNCH_CHECKLIST.md`](audits/LAUNCH_CHECKLIST.md) | Auditoría y checklist de lanzamiento. |

## 5. La matemática del modelo (finanzas)

| Documento | Qué contiene |
|---|---|
| **[`quant/quants.md`](quant/quants.md)** | Logit de probabilidades, clipping, OLS, errores estándar HAC, diagnósticos, attribution, alineación temporal, y qué es / qué no es el modelo. Incluye el modelo crypto 5-min y la implied PDF. |
| [`quant/quant_rigor_advanced.md`](quant/quant_rigor_advanced.md) | Rigor estadístico (deflated Sharpe, reality checks). |
| [`regression-methodology-improvements.md`](regression-methodology-improvements.md), [`quant/regression-cookbook.md`](quant/regression-cookbook.md) | Metodología y recetas de regresión. |
| [`quant/garch_asymmetric_theory.md`](quant/garch_asymmetric_theory.md) | GARCH asimétrico (volatilidad). |
| [`quant/multi_event_theory.md`](quant/multi_event_theory.md), [`quant/event_on_event_theory.md`](quant/event_on_event_theory.md), [`quant/advanced_event_theory.md`](quant/advanced_event_theory.md) | Modelos multi-evento y evento-sobre-evento. |
| [`binary-pricing-results.md`](binary-pricing-results.md) | Pricing de contratos binarios. |

## 6. Estrategias y alphas

| Documento | Qué contiene |
|---|---|
| [`strategies/strategies.md`](strategies/strategies.md) | Catálogo de estrategias. |
| [`strategies/STRATEGY_LIFECYCLE.md`](strategies/STRATEGY_LIFECYCLE.md) | Ciclo de vida (propuesta → robustez → deploy). |
| [`alpha-report-v22.md`](alpha-report-v22.md) | **Reporte de alphas vigente.** Historial: [v18](alpha-reports/alpha-report-v18.md) · [v19](alpha-reports/alpha-report-v19.md) · [v20](alpha-reports/alpha-report-v20.md) · [v21](alpha-reports/alpha-report-v21.md). Serie completa versionada en [`alpha-reports/`](alpha-reports/). |
| [`strategies/robustness-lab-report.md`](strategies/robustness-lab-report.md) | Laboratorio de robustez (4-quarter, deflated Sharpe). |
| [`strategies/factor-curation-guide.md`](strategies/factor-curation-guide.md), [`strategies/factor-catalog-gaps.md`](strategies/factor-catalog-gaps.md) | Curación del catálogo de factores. |

## 7. Notas de investigación y futuro

Exploraciones (no necesariamente deployables) en [`research-notes/`](research-notes/):
[`markov-switching-research-note.md`](research-notes/markov-switching-research-note.md) ·
[`cross-sectional-sweep.md`](research-notes/cross-sectional-sweep.md) ·
[`exotic-regressions-report.md`](research-notes/exotic-regressions-report.md) ·
[`regression_audit_findings.md`](research-notes/regression_audit_findings.md) ·
[`multi_source_factors.md`](research-notes/multi_source_factors.md) ·
[`pm_vix_slug_management.md`](research-notes/pm_vix_slug_management.md) ·
[`vol-event-validation.md`](research-notes/vol-event-validation.md) ·
[`vol-pm-iv-validation.md`](research-notes/vol-pm-iv-validation.md) ·
[`vol-strategies-findings.md`](research-notes/vol-strategies-findings.md) ·
[`future-work.md`](future-work.md)
