# Instrucciones — Prediction Terminal

Guía para correr y evaluar el proyecto. Pensada para que **un solo comando**
levante todo y se vea funcionando en minutos.

---

## 1. Requisitos previos

Solo necesitas **Docker** con **Docker Compose**:

- Docker Desktop (Mac / Windows), o Docker Engine + `docker compose` (Linux).
- Que los puertos **8080** (web) y **8000** (API) estén libres.

No hace falta instalar Python, Node ni nada más — todo corre en contenedores.

---

## 2. Cómo correr (1 comando)

Desde la raíz del repo:

```bash
docker-compose up -d --build
```

Esto construye y levanta **3 servicios**: `web` (nginx), `api` (FastAPI) y
`redis` (caché). La primera vez tarda unos minutos (build); las siguientes son
instantáneas.

Cuando termine, abre en el navegador:

- **UI (la app):** http://localhost:8080
- **API + Swagger:** http://localhost:8000/docs
- **OpenAPI JSON:** http://localhost:8000/openapi.json

> **Importante:** abre la UI en **http://localhost:8080** (no `127.0.0.1` ni
> otro puerto) para que el navegador permita las llamadas al API según el CORS
> y el Content-Security-Policy configurados.

Para detener todo:

```bash
docker-compose down
```

---

## 3. Verificación rápida (que todo sirve)

Con el stack arriba:

```bash
# 1. Salud del API (incluye Redis + git SHA)
curl -s http://localhost:8000/health

# 2. Catálogo de factores
curl -s "http://localhost:8000/factors?limit=1"

# 3. Cantidad de endpoints documentados en OpenAPI
curl -s http://localhost:8000/openapi.json | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["paths"]),"endpoints")'
```

En la UI (`http://localhost:8080`) deberías ver el **Terminal** (data de
mercados), el **α Hub** (estrategias) y el modo **Regression** — todos con data
real. `Cmd/Ctrl + K` abre la búsqueda global.

---

## 4. Dónde está cada requisito del curso

| Requisito | Dónde verlo |
|---|---|
| **CI/CD** (lint + test + build) | `.github/workflows/ci.yml` y la pestaña *Actions* en GitHub |
| **Docker / `docker-compose up`** | `docker-compose.yml`, `api/Dockerfile`, `web/Dockerfile` |
| **OpenAPI auto-generado** | http://localhost:8000/openapi.json (y Swagger en `/docs`) |
| **Documentación (matemática + técnica)** | [`docs/README.md`](docs/README.md) — índice; matemática en `docs/quant/quants.md`, técnica en `docs/architecture/architecture.md` |
| **ADRs** (decisiones de arquitectura) | [`docs/adrs/`](docs/adrs/) — 18 ADRs |
| **README** | [`README.md`](README.md) |
| **Guión de demo (15 min)** | [`docs/guides/DEMO_SCRIPT.md`](docs/guides/DEMO_SCRIPT.md) |

---

## 5. Problemas comunes

| Síntoma | Solución |
|---|---|
| `port is already allocated` (8080/8000) | Algo más usa esos puertos. Ciérralo, o cambia el mapeo en `docker-compose.yml` (`"8081:80"` / `"8001:8000"`). |
| La UI carga pero sin data | Asegúrate de abrir **http://localhost:8080** (no `127.0.0.1`) y que el contenedor `api` esté `healthy` (`docker-compose ps`). |
| El build falla | Confirma que Docker está corriendo: `docker ps`. Reintenta `docker-compose up -d --build`. |
| Quiero ver los logs | `docker-compose logs -f api` (o `web` / `redis`). |

---

## 6. Estructura del repo (resumen)

```
.
├── api/                 # Backend FastAPI (Python 3.12) — core quant + endpoints
│   └── src/pfm/         # model.py (OLS+HAC), attribution.py, sources/, strategies/, terminal/
├── web/                 # Frontend (HTML + Plotly, sin build step) servido por nginx
├── docs/                # Documentación (ver docs/README.md como índice)
│   ├── adrs/            # 18 Architecture Decision Records
│   ├── quant/           # Matemática del modelo
│   └── architecture/    # Explicación técnica
├── docker-compose.yml   # Levanta web + api + redis
└── README.md            # Visión general del proyecto
```

Para más detalle, ver el índice de documentación en
[`docs/README.md`](docs/README.md).
