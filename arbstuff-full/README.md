# arbstuff-full

Monorepo del stack completo de detección de arbitrajes Kalshi × Polymarket: backend Python (detección + Flask SSE bridge) y frontend React (dashboard terminal-style). Pensado para integrarse a una terminal de prediction markets.

```
arbstuff-full/
├── arb/            ← backend Python: engine, discovery, mappings, Flask bridge
└── dashboard/      ← frontend React: SPA con SSE, tabs, design system propio
```

Cada subfolder es self-contained y tiene su propio `README.md` con detalles. Este archivo es solo el mapa.

---

## Arranque rápido (3 terminales)

```bash
# 1. Backend - detección de arbs
cd arb
pip install -r requirements.txt
python arb_engine.py
# → escribe dashboard_state.json cada ciclo (60-90s)

# 2. Backend - Flask SSE bridge
cd arb
python review_app.py
# → API en http://localhost:5000

# 3. Frontend - React dashboard
cd dashboard
npm install
npm run dev
# → UI en http://localhost:5173 (vite proxy → :5060)
```

**Ojo con los puertos:** `review_app.py` arranca en `:5000` pero `dashboard/vite.config.js` proxea a `:5060`. Si lo dejás default, ajustá uno de los dos:

```js
// dashboard/vite.config.js
proxy: { '/api': { target: 'http://localhost:5000', ... } }
```

---

## Flujo end-to-end

```
   ┌────────────────────┐
   │  arb_engine.py     │  detecta arbs cada 60-90s
   │  (loop principal)  │  fetch público Kalshi + Polymarket
   └─────────┬──────────┘
             │ escribe
             ▼
   ┌────────────────────────────────┐
   │  dashboard_state.json (runtime)│
   └─────────┬──────────────────────┘
             │ lectura cada 2s
             ▼
   ┌────────────────────┐
   │  review_app.py     │  Flask, expone /api/*
   │  (SSE bridge)      │  CORS abierto en dev
   └─────────┬──────────┘
             │ SSE /api/dashboard/stream
             ▼
   ┌────────────────────┐
   │  dashboard/        │  React SPA, JetBrains Mono
   │  (terminal UI)     │  tabs: Opp/Politics/PnL/...
   └────────────────────┘
```

---

## Seguridad — qué subir y qué NO

Este repo es **privado** porque los `markets_config_*.json` reflejan el universo de mercados mapeados (info competitivamente valiosa). Pero **el código en sí no necesita credenciales** para detectar arbs — solo orderbooks públicos.

Lo que SIEMPRE queda fuera (gitignored):

- `arb/.env`, `arb/kalshi_private_key.pem` — credenciales tuyas
- `arb/dashboard_state.json`, `arb/arb_pnl_log.json`, `arb/arb_detection_history.json` — runtime state, regenerado por el engine
- `arb/arb_engine.log`, `arb/crypto_jump_log.jsonl` — logs
- `dashboard/node_modules/`, `dashboard/dist/` — regenerables

Si vas a desplegar la UI a algún sitio público (Vercel, Netlify, etc.) **NO uses este repo entero** — separá solo `dashboard/` y serví el backend desde una VPS privada. El backend incluye los mappings completos, que no deberían ser públicos.

---

## Repos relacionados

Este monorepo combina dos repos atómicos que mantenés en paralelo por si querés versionar partes por separado:

- [`arbstuff`](https://github.com/DGallardoL/arbstuff) — solo backend
- [`arbstuff-ui`](https://github.com/DGallardoL/arbstuff-ui) — solo frontend

Para nuevos cambios elegí dónde commitear primero (atómico) y replicá al `-full`, o trabajá directo en `-full` y splitteá después con `git subtree split`.
