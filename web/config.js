// Runtime API base configuration.
// Loaded by index.html before any other script. The same static bundle must
// work across these deployment shapes:
//
//   1) docker-compose: nginx on :8080 proxies /api/* to api:8000
//      browser at http://localhost:8080  ->  PFM_API_BASE = "/api"
//
//   2) uvicorn standalone (dev): serves both /ui/ and the API on the same port
//      browser at http://localhost:8000/ui/  ->  PFM_API_BASE = "" (same origin)
//
//   3) Local dev split: python http.server on :8080 + gunicorn on :8000.
//      No /api proxy exists, so /api/* would 404. Point directly at :8000.
//
//   4) production behind reverse proxy (Fly.io / Render): /api/* proxied
//      browser at https://yourdomain  ->  PFM_API_BASE = "/api"
//
// Detection rule:
//   - localhost/127.0.0.1 on :8080 (dev split)         -> "http://<host>:8000"
//   - non-local host on :8080 / 80 / 443 (nginx)        -> "/api"
//   - localhost on any other port (uvicorn unified)     -> ""
//   - non-local custom deployment                       -> "/api"
//
// Override manually by setting window.PFM_API_BASE before any fetch call.
(function () {
  var loc = window.location;
  var port = loc.port;
  var host = loc.hostname;
  var isLocal = host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0";

  // Local dev split — python http.server (or any static server) on :8080
  // alongside gunicorn on :8000. No nginx, no /api proxy. The frontend must
  // point cross-origin at the backend.
  if (isLocal && port === "8080") {
    window.PFM_API_BASE = loc.protocol + "//" + host + ":8000";
    return;
  }

  // Unified single-app deploy (e.g. Fly.io single machine): the API serves the
  // static UI under /ui AND the API itself at the same origin's ROOT — there is
  // no /api proxy. Detect by the page path so this wins over the /api rule
  // below for any host/port (incl. https://<app>.fly.dev/ui/).
  if (loc.pathname.indexOf("/ui") === 0) {
    window.PFM_API_BASE = "";
    return;
  }

  // Production (no port or 80/443) and docker-compose nginx (non-local :8080):
  // /api/* is proxied to the backend.
  if (port === "" || port === "80" || port === "443" || port === "8080") {
    window.PFM_API_BASE = "/api";
    return;
  }

  // Local dev with uvicorn directly serving both /ui/ and the API at the
  // same origin (any other port). Use empty string so fetch("/factors")
  // resolves against the current page origin.
  if (isLocal) {
    window.PFM_API_BASE = "";
    return;
  }

  // Fallback for non-localhost custom deployments: assume /api proxy.
  window.PFM_API_BASE = "/api";
})();
