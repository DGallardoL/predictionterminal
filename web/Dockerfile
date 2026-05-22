FROM nginx:1.27-alpine

RUN apk add --no-cache curl

COPY nginx.conf /etc/nginx/conf.d/default.conf
# Bundle the ENTIRE web/ tree into the served root. index.html pulls ~38 CSS +
# ~32 JS modules from css/ and js/, plus root-level assets (plotly-theme.js,
# cmdk.{js,css}, onboarding.js, realtime-tickers.js, *-premium.css), PWA icons,
# manifest.json, components/, public/ and docs/. Copying everything (rather than
# enumerating files) means a newly-referenced asset can never 404 in the image.
# .dockerignore keeps the Dockerfile itself out of the served root.
COPY . /usr/share/nginx/html/

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost/ >/dev/null || exit 1
