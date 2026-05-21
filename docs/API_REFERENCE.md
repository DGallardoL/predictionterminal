# Prediction Factor Model Reference

Auto-generated from `http://localhost:8000/openapi.json` on 2026-05-16 15:12 UTC. Do not edit by hand — re-run `python scripts/gen_api_reference.py`.

**API version**: `0.1.0`  
**Total endpoints**: 276  
**Groups**: 35

## Endpoints by Group

### Terminal (67 endpoints)
- [`POST /terminal/backtest-compare`](#post-terminalbacktest-compare) — Compare N pairs-trading strategies side-by-side on the same data.
- [`POST /terminal/backtest/{slug}`](#post-terminalbacktestslug) — Inline mean-reversion backtest (pair / rolling-z / bollinger).
- [`GET /terminal/book/{slug}`](#get-terminalbookslug) — Get Book Ladder
- [`GET /terminal/calendar`](#get-terminalcalendar) — Unified Calendar
- [`GET /terminal/calendar-curated/clusters`](#get-terminalcalendar-curatedclusters) — List Clusters
- [`GET /terminal/calendar-curated/{cluster_id}`](#get-terminalcalendar-curatedcluster-id) — Get Cluster
- [`GET /terminal/calendar-pair/{slug}`](#get-terminalcalendar-pairslug) — Get Calendar Pair
- [`GET /terminal/calendar-scanner/active`](#get-terminalcalendar-scanneractive) — Get Active Signals
- [`GET /terminal/calendar-scanner/historical`](#get-terminalcalendar-scannerhistorical) — Get Historical Backtest
- [`GET /terminal/calendar/upcoming`](#get-terminalcalendarupcoming) — Get Upcoming Events
- [`GET /terminal/compare`](#get-terminalcompare) — Get Compare
- [`GET /terminal/correlations/{slug}`](#get-terminalcorrelationsslug) — Get Correlations
- [`GET /terminal/countdown`](#get-terminalcountdown) — Get Countdown
- [`GET /terminal/countdown/{slug}`](#get-terminalcountdownslug) — Get Market Countdown
- [`GET /terminal/equity-curve/{slug}`](#get-terminalequity-curveslug) — Get Terminal Equity
- [`GET /terminal/equity/{slug}`](#get-terminalequityslug) — Get Terminal Equity
- [`POST /terminal/export/bulk`](#post-terminalexportbulk) — Bulk Export
- [`GET /terminal/factor-clusters`](#get-terminalfactor-clusters) — Hierarchical clustering of factors by Δlogit-return correlation.
- [`GET /terminal/fair-price/{slug}`](#get-terminalfair-priceslug) — Get Fair Prices
- [`GET /terminal/fair/{slug}`](#get-terminalfairslug) — Get Fair Prices
- [`GET /terminal/flow/{slug}`](#get-terminalflowslug) — Trade-flow analytics (informed/aggressive flow) for a Polymarket market.
- [`GET /terminal/gdelt/breaking`](#get-terminalgdeltbreaking) — Top global breaking-news headlines from GDELT (last 6 hours).
- [`GET /terminal/gdelt/{slug}`](#get-terminalgdeltslug) — GDELT 2.0 global news for a Polymarket market's topic.
- [`GET /terminal/homepage`](#get-terminalhomepage) — Get Homepage
- [`GET /terminal/jumps/cluster`](#get-terminaljumpscluster) — Group jumps across many slugs into macro-event clusters.
- [`GET /terminal/jumps/{slug}`](#get-terminaljumpsslug) — Detect price-series jumps and attach matching GDELT articles.
- [`GET /terminal/jumps/{slug}/backtest`](#get-terminaljumpsslugbacktest) — Paper-PnL backtest of the disagrees-jump reversion signal.
- [`GET /terminal/live-stream`](#get-terminallive-stream) — Live Stream
- [`GET /terminal/macro-overlay/{slug}`](#get-terminalmacro-overlayslug) — Get Macro Overlay
- [`GET /terminal/market/{slug}`](#get-terminalmarketslug) — Terminal Market
- [`GET /terminal/market/{slug}/history`](#get-terminalmarketslughistory) — Terminal Market History
- [`GET /terminal/news-impact/{slug}`](#get-terminalnews-impactslug) — GDELT news events with Polymarket price-reaction windows.
- [`GET /terminal/news/{slug}`](#get-terminalnewsslug) — Recent Reddit + HN posts mentioning a Polymarket market's topic.
- [`GET /terminal/orderbook/{slug}`](#get-terminalorderbookslug) — Get Book Ladder
- [`GET /terminal/overview`](#get-terminaloverview) — Terminal Overview
- [`GET /terminal/peers/{slug}`](#get-terminalpeersslug) — Get Peers
- [`POST /terminal/portfolio-sim`](#post-terminalportfolio-sim) — Post Portfolio Sim
- [`GET /terminal/prob-fan/{slug}`](#get-terminalprob-fanslug) — Get Prob Fan
- [`GET /terminal/quality/{slug}`](#get-terminalqualityslug) — Get Quality
- [`GET /terminal/quote/{slug}`](#get-terminalquoteslug) — Get Quote
- [`GET /terminal/rss-news`](#get-terminalrss-news) — Discoverability alias for /headlines with optional ``q`` keyword.
- [`GET /terminal/rss/headlines`](#get-terminalrssheadlines) — Unified, ranked RSS headlines across every active wire source.
- [`GET /terminal/rss/sources`](#get-terminalrsssources) — List all RSS sources and their current ok/error status.
- [`GET /terminal/rss/{slug}`](#get-terminalrssslug) — Headlines matching a Polymarket market's question keywords.
- [`GET /terminal/search`](#get-terminalsearch) — Terminal Search
- [`GET /terminal/search-index`](#get-terminalsearch-index) — Get Search Index
- [`GET /terminal/search-index/chunked`](#get-terminalsearch-indexchunked) — Get Search Index Chunked
- [`GET /terminal/sentiment-leaderboard`](#get-terminalsentiment-leaderboard) — Rank top-volume markets by news-sentiment / price-jump disagreement density.
- [`GET /terminal/sentiment-trend/spike-alerts`](#get-terminalsentiment-trendspike-alerts) — Markets where mean tone has shifted by more than 3.0 in the last N days.
- [`GET /terminal/sentiment-trend/{slug}`](#get-terminalsentiment-trendslug) — GDELT tone series for a market, with lag-correlation against price.
- [`GET /terminal/stream`](#get-terminalstream) — Stream
- [`GET /terminal/themes`](#get-terminalthemes) — Get Themes
- [`GET /terminal/theta/cluster`](#get-terminalthetacluster) — Get Cluster Theta
- [`GET /terminal/theta/{slug}`](#get-terminalthetaslug) — Get Market Theta
- [`GET /terminal/trade-ticket/scan`](#get-terminaltrade-ticketscan) — Scan Trade Tickets
- [`GET /terminal/trade-ticket/{cluster_id}`](#get-terminaltrade-ticketcluster-id) — Get Trade Ticket
- [`GET /terminal/trades/{slug}`](#get-terminaltradesslug) — Recent classified trades for a Polymarket market.
- [`GET /terminal/vol-cone/{slug}`](#get-terminalvol-coneslug) — Get Vol Cone
- [`GET /terminal/vol-distribution/{slug}`](#get-terminalvol-distributionslug) — Get Vol Distribution
- [`GET /terminal/volume-tape/{slug}`](#get-terminalvolume-tapeslug) — Recent classified trades (alias of /trades/{slug}).
- [`POST /terminal/watchlist`](#post-terminalwatchlist) — Add To Watchlist
- [`GET /terminal/watchlist/quotes`](#get-terminalwatchlistquotes) — Watchlist Quotes
- [`GET /terminal/watchlist/{user_id}`](#get-terminalwatchlistuser-id) — List Watchlist
- [`GET /terminal/watchlist/{user_id}/alerts`](#get-terminalwatchlistuser-idalerts) — List Triggered Alerts
- [`DELETE /terminal/watchlist/{user_id}/{slug}`](#delete-terminalwatchlistuser-idslug) — Remove From Watchlist
- [`GET /terminal/whales/recent-large-trades`](#get-terminalwhalesrecent-large-trades) — Recent large trades over the last N hours for one market.
- [`GET /terminal/whales/{slug}`](#get-terminalwhalesslug) — Large positions per address for a Polymarket market.

### Strategies (59 endpoints)
- [`POST /strategies/almgren-chriss`](#post-strategiesalmgren-chriss) — Strategies Almgren Chriss
- [`DELETE /strategies/arb/blacklist`](#delete-strategiesarbblacklist) — Clear the blacklist
- [`GET /strategies/arb/blacklist`](#get-strategiesarbblacklist) — List blacklisted arb_keys
- [`POST /strategies/arb/blacklist`](#post-strategiesarbblacklist) — Append an arb_key to the blacklist
- [`GET /strategies/arb/config`](#get-strategiesarbconfig) — Current scan threshold + mode + last-known control
- [`GET /strategies/arb/config-events`](#get-strategiesarbconfig-events) — Merged mapped-event universe
- [`GET /strategies/arb/config-stats`](#get-strategiesarbconfig-stats) — Mapping counts per source file
- [`GET /strategies/arb/detection-history`](#get-strategiesarbdetection-history) — Rolling history of detected arbs (newest-first)
- [`GET /strategies/arb/markets`](#get-strategiesarbmarkets) — All mapped Kalshi↔Polymarket pairs (paginated)
- [`GET /strategies/arb/orderbook`](#get-strategiesarborderbook) — Live Kalshi + Polymarket orderbook proxy
- [`GET /strategies/arb/pnl`](#get-strategiesarbpnl) — Simulated PnL log from arb_engine test-mode trades
- [`GET /strategies/arb/politics-events`](#get-strategiesarbpolitics-events) — Politics specialist mapping universe
- [`POST /strategies/arb/settings`](#post-strategiesarbsettings) — Merge runtime control toggles
- [`GET /strategies/arb/state`](#get-strategiesarbstate) — Live arb engine state — opportunities + scan log
- [`GET /strategies/arb/stream`](#get-strategiesarbstream) — SSE stream of /state every 5s
- [`POST /strategies/auto-backtest`](#post-strategiesauto-backtest) — Strategies Auto Backtest
- [`POST /strategies/basket-stat-arb`](#post-strategiesbasket-stat-arb) — Strategies Basket Stat Arb
- [`POST /strategies/bounds`](#post-strategiesbounds) — Strategies Bounds
- [`POST /strategies/cointegration`](#post-strategiescointegration) — Strategies Cointegration
- [`POST /strategies/conditional`](#post-strategiesconditional) — Strategies Conditional
- [`GET /strategies/crypto/5min/compare`](#get-strategiescrypto5mincompare) — Side-by-side model vs market for every BTC/ETH × 5m/15m combo
- [`GET /strategies/crypto/5min/diag`](#get-strategiescrypto5mindiag) — Spot-buffer diagnostics for the 5min predictor
- [`GET /strategies/crypto/5min/markets`](#get-strategiescrypto5minmarkets) — Live model-vs-market table for every open 5m/15m crypto market
- [`GET /strategies/crypto/5min/predict/{symbol}`](#get-strategiescrypto5minpredictsymbol) — Pure-model P(up by end of next 5m/15m window) for one Binance pair
- [`GET /strategies/crypto/events`](#get-strategiescryptoevents) — Live whale + mean-reversion events from the WS engine (last N min)
- [`GET /strategies/crypto/model-state/{symbol}`](#get-strategiescryptomodel-statesymbol) — Live cryptostuff signals + annualized σ for the GBM model-prob calc
- [`GET /strategies/crypto/signals`](#get-strategiescryptosignals) — Catalogue of the 9 microstructure signals computed by the WS engine
- [`GET /strategies/crypto/snapshot`](#get-strategiescryptosnapshot) — Live 10-pair microstructure snapshot (Binance REST)
- [`GET /strategies/crypto/spec`](#get-strategiescryptospec) — How to launch the WS engine locally + what to expect
- [`POST /strategies/cusum`](#post-strategiescusum) — Strategies Cusum
- [`POST /strategies/dfa`](#post-strategiesdfa) — Strategies Dfa
- [`GET /strategies/discovery`](#get-strategiesdiscovery) — Filter the strategies catalog by tag.
- [`POST /strategies/distance-method`](#post-strategiesdistance-method) — Strategies Distance Method
- [`POST /strategies/event-model`](#post-strategiesevent-model) — Strategies Event Model
- [`POST /strategies/factor-model-pro`](#post-strategiesfactor-model-pro) — Strategies Factor Model Pro
- [`POST /strategies/fractional-diff`](#post-strategiesfractional-diff) — Strategies Fractional Diff
- [`POST /strategies/fred-cointegration`](#post-strategiesfred-cointegration) — Strategies Fred Cointegration
- [`POST /strategies/garch`](#post-strategiesgarch) — Strategies Garch
- [`POST /strategies/granger`](#post-strategiesgranger) — Strategies Granger
- [`POST /strategies/implication`](#post-strategiesimplication) — Strategies Implication
- [`POST /strategies/info-share`](#post-strategiesinfo-share) — Strategies Info Share
- [`POST /strategies/kalman-hedge`](#post-strategieskalman-hedge) — Strategies Kalman Hedge
- [`GET /strategies/list`](#get-strategieslist) — Enumerate every /strategies/* endpoint with metadata.
- [`POST /strategies/mean-reversion`](#post-strategiesmean-reversion) — Strategies Mean Reversion
- [`POST /strategies/ml-predictor`](#post-strategiesml-predictor) — Strategies Ml Predictor
- [`POST /strategies/optimize`](#post-strategiesoptimize) — Optimize
- [`POST /strategies/ou-bands`](#post-strategiesou-bands) — Strategies Ou Bands
- [`POST /strategies/pairs-backtest`](#post-strategiespairs-backtest) — Strategies Pairs Backtest
- [`POST /strategies/patterns`](#post-strategiespatterns) — Strategies Patterns
- [`POST /strategies/portfolio`](#post-strategiesportfolio) — Strategies Portfolio
- [`GET /strategies/presets`](#get-strategiespresets) — Strategies Presets
- [`POST /strategies/regime-switching`](#post-strategiesregime-switching) — Strategies Regime Switching
- [`POST /strategies/robust-validation`](#post-strategiesrobust-validation) — Strategies Robust Validation
- [`POST /strategies/scan`](#post-strategiesscan) — Strategies Scan
- [`POST /strategies/sharpe-bootstrap`](#post-strategiessharpe-bootstrap) — Strategies Sharpe Bootstrap
- [`POST /strategies/sharpe-permutation`](#post-strategiessharpe-permutation) — Strategies Sharpe Permutation
- [`POST /strategies/spot-vs-implied`](#post-strategiesspot-vs-implied) — Strategies Spot Vs Implied
- [`POST /strategies/triple-barrier`](#post-strategiestriple-barrier) — Strategies Triple Barrier
- [`POST /strategies/walk-forward`](#post-strategieswalk-forward) — Strategies Walk Forward

### Alpha Hub (7 endpoints)
- [`GET /alpha-hub/graveyard`](#get-alpha-hubgraveyard) — List dead / downgraded alpha strategies
- [`GET /alpha-hub/graveyard/{pair_id}`](#get-alpha-hubgraveyardpair-id) — Fetch a single death certificate
- [`GET /alpha-hub/leaderboard`](#get-alpha-hubleaderboard) — Paginated, filtered, sortable view of curated alpha strategies.
- [`GET /alpha-hub/live-panel`](#get-alpha-hublive-panel) — Composite payload: top production alphas + watchlist + recent graveyard.
- [`POST /alpha-hub/regenerate-tiers`](#post-alpha-hubregenerate-tiers) — Re-run the walk-forward harness over alpha_strategies.json
- [`GET /alpha-hub/regenerate-tiers/{job_id}`](#get-alpha-hubregenerate-tiersjob-id) — Fetch status / summary of a regen job
- [`GET /alpha-hub/strategy/{pair_id}`](#get-alpha-hubstrategypair-id) — Full per-strategy detail (all fields from alpha_strategies.json).

### Alpha (7 endpoints)
- [`GET /alpha/decay`](#get-alphadecay) — List Decay Status
- [`GET /alpha/earnings-calendar`](#get-alphaearnings-calendar) — Get Earnings Calendar
- [`GET /alpha/earnings-whisper-dashboard`](#get-alphaearnings-whisper-dashboard) — Get Whisper Dashboard
- [`GET /alpha/earnings-whisper/{ticker}`](#get-alphaearnings-whisperticker) — Get Whisper
- [`POST /alpha/prediction-driven`](#post-alphaprediction-driven) — Prediction Driven Endpoint
- [`POST /alpha/{pair_id}/recompute-decay`](#post-alphapair-idrecompute-decay) — Recompute Decay
- [`GET /alpha/{pair_id}/rolling-sharpe`](#get-alphapair-idrolling-sharpe) — Get Rolling Sharpe

### Archive (12 endpoints)
- [`GET /archive/cross-venue/concepts`](#get-archivecross-venueconcepts) — Catalog of pre-mapped cross-venue concepts (PM vs Kalshi).
- [`GET /archive/cross-venue/{concept}`](#get-archivecross-venueconcept) — Polymarket vs Kalshi divergence metrics for a resolved concept.
- [`GET /archive/kalshi/market/{ticker}`](#get-archivekalshimarketticker) — Per-market detail (metadata + history + stats), optionally as CSV.
- [`GET /archive/kalshi/markets`](#get-archivekalshimarkets) — Paginated list of settled Kalshi markets.
- [`GET /archive/kalshi/series`](#get-archivekalshiseries) — Per-series stats over all settled Kalshi markets.
- [`GET /archive/list`](#get-archivelist) — Alias of /archive/polymarket/markets (footer pill).
- [`POST /archive/polymarket/export-bulk`](#post-archivepolymarketexport-bulk) — Bulk-export N archive markets as a ZIP of per-slug files.
- [`GET /archive/polymarket/market/{slug}`](#get-archivepolymarketmarketslug) — Full archive detail (history + stats) for one resolved market.
- [`GET /archive/polymarket/markets`](#get-archivepolymarketmarkets) — Paginated list of resolved Polymarket markets in a date range.
- [`GET /archive/polymarket/resolutions/{slug}`](#get-archivepolymarketresolutionsslug) — Resolution outcome only (no price history).
- [`GET /archive/polymarket/search`](#get-archivepolymarketsearch) — Substring search over resolved-market slug + question.
- [`GET /archive/polymarket/themes`](#get-archivepolymarketthemes) — Aggregate stats per theme across the most recent resolved markets.

### Factors (8 endpoints)
- [`GET /factors`](#get-factors) — List Factors
- [`GET /factors/all`](#get-factorsall) — List Factors All
- [`POST /factors/best`](#post-factorsbest) — Best Model
- [`GET /factors/discover`](#get-factorsdiscover) — Discover Factors
- [`POST /factors/permutation`](#post-factorspermutation) — Factors Permutation
- [`POST /factors/preview`](#post-factorspreview) — Preview Factor
- [`POST /factors/rank`](#post-factorsrank) — Rank Factors
- [`POST /factors/suggest-for-ticker`](#post-factorssuggest-for-ticker) — Suggest Factors For Ticker

### Auth (7 endpoints)
- [`POST /auth/demo-key`](#post-authdemo-key) — Mint a 24h Free-tier demo key (open, no admin token required)
- [`GET /auth/first-boot-info`](#get-authfirst-boot-info) — One-shot retrieval of the autogenerated admin token (prod only)
- [`POST /auth/keys`](#post-authkeys) — Create a new API key (admin only)
- [`GET /auth/keys/me`](#get-authkeysme) — Inspect the API key in use on this request
- [`GET /auth/keys/me/usage`](#get-authkeysmeusage) — Usage stats for the API key in use
- [`DELETE /auth/keys/{key}`](#delete-authkeyskey) — Revoke a key (admin only)
- [`GET /auth/usage/dashboard`](#get-authusagedashboard) — Aggregated org-wide usage (admin only)

### Macro (7 endpoints)
- [`GET /macro/bls/catalog`](#get-macroblscatalog) — Bls Catalog
- [`GET /macro/bls/{series_id}`](#get-macroblsseries-id) — Bls Series Endpoint
- [`GET /macro/calendar/export.ics`](#get-macrocalendarexportics) — iCalendar export of the macro calendar (Google Calendar friendly).
- [`GET /macro/fred/catalog`](#get-macrofredcatalog) — Fred Catalog
- [`GET /macro/fred/series/{series_id}`](#get-macrofredseriesseries-id) — Fred Series Endpoint
- [`GET /macro/overlay`](#get-macrooverlay) — Macro Overlay
- [`GET /macro/upcoming`](#get-macroupcoming) — Macro Upcoming

### Arbitrage (8 endpoints)
- [`GET /arb/4way-arbs`](#get-arb4way-arbs) — Get 4Way Arbs
- [`GET /arb/auto-discover`](#get-arbauto-discover) — Get Auto Discover
- [`GET /arb/concept/{concept_id}`](#get-arbconceptconcept-id) — Get 4Way Concept
- [`GET /arb/concepts`](#get-arbconcepts) — List 4Way Concepts
- [`GET /arb/confirmed-matches`](#get-arbconfirmed-matches) — Get Confirmed Matches
- [`POST /arb/match`](#post-arbmatch) — Post Match
- [`GET /arb/matched`](#get-arbmatched) — Get Matched
- [`GET /arb/scanner`](#get-arbscanner) — Get Scanner

### Reverse Finder (2 endpoints)
- [`POST /reverse-finder`](#post-reverse-finder) — Reverse Finder Endpoint
- [`POST /reverse-finder/stream`](#post-reverse-finderstream) — Reverse Finder Stream Endpoint

### Advanced Model (6 endpoints)
- [`POST /advanced-model/conditional`](#post-advanced-modelconditional) — Post Conditional
- [`POST /advanced-model/garch-x`](#post-advanced-modelgarch-x) — Post Garch X
- [`POST /advanced-model/polynomial`](#post-advanced-modelpolynomial) — Post Polynomial
- [`POST /advanced-model/regime-switching`](#post-advanced-modelregime-switching) — Post Regime Switching
- [`POST /advanced-model/tail-dependence`](#post-advanced-modeltail-dependence) — Post Tail Dependence
- [`POST /advanced-model/vecm`](#post-advanced-modelvecm) — Post Vecm

### Event Model (5 endpoints)
- [`POST /event-model/correlation-matrix`](#post-event-modelcorrelation-matrix) — Event Model Correlation
- [`POST /event-model/fit`](#post-event-modelfit) — Event Model Fit
- [`POST /event-model/lead-lag`](#post-event-modellead-lag) — Event Model Lead Lag
- [`POST /event-model/pca`](#post-event-modelpca) — Event Model Pca
- [`POST /event-model/var`](#post-event-modelvar) — Event Model Var

### Multi-Event (5 endpoints)
- [`POST /multi-event/chains`](#post-multi-eventchains) — Find Granger-significant chains start_factor -> ... -> ticker.
- [`POST /multi-event/lasso`](#post-multi-eventlasso) — Fit LassoCV across N PM-factor Δlogits to predict ticker log returns.
- [`POST /multi-event/macro-correlation`](#post-multi-eventmacro-correlation) — Δlogit(factor) vs Δ(macro) correlation, t-stat, and lead-lag.
- [`POST /multi-event/sector-attribution`](#post-multi-eventsector-attribution) — Per-sector OLS-HAC and variance attribution across PM factors.
- [`POST /multi-event/systemic-factor`](#post-multi-eventsystemic-factor) — Extract a PM-PCA systemic risk-on/off factor from N PM factors.

### News (6 endpoints)
- [`POST /news/causal-chain`](#post-newscausal-chain) — Build news -> Δprob -> Δlogit -> ticker-impact chain for a factor.
- [`GET /news/entity/{entity}/factors`](#get-newsentityentityfactors) — Top factors associated with a named entity.
- [`GET /news/factor/{factor_id}/recent`](#get-newsfactorfactor-idrecent) — Recently tagged news items for a factor.
- [`GET /news/movers`](#get-newsmovers) — Top news items by |expected stock impact| across registered factors.
- [`POST /news/tag`](#post-newstag) — Tag a single news headline -> entities + matched factors + sentiment.
- [`POST /news/tag-batch`](#post-newstag-batch) — Bulk-tag a list of news items.

### Indices (5 endpoints)
- [`GET /indices/pm-vix`](#get-indicespm-vix) — Get Pm Vix
- [`GET /indices/pm-vix/components`](#get-indicespm-vixcomponents) — Get Pm Vix Components
- [`GET /indices/pm-vix/history`](#get-indicespm-vixhistory) — Get Pm Vix History
- [`POST /indices/pm-vix/refresh-slugs`](#post-indicespm-vixrefresh-slugs) — Validate hardcoded bucket slugs against Polymarket and persist replacements.
- [`GET /indices/pm-vix/slugs`](#get-indicespm-vixslugs) — Return the current per-bucket slug map (live or fallback).

### Quant (5 endpoints)
- [`POST /quant/diebold-mariano`](#post-quantdiebold-mariano) — Post Diebold Mariano
- [`POST /quant/multitest/bh`](#post-quantmultitestbh) — Post Bh Multitest
- [`POST /quant/oos-r-squared`](#post-quantoos-r-squared) — Post Oos R Squared
- [`POST /quant/quarterly-stability`](#post-quantquarterly-stability) — Post Quarterly Stability
- [`POST /quant/whites-reality-check`](#post-quantwhites-reality-check) — Post Whites Reality Check

### Lab (4 endpoints)
- [`POST /lab/discover`](#post-labdiscover) — Kick off an alpha-discovery run (background task)
- [`POST /lab/promote/{candidate_id}`](#post-labpromotecandidate-id) — Mark a candidate for human review (does NOT auto-promote)
- [`GET /lab/queue`](#get-labqueue) — Get the lab's current runtime state
- [`GET /lab/results/{job_id}`](#get-labresultsjob-id) — Fetch results for a specific job

### Signals (4 endpoints)
- [`GET /signals/connectivity-check`](#get-signalsconnectivity-check) — Probe Polymarket Gamma + CLOB end-to-end with a sample slug.
- [`GET /signals/live`](#get-signalslive) — Return the current live_signals.json contents (cached 30s).
- [`POST /signals/recompute-now`](#post-signalsrecompute-now) — Trigger one live-signals recompute synchronously.
- [`GET /signals/status`](#get-signalsstatus) — Last live-signals run status (cron health).

### Alerts (8 endpoints)
- [`GET /alerts`](#get-alerts) — List alert rules for a user
- [`POST /alerts`](#post-alerts) — Create a new alert rule
- [`GET /alerts/events`](#get-alertsevents) — List alert events
- [`POST /alerts/events/{event_id}/ack`](#post-alertseventsevent-idack) — Acknowledge an event
- [`DELETE /alerts/{id}`](#delete-alertsid) — Delete an alert rule
- [`GET /alerts/{id}`](#get-alertsid) — Get an alert rule by id
- [`PATCH /alerts/{id}`](#patch-alertsid) — Partial-update an alert rule
- [`POST /alerts/{id}/test`](#post-alertsidtest) — Dry-run dispatch to the rule's channels

### Embed (7 endpoints)
- [`POST /embed/beacon`](#post-embedbeacon) — Embed-impression beacon (best-effort tracking, no PII).
- [`GET /embed/compare`](#get-embedcompare) — Embeddable overlay of 2+ market price histories (normalised).
- [`GET /embed/market/{slug}`](#get-embedmarketslug) — Embeddable mini-card for a Polymarket market.
- [`GET /embed/og/factor/{factor_id}`](#get-embedogfactorfactor-id) — Open-Graph PNG (1200x630) for a factor share link.
- [`GET /embed/og/market/{slug}.png`](#get-embedogmarketslugpng) — Open-Graph PNG (1200x630) for a market — used in social unfurls.
- [`GET /embed/og/strategy/{strategy_id}`](#get-embedogstrategystrategy-id) — Open-Graph PNG (1200x630) for a strategy share link.
- [`GET /embed/strategy/{pair_id}`](#get-embedstrategypair-id) — Embeddable card for a validated alpha strategy.

### Replay (7 endpoints)
- [`POST /replay/order`](#post-replayorder) — Simulate a paper-trade order against historical prices
- [`GET /replay/scenario/{scenario_name}`](#get-replayscenarioscenario-name) — Hydrate a pre-baked scenario
- [`GET /replay/scenario/{scenario_name}/pnl`](#get-replayscenarioscenario-namepnl) — Realized historical basket PnL for the scenario window
- [`GET /replay/scenario/{scenario_name}/preflight`](#get-replayscenarioscenario-namepreflight) — Verify each scenario slug is still resolvable on Polymarket
- [`GET /replay/scenarios`](#get-replayscenarios) — List pre-baked replay scenarios
- [`GET /replay/sessions`](#get-replaysessions) — Alias of /replay/scenarios (footer pill).
- [`GET /replay/state`](#get-replaystate) — Snapshot of PM + equity state at a past timestamp

### Fit (2 endpoints)
- [`POST /fit`](#post-fit) — Fit Endpoint
- [`POST /fit/preview`](#post-fitpreview) — Fit Preview Endpoint

### Attribution (1 endpoints)
- [`POST /attribution`](#post-attribution) — Attribution Endpoint

### Health (2 endpoints)
- [`GET /health`](#get-health) — Health
- [`GET /health/detail`](#get-healthdetail) — Health Detail

### Counterfactual (2 endpoints)
- [`POST /counterfactual`](#post-counterfactual) — Post Counterfactual
- [`POST /counterfactual/multi`](#post-counterfactualmulti) — Post Multi

### Divergence (2 endpoints)
- [`GET /divergence/smart-money`](#get-divergencesmart-money) — Top PM-vs-equity flow divergences across the universe.
- [`GET /divergence/{slug}`](#get-divergenceslug) — Divergence snapshot for a single (slug, default-ticker) pair.

### Export (1 endpoints)
- [`POST /export/chart-png`](#post-exportchart-png) — Chart Png

### Hedge (2 endpoints)
- [`POST /hedge/auto-config`](#post-hedgeauto-config) — Solve PM hedge sizes that neutralise a portfolio's factor β.
- [`POST /hedge/simulate`](#post-hedgesimulate) — Paper-trade a daily-rebalance hedge over N days.

### Multi Venue (3 endpoints)
- [`GET /multi-venue/concept/{concept_id}`](#get-multi-venueconceptconcept-id) — Get Concept
- [`GET /multi-venue/concepts`](#get-multi-venueconcepts) — List Concepts
- [`GET /multi-venue/search`](#get-multi-venuesearch) — Get Search

### Portfolio (2 endpoints)
- [`POST /portfolio/pnl-monte-carlo`](#post-portfoliopnl-monte-carlo) — Monte-Carlo P&L distribution from N bootstrapped Δlogit paths.
- [`POST /portfolio/resolution-tree`](#post-portfolioresolution-tree) — Conditional MTM tree (YES vs NO outcome) for a portfolio on a factor.

### Sources (3 endpoints)
- [`GET /sources/delisted`](#get-sourcesdelisted) — Get Delisted
- [`POST /sources/delisted/{ticker}`](#post-sourcesdelistedticker) — Post Delisted
- [`GET /sources/health`](#get-sourceshealth) — Sources Health

### Strategy Verdict (2 endpoints)
- [`POST /strategy-verdict/cointegration`](#post-strategy-verdictcointegration) — Post Cointegration Verdict
- [`POST /strategy-verdict/pairs`](#post-strategy-verdictpairs) — Post Pairs Verdict

### Vol (3 endpoints)
- [`POST /vol/egarch`](#post-volegarch) — Post Egarch
- [`POST /vol/garch-compare`](#post-volgarch-compare) — Post Garch Compare
- [`POST /vol/gjr-garch`](#post-volgjr-garch) — Post Gjr Garch

### Vol Surface (2 endpoints)
- [`GET /vol-surface/compare`](#get-vol-surfacecompare) — Compare
- [`GET /vol-surface/pm/{slug_pattern}`](#get-vol-surfacepmslug-pattern) — Get Pm Distribution

### Whales (3 endpoints)
- [`POST /whales/mirror`](#post-whalesmirror) — Build a mirror portfolio over a whale's current positions.
- [`GET /whales/top`](#get-whalestop) — Top whales by absolute 7d PnL.
- [`GET /whales/{address}/history`](#get-whalesaddresshistory) — Cumulative-PnL trace for a single whale over N days.

## Endpoint Details

## Terminal

### POST /terminal/backtest-compare

**Summary**: Compare N pairs-trading strategies side-by-side on the same data.

Run each strategy through ``pairs_backtest`` and compare them.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "strategies": [
    {
      "slug": "string",
      "side": "both",
      "entry_z": 2.0,
      "exit_z": 0.5,
      "stop_z": 4.0,
      "window": 20
    }
  ],
  "days": 180
}
```

**Response 200** (`application/json`):

```json
{
  "strategies": [
    {
      "spec": null,
      "peer_slug": "string",
      "beta_hedge": 0.0,
      "n_obs": 0,
      "n_trades": 0,
      "sharpe": 0.0
    }
  ],
  "correlation": [
    [
      0.0
    ]
  ],
  "combined_portfolio": {
    "sharpe": 0.0,
    "dd": 0.0
  }
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/terminal/backtest-compare -H 'Content-Type: application/json' -d '{"strategies": [{"slug": "string", "side": "both", "entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "window": 20}], "days": 180}'
```


### POST /terminal/backtest/{slug}

**Summary**: Inline mean-reversion backtest (pair / rolling-z / bollinger).

Run a backtest of ``slug`` in one of three modes.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |

**Request Body** (`application/json`):

```json
{
  "entry_z": 2.0,
  "exit_z": 0.5,
  "stop_z": 4.0,
  "window": 20,
  "hold_days": 0,
  "side": "both"
}
```

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "mode_used": "pair",
  "peer_slug": "string",
  "beta_hedge": 0.0,
  "n_obs": 0,
  "n_trades": 0
}
```

**Example**:

```bash
curl -X POST 'http://localhost:8000/terminal/backtest/<slug>' -H 'Content-Type: application/json' -d '{"entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "window": 20, "hold_days": 0, "side": "both"}'
```


### GET /terminal/book/{slug}

**Summary**: Get Book Ladder

Return a 10-level ladder, depth bands, fill costs, and top-5 imbalance.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `timeout` | query | number | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "token_id": "string",
  "bid_levels": [
    {
      "price": 0.0,
      "size": 0.0,
      "cumulative": 0.0
    }
  ],
  "ask_levels": [
    {
      "price": 0.0,
      "size": 0.0,
      "cumulative": 0.0
    }
  ],
  "mid": 0.0,
  "spread_cents": 0.0
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/book/<slug>'
```


### GET /terminal/calendar

**Summary**: Unified Calendar

Return a chronologically-sorted, multi-source calendar slice.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `start` | query | string | no | Inclusive start date, ISO-8601. Defaults to *today* when omitted — combined with the default 7-day window this makes the bare ``/terminal/calendar`` URL a useful one-shot. |
| `end` | query | string | no | Inclusive end date, ISO-8601. Defaults to ``start + 7 days`` when omitted. |
| `kinds` | query | string | no | Comma-separated subset of ``resolution,earnings,macro``. Defaults to all three when omitted. |
| `theme` | query | string | no | Optional case-insensitive substring filter applied to each item's theme tags (rates, fed, inflation, oil, NVDA, …). |

**Response 200** (`application/json`):

```json
{
  "start": "string",
  "end": "string",
  "items": [
    {
      "date": "string",
      "kind": "resolution",
      "title": "string",
      "slug": null,
      "ticker": null,
      "importance": 2
    }
  ],
  "total": 0
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/calendar
```


### GET /terminal/calendar-curated/clusters

**Summary**: List Clusters

Return one summary per curated cluster.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
[
  {
    "cluster_id": "string",
    "title": "string",
    "theory": "string",
    "legs": [
      null
    ],
    "n_legs": 0,
    "lambda_range": [
      null
    ]
  }
]
```

**Example**:

```bash
curl http://localhost:8000/terminal/calendar-curated/clusters
```


### GET /terminal/calendar-curated/{cluster_id}

**Summary**: Get Cluster

Detail payload for a single cluster, including 90-day λ-ratio.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `cluster_id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "cluster_id": "string",
  "title": "string",
  "theory": "string",
  "legs": [
    {
      "factor_id": "string",
      "slug": "string",
      "name": "string",
      "source": "string",
      "deadline": "string",
      "days_to_resolve": 0
    }
  ],
  "n_legs": 0,
  "lambda_range": [
    null
  ]
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/calendar-curated/<cluster_id>'
```


### GET /terminal/calendar-pair/{slug}

**Summary**: Get Calendar Pair

Return the calendar-pair surface for ``slug`` or ``null``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "event_token": "string",
  "surface": [
    null
  ],
  "lambda_near": 0.0,
  "lambda_far": 0.0,
  "log_lambda_ratio": 0.0
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/calendar-pair/<slug>'
```


### GET /terminal/calendar-scanner/active

**Summary**: Get Active Signals

Return every currently-actionable calendar arb across curated clusters.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
[
  {
    "cluster_id": "string",
    "title": "string",
    "trade_type": "FLATTEN_CURVE",
    "long_leg": {
      "slug": null,
      "name": null,
      "current_p": null,
      "implied_lambda": null
    },
    "short_leg": {
      "slug": null,
      "name": null,
      "current_p": null,
      "implied_lambda": null
    },
    "log_lambda_ratio": 0.0
  }
]
```

**Example**:

```bash
curl http://localhost:8000/terminal/calendar-scanner/active
```


### GET /terminal/calendar-scanner/historical

**Summary**: Get Historical Backtest

90-day cluster-level PnL backtest of the threshold-crossing signal.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `cluster_id` | query | string | yes |  |
| `lookback_days` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "cluster_id": "string",
  "n_days": 0,
  "n_trades": 0,
  "cum_pnl": 0.0,
  "sharpe": 0.0,
  "points": [
    {
      "date": "string",
      "log_lambda_ratio": 0.0,
      "in_trade": true,
      "pnl_today": 0.0,
      "cum_pnl": 0.0
    }
  ]
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/calendar-scanner/historical?cluster_id=string'
```


### GET /terminal/calendar/upcoming

**Summary**: Get Upcoming Events

List upcoming scheduled macro/political/crypto events that prediction markets react to.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `days` | query | integer | no | Look-ahead horizon in days. |

**Response 200** (`application/json`):

```json
{
  "as_of": "string",
  "horizon_days": 0,
  "n_events": 0,
  "events": [
    {
      "date": "string",
      "time_et": "string",
      "name": "string",
      "category": "string",
      "expected_impact_themes": [
        null
      ],
      "related_markets": [
        null
      ]
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/calendar/upcoming
```


### GET /terminal/compare

**Summary**: Get Compare

Side-by-side comparison of N≤4 prediction-market contracts.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slugs` | query | string | yes | Comma-separated slugs (2..4). |
| `days` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "slugs": [
    "string"
  ],
  "days": 0,
  "legs": [
    {
      "slug": "string",
      "live": null,
      "meta": null,
      "stats": null,
      "history": [
        null
      ]
    }
  ],
  "correlation_matrix": {},
  "pairs_trade": {
    "a": "string",
    "b": "string",
    "beta_hedge": null,
    "intercept": null,
    "spread_now": null,
    "spread_mean": null
  }
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/compare?slugs=string'
```


### GET /terminal/correlations/{slug}

**Summary**: Get Correlations

Return the cross-asset correlation card for a Polymarket slug.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `days` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/correlations/<slug>'
```


### GET /terminal/countdown

**Summary**: Get Countdown

Polymarket factors resolving in the next ``days`` days. Cached 5 min,
fan-out capped at ``_COUNTDOWN_MAX_FACTORS`` slugs to avoid gamma 429s.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `days` | query | integer | no | Look-ahead horizon in days. |

**Response 200** (`application/json`):

```json
{
  "as_of": "string",
  "horizon_days": 0,
  "n_markets": 0,
  "groups": [
    {
      "bucket": "today",
      "n_markets": 0,
      "markets": [
        null
      ]
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/countdown
```


### GET /terminal/countdown/{slug}

**Summary**: Get Market Countdown

Real-time countdown + expected payoff for one market.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "question": "string",
  "current_p": 0.0,
  "days": 0,
  "hours": 0,
  "minutes": 0
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/countdown/<slug>'
```


### GET /terminal/equity-curve/{slug}

**Summary**: Get Terminal Equity

Return the equity-overlay payload for a Polymarket slug.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `days` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/equity-curve/<slug>'
```


### GET /terminal/equity/{slug}

**Summary**: Get Terminal Equity

Return the equity-overlay payload for a Polymarket slug.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `days` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/equity/<slug>'
```


### POST /terminal/export/bulk

**Summary**: Bulk Export

Fetch ``scope`` for every slug in parallel and return a combined blob.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "slugs": [
    "string"
  ],
  "format": "csv",
  "scope": [
    "live"
  ]
}
```

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl -X POST http://localhost:8000/terminal/export/bulk -H 'Content-Type: application/json' -d '{"slugs": ["string"], "format": "csv", "scope": ["live"]}'
```


### GET /terminal/factor-clusters

**Summary**: Hierarchical clustering of factors by Δlogit-return correlation.

Cluster factors by return correlation and surface a leader per cluster.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `theme` | query | string | no | Filter by factors.yml theme tag (e.g. 'politics'). |
| `min_corr` | query | number | no | \|corr\| threshold cutting the dendrogram. |

**Response 200** (`application/json`):

```json
{
  "n_factors_in": 0,
  "n_clusters": 0,
  "clusters": [
    {
      "cluster_id": "string",
      "n_factors": 0,
      "avg_intra_corr": 0.0,
      "leader": null,
      "members": [
        null
      ],
      "theme_centroid": "string"
    }
  ],
  "theme": "string",
  "min_corr": 0.0,
  "degraded_mode": false
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/factor-clusters
```


### GET /terminal/fair-price/{slug}

**Summary**: Get Fair Prices

Return multi-model fair-price estimates for a market.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `p_market` | query | number | no |  |
| `peer_price` | query | number | no |  |
| `btc_t` | query | number | no |  |
| `btc_0` | query | number | no |  |
| `seconds_remaining` | query | number | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/fair-price/<slug>'
```


### GET /terminal/fair/{slug}

**Summary**: Get Fair Prices

Return multi-model fair-price estimates for a market.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `p_market` | query | number | no |  |
| `peer_price` | query | number | no |  |
| `btc_t` | query | number | no |  |
| `btc_0` | query | number | no |  |
| `seconds_remaining` | query | number | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/fair/<slug>'
```


### GET /terminal/flow/{slug}

**Summary**: Trade-flow analytics (informed/aggressive flow) for a Polymarket market.

Return flow analytics for the trailing ``window_minutes`` of trades.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes | Polymarket market slug. |
| `window_minutes` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "window_minutes": 0,
  "n_trades_total": 0,
  "n_trades_buy": 0,
  "n_trades_sell": 0,
  "buy_ratio": 0.0
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/flow/<slug>'
```


### GET /terminal/gdelt/breaking

**Summary**: Top global breaking-news headlines from GDELT (last 6 hours).

Return up to ``limit`` recent global headlines (English, hybridrel-sorted).

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `limit` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "timespan": "string",
  "n_articles": 0,
  "articles": [
    {
      "url": "string",
      "title": "string",
      "source": "string",
      "country": "string",
      "ts": "string",
      "tone": 0.0
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/gdelt/breaking
```


### GET /terminal/gdelt/{slug}

**Summary**: GDELT 2.0 global news for a Polymarket market's topic.

Return GDELT articles relevant to the topic of ``slug`` plus aggregates.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes | Polymarket market slug. |
| `limit` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "query_used": "string",
  "n_articles": 0,
  "articles": [
    {
      "url": "string",
      "title": "string",
      "source": "string",
      "country": "string",
      "ts": "string",
      "tone": 0.0
    }
  ],
  "mean_tone": 0.0,
  "dominant_topic": "string"
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/gdelt/<slug>'
```


### GET /terminal/homepage

**Summary**: Get Homepage

Composed homepage payload (gainers/losers/most-active + sparklines).

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `theme` | query | string | no |  |
| `hours` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "theme": "string",
  "hours": 0,
  "n_markets_considered": 0,
  "gainers": [
    {
      "slug": "string",
      "name": "string",
      "theme": null,
      "price": null,
      "change_pct": null,
      "volume_24h": null
    }
  ],
  "losers": [
    {
      "slug": "string",
      "name": "string",
      "theme": null,
      "price": null,
      "change_pct": null,
      "volume_24h": null
    }
  ],
  "most_active": [
    {
      "slug": "string",
      "name": "string",
      "theme": null,
      "price": null,
      "change_pct": null,
      "volume_24h": null
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/homepage
```


### GET /terminal/jumps/cluster

**Summary**: Group jumps across many slugs into macro-event clusters.

For a list of slugs, run jump detection then cluster the results.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slugs` | query | string | no | Comma-separated Polymarket slugs. Omit to default to the top-20 by 24h volume. |
| `days` | query | integer | no |  |
| `time_tol_minutes` | query | number | no |  |
| `kw_min_jaccard` | query | number | no |  |
| `mad_k` | query | number | no |  |
| `min_jump_pp` | query | number | no |  |

**Response 200** (`application/json`):

```json
{
  "slugs": [
    "string"
  ],
  "days": 0,
  "time_tol_minutes": 0.0,
  "kw_min_jaccard": 0.0,
  "n_jumps_total": 0,
  "n_clusters": 0
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/jumps/cluster
```


### GET /terminal/jumps/{slug}

**Summary**: Detect price-series jumps and attach matching GDELT articles.

For a Polymarket slug, return jumps (∆logit outliers) with matching news.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `days` | query | integer | no |  |
| `mad_k` | query | number | no |  |
| `min_jump_pp` | query | number | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "days": 0,
  "threshold_mad_k": 0.0,
  "threshold_min_jump_pp": 0.0,
  "n_jumps": 0,
  "n_explained": 0
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/jumps/<slug>'
```


### GET /terminal/jumps/{slug}/backtest

**Summary**: Paper-PnL backtest of the disagrees-jump reversion signal.

Run the disagrees-jump paper-PnL backtest for one slug.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `days` | query | integer | no |  |
| `hold_hours` | query | integer | no |  |
| `mad_k` | query | number | no |  |
| `min_jump_pp` | query | number | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "hold_hours": 0,
  "n_disagrees": 0,
  "n_agrees": 0,
  "disagrees_pnl": {
    "n_trades": 0,
    "mean_return": 0.0,
    "std_return": 0.0,
    "sharpe_naive": 0.0,
    "hit_rate": 0.0,
    "avg_win": 0.0
  },
  "agrees_pnl": {
    "n_trades": 0,
    "mean_return": 0.0,
    "std_return": 0.0,
    "sharpe_naive": 0.0,
    "hit_rate": 0.0,
    "avg_win": 0.0
  }
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/jumps/<slug>/backtest'
```


### GET /terminal/live-stream

**Summary**: Live Stream

Open a Server-Sent Events stream of live midpoints for ``slugs``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slugs` | query | string | yes | Comma-separated Polymarket slugs (max 30). |
| `hz` | query | number | no |  |

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/live-stream?slugs=string'
```


### GET /terminal/macro-overlay/{slug}

**Summary**: Get Macro Overlay

Return the macro-overlay payload for a Polymarket macro slug.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `days` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/macro-overlay/<slug>'
```


### GET /terminal/market/{slug}

**Summary**: Terminal Market

Merged data hub for a single market (live + meta + stats + peers).

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `format` | query | string | no |  |

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/market/<slug>'
```


### GET /terminal/market/{slug}/history

**Summary**: Terminal Market History

Pass-through to CLOB ``/prices-history`` with TTL caching.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `fidelity` | query | integer | no |  |
| `start` | query | string | no |  |
| `end` | query | string | no |  |
| `format` | query | string | no |  |

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/market/<slug>/history'
```


### GET /terminal/news-impact/{slug}

**Summary**: GDELT news events with Polymarket price-reaction windows.

Return per-event price reactions (1h/6h/24h) for a Polymarket slug.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `days` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "days": 0,
  "events": [
    {
      "ts_iso": "string",
      "headline": "string",
      "source": "string",
      "tone": 0.0,
      "price_before": null,
      "price_1h_after": null
    }
  ],
  "n_events": 0,
  "n_attributable": 0,
  "attributable_pct": 0.0
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/news-impact/<slug>'
```


### GET /terminal/news/{slug}

**Summary**: Recent Reddit + HN posts mentioning a Polymarket market's topic.

Return recent Reddit + HN posts for the topic of ``slug``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes | Polymarket market slug. |
| `limit` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "question": "string",
  "keywords": [
    "string"
  ],
  "n_items": 0,
  "items": [
    {
      "source": "reddit",
      "title": "string",
      "url": "string",
      "ts": "string",
      "score": 0,
      "sentiment": "positive"
    }
  ],
  "anchors": [
    "string"
  ]
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/news/<slug>'
```


### GET /terminal/orderbook/{slug}

**Summary**: Get Book Ladder

Return a 10-level ladder, depth bands, fill costs, and top-5 imbalance.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `timeout` | query | number | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "token_id": "string",
  "bid_levels": [
    {
      "price": 0.0,
      "size": 0.0,
      "cumulative": 0.0
    }
  ],
  "ask_levels": [
    {
      "price": 0.0,
      "size": 0.0,
      "cumulative": 0.0
    }
  ],
  "mid": 0.0,
  "spread_cents": 0.0
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/orderbook/<slug>'
```


### GET /terminal/overview

**Summary**: Terminal Overview

Markets overview: theme heatmap + movers + most-traded + new + soon-to-resolve.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `pages` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "n_markets_considered": 0,
  "theme_heatmap": [
    {
      "theme": "string",
      "n_markets": 0,
      "median_24h_change": null,
      "median_volume_24hr": null,
      "total_volume_24hr": null,
      "median_yes_price": null
    }
  ],
  "top_movers": [
    {
      "slug": "string",
      "question": "string",
      "theme": null,
      "price": null,
      "one_day_price_change": null,
      "volume_24hr": null
    }
  ],
  "most_traded": [
    {
      "slug": "string",
      "question": "string",
      "theme": null,
      "price": null,
      "one_day_price_change": null,
      "volume_24hr": null
    }
  ],
  "recently_launched": [
    {
      "slug": "string",
      "question": "string",
      "theme": null,
      "price": null,
      "created_at": null,
      "age_days": null
    }
  ],
  "upcoming_resolutions": [
    {
      "slug": "string",
      "question": "string",
      "theme": null,
      "price": null,
      "end_date": null,
      "days_to_resolve": null
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/overview
```


### GET /terminal/peers/{slug}

**Summary**: Get Peers

Cointegrated-peer lookup for a factor slug.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `top` | query | integer | no |  |
| `min_sharpe` | query | number | no |  |
| `format` | query | string | no |  |

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/peers/<slug>'
```


### POST /terminal/portfolio-sim

**Summary**: Post Portfolio Sim

Simulate a multi-position Polymarket book over a daily history window.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `format` | query | string | no |  |

**Request Body** (`application/json`):

```json
{
  "positions": [
    {
      "slug": "string",
      "side": "YES",
      "size_usd": 0.0
    }
  ],
  "days": 180
}
```

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl -X POST http://localhost:8000/terminal/portfolio-sim -H 'Content-Type: application/json' -d '{"positions": [{"slug": "string", "side": "YES", "size_usd": 0.0}], "days": 180}'
```


### GET /terminal/prob-fan/{slug}

**Summary**: Get Prob Fan

Return percentile paths of the YES probability under a Brownian bridge.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `n_paths` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/prob-fan/<slug>'
```


### GET /terminal/quality/{slug}

**Summary**: Get Quality

Compute and return the composite quality score for ``slug``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `timeout` | query | number | no |  |
| `format` | query | string | no |  |

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/quality/<slug>'
```


### GET /terminal/quote/{slug}

**Summary**: Get Quote

Composed quote-page payload.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `days` | query | integer | no |  |
| `include` | query | string | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "days": 0,
  "includes": [
    "string"
  ],
  "live": {
    "price": 0.0,
    "best_bid": 0.0,
    "best_ask": 0.0,
    "spread_cents": 0.0,
    "change_24h": 0.0,
    "change_7d": 0.0
  },
  "meta": {
    "slug": "string",
    "title": "string",
    "theme": "string",
    "end_date": "string",
    "days_to_resolve": 0,
    "total_volume": 0.0
  },
  "stats": {
    "n_obs": 0,
    "rv_30d": 0.0,
    "half_life": 0.0,
    "hurst": 0.0,
    "dfa_alpha": 0.0,
    "vif_max": 0.0
  }
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/quote/<slug>'
```


### GET /terminal/rss-news

**Summary**: Discoverability alias for /headlines with optional ``q`` keyword.

Same backing logic as ``/headlines`` plus a case-insensitive ``q`` filter.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `q` | query | string | no |  |
| `limit` | query | integer | no |  |
| `category` | query | string | no |  |

**Response 200** (`application/json`):

```json
{
  "n_items": 0,
  "category": "all",
  "sources_used": [
    "string"
  ],
  "items": [
    {
      "source": "string",
      "source_name": "string",
      "category": "all",
      "title": "string",
      "link": "string",
      "pub_date": "string"
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/rss-news
```


### GET /terminal/rss/headlines

**Summary**: Unified, ranked RSS headlines across every active wire source.

Aggregate every source's RSS, filter by category, rank by recency.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `limit` | query | integer | no |  |
| `category` | query | string | no |  |

**Response 200** (`application/json`):

```json
{
  "n_items": 0,
  "category": "all",
  "sources_used": [
    "string"
  ],
  "items": [
    {
      "source": "string",
      "source_name": "string",
      "category": "all",
      "title": "string",
      "link": "string",
      "pub_date": "string"
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/rss/headlines
```


### GET /terminal/rss/sources

**Summary**: List all RSS sources and their current ok/error status.

Probe each source and report status. Cache hits count as ``ok``.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "n_sources": 0,
  "n_ok": 0,
  "sources": [
    {
      "slug": "string",
      "name": "string",
      "url": "string",
      "category": "all",
      "status": "ok",
      "n_items": 0
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/rss/sources
```


### GET /terminal/rss/{slug}

**Summary**: Headlines matching a Polymarket market's question keywords.

Resolve the slug → question, score every headline by token overlap.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes | Polymarket market slug. |
| `limit` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "question": "string",
  "keywords": [
    "string"
  ],
  "n_items": 0,
  "items": [
    {
      "source": "string",
      "source_name": "string",
      "category": "all",
      "title": "string",
      "link": "string",
      "pub_date": "string"
    }
  ],
  "anchors": [
    "string"
  ]
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/rss/<slug>'
```


### GET /terminal/search

**Summary**: Terminal Search

Fuzzy search across factor catalog (name + slug). Token-overlap scoring.
With empty q, returns the first ``limit`` factors filtered by ``theme``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `q` | query | string | no |  |
| `theme` | query | string | no |  |
| `limit` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "query": "string",
  "n_results": 0,
  "results": [
    {
      "factor_id": "string",
      "name": "string",
      "slug": "string",
      "theme": null,
      "score": 0.0,
      "current_price": null
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/search
```


### GET /terminal/search-index

**Summary**: Get Search Index

Compact palette dump: factors + strategies + pages + actions.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "version": "string",
  "n_factors": 0,
  "factors": [
    {
      "i": "string",
      "s": "string",
      "n": "string",
      "t": null,
      "p": null,
      "v": null
    }
  ],
  "strategies": [
    {
      "i": "string",
      "n": "string",
      "t": null
    }
  ],
  "pages": [
    {
      "i": "string",
      "n": "string",
      "u": "string"
    }
  ],
  "actions": [
    {
      "i": "string",
      "n": "string",
      "k": "string"
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/search-index
```


### GET /terminal/search-index/chunked

**Summary**: Get Search Index Chunked

Lazy-loadable slice of the palette factor catalogue.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `chunk` | query | integer | no |  |
| `size` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "version": "string",
  "n_factors": 0,
  "chunk": 0,
  "chunk_size": 0,
  "total_chunks": 0,
  "factors": [
    {
      "i": "string",
      "s": "string",
      "n": "string",
      "t": null,
      "p": null,
      "v": null
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/search-index/chunked
```


### GET /terminal/sentiment-leaderboard

**Summary**: Rank top-volume markets by news-sentiment / price-jump disagreement density.

Top-25 markets where news-sentiment most often disagrees with price jumps.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `days` | query | integer | no |  |
| `min_jumps` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "days": 0,
  "min_jumps": 0,
  "n_markets_considered": 0,
  "n_markets_qualified": 0,
  "rows": [
    {
      "rank": 0,
      "slug": "string",
      "name": null,
      "theme": null,
      "volume_24h": null,
      "n_jumps": 0
    }
  ],
  "interpretation": "string"
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/sentiment-leaderboard
```


### GET /terminal/sentiment-trend/spike-alerts

**Summary**: Markets where mean tone has shifted by more than 3.0 in the last N days.

Scan high-volume markets for sudden tone shifts.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `days` | query | integer | no |  |
| `min_n_articles` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "days": 0,
  "min_n_articles": 0,
  "n_alerts": 0,
  "alerts": [
    {
      "slug": "string",
      "question": "string",
      "tone_start": 0.0,
      "tone_end": 0.0,
      "tone_shift": 0.0,
      "n_articles": 0
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/sentiment-trend/spike-alerts
```


### GET /terminal/sentiment-trend/{slug}

**Summary**: GDELT tone series for a market, with lag-correlation against price.

Return a daily tone series and its best-lag correlation with YES price.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes | Polymarket market slug. |
| `days` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "current_tone": 0.0,
  "tone_series": [
    {
      "date": "string",
      "mean_tone": 0.0,
      "n_articles": 0,
      "dominant_topic": "string"
    }
  ],
  "sentiment_regime": "string",
  "correlation_with_price": 0.0,
  "lead_lag_days": 0
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/sentiment-trend/<slug>'
```


### GET /terminal/stream

**Summary**: Stream

Multiplexed SSE stream over the realtime hub.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `subs` | query | string | yes | Comma-separated 'kind:slug' subscriptions. |

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/stream?subs=string'
```


### GET /terminal/themes

**Summary**: Get Themes

Themes sidebar — markets-by-theme rollup.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `hours` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "n_themes": 0,
  "themes": [
    {
      "theme": "string",
      "n_markets": 0,
      "avg_change_24h": null,
      "total_volume_24h": null
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/themes
```


### GET /terminal/theta/cluster

**Summary**: Get Cluster Theta

Aggregate theta for all markets matching ``theme`` + ``resolution_period``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `theme` | query | string | no | filter by factor theme |
| `resolution_period` | query | string | no | quarter filter, e.g. "2026Q3" |
| `days` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/terminal/theta/cluster
```


### GET /terminal/theta/{slug}

**Summary**: Get Market Theta

Time-decay analytics card for one Polymarket binary market.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `days` | query | integer | no | empirical lookback in days |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/theta/<slug>'
```


### GET /terminal/trade-ticket/scan

**Summary**: Scan Trade Tickets

List **only** the currently-actionable tickets across every cluster.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `bankroll` | query | number | no |  |
| `risk_per_trade` | query | number | no |  |

**Response 200** (`application/json`):

```json
{
  "bankroll_usd": 0.0,
  "n_clusters_scanned": 0,
  "n_actionable": 0,
  "tickets": [
    {
      "cluster_id": "string",
      "title": "string",
      "action": "OPEN_PAIR",
      "rationale": "string",
      "tickets": [
        null
      ],
      "total_capital_at_risk_usd": 0.0
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/terminal/trade-ticket/scan
```


### GET /terminal/trade-ticket/{cluster_id}

**Summary**: Get Trade Ticket

Build a printable trade ticket for one cluster.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `cluster_id` | path | string | yes |  |
| `bankroll` | query | number | no |  |
| `risk_per_trade` | query | number | no |  |

**Response 200** (`application/json`):

```json
{
  "cluster_id": "string",
  "title": "string",
  "action": "OPEN_PAIR",
  "rationale": "string",
  "tickets": [
    {
      "slug": "string",
      "side": "BUY_YES",
      "current_price_cents": 0.0,
      "size_usd": 0.0,
      "size_contracts": 0,
      "entry_target_cents": 0.0
    }
  ],
  "total_capital_at_risk_usd": 0.0
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/trade-ticket/<cluster_id>'
```


### GET /terminal/trades/{slug}

**Summary**: Recent classified trades for a Polymarket market.

Return the last ``limit`` trades for ``slug`` with Lee-Ready sides.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes | Polymarket market slug. |
| `limit` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "condition_id": "string",
  "n_trades": 0,
  "trades": [
    {
      "timestamp": "string",
      "price": 0.0,
      "size": 0.0,
      "side": "BUY"
    }
  ],
  "rolling_buy_ratio": [
    {
      "timestamp": "string",
      "buy_ratio": 0.0,
      "informed": true
    }
  ],
  "informed_flow_alert": true
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/trades/<slug>'
```


### GET /terminal/vol-cone/{slug}

**Summary**: Get Vol Cone

Return the realized-volatility cone for a single Polymarket slug.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `epsilon` | query | number | no | logit clip ε |
| `lookback_days` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/vol-cone/<slug>'
```


### GET /terminal/vol-distribution/{slug}

**Summary**: Get Vol Distribution

Return the cross-sectional realised-vol distribution for a single slug.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `window` | query | integer | no | rolling-σ window in days |
| `epsilon` | query | number | no | logit clip ε |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/vol-distribution/<slug>'
```


### GET /terminal/volume-tape/{slug}

**Summary**: Recent classified trades (alias of /trades/{slug}).

Return the last ``limit`` trades for ``slug`` with Lee-Ready sides.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes | Polymarket market slug. |
| `limit` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "condition_id": "string",
  "n_trades": 0,
  "trades": [
    {
      "timestamp": "string",
      "price": 0.0,
      "size": 0.0,
      "side": "BUY"
    }
  ],
  "rolling_buy_ratio": [
    {
      "timestamp": "string",
      "buy_ratio": 0.0,
      "informed": true
    }
  ],
  "informed_flow_alert": true
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/volume-tape/<slug>'
```


### POST /terminal/watchlist

**Summary**: Add To Watchlist

Add ``slug`` to ``user_id``'s watchlist (idempotent).

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "user_id": "default",
  "slug": "string",
  "alert_z": 0.0
}
```

**Response 200** (`application/json`):

```json
{
  "user_id": "string",
  "slug": "string",
  "alert_z": 0.0,
  "added": true
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/terminal/watchlist -H 'Content-Type: application/json' -d '{"user_id": "default", "slug": "string", "alert_z": 0.0}'
```


### GET /terminal/watchlist/quotes

**Summary**: Watchlist Quotes

Bulk-quote a CSV of slugs from the sidebar watchlist widget.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slugs` | query | string | yes | Comma-separated Polymarket slugs (max 50). |

**Response 200** (`application/json`):

```json
{
  "n_items": 0,
  "items": [
    {
      "slug": "string",
      "name": null,
      "theme": null,
      "price": null,
      "change_24h": null,
      "volume_24h": null
    }
  ]
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/watchlist/quotes?slugs=string'
```


### GET /terminal/watchlist/{user_id}

**Summary**: List Watchlist

List ``user_id``'s watchlist with current prices and z-scores.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `user_id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "user_id": "string",
  "n_items": 0,
  "items": [
    {
      "slug": "string",
      "alert_z": null,
      "current_p": null,
      "z_score": null,
      "alert_triggered": false
    }
  ]
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/watchlist/<user_id>'
```


### GET /terminal/watchlist/{user_id}/alerts

**Summary**: List Triggered Alerts

Return only the watchlist rows whose z-score has breached their ``alert_z``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `user_id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "user_id": "string",
  "n_alerts": 0,
  "alerts": [
    {
      "slug": "string",
      "alert_z": null,
      "current_p": null,
      "z_score": null,
      "alert_triggered": false
    }
  ]
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/watchlist/<user_id>/alerts'
```


### DELETE /terminal/watchlist/{user_id}/{slug}

**Summary**: Remove From Watchlist

Remove ``slug`` from ``user_id``'s watchlist; ``removed=False`` if it wasn't there.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `user_id` | path | string | yes |  |
| `slug` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "user_id": "string",
  "slug": "string",
  "removed": true
}
```

**Example**:

```bash
curl -X DELETE 'http://localhost:8000/terminal/watchlist/<user_id>/<slug>'
```


### GET /terminal/whales/recent-large-trades

**Summary**: Recent large trades over the last N hours for one market.

Return the largest recent trades on a market.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | query | string | yes | Polymarket market slug. |
| `min_size_usd` | query | number | no |  |
| `hours` | query | integer | no |  |
| `limit` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "condition_id": "string",
  "hours": 0,
  "min_size_usd": 0.0,
  "n_trades": 0,
  "total_notional_usd": 0.0
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/whales/recent-large-trades?slug=string'
```


### GET /terminal/whales/{slug}

**Summary**: Large positions per address for a Polymarket market.

Return whales (positions ≥ ``min_position_usd``) for ``slug``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes | Polymarket market slug. |
| `min_position_usd` | query | number | no |  |
| `limit` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "condition_id": "string",
  "n_whales": 0,
  "total_whale_notional_usd": 0.0,
  "whales": [
    {
      "address": "string",
      "position_yes_usd": 0.0,
      "position_no_usd": 0.0,
      "net_usd": 0.0,
      "last_active_iso": null,
      "n_trades_24h": 0
    }
  ],
  "net_directional_skew": 0.0
}
```

**Example**:

```bash
curl 'http://localhost:8000/terminal/whales/<slug>'
```


## Strategies

### POST /strategies/almgren-chriss

**Summary**: Strategies Almgren Chriss

Closed-form Almgren-Chriss (2001) optimal execution trajectory. No
historical data needed — pure mathematical optimisation given the
target position and impact/risk parameters. Use to schedule entry of a
large pairs trade without telegraphing it to the market.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "target_position": 0.0,
  "n_intervals": 10,
  "time_horizon": 1.0,
  "sigma": 0.1,
  "eta": 0.01,
  "epsilon": 0.005
}
```

**Response 200** (`application/json`):

```json
{
  "n_intervals": 0,
  "x_remaining": [
    0.0
  ],
  "n_per_interval": [
    0.0
  ],
  "kappa": 0.0,
  "time_horizon": 0.0,
  "expected_cost": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/almgren-chriss -H 'Content-Type: application/json' -d '{"target_position": 0.0, "n_intervals": 10, "time_horizon": 1.0, "sigma": 0.1, "eta": 0.01, "epsilon": 0.005}'
```


### DELETE /strategies/arb/blacklist

**Summary**: Clear the blacklist

Truncate both the file and the Redis SET.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `x-admin-token` | header | string | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X DELETE http://localhost:8000/strategies/arb/blacklist
```


### GET /strategies/arb/blacklist

**Summary**: List blacklisted arb_keys

Read the union of file + Redis blacklist.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/arb/blacklist
```


### POST /strategies/arb/blacklist

**Summary**: Append an arb_key to the blacklist

Add ``arb_key`` (idempotent) to ``arb_blacklist.json`` AND a Redis SET.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `x-admin-token` | header | string | no |  |

**Request Body** (`application/json`):

```json
{
  "arb_key": "string"
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/arb/blacklist -H 'Content-Type: application/json' -d '{"arb_key": "string"}'
```


### GET /strategies/arb/config

**Summary**: Current scan threshold + mode + last-known control

Return ``dashboard_control.json`` — runtime toggles.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/arb/config
```


### GET /strategies/arb/config-events

**Summary**: Merged mapped-event universe

Return ``{events: [...]}`` merged across all four config files.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/arb/config-events
```


### GET /strategies/arb/config-stats

**Summary**: Mapping counts per source file

Return ``{reviewed, main, politics, discovered, combined_mapped}``.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/arb/config-stats
```


### GET /strategies/arb/detection-history

**Summary**: Rolling history of detected arbs (newest-first)

Return ``{items, count}``.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/arb/detection-history
```


### GET /strategies/arb/markets

**Summary**: All mapped Kalshi↔Polymarket pairs (paginated)

Paginated view of curated Kalshi↔Polymarket market pairs.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `limit` | query | integer | no |  |
| `offset` | query | integer | no |  |
| `source` | query | string | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/arb/markets
```


### GET /strategies/arb/orderbook

**Summary**: Live Kalshi + Polymarket orderbook proxy

Fetch both sides' orderbooks. At least one identifier required.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `kalshi_ticker` | query | string | no |  |
| `poly_token` | query | string | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/arb/orderbook
```


### GET /strategies/arb/pnl

**Summary**: Simulated PnL log from arb_engine test-mode trades

Return ``{trades, total_pnl, count}`` reading ``arb_pnl_log.json``.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/arb/pnl
```


### GET /strategies/arb/politics-events

**Summary**: Politics specialist mapping universe

Return events from ``markets_config_politics.json`` with parsed fields.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/arb/politics-events
```


### POST /strategies/arb/settings

**Summary**: Merge runtime control toggles

Merge keys into ``dashboard_control.json`` for the engine to pick up.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `x-admin-token` | header | string | no |  |

**Request Body** (`application/json`):

```json
{
  "email_enabled": true,
  "threshold": 0.0,
  "min_alert_profit": 0.0,
  "scan_mode": "string"
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/arb/settings -H 'Content-Type: application/json' -d '{"email_enabled": true, "threshold": 0.0, "min_alert_profit": 0.0, "scan_mode": "string"}'
```


### GET /strategies/arb/state

**Summary**: Live arb engine state — opportunities + scan log

Return current detected arbs + engine status.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl http://localhost:8000/strategies/arb/state
```


### GET /strategies/arb/stream

**Summary**: SSE stream of /state every 5s

Server-sent events stream pushing the (trimmed) ``/state`` envelope.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl http://localhost:8000/strategies/arb/stream
```


### POST /strategies/auto-backtest

**Summary**: Strategies Auto Backtest

Auto-pipeline: scan catalog for cointegrated pairs, backtest each,
rank the leaderboard by Sharpe.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "theme": "string",
  "factor_ids": [
    "string"
  ],
  "start": "date",
  "end": "date",
  "max_pairs": 300,
  "max_to_backtest": 15
}
```

**Response 200** (`application/json`):

```json
{
  "n_factors_scanned": 0,
  "n_coint_hits": 0,
  "n_backtested": 0,
  "runtime_seconds": 0.0,
  "leaderboard": [
    {
      "a_id": "string",
      "b_id": "string",
      "sharpe": 0.0,
      "sharpe_is": 0.0,
      "sharpe_oos": 0.0,
      "oos_to_is_ratio": 0.0
    }
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/auto-backtest -H 'Content-Type: application/json' -d '{"theme": "string", "factor_ids": ["string"], "start": "date", "end": "date", "max_pairs": 300, "max_to_backtest": 15}'
```


### POST /strategies/basket-stat-arb

**Summary**: Strategies Basket Stat Arb

PCA-residual statistical arbitrage on a basket of related events.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "factor_ids": [
    "string"
  ],
  "start": "date",
  "end": "date",
  "n_components": 0,
  "explained_variance_target": 0.7,
  "z_window": 20
}
```

**Response 200** (`application/json`):

```json
{
  "factor_ids": [
    "string"
  ],
  "n_obs": 0,
  "n_components_used": 0,
  "explained_variance_ratio": [
    0.0
  ],
  "loadings": [
    [
      0.0
    ]
  ],
  "kelly_fraction_per_market": {}
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/basket-stat-arb -H 'Content-Type: application/json' -d '{"factor_ids": ["string"], "start": "date", "end": "date", "n_components": 0, "explained_variance_target": 0.7, "z_window": 20}'
```


### POST /strategies/bounds

**Summary**: Strategies Bounds

Per-date Fréchet-Hoeffding bounds on the joint ``P(A ∩ B)``.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string"
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "n_obs": 0,
  "mean_lower": 0.0,
  "mean_upper": 0.0,
  "mean_width": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/bounds -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string"}'
```


### POST /strategies/cointegration

**Summary**: Strategies Cointegration

Engle-Granger 2-step cointegration test on a probability pair.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "significance": 0.05
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "n_obs": 0,
  "cointegrated": true,
  "verdict": "cointegrated",
  "beta_hedge": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/cointegration -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "significance": 0.05}'
```


### POST /strategies/conditional

**Summary**: Strategies Conditional

HAC-OLS regression of P_A on P_B (β interpretable as conditional sensitivity).

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "hac_lag": 5
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "n_obs": 0,
  "beta": 0.0,
  "beta_hac_se": 0.0,
  "beta_ci_lo": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/conditional -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "hac_lag": 5}'
```


### GET /strategies/crypto/5min/compare

**Summary**: Side-by-side model vs market for every BTC/ETH × 5m/15m combo

Always-on table of (asset × window) rows with both probabilities.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `edge_threshold` | query | number | no |  |
| `assets` | query | string | no | CSV of assets, default 'BTC,ETH' |
| `window_minutes_csv` | query | string | no | CSV of window sizes |
| `use_cache` | query | boolean | no | Skip the in-memory cache for fresh data |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/crypto/5min/compare
```


### GET /strategies/crypto/5min/diag

**Summary**: Spot-buffer diagnostics for the 5min predictor

Internal health surface: per-symbol sample count + age.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/crypto/5min/diag
```


### GET /strategies/crypto/5min/markets

**Summary**: Live model-vs-market table for every open 5m/15m crypto market

Discover & price every open short-dated crypto market.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `edge_threshold` | query | number | no |  |
| `assets` | query | string | no | CSV of assets, e.g. 'BTC,ETH' |
| `window_minutes_csv` | query | string | no | CSV of window sizes |
| `use_cache` | query | boolean | no | Skip the in-memory cache for fresh data |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/crypto/5min/markets
```


### GET /strategies/crypto/5min/predict/{symbol}

**Summary**: Pure-model P(up by end of next 5m/15m window) for one Binance pair

Return our model's up-probability for ``symbol`` for the *next* boundary.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `symbol` | path | string | yes |  |
| `window_minutes` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/strategies/crypto/5min/predict/<symbol>'
```


### GET /strategies/crypto/events

**Summary**: Live whale + mean-reversion events from the WS engine (last N min)

Return event-class signals captured by the in-process WS engine.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `window_min` | query | number | no |  |
| `symbol` | query | string | no |  |
| `kinds` | query | string | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/crypto/events
```


### GET /strategies/crypto/model-state/{symbol}

**Summary**: Live cryptostuff signals + annualized σ for the GBM model-prob calc

Expose the engine's live state for one symbol.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `symbol` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/strategies/crypto/model-state/<symbol>'
```


### GET /strategies/crypto/signals

**Summary**: Catalogue of the 9 microstructure signals computed by the WS engine

Static reference card — the signal taxonomy the WS engine produces.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/crypto/signals
```


### GET /strategies/crypto/snapshot

**Summary**: Live 10-pair microstructure snapshot (Binance REST)

Return live midprice/spread/OBI/24h-change for all 10 pairs.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/crypto/snapshot
```


### GET /strategies/crypto/spec

**Summary**: How to launch the WS engine locally + what to expect

Plain instructions the UI panel can render verbatim.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/strategies/crypto/spec
```


### POST /strategies/cusum

**Summary**: Strategies Cusum

Brown-Durbin-Evans CUSUM-OLS structural-break test on the
Engle-Granger spread. Detects level shifts / regime changes in the
cointegrating relationship — useful before deploying capital on a
pair: a recent break makes the historical β_hedge unreliable.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string"
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "n_obs": 0,
  "verdict": "stable",
  "rejected": true,
  "max_abs_cusum": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/cusum -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string"}'
```


### POST /strategies/dfa

**Summary**: Strategies Dfa

Peng et al. (1994) Detrended Fluctuation Analysis — robust Hurst
exponent on the integrated/cumulative-sum series. Robust to
non-stationary trends. α<0.5 = mean-reverting; α≈0.5 = random walk;
α>0.5 = persistent; α>1 = non-stationary.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "factor_id": "string",
  "start": "date",
  "end": "date",
  "poly_order": 1
}
```

**Response 200** (`application/json`):

```json
{
  "factor_id": "string",
  "n_obs": 0,
  "alpha_dfa": 0.0,
  "r_squared_log_log": 0.0,
  "interpretation": "mean_reverting",
  "log_n": [
    0.0
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/dfa -H 'Content-Type: application/json' -d '{"factor_id": "string", "start": "date", "end": "date", "poly_order": 1}'
```


### GET /strategies/discovery

**Summary**: Filter the strategies catalog by tag.

Return the catalog filtered by ``tag``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `tag` | query | string | no | Tag filter; 'all' disables. |

**Response 200** (`application/json`):

```json
{
  "total": 0,
  "items": [
    {
      "id": "string",
      "endpoint": "string",
      "method": "POST",
      "description": "string",
      "tag": "string"
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/strategies/discovery
```


### POST /strategies/distance-method

**Summary**: Strategies Distance Method

Gatev-Goetzmann-Rouwenhorst (2006) Distance Method.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "formation_fraction": 0.5
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "formation_ssd": 0.0,
  "formation_sigma": 0.0,
  "n_trading_bars": 0,
  "n_trades": 0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/distance-method -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "formation_fraction": 0.5}'
```


### POST /strategies/event-model

**Summary**: Strategies Event Model

HAC-OLS regression of one event probability on N other events.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "target_id": "string",
  "factor_ids": [
    "string"
  ],
  "start": "date",
  "end": "date",
  "hac_lag": 5
}
```

**Response 200** (`application/json`):

```json
{
  "target_id": "string",
  "factor_ids": [
    "string"
  ],
  "n_obs": 0,
  "intercept": 0.0,
  "intercept_se": 0.0,
  "coefficients": [
    {
      "factor_id": "string",
      "beta": 0.0,
      "hac_se": 0.0,
      "t_stat": 0.0,
      "p_value": 0.0,
      "ci_lo": 0.0
    }
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/event-model -H 'Content-Type: application/json' -d '{"target_id": "string", "factor_ids": ["string"], "start": "date", "end": "date", "hac_lag": 5}'
```


### POST /strategies/factor-model-pro

**Summary**: Strategies Factor Model Pro

Production-grade multi-event factor model.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "target_id": "string",
  "factor_ids": [
    "string"
  ],
  "start": "date",
  "end": "date",
  "estimator": "ols",
  "alpha": 1.0
}
```

**Response 200** (`application/json`):

```json
{
  "target_id": "string",
  "estimator": "string",
  "transform": "string",
  "use_pca": true,
  "n_obs": 0,
  "n_factors": 0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/factor-model-pro -H 'Content-Type: application/json' -d '{"target_id": "string", "factor_ids": ["string"], "start": "date", "end": "date", "estimator": "ols", "alpha": 1.0}'
```


### POST /strategies/fractional-diff

**Summary**: Strategies Fractional Diff

Hosking (1981) / López de Prado (2018 §5) fractional differentiation.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "factor_id": "string",
  "start": "date",
  "end": "date",
  "d": 0.0,
  "threshold": 0.001
}
```

**Response 200** (`application/json`):

```json
{
  "factor_id": "string",
  "minimal_d": 0.0,
  "adf_p_at_minimal_d": 0.0,
  "correlation_with_original": 0.0,
  "weights_width": 0,
  "grid": [
    {
      "d": 0.0,
      "adf_p": 0.0,
      "corr_with_original": 0.0,
      "n_after_filter": 0
    }
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/fractional-diff -H 'Content-Type: application/json' -d '{"factor_id": "string", "start": "date", "end": "date", "d": 0.0, "threshold": 0.001}'
```


### POST /strategies/fred-cointegration

**Summary**: Strategies Fred Cointegration

Engle-Granger cointegration test between a factor and a FRED macro series.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "factor_id": "string",
  "fred_series": "DFF",
  "start": "date",
  "end": "date",
  "transform": "raw"
}
```

**Response 200** (`application/json`):

```json
{
  "factor_id": "string",
  "fred_series": "string",
  "n_obs": 0,
  "adf_pvalue": 0.0,
  "beta_hedge": 0.0,
  "half_life_days": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/fred-cointegration -H 'Content-Type: application/json' -d '{"factor_id": "string", "fred_series": "DFF", "start": "date", "end": "date", "transform": "raw"}'
```


### POST /strategies/garch

**Summary**: Strategies Garch

Bollerslev (1986) GARCH(1,1) — conditional volatility on Δ-series.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "factor_id": "string",
  "start": "date",
  "end": "date"
}
```

**Response 200** (`application/json`):

```json
{
  "factor_id": "string",
  "n_obs": 0,
  "converged": true,
  "is_stationary": true,
  "mu": 0.0,
  "omega": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/garch -H 'Content-Type: application/json' -d '{"factor_id": "string", "start": "date", "end": "date"}'
```


### POST /strategies/granger

**Summary**: Strategies Granger

Bivariate Granger causality between two event probability series.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "max_lag": 5
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "n_obs": 0,
  "direction": "B_causes_A",
  "best_lag_b_to_a": 0,
  "best_pvalue_b_to_a": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/granger -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "max_lag": 5}'
```


### POST /strategies/implication

**Summary**: Strategies Implication

Test the logical-implication invariant ``A ⇒ B`` ⇒ ``P(A) ≤ P(B)``.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "antecedent_id": "string",
  "consequent_id": "string",
  "tolerance": 0.02
}
```

**Response 200** (`application/json`):

```json
{
  "antecedent_id": "string",
  "consequent_id": "string",
  "n_obs": 0,
  "verdict": "consistent",
  "n_violations": 0,
  "violation_dates": [
    "date"
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/implication -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "antecedent_id": "string", "consequent_id": "string", "tolerance": 0.02}'
```


### POST /strategies/info-share

**Summary**: Strategies Info Share

Hasbrouck (1995) Information Share — for two cointegrated price
series, decomposes the proportion of long-run price discovery each
venue contributes. The leader's IS is closer to 1; the follower's
closer to 0. Use to identify which venue (Kalshi vs Polymarket) drives
a cross-platform Fed-cut basis.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "var_lags": 5
}
```

**Response 200** (`application/json`):

```json
{
  "venue_a_id": "string",
  "venue_b_id": "string",
  "n_obs": 0,
  "is_a_lower": 0.0,
  "is_a_upper": 0.0,
  "is_b_lower": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/info-share -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "var_lags": 5}'
```


### POST /strategies/kalman-hedge

**Summary**: Strategies Kalman Hedge

Time-varying hedge ratio β_t via Kalman filter.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "delta": 0.0001
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "n_obs": 0,
  "delta": 0.0,
  "r": 0.0,
  "q": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/kalman-hedge -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "delta": 0.0001}'
```


### GET /strategies/list

**Summary**: Enumerate every /strategies/* endpoint with metadata.

Return the full catalog enumeration.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "total": 0,
  "items": [
    {
      "id": "string",
      "endpoint": "string",
      "method": "POST",
      "description": "string",
      "tag": "string"
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/strategies/list
```


### POST /strategies/mean-reversion

**Summary**: Strategies Mean Reversion

Hurst exponent (R/S) + Lo-MacKinlay variance-ratio test on a single
factor's probability series. Both quantify mean-reversion strength
model-free.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "factor_id": "string",
  "start": "date",
  "end": "date",
  "vr_q": 2
}
```

**Response 200** (`application/json`):

```json
{
  "factor_id": "string",
  "n_obs": 0,
  "hurst": 0.0,
  "hurst_r_squared": 0.0,
  "hurst_interpretation": "mean_reverting",
  "vr_q": 0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/mean-reversion -H 'Content-Type: application/json' -d '{"factor_id": "string", "start": "date", "end": "date", "vr_q": 2}'
```


### POST /strategies/ml-predictor

**Summary**: Strategies Ml Predictor

Gradient-boosted regressor predicting next-bar Δspread from
engineered features (lag-z, rolling vol, autocorrelation, momentum,
long-window distance-from-mean). TimeSeriesSplit cross-validation.
Reports R², direction accuracy, information coefficient, beats-baseline,
feature importances, and a forward-looking last_prediction.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "n_folds": 5
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "n_obs": 0,
  "n_features": 0,
  "feature_names": [
    "string"
  ],
  "n_folds": 0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/ml-predictor -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "n_folds": 5}'
```


### POST /strategies/optimize

**Summary**: Optimize

Suggest optimal weights for a basket of curated alphas.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `format` | query | string | no |  |
| `authorization` | header | string | no |  |
| `x-api-key` | header | string | no |  |

**Request Body** (`application/json`):

```json
{
  "pair_ids": [
    "string"
  ],
  "method": "hrp",
  "lookback_days": 252,
  "risk_free_rate": 0.045,
  "max_weight": 0.3,
  "min_weight": 0.0
}
```

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/optimize -H 'Content-Type: application/json' -d '{"pair_ids": ["string"], "method": "hrp", "lookback_days": 252, "risk_free_rate": 0.045, "max_weight": 0.3, "min_weight": 0.0}'
```


### POST /strategies/ou-bands

**Summary**: Strategies Ou Bands

Calibrate OU dynamics on the cointegration spread + Bertram (2010)
optimal entry/exit z-bands.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "transaction_cost_sigma": 0.1
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "n_obs": 0,
  "cointegrated": true,
  "kappa": 0.0,
  "mu": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/ou-bands -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "transaction_cost_sigma": 0.1}'
```


### POST /strategies/pairs-backtest

**Summary**: Strategies Pairs Backtest

Walk-forward z-score pairs trade on the spread of two probability series.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "window": 20
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "n_obs": 0,
  "n_trades": 0,
  "sharpe": 0.0,
  "sortino": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/pairs-backtest -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "window": 20}'
```


### POST /strategies/patterns

**Summary**: Strategies Patterns

Cross-pair structural-pattern analysis: PnL correlation matrix,
day-of-week effects per pair, pre-resolution regime shifts, k-means
clustering on pair signatures.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "pairs": [
    {
      "a_id": "string",
      "b_id": "string"
    }
  ],
  "start": "date",
  "end": "date",
  "window": 20,
  "entry_z": 2.0,
  "exit_z": 0.5
}
```

**Response 200** (`application/json`):

```json
{
  "n_pairs_analysed": 0,
  "correlation": {
    "pair_labels": [
      "string"
    ],
    "correlation_matrix": [
      [
        null
      ]
    ],
    "mean_off_diagonal": 0.0,
    "max_off_diagonal": 0.0,
    "most_correlated_a": "string",
    "most_correlated_b": "string"
  },
  "day_of_week": [
    {
      "pair": "string",
      "means": {},
      "counts": {},
      "t_stats": {},
      "p_values": {},
      "best_day": null
    }
  ],
  "pre_resolution": [
    {
      "pair": "string",
      "far_n": 0,
      "near_n": 0,
      "far_std": null,
      "near_std": null,
      "vol_ratio": null
    }
  ],
  "clusters": [
    {
      "cluster_id": 0,
      "pair_labels": [
        null
      ],
      "centroid": {},
      "n_members": 0
    }
  ],
  "silhouette_proxy": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/patterns -H 'Content-Type: application/json' -d '{"pairs": [{"a_id": "string", "b_id": "string"}], "start": "date", "end": "date", "window": 20, "entry_z": 2.0, "exit_z": 0.5}'
```


### POST /strategies/portfolio

**Summary**: Strategies Portfolio

Vol-targeted portfolio combiner. Aggregates the per-bar PnLs of N
pair-trading strategies into a single equity curve, weighted so each
leg contributes ``target_per_leg_vol`` annualised volatility.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "pairs": [
    {
      "a_id": "string",
      "b_id": "string",
      "signal_type": "zscore",
      "window": 20
    }
  ],
  "start": "date",
  "end": "date",
  "target_per_leg_vol": 0.1,
  "walk_forward_folds": 5
}
```

**Response 200** (`application/json`):

```json
{
  "n_pairs": 0,
  "pair_labels": [
    "string"
  ],
  "weights": {},
  "individual_sharpes": {},
  "correlation_matrix": [
    [
      0.0
    ]
  ],
  "n_obs": 0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/portfolio -H 'Content-Type: application/json' -d '{"pairs": [{"a_id": "string", "b_id": "string", "signal_type": "zscore", "window": 20}], "start": "date", "end": "date", "target_per_leg_vol": 0.1, "walk_forward_folds": 5}'
```


### GET /strategies/presets

**Summary**: Strategies Presets

Curated example inputs for every Strategies sub-tool.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "cointegration": [
    {
      "label": "string",
      "description": "string",
      "inputs": {},
      "metric": null,
      "tier": null
    }
  ],
  "pairs": [
    {
      "label": "string",
      "description": "string",
      "inputs": {},
      "metric": null,
      "tier": null
    }
  ],
  "pair_explorer": [
    {
      "label": "string",
      "description": "string",
      "inputs": {},
      "metric": null,
      "tier": null
    }
  ],
  "event_model": [
    {
      "label": "string",
      "description": "string",
      "inputs": {},
      "metric": null,
      "tier": null
    }
  ],
  "basket": [
    {
      "label": "string",
      "description": "string",
      "inputs": {},
      "metric": null,
      "tier": null
    }
  ],
  "spot_vs_implied": [
    {
      "label": "string",
      "description": "string",
      "inputs": {},
      "metric": null,
      "tier": null
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/strategies/presets
```


### POST /strategies/regime-switching

**Summary**: Strategies Regime Switching

Hamilton (1989) Markov-switching variance model on the cointegration
spread. State 0 = tight mean-reversion (low σ, tradeable); state 1 =
broken (high σ, regime change risk). Returns smoothed P(state=1) per bar.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "k_regimes": 2
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "n_obs": 0,
  "n_state0": 0,
  "n_state1": 0,
  "sigma_state0": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/regime-switching -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "k_regimes": 2}'
```


### POST /strategies/robust-validation

**Summary**: Strategies Robust Validation

Comprehensive robustness battery on a portfolio of pair trades.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "pairs": [
    {
      "a_id": "string",
      "b_id": "string",
      "signal_type": "zscore",
      "window": 20
    }
  ],
  "start": "date",
  "end": "date",
  "target_per_leg_vol": 0.1,
  "annualisation": 252.0,
  "n_trials_searched": 100
}
```

**Response 200** (`application/json`):

```json
{
  "portfolio_sharpe": 0.0,
  "n_obs": 0,
  "overall_verdict": "STRONG ALPHA",
  "n_tests_passed": 0,
  "lo_sharpe": 0.0,
  "lo_se": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/robust-validation -H 'Content-Type: application/json' -d '{"pairs": [{"a_id": "string", "b_id": "string", "signal_type": "zscore", "window": 20}], "start": "date", "end": "date", "target_per_leg_vol": 0.1, "annualisation": 252.0, "n_trials_searched": 100}'
```


### POST /strategies/scan

**Summary**: Strategies Scan

Cartesian inefficiency scanner across the factor catalog.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "mode": "all",
  "theme": "string",
  "factor_ids": [
    "string"
  ],
  "start": "date",
  "end": "date",
  "max_pairs": 500
}
```

**Response 200** (`application/json`):

```json
{
  "mode": "implication",
  "n_factors_scanned": 0,
  "n_pairs_evaluated": 0,
  "runtime_seconds": 0.0,
  "implication": [
    {
      "kind": "implication",
      "a_id": "string",
      "b_id": "string",
      "score": 0.0,
      "n_obs": 0,
      "summary": "string"
    }
  ],
  "conditional": [
    {
      "kind": "implication",
      "a_id": "string",
      "b_id": "string",
      "score": 0.0,
      "n_obs": 0,
      "summary": "string"
    }
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/scan -H 'Content-Type: application/json' -d '{"mode": "all", "theme": "string", "factor_ids": ["string"], "start": "date", "end": "date", "max_pairs": 500}'
```


### POST /strategies/sharpe-bootstrap

**Summary**: Strategies Sharpe Bootstrap

Stationary block-bootstrap (Politis-Romano 1994) CI on the Sharpe
of a pair's z-score backtest. CI excluding zero ⇒ Sharpe is statistically
distinguishable from random.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "window": 20
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "sharpe_point": 0.0,
  "sharpe_mean": 0.0,
  "sharpe_std": 0.0,
  "sharpe_ci_lo_90": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/sharpe-bootstrap -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "window": 20}'
```


### POST /strategies/sharpe-permutation

**Summary**: Strategies Sharpe Permutation

Permutation null distribution of the Sharpe ratio. Sign-flips the
spread's first differences; rebuilds; runs the same strategy; computes
Sharpe. ``p = P(null Sharpe ≥ real Sharpe)``. p < 0.05 ⇒ the real
Sharpe doesn't come from random fluctuations of the spread.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "window": 20
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "real_sharpe": 0.0,
  "null_sharpes": [
    0.0
  ],
  "null_median": 0.0,
  "null_pct95": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/sharpe-permutation -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "window": 20}'
```


### POST /strategies/spot-vs-implied

**Summary**: Strategies Spot Vs Implied

Compare a live underlying (Binance daily klines) to a market-implied
YES-price for a price-target binary outcome.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "symbol": "string",
  "strike": 0.0,
  "expiry": "date",
  "geometry": "terminal",
  "market_prob": 0.0,
  "drift_annual": 0.0
}
```

**Response 200** (`application/json`):

```json
{
  "symbol": "string",
  "interval": "string",
  "spot": 0.0,
  "strike": 0.0,
  "expiry": "date",
  "geometry": "terminal"
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/spot-vs-implied -H 'Content-Type: application/json' -d '{"symbol": "string", "strike": 0.0, "expiry": "date", "geometry": "terminal", "market_prob": 0.0, "drift_annual": 0.0}'
```


### POST /strategies/triple-barrier

**Summary**: Strategies Triple Barrier

López de Prado (2018) Triple Barrier Method on a cointegration spread.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "window": 20
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "n_trades": 0,
  "n_profit_hits": 0,
  "n_stop_hits": 0,
  "n_time_hits": 0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/triple-barrier -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "window": 20}'
```


### POST /strategies/walk-forward

**Summary**: Strategies Walk Forward

K-fold walk-forward backtest. Reports the *distribution* of test-fold
Sharpes — much more credible than a single train/test split. Stable =
min(test Sharpe) > 0 AND std(test Sharpe) < |mean|.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start": "date",
  "end": "date",
  "epsilon": 0.01,
  "a_id": "string",
  "b_id": "string",
  "n_folds": 5
}
```

**Response 200** (`application/json`):

```json
{
  "a_id": "string",
  "b_id": "string",
  "n_obs": 0,
  "n_folds": 0,
  "folds": [
    {
      "fold": 0,
      "test_start": "date",
      "test_end": "date",
      "train_sharpe": 0.0,
      "test_sharpe": 0.0,
      "n_train": 0
    }
  ],
  "train_sharpe_mean": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategies/walk-forward -H 'Content-Type: application/json' -d '{"start": "date", "end": "date", "epsilon": 0.01, "a_id": "string", "b_id": "string", "n_folds": 5}'
```


## Alpha Hub

### GET /alpha-hub/graveyard

**Summary**: List dead / downgraded alpha strategies

Return all entries in the alpha graveyard, optionally filtered by cause.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `cause` | query | string | no | Filter by failure mode; 'all' returns the full list. |

**Response 200** (`application/json`):

```json
{
  "n_entries": 0,
  "cause_filter": "all",
  "entries": [
    {
      "pair_id": "string",
      "name": "string",
      "killed_iso": "string",
      "killed_in_wave": 0,
      "cause": "regime",
      "claimed_sharpe": 0.0
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/alpha-hub/graveyard
```


### GET /alpha-hub/graveyard/{pair_id}

**Summary**: Fetch a single death certificate

Return the death certificate for a single ``pair_id``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `pair_id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "pair_id": "string",
  "name": "string",
  "killed_iso": "string",
  "killed_in_wave": 0,
  "cause": "regime",
  "claimed_sharpe": 0.0
}
```

**Example**:

```bash
curl 'http://localhost:8000/alpha-hub/graveyard/<pair_id>'
```


### GET /alpha-hub/leaderboard

**Summary**: Paginated, filtered, sortable view of curated alpha strategies.

Return a paginated leaderboard slice for the discovery UI.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `tier` | query | string | no | Tier filter; 'all' disables. |
| `theme` | query | string | no | Match theme_a or theme_b (case-insensitive). |
| `min_sharpe` | query | number | no | Drop rows where oos_sharpe < min_sharpe. |
| `sort` | query | string | no | Sort key. |
| `order` | query | string | no | Sort order. |
| `limit` | query | integer | no |  |
| `offset` | query | integer | no |  |
| `full` | query | boolean | no | When true, items contain the raw catalog dicts (every field preserved) and a ``meta`` block of top-level summary counts is included. Used by the frontend so the API is the single source of truth for the discovery panel. |

**Response 200** (`application/json`):

```json
{
  "total": 0,
  "n_returned": 0,
  "offset": 0,
  "limit": 0,
  "sort": "oos_sharpe",
  "order": "desc"
}
```

**Example**:

```bash
curl http://localhost:8000/alpha-hub/leaderboard
```


### GET /alpha-hub/live-panel

**Summary**: Composite payload: top production alphas + watchlist + recent graveyard.

Return a small dashboard payload suitable for the hub landing card.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "production": [
    {
      "pair_id": "string",
      "tier": "string",
      "theme_a": null,
      "theme_b": null,
      "category": null,
      "oos_sharpe": null
    }
  ],
  "watchlist": [
    {
      "pair_id": "string",
      "tier": "string",
      "theme_a": null,
      "theme_b": null,
      "category": null,
      "oos_sharpe": null
    }
  ],
  "graveyard": [
    {}
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/alpha-hub/live-panel
```


### POST /alpha-hub/regenerate-tiers

**Summary**: Re-run the walk-forward harness over alpha_strategies.json

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `x-admin-token` | header | string | no |  |

**Request Body** (`application/json`):

```json
{
  "output_mode": "backup",
  "max_runtime_seconds": 600,
  "history_days": 120,
  "n_folds": 4,
  "perm_iters": 200,
  "fetch_concurrency": 10
}
```

**Response 200** (`application/json`):

```json
{
  "job_id": "string",
  "status": "string",
  "started_at": "string",
  "params": {}
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/alpha-hub/regenerate-tiers -H 'Content-Type: application/json' -d '{"output_mode": "backup", "max_runtime_seconds": 600, "history_days": 120, "n_folds": 4, "perm_iters": 200, "fetch_concurrency": 10}'
```


### GET /alpha-hub/regenerate-tiers/{job_id}

**Summary**: Fetch status / summary of a regen job

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `job_id` | path | string | yes |  |
| `x-admin-token` | header | string | no |  |

**Response 200** (`application/json`):

```json
{
  "job_id": "string",
  "status": "string",
  "started_at": "string",
  "completed_at": "string",
  "summary": {},
  "written_path": "string"
}
```

**Example**:

```bash
curl 'http://localhost:8000/alpha-hub/regenerate-tiers/<job_id>'
```


### GET /alpha-hub/strategy/{pair_id}

**Summary**: Full per-strategy detail (all fields from alpha_strategies.json).

Return the full, untrimmed strategy entry for ``pair_id``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `pair_id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/alpha-hub/strategy/<pair_id>'
```


## Alpha

### GET /alpha/decay

**Summary**: List Decay Status

List the decay status of every strategy in the catalog.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `window` | query | integer | no |  |
| `alpha_strategies_path` | query | string | no |  |
| `data_source` | query | string | no |  |
| `live_signals_path` | query | string | no |  |
| `allow_polymarket` | query | boolean | no |  |

**Response 200** (`application/json`):

```json
{
  "n_total": 0,
  "n_fresh": 0,
  "n_stable": 0,
  "n_decaying": 0,
  "n_dead": 0,
  "n_using_real_data": 0
}
```

**Example**:

```bash
curl http://localhost:8000/alpha/decay
```


### GET /alpha/earnings-calendar

**Summary**: Get Earnings Calendar

Upcoming earnings calendar (Polygon when configured, hardcoded fallback).

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `days` | query | integer | no |  |
| `source` | query | string | no |  |

**Response 200** (`application/json`):

```json
{
  "n": 0,
  "horizon_days": 0,
  "source": "cached",
  "rows": [
    {
      "ticker": "string",
      "earnings_date": "string",
      "consensus_eps": null,
      "n_analysts": 0
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/alpha/earnings-calendar
```


### GET /alpha/earnings-whisper-dashboard

**Summary**: Get Whisper Dashboard

Whisper rows for every ticker with earnings inside ``days``, sorted by |edge|.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `days` | query | integer | no |  |
| `source` | query | string | no |  |

**Response 200** (`application/json`):

```json
{
  "n": 0,
  "horizon_days": 0,
  "source": "cached",
  "rows": [
    {
      "ticker": "string",
      "earnings_date": "string",
      "consensus_eps": 0.0,
      "consensus_source": "hardcoded_snapshot",
      "pm_beat_prob": 0.0,
      "expected_beat_pct": 0.0
    }
  ],
  "cache_age_seconds": 0,
  "is_stale": false
}
```

**Example**:

```bash
curl http://localhost:8000/alpha/earnings-whisper-dashboard
```


### GET /alpha/earnings-whisper/{ticker}

**Summary**: Get Whisper

Compute the whisper EPS for one ticker.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `ticker` | path | string | yes |  |
| `date` | query | string | no |  |
| `source` | query | string | no |  |

**Response 200** (`application/json`):

```json
{
  "ticker": "string",
  "earnings_date": "string",
  "consensus_eps": 0.0,
  "consensus_source": "hardcoded_snapshot",
  "pm_beat_prob": 0.0,
  "expected_beat_pct": 0.0
}
```

**Example**:

```bash
curl 'http://localhost:8000/alpha/earnings-whisper/<ticker>'
```


### POST /alpha/prediction-driven

**Summary**: Prediction Driven Endpoint

Univariate β / R² scan: which equities load on a single PM factor.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "factor_id": "string",
  "candidate_tickers": [
    "string"
  ],
  "window_days": 252,
  "top_n": 12,
  "delta_logit_assumed": 0.0,
  "return_type": "log"
}
```

**Response 200** (`application/json`):

```json
{
  "factor_id": "string",
  "factor_name": "string",
  "tickers": [
    {
      "ticker": "string",
      "beta": 0.0,
      "r_squared": 0.0,
      "t_stat": 0.0,
      "n_obs": 0,
      "expected_return_pct": null
    }
  ],
  "ranked_by": "string",
  "delta_logit_assumed": 0.0,
  "window_days": 0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/alpha/prediction-driven -H 'Content-Type: application/json' -d '{"factor_id": "string", "candidate_tickers": ["string"], "window_days": 252, "top_n": 12, "delta_logit_assumed": 0.0, "return_type": "log"}'
```


### POST /alpha/{pair_id}/recompute-decay

**Summary**: Recompute Decay

Force a recompute of the decay status for one strategy.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `pair_id` | path | string | yes |  |
| `window` | query | integer | no |  |
| `alpha_strategies_path` | query | string | no |  |
| `data_source` | query | string | no |  |
| `live_signals_path` | query | string | no |  |
| `allow_polymarket` | query | boolean | no |  |

**Response 200** (`application/json`):

```json
{
  "pair_id": "string",
  "status": {
    "pair_id": "string",
    "tier": "string",
    "current_sharpe": 0.0,
    "baseline_sharpe": 0.0,
    "ratio": 0.0,
    "decay_indicator": "FRESH"
  },
  "forced": true
}
```

**Example**:

```bash
curl -X POST 'http://localhost:8000/alpha/<pair_id>/recompute-decay'
```


### GET /alpha/{pair_id}/rolling-sharpe

**Summary**: Get Rolling Sharpe

Return the daily rolling-Sharpe series for one strategy.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `pair_id` | path | string | yes |  |
| `window` | query | integer | no |  |
| `alpha_strategies_path` | query | string | no |  |
| `data_source` | query | string | no |  |
| `live_signals_path` | query | string | no |  |
| `allow_polymarket` | query | boolean | no |  |

**Response 200** (`application/json`):

```json
{
  "pair_id": "string",
  "window": 0,
  "baseline_sharpe": 0.0,
  "n_obs": 0,
  "series": [
    {
      "date": "string",
      "rolling_sharpe": null
    }
  ],
  "source_used": "synthetic_fallback"
}
```

**Example**:

```bash
curl 'http://localhost:8000/alpha/<pair_id>/rolling-sharpe'
```


## Archive

### GET /archive/cross-venue/concepts

**Summary**: Catalog of pre-mapped cross-venue concepts (PM vs Kalshi).

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "concepts": [
    {
      "concept": "string",
      "description": null,
      "polymarket_slug": null,
      "kalshi_ticker": null,
      "resolved_outcome": null
    }
  ],
  "n": 0
}
```

**Example**:

```bash
curl http://localhost:8000/archive/cross-venue/concepts
```


### GET /archive/cross-venue/{concept}

**Summary**: Polymarket vs Kalshi divergence metrics for a resolved concept.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `concept` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "concept": "string",
  "description": "string",
  "polymarket_slug": "string",
  "kalshi_ticker": "string",
  "resolved_outcome": "string",
  "n_overlap_days": 0
}
```

**Example**:

```bash
curl 'http://localhost:8000/archive/cross-venue/<concept>'
```


### GET /archive/kalshi/market/{ticker}

**Summary**: Per-market detail (metadata + history + stats), optionally as CSV.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `ticker` | path | string | yes |  |
| `format` | query | string | no |  |

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl 'http://localhost:8000/archive/kalshi/market/<ticker>'
```


### GET /archive/kalshi/markets

**Summary**: Paginated list of settled Kalshi markets.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `start` | query | string | no | Lower bound on settle date. |
| `end` | query | string | no | Upper bound on settle date. |
| `series` | query | string | no | Restrict to one series ticker (e.g. KXFEDDECISION). |
| `limit` | query | integer | no |  |
| `offset` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "items": [
    {
      "ticker": "string",
      "title": "string",
      "series": "string",
      "settle_date": null,
      "settle_value": null,
      "open_interest": 0.0
    }
  ],
  "n": 0,
  "limit": 0,
  "offset": 0,
  "series_ticker": "string",
  "start": "string"
}
```

**Example**:

```bash
curl http://localhost:8000/archive/kalshi/markets
```


### GET /archive/kalshi/series

**Summary**: Per-series stats over all settled Kalshi markets.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "series": {},
  "n_total_markets": 0,
  "n_series": 0
}
```

**Example**:

```bash
curl http://localhost:8000/archive/kalshi/series
```


### GET /archive/list

**Summary**: Alias of /archive/polymarket/markets (footer pill).

Footer-pill friendly alias — delegates to ``list_resolved_markets``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `start` | query | string | no |  |
| `end` | query | string | no |  |
| `theme` | query | string | no |  |
| `limit` | query | integer | no |  |
| `offset` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "n_markets": 0,
  "limit": 0,
  "offset": 0,
  "markets": [
    {
      "id": "string",
      "slug": null,
      "question": "string",
      "theme": "string",
      "end_date": null,
      "resolution": "YES"
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/archive/list
```


### POST /archive/polymarket/export-bulk

**Summary**: Bulk-export N archive markets as a ZIP of per-slug files.

Build a ZIP with one file per slug in the requested ``format``.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "slugs": [
    "string"
  ],
  "format": "csv"
}
```

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl -X POST http://localhost:8000/archive/polymarket/export-bulk -H 'Content-Type: application/json' -d '{"slugs": ["string"], "format": "csv"}'
```


### GET /archive/polymarket/market/{slug}

**Summary**: Full archive detail (history + stats) for one resolved market.

Return market detail. ``?format=csv`` streams the daily-history table.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `format` | query | string | no | ``json`` (default) or ``csv`` for the price-history table. |

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl 'http://localhost:8000/archive/polymarket/market/<slug>'
```


### GET /archive/polymarket/markets

**Summary**: Paginated list of resolved Polymarket markets in a date range.

List resolved markets with paging + optional theme filter.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `start` | query | string | no | Lower bound on resolution end-date (ISO YYYY-MM-DD). Defaults to 1 year ago. |
| `end` | query | string | no | Upper bound on resolution end-date (ISO YYYY-MM-DD). Defaults to today. |
| `theme` | query | string | no | Optional theme filter, e.g. ``politics``, ``crypto``, ``sports``. |
| `limit` | query | integer | no |  |
| `offset` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "n_markets": 0,
  "limit": 0,
  "offset": 0,
  "markets": [
    {
      "id": "string",
      "slug": null,
      "question": "string",
      "theme": "string",
      "end_date": null,
      "resolution": "YES"
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/archive/polymarket/markets
```


### GET /archive/polymarket/resolutions/{slug}

**Summary**: Resolution outcome only (no price history).

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "resolution": "YES",
  "resolution_date": "string",
  "resolution_source": "string",
  "payout_per_share": 0.0,
  "dispute_history": [
    {}
  ]
}
```

**Example**:

```bash
curl 'http://localhost:8000/archive/polymarket/resolutions/<slug>'
```


### GET /archive/polymarket/search

**Summary**: Substring search over resolved-market slug + question.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `q` | query | string | yes |  |
| `limit` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "q": "string",
  "n_results": 0,
  "results": [
    {
      "id": "string",
      "slug": null,
      "question": "string",
      "theme": "string",
      "end_date": null,
      "resolution": "YES"
    }
  ]
}
```

**Example**:

```bash
curl 'http://localhost:8000/archive/polymarket/search?q=string'
```


### GET /archive/polymarket/themes

**Summary**: Aggregate stats per theme across the most recent resolved markets.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "n_markets_total": 0,
  "themes": [
    {
      "theme": "string",
      "n_markets": 0,
      "pct_yes": 0.0,
      "pct_no": 0.0,
      "pct_ambiguous": 0.0,
      "avg_duration_days": null
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/archive/polymarket/themes
```


## Factors

### GET /factors

**Summary**: List Factors

Paginated factor catalog (default page=50, cap=500). See ``/factors/all`` for full dump.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `limit` | query | integer | no |  |
| `offset` | query | integer | no |  |
| `theme` | query | string | no |  |
| `source` | query | string | no |  |
| `search` | query | string | no |  |

**Response 200** (`application/json`):

```json
{
  "factors": [
    {
      "id": "string",
      "name": "string",
      "slug": "string",
      "source": "string",
      "description": "string",
      "theme": "other"
    }
  ],
  "total": 0,
  "limit": 0,
  "offset": 0,
  "next_offset": 0
}
```

**Example**:

```bash
curl http://localhost:8000/factors
```


### GET /factors/all

**Summary**: List Factors All

Full factor dump (~500 KB at ~1 360 entries). Prefer ``/factors`` with pagination.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "factors": [
    {
      "id": "string",
      "name": "string",
      "slug": "string",
      "source": "string",
      "description": "string",
      "theme": "other"
    }
  ],
  "total": 0,
  "limit": 0,
  "offset": 0,
  "next_offset": 0
}
```

**Example**:

```bash
curl http://localhost:8000/factors/all
```


### POST /factors/best

**Summary**: Best Model

Forward stepwise selection — greedily build a multi-factor model by R²adj or OOS-R².

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `epsilon` | query | number | no |  |

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "start": "date",
  "end": "date",
  "return_type": "log",
  "regression": "hac",
  "alignment": "strict"
}
```

**Response 200** (`application/json`):

```json
{
  "ticker": "string",
  "start": "date",
  "end": "date",
  "selected": [
    "string"
  ],
  "final_r_squared": 0.0,
  "final_r_squared_adj": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/factors/best -H 'Content-Type: application/json' -d '{"ticker": "string", "start": "date", "end": "date", "return_type": "log", "regression": "hac", "alignment": "strict"}'
```


### GET /factors/discover

**Summary**: Discover Factors

Surface high-volume active markets as candidate factors.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `min_volume` | query | number | no |  |
| `limit` | query | integer | no |  |
| `keyword` | query | string | no |  |

**Response 200** (`application/json`):

```json
{
  "markets": [
    {
      "slug": "string",
      "question": "string",
      "volume": 0.0,
      "end_date": null,
      "active": true,
      "closed": true
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/factors/discover
```


### POST /factors/permutation

**Summary**: Factors Permutation

Standalone permutation test — shuffle factor values, refit, return p-value.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `epsilon` | query | number | no |  |

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "factors": [
    "string"
  ],
  "custom_factors": [
    {
      "id": "string",
      "slug": "string",
      "name": null
    }
  ],
  "start": "date",
  "end": "date",
  "return_type": "log"
}
```

**Response 200** (`application/json`):

```json
{
  "real_test_r2": 0.0,
  "null_test_r2s": [
    0.0
  ],
  "null_median": 0.0,
  "null_pct95": 0.0,
  "null_max": 0.0,
  "p_value": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/factors/permutation -H 'Content-Type: application/json' -d '{"ticker": "string", "factors": ["string"], "custom_factors": [{"id": "string", "slug": "string", "name": null}], "start": "date", "end": "date", "return_type": "log"}'
```


### POST /factors/preview

**Summary**: Preview Factor

Look up a slug, return metadata + recent price history for the UI.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "slug": "string",
  "source": "polymarket"
}
```

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "question": "string",
  "yes_token_id": "string",
  "active": true,
  "closed": true,
  "n_bars": 0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/factors/preview -H 'Content-Type: application/json' -d '{"slug": "string", "source": "polymarket"}'
```


### POST /factors/rank

**Summary**: Rank Factors

Rank curated factors by single-factor R² for ``body.ticker``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `epsilon` | query | number | no |  |

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "start": "date",
  "end": "date",
  "return_type": "log",
  "regression": "hac",
  "alignment": "strict"
}
```

**Response 200** (`application/json`):

```json
{
  "ticker": "string",
  "start": "date",
  "end": "date",
  "return_type": "log",
  "regression": "ols",
  "items": [
    {
      "factor_id": "string",
      "name": "string",
      "slug": "string",
      "theme": "string",
      "n_obs": 0,
      "r_squared": 0.0
    }
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/factors/rank -H 'Content-Type: application/json' -d '{"ticker": "string", "start": "date", "end": "date", "return_type": "log", "regression": "hac", "alignment": "strict"}'
```


### POST /factors/suggest-for-ticker

**Summary**: Suggest Factors For Ticker

Smart factor picker — top-K factors most correlated with a ticker.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "lookback_days": 90,
  "top_k": 10,
  "min_n_obs": 30
}
```

**Response 200** (`application/json`):

```json
{
  "ticker": "string",
  "lookback_days": 0,
  "n_factors_scanned": 0,
  "n_factors_skipped": 0,
  "top_factors": [
    {
      "factor_id": "string",
      "name": "string",
      "source": "string",
      "theme": null,
      "r": 0.0,
      "abs_r": 0.0
    }
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/factors/suggest-for-ticker -H 'Content-Type: application/json' -d '{"ticker": "string", "lookback_days": 90, "top_k": 10, "min_n_obs": 30}'
```


## Auth

### POST /auth/demo-key

**Summary**: Mint a 24h Free-tier demo key (open, no admin token required)

Hands out a short-lived Free key for in-browser demos.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/auth/demo-key
```


### GET /auth/first-boot-info

**Summary**: One-shot retrieval of the autogenerated admin token (prod only)

Return the active admin token exactly once after a fresh boot.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/auth/first-boot-info
```


### POST /auth/keys

**Summary**: Create a new API key (admin only)

Mint a new key. The plaintext is returned exactly once.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `x-admin-token` | header | string | no |  |

**Request Body** (`application/json`):

```json
{
  "user_id": "string",
  "tier": "free"
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/auth/keys -H 'Content-Type: application/json' -d '{"user_id": "string", "tier": "free"}'
```


### GET /auth/keys/me

**Summary**: Inspect the API key in use on this request

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `authorization` | header | string | no |  |
| `x-api-key` | header | string | no |  |

**Response 200** (`application/json`):

```json
{
  "key_masked": "string",
  "user_id": "string",
  "tier": "free",
  "created_at": "date-time",
  "last_used_at": "date-time",
  "enabled": true
}
```

**Example**:

```bash
curl http://localhost:8000/auth/keys/me
```


### GET /auth/keys/me/usage

**Summary**: Usage stats for the API key in use

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `authorization` | header | string | no |  |
| `x-api-key` | header | string | no |  |

**Response 200** (`application/json`):

```json
{
  "user_id": "string",
  "tier": "free",
  "requests_this_minute": 0,
  "requests_today": 0,
  "rate_limit_per_minute": 0,
  "daily_quota": 0
}
```

**Example**:

```bash
curl http://localhost:8000/auth/keys/me/usage
```


### DELETE /auth/keys/{key}

**Summary**: Revoke a key (admin only)

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `key` | path | string | yes |  |
| `x-admin-token` | header | string | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X DELETE 'http://localhost:8000/auth/keys/<key>'
```


### GET /auth/usage/dashboard

**Summary**: Aggregated org-wide usage (admin only)

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `x-admin-token` | header | string | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/auth/usage/dashboard
```


## Macro

### GET /macro/bls/catalog

**Summary**: Bls Catalog

List the curated BLS series this service supports.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/macro/bls/catalog
```


### GET /macro/bls/{series_id}

**Summary**: Bls Series Endpoint

Fetch a curated BLS series.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `series_id` | path | string | yes |  |
| `start` | query | integer | no | Start year |
| `end` | query | integer | no | End year |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/macro/bls/<series_id>'
```


### GET /macro/calendar/export.ics

**Summary**: iCalendar export of the macro calendar (Google Calendar friendly).

Return all matching events as a ``text/calendar`` ICS file.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `days` | query | integer | no |  |
| `kind` | query | string | no |  |
| `importance_min` | query | integer | no |  |
| `region` | query | string | no |  |

**Response 200** (`text/plain`):

```json
"string"
```

**Example**:

```bash
curl http://localhost:8000/macro/calendar/export.ics
```


### GET /macro/fred/catalog

**Summary**: Fred Catalog

List every FRED series this service supports.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "count": 0,
  "series": [
    {
      "series_id": "string",
      "name": "string",
      "frequency": "string",
      "units": "string",
      "last_updated": null,
      "citation": "string"
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/macro/fred/catalog
```


### GET /macro/fred/series/{series_id}

**Summary**: Fred Series Endpoint

Fetch a single FRED series in JSON form.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `series_id` | path | string | yes |  |
| `start` | query | string | yes | ISO date YYYY-MM-DD |
| `end` | query | string | yes | ISO date YYYY-MM-DD |
| `transform` | query | string | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/macro/fred/series/<series_id>?start=string&end=string'
```


### GET /macro/overlay

**Summary**: Macro Overlay

Fetch multiple FRED series aligned on a daily UTC calendar.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `series` | query | string | yes | Comma-separated FRED series ids, e.g. 'DFF,DGS10,VIXCLS' |
| `start` | query | string | yes | ISO date YYYY-MM-DD |
| `end` | query | string | yes | ISO date YYYY-MM-DD |

**Response 200** (`application/json`):

```json
{
  "start": "string",
  "end": "string",
  "count": 0,
  "series": [
    {
      "id": "string",
      "name": "string",
      "units": "string",
      "frequency": "string",
      "dates": [
        null
      ],
      "values": [
        null
      ]
    }
  ]
}
```

**Example**:

```bash
curl 'http://localhost:8000/macro/overlay?series=string&start=string&end=string'
```


### GET /macro/upcoming

**Summary**: Macro Upcoming

Return the upcoming macro events within ``days`` days.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `days` | query | integer | no | Lookahead window in days |
| `kind` | query | string | no | Keep only this kind (e.g. fomc, cpi). |
| `importance_min` | query | integer | no | Drop events with importance < this (1=low,3=high). |
| `region` | query | string | no | Region filter: US, EU, JP, CN, GLOBAL. |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/macro/upcoming
```


## Arbitrage

### GET /arb/4way-arbs

**Summary**: Get 4Way Arbs

Active 4-venue arb opportunities across the curated concept maps.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `min_spread_pct` | query | number | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/arb/4way-arbs
```


### GET /arb/auto-discover

**Summary**: Get Auto Discover

Auto-discovered cross-venue arb pairs (5-min cache).

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `min_similarity` | query | number | no |  |
| `min_volume` | query | number | no |  |
| `max_pairs` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/arb/auto-discover
```


### GET /arb/concept/{concept_id}

**Summary**: Get 4Way Concept

Return the 4-venue concept map and a snapshot of available legs.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `concept_id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/arb/concept/<concept_id>'
```


### GET /arb/concepts

**Summary**: List 4Way Concepts

List every hardcoded 4-venue concept map.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/arb/concepts
```


### GET /arb/confirmed-matches

**Summary**: Get Confirmed Matches

Persistent registry of cross-venue pairs confirmed across N consecutive fetches.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `only_confirmed` | query | boolean | no |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/arb/confirmed-matches
```


### POST /arb/match

**Summary**: Post Match

Manually register a (pm_slug, kalshi_slug) pair.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "pm_slug": "string",
  "kalshi_slug": "string",
  "label": "",
  "theme": ""
}
```

**Response 200** (`application/json`):

```json
{
  "pm_slug": "string",
  "kalshi_slug": "string",
  "label": "",
  "theme": "",
  "source": "string"
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/arb/match -H 'Content-Type: application/json' -d '{"pm_slug": "string", "kalshi_slug": "string", "label": "", "theme": ""}'
```


### GET /arb/matched

**Summary**: Get Matched

List all matched pairs (hardcoded + manually confirmed).

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "n": 0,
  "pairs": [
    {
      "pm_slug": "string",
      "kalshi_slug": "string",
      "label": "",
      "theme": "",
      "source": "string"
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/arb/matched
```


### GET /arb/scanner

**Summary**: Get Scanner

Top cross-venue arbs ranked by ``spread_pct × tradeable_size_usd``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `min_spread_pct` | query | number | no |  |
| `n` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "as_of": "string",
  "n": 0,
  "min_spread_pct": 0.0,
  "arbs": [
    {
      "pm_slug": "string",
      "kalshi_slug": "string",
      "label": "",
      "pm_price": 0.0,
      "kalshi_price": 0.0,
      "spread_pct": 0.0
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/arb/scanner
```


## Reverse Finder

### POST /reverse-finder

**Summary**: Reverse Finder Endpoint

Top-k Polymarket / Kalshi markets that best explain a ticker's returns.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `pool` | query | string | no | Candidate-pool selection mode when ``candidate_factor_ids`` is null. ``curated`` (default) uses a source-prioritised, theme-balanced ≤200-factor pool (~3s end-to-end). ``top_volume`` uses the top-N factors by 24h volume from the homepage cache. ``all`` iterates every factor in factors.yml (~1360, slow). ``theme`` is reserved for future per-theme pools. |

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "start": "date",
  "end": "date",
  "candidate_factor_ids": [
    "string"
  ],
  "k": 5,
  "return_type": "log"
}
```

**Response 200** (`application/json`):

```json
{
  "ticker": "string",
  "top_factors": [
    {
      "factor_id": "string",
      "factor_name": null,
      "delta_r_squared": 0.0,
      "beta": 0.0,
      "t_stat": 0.0,
      "vif": 0.0
    }
  ],
  "total_r_squared": 0.0,
  "n_obs": 0,
  "rejected": [
    "string"
  ],
  "note": "string"
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/reverse-finder -H 'Content-Type: application/json' -d '{"ticker": "string", "start": "date", "end": "date", "candidate_factor_ids": ["string"], "k": 5, "return_type": "log"}'
```


### POST /reverse-finder/stream

**Summary**: Reverse Finder Stream Endpoint

Server-Sent Events variant of ``/reverse-finder``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `pool` | query | string | no | Same as ``/reverse-finder``. ``curated`` (default) is recommended for interactive use. |

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "start": "date",
  "end": "date",
  "candidate_factor_ids": [
    "string"
  ],
  "k": 5,
  "return_type": "log"
}
```

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl -X POST http://localhost:8000/reverse-finder/stream -H 'Content-Type: application/json' -d '{"ticker": "string", "start": "date", "end": "date", "candidate_factor_ids": ["string"], "k": 5, "return_type": "log"}'
```


## Advanced Model

### POST /advanced-model/conditional

**Summary**: Post Conditional

Bucketed conditional model: separate (alpha, beta) per probability regime.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "factor_id": "string",
  "start": "date",
  "end": "date",
  "conditioning_thresholds": [
    0.0
  ],
  "epsilon": 0.01
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/advanced-model/conditional -H 'Content-Type: application/json' -d '{"ticker": "string", "factor_id": "string", "start": "date", "end": "date", "conditioning_thresholds": [0.0], "epsilon": 0.01}'
```


### POST /advanced-model/garch-x

**Summary**: Post Garch X

GARCH(1,1) augmented with |Δlogit| as exogenous variance regressor.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "factor_id": "string",
  "start": "date",
  "end": "date",
  "epsilon": 0.01
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/advanced-model/garch-x -H 'Content-Type: application/json' -d '{"ticker": "string", "factor_id": "string", "start": "date", "end": "date", "epsilon": 0.01}'
```


### POST /advanced-model/polynomial

**Summary**: Post Polynomial

Polynomial-in-Δlogit factor model with HAC SEs and an LR test vs linear.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "factor_id": "string",
  "start": "date",
  "end": "date",
  "degree": 2,
  "epsilon": 0.01
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/advanced-model/polynomial -H 'Content-Type: application/json' -d '{"ticker": "string", "factor_id": "string", "start": "date", "end": "date", "degree": 2, "epsilon": 0.01}'
```


### POST /advanced-model/regime-switching

**Summary**: Post Regime Switching

Hamilton (1989) Markov-switching regression on r_t = alpha_s + beta_s · Δlogit_t.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "factor_id": "string",
  "start": "date",
  "end": "date",
  "n_regimes": 2,
  "epsilon": 0.01
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/advanced-model/regime-switching -H 'Content-Type: application/json' -d '{"ticker": "string", "factor_id": "string", "start": "date", "end": "date", "n_regimes": 2, "epsilon": 0.01}'
```


### POST /advanced-model/tail-dependence

**Summary**: Post Tail Dependence

Empirical lower- and upper-tail dependence between equity returns and Δlogit.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "factor_id": "string",
  "start": "date",
  "end": "date",
  "quantile": 0.05,
  "epsilon": 0.01
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/advanced-model/tail-dependence -H 'Content-Type: application/json' -d '{"ticker": "string", "factor_id": "string", "start": "date", "end": "date", "quantile": 0.05, "epsilon": 0.01}'
```


### POST /advanced-model/vecm

**Summary**: Post Vecm

Johansen + VECM(1) on (log P_equity, logit p).

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "factor_id": "string",
  "start": "date",
  "end": "date",
  "det_order": 0,
  "k_ar_diff": 1
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/advanced-model/vecm -H 'Content-Type: application/json' -d '{"ticker": "string", "factor_id": "string", "start": "date", "end": "date", "det_order": 0, "k_ar_diff": 1}'
```


## Event Model

### POST /event-model/correlation-matrix

**Summary**: Event Model Correlation

Pairwise correlation matrix on Δlogit (or level) probability series.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "factor_ids": [
    "string"
  ],
  "start": "date",
  "end": "date",
  "method": "pearson",
  "on": "delta_logit",
  "epsilon": 0.01
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/event-model/correlation-matrix -H 'Content-Type: application/json' -d '{"factor_ids": ["string"], "start": "date", "end": "date", "method": "pearson", "on": "delta_logit", "epsilon": 0.01}'
```


### POST /event-model/fit

**Summary**: Event Model Fit

Fit Δlogit(target) ~ Σ β · Δlogit(predictors) with HAC SEs.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "target_factor_id": "string",
  "predictor_factor_ids": [
    "string"
  ],
  "start": "date",
  "end": "date",
  "return_type": "delta_logit",
  "epsilon": 0.01
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/event-model/fit -H 'Content-Type: application/json' -d '{"target_factor_id": "string", "predictor_factor_ids": ["string"], "start": "date", "end": "date", "return_type": "delta_logit", "epsilon": 0.01}'
```


### POST /event-model/lead-lag

**Summary**: Event Model Lead Lag

Cross-correlation function and Granger causality between two events.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "target_id": "string",
  "predictor_id": "string",
  "start": "date",
  "end": "date",
  "max_lag": 5,
  "epsilon": 0.01
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/event-model/lead-lag -H 'Content-Type: application/json' -d '{"target_id": "string", "predictor_id": "string", "start": "date", "end": "date", "max_lag": 5, "epsilon": 0.01}'
```


### POST /event-model/pca

**Summary**: Event Model Pca

PCA decomposition of Δlogit innovations across N events.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "factor_ids": [
    "string"
  ],
  "start": "date",
  "end": "date",
  "n_components": 5,
  "epsilon": 0.01
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/event-model/pca -H 'Content-Type: application/json' -d '{"factor_ids": ["string"], "start": "date", "end": "date", "n_components": 5, "epsilon": 0.01}'
```


### POST /event-model/var

**Summary**: Event Model Var

VAR(p) on a Δlogit panel of N events.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "factor_ids": [
    "string"
  ],
  "start": "date",
  "end": "date",
  "lags": 5,
  "epsilon": 0.01
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/event-model/var -H 'Content-Type: application/json' -d '{"factor_ids": ["string"], "start": "date", "end": "date", "lags": 5, "epsilon": 0.01}'
```


## Multi-Event

### POST /multi-event/chains

**Summary**: Find Granger-significant chains start_factor -> ... -> ticker.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "start_factor": "string",
  "end_ticker": "string",
  "candidate_intermediate_factors": [
    "string"
  ],
  "max_depth": 3,
  "start": "string",
  "end": "string"
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/multi-event/chains -H 'Content-Type: application/json' -d '{"start_factor": "string", "end_ticker": "string", "candidate_intermediate_factors": ["string"], "max_depth": 3, "start": "string", "end": "string"}'
```


### POST /multi-event/lasso

**Summary**: Fit LassoCV across N PM-factor Δlogits to predict ticker log returns.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "factor_ids": [
    "string"
  ],
  "start": "string",
  "end": "string",
  "alpha": 0.01
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/multi-event/lasso -H 'Content-Type: application/json' -d '{"ticker": "string", "factor_ids": ["string"], "start": "string", "end": "string", "alpha": 0.01}'
```


### POST /multi-event/macro-correlation

**Summary**: Δlogit(factor) vs Δ(macro) correlation, t-stat, and lead-lag.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "factor_id": "string",
  "macro_series": [
    "string"
  ],
  "start": "string",
  "end": "string"
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/multi-event/macro-correlation -H 'Content-Type: application/json' -d '{"factor_id": "string", "macro_series": ["string"], "start": "string", "end": "string"}'
```


### POST /multi-event/sector-attribution

**Summary**: Per-sector OLS-HAC and variance attribution across PM factors.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "sectors_etfs": [
    "string"
  ],
  "factor_ids": [
    "string"
  ],
  "start": "string",
  "end": "string"
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/multi-event/sector-attribution -H 'Content-Type: application/json' -d '{"sectors_etfs": ["string"], "factor_ids": ["string"], "start": "string", "end": "string"}'
```


### POST /multi-event/systemic-factor

**Summary**: Extract a PM-PCA systemic risk-on/off factor from N PM factors.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "factor_ids": [
    "string"
  ],
  "n_factors": 1,
  "start": "string",
  "end": "string"
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/multi-event/systemic-factor -H 'Content-Type: application/json' -d '{"factor_ids": ["string"], "n_factors": 1, "start": "string", "end": "string"}'
```


## News

### POST /news/causal-chain

**Summary**: Build news -> Δprob -> Δlogit -> ticker-impact chain for a factor.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `authorization` | header | string | no |  |
| `x-api-key` | header | string | no |  |

**Request Body** (`application/json`):

```json
{
  "factor_id": "string",
  "news_items": [
    {}
  ],
  "lookback_hours": 48,
  "beta_map": {}
}
```

**Response 200** (`application/json`):

```json
{
  "factor_id": "string",
  "lookback_hours": 0,
  "n_items": 0,
  "n_tagged": 0,
  "chain": [
    {
      "news_item": null,
      "tagged_factor": null,
      "keyword_overlap": 0,
      "delta_prob": null,
      "delta_logit": null,
      "affected_tickers": [
        null
      ]
    }
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/news/causal-chain -H 'Content-Type: application/json' -d '{"factor_id": "string", "news_items": [{}], "lookback_hours": 48, "beta_map": {}}'
```


### GET /news/entity/{entity}/factors

**Summary**: Top factors associated with a named entity.

Return the top-N factors most associated with ``entity``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `entity` | path | string | yes |  |
| `n` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "entity": "string",
  "n_returned": 0,
  "factors": [
    {
      "factor_id": "string",
      "factor_name": "string",
      "match_score": 0.0
    }
  ]
}
```

**Example**:

```bash
curl 'http://localhost:8000/news/entity/<entity>/factors'
```


### GET /news/factor/{factor_id}/recent

**Summary**: Recently tagged news items for a factor.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `factor_id` | path | string | yes |  |
| `hours` | query | integer | no |  |
| `n` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "factor_id": "string",
  "hours": 0,
  "n_returned": 0,
  "items": [
    {}
  ]
}
```

**Example**:

```bash
curl 'http://localhost:8000/news/factor/<factor_id>/recent'
```


### GET /news/movers

**Summary**: Top news items by |expected stock impact| across registered factors.

Scan every β-registered factor, hydrate news, rank by |impact|.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `hours` | query | integer | no |  |
| `n` | query | integer | no |  |
| `min_impact_pct` | query | number | no |  |

**Response 200** (`application/json`):

```json
{
  "window_hours": 0,
  "n_total": 0,
  "n_returned": 0,
  "min_impact_pct": 0.0,
  "movers": [
    {
      "factor_id": "string",
      "headline": "string",
      "ts": "string",
      "source": "string",
      "expected_impact_pct": 0.0,
      "delta_prob": null
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/news/movers
```


### POST /news/tag

**Summary**: Tag a single news headline -> entities + matched factors + sentiment.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "news_text": "string",
  "factor_ids": [
    "string"
  ],
  "threshold": 0.3
}
```

**Response 200** (`application/json`):

```json
{
  "news_text": "string",
  "entities": {
    "tickers": [
      "string"
    ],
    "politicians": [
      "string"
    ],
    "countries": [
      "string"
    ],
    "events": [
      "string"
    ],
    "commodities": [
      "string"
    ]
  },
  "matched_factors": [
    {
      "factor_id": "string",
      "factor_name": "string",
      "match_score": 0.0
    }
  ],
  "sentiment": {}
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/news/tag -H 'Content-Type: application/json' -d '{"news_text": "string", "factor_ids": ["string"], "threshold": 0.3}'
```


### POST /news/tag-batch

**Summary**: Bulk-tag a list of news items.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "news_items": [
    {
      "title": "string",
      "description": "",
      "ts": "",
      "url": "",
      "source": ""
    }
  ],
  "factor_ids": [
    "string"
  ],
  "threshold": 0.3
}
```

**Response 200** (`application/json`):

```json
{
  "n_items": 0,
  "n_with_matches": 0,
  "results": [
    {
      "news_item": null,
      "entities": null,
      "matched_factors": [
        null
      ],
      "sentiment": {}
    }
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/news/tag-batch -H 'Content-Type: application/json' -d '{"news_items": [{"title": "string", "description": "", "ts": "", "url": "", "source": ""}], "factor_ids": ["string"], "threshold": 0.3}'
```


## Indices

### GET /indices/pm-vix

**Summary**: Get Pm Vix

Current PM-VIX snapshot. Cache 300s on the resolved ``as_of``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `as_of` | query | string | no |  |

**Response 200** (`application/json`):

```json
{
  "as_of": "string",
  "score": 0.0,
  "components": [
    {
      "bucket": "string",
      "avg_prob": 0.0,
      "weight": 0.0,
      "n_used": 0,
      "n_total": 0,
      "sub_score": 0.0
    }
  ],
  "history_30d": [
    0.0
  ],
  "change_24h": 0.0,
  "regime": "RISK_ON"
}
```

**Example**:

```bash
curl http://localhost:8000/indices/pm-vix
```


### GET /indices/pm-vix/components

**Summary**: Get Pm Vix Components

Per-bucket breakdown without the history series.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "as_of": "string",
  "score": 0.0,
  "components": [
    {
      "bucket": "string",
      "avg_prob": 0.0,
      "weight": 0.0,
      "n_used": 0,
      "n_total": 0,
      "sub_score": 0.0
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/indices/pm-vix/components
```


### GET /indices/pm-vix/history

**Summary**: Get Pm Vix History

Synthesised PM-VIX history (deterministic seed).

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `days` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "n": 0,
  "points": [
    {
      "date": "string",
      "score": 0.0
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/indices/pm-vix/history
```


### POST /indices/pm-vix/refresh-slugs

**Summary**: Validate hardcoded bucket slugs against Polymarket and persist replacements.

Trigger one ``validate_and_refresh_buckets`` cycle synchronously.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "as_of": "string",
  "n_kept": 0,
  "n_dead_replaced": 0,
  "buckets": {}
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/indices/pm-vix/refresh-slugs
```


### GET /indices/pm-vix/slugs

**Summary**: Return the current per-bucket slug map (live or fallback).

Surface the slug map currently driving the score.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "as_of": "string",
  "source": "fallback",
  "buckets": {}
}
```

**Example**:

```bash
curl http://localhost:8000/indices/pm-vix/slugs
```


## Quant

### POST /quant/diebold-mariano

**Summary**: Post Diebold Mariano

Compare two forecast-error series via the Diebold-Mariano test.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "forecast_errors_1": [
    0.0
  ],
  "forecast_errors_2": [
    0.0
  ],
  "h": 1,
  "loss": "MSE",
  "hac_lag": 0
}
```

**Response 200** (`application/json`):

```json
{
  "dm_stat": 0.0,
  "p_value": 0.0,
  "dm_stat_hln": 0.0,
  "p_value_hln": 0.0,
  "mean_loss_diff": 0.0,
  "prefer_model": 0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/quant/diebold-mariano -H 'Content-Type: application/json' -d '{"forecast_errors_1": [0.0], "forecast_errors_2": [0.0], "h": 1, "loss": "MSE", "hac_lag": 0}'
```


### POST /quant/multitest/bh

**Summary**: Post Bh Multitest

Apply Benjamini-Hochberg FDR to the supplied p-values.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "p_values": [
    0.0
  ],
  "alpha": 0.05
}
```

**Response 200** (`application/json`):

```json
{
  "rejected_idx": [
    0
  ],
  "q_values": [
    0.0
  ],
  "n_significant": 0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/quant/multitest/bh -H 'Content-Type: application/json' -d '{"p_values": [0.0], "alpha": 0.05}'
```


### POST /quant/oos-r-squared

**Summary**: Post Oos R Squared

Compute the Campbell-Thompson R^2_OOS and Clark-West stat.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "y_actual": [
    0.0
  ],
  "y_pred_model": [
    0.0
  ],
  "y_pred_baseline": [
    0.0
  ],
  "nested": true,
  "hac_lag": 0
}
```

**Response 200** (`application/json`):

```json
{
  "r_squared_oos": 0.0,
  "mse_model": 0.0,
  "mse_baseline": 0.0,
  "n_obs": 0,
  "hac_t_stat_clark_west": 0.0,
  "hac_p_value": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/quant/oos-r-squared -H 'Content-Type: application/json' -d '{"y_actual": [0.0], "y_pred_model": [0.0], "y_pred_baseline": [0.0], "nested": true, "hac_lag": 0}'
```


### POST /quant/quarterly-stability

**Summary**: Post Quarterly Stability

Score a strategy's per-quarter Sharpe record for tier promotion.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "quarterly_sharpes": [
    0.0
  ],
  "threshold": 0.5
}
```

**Response 200** (`application/json`):

```json
{
  "n_quarters": 0,
  "n_positive": 0,
  "sign_flips": 0,
  "passes_4q_gold": true,
  "passes_4q_silver": true,
  "tier_recommendation": "string"
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/quant/quarterly-stability -H 'Content-Type: application/json' -d '{"quarterly_sharpes": [0.0], "threshold": 0.5}'
```


### POST /quant/whites-reality-check

**Summary**: Post Whites Reality Check

Run White's RC + Hansen SPA + (optional) Romano-Wolf stepwise SPA.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "strategy_returns_matrix": [
    [
      0.0
    ]
  ],
  "benchmark_returns": [
    0.0
  ],
  "n_bootstrap": 1000,
  "block_size": 0.0,
  "seed": 42,
  "run_stepwise_spa": true
}
```

**Response 200** (`application/json`):

```json
{
  "n_strategies": 0,
  "n_obs": 0,
  "best_strategy_idx": 0,
  "best_excess_return": 0.0,
  "test_statistic_v_t": 0.0,
  "white_pvalue": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/quant/whites-reality-check -H 'Content-Type: application/json' -d '{"strategy_returns_matrix": [[0.0]], "benchmark_returns": [0.0], "n_bootstrap": 1000, "block_size": 0.0, "seed": 42, "run_stepwise_spa": true}'
```


## Lab

### POST /lab/discover

**Summary**: Kick off an alpha-discovery run (background task)

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `authorization` | header | string | no |  |
| `x-api-key` | header | string | no |  |

**Request Body** (`application/json`):

```json
{
  "n_combos": 20,
  "min_oos_sharpe": 1.0,
  "min_quarters_positive": 3,
  "max_runtime_seconds": 60,
  "seed": 17
}
```

**Response 200** (`application/json`):

```json
{
  "job_id": "string",
  "status": "string",
  "started_at": "string",
  "params": {}
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/lab/discover -H 'Content-Type: application/json' -d '{"n_combos": 20, "min_oos_sharpe": 1.0, "min_quarters_positive": 3, "max_runtime_seconds": 60, "seed": 17}'
```


### POST /lab/promote/{candidate_id}

**Summary**: Mark a candidate for human review (does NOT auto-promote)

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `candidate_id` | path | string | yes |  |
| `job_id` | query | string | no |  |

**Response 200** (`application/json`):

```json
{
  "candidate_id": "string",
  "job_id": "string",
  "promoted_at": "string",
  "review_status": "string",
  "pending_file": "string"
}
```

**Example**:

```bash
curl -X POST 'http://localhost:8000/lab/promote/<candidate_id>'
```


### GET /lab/queue

**Summary**: Get the lab's current runtime state

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "running": true,
  "last_job_id": "string",
  "last_run_at": "string",
  "last_results_summary": {},
  "jobs_file": "string",
  "pending_file": "string"
}
```

**Example**:

```bash
curl http://localhost:8000/lab/queue
```


### GET /lab/results/{job_id}

**Summary**: Fetch results for a specific job

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `job_id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "job_id": "string",
  "status": "string",
  "started_at": "string",
  "completed_at": "string",
  "params": {},
  "n_tested": 0
}
```

**Example**:

```bash
curl 'http://localhost:8000/lab/results/<job_id>'
```


## Signals

### GET /signals/connectivity-check

**Summary**: Probe Polymarket Gamma + CLOB end-to-end with a sample slug.

Verify the real fetcher path can reach Polymarket.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | query | string | no |  |

**Response 200** (`application/json`):

```json
{
  "ok": true,
  "sample_size": 0,
  "error": "string",
  "latency_ms": 0.0,
  "slug": "string"
}
```

**Example**:

```bash
curl http://localhost:8000/signals/connectivity-check
```


### GET /signals/live

**Summary**: Return the current live_signals.json contents (cached 30s).

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/signals/live
```


### POST /signals/recompute-now

**Summary**: Trigger one live-signals recompute synchronously.

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "last_run_iso": "string",
  "last_duration_seconds": 0.0,
  "n_alphas_total": 0,
  "n_alphas_updated": 0,
  "n_alphas_failed": 0,
  "n_alphas_actionable": 0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/signals/recompute-now
```


### GET /signals/status

**Summary**: Last live-signals run status (cron health).

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "last_run_iso": "string",
  "last_duration_seconds": 0.0,
  "n_alphas_total": 0,
  "n_alphas_updated": 0,
  "n_alphas_failed": 0,
  "n_alphas_actionable": 0
}
```

**Example**:

```bash
curl http://localhost:8000/signals/status
```


## Alerts

### GET /alerts

**Summary**: List alert rules for a user

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `user_id` | query | string | yes |  |
| `enabled` | query | boolean | no |  |

**Response 200** (`application/json`):

```json
[
  {}
]
```

**Example**:

```bash
curl 'http://localhost:8000/alerts?user_id=string'
```


### POST /alerts

**Summary**: Create a new alert rule

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "id": "string",
  "user_id": "string",
  "name": "string",
  "cooldown_seconds": 300,
  "channels": [
    null
  ],
  "enabled": true
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/alerts -H 'Content-Type: application/json' -d '{"id": "string", "user_id": "string", "name": "string", "cooldown_seconds": 300, "channels": [null], "enabled": true}'
```


### GET /alerts/events

**Summary**: List alert events

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `user_id` | query | string | yes |  |
| `unack` | query | integer | no |  |
| `limit` | query | integer | no |  |

**Response 200** (`application/json`):

```json
[
  {}
]
```

**Example**:

```bash
curl 'http://localhost:8000/alerts/events?user_id=string'
```


### POST /alerts/events/{event_id}/ack

**Summary**: Acknowledge an event

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `event_id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST 'http://localhost:8000/alerts/events/<event_id>/ack'
```


### DELETE /alerts/{id}

**Summary**: Delete an alert rule

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X DELETE 'http://localhost:8000/alerts/<id>'
```


### GET /alerts/{id}

**Summary**: Get an alert rule by id

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl 'http://localhost:8000/alerts/<id>'
```


### PATCH /alerts/{id}

**Summary**: Partial-update an alert rule

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `id` | path | string | yes |  |

**Request Body** (`application/json`):

```json
{
  "name": "string",
  "cooldown_seconds": 0,
  "channels": [
    {
      "type": null,
      "target": null,
      "enabled": null
    }
  ],
  "enabled": true
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X PATCH 'http://localhost:8000/alerts/<id>' -H 'Content-Type: application/json' -d '{"name": "string", "cooldown_seconds": 0, "channels": [{"type": null, "target": null, "enabled": null}], "enabled": true}'
```


### POST /alerts/{id}/test

**Summary**: Dry-run dispatch to the rule's channels

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST 'http://localhost:8000/alerts/<id>/test'
```


## Embed

### POST /embed/beacon

**Summary**: Embed-impression beacon (best-effort tracking, no PII).

Append one beacon row to a JSONL log. Always returns 204, even on
write failure — we never want a tracking error to surface to the host page.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "slug": "string",
  "pair_id": "string",
  "referrer": "string",
  "utm_source": "string",
  "utm_medium": "string",
  "utm_campaign": "string"
}
```

**Response 204**: Successful Response

**Example**:

```bash
curl -X POST http://localhost:8000/embed/beacon -H 'Content-Type: application/json' -d '{"slug": "string", "pair_id": "string", "referrer": "string", "utm_source": "string", "utm_medium": "string", "utm_campaign": "string"}'
```


### GET /embed/compare

**Summary**: Embeddable overlay of 2+ market price histories (normalised).

Render an overlay sparkline normalising each leg to its first
observation (so the y-axis shows pct change from t0).

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slugs` | query | string | yes | Comma-separated slugs. |
| `theme` | query | string | no |  |

**Response 200** (`text/html`):

```json
"string"
```

**Example**:

```bash
curl 'http://localhost:8000/embed/compare?slugs=string'
```


### GET /embed/market/{slug}

**Summary**: Embeddable mini-card for a Polymarket market.

Render a self-contained HTML card for ``slug``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `theme` | query | string | no |  |
| `height` | query | integer | no |  |
| `autorefresh` | query | boolean | no |  |

**Response 200** (`text/html`):

```json
"string"
```

**Example**:

```bash
curl 'http://localhost:8000/embed/market/<slug>'
```


### GET /embed/og/factor/{factor_id}

**Summary**: Open-Graph PNG (1200x630) for a factor share link.

Return a cached PNG OG image for a factor.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `factor_id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl 'http://localhost:8000/embed/og/factor/<factor_id>'
```


### GET /embed/og/market/{slug}.png

**Summary**: Open-Graph PNG (1200x630) for a market — used in social unfurls.

Return a cached PNG OG image for ``slug``.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl 'http://localhost:8000/embed/og/market/<slug>.png'
```


### GET /embed/og/strategy/{strategy_id}

**Summary**: Open-Graph PNG (1200x630) for a strategy share link.

Return a cached PNG OG image for an alpha strategy.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `strategy_id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl 'http://localhost:8000/embed/og/strategy/<strategy_id>'
```


### GET /embed/strategy/{pair_id}

**Summary**: Embeddable card for a validated alpha strategy.

Render an alpha-strategy card. Falls back to a placeholder if the pair
isn't in the curated catalog yet.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `pair_id` | path | string | yes |  |
| `theme` | query | string | no |  |

**Response 200** (`text/html`):

```json
"string"
```

**Example**:

```bash
curl 'http://localhost:8000/embed/strategy/<pair_id>'
```


## Replay

### POST /replay/order

**Summary**: Simulate a paper-trade order against historical prices

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "slug": "string",
  "side": "LONG",
  "size_usd": 0.0,
  "at_timestamp": "date-time",
  "hold_until": "date-time",
  "slippage_bps": 100.0
}
```

**Response 200** (`application/json`):

```json
{
  "status": "string",
  "slug": "string",
  "side": "string",
  "size_usd": 0.0,
  "entry_price": 0.0,
  "exit_price": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/replay/order -H 'Content-Type: application/json' -d '{"slug": "string", "side": "LONG", "size_usd": 0.0, "at_timestamp": "date-time", "hold_until": "date-time", "slippage_bps": 100.0}'
```


### GET /replay/scenario/{scenario_name}

**Summary**: Hydrate a pre-baked scenario

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `scenario_name` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "as_of": "string",
  "markets": [
    {
      "slug": "string",
      "name": "string",
      "prob": 0.0,
      "vol": 0.0,
      "theme": "string",
      "last_change": 0.0
    }
  ],
  "equities": [
    {
      "ticker": "string",
      "price": 0.0,
      "change_24h": 0.0,
      "as_of_obs": null
    }
  ],
  "headline_news": [
    {}
  ],
  "scenario": {},
  "cache_age_seconds": 0
}
```

**Example**:

```bash
curl 'http://localhost:8000/replay/scenario/<scenario_name>'
```


### GET /replay/scenario/{scenario_name}/pnl

**Summary**: Realized historical basket PnL for the scenario window

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `scenario_name` | path | string | yes |  |
| `capital` | query | number | no |  |

**Response 200** (`application/json`):

```json
{
  "scenario_id": "string",
  "capital_usd": 0.0,
  "ticker_returns": {},
  "basket_pnl_long_only": 0.0,
  "basket_pnl_equal_weighted": 0.0,
  "as_of_iso": "string"
}
```

**Example**:

```bash
curl 'http://localhost:8000/replay/scenario/<scenario_name>/pnl'
```


### GET /replay/scenario/{scenario_name}/preflight

**Summary**: Verify each scenario slug is still resolvable on Polymarket

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `scenario_name` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "scenario_id": "string",
  "slugs_status": [
    {
      "slug": "string",
      "status": "live"
    }
  ],
  "can_replay": true,
  "substitutes": {}
}
```

**Example**:

```bash
curl 'http://localhost:8000/replay/scenario/<scenario_name>/preflight'
```


### GET /replay/scenarios

**Summary**: List pre-baked replay scenarios

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "n_scenarios": 0,
  "scenarios": [
    {
      "id": "string",
      "name": "string",
      "title": "string",
      "timestamp": "string",
      "as_of_iso": "string",
      "end_iso": null
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/replay/scenarios
```


### GET /replay/sessions

**Summary**: Alias of /replay/scenarios (footer pill).

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "n_scenarios": 0,
  "scenarios": [
    {
      "id": "string",
      "name": "string",
      "title": "string",
      "timestamp": "string",
      "as_of_iso": "string",
      "end_iso": null
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/replay/sessions
```


### GET /replay/state

**Summary**: Snapshot of PM + equity state at a past timestamp

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `as_of` | query | string | yes | UTC timestamp to replay |
| `slugs` | query | string | no | Comma-separated PM slugs |
| `tickers` | query | string | no | Comma-separated yfinance tickers |
| `authorization` | header | string | no |  |
| `x-api-key` | header | string | no |  |

**Response 200** (`application/json`):

```json
{
  "as_of": "string",
  "markets": [
    {
      "slug": "string",
      "name": "string",
      "prob": 0.0,
      "vol": 0.0,
      "theme": "string",
      "last_change": 0.0
    }
  ],
  "equities": [
    {
      "ticker": "string",
      "price": 0.0,
      "change_24h": 0.0,
      "as_of_obs": null
    }
  ],
  "headline_news": [
    {}
  ],
  "scenario": {},
  "cache_age_seconds": 0
}
```

**Example**:

```bash
curl 'http://localhost:8000/replay/state?as_of=date-time'
```


## Fit

### POST /fit

**Summary**: Fit Endpoint

Fit OLS+HAC factor model of stock returns on prediction-market Δlogit factors. The optional ``X-Session-Test-Count`` request header (integer ≥1, default 1) lets the client tell the server how many fits have been run in this session — the server echoes back a Bonferroni-style α/N threshold in the ``multitest_hint`` field and an ``X-Session-Test-Hint`` response header.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `epsilon` | query | number | no |  |
| `prune_collinear` | query | boolean | no | If true, iteratively drop the factor with the highest VIF until every remaining VIF is < 5. Dropped ids are surfaced in the response under `auto_pruned`. Useful when the user throws 8+ correlated factors at the model and wants the server to keep only the identifiable subset. |
| `X-Session-Test-Count` | header | integer | no | Optional client-supplied counter for the number of /fit calls already made in this session. Used to compute the Bonferroni-style α/N threshold returned in ``multitest_hint``. Defaults to 1 when absent. |

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "factors": [
    "string"
  ],
  "custom_factors": [
    {
      "id": "string",
      "slug": "string",
      "name": null
    }
  ],
  "start": "date",
  "end": "date",
  "return_type": "log"
}
```

**Response 200** (`application/json`):

```json
{
  "ticker": "string",
  "n_obs": 0,
  "start": "date",
  "end": "date",
  "epsilon": 0.0,
  "return_type": "log"
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/fit -H 'Content-Type: application/json' -d '{"ticker": "string", "factors": ["string"], "custom_factors": [{"id": "string", "slug": "string", "name": null}], "start": "date", "end": "date", "return_type": "log"}'
```


### POST /fit/preview

**Summary**: Fit Preview Endpoint

Fast pre-flight check for /fit.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "factors": [
    "string"
  ],
  "custom_factors": [
    {
      "id": "string",
      "slug": "string",
      "name": null
    }
  ],
  "start": "date",
  "end": "date",
  "return_type": "log"
}
```

**Response 200** (`application/json`):

```json
{
  "ticker": "string",
  "start": "date",
  "end": "date",
  "equity_n_obs": 0,
  "equity_first_date": "date",
  "equity_last_date": "date"
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/fit/preview -H 'Content-Type: application/json' -d '{"ticker": "string", "factors": ["string"], "custom_factors": [{"id": "string", "slug": "string", "name": null}], "start": "date", "end": "date", "return_type": "log"}'
```


## Attribution

### POST /attribution

**Summary**: Attribution Endpoint

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `epsilon` | query | number | no |  |

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "factors": [
    "string"
  ],
  "custom_factors": [
    {
      "id": "string",
      "slug": "string",
      "name": null
    }
  ],
  "start": "date",
  "end": "date",
  "date": "date"
}
```

**Response 200** (`application/json`):

```json
{
  "date": "date",
  "observed_return": 0.0,
  "predicted_return": 0.0,
  "residual": 0.0,
  "contributions": [
    {
      "id": "string",
      "delta_logit": null,
      "beta": null,
      "contribution": 0.0
    }
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/attribution -H 'Content-Type: application/json' -d '{"ticker": "string", "factors": ["string"], "custom_factors": [{"id": "string", "slug": "string", "name": null}], "start": "date", "end": "date", "date": "date"}'
```


## Health

### GET /health

**Summary**: Health

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "status": "ok",
  "version": "string"
}
```

**Example**:

```bash
curl http://localhost:8000/health
```


### GET /health/detail

**Summary**: Health Detail

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/health/detail
```


## Counterfactual

### POST /counterfactual

**Summary**: Post Counterfactual

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "factor_id": "string",
  "scenario": "YES",
  "start": "date",
  "end": "date",
  "actual_resolution": "YES"
}
```

**Response 200** (`application/json`):

```json
{
  "ticker": "string",
  "factor_id": "string",
  "scenario": "YES",
  "actual_resolution": "YES",
  "beta": 0.0,
  "n_obs": 0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/counterfactual -H 'Content-Type: application/json' -d '{"ticker": "string", "factor_id": "string", "scenario": "YES", "start": "date", "end": "date", "actual_resolution": "YES"}'
```


### POST /counterfactual/multi

**Summary**: Post Multi

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "factors_list": [
    "string"
  ],
  "start": "date",
  "end": "date",
  "betas": {}
}
```

**Response 200** (`application/json`):

```json
{
  "ticker": "string",
  "start": "string",
  "end": "string",
  "n_factors": 0,
  "total_return_pct": 0.0,
  "residual_pct": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/counterfactual/multi -H 'Content-Type: application/json' -d '{"ticker": "string", "factors_list": ["string"], "start": "date", "end": "date", "betas": {}}'
```


## Divergence

### GET /divergence/smart-money

**Summary**: Top PM-vs-equity flow divergences across the universe.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `min_strength` | query | number | no |  |

**Response 200** (`application/json`):

```json
{
  "min_strength": 0.0,
  "n_results": 0,
  "results": [
    {
      "slug": "string",
      "ticker_proxy": "string",
      "lookback_hours": 0,
      "whale_flow_pm": 0.0,
      "equity_flow": 0.0,
      "divergence_strength": 0.0
    }
  ]
}
```

**Example**:

```bash
curl http://localhost:8000/divergence/smart-money
```


### GET /divergence/{slug}

**Summary**: Divergence snapshot for a single (slug, default-ticker) pair.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug` | path | string | yes |  |
| `ticker_proxy` | query | string | no |  |
| `lookback_hours` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "slug": "string",
  "ticker_proxy": "string",
  "lookback_hours": 0,
  "whale_flow_pm": 0.0,
  "equity_flow": 0.0,
  "divergence_strength": 0.0
}
```

**Example**:

```bash
curl 'http://localhost:8000/divergence/<slug>'
```


## Export

### POST /export/chart-png

**Summary**: Chart Png

Render a single chart as PNG and return ``image/png`` bytes.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "title": "",
  "x": [
    0.0
  ],
  "y": [
    0.0
  ],
  "kind": "line",
  "width": 1200,
  "height": 600
}
```

**Response 200** (`application/json`):

```json
null
```

**Example**:

```bash
curl -X POST http://localhost:8000/export/chart-png -H 'Content-Type: application/json' -d '{"title": "", "x": [0.0], "y": [0.0], "kind": "line", "width": 1200, "height": 600}'
```


## Hedge

### POST /hedge/auto-config

**Summary**: Solve PM hedge sizes that neutralise a portfolio's factor β.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "portfolio": [
    {
      "ticker": "string",
      "size_usd": 0.0
    }
  ],
  "hedge_factors": [
    "string"
  ],
  "target_beta": 0.0
}
```

**Response 200** (`application/json`):

```json
{
  "target_beta": 0.0,
  "current_betas": {},
  "hedge_positions": [
    {
      "slug": "string",
      "size_usd": 0.0,
      "side": "YES",
      "expected_drift_pct_per_day": 0.0
    }
  ],
  "net_beta_after_hedge": {},
  "gross_hedge_notional_usd": 0.0,
  "slippage_30d_estimate_bps": 0.0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/hedge/auto-config -H 'Content-Type: application/json' -d '{"portfolio": [{"ticker": "string", "size_usd": 0.0}], "hedge_factors": ["string"], "target_beta": 0.0}'
```


### POST /hedge/simulate

**Summary**: Paper-trade a daily-rebalance hedge over N days.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "portfolio": [
    {
      "ticker": "string",
      "size_usd": 0.0
    }
  ],
  "hedge_factors": [
    "string"
  ],
  "target_beta": 0.0,
  "days": 30
}
```

**Response 200** (`application/json`):

```json
{
  "days": 0,
  "final_portfolio_pnl_usd": 0.0,
  "final_hedged_pnl_usd": 0.0,
  "final_slippage_usd": 0.0,
  "vol_reduction_ratio": 0.0,
  "path": [
    {
      "day": 0,
      "portfolio_pnl_usd": 0.0,
      "hedged_pnl_usd": 0.0,
      "cumulative_slippage_usd": 0.0
    }
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/hedge/simulate -H 'Content-Type: application/json' -d '{"portfolio": [{"ticker": "string", "size_usd": 0.0}], "hedge_factors": ["string"], "target_beta": 0.0, "days": 30}'
```


## Multi Venue

### GET /multi-venue/concept/{concept_id}

**Summary**: Get Concept

Unified per-venue view for a curated 4-venue concept map.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `concept_id` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "concept_id": "string",
  "label": "",
  "theme": "",
  "venues": {
    "polymarket": "string",
    "kalshi": "string",
    "manifold": "string",
    "predictit": 0
  },
  "n_legs_present": 0
}
```

**Example**:

```bash
curl 'http://localhost:8000/multi-venue/concept/<concept_id>'
```


### GET /multi-venue/concepts

**Summary**: List Concepts

List every curated 4-venue concept (id + label + theme).

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl http://localhost:8000/multi-venue/concepts
```


### GET /multi-venue/search

**Summary**: Get Search

Free-text search fanned out across all four venues.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `q` | query | string | yes | free-text search term |
| `limit` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "query": "string",
  "n_total": 0,
  "polymarket": [
    {
      "venue": "string",
      "id": "string",
      "slug": "",
      "title": "",
      "end_date": null
    }
  ],
  "kalshi": [
    {
      "venue": "string",
      "id": "string",
      "slug": "",
      "title": "",
      "end_date": null
    }
  ],
  "manifold": [
    {
      "venue": "string",
      "id": "string",
      "slug": "",
      "title": "",
      "end_date": null
    }
  ],
  "predictit": [
    {
      "venue": "string",
      "id": "string",
      "slug": "",
      "title": "",
      "end_date": null
    }
  ]
}
```

**Example**:

```bash
curl 'http://localhost:8000/multi-venue/search?q=string'
```


## Portfolio

### POST /portfolio/pnl-monte-carlo

**Summary**: Monte-Carlo P&L distribution from N bootstrapped Δlogit paths.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "positions": [
    {
      "ticker": "string",
      "size_usd": 0.0,
      "beta_factor": 0.0
    }
  ],
  "factor_id": "string",
  "n_paths": 10000,
  "current_prob": 0.0,
  "beta_map": {},
  "epsilon": 0.01
}
```

**Response 200** (`application/json`):

```json
{
  "factor_id": "string",
  "current_prob": 0.0,
  "n_paths": 0,
  "bootstrap_sigma": 0.0,
  "epsilon": 0.0,
  "n_positions": 0
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/portfolio/pnl-monte-carlo -H 'Content-Type: application/json' -d '{"positions": [{"ticker": "string", "size_usd": 0.0, "beta_factor": 0.0}], "factor_id": "string", "n_paths": 10000, "current_prob": 0.0, "beta_map": {}, "epsilon": 0.01}'
```


### POST /portfolio/resolution-tree

**Summary**: Conditional MTM tree (YES vs NO outcome) for a portfolio on a factor.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "positions": [
    {
      "ticker": "string",
      "size_usd": 0.0,
      "beta_factor": 0.0
    }
  ],
  "factor_id": "string",
  "current_prob": 0.0,
  "beta_map": {},
  "epsilon": 0.01
}
```

**Response 200** (`application/json`):

```json
{
  "factor_id": "string",
  "current_prob": 0.0,
  "epsilon": 0.0,
  "n_positions": 0,
  "gross_notional_usd": 0.0,
  "scenarios": [
    {
      "outcome": "YES",
      "prob": 0.0,
      "delta_logit": 0.0,
      "mtm_total_usd": 0.0,
      "by_ticker": [
        null
      ]
    }
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/portfolio/resolution-tree -H 'Content-Type: application/json' -d '{"positions": [{"ticker": "string", "size_usd": 0.0, "beta_factor": 0.0}], "factor_id": "string", "current_prob": 0.0, "beta_map": {}, "epsilon": 0.01}'
```


## Sources

### GET /sources/delisted

**Summary**: Get Delisted

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "tickers": [
    "string"
  ],
  "count": 0
}
```

**Example**:

```bash
curl http://localhost:8000/sources/delisted
```


### POST /sources/delisted/{ticker}

**Summary**: Post Delisted

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `ticker` | path | string | yes |  |

**Response 200** (`application/json`):

```json
{
  "ticker": "string",
  "marked": true,
  "tickers": [
    "string"
  ]
}
```

**Example**:

```bash
curl -X POST 'http://localhost:8000/sources/delisted/<ticker>'
```


### GET /sources/health

**Summary**: Sources Health

**Parameters**: (none)

**Response 200** (`application/json`):

```json
{
  "sources": {},
  "summary": {}
}
```

**Example**:

```bash
curl http://localhost:8000/sources/health
```


## Strategy Verdict

### POST /strategy-verdict/cointegration

**Summary**: Post Cointegration Verdict

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "adf_p": 0.0,
  "half_life_days": 0.0,
  "rho_ar1": 0.0,
  "n_obs": 0,
  "beta_hedge": 0.0,
  "current_z": 0.0
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategy-verdict/cointegration -H 'Content-Type: application/json' -d '{"adf_p": 0.0, "half_life_days": 0.0, "rho_ar1": 0.0, "n_obs": 0, "beta_hedge": 0.0, "current_z": 0.0}'
```


### POST /strategy-verdict/pairs

**Summary**: Post Pairs Verdict

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "current_z": 0.0,
  "entry_z": 2.0,
  "exit_z": 0.5,
  "stop_z": 4.0,
  "in_position": false,
  "cointegration_passed": true
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/strategy-verdict/pairs -H 'Content-Type: application/json' -d '{"current_z": 0.0, "entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "in_position": false, "cointegration_passed": true}'
```


## Vol

### POST /vol/egarch

**Summary**: Post Egarch

Fit EGARCH(1,1) to the ticker's daily log-returns.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "start": "string",
  "end": "string"
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/vol/egarch -H 'Content-Type: application/json' -d '{"ticker": "string", "start": "string", "end": "string"}'
```


### POST /vol/garch-compare

**Summary**: Post Garch Compare

Fit multiple GARCH-family models and pick the AIC / BIC winner.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "start": "string",
  "end": "string",
  "models": [
    "garch11"
  ]
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/vol/garch-compare -H 'Content-Type: application/json' -d '{"ticker": "string", "start": "string", "end": "string", "models": ["garch11"]}'
```


### POST /vol/gjr-garch

**Summary**: Post Gjr Garch

Fit GJR-GARCH(1,1) to the ticker's daily log-returns.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "ticker": "string",
  "start": "string",
  "end": "string",
  "distribution": "normal"
}
```

**Response 200** (`application/json`):

```json
{}
```

**Example**:

```bash
curl -X POST http://localhost:8000/vol/gjr-garch -H 'Content-Type: application/json' -d '{"ticker": "string", "start": "string", "end": "string", "distribution": "normal"}'
```


## Vol Surface

### GET /vol-surface/compare

**Summary**: Compare

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `ticker` | query | string | yes |  |
| `pm_pattern` | query | string | yes |  |
| `current_price` | query | number | no |  |
| `options_iv_annual` | query | number | no |  |

**Response 200** (`application/json`):

```json
{
  "ticker": "string",
  "current_price": 0.0,
  "pm_lognormal_sigma": 0.0,
  "options_iv_annual": 0.0,
  "spread_sigma": 0.0,
  "direction": "pm_richer"
}
```

**Example**:

```bash
curl 'http://localhost:8000/vol-surface/compare?ticker=string&pm_pattern=string'
```


### GET /vol-surface/pm/{slug_pattern}

**Summary**: Get Pm Distribution

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `slug_pattern` | path | string | yes |  |
| `market_value` | query | number | no |  |

**Response 200** (`application/json`):

```json
{
  "slug_pattern": "string",
  "strikes": [
    0.0
  ],
  "implied_probs": [
    0.0
  ],
  "fitted_mean": 0.0,
  "fitted_std": 0.0,
  "implied_skew": 0.0
}
```

**Example**:

```bash
curl 'http://localhost:8000/vol-surface/pm/<slug_pattern>'
```


## Whales

### POST /whales/mirror

**Summary**: Build a mirror portfolio over a whale's current positions.

**Parameters**: (none)

**Request Body** (`application/json`):

```json
{
  "whale_address": "string",
  "capital_usd": 0.0,
  "max_positions": 10
}
```

**Response 200** (`application/json`):

```json
{
  "whale_address": "string",
  "capital_usd": 0.0,
  "suggested_positions": [
    {
      "slug": "string",
      "side": "YES",
      "size_usd": 0.0,
      "current_price": 0.0,
      "target_price": 0.0,
      "equity_beta": 0.0
    }
  ],
  "total_exposure": 0.0,
  "equivalent_equity_beta_estimate": 0.0,
  "source": "synthetic"
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/whales/mirror -H 'Content-Type: application/json' -d '{"whale_address": "string", "capital_usd": 0.0, "max_positions": 10}'
```


### GET /whales/top

**Summary**: Top whales by absolute 7d PnL.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `window_days` | query | integer | no |  |
| `min_pnl_usd` | query | number | no |  |
| `limit` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "window_days": 0,
  "min_pnl_usd": 0.0,
  "n_whales": 0,
  "whales": [
    {
      "address": "string",
      "pnl_7d_usd": 0.0,
      "positions_value_usd": 0.0,
      "win_rate": 0.0,
      "num_active_positions": 0,
      "last_active_iso": null
    }
  ],
  "source": "synthetic"
}
```

**Example**:

```bash
curl http://localhost:8000/whales/top
```


### GET /whales/{address}/history

**Summary**: Cumulative-PnL trace for a single whale over N days.

**Parameters**:

| Name | In | Type | Required | Description |
| --- | --- | --- | --- | --- |
| `address` | path | string | yes |  |
| `days` | query | integer | no |  |

**Response 200** (`application/json`):

```json
{
  "whale_address": "string",
  "days": 0,
  "trace": [
    {
      "date_iso": "string",
      "cumulative_pnl_usd": 0.0,
      "positions_value_usd": 0.0
    }
  ],
  "source": "synthetic"
}
```

**Example**:

```bash
curl 'http://localhost:8000/whales/<address>/history'
```
