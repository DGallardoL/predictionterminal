# Regression Cookbook — Recipes for `/fit`

> Ten battle-tested patterns for fitting factor models against prediction-market and
> macro factors using the Prediction Terminal API. Every recipe contains a copy-pasteable
> `curl` call, a plain-English interpretation of what the result means, and the single
> most common pitfall we have seen in practice. Treat this document as a working
> reference — when in doubt, replicate a recipe verbatim, then mutate one parameter at a
> time.

The API base URL throughout is `http://localhost:8000`. All endpoints accept and return
JSON. Dates are ISO-8601 (`YYYY-MM-DD`). Returns are computed as log differences of
adjusted close prices (`r_t = log(P_t / P_{t-1})`); prediction-market factors are
clipped to `[ε, 1-ε]` with default `ε=0.01` and converted to logit space before
differencing. Cross-check the math in `docs/quants.md` if any coefficient looks
implausible.

---

## Recipe 1 — Single factor: "How does NVDA respond to Bitcoin price?"

The simplest possible call: one ticker, one factor, default HAC standard errors. Use
this as a smoke test whenever you suspect the API is misbehaving — if this returns
sensible numbers, downstream complexity is the problem, not your environment.

```bash
curl -s -X POST http://localhost:8000/fit \
  -H 'Content-Type: application/json' \
  -d '{
    "ticker": "NVDA",
    "factors": ["macro:bitcoin"],
    "start": "2024-01-01",
    "end": "2026-05-01",
    "frequency": "daily",
    "cov_type": "HAC",
    "hac_maxlags": 5
  }' | jq '.'
```

**Interpretation.** Look at `betas["macro:bitcoin"].coef` first. A value near `0.15`
means a 1-log-point move in BTC is associated with a 0.15-log-point co-move in NVDA on
the same day; the `t_stat` should clear 2 in absolute value before you read any further.
`r_squared` for a one-factor model on daily data above 0.10 is genuinely informative;
anything above 0.30 is suspicious and probably points at look-ahead bias, a mis-aligned
calendar, or BTC being a near-proxy for something deeper (semis risk-on/off). Compare
`n_obs` to your expected trading-day count — a 30% gap usually means weekend BTC
observations are not being aligned with US close.

**Pitfall.** BTC trades 24/7; NVDA does not. The API normalizes both to UTC close, but
if you accidentally pass a sub-daily frequency you will get a spurious negative β
because Friday's late-evening BTC tick is being regressed against Monday's NVDA open.
Always confirm `frequency=daily` in the echoed request payload.

---

## Recipe 2 — Multi-factor: combining macro and sector factors

Most real research uses 3–8 factors. The trade-off is variance reduction vs.
multicollinearity. The API reports a `vif` block per factor so you can spot collinear
pairs before they corrupt your standard errors.

```bash
curl -s -X POST http://localhost:8000/fit \
  -H 'Content-Type: application/json' \
  -d '{
    "ticker": "NVDA",
    "factors": [
      "macro:spy",
      "macro:qqq",
      "macro:vix",
      "macro:dxy",
      "macro:10y-yield",
      "sector:semis"
    ],
    "start": "2024-01-01",
    "end": "2026-05-01",
    "cov_type": "HAC",
    "hac_maxlags": 5
  }' | jq '.betas, .vif, .r_squared'
```

**Interpretation.** Read VIFs first. Any factor with `vif > 10` is a red flag: drop the
weaker-theory factor (usually QQQ when SPY is present, since QQQ is ~60% tech).
Inspect `betas["sector:semis"].coef`; if it's near 1.0 with t-stat above 10 you've
essentially regressed NVDA on itself — pull semis out and re-fit. Compare adjusted R² to
the single-factor benchmark; if the jump is less than 0.05 the marginal factors are not
earning their degrees of freedom.

**Pitfall.** Adding factors mechanically inflates R² even if every new factor is noise.
Always inspect adjusted R² (the API returns it as `r_squared_adj`) and only declare a
factor "useful" if its inclusion improves it by ≥0.01 and its individual t-stat is
robust.

---

## Recipe 3 — Sentiment factor: `sentiment:bitcoin`

The `sentiment:` source is a hybrid VADER + financial-lexicon scorer (see
`pfm/terminal/sentiment_nlp.py`). It produces a daily-aggregated score in `[-1, +1]`
which is then z-scored within the request window. Use it when you suspect retail
narrative — not fundamentals — is driving moves.

```bash
curl -s -X POST http://localhost:8000/fit \
  -H 'Content-Type: application/json' \
  -d '{
    "ticker": "COIN",
    "factors": [
      "macro:bitcoin",
      "sentiment:bitcoin",
      "sentiment:crypto-regulation"
    ],
    "start": "2024-06-01",
    "end": "2026-05-01",
    "cov_type": "HAC",
    "hac_maxlags": 7
  }' | jq '.betas, .factor_meta'
```

**Interpretation.** `betas["sentiment:bitcoin"].coef` represents the marginal effect of a
one-standard-deviation jump in narrative sentiment on COIN's daily log-return,
controlling for the actual BTC tape. If the coefficient is meaningfully positive
(say >0.005) with a t-stat above 2 _after_ macro:bitcoin is already in the model, you
have evidence the narrative carries information beyond price. Check `factor_meta` for
the headline-count per day; days with fewer than 5 headlines have noisy scores and may
be driving the result.

**Pitfall.** Sentiment leaks future information if you accidentally use a query that
includes outcome language ("crash", "rally" used after the fact in retrospective
journalism). Stick to curated slugs (`sentiment:fed-hawkish`, etc.) for production
work; free-form queries are for exploration.

---

## Recipe 4 — High-dimension: 100 factors with elastic net

When you have more factors than you can defend individually, switch the estimator from
OLS to elastic net. The API exposes this through `cov_type="elasticnet"` with an
`alpha` (overall regularization strength) and `l1_ratio` (lasso vs. ridge mix).

```bash
curl -s -X POST http://localhost:8000/fit \
  -H 'Content-Type: application/json' \
  -d '{
    "ticker": "SPY",
    "factors": ["category:all-macro", "category:all-sentiment"],
    "start": "2024-01-01",
    "end": "2026-05-01",
    "estimator": "elasticnet",
    "alpha": 0.01,
    "l1_ratio": 0.5,
    "cv_folds": 5
  }' | jq '.betas | to_entries | sort_by(-(.value.coef|fabs)) | .[0:10]'
```

**Interpretation.** Inspect the top-ten non-zero coefficients ranked by absolute size.
These are the factors the elastic net believes carry signal after cross-validated
shrinkage. The set should be _stable_ across reasonable α values; if it churns
violently between α=0.005 and α=0.02 your data does not actually support a sparse
model. The API reports `cv_score` (out-of-fold R²) — if it is negative, no model is
better than the mean and you should not deploy.

**Pitfall.** Elastic net standardizes features internally; absolute coefficients are
on the standardized scale, not the raw scale. If you compare them to OLS coefficients
from Recipe 2 they will appear an order of magnitude smaller. Use the
`coef_unstandardized` field for apples-to-apples comparisons.

---

## Recipe 5 — Tail risk: quantile regression for τ=0.05 and τ=0.95

OLS estimates conditional means. When you care about how a factor moves the tails of
the return distribution — the downside fat tail at τ=0.05 or the upside at τ=0.95 —
switch to quantile regression. This is essential for tail-hedging strategies.

```bash
curl -s -X POST http://localhost:8000/fit \
  -H 'Content-Type: application/json' \
  -d '{
    "ticker": "NVDA",
    "factors": ["macro:vix", "macro:spy"],
    "start": "2024-01-01",
    "end": "2026-05-01",
    "estimator": "quantile",
    "quantiles": [0.05, 0.5, 0.95]
  }' | jq '.quantile_betas'
```

**Interpretation.** Compare β on VIX across the three quantiles. If `β_0.05` is
sharply negative (e.g. -0.08) while `β_0.95` is near zero, VIX is informative about
downside risk in NVDA but not upside — exactly the asymmetry you would expect from a
fear gauge. The median (`β_0.50`) is robust to outliers and should roughly match the
OLS coefficient from Recipe 2; large divergence means a few outlier days are dragging
the OLS estimate.

**Pitfall.** Quantile regression standard errors are bootstrap-based and slow to
converge. Use `bootstrap_reps >= 500` for stable inference. If you need confidence
intervals on a tail coefficient, increase reps to 1000+ and expect a 20-second wait.

---

## Recipe 6 — Uncertainty quantification: Bayesian for honest credible intervals

Frequentist confidence intervals require well-behaved sampling distributions and tell
you nothing about the probability the coefficient is positive. Bayesian posteriors
give you exactly that. The API ships a weakly informative normal prior centered on
zero with sd=1 on standardized features.

```bash
curl -s -X POST http://localhost:8000/fit \
  -H 'Content-Type: application/json' \
  -d '{
    "ticker": "NVDA",
    "factors": ["macro:bitcoin", "macro:spy", "sentiment:ai-hype"],
    "start": "2024-01-01",
    "end": "2026-05-01",
    "estimator": "bayesian",
    "prior": "normal",
    "prior_sd": 1.0,
    "draws": 2000
  }' | jq '.posterior_summary'
```

**Interpretation.** `posterior_summary` reports the posterior mean, median, and the
90% credible interval `[q_05, q_95]`. The key field is `prob_positive`: the share of
posterior draws where β > 0. A coefficient with `prob_positive=0.97` is "probably
positive"; one with 0.55 is essentially zero regardless of how the point estimate
reads. Use credible intervals — not p-values — when communicating to non-quants.

**Pitfall.** Bayesian results are sensitive to the prior. If your `prior_sd` is too
tight (e.g. 0.1) you will shrink real signal toward zero; if too loose (e.g. 10) you
recover OLS with extra compute. Always run the same regression with two different
priors and report both if they disagree.

---

## Recipe 7 — Robustness: walk-forward to check stability

A coefficient that is significant on the full sample but flips sign in three of four
quarters is not deployable. The walk-forward harness refits the model on a rolling
window and reports the trajectory of every coefficient.

```bash
curl -s -X POST http://localhost:8000/fit/walkforward \
  -H 'Content-Type: application/json' \
  -d '{
    "ticker": "NVDA",
    "factors": ["macro:bitcoin", "macro:vix", "sector:semis"],
    "start": "2023-01-01",
    "end": "2026-05-01",
    "window_days": 252,
    "step_days": 63
  }' | jq '.coefficient_path["macro:bitcoin"]'
```

**Interpretation.** The returned array is a time series of β estimates with 90% CIs.
Inspect the sign-flip count: if `macro:bitcoin` β switches sign more than once across
the windows, the relationship is regime-driven, not structural. Stable coefficients
look like a noisy horizontal band; unstable ones drift or oscillate. Walk-forward is
the cheapest defense against the Wave-5 failure mode where a stress-tested A_GOLD
claim collapsed in three of four quarters.

**Pitfall.** Choosing the window too short (≤126 days) inflates the variance of each
estimate and makes everything look unstable. Choosing it too long (≥500) hides regime
changes inside the window. 252 trading days (~1 year) is the safe default.

---

## Recipe 8 — Cointegration check: pairs trading hypothesis

Before running a pairs trade you must verify the two series are cointegrated, not
merely correlated. The `/fit/cointegration` endpoint runs an Engle-Granger test plus a
Johansen check with two lags.

```bash
curl -s -X POST http://localhost:8000/fit/cointegration \
  -H 'Content-Type: application/json' \
  -d '{
    "tickers": ["KO", "PEP"],
    "start": "2020-01-01",
    "end": "2026-05-01"
  }' | jq '.engle_granger, .johansen, .half_life_days'
```

**Interpretation.** Reject the null of no cointegration when the Engle-Granger ADF
p-value is below 0.05 _and_ Johansen's trace statistic exceeds the 5% critical value.
`half_life_days` quantifies how quickly the spread mean-reverts; a half-life of 7–30
days is workable for daily trading, while >90 days means the spread drifts faster than
it reverts and the pair is not tradable even if statistically cointegrated.

**Pitfall.** Cointegration is fragile to structural breaks. Always re-run on at least
two sub-samples (e.g. pre-2022 and post-2022). If one passes and the other does not,
the cointegration is a Phase artifact, not a deployable signal.

---

## Recipe 9 — Event study: how did NVDA react to FOMC meetings?

Event studies estimate abnormal returns in a window around scheduled events. Pass the
event slug and the API will fetch the dates, align to trading days, and compute
cumulative abnormal returns (CAR) using a 120-day pre-event estimation window.

```bash
curl -s -X POST http://localhost:8000/fit/event-study \
  -H 'Content-Type: application/json' \
  -d '{
    "ticker": "NVDA",
    "event": "fomc-meetings",
    "event_window": [-3, 5],
    "estimation_window": 120,
    "start": "2022-01-01",
    "end": "2026-05-01"
  }' | jq '.car_series, .cross_event_stats'
```

**Interpretation.** Examine `car_series` for the average cumulative abnormal return at
each offset day. A statistically meaningful FOMC reaction shows up as a step at t=0
(the announcement) with a CAR magnitude that clears the 95% CI bound reported in
`cross_event_stats.car_ci`. If the CAR drifts but does not jump, the market is
trading the macro consensus into the meeting and the announcement itself carries no
incremental information.

**Pitfall.** Event windows that span earnings or other firm-specific news produce
contaminated CARs. Use `exclude_overlapping_events=true` to drop events that fall
within ±3 days of any earnings date for the same ticker.

---

## Recipe 10 — Sharpe with deflation: honest performance claims

A backtested Sharpe of 2.0 across 100 trial strategies is essentially noise. The
deflated Sharpe ratio (Bailey & López de Prado, 2014) corrects for selection bias by
penalizing the realized Sharpe by the number of trials you ran.

```bash
curl -s -X POST http://localhost:8000/fit/sharpe-deflated \
  -H 'Content-Type: application/json' \
  -d '{
    "returns_endpoint": "/fit",
    "ticker": "NVDA",
    "factors": ["macro:bitcoin", "macro:vix"],
    "start": "2024-01-01",
    "end": "2026-05-01",
    "n_trials": 50,
    "skewness": -0.4,
    "kurtosis": 4.2
  }' | jq '.observed_sharpe, .deflated_sharpe, .probability_skill'
```

**Interpretation.** `observed_sharpe` is the raw realized Sharpe; `deflated_sharpe` is
the corrected version after penalizing for `n_trials=50` independent variants and the
strategy's higher-moment shape. `probability_skill` is the posterior probability that
the true Sharpe exceeds zero. Only claim a strategy is deployable if
`probability_skill ≥ 0.95` _and_ `deflated_sharpe ≥ 0.5`.

**Pitfall.** Lying about `n_trials` is the single most common abuse. The number you
pass must include every variant you tried, not just the ones you remembered to log.
If you tested seven factor combinations across three estimators and four windows, that
is 84 trials, not 1. Be honest with yourself before the market is honest with you.

---

## Cross-recipe defaults

- **Frequency.** Always `daily` unless you have a specific reason. Higher-frequency
  data introduces microstructure noise that swamps macro factor signal.
- **HAC lags.** Use `5` for daily and `2` for weekly returns. Set higher only if the
  Newey-West auto-bandwidth in `factor_meta.recommended_hac_lags` suggests it.
- **Clipping ε.** Default 0.01 is fine for liquid markets. For thin contracts that
  trade below 0.02 set `epsilon=0.005` and document the choice — sensitivity to ε
  should be reported alongside coefficients.
- **Calendars.** UTC everywhere. If you observe a fit that mysteriously has half the
  expected `n_obs`, you almost certainly have a timezone mismatch and the join
  silently dropped the unmatched rows.

When in doubt, refit Recipe 1 first, then layer complexity. Most "the model is
broken" reports turn out to be timezone bugs, missing data on weekends, or a single
hardcoded slug that got resolved. The recipes above are designed to fail loudly, not
quietly — trust the t-stats, trust the VIF, trust the walk-forward.
