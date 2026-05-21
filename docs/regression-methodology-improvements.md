# Regression Methodology Improvements for `/fit`

*Status: research-only · Author: research desk · Date: 2026-05-16 · Task: T79*

## 0. Why this document exists

The user asked **"¿qué tipo de regresión sí podría servir más igual y ser más informativa?"** — i.e. which regression families would tell us more, more honestly, about the relationship between prediction-market factors and stock returns than the current default of OLS with Newey–West (HAC) standard errors.

The current `/fit` endpoint runs `statsmodels.OLS(...).fit(cov_type='HAC', cov_kwds={'maxlags': lag})` with optional Andrews bandwidth, log returns on the target, Δlogit on probability factors, ε-clipping, VIF reporting, and a bootstrap option. That is correct, defensible, and the right *default*. But it is not the right *only* tool when:

- The catalog has **1228 factors** and a typical fit uses 3–6 — collinearity and search-space concerns dominate.
- Time series are **short** (often 30–120 daily observations) — degrees of freedom are scarce.
- Returns are **fat-tailed and event-driven** — OLS minimises the squared loss that those tails dominate.
- Market liquidity is **heterogeneous** across factors — error variance is structurally non-constant.
- Users want **honest uncertainty**, not p-values from a single point estimate that pretends the design matrix was given.

This document surveys 11 alternative or complementary regression families, picks the three worth shipping behind a `method=` query parameter, sketches the math, and writes a roadmap. It is research-only — no production code is changed.

## 1. Survey of methods

For each method: **what it adds**, **when it shines** in our prediction-market factor context, **when it fails**.

### 1.1 Weighted Least Squares (WLS)

**Adds.** A diagonal weight matrix $W = \mathrm{diag}(w_i)$ inside the normal equations: $\hat\beta = (X^\top W X)^{-1} X^\top W y$. With $w_i \propto 1/\sigma_i^2$, WLS is BLUE under heteroskedasticity without needing HAC. HAC is a *sandwich on top of OLS*; WLS is a *change of estimator*. They are complementary: WLS with HAC standard errors is fine.

**Shines for prediction-market factors.** Polymarket liquidity varies by 3–4 orders of magnitude across our catalog (BTC daily volume $≈ $10M; obscure geopolitical binaries $≈ $200). A factor with thin liquidity has noisier Δlogit, so its residual variance is structurally higher. WLS with $w_i = $ (sum of bid + ask depth)$_i$ or $1/\text{spread}_i^2$ shifts the fit toward days when the factor was actually informative.

**Fails when.** (a) Weights are estimated from the same data and feedback-loop into β (use one-step feasible WLS, never iterated); (b) Weights are zero or near-zero for a long stretch (the matrix becomes ill-conditioned); (c) The user wants residual diagnostics — WLS residuals are *weighted* and standard plots are misleading without rescaling.

### 1.2 Quantile Regression (QR)

**Adds.** Estimates the τ-th conditional quantile, not the mean: $\hat\beta(\tau) = \arg\min_\beta \sum_i \rho_\tau(y_i - x_i^\top\beta)$ with $\rho_\tau(u) = u(\tau - \mathbb{1}\{u<0\})$. Fitting τ ∈ {0.1, 0.25, 0.5, 0.75, 0.9} gives a *shape*, not a *point*, of how factors move the distribution.

**Shines for prediction-market factors.** Event-driven days (FOMC, jobs print, geopolitical surprise) are exactly the ones where Polymarket moves matter. On those days returns live in the tails. A factor whose mean β is small but whose 0.9-quantile β is large is a "right-tail event amplifier" — that is a real and useful pattern OLS cannot see.

**Fails when.** (a) n is small (<60) — quantile regression is data-hungry, more so than OLS; (b) The user wants R² — there is no canonical equivalent, only pseudo-R² (Koenker–Machado 1999); (c) Standard errors are usually bootstrapped (Markov-chain or paired), not analytic.

### 1.3 Elastic Net (LASSO + Ridge)

**Adds.** Penalised objective $\hat\beta = \arg\min_\beta \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda\big[\alpha\|\beta\|_1 + \tfrac{1-\alpha}{2}\|\beta\|_2^2\big]$. The L1 piece performs variable selection (sets coefficients exactly to zero); the L2 piece handles correlated regressors (which pure LASSO mis-selects between).

**Shines for prediction-market factors.** With 1228 factors in the catalog and many of them near-duplicates (`fed_cuts_2_2026`, `no_fed_cuts_2026`, `fed_cuts_jun`, `fed_cuts_dec` all proxy the same latent macro state), Elastic Net is the *right* tool to do factor selection inside a single endpoint call. Users could pass `factors=auto` and `top_k=10` and have the system pick from the catalog under a transparent regularisation regime.

**Fails when.** (a) Inference is not free — naive Elastic Net does not yield valid p-values; one needs post-selection inference (Belloni–Chernozhukov 2013) or de-sparsification (van de Geer et al. 2014); (b) Standardisation matters — must z-score factors before fitting, then un-standardise coefficients on report; (c) Cross-validation on small time series is fraught with look-ahead.

### 1.4 Rolling-window OLS

**Adds.** Fits OLS on $\{t-W+1, \ldots, t\}$ for each $t$, yielding $\hat\beta_t$ as a time series. Already partly exposed via `rolling_window` parameter in the existing fit response.

**Shines for prediction-market factors.** Regime drift is the rule, not the exception, in our factor catalog — pre-election vs post-election, calm vs FOMC week, summer vs winter geopolitical news cycle. A factor that flips sign in rolling windows is a regime trade, not a structural alpha — this is precisely the test that demoted favorites-bias from A_GOLD to B_VALIDATED (see `project_wave5_stress_test_findings.md`).

**Fails when.** (a) Window is too short → coefficients are noise; too long → can't detect regime change; (b) Forward-looking centring of the window (a classic bug — must end at $t$, not be centred at $t$); (c) Multiple-testing if every rolling β is treated as a hypothesis.

### 1.5 Markov-switching regression (MS)

**Adds.** Postulates K latent states $s_t \in \{1,\ldots,K\}$ following a Markov chain with transition matrix $P$. Within state $k$ the model is OLS with state-specific $\beta^{(k)}$. Estimated by EM or by gradient methods on the filtered likelihood (Hamilton 1989).

**Shines for prediction-market factors.** In principle this is the *right* model for "calm vs event" days — exactly the regime story behind half our anti-alpha list. State 1 could be "low Polymarket volume, factor doesn't bite", State 2 "event week, factor whips the stock".

**Fails when.** (Almost always for us.) Our time series are short. A 2-state MS on n=120 daily observations has 2K + K(K-1) + 1 parameters (state intercepts, transition probs, error variance) plus per-factor state-specific betas. For 5 factors and 2 states that is ≥ 13 parameters. EM convergence is fragile, identification (state 1 vs state 2) is by convention, and tests for the number of states are non-standard (likelihood is unbounded on the boundary). **This is the kind of model that produces beautiful in-sample plots and zero out-of-sample edge.** See §4 — we don't ship it.

### 1.6 Bayesian linear regression (NUTS or conjugate)

**Adds.** Returns the full posterior $p(\beta \mid y, X)$, not a point. With normal-inverse-gamma priors and Gaussian likelihood the posterior is analytic; with non-conjugate priors a NUTS sampler (PyMC) does the job. Posterior credible intervals are *honest* in the small-n regime where the OLS sampling distribution is itself uncertain.

**Shines for prediction-market factors.** When n=33 (e.g. the XLE/Iran fit in `exotic-regressions-report.md` §2), the frequentist 95 % CI on β is a lie — the t-distribution it relies on assumes the variance estimator is itself well-estimated. A Bayesian fit with weakly informative priors (β ~ N(0, 1), σ² ~ HalfCauchy(1)) returns a wider, more honest credible interval, and the user can read the posterior probability of sign-correct-and-non-trivial directly: $\Pr(\beta > 0.01 \mid y)$.

**Fails when.** (a) Priors matter and we have to defend them (use weakly-informative defaults and let advanced users override); (b) Sampling is slow (NUTS on n=120, p=6 takes ~5 s — acceptable for an async endpoint but not for a 200 ms warm hit); (c) The user just wants a p-value — Bayesian credible intervals are not p-values, and pretending they are is dishonest.

### 1.7 Generalized Least Squares (GLS)

**Adds.** Models the residual covariance directly: $\hat\beta = (X^\top \Omega^{-1} X)^{-1} X^\top \Omega^{-1} y$ with $\Omega$ specified (AR(1), MA, or estimated by Cochrane–Orcutt or Prais–Winsten iteration).

**Shines for prediction-market factors.** When Durbin–Watson is far from 2 and HAC bandwidth is large (>5 lags), HAC is correcting for autocorrelation in the *standard errors* but not in the *point estimate*. GLS with an explicit AR(1) on the residuals yields a more efficient β when the AR(1) is correct.

**Fails when.** (a) The error process is mis-specified (real data is rarely pure AR(1)); (b) The user wants robustness to *unknown* serial correlation — that is exactly what HAC is for, so GLS is a *bet*, not a hedge; (c) Small sample → iterative estimation of ρ is unstable.

### 1.8 Robust regression (Huber, M-estimator)

**Adds.** Replaces the squared loss with a piecewise function that down-weights outliers: $\hat\beta = \arg\min_\beta \sum_i \rho(r_i)$ with Huber's $\rho(u) = u^2/2$ for $|u|\le k$, $k|u| - k^2/2$ for $|u|>k$. Iteratively reweighted least squares (IRLS) computes this.

**Shines for prediction-market factors.** A single 8 % daily move on FOMC day can swing an OLS β by 30 %. Huber regression with k = 1.345·σ̂ down-weights such observations. This is *not* the same as dropping them: they still inform β but with reduced leverage.

**Fails when.** (a) The outlier *is* the signal (FOMC week is the most informative week!) — Huber would discard the most useful observation; (b) Choice of tuning constant k is somewhat arbitrary; (c) Standard errors require sandwich estimator (statsmodels does this, but the formula differs from HAC).

### 1.9 Distributed-lag / VAR

**Adds.** Models $y_t = \alpha + \sum_{k=0}^{K} \gamma_k x_{t-k} + \epsilon_t$ (distributed lag) or jointly $\mathbf{y}_t = A_1 \mathbf{y}_{t-1} + \ldots + A_p \mathbf{y}_{t-p} + \epsilon_t$ (VAR). Granger-causality tests fall out naturally.

**Shines for prediction-market factors.** The lead-lag question — *does Polymarket move first or does the stock?* — is one of the most economically interesting questions our data can answer. A VAR(1) on (Δlogit_factor, log-return_stock) with a Granger-causality test gives a clean answer. If Polymarket leads, that is alpha. If the stock leads, the factor is descriptive, not predictive.

**Fails when.** (a) Lag length is hard to choose (AIC/BIC give different answers); (b) VAR coefficients are hard to interpret without impulse-response functions; (c) Stationarity is required — Polymarket prices are not stationary, Δlogit is.

### 1.10 Error-correction / Engle–Granger cointegration

**Adds.** When two series share a stochastic trend (cointegrated), regressing levels yields a valid relationship even though each is non-stationary. The ECM form is $\Delta y_t = \alpha(y_{t-1} - \beta x_{t-1}) + \gamma \Delta x_t + \epsilon_t$ where $(y_{t-1} - \beta x_{t-1})$ is the equilibrium error.

**Shines for prediction-market factors.** Some pairs really do co-move at the *price level*, not just at returns. E.g. CME Bitcoin futures front-month and Polymarket "BTC > $100k by EOY" odds share a latent BTC-spot trend. Engle–Granger or Johansen captures that. The current `/fit` works on returns by design, which throws away this information.

**Fails when.** (a) The cointegration relationship breaks (regime change in the spread); (b) Most stock/Polymarket pairs are *not* cointegrated; (c) Adding ECM to the catalog is a fundamental rethink of the endpoint, not a method= flag.

### 1.11 Bayesian model averaging (BMA)

**Adds.** Instead of picking one model, average over the space of $2^p$ possible factor subsets, weighting each by its posterior probability $\Pr(M_k \mid y)$. The posterior on β is $\sum_k \hat\beta_k \cdot \Pr(M_k \mid y)$.

**Shines for prediction-market factors.** When the user honestly doesn't know which of 10 candidate factors matter, BMA gives the posterior *inclusion probability* for each — a direct, calibrated answer to "which factors matter?" that is much more honest than running OLS on a hand-picked subset and reporting p-values.

**Fails when.** (a) The model space is huge (2^10 = 1024 is fine; 2^1228 is not — needs MCMC sampling over the model space); (b) Priors over models matter; (c) Hard to explain in a UI — "inclusion probability 0.34" reads ambiguously to a non-quant user.

## 2. Top three picks worth shipping

Picks are constrained by (a) marginal information gain vs OLS+HAC, (b) implementation cost, (c) shipping in a single `method=` query parameter without breaking the existing response schema. Picks: **Elastic Net, Quantile Regression, Bayesian linear**.

### 2.1 Elastic Net — the auto-selector

**Math.**
$$\hat\beta^{EN} = \arg\min_\beta \frac{1}{2n}\sum_{i=1}^{n}(y_i - x_i^\top \beta)^2 + \lambda\left[\alpha \sum_{j=1}^{p}|\beta_j| + \frac{1-\alpha}{2}\sum_{j=1}^{p}\beta_j^2\right]$$

with $\lambda \ge 0$ the overall regularisation and $\alpha \in [0,1]$ mixing L1 (LASSO) and L2 (Ridge). Standardise factors to unit variance before fit; report coefficients on the original scale.

**New diagnostics produced.**
- Per-factor `selected: bool` (1 if $\hat\beta_j \ne 0$).
- `regularisation_path` — λ values along the cross-validated regularisation path with corresponding R² (CV).
- `optimal_lambda`, `optimal_alpha` (if user passes `alpha=auto`, use 5-fold CV with `TimeSeriesSplit`).
- Post-selection inference is *off by default*; if `inference=desparsified` is passed, run the de-sparsified Lasso (Javanmard–Montanari 2014) for valid CIs.

**Existing diagnostics replaced.** `t_stats` is replaced by `coefficient_path` plot; VIF becomes irrelevant (Ridge penalty absorbs collinearity); `r_squared` is reported as `r_squared_cv` (5-fold CV) rather than in-sample.

**API sketch.**

```
POST /fit?method=enet&alpha=0.5&lambda=auto
```

Response shape: existing fields preserved where defined; new fields `selected_factors: list[str]`, `regularisation_path: list[{lambda, r2_cv, n_selected}]`, `optimal_lambda: float`.

### 2.2 Quantile Regression — the tail telescope

**Math.**
$$\hat\beta(\tau) = \arg\min_\beta \sum_{i=1}^{n} \rho_\tau\bigl(y_i - x_i^\top\beta\bigr),\quad \rho_\tau(u) = u\bigl(\tau - \mathbb{1}\{u<0\}\bigr)$$

solved by interior-point or simplex method (statsmodels `QuantReg`). Fit at τ ∈ {0.1, 0.25, 0.5, 0.75, 0.9} by default; user can override.

**New diagnostics produced.**
- A `coefficients_by_quantile` table: one row per factor, one column per τ. Reveals tail asymmetry (a factor whose β(0.5) is small but β(0.9) is large is a right-tail amplifier).
- `pseudo_r2_by_quantile` (Koenker–Machado 1999).
- Bootstrap CIs on β(τ) — 500 paired-bootstrap replicates is the standard default.
- A `tail_asymmetry: float` summary = β(0.9) − β(0.1), signed by user direction.

**Existing diagnostics replaced.** None — quantile regression is *additive* to OLS, not a replacement. The endpoint should still return the OLS fit as the headline and append the quantile shape as a supplement.

**API sketch.**

```
POST /fit?method=quantile&taus=0.1,0.25,0.5,0.75,0.9
```

### 2.3 Bayesian linear regression — the honest CI

**Math.** Conjugate normal-inverse-gamma prior:

$$\beta \mid \sigma^2 \sim \mathcal{N}(\mu_0,\, \sigma^2 \Lambda_0^{-1}),\quad \sigma^2 \sim \mathrm{InvGamma}(a_0, b_0)$$

Posterior is analytic:

$$\beta \mid y, X, \sigma^2 \sim \mathcal{N}(\mu_n,\, \sigma^2 \Lambda_n^{-1}),\quad \Lambda_n = X^\top X + \Lambda_0,\quad \mu_n = \Lambda_n^{-1}(\Lambda_0\mu_0 + X^\top y)$$

Default prior: $\mu_0 = 0$, $\Lambda_0 = \kappa \cdot I$ with $\kappa = 0.01$ (weakly informative on β), $a_0 = b_0 = 1$ (broad on σ²). For non-conjugate priors (e.g. horseshoe for sparsity) fall back to NUTS via PyMC.

**New diagnostics produced.**
- Per-factor `posterior_mean`, `posterior_sd`, `credible_interval_95: [lo, hi]`.
- `prob_sign_correct: float` — $\Pr(\beta > 0 \mid y)$ for factors with prior-positive sign expectations.
- `prob_practical: float` — $\Pr(|\beta| > \beta_\mathrm{threshold} \mid y)$ where $\beta_\mathrm{threshold}$ is a user-supplied "practical significance" cutoff (default 0.005).
- Posterior predictive check: a single sample of $\tilde y$ from $p(\tilde y \mid y)$ for visual diagnostic.

**Existing diagnostics replaced.** `t_stats` → `prob_sign_correct`; HAC standard errors → posterior SDs (which already correctly reflect the uncertainty in σ²); `p_values` → `credible_intervals`.

**API sketch.**

```
POST /fit?method=bayes&prior=weakly_informative&n_samples=2000
```

For the conjugate case `n_samples` is ignored (analytic); for NUTS it's the chain length.

## 3. Implementation roadmap

**Ship order.** Bayesian linear (conjugate) → Elastic Net → Quantile Regression. Rationale below.

**Wave 1 — Bayesian linear (conjugate).** Lowest dependency cost (numpy + scipy, both already in tree), analytic posterior so no MCMC infrastructure needed, immediate value to users with n<60 fits where current t-stats are dishonest. Estimated effort: 1 day code + 1 day tests. New module: `pfm/quant/regression_bayes.py`. New tests: synthetic-DGP recovery of (μ, Σ); flat-prior limit recovers OLS β; tight-prior shrinks toward μ₀; posterior CI coverage on 1000-iter synthetic.

**Wave 2 — Elastic Net.** Requires `scikit-learn` (lightweight, already a transitive dependency through some test fixtures — verify before adding to `pyproject.toml`). Implement `ElasticNetCV` with `TimeSeriesSplit` to avoid look-ahead. Estimated effort: 1 day code + 1 day tests + 1 day endpoint plumbing. New module: `pfm/quant/regression_enet.py`. New tests: synthetic with 20 factors, 3 truly active → recover sparse pattern; check `TimeSeriesSplit` is used (no random CV on a time series); standardisation round-trip preserves response scale.

**Wave 3 — Quantile Regression.** Requires `statsmodels.regression.quantile_regression` (already in tree). Bootstrap CIs need a `concurrent.futures` pool sized to ~500 reps × ~10 ms = ~5 s warm. Estimated effort: 1 day code + 1 day tests + 1 day UI work for the tail-shape plot. New module: `pfm/quant/regression_quantile.py`. New tests: synthetic heteroskedastic-tail DGP recovers β(0.9) > β(0.5); bootstrap CIs cover the true β(τ) at the nominal rate.

**Dependencies summary.** `scikit-learn` (likely already pulled), `scipy.stats` (in tree), `statsmodels` (in tree). PyMC would be needed only for non-conjugate Bayesian; defer to a later wave if/when a user requests sparse-shrinkage priors. **No new heavy dependencies in Waves 1–3.**

**Test plan.** Every method gets the same three-layer test stack: (1) **synthetic DGP recovery** — generate y from known β, check we recover within a 2σ band; (2) **degenerate-input fuzzing** — n=p, n<p, perfect collinearity, all-zeros factor, all-NaN target; (3) **endpoint integration** — `TestClient.post('/fit?method=...')` with mocked Polymarket/yfinance data, snapshot response shape. Target ≥ 90 % coverage on each new module.

**Schema discipline.** All new response fields are *additive* and *optional* — existing clients continue to parse the response unchanged. The `method=ols_hac` default keeps the current behaviour exact (no `Optional` fields populated). Schema additions go at the END of `schemas.py` per protocol.

## 4. What NOT to ship (and why)

- **Markov-switching regression.** Killer combo of small n (often 30–120 daily obs), parameter explosion (≥ 13 parameters for a 2-state 5-factor model), fragile EM convergence, non-standard tests for K, and known propensity to over-fit. Producing a polished MS endpoint would mislead users into thinking we can identify regimes that we statistically cannot. Belongs to the same family of mistakes as the Wave-5-killed alphas (regime-driven results sold as structural).

- **Bayesian model averaging at full scale.** With 1228 factors the model space is 2^1228 — astronomically infeasible. A restricted BMA over a user-supplied candidate set of ≤10 factors *would* be tractable (1024 sub-models), but it duplicates a lot of what Elastic Net already provides at lower compute cost. Defer until a user explicitly asks for inclusion probabilities.

- **VAR / ECM as a `method=` flag.** These are not point-estimator swaps — they are *different econometric questions* (Granger causality, cointegration). They deserve dedicated endpoints (`/leadlag`, `/cointegration`) with their own response shapes. Squeezing them into `/fit?method=var` would produce a Frankenstein response that is hard to document and hard to consume.

- **GLS with AR(1) errors.** Marginal gain over HAC for our use case (HAC is robust to *unspecified* serial correlation; GLS is efficient only if AR(1) is correct, otherwise it's worse). Not worth a code path.

- **Robust (Huber) regression.** The "outlier" days (FOMC, jobs, geopolitical) are exactly when prediction-market factors are most informative. Down-weighting them is exactly the wrong thing for a prediction-market-factor model. Robust regression is a tool for *contamination*, not for *fat tails of the actual signal* — that's what Quantile Regression is for.

## 5. Risks and mitigations

**Data sparsity.** Some factors have ≤ 30 daily observations. Elastic Net's CV needs ≥ 5 folds × ≥ 5 obs/fold = 25 obs minimum; Bayesian conjugate degrades gracefully (the prior dominates); Quantile Regression's bootstrap CI widths blow up below n=60. Mitigation: hard-fail Quantile Regression below n=60 with an explanatory 422; warn-degrade Elastic Net (skip CV, use a default λ) below n=60; allow Bayesian on any n with a prior-dominated CI warning.

**Look-ahead bias in rolling/CV.** Random k-fold CV on a time series is a data-leakage bug. Must use `sklearn.model_selection.TimeSeriesSplit` (forward chaining) for all Elastic Net CV. Test for this explicitly: a fixture where future leaks into past should produce a higher CV R² than the correct walk-forward, and the test asserts that we get the *lower*, correct number.

**Multiple testing over 1228 factors.** Users will absolutely run "give me the best factor for NVDA from the whole catalog" prompts. Without multiplicity correction, the best p-value of 1228 univariate fits is ≈ 0 by chance alone. Mitigations: (a) Elastic Net is the *right* tool for this — it implicitly handles the search-space via regularisation; (b) any endpoint that scans the whole catalog *must* return Benjamini–Hochberg-FDR-adjusted p-values (Bailey & López de Prado 2014 deflated-Sharpe ratio is the spiritual cousin in finance); (c) the response must include `n_tested: int` so users see the search space.

**Posterior abuse.** Bayesian credible intervals are not p-values, and Bayesian "probability the factor matters" is not the same as a frequentist test. Documentation must be explicit about this. Default priors must be transparent and visible in the response (`prior: {beta: "N(0, 100)", sigma2: "InvGamma(1, 1)"}`).

**Compute cost.** Bootstrap Quantile Regression at 5 quantiles × 500 reps = 2500 fits per request. On n=120 this is ~5 s warm, which is acceptable but not free. Mitigations: parallelise bootstrap with `concurrent.futures.ProcessPoolExecutor` (Python GIL releases under numpy ops anyway); cache the design matrix; rate-limit `/fit?method=quantile` requests per IP.

**Communication.** The current `/fit` response is already dense (factor metadata, warnings, clipping events, VIF, HAC lag, …). Adding posterior summaries, regularisation paths, and quantile tables risks making the JSON unreadable. Mitigation: the frontend must opt into the richer payload via `?verbose=1`; the default response stays terse, and method-specific fields only appear when the relevant method was chosen.

## 6. Selected references

- Belloni, A. and Chernozhukov, V. (2013). *Least squares after model selection in high-dimensional sparse models.* Bernoulli, 19(2): 521–547. **Post-LASSO inference, motivates the de-sparsification flag.**
- Koenker, R. (2005). *Quantile Regression.* Cambridge University Press. **The standard reference; chapter 3 on bootstrap inference is the practical default we should follow.**
- Koenker, R. and Machado, J. A. F. (1999). *Goodness of fit and related inference processes for quantile regression.* JASA 94(448): 1296–1310. **Pseudo-R² for QR.**
- Bailey, D. H. and López de Prado, M. (2014). *The deflated Sharpe ratio.* Journal of Portfolio Management 40(5): 94–107. **Multiple-testing adjustment for trading-strategy backtests; the family our /fit-over-1228-factors workflow belongs to.**
- Zou, H. and Hastie, T. (2005). *Regularization and variable selection via the elastic net.* JRSS-B 67(2): 301–320. **The Elastic Net paper itself.**
- Javanmard, A. and Montanari, A. (2014). *Confidence intervals and hypothesis testing for high-dimensional regression.* JMLR 15: 2869–2909. **De-sparsification for valid post-selection inference.**
- Hamilton, J. D. (1989). *A new approach to the economic analysis of nonstationary time series and the business cycle.* Econometrica 57(2): 357–384. **MS regression foundational — we cite to explain why we are *not* shipping it.**
- van de Geer, S., Bühlmann, P., Ritov, Y., and Dezeure, R. (2014). *On asymptotically optimal confidence regions and tests for high-dimensional models.* Annals of Statistics 42(3): 1166–1202. **Companion to Javanmard–Montanari.**

## 7. Companion stub

A function-signature-only stub of the top 3 picks lives at `api/src/pfm/quant/regression_methods.py`. It contains no implementation — just typed signatures, docstrings, and `raise NotImplementedError` bodies — so that downstream design can proceed in parallel with this proposal.
