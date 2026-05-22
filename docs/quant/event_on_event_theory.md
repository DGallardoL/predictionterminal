# Event-on-Event Factor Model — Theory

The classical regression in `pfm.model` regresses **equity log-returns** on
prediction-market $\Delta\text{logit}$ innovations. The event-on-event
variant flips one side of that equation: both target and regressors are
**prediction-market probability series**, and the question becomes "how
does event A's probability path co-move with the path of related events
B, C, D?".

## Δlogit transform

Probabilities live in $(0, 1)$, are bounded, heteroskedastic in the tails,
and asymmetric under shocks. The logit map

$$
\ell_t \;=\; \log \frac{p_t}{1 - p_t}
$$

stretches $(0, 1)$ onto $\mathbb{R}$, restoring symmetry: a move from
$p = 0.10$ to $p = 0.20$ has the same Δlogit magnitude as the symmetric
move from $0.80$ to $0.90$. We work on first differences

$$
\Delta\text{logit}(p_t) \;=\; \ell_t - \ell_{t-1}
\;=\; \log \frac{p_t}{1 - p_t} - \log \frac{p_{t-1}}{1 - p_{t-1}}
$$

so the resulting series is approximately stationary. Probabilities are
clipped to $[\varepsilon, 1 - \varepsilon]$ before transformation
($\varepsilon = 0.01$ by default) to avoid blow-up at resolution
boundaries.

## Multivariate model

Stack the target and $J$ predictors into one panel and fit

$$
\Delta\text{logit}(p_{i, t}) \;=\;
   \alpha + \sum_{j=1}^{J} \beta_j \cdot \Delta\text{logit}(p_{j, t}) + \epsilon_t
$$

via OLS with Newey-West HAC standard errors (Andrews 1991 bandwidth). The
intercept $\alpha$ captures any drift that survives the differencing; the
$\beta_j$ are partial sensitivities — the Δlogit move in event $i$ per
unit Δlogit move in event $j$, holding the other predictors fixed. VIF is
reported alongside each coefficient so the user can spot collinear
predictors (e.g. two near-duplicate election markets).

## Why Δlogit, not Δprob

Two reasons:

1. **Symmetry.** Δprob = $0.05$ at the centre is qualitatively different
   from Δprob = $0.05$ near a boundary. Δlogit normalises the metric.
2. **Stationarity.** Levels of probabilities are I(1)-ish for active
   markets — running OLS on levels is a textbook spurious-regression
   trap. Differenced logits are I(0).

## VAR(p)

For the full panel we additionally fit a vector autoregression

$$
X_t \;=\; c + \sum_{\ell = 1}^{p} A_\ell \, X_{t-\ell} + \epsilon_t,
\qquad X_t \in \mathbb{R}^k
$$

and report (a) per-pair Granger F-test p-values, (b) impulse responses
for the first three horizons, and (c) forecast-error-variance
decomposition. The VAR is the natural multivariate generalisation of the
bivariate lead-lag test.

## PCA on Δlogit innovations

Stacking the $T \times k$ matrix of Δlogit innovations and running PCA
yields $k$ orthogonal latent factors with shares of total variance
explained. A typical political-events panel is dominated by a single
"broad market" factor (all loadings same sign — every event moved
together on news days) plus 2–3 spread/rotation factors that distinguish
clusters (e.g. "Fed-cut basket" vs "Senate-control basket").

## Use cases

* **Conditional expectations** — *if Fed-cut-Mar resolves YES, how much
  does Fed-cut-Jun move?* Run `fit_event_on_event` with `target = Jun`,
  `predictors = [Mar]` over the live window, then plug Mar's resolution
  Δlogit ($\approx +5$ in absolute terms) into the fitted equation.
* **Lead-lag discovery** — call `event_lead_lag` to find which of two
  related events (e.g. election-odds vs senate-control) leads the other,
  with both a CCF panel and Granger p-values. Trade the follower, hedge
  in the leader.
* **Risk-factor decomposition** — call `event_pca_decomposition` on the
  full daily panel to ask: *what fraction of the average daily PM move
  is explained by 3 latent factors?* If the first PC explains >60%, the
  market is in a regime where everything moves on macro news; if shares
  are spread across components, idiosyncratic event risk dominates.
* **Implied joint distributions** — combine pairwise Δlogit correlations
  with the marginal Polymarket implied probabilities to construct an
  implied joint distribution over outcomes (under a Gaussian copula
  approximation). Useful for pricing baskets and conditional payoffs.

The five primitives compose: a daily-refreshed correlation heatmap plus
a PCA scree plot are sufficient to monitor whether the prediction-market
universe is in a "co-moving" or "idiosyncratic" regime, and the
event-on-event regression operationalises the bilateral relationships.
