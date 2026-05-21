# Cross-asset multi-event chain models

This note documents the math behind `pfm.multi_event_chain`. We connect a
universe of $F$ prediction-market factors with a universe of equity/sector
return series, search Granger-significant chains across events, and
extract a low-dimensional systemic factor from the joint Δlogit panel.

## 1. Sparse multi-event factor model (LASSO)

Let $x_{i,t} = \Delta\,\mathrm{logit}(p_{i,t})$ be the Δlogit of PM
factor $i$ at time $t$, $i = 1,\dots,F$, and $r_{j,t}$ the log return of
asset $j$. The Lasso estimator solves

$$\hat\beta_j = \arg\min_{\beta\in\mathbb R^F}\;\;\frac{1}{2N}\;\bigl\Vert r_j - X\beta\bigr\Vert_2^2 + \alpha\,\Vert\beta\Vert_1.$$

The L1 penalty produces a sparse $\hat\beta_j$: only the factors that
materially explain $r_j$ stay non-zero. We pick $\alpha$ by
`LassoCV` (5-fold) on the training panel after standardising $X$ so the
penalty is unit-free across factors with different Δlogit volatilities.
After fitting we un-standardise back to the original scale so reported
betas have the usual interpretation (return per unit Δlogit).

## 2. Sector attribution decomposition

For each sector ETF $j \in \{\mathrm{XLF},\mathrm{XLK},\dots\}$ we run a
Newey-West HAC-OLS on the dense factor matrix $X$:

$$r_{j,t} = \alpha_j + \sum_{i=1}^F \beta_{j,i}\,x_{i,t} + \varepsilon_{j,t}.$$

The variance attributed to factor $i$ in sector $j$ is

$$A_{j,i} = \frac{\beta_{j,i}^2\,\mathrm{Var}(x_i)}{\mathrm{Var}(r_j)},$$

i.e. the share of the sector's variance "explained" by factor $i$ in
isolation. $A$ is an $S\times F$ heat-map; row sums approximate the
regression $R^2$ (covariance between regressors is folded into the
residual). We surface the row-argmax (`dominant_factor_per_sector`) and
column-argmax (`dominant_sector_per_factor`) for fast UI summarisation.

## 3. Multi-event chains via Granger pathfinding

Given a starting factor $A$, candidate intermediates
$\{B_1,\dots,B_K\}$, and a terminal ticker $T$, we DFS up to depth
$D$ over factor-to-factor and factor-to-return edges. An edge
$A\to B$ exists when Bivariate Granger F-test gives $p_{A\to B}<0.10$ at
the lag with smallest $p$:

$$F = \frac{(\mathrm{SSR}_R - \mathrm{SSR}_U)/L}{\mathrm{SSR}_U/(T-2L-1)}.$$

The terminal edge tests "factor Granger-causes log returns of $T$". Path
strength is reported as the worst-link p (`granger_p_max`) and the
product of pairwise correlations (`total_correlation`).

## 4. Event--macro overlay

For factor $i$ and macro series $m\in\{\mathrm{DFF},\mathrm{DGS10},\dots\}$
we correlate $x_{i,t}$ with $\Delta m_{t-k}$ for
$k\in[-K,K]$ and pick $k^\star = \arg\max_k|\rho_k|$. The reported
$t$-statistic uses Newey-West HAC SEs on the univariate slope at $k^\star$.

## 5. PCA-derived systemic factor

We standardise the Δlogit panel and run PCA. The first principal
component $\xi^{(1)}_t$ is interpreted as a PM-implied risk-on/off
signal:

$$\xi^{(1)}_t = \sum_{i=1}^F w^{(1)}_i\, \tilde x_{i,t},\qquad \mathrm{Var}(\xi^{(1)}) = \lambda_1.$$

When the first eigenvalue captures $\geq 20\%$ of total variance the
factor is flagged `can_use_as_factor=True` so callers can plug
$\xi^{(1)}$ directly into the regression / Strategies layer as a new
synthetic factor. Loadings $w^{(1)}$ are returned for interpretability.
