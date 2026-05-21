# Advanced quant rigor: continuous p-values, OOS-R², DM, White RC, deflated Sharpe

This document covers the second-tier statistical machinery layered on top of
the core regression. Each section gives the formula and the file/function that
implements it.

## 1. MacKinnon-Haug-Michelis Johansen p-values

The Johansen trace and max-eigenvalue tests have asymptotic null distributions
without a closed-form CDF. MacKinnon, Haug & Michelis (1999) fit response
surfaces of the form

$$
\Pr\left( \mathrm{LR} \le q \right) \;\approx\; F_{\Gamma}\!\left( q ; \;
\mu = b_{0} + \frac{b_{1}}{T} + \frac{b_{2}}{T^{2}}, \;
v = w_{0} + \frac{w_{1}}{T} + \frac{w_{2}}{T^{2}} \right)
$$

with the Gamma parameterised so its mean equals $\mu$ and variance $v$
(equivalently $\mathrm{shape}=\mu^{2}/v$, $\mathrm{scale}=v/\mu$). The
asymptotic terms $b_{0}, w_{0}$ are tabulated for each combination of

* test $\in$ {trace, eigen}
* deterministic-trend order $d \in \{-1, 0, 1\}$ (no constant / constant /
  linear)
* $n - r$ = number of common stochastic trends under the null

In `pfm.mhm_critical` we pin the Gamma at each cell against the
Osterwald-Lenum / Johansen 1995 published 90% and 99% critical values so the
recovered p-value is **exact at the bucketed boundaries** and continuous in
between. This replaces the old four-bucket lookup
($\{0.005, 0.025, 0.075, 0.20\}$) with a smooth p-value usable in
BH-FDR pipelines and in BLDP deflated-Sharpe accounting.

## 2. Campbell-Thompson out-of-sample R²

In-sample $R^{2}$ over-states predictive ability. Campbell & Thompson (2008)
define

$$
R^{2}_{OOS} = 1 - \frac{ \sum_{t} \left( y_{t} - \hat{y}^{model}_{t} \right)^{2} }
                       { \sum_{t} \left( y_{t} - \hat{y}^{base}_{t} \right)^{2} }
$$

Positive $R^{2}_{OOS}$ means the model beats the baseline (typically the
recursive sample mean) out of sample. Negative values are informative:
the model adds noise.

For *nested* comparisons the standard Diebold-Mariano test biases against
the larger model. Clark & West (2007) build a corrected loss differential

$$
\hat{f}_{t} = \left( y_{t} - \hat{y}^{base}_{t} \right)^{2} - \left[
  \left( y_{t} - \hat{y}^{model}_{t} \right)^{2}
  - \left( \hat{y}^{base}_{t} - \hat{y}^{model}_{t} \right)^{2} \right]
$$

whose sample mean is studentised by a Newey-West HAC variance with
bandwidth $\sim T^{1/3}$. Under $H_{0}$: equal predictive accuracy, the
ratio is asymptotically standard normal.

Implementation: `pfm.oos_metrics.oos_r_squared_campbell_thompson`. Endpoint:
`POST /quant/oos-r-squared`.

## 3. Diebold-Mariano test with HLN correction

Given two forecasters with errors $e_{1,t}, e_{2,t}$ and a loss $L(\cdot)$,
let $d_{t} = L(e_{1,t}) - L(e_{2,t})$. The DM statistic is

$$
\mathrm{DM} = \frac{\bar{d}}{ \sqrt{ \widehat{V}_{HAC}(\bar{d}) } } \quad
\stackrel{a}{\sim}\quad \mathcal{N}(0, 1)
$$

The Newey-West bandwidth defaults to $h - 1$ for $h$-step forecasts. Harvey,
Leybourne & Newbold (1997) note finite-sample over-rejection at $h>1$ and
provide the multiplier

$$
\mathrm{DM}^{*} = \mathrm{DM} \cdot \sqrt{ \frac{T + 1 - 2h + h(h-1)/T}{T} }
$$

with reference distribution $t_{T-1}$. Implementation:
`pfm.forecast_comparison.diebold_mariano`. Endpoint: `POST /quant/diebold-mariano`.

## 4. White's Reality Check, Hansen SPA, Romano-Wolf stepwise

White (2000) tests $H_{0}: \max_{k} \mathbb{E}[r_{k,t} - r_{bench,t}] \le 0$
across $K$ candidate strategies via the stationary block bootstrap of Politis
and Romano (1994). The test statistic is

$$
V_{T} = \max_{k=1..K} \sqrt{T} \, \overline{(r_{k} - r_{bench})}
$$

and the bootstrap p-value is

$$
p_{RC} = \frac{1}{B} \sum_{b=1}^{B} \mathbb{1}\!\left( V_{T}^{*(b)} \ge V_{T} \right)
$$

with $V_{T}^{*(b)} = \max_{k} \sqrt{T} (\bar{r}^{*(b)}_{k} - \bar{r}_{k})$
the centred bootstrap maximum. Hansen's SPA (2005) recenters only strategies
that pass a soft consistency screen $\bar{r}_{k} \ge -\sigma_{k}\sqrt{2\log\log T / T}$,
which is less conservative when the family contains losers.

Romano & Wolf (2005) extend to identifying *all* strategies that beat the
benchmark: at each step take the surviving family, find the
$(1-\alpha)$-quantile of the bootstrap-max $t$-statistic, reject any whose
own $t$-stat exceeds it, repeat until no new rejections. The procedure
**strongly controls FWER** at level $\alpha$.

Implementation: `pfm.whites_reality_check.{whites_reality_check,stepwise_spa}`.
Endpoint: `POST /quant/whites-reality-check`.

## 5. Deflated Sharpe ratio (Bailey & Lopez de Prado 2014, full)

The expected maximum Sharpe under the null over $N$ trials follows from a
Mill's-ratio expansion:

$$
\mathbb{E}[\widehat{SR}_{\max}]
\;\approx\; \sigma_{SR} \left[ (1-\gamma) \, \Phi^{-1}\!\left(1 - \tfrac{1}{N}\right)
  + \gamma \, \Phi^{-1}\!\left(1 - \tfrac{1}{Ne}\right) \right]
$$

where $\gamma = 0.5772\ldots$ is the Euler-Mascheroni constant and
$\sigma_{SR}$ is the cross-trial Sharpe dispersion. The Edgeworth-expansion
finite-sample SE of the per-period Sharpe estimator is

$$
\widehat{SE}(\widehat{SR}) =
\sqrt{ \frac{ 1 - \gamma_{3} \widehat{SR} +
              \frac{ \gamma_{4} - 1 }{ 4 } \widehat{SR}^{2} }{ T - 1 } }
$$

with $\gamma_{3}$ skew and $\gamma_{4}$ Pearson kurtosis (3 = Gaussian).
The deflated Sharpe statistic and p-value are

$$
\mathrm{DSR} = \Phi\!\left( \frac{ \widehat{SR} - \mathbb{E}[\widehat{SR}_{\max}] }{ \widehat{SE}(\widehat{SR}) } \right) \quad,\quad
p_{def} = 1 - \mathrm{DSR}
$$

Implementation: `pfm.multitest.deflated_sharpe_full`. The legacy
`pfm.robust_validation.deflated_sharpe_ratio` is preserved for backward
compatibility; both agree to within ~0.10 deflated p-value across the
operating range.

## References

* MacKinnon, J. G., Haug, A. A., Michelis, L. (1999). *Numerical Distribution
  Functions of Likelihood Ratio Tests for Cointegration*, J. Appl. Econ.
* Campbell, J. Y., Thompson, S. B. (2008). *Predicting Excess Stock Returns
  Out of Sample*, RFS.
* Clark, T. E., West, K. D. (2007). *Approximately Normal Tests for Equal
  Predictive Accuracy in Nested Models*, JoE.
* Diebold, F. X., Mariano, R. S. (1995). *Comparing Predictive Accuracy*, JBES.
* Harvey, D., Leybourne, S., Newbold, P. (1997). *Testing the equality of
  prediction mean squared errors*, IJF.
* White, H. (2000). *A Reality Check for Data Snooping*, Econometrica.
* Hansen, P. R. (2005). *A Test for Superior Predictive Ability*, JBES.
* Romano, J. P., Wolf, M. (2005). *Stepwise Multiple Testing as Formalized
  Data Snooping*, Econometrica.
* Politis, D. N., Romano, J. P. (1994). *The Stationary Bootstrap*, JASA.
* Bailey, D. H., Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio*, JPM.
