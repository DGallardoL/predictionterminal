# Advanced Event-Conditioned Quant Theory

The baseline factor model fitted by `pfm.model.fit_ols_hac` regresses log
equity returns on the first difference of the logit-transformed
prediction-market probability:

$$
r_t = \alpha + \beta\,\Delta\!\operatorname{logit}(p_t) + \varepsilon_t,
\qquad \operatorname{logit}(p) = \log\frac{p}{1-p}.
$$

This is honest and identifiable, but it imposes (i) a single global
$\beta$, (ii) linearity in $\Delta\!\operatorname{logit}$, (iii) constant
conditional variance, and (iv) no long-run equilibrium between the equity
level and the implied probability. The six specifications below
relax those restrictions one at a time.

## A. Conditional model

Partition the support of $p$ into $K{+}1$ buckets $B_0,\dots,B_K$ defined
by ascending thresholds $0<\tau_1<\dots<\tau_K<1$. Within each bucket
fit

$$
r_t = \alpha_b + \beta_b\,\Delta\!\operatorname{logit}(p_t) + \varepsilon_t,\quad b\in\{0,\dots,K\}.
$$

A Breusch-Pagan test on the pooled residuals flags
heteroscedasticity that justifies the partition. **Use case:** equity
sensitivity to favourite-side news ($p>0.7$) is empirically smaller than
to longshot-side news.

## B. Polynomial / non-linear

Fit

$$
r_t = \alpha + \sum_{k=1}^{d} \beta_k\,(\Delta\!\operatorname{logit}(p_t))^k + \varepsilon_t,
$$

with HAC SEs and a robust $F$-test of $H_0: \beta_2=\dots=\beta_d=0$
against the linear $d{=}1$ model. AIC selects the best degree on
$\{1,\dots,5\}$. **Use case:** asymmetric or kinked response — large
$\Delta\!\operatorname{logit}$ shocks move the equity disproportionately.

## C. Markov regime-switching

Hamilton (1989). With latent state $s_t\in\{0,\dots,K{-}1\}$ following an
$K\times K$ Markov chain $P$,

$$
r_t \mid s_t = \alpha_{s_t} + \beta_{s_t}\,\Delta\!\operatorname{logit}(p_t) + \sigma_{s_t}\,\eta_t,\quad \eta_t\sim\mathcal N(0,1),
$$

estimated by EM. We surface per-state $(\alpha,\beta,\sigma)$, $P$, the
ergodic distribution $\pi=\pi P$, and smoothed $\Pr(s_t=k\mid r_{1:T})$.
**Use case:** the same factor flips sign across calm/turbulent regimes.

## D. Vector Error Correction (VECM)

Stack $y_t=(\log P_t^{\text{eq}},\,\operatorname{logit}(p_t))^\top$.
Johansen tests the rank of $\Pi$ in

$$
\Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{k-1}\Gamma_i\,\Delta y_{t-i} + \varepsilon_t,
\quad \Pi=\alpha\beta'.
$$

If rank $1$, $\beta$ is the cointegrating vector and $\alpha$ is the
loading vector that pulls the system back to equilibrium. Half-life of
adjustment for the target equation: $T_{1/2}=-\log 2/\log(1+\alpha_1)$.
**Use case:** prediction-market belief and equity price share a stochastic
trend; tradeable spread.

## E. GARCH-X

Bollerslev (1986) with the prediction-market signal as exogenous variance
regressor:

$$
\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2 + \gamma\,X_t,
\quad X_t = |\Delta\!\operatorname{logit}(p_t)|.
$$

Stationary if $\alpha+\beta<1$. The reported "% conditional variance
explained by factor" is $\gamma\,\bar X / \overline{\sigma^2}$. **Use
case:** Polymarket flow predicts realised vol on the underlying.

## F. Tail dependence (copula)

Empirical lower-tail coefficient

$$
\hat\lambda_L(q) = \frac{|\,r_t<F_R^{-1}(q)\ \wedge\ X_t<F_X^{-1}(q)|}{|\,X_t<F_X^{-1}(q)|}.
$$

Independence baseline $=q$. A ratio $\hat\lambda_L(q)/q\gtrsim 2$ signals
joint left-tail clustering not captured by Pearson correlation. We
report lower, upper, and the asymmetry $\hat\lambda_L-\hat\lambda_U$.
**Use case:** pricing crash insurance from binary catastrophe contracts.
