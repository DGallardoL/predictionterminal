# Asymmetric GARCH: GJR-GARCH and EGARCH

The plain GARCH(1,1) model assumes symmetric response of conditional variance
to past shocks: a positive innovation of magnitude $|\epsilon_{t-1}|$ raises
$\sigma_t^2$ by exactly the same amount as a negative innovation of the same
magnitude. Empirical equity returns systematically violate this: negative
returns precede *higher* future volatility than positive returns. The
phenomenon is known as the **leverage effect** (Black 1976) and it is the
single most robust stylised fact about equity-return volatility, second only
to vol clustering itself.

## GJR-GARCH(1,1)

Glosten-Jagannathan-Runkle (1993) capture the leverage effect with an
indicator-style asymmetry term:

$$
\sigma_t^2 = \omega + \alpha\, \epsilon_{t-1}^2
              + \gamma\, I_{[\epsilon_{t-1} < 0]}\, \epsilon_{t-1}^2
              + \beta\, \sigma_{t-1}^2.
$$

The leverage indicator $I_{[\epsilon_{t-1}<0]}$ adds an extra $\gamma$ to the
ARCH coefficient when the previous shock was negative. **A positive $\gamma$
identifies the leverage effect.** Stationarity requires $\omega>0$,
$\alpha\ge 0$, $\beta\ge 0$ and $\alpha + \gamma/2 + \beta < 1$.

## EGARCH(1,1)

Nelson (1991) parameterises the *log* of the conditional variance, which
removes the positivity constraints on the ARCH/GARCH coefficients and lets
the model accommodate sign-dependent responses naturally:

$$
\log \sigma_t^2 = \omega
                  + \alpha\,\bigl(|z_{t-1}| - \mathbb{E}|z|\bigr)
                  + \gamma\, z_{t-1}
                  + \beta\, \log \sigma_{t-1}^2,
\qquad z_{t-1} = \epsilon_{t-1}/\sigma_{t-1}.
$$

For standard normal innovations $\mathbb{E}|z| = \sqrt{2/\pi}$. Crucially, the
$\gamma\, z_{t-1}$ term is signed: a positive $z$ contributes
$+\gamma$ while a negative $z$ contributes $-\gamma$. **A negative $\gamma$
identifies the leverage effect** in EGARCH (the opposite sign convention from
GJR). Stationarity only requires $|\beta|<1$; $\alpha$ and $\gamma$ are
unconstrained.

## Identification: signs you should expect on equity series

| Model       | Leverage parameter | Equity sign  | Typical magnitude |
|-------------|--------------------|--------------|-------------------|
| GJR-GARCH   | $\gamma$           | $\gamma > 0$ | $0.05 - 0.15$     |
| EGARCH      | $\gamma$           | $\gamma < 0$ | $-0.20$ to $-0.05$|

If a leverage parameter has the *wrong* sign on a long-history equity
series, the most likely causes are: (i) the sample is dominated by a single
positive-shock regime (rare); (ii) the optimiser landed in a poor local
optimum; (iii) the asset class is not actually leverage-prone (commodities,
some FX pairs).

## Model selection

In practice we fit GARCH(1,1), GJR-GARCH(1,1) and EGARCH(1,1) and select on
AIC/BIC. The asymmetric models add one parameter so they need to deliver a
non-trivial likelihood improvement — otherwise we keep the parsimonious
symmetric fit. The auxiliary leverage-coefficient $t$-statistic is reported
so users can decide whether to deploy the asymmetric model on its own
merits. If $|t_\gamma| < 1.96$ the leverage effect is not statistically
identified at the 5% level and the symmetric GARCH should be preferred.

## References

- Black, F. (1976). "Studies of Stock Price Volatility Changes."
- Glosten, L., Jagannathan, R., Runkle, D. (1993). *Journal of Finance* 48.
- Nelson, D. (1991). *Econometrica* 59, 347-370.
