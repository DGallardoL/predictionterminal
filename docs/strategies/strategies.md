# Strategies — theoretical reference

This document accompanies `pfm.strategies` and the `/strategies/*` endpoints.
It explains *what each detector measures*, *under what assumptions*, and *what
it does NOT measure*. We avoid the common pitfall of treating prediction-market
prices as point estimates of true probabilities — under LMSR market-making
and risk-aversion biases, they are not.

## 1. What does a Polymarket / Kalshi YES-price represent?

For a binary event $E$ with YES contract paying $\$1$ if $E$ occurs and $\$0$
otherwise, the theoretical risk-neutral price under no transaction costs is
$P(E)$ — the market's consensus probability. In practice prices deviate
from $P(E)$ for at least four reasons:

1. **LMSR cost-function curvature**. Both Polymarket (CLOB-overlay) and Kalshi
   use Hanson's logarithmic market-scoring rule. The price is not the
   arithmetic mean of agent beliefs but the gradient of a strictly convex cost
   function $C(q) = b \log\!\sum_i e^{q_i / b}$. Equilibrium prices clear at
   the marginal cost; deep liquidity ($b$ large) makes the price near-linear
   in trader beliefs but in thin markets curvature distortions are significant
   (Hanson 2003).

2. **Risk aversion**. If traders have concave utility, the equilibrium price
   $\pi$ satisfies $\pi = \mathbb{E}[u'(W) \cdot \mathbf{1}_E] / \mathbb{E}[u'(W)]$
   under no-arbitrage rather than $\pi = \mathbb{E}[\mathbf{1}_E] = P(E)$
   (Manski 2006). For events that correlate with marginal utility — recessions,
   pandemics — the bias is non-trivial.

3. **Bid-ask spreads & fees**. Polymarket charges no taker fee but the
   spread is typically 1–3¢ on liquid markets and 5–10¢ on illiquid ones.
   Kalshi takes a per-contract fee. Naive $\sum_i p_i \neq 1$ "arbitrage"
   ignores both spreads and the cost of capital tied up to settlement
   (Wolfers & Zitzewitz 2006).

4. **Resolution-source variance**. "BTC > \$100k by Date X" markets settle on
   a specific oracle (UMA-resolved; usually a Coinbase or aggregated index).
   A live spot from another exchange (Binance) can deviate from the resolution
   source by 1–5 bps, which compounds near the strike.

We treat YES-prices as **the market's consensus probability under all of the
above frictions**, not as $P(E)$ itself. The strategies below extract
information that is *robust* to these frictions, or quantifies the impact of
ignoring them.

## 2. Logical-implication test

### Claim

If event $A$ logically implies event $B$ (i.e. $A \subseteq B$ as outcome
sets), then for *any* probability measure

$$P(A) \le P(B). \tag{1}$$

If we observe a market price $\pi_A$ and another $\pi_B$ satisfying
$\pi_A > \pi_B + \tau$ on date $t$ (for tolerance $\tau \approx$ half-spread),
either the market is mispriced or our supposed implication does not hold
(market interpretation differs from ours).

### Diagnostics we report

- $\text{gap}_t = \pi_A^{(t)} - \pi_B^{(t)}$ on the linear scale.
- $\text{logit-gap}_t = \mathrm{logit}(\pi_A^{(t)}) - \mathrm{logit}(\pi_B^{(t)})$
  — scale-stable comparison; a gap of 0.05 at $\pi=0.50$ means something very
  different than at $\pi=0.05$.
- Number of dates with $\text{gap}_t > \tau$, bucketed into a verdict:
  *consistent* (0–0 violations), *borderline* (1–4), *violated* ($\ge 5$).

### Why we don't auto-trade on this

A persistent violation can mean: (a) one market's resolution criterion is
subtly different from the other (e.g. "by Dec 31 23:59 UTC" vs "by close of
business Dec 31"); (b) the markets clear on different LMSR curves and the
gap is the curvature wedge; (c) genuine market mispricing. (a) and (b) are
not arbitrageable.

### Tolerance choice

Default $\tau = 0.02$ comes from typical Polymarket bid-ask half-spreads on
liquid markets. For thinner markets, raise it. For markets in the 0.01–0.10
or 0.90–0.99 range where logit-scale moves correspond to tiny price moves,
inspect the `logit-gap` series instead.

## 3. Conditional probability via co-move regression

### Setup

Let $\pi_A, \pi_B$ be daily YES-prices for events $A, B$. We fit

$$\pi_A^{(t)} = \alpha + \beta \pi_B^{(t)} + \varepsilon_t \tag{2}$$

with **HAC (Newey-West) standard errors** because $\pi$-series are
auto-correlated by construction (overlapping news effects across dates).

### Interpretation of $\beta$

Two complementary readings:

1. **Linear-projection slope**. $\hat{\beta} = \widehat{\mathrm{Cov}}(\pi_A, \pi_B) / \widehat{\mathrm{Var}}(\pi_B)$.
   This is just "how much does the market for $A$ move when the market for $B$
   moves by $\Delta$". Always defined as long as $\pi_B$ has positive sample
   variance.

2. **Conditional-mean slope under binarisation**. Suppose we treat
   $A_t = \mathbf{1}\{\pi_A^{(t)} > 0.5\}$ and $B_t$ similarly; under
   stationarity and joint Bernoulli structure, the population analogue of
   $\beta$ is

   $$\beta = P(A=1 \mid B=1) - P(A=1 \mid B=0). \tag{3}$$

   That is, $\beta$ is the *conditional risk difference*, which we report
   alongside the empirical conditional means (split at $\pi_B = 0.5$).

We caution that (3) holds in the limit and under simplifying assumptions; the
HAC-CI on $\beta$ from (2) is the rigorous statistical statement, while the
binarised conditional means are descriptive.

### HAC lag choice

Default lag is 5. For long-running markets with multi-week news cycles
(e.g. macro markets driven by FOMC schedule), use 10–20. The published p-values
are robust to lag choice within an order of magnitude.

## 4. Fréchet-Hoeffding bounds

For *any* bivariate distribution with marginals $P(A), P(B)$, the joint
satisfies

$$\max(0, P(A) + P(B) - 1) \le P(A \cap B) \le \min(P(A), P(B)). \tag{4}$$

These are the **distribution-free** sharp bounds (Fréchet 1951; Hoeffding 1940).
The width

$$w(t) = \min(\pi_A^{(t)}, \pi_B^{(t)}) - \max(0, \pi_A^{(t)} + \pi_B^{(t)} - 1)$$

quantifies how much the marginals already pin down the joint. When $w(t)$ is
small, observing marginal prices effectively determines $P(A \cap B)$; when
large, the joint is unidentified by marginals alone.

The independence reference $\pi_A^{(t)} \cdot \pi_B^{(t)}$ always lies inside
the band, so reading off "is the joint above or below independence?" requires
*observing the actual joint* — which prediction markets typically do not price.
What we *can* show is the **range of valid joint probabilities** consistent
with the observed marginals, which is informative for portfolio construction
on combination outcomes.

### Use cases

- **"Conjunction" portfolios**. If you want exposure to "Trump wins AND BTC
  above 100k", and only the marginals trade, the bounds tell you what joint
  prices would be self-consistent.
- **Detecting joint-pricing impossibilities**. If a prediction market *does*
  list a combo "A and B" with price $\pi_{AB}$, check $\pi_{AB} \in [\text{lower}_t, \text{upper}_t]$.
  Violations are arbitrageable in principle.

## 5. Why we deliberately skip naive arbitrage detection

A common first idea: scan tuples of mutually-exclusive events $\{E_1, \ldots, E_k\}$
and flag any date where $\sum_i \pi_i \neq 1$. We deliberately do not
implement this because:

1. **Spreads dominate**. If the typical half-spread is $\tau \approx 0.02$,
   then $|\sum \pi_i - 1| < k \tau$ is *normal*. To call something
   "arbitrageable", you need the deviation to exceed the *round-trip* spread
   plus fees plus capital opportunity cost — which on a long-dated market
   could be several percent. Naive flagging produces false positives.

2. **Settlement risk**. Even if you sell each YES of a sum-to-1 set at the
   ask, you only realise the arbitrage *at settlement*. A 60-day capital
   lockup at 5% borrow = 0.83% headwind. Polymarket's UMA dispute window
   adds further uncertainty.

3. **LMSR curvature**. As you trade, you move prices against you. A nominal
   0.05 arb on screen typically delivers 0.005 of realised P&L after price
   impact in thin markets.

The honest replacement for naive arbitrage detection is the **logical
implication test** (§2) and **Fréchet-bound violation check** (§4) — both
of which surface mispricing patterns *without* claiming that the user can
trade them at the screen price.

## 6. Future direction: spot-vs-implied (rigorous GBM)

A genuinely interesting strategy compares a *live underlying* (e.g. BTC spot
from Binance) against the prediction-market-implied probability of a
price-touch event. Done rigorously this requires:

- A vol estimator that handles overnight gaps (we plan **Yang-Zhang OHLC**
  per Yang & Zhang 2000, which is ~5× more efficient than close-to-close).
- Closed-form GBM probabilities for *terminal* and *one-touch* market shapes.
- A **bootstrap CI** on the model probability (block-bootstrap to preserve
  vol clustering).
- Explicit accounting for resolution-source basis (Coinbase index ≠ Binance
  spot) and a configurable spread input.

This is documented as a planned-not-implemented strategy in the task list
(`spot-vs-market-implied`). The current module exposes only the three
distribution-free / non-parametric tests above, which are honest about the
data and don't require a model of the underlying.

## References

- Fréchet, M. (1951). *Sur les tableaux de corrélation dont les marges sont données.* Annales de l'Université de Lyon.
- Hanson, R. (2003). *Combinatorial Information Market Design.* Information Systems Frontiers 5(1), 107–119.
- Hoeffding, W. (1940). *Maßstabinvariante Korrelationstheorie.* Schriften des Mathematischen Instituts der Universität Berlin.
- Manski, C. F. (2006). *Interpreting the Predictions of Prediction Markets.* Economics Letters 91(3), 425–429.
- Newey, W. K. & West, K. D. (1987). *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica 55(3), 703–708.
- Wolfers, J. & Zitzewitz, E. (2006). *Interpreting Prediction Market Prices as Probabilities.* NBER Working Paper 12200.
- Yang, D. & Zhang, Q. (2000). *Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices.* Journal of Business 73(3), 477–491.
