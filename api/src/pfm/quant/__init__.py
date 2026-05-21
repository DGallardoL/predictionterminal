"""pfm.quant — alternative regression methods.

This subpackage holds research stubs and (eventually) implementations for
regression families beyond the default OLS + HAC. See
``docs/regression-methodology-improvements.md`` for the full design.

Currently the only module here is ``regression_methods``, which exposes
signature-only stubs for the three picks (Bayesian linear, Elastic Net,
Quantile Regression). No code path imports this subpackage in production yet.
"""

__all__ = ["deflated_sharpe", "regression_methods"]
