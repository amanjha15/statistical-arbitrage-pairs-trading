"""Hedge-ratio estimation, spread construction, and stationarity/cointegration tests.

Implements the core relationship:

    A_t = alpha + beta * B_t + eps_t
    S_t = A_t - alpha - beta * B_t

with an ADF test on S_t used as evidence of cointegration (Engle-Granger
two-step method), plus a half-life estimate of mean-reversion speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class HedgeRatio:
    alpha: float
    beta: float


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> HedgeRatio:
    """OLS estimate of A_t = alpha + beta * B_t + eps_t.

    y is the dependent series (A), x is the independent series (B).
    """
    x_with_const = sm.add_constant(x.values)
    model = sm.OLS(y.values, x_with_const).fit()
    alpha, beta = model.params
    return HedgeRatio(alpha=float(alpha), beta=float(beta))


def compute_spread(y: pd.Series, x: pd.Series, hedge: HedgeRatio) -> pd.Series:
    """S_t = A_t - alpha - beta * B_t."""
    spread = y - hedge.alpha - hedge.beta * x
    spread.name = "spread"
    return spread


@dataclass
class ADFResult:
    statistic: float
    pvalue: float
    used_lag: int
    n_obs: int
    critical_values: dict[str, float]
    is_stationary_5pct: bool


def adf_test(series: pd.Series, regression: str = "c") -> ADFResult:
    """Augmented Dickey-Fuller unit-root test.

    H0: series has a unit root (non-stationary), H1: stationary/mean-reverting.
    """
    stat, pvalue, used_lag, n_obs, crit, _ = adfuller(
        series.dropna().values, regression=regression, autolag="AIC", result_object=False
    )
    return ADFResult(
        statistic=float(stat),
        pvalue=float(pvalue),
        used_lag=int(used_lag),
        n_obs=int(n_obs),
        critical_values={k: float(v) for k, v in crit.items()},
        is_stationary_5pct=pvalue < 0.05,
    )


@dataclass
class CointegrationResult:
    hedge: HedgeRatio
    spread: pd.Series
    adf: ADFResult
    engle_granger_pvalue: float
    is_cointegrated: bool


def test_cointegration(y: pd.Series, x: pd.Series, alpha: float = 0.05) -> CointegrationResult:
    """Full Engle-Granger pipeline: OLS hedge ratio -> spread -> ADF on the residual.

    Also cross-checks with statsmodels' `coint`, which runs the same
    two-step test with appropriate critical values for a generated residual.
    """
    hedge = estimate_hedge_ratio(y, x)
    spread = compute_spread(y, x, hedge)
    adf = adf_test(spread)

    _, eg_pvalue, _ = coint(y.values, x.values)

    is_cointegrated = adf.is_stationary_5pct and eg_pvalue < alpha
    return CointegrationResult(
        hedge=hedge,
        spread=spread,
        adf=adf,
        engle_granger_pvalue=float(eg_pvalue),
        is_cointegrated=is_cointegrated,
    )


def half_life(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life from delta_S_t = gamma * S_{t-1} + eps_t.

    t_half = -ln(2) / gamma, defined only when gamma < 0 (mean-reverting).
    Returns np.inf if gamma >= 0 (no mean reversion detected).
    """
    s = spread.dropna()
    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()

    lagged, delta = lagged.align(delta, join="inner")

    x_with_const = sm.add_constant(lagged.values)
    model = sm.OLS(delta.values, x_with_const).fit()
    gamma = model.params[1]

    if gamma >= 0:
        return float("inf")
    return float(-np.log(2) / gamma)
