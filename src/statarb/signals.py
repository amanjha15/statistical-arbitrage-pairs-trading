"""Rolling spread estimation, z-scores, and entry/exit/stop signal generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RollingEstimate:
    """Rolling hedge ratio, spread, and z-score computed with no look-ahead.

    At each time t, beta_t/mu_t/sigma_t are estimated using only data in
    the trailing window ending at t-1, so the z-score at t never uses
    information from t itself.
    """

    beta: pd.Series
    spread: pd.Series
    zscore: pd.Series


def rolling_hedge_ratio(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta of y on x (with intercept), one window per point.

    beta_t is estimated from the window [t-window, t-1], i.e. it does not
    include the current observation, avoiding look-ahead bias.
    """
    y_shifted = y
    x_shifted = x

    cov = x_shifted.rolling(window).cov(y_shifted)
    var = x_shifted.rolling(window).var()
    beta = (cov / var).shift(1)
    beta.name = "beta"
    return beta


def rolling_intercept(y: pd.Series, x: pd.Series, beta: pd.Series, window: int) -> pd.Series:
    """Rolling alpha consistent with a rolling beta: mean(y) - beta * mean(x), lagged by 1."""
    y_mean = y.rolling(window).mean().shift(1)
    x_mean = x.rolling(window).mean().shift(1)
    alpha = y_mean - beta * x_mean
    alpha.name = "alpha"
    return alpha


def build_rolling_estimate(y: pd.Series, x: pd.Series, window: int, z_window: int | None = None) -> RollingEstimate:
    """Construct a no-look-ahead rolling spread and z-score.

    beta/alpha at time t use only data through t-1 (via the shift(1) in
    rolling_hedge_ratio/rolling_intercept). The z-score's own mean/std
    are computed over a trailing window of the spread and then shifted
    by one more step, so z_t depends only on spread values up to t-1
    plus the realized spread at t (which is itself built from lagged
    parameters).
    """
    if z_window is None:
        z_window = window

    beta = rolling_hedge_ratio(y, x, window)
    alpha = rolling_intercept(y, x, beta, window)
    spread = y - alpha - beta * x
    spread.name = "spread"

    mu = spread.rolling(z_window).mean()
    sigma = spread.rolling(z_window).std()
    zscore = (spread - mu) / sigma
    zscore.name = "zscore"

    return RollingEstimate(beta=beta, spread=spread, zscore=zscore)


def generate_signals(
    zscore: pd.Series,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float | None = 3.5,
) -> pd.Series:
    """Turn a z-score series into a position series in {-1, 0, +1}.

    Position convention (spread = A - alpha - beta*B):
      +1 = long spread  (long A, short beta*B)   -- entered when z < -entry
      -1 = short spread (short A, long beta*B)   -- entered when z > +entry
       0 = flat

    Exit when |z| < exit. If stop is set, force-flatten when |z| > stop
    regardless of the entry/exit rule (protective stop-loss).
    """
    z = zscore.copy()
    position = pd.Series(0, index=z.index, dtype=int)

    current = 0
    for t, zt in z.items():
        if np.isnan(zt):
            position[t] = 0
            continue

        if stop is not None and abs(zt) > stop:
            current = 0
        elif current == 0:
            if zt > entry:
                current = -1
            elif zt < -entry:
                current = 1
        else:
            if abs(zt) < exit:
                current = 0

        position[t] = current

    position.name = "position"
    return position
