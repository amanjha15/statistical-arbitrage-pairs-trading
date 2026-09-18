"""Performance metrics: Sharpe ratio, drawdown, and equity-curve summary stats."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    annualize: bool = True,
) -> float:
    """Sharpe = E[R - R_f] / sigma_R, optionally annualized by sqrt(periods_per_year).

    `returns` and `risk_free` should be in the same units (e.g. both daily
    simple returns). risk_free may be a scalar per-period rate.
    """
    excess = returns - risk_free
    sigma = excess.std()
    if np.isnan(sigma) or sigma < 1e-12:
        return 0.0

    sharpe = excess.mean() / sigma
    if annualize:
        sharpe *= np.sqrt(periods_per_year)
    return float(sharpe)


@dataclass
class DrawdownResult:
    drawdown_series: pd.Series
    max_drawdown: float
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp


def max_drawdown(equity_curve: pd.Series) -> DrawdownResult:
    """Largest peak-to-trough decline of the equity curve, as a positive magnitude.

    DD_t = (P_t - P_t^max) / P_t^max, P_t^max = running max up to t.
    """
    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max

    trough_date = dd.idxmin()
    peak_date = equity_curve.loc[:trough_date].idxmax()
    max_dd = float(-dd.min())

    return DrawdownResult(
        drawdown_series=dd,
        max_drawdown=max_dd,
        peak_date=peak_date,
        trough_date=trough_date,
    )


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate implied by the equity curve's start/end and length."""
    n_periods = len(equity_curve) - 1
    if n_periods <= 0 or equity_curve.iloc[0] <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std() * np.sqrt(periods_per_year))


@dataclass
class PerformanceSummary:
    total_return: float
    cagr: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    turnover: float


def summarize_performance(
    equity_curve: pd.Series,
    returns: pd.Series,
    turnover: pd.Series | None = None,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> PerformanceSummary:
    dd = max_drawdown(equity_curve)
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    return PerformanceSummary(
        total_return=total_return,
        cagr=cagr(equity_curve, periods_per_year),
        annualized_vol=annualized_volatility(returns, periods_per_year),
        sharpe=sharpe_ratio(returns, risk_free, periods_per_year),
        max_drawdown=dd.max_drawdown,
        turnover=float(turnover.mean()) if turnover is not None else float("nan"),
    )
