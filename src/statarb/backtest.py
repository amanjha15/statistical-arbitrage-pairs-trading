"""Cost-aware backtest engine for a single pairs-trading spread.

Implements the pipeline stages: strict time-ordered signal -> position,
risk-based position sizing, transaction costs/slippage, and the resulting
equity curve. All position/size values used to compute P&L at time t are
taken from t-1, so no future information leaks into a historical decision
(see notes section 18-19, look-ahead bias).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    pnl: pd.Series
    costs: pd.Series
    units: pd.Series
    turnover: pd.Series


def inverse_vol_sizing(
    spread: pd.Series,
    vol_window: int,
    target_dollar_risk: float,
) -> pd.Series:
    """Risk-based sizing: size ∝ 1 / rolling spread volatility (notes section 30).

    Returns the number of "spread units" to hold when a signal is active,
    where one spread unit = long 1 share of A, short beta shares of B.
    Lagged by one period so size_t is known before t's return is realized.
    """
    vol = spread.rolling(vol_window).std().shift(1)
    size = target_dollar_risk / vol
    size = size.replace([np.inf, -np.inf], np.nan)
    size.name = "units"
    return size


def run_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    beta: pd.Series,
    spread: pd.Series,
    position: pd.Series,
    initial_capital: float = 100_000.0,
    cost_bps: float = 5.0,
    vol_window: int = 20,
    target_dollar_risk: float = 10_000.0,
) -> BacktestResult:
    """Run a dollar-notional-scaled pairs backtest.

    position: series in {-1, 0, +1}, meaning long/flat/short the spread
        (long spread = long A, short beta*B).
    beta: hedge ratio series aligned with position (may be rolling or static).
    spread: A - alpha - beta*B, aligned with position.
    cost_bps: round-trip-style transaction cost in basis points of traded
        gross notional, applied whenever the position or size changes.
    """
    idx = price_a.index
    beta = beta.reindex(idx)
    spread = spread.reindex(idx)
    position = position.reindex(idx).fillna(0)

    size = inverse_vol_sizing(spread, vol_window, target_dollar_risk)
    size = size.reindex(idx)

    signed_units = (position * size).fillna(0.0)
    signed_units_lagged = signed_units.shift(1).fillna(0.0)

    spread_change = spread.diff().fillna(0.0)
    pnl = signed_units_lagged * spread_change
    pnl.name = "pnl"

    gross_notional_per_unit = price_a + beta.abs() * price_b
    traded_units = signed_units.diff().fillna(signed_units.iloc[0] if len(signed_units) else 0.0).abs()
    costs = traded_units * gross_notional_per_unit * (cost_bps / 10_000.0)
    costs = costs.fillna(0.0)
    costs.name = "costs"

    net_pnl = pnl - costs
    equity_curve = initial_capital + net_pnl.cumsum()
    equity_curve.name = "equity"

    returns = equity_curve.pct_change().fillna(0.0)
    returns.name = "returns"

    denom = (equity_curve.shift(1) / gross_notional_per_unit.replace(0, np.nan))
    turnover = (traded_units / denom.replace(0, np.nan)).fillna(0.0)
    turnover.name = "turnover"

    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns,
        pnl=net_pnl,
        costs=costs,
        units=signed_units,
        turnover=turnover,
    )
