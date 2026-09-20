"""Cost-aware backtest engine for a single pairs-trading spread.

Implements the pipeline stages: strict time-ordered signal -> position,
risk-based position sizing under a hard leverage cap, transaction
costs/slippage, and the resulting equity curve. All position/size values
used to compute P&L at time t are taken from t-1, so no future information
leaks into a historical decision (look-ahead bias).
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


def run_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    beta: pd.Series,
    spread: pd.Series,
    position: pd.Series,
    initial_capital: float = 100_000.0,
    cost_bps: float = 5.0,
    vol_window: int = 20,
    target_risk_fraction: float = 0.02,
    max_leverage: float = 4.0,
) -> BacktestResult:
    """Run a leverage-capped, equity-relative pairs backtest.

    position: series in {-1, 0, +1}, meaning long/flat/short the spread
        (long spread = long A, short beta*B).
    beta: hedge ratio series aligned with position (may be rolling or static).
    spread: A - alpha - beta*B, aligned with position.
    cost_bps: transaction cost in basis points of traded gross notional,
        applied whenever the held size changes.

    Sizing: the number of "spread units" held (1 unit = long 1 share of A,
    short beta shares of B) targets `target_risk_fraction * equity` of
    expected daily P&L variation, based on trailing spread volatility, but
    is never allowed to push gross notional exposure (|units| * (price_a +
    |beta|*price_b)) above `max_leverage * equity`. Because the cap depends
    on the realized equity path (not just starting capital), sizing and
    equity are computed together in a single forward pass rather than
    vectorized, so each day's cap reflects gains/losses actually realized
    so far — never a future value.
    """
    idx = price_a.index
    beta = beta.reindex(idx)
    spread = spread.reindex(idx)
    position = position.reindex(idx).fillna(0)
    price_a = price_a.reindex(idx)
    price_b = price_b.reindex(idx)

    vol = spread.rolling(vol_window).std().shift(1)
    gross_notional_per_unit = price_a + beta.abs() * price_b

    equity = pd.Series(index=idx, dtype=float)
    pnl = pd.Series(index=idx, dtype=float)
    costs = pd.Series(index=idx, dtype=float)
    units = pd.Series(index=idx, dtype=float)

    prev_equity = initial_capital
    prev_units = 0.0
    prev_spread = None

    for t in idx:
        v = vol.loc[t]
        gnpu = gross_notional_per_unit.loc[t]
        target_units = 0.0

        if position.loc[t] != 0 and gnpu > 0 and not np.isnan(v) and v > 0:
            raw_units = (target_risk_fraction * prev_equity) / v
            max_units = (max_leverage * prev_equity) / gnpu
            target_units = float(position.loc[t]) * min(raw_units, max_units)

        current_spread = spread.loc[t]
        if prev_spread is None or np.isnan(current_spread) or np.isnan(prev_spread):
            spread_change = 0.0
        else:
            spread_change = current_spread - prev_spread
        period_pnl = prev_units * spread_change

        traded = abs(target_units - prev_units)
        period_cost = traded * gnpu * (cost_bps / 10_000.0) if gnpu > 0 else 0.0

        new_equity = prev_equity + period_pnl - period_cost

        pnl.loc[t] = period_pnl - period_cost
        costs.loc[t] = period_cost
        units.loc[t] = target_units
        equity.loc[t] = new_equity

        prev_equity = new_equity
        prev_units = target_units
        prev_spread = current_spread

    equity.name = "equity"
    pnl.name = "pnl"
    costs.name = "costs"
    units.name = "units"

    returns = equity.pct_change(fill_method=None).fillna(0.0)
    returns.name = "returns"

    traded_units = units.diff().fillna(units.iloc[0] if len(units) else 0.0).abs()
    prev_equity_series = equity.shift(1).fillna(initial_capital)
    turnover = (traded_units * gross_notional_per_unit / prev_equity_series).fillna(0.0)
    turnover.name = "turnover"

    return BacktestResult(
        equity_curve=equity,
        returns=returns,
        pnl=pnl,
        costs=costs,
        units=units,
        turnover=turnover,
    )
