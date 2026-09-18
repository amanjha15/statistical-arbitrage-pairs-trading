"""Out-of-sample and walk-forward validation (notes section 21).

Splits history into a development/training window and one or more unseen
test windows, freezing strategy rules (thresholds, sizing) after the
training step and only ever evaluating them on the following, not-yet-seen
period.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import cointegration, signals, backtest, metrics


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WalkForwardFoldResult:
    window: WalkForwardWindow
    hedge: cointegration.HedgeRatio
    adf_pvalue: float
    is_cointegrated: bool
    backtest: backtest.BacktestResult
    performance: metrics.PerformanceSummary


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFoldResult] = field(default_factory=list)
    combined_equity_curve: pd.Series | None = None


def generate_windows(
    index: pd.DatetimeIndex,
    train_periods: int,
    test_periods: int,
    step_periods: int | None = None,
) -> list[WalkForwardWindow]:
    """Non-overlapping (or rolling, if step < test_periods) train/test windows."""
    if step_periods is None:
        step_periods = test_periods

    windows = []
    start = 0
    n = len(index)
    while start + train_periods + test_periods <= n:
        train_start = index[start]
        train_end = index[start + train_periods - 1]
        test_start = index[start + train_periods]
        test_end = index[start + train_periods + test_periods - 1]
        windows.append(WalkForwardWindow(train_start, train_end, test_start, test_end))
        start += step_periods

    return windows


def run_walk_forward(
    price_a: pd.Series,
    price_b: pd.Series,
    train_periods: int,
    test_periods: int,
    step_periods: int | None = None,
    z_window: int = 20,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float | None = 3.5,
    cost_bps: float = 5.0,
    initial_capital: float = 100_000.0,
) -> WalkForwardResult:
    """Estimate the hedge ratio on each training window, then trade the
    following test window with those frozen parameters, re-estimating the
    z-score's rolling mean/std within the test window (still no look-ahead,
    since z_window only looks backward from each point).
    """
    windows = generate_windows(price_a.index, train_periods, test_periods, step_periods)
    result = WalkForwardResult()

    equity_pieces = []
    running_capital = initial_capital

    for window in windows:
        train_a = price_a.loc[window.train_start:window.train_end]
        train_b = price_b.loc[window.train_start:window.train_end]

        coint_result = cointegration.test_cointegration(train_a, train_b)
        hedge = coint_result.hedge

        test_a = price_a.loc[window.test_start:window.test_end]
        test_b = price_b.loc[window.test_start:window.test_end]

        test_spread = cointegration.compute_spread(test_a, test_b, hedge)
        mu = test_spread.rolling(z_window).mean()
        sigma = test_spread.rolling(z_window).std()
        zscore = (test_spread - mu) / sigma

        position = signals.generate_signals(zscore, entry=entry, exit=exit, stop=stop)
        beta_series = pd.Series(hedge.beta, index=test_a.index)

        bt = backtest.run_backtest(
            test_a,
            test_b,
            beta_series,
            test_spread,
            position,
            initial_capital=running_capital,
            cost_bps=cost_bps,
        )
        perf = metrics.summarize_performance(bt.equity_curve, bt.returns, bt.turnover)

        result.folds.append(
            WalkForwardFoldResult(
                window=window,
                hedge=hedge,
                adf_pvalue=coint_result.adf.pvalue,
                is_cointegrated=coint_result.is_cointegrated,
                backtest=bt,
                performance=perf,
            )
        )

        equity_pieces.append(bt.equity_curve)
        running_capital = float(bt.equity_curve.iloc[-1])

    if equity_pieces:
        result.combined_equity_curve = pd.concat(equity_pieces)

    return result
