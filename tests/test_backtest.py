import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statarb import data, cointegration, signals, backtest


def test_backtest_runs_end_to_end_on_synthetic_pair():
    prices = data.synthetic_cointegrated_pair(n=600, phi=0.85, seed=11)
    est = signals.build_rolling_estimate(prices["A"], prices["B"], window=40)
    position = signals.generate_signals(est.zscore, entry=1.5, exit=0.3, stop=3.5)

    result = backtest.run_backtest(
        prices["A"], prices["B"], est.beta, est.spread, position,
        initial_capital=100_000.0, cost_bps=5.0,
    )

    assert len(result.equity_curve) == len(prices)
    assert result.equity_curve.iloc[0] > 0
    assert not result.equity_curve.isna().any()


def test_zero_position_produces_zero_pnl_and_flat_equity():
    prices = data.synthetic_cointegrated_pair(n=200, seed=5)
    hedge = cointegration.estimate_hedge_ratio(prices["A"], prices["B"])
    spread = cointegration.compute_spread(prices["A"], prices["B"], hedge)
    beta_series = pd.Series(hedge.beta, index=prices.index)
    flat_position = pd.Series(0, index=prices.index)

    result = backtest.run_backtest(
        prices["A"], prices["B"], beta_series, spread, flat_position,
        initial_capital=50_000.0,
    )

    assert (result.pnl == 0).all()
    assert (result.equity_curve == 50_000.0).all()


def test_higher_transaction_costs_reduce_final_equity():
    prices = data.synthetic_cointegrated_pair(n=600, phi=0.8, seed=9)
    est = signals.build_rolling_estimate(prices["A"], prices["B"], window=40)
    position = signals.generate_signals(est.zscore, entry=1.5, exit=0.3, stop=3.5)

    cheap = backtest.run_backtest(
        prices["A"], prices["B"], est.beta, est.spread, position, cost_bps=1.0
    )
    expensive = backtest.run_backtest(
        prices["A"], prices["B"], est.beta, est.spread, position, cost_bps=50.0
    )

    assert expensive.equity_curve.iloc[-1] <= cheap.equity_curve.iloc[-1]
