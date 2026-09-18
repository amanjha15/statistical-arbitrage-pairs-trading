import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statarb import metrics


def test_max_drawdown_matches_hand_computed_example():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    equity = pd.Series([100, 120, 110, 130, 90], index=idx, dtype=float)

    result = metrics.max_drawdown(equity)

    assert abs(result.max_drawdown - 0.3077) < 1e-3
    assert result.peak_date == idx[3]
    assert result.trough_date == idx[4]


def test_sharpe_ratio_zero_for_zero_volatility():
    returns = pd.Series([0.001] * 10)
    sharpe = metrics.sharpe_ratio(returns)
    assert sharpe == 0.0


def test_sharpe_ratio_positive_for_positive_mean_returns():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, size=500))
    sharpe = metrics.sharpe_ratio(returns)
    assert sharpe > 0


def test_cagr_matches_simple_growth_example():
    idx = pd.date_range("2020-01-01", periods=253, freq="D")
    equity = pd.Series(np.linspace(100_000, 110_000, 253), index=idx)

    result = metrics.cagr(equity, periods_per_year=252)

    assert abs(result - 0.10) < 0.01
