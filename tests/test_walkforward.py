import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statarb import data, walkforward


def test_generate_windows_covers_expected_span():
    prices = data.synthetic_cointegrated_pair(n=1000, seed=1)
    windows = walkforward.generate_windows(prices.index, train_periods=300, test_periods=100)

    assert len(windows) > 0
    for w in windows:
        assert w.train_start < w.train_end < w.test_start <= w.test_end


def test_run_walk_forward_produces_one_fold_per_window():
    prices = data.synthetic_cointegrated_pair(n=1200, phi=0.85, seed=21)
    result = walkforward.run_walk_forward(
        prices["A"], prices["B"], train_periods=400, test_periods=150, z_window=30
    )

    windows = walkforward.generate_windows(prices.index, train_periods=400, test_periods=150)
    assert len(result.folds) == len(windows)
    assert result.combined_equity_curve is not None
    assert len(result.combined_equity_curve) == sum(150 for _ in windows)


def test_walk_forward_equity_chains_across_folds():
    prices = data.synthetic_cointegrated_pair(n=1200, phi=0.85, seed=21)
    result = walkforward.run_walk_forward(
        prices["A"], prices["B"], train_periods=400, test_periods=150,
        z_window=30, initial_capital=100_000.0,
    )

    assert result.folds[0].backtest.equity_curve.iloc[0] == 100_000.0
    for prev_fold, next_fold in zip(result.folds, result.folds[1:]):
        prev_final = prev_fold.backtest.equity_curve.iloc[-1]
        next_start = next_fold.backtest.equity_curve.iloc[0]
        assert abs(next_start - prev_final) < 1e-6
