import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statarb import data, cointegration


def test_synthetic_pair_is_cointegrated():
    prices = data.synthetic_cointegrated_pair(n=800, alpha=5.0, beta=1.5, phi=0.85, seed=42)
    result = cointegration.test_cointegration(prices["A"], prices["B"])

    assert result.is_cointegrated
    assert result.adf.pvalue < 0.05
    assert abs(result.hedge.beta - 1.5) < 0.1
    assert abs(result.hedge.alpha - 5.0) < 3.0


def test_random_walk_pair_is_not_cointegrated():
    prices_a = data.synthetic_cointegrated_pair(n=500, phi=0.0, noise_std=0.01, seed=1)
    prices_b = data.synthetic_cointegrated_pair(n=500, phi=0.0, noise_std=0.01, seed=2)

    result = cointegration.test_cointegration(prices_a["B"], prices_b["B"])

    assert not result.is_cointegrated


def test_half_life_positive_for_mean_reverting_spread():
    prices = data.synthetic_cointegrated_pair(n=1000, phi=0.8, seed=7)
    result = cointegration.test_cointegration(prices["A"], prices["B"])

    hl = cointegration.half_life(result.spread)

    assert hl > 0
    assert hl < 50
