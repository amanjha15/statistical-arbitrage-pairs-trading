"""Price data loading for pairs-trading research."""

from __future__ import annotations

import pandas as pd


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    field: str = "Close",
) -> pd.DataFrame:
    """Download adjusted daily prices for a list of tickers via yfinance.

    Returns a DataFrame indexed by date with one column per ticker,
    forward-filled and with rows containing any remaining NaNs dropped.
    """
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw[field]
    else:
        # Single ticker: yfinance returns a flat-column frame.
        prices = raw[[field]]
        prices.columns = tickers

    prices = prices.ffill().dropna()
    prices.index.name = "date"
    return prices


def synthetic_cointegrated_pair(
    n: int = 1000,
    alpha: float = 5.0,
    beta: float = 1.5,
    phi: float = 0.9,
    noise_std: float = 1.0,
    start_price: float = 100.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic cointegrated pair (B random walk, A = alpha + beta*B + stationary noise).

    Useful for deterministic, offline tests and demos of the pipeline
    without depending on network access.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    b_shocks = rng.normal(0, 1.0, size=n)
    b = start_price + np.cumsum(b_shocks)

    spread = np.zeros(n)
    eps = rng.normal(0, noise_std, size=n)
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + eps[t]

    a = alpha + beta * b + spread

    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": a, "B": b}, index=idx)
