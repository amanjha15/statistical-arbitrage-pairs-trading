# Statistical Arbitrage — Pairs Trading

A research/backtesting pipeline for cointegration-based pairs trading:
find two assets with a stable long-run relationship, trade deviations
from that relationship, and validate the strategy out-of-sample.

## Pipeline

```
Price data (A, B)
  -> OLS hedge ratio:        A_t = alpha + beta * B_t + eps_t
  -> Spread:                 S_t = A_t - alpha - beta * B_t
  -> ADF / Engle-Granger cointegration test on S_t
  -> Half-life of mean reversion
  -> Rolling beta / z-score (no look-ahead: uses only data through t-1)
  -> Entry / exit / stop-loss signals on the z-score
  -> Backtest: inverse-vol position sizing, transaction costs, equity curve
  -> Metrics: total return, CAGR, Sharpe ratio, max drawdown, turnover
  -> Walk-forward validation: re-estimate on train windows, trade unseen test windows
```

## Project layout

```
src/statarb/
  data.py          price loading (yfinance) + synthetic cointegrated pair generator
  cointegration.py OLS hedge ratio, spread, ADF test, Engle-Granger, half-life
  signals.py       rolling hedge ratio/z-score, entry/exit/stop signal generation
  backtest.py       cost-aware backtest engine with inverse-volatility position sizing
  metrics.py         Sharpe ratio, max drawdown, CAGR, performance summary
  walkforward.py     train/test window generation and walk-forward backtest loop
scripts/
  run_backtest.py    CLI: download real tickers, run the full pipeline, save charts
tests/
  test_*.py           unit tests against a deterministic synthetic cointegrated pair
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the example backtest

```bash
python scripts/run_backtest.py --tickers KO PEP --start 2018-01-01 --end 2024-01-01
```

Prints the cointegration test, half-life, and backtest performance summary,
and saves `equity_curve.png` with price/z-score/equity charts.

Key flags: `--z-window` (rolling beta/z-score window), `--entry`/`--exit`/`--stop`
(z-score thresholds), `--cost-bps` (transaction cost), `--capital`.

## Run tests

```bash
pytest tests/ -v
```

Tests run entirely offline against a synthetic cointegrated pair
(`statarb.data.synthetic_cointegrated_pair`), so they don't depend on
network access or Yahoo Finance availability.

## Methodology

- **No look-ahead**: rolling beta, spread mean/std are computed from a
  trailing window and shifted by one period before being used to size or
  signal a trade at time *t*.
- **Position sizing**: inverse to rolling spread volatility, targeting a
  fraction of *current equity* per trade (not a fixed dollar amount), and
  hard-capped at a maximum gross leverage multiple of equity. The cap is
  computed sequentially against the realized equity path, since a
  leverage limit checked only against starting capital would silently
  drift as equity compounds.
- **Costs**: transaction costs are charged in basis points of traded gross
  notional whenever the position size changes, not just on binary entry/exit.
- **Walk-forward**: the hedge ratio and cointegration test are estimated
  only on the training window of each fold and then frozen while trading
  the following, unseen test window.
