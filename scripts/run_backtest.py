"""End-to-end example: download two tickers, test cointegration, backtest the spread.

Usage:
    python scripts/run_backtest.py --tickers KO PEP --start 2018-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt

from statarb import data, cointegration, signals, backtest, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 6 pairs-trading pipeline on real data.")
    parser.add_argument("--tickers", nargs=2, default=["KO", "PEP"], metavar=("A", "B"))
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--z-window", type=int, default=60, help="Rolling window for beta/z-score estimation")
    parser.add_argument("--entry", type=float, default=2.0)
    parser.add_argument("--exit", type=float, default=0.5)
    parser.add_argument("--stop", type=float, default=3.5)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--no-plot", action="store_true", help="Skip showing matplotlib charts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker_a, ticker_b = args.tickers

    print(f"Downloading {ticker_a} and {ticker_b} from {args.start} to {args.end or 'today'}...")
    prices = data.download_prices([ticker_a, ticker_b], start=args.start, end=args.end)
    price_a, price_b = prices[ticker_a], prices[ticker_b]

    print("\n--- Full-sample cointegration test ---")
    coint_result = cointegration.test_cointegration(price_a, price_b)
    print(f"Hedge ratio: alpha={coint_result.hedge.alpha:.4f}, beta={coint_result.hedge.beta:.4f}")
    print(f"ADF statistic={coint_result.adf.statistic:.4f}, p-value={coint_result.adf.pvalue:.4f}")
    print(f"Engle-Granger p-value: {coint_result.engle_granger_pvalue:.4f}")
    print(f"Cointegrated at 5%: {coint_result.is_cointegrated}")

    hl = cointegration.half_life(coint_result.spread)
    print(f"Half-life of mean reversion: {hl:.2f} periods")

    print("\n--- Building rolling spread / z-score (no look-ahead) ---")
    est = signals.build_rolling_estimate(price_a, price_b, window=args.z_window)
    position = signals.generate_signals(est.zscore, entry=args.entry, exit=args.exit, stop=args.stop)
    n_trades = int((position.diff().fillna(0) != 0).sum())
    print(f"Number of position changes: {n_trades}")

    print("\n--- Backtest ---")
    bt = backtest.run_backtest(
        price_a,
        price_b,
        est.beta,
        est.spread,
        position,
        initial_capital=args.capital,
        cost_bps=args.cost_bps,
    )
    perf = metrics.summarize_performance(bt.equity_curve, bt.returns, bt.turnover)
    print(f"Total return:      {perf.total_return:.2%}")
    print(f"CAGR:              {perf.cagr:.2%}")
    print(f"Annualized vol:    {perf.annualized_vol:.2%}")
    print(f"Sharpe ratio:      {perf.sharpe:.2f}")
    print(f"Max drawdown:      {perf.max_drawdown:.2%}")
    print(f"Avg daily turnover:{perf.turnover:.4f}")

    if not args.no_plot:
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

        axes[0].plot(price_a.index, price_a, label=ticker_a)
        axes[0].plot(price_b.index, price_b, label=ticker_b)
        axes[0].set_title(f"{ticker_a} vs {ticker_b} — Prices")
        axes[0].legend()

        axes[1].plot(est.zscore.index, est.zscore, label="z-score", color="tab:purple")
        axes[1].axhline(args.entry, color="red", linestyle="--", linewidth=1)
        axes[1].axhline(-args.entry, color="red", linestyle="--", linewidth=1)
        axes[1].axhline(args.exit, color="grey", linestyle=":", linewidth=1)
        axes[1].axhline(-args.exit, color="grey", linestyle=":", linewidth=1)
        axes[1].set_title("Rolling Spread Z-Score")
        axes[1].legend()

        axes[2].plot(bt.equity_curve.index, bt.equity_curve, label="Equity", color="tab:green")
        axes[2].set_title("Equity Curve")
        axes[2].legend()

        plt.tight_layout()
        out_path = Path(__file__).resolve().parents[1] / "equity_curve.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved chart to {out_path}")


if __name__ == "__main__":
    main()
