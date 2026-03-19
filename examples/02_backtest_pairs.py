"""
Example 2: Pairs Trading Backtest

Backtests a pairs trading strategy on historical data.
"""

import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from python.backtest.data_handler import DataHandler
from python.backtest.engine import BacktestEngine
from python.backtest.metrics import MetricsCalculator
from python.strategy.pairs_strategy import PairsStrategy


def main():
    """Run pairs trading backtest."""

    print("=" * 70)
    print("LATENCYARB: Pairs Trading Backtest Example")
    print("=" * 70)
    print()

    # Configuration
    symbol1 = 'AAPL'
    symbol2 = 'MSFT'
    initial_capital = 100_000
    entry_threshold = 2.0
    exit_threshold = 0.5

    print(f"[*] Configuration:")
    print(f"    - Pair: {symbol1} / {symbol2}")
    print(f"    - Capital: ${initial_capital:,.0f}")
    print(f"    - Entry Z-score: {entry_threshold}")
    print(f"    - Exit Z-score: {exit_threshold}")
    print()

    # Download data
    print("[*] Downloading data...")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        data1 = DataHandler.download_data([symbol1], start_date, end_date)['Close'].values
        data2 = DataHandler.download_data([symbol2], start_date, end_date)['Close'].values

        # Align lengths
        min_len = min(len(data1), len(data2))
        data1 = data1[-min_len:]
        data2 = data2[-min_len:]

        print(f"    - {symbol1}: {len(data1)} bars")
        print(f"    - {symbol2}: {len(data2)} bars")

    except Exception as e:
        print(f"    ERROR: {e}")
        print("    Using synthetic data for demonstration...")

        # Synthetic data
        n = 252
        prices1 = 100 + np.cumsum(np.random.randn(n) * 1)
        prices2 = 100 + np.cumsum(np.random.randn(n) * 1)
        data1 = prices1
        data2 = prices2

    print()

    # Initialize strategy
    print("[*] Initializing strategy...")
    strategy = PairsStrategy(
        pair_id="AAPL-MSFT",
        symbol1=symbol1,
        symbol2=symbol2,
        initial_capital=initial_capital,
        entry_z_threshold=entry_threshold,
        exit_z_threshold=exit_threshold,
    )

    strategy.risk_manager.initialize(initial_capital)
    print("    - Strategy initialized")
    print()

    # Run backtest
    print("[*] Running backtest...")
    equity_curve = []
    trades_log = []
    signals_log = []

    for i in range(lookback := 20, len(data1)):
        price1 = data1[i]
        price2 = data2[i]

        # Update strategy
        signal = strategy.update(price1, price2, timestamp=i)

        if signal:
            signals_log.append({
                'timestamp': i,
                'signal': signal.signal_type,
                'z_score': signal.z_score,
            })

        # Mark-to-market
        pnl = strategy.get_current_pnl(price1, price2)
        current_capital = initial_capital + pnl

        strategy.risk_manager.update_capital(current_capital)
        equity_curve.append(current_capital)

    print(f"    - Backtest complete: {len(data1) - lookback} bars processed")
    print()

    # Calculate metrics
    print("[*] Performance Analysis:")
    print("-" * 70)

    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    total_return = (equity[-1] - initial_capital) / initial_capital
    annual_return = total_return * (252 / len(returns)) if len(returns) > 0 else 0

    print(f"Total Return: {total_return:.2%}")
    print(f"Annual Return: {annual_return:.2%}")

    if len(returns) > 0:
        sharpe = (np.mean(returns) * 252 - 0.04) / (np.std(returns) * np.sqrt(252))
        print(f"Sharpe Ratio: {sharpe:.2f}")

        # Drawdown
        running_max = np.maximum.accumulate(equity)
        max_dd = np.min((equity - running_max) / running_max)
        print(f"Max Drawdown: {max_dd:.2%}")

    print(f"Final Equity: ${equity[-1]:,.0f}")
    print(f"Trades Generated: {len(signals_log)}")
    print()

    # Position metrics
    print("[*] Current Position:")
    metrics = strategy.get_position_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"    - {key}: {value:.4f}")
        else:
            print(f"    - {key}: {value}")

    print()
    print("[✓] Backtest complete!")
    print()


if __name__ == "__main__":
    main()
