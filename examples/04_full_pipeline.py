"""
Example 4: Complete Pipeline

End-to-end workflow from pair selection through backtesting to analysis.
"""

import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def main():
    """Run complete pipeline."""

    print("=" * 70)
    print("LATENCYARB: Complete Pipeline Example")
    print("=" * 70)
    print()

    # Step 1: Pair Selection
    print("[STEP 1/4] Pair Selection")
    print("-" * 70)

    from python.research.pair_selection import PairSelector
    from python.backtest.data_handler import DataHandler

    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    print(f"Screening universe: {universe}")

    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        data_dict = {}
        for ticker in universe:
            df = DataHandler.download_data([ticker], start_date, end_date)
            data_dict[ticker] = df

        aligned_data = DataHandler.align_data(data_dict)
        pairs = PairSelector.screen_universe(aligned_data, min_score=0.2, max_pairs=5)

        if pairs:
            print(f"Found {len(pairs)} candidate pairs")
            selected_pair = pairs[0]
            print(f"Selected: {selected_pair.symbol1} / {selected_pair.symbol2}")
            print(f"  - Score: {selected_pair.total_score:.4f}")
            print(f"  - P-value: {selected_pair.cointegration_pvalue:.4f}")
            print(f"  - Half-life: {selected_pair.half_life:.1f} periods")
        else:
            print("No pairs found, using synthetic pair")
            selected_pair = type('PairScore', (), {
                'symbol1': 'SYNTHETIC_1',
                'symbol2': 'SYNTHETIC_2',
            })()

    except Exception as e:
        print(f"Data download failed: {e}")
        print("Using synthetic data for demonstration...")
        selected_pair = type('PairScore', (), {
            'symbol1': 'SYNTHETIC_1',
            'symbol2': 'SYNTHETIC_2',
        })()
        aligned_data = pd.DataFrame({
            'SYNTHETIC_1': 100 + np.cumsum(np.random.randn(252) * 1),
            'SYNTHETIC_2': 100 + np.cumsum(np.random.randn(252) * 1),
        })

    print()

    # Step 2: Strategy Setup
    print("[STEP 2/4] Strategy Setup")
    print("-" * 70)

    from python.strategy.pairs_strategy import PairsStrategy

    strategy = PairsStrategy(
        pair_id="DEMO-PAIR",
        symbol1=selected_pair.symbol1,
        symbol2=selected_pair.symbol2,
        initial_capital=100_000,
        entry_z_threshold=2.0,
        exit_z_threshold=0.5,
        lookback=20,
    )

    print(f"Strategy initialized:")
    print(f"  - Pair: {selected_pair.symbol1} / {selected_pair.symbol2}")
    print(f"  - Capital: $100,000")
    print(f"  - Entry threshold: 2.0σ")
    print(f"  - Exit threshold: 0.5σ")
    print()

    # Step 3: Backtesting
    print("[STEP 3/4] Backtesting")
    print("-" * 70)

    try:
        prices1 = aligned_data[selected_pair.symbol1].values
        prices2 = aligned_data[selected_pair.symbol2].values
    except:
        n = 252
        prices1 = 100 + np.cumsum(np.random.randn(n) * 1)
        prices2 = 100 + np.cumsum(np.random.randn(n) * 1)

    equity_curve = []
    signals = []
    lookback = 20

    for i in range(lookback, len(prices1)):
        price1 = prices1[i]
        price2 = prices2[i]

        # Update strategy
        signal = strategy.update(price1, price2, timestamp=i)

        if signal:
            signals.append(signal)

        # Mark-to-market
        pnl = strategy.get_current_pnl(price1, price2)
        current_capital = 100_000 + pnl
        equity_curve.append(current_capital)

    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    total_return = (equity[-1] - 100_000) / 100_000
    print(f"Backtest complete:")
    print(f"  - Bars processed: {len(prices1) - lookback}")
    print(f"  - Signals generated: {len(signals)}")
    print(f"  - Final equity: ${equity[-1]:,.0f}")
    print(f"  - Total return: {total_return:.2%}")

    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = (np.mean(returns) * 252 - 0.04) / (np.std(returns) * np.sqrt(252))
        print(f"  - Sharpe ratio: {sharpe:.2f}")

        running_max = np.maximum.accumulate(equity)
        max_dd = np.min((equity - running_max) / running_max)
        print(f"  - Max drawdown: {max_dd:.2%}")

    print()

    # Step 4: Risk Analysis
    print("[STEP 4/4] Risk Analysis")
    print("-" * 70)

    from python.backtest.monte_carlo import MonteCarloAnalyzer

    if len(returns) > 20:
        mc_result = MonteCarloAnalyzer.analyze_strategy(
            returns,
            n_simulations=100,
            random_state=42,
        )

        print("Monte Carlo Analysis (100 simulations):")
        print(f"  - Sharpe ratio (mean): {mc_result.sharpe_mean:.3f}")
        print(f"  - Sharpe ratio (95% CI): [{mc_result.sharpe_ci_lower:.3f}, {mc_result.sharpe_ci_upper:.3f}]")
        print(f"  - Max drawdown (mean): {mc_result.max_drawdown_mean:.2%}")
        print(f"  - Probability of ruin: {mc_result.probability_ruin:.2%}")
        print(f"  - Expected shortfall: {mc_result.expected_shortfall:.2%}")
    else:
        print("Insufficient data for Monte Carlo (need >20 returns)")

    print()

    # Summary
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Optimize entry/exit thresholds")
    print("  2. Add position sizing based on Kelly criterion")
    print("  3. Implement adaptive Kalman filter hedge ratio")
    print("  4. Deploy on live data with strict risk controls")
    print()


if __name__ == "__main__":
    main()
