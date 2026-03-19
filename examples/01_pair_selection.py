"""
Example 1: Pair Selection and Screening

Screens a universe of stocks to identify cointegrated pairs
for statistical arbitrage trading.
"""

import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from python.backtest.data_handler import DataHandler
from python.research.pair_selection import PairSelector
from python.research.cointegration import EngleGrangerTest, JohansenTest
from python.research.mean_reversion import OrnsteinUhlenbeckEstimator


def main():
    """Run pair selection example."""

    print("=" * 70)
    print("LATENCYARB: Pair Selection Example")
    print("=" * 70)
    print()

    # Define universe
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TESLA', 'NVDA', 'AMD']

    # Download historical data
    print(f"[*] Downloading data for {len(universe)} stocks...")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        data_dict = {}
        for ticker in universe:
            print(f"    - {ticker}...", end="", flush=True)
            df = DataHandler.download_data([ticker], start_date, end_date, interval="1d")
            data_dict[ticker] = df
            print(" OK")

        # Align data
        print("\n[*] Aligning time series...")
        aligned_data = DataHandler.align_data(data_dict)
        print(f"    - Final dataset: {len(aligned_data)} rows x {len(aligned_data.columns)} columns")

    except Exception as e:
        print(f"\n    ERROR: {e}")
        print("    Using synthetic data for demonstration...")
        # Generate synthetic data
        n_days = 252
        aligned_data = pd.DataFrame({
            ticker: 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))
            for ticker in universe
        })

    print()

    # Screen for pairs
    print("[*] Screening universe for cointegrated pairs...")
    pairs = PairSelector.screen_universe(aligned_data, min_score=0.3, max_pairs=20)

    print(f"\n    Found {len(pairs)} high-quality pairs:")
    print()

    # Create and display report
    report = PairSelector.generate_report(pairs[:10])
    print(report.to_string(index=False))

    print()

    # Analyze top pair
    if pairs:
        top_pair = pairs[0]
        print(f"\n[*] Detailed Analysis: {top_pair.symbol1} / {top_pair.symbol2}")
        print("-" * 70)

        # Get price data
        prices1 = aligned_data[top_pair.symbol1].values
        prices2 = aligned_data[top_pair.symbol2].values

        # Engle-Granger test
        print("\n1. Cointegration Test (Engle-Granger):")
        coint_result = EngleGrangerTest.test(prices1, prices2)
        print(f"   - Test Statistic: {coint_result.test_statistic:.4f}")
        print(f"   - P-Value: {coint_result.p_value:.4f}")
        print(f"   - Critical Values (5%): {coint_result.critical_values['5%']:.4f}")
        print(f"   - Cointegrated: {coint_result.is_cointegrated}")

        if coint_result.cointegration_vector is not None:
            print(f"   - Cointegration Vector: {coint_result.cointegration_vector}")

        # Mean reversion analysis
        print("\n2. Mean Reversion Analysis:")
        spread = prices2 - top_pair.cointegration_pvalue * prices1
        ou_params = OrnsteinUhlenbeckEstimator.estimate_regression(spread)

        print(f"   - Long-term Mean: {ou_params.mu:.4f}")
        print(f"   - Mean Reversion Speed (θ): {ou_params.theta:.6f}")
        print(f"   - Volatility (σ): {ou_params.sigma:.6f}")
        print(f"   - Half-life: {ou_params.half_life:.2f} periods")

        # Hurst exponent
        hurst = PairSelector.hurst_exponent(spread)
        print(f"\n3. Hurst Exponent: {hurst:.3f}")
        print(f"   - Interpretation: ", end="")
        if hurst < 0.5:
            print("Mean-reverting (GOOD for pairs trading)")
        elif hurst > 0.5:
            print("Trending (BAD for pairs trading)")
        else:
            print("Random walk")

        # Summary
        print(f"\n4. Overall Score: {top_pair.total_score:.4f}")
        print(f"   - Cointegration Score: {max(0, 1 - top_pair.cointegration_pvalue) * 0.4:.4f}")

        # Calculate spread statistics
        print(f"\n5. Spread Statistics:")
        print(f"   - Mean: {np.mean(spread):.6f}")
        print(f"   - Std Dev: {np.std(spread):.6f}")
        print(f"   - Min: {np.min(spread):.6f}")
        print(f"   - Max: {np.max(spread):.6f}")

        print()

    print("[✓] Pair selection complete!")
    print()


if __name__ == "__main__":
    main()
