"""
Example 3: Risk Analysis and Monte Carlo Simulation

Performs comprehensive risk analysis including Monte Carlo simulation,
Value-at-Risk, and probability of ruin calculations.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from python.backtest.monte_carlo import MonteCarloAnalyzer


def main():
    """Run risk analysis example."""

    print("=" * 70)
    print("LATENCYARB: Risk Analysis Example")
    print("=" * 70)
    print()

    # Generate sample returns
    print("[*] Generating sample strategy returns...")
    np.random.seed(42)

    n_days = 252
    mu = 0.0005  # 0.05% daily return
    sigma = 0.015  # 1.5% daily volatility
    returns = np.random.normal(mu, sigma, n_days)

    print(f"    - Sample size: {len(returns)} days")
    print(f"    - Mean return: {np.mean(returns):.4%}")
    print(f"    - Volatility: {np.std(returns):.4%}")
    print()

    # Monte Carlo analysis
    print("[*] Running Monte Carlo simulation (1000 paths)...")
    mc_result = MonteCarloAnalyzer.analyze_strategy(
        returns,
        n_simulations=1000,
        confidence_intervals=0.95,
        random_state=42,
    )

    print("    - Simulation complete")
    print()

    # Display results
    print("[*] Risk Analysis Results:")
    print("-" * 70)

    results_dict = MonteCarloAnalyzer.format_results(mc_result)
    for key, value in results_dict.items():
        print(f"{key:.<45} {value:>20}")

    print()

    # Detailed metrics
    print("[*] Detailed Metrics:")
    print("-" * 70)

    print(f"Sharpe Ratio Distribution:")
    print(f"    - Mean: {mc_result.sharpe_mean:.3f}")
    print(f"    - Std Dev: {mc_result.sharpe_std:.3f}")
    print(f"    - 95% CI: [{mc_result.sharpe_ci_lower:.3f}, {mc_result.sharpe_ci_upper:.3f}]")

    print(f"\nMax Drawdown Distribution:")
    print(f"    - Mean: {mc_result.max_drawdown_mean:.2%}")
    print(f"    - 95% CI: [{mc_result.max_drawdown_ci_lower:.2%}, {mc_result.max_drawdown_ci_upper:.2%}]")

    print(f"\nRuin Analysis:")
    print(f"    - Probability of Ruin (25% loss): {mc_result.probability_ruin:.2%}")
    print(f"    - Expected Shortfall (CVaR): {mc_result.expected_shortfall:.2%}")

    print()

    # Bootstrap analysis
    print("[*] Bootstrap Analysis of Return Paths:")
    print("-" * 70)

    paths = MonteCarloAnalyzer.bootstrap_returns(returns, n_simulations=100)
    sharpes = MonteCarloAnalyzer.compute_sharpe_distribution(paths)
    drawdowns = MonteCarloAnalyzer.compute_drawdown_distribution(paths)

    print(f"Sharpe Ratio Statistics:")
    print(f"    - Min: {np.min(sharpes):.3f}")
    print(f"    - Median: {np.median(sharpes):.3f}")
    print(f"    - Max: {np.max(sharpes):.3f}")
    print(f"    - Std Dev: {np.std(sharpes):.3f}")

    print(f"\nMax Drawdown Statistics:")
    print(f"    - Min: {np.min(drawdowns):.2%}")
    print(f"    - Median: {np.median(drawdowns):.2%}")
    print(f"    - Max: {np.max(drawdowns):.2%}")
    print(f"    - Std Dev: {np.std(drawdowns):.2%}")

    print()

    # Percentile analysis
    print("[*] Percentile Analysis:")
    print("-" * 70)

    percentiles = [10, 25, 50, 75, 90]
    print("Sharpe Ratio Percentiles:")
    for p in percentiles:
        value = np.percentile(sharpes, p)
        print(f"    - {p:>2}th percentile: {value:.3f}")

    print("\nMax Drawdown Percentiles:")
    for p in percentiles:
        value = np.percentile(drawdowns, p)
        print(f"    - {p:>2}th percentile: {value:.2%}")

    print()
    print("[✓] Risk analysis complete!")
    print()


if __name__ == "__main__":
    main()
