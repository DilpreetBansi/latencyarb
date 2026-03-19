"""
Monte Carlo simulation for risk analysis.

Bootstraps returns to estimate confidence intervals and probability of ruin.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, NamedTuple


class MonteCarloResult(NamedTuple):
    """Results from Monte Carlo simulation."""
    sharpe_mean: float
    sharpe_std: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    max_drawdown_mean: float
    max_drawdown_ci_lower: float
    max_drawdown_ci_upper: float
    probability_ruin: float
    expected_shortfall: float


class MonteCarloAnalyzer:
    """
    Monte Carlo analysis of strategy risk.

    Uses bootstrap resampling to estimate:
    - Confidence intervals on Sharpe ratio
    - Distribution of max drawdowns
    - Probability of ruin
    - Expected shortfall (CVaR)
    """

    @staticmethod
    def bootstrap_returns(
        returns: np.ndarray,
        n_simulations: int = 1000,
        path_length: int = None,
        random_state: int = None,
    ) -> np.ndarray:
        """
        Bootstrap return sequences.

        Samples returns with replacement to generate alternative paths.

        Parameters:
        -----------
        returns : array-like
            Historical returns
        n_simulations : int
            Number of simulated paths
        path_length : int
            Length of each path (default: same as input)
        random_state : int
            Random seed

        Returns:
        --------
        Array of shape (n_simulations, path_length)
        """
        if random_state is not None:
            np.random.seed(random_state)

        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]

        if path_length is None:
            path_length = len(returns)

        simulated_paths = np.zeros((n_simulations, path_length))

        for i in range(n_simulations):
            # Random sampling with replacement
            indices = np.random.choice(len(returns), size=path_length, replace=True)
            simulated_paths[i] = returns[indices]

        return simulated_paths

    @staticmethod
    def compute_sharpe_distribution(
        paths: np.ndarray,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.04,
    ) -> np.ndarray:
        """
        Compute Sharpe ratio for each simulated path.

        Parameters:
        -----------
        paths : ndarray
            Simulated return paths
        periods_per_year : int
            Trading periods per year
        risk_free_rate : float
            Annual risk-free rate

        Returns:
        --------
        Array of Sharpe ratios
        """
        sharpes = np.zeros(len(paths))

        for i, path in enumerate(paths):
            mean_ret = np.mean(path)
            std_ret = np.std(path)

            if std_ret > 0:
                annual_mean = mean_ret * periods_per_year
                annual_std = std_ret * np.sqrt(periods_per_year)
                sharpes[i] = (annual_mean - risk_free_rate) / annual_std
            else:
                sharpes[i] = 0.0

        return sharpes

    @staticmethod
    def compute_drawdown_distribution(paths: np.ndarray) -> np.ndarray:
        """
        Compute maximum drawdown for each simulated path.

        Parameters:
        -----------
        paths : ndarray
            Simulated return paths

        Returns:
        --------
        Array of maximum drawdowns
        """
        max_drawdowns = np.zeros(len(paths))

        for i, path in enumerate(paths):
            # Convert returns to equity curve
            equity = np.cumprod(1 + path)
            running_max = np.maximum.accumulate(equity)
            drawdown = 1 - equity / running_max

            max_drawdowns[i] = np.max(drawdown)

        return max_drawdowns

    @staticmethod
    def compute_probability_of_ruin(
        paths: np.ndarray,
        initial_capital: float = 100,
        ruin_level: float = 0.25,  # 25% loss = ruin
    ) -> float:
        """
        Estimate probability of portfolio decline exceeding threshold.

        Parameters:
        -----------
        paths : ndarray
            Simulated return paths
        initial_capital : float
            Starting capital (for scaling)
        ruin_level : float
            Drawdown threshold to consider as ruin

        Returns:
        --------
        Probability of ruin (0 to 1)
        """
        ruined = 0

        for path in paths:
            # Equity curve from returns
            equity = initial_capital * np.cumprod(1 + path)
            peak = np.maximum.accumulate(equity)
            drawdown = 1 - equity / peak

            if np.max(drawdown) > ruin_level:
                ruined += 1

        return ruined / len(paths)

    @staticmethod
    def compute_expected_shortfall(
        returns: np.ndarray,
        confidence_level: float = 0.95,
    ) -> float:
        """
        Compute expected shortfall (CVaR).

        Average of returns worse than the VaR level.

        Parameters:
        -----------
        returns : array-like
            Return distribution
        confidence_level : float
            Confidence level

        Returns:
        --------
        Expected shortfall as fraction
        """
        returns = np.asarray(returns, dtype=np.float64)

        # VaR at confidence level
        var = np.percentile(returns, (1 - confidence_level) * 100)

        # CVaR: mean of returns worse than VaR
        tail_returns = returns[returns <= var]

        if len(tail_returns) > 0:
            return np.mean(tail_returns)
        else:
            return 0.0

    @staticmethod
    def analyze_strategy(
        returns: np.ndarray,
        n_simulations: int = 1000,
        confidence_intervals: float = 0.95,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.04,
        random_state: int = None,
    ) -> MonteCarloResult:
        """
        Perform complete Monte Carlo analysis.

        Parameters:
        -----------
        returns : array-like
            Historical returns
        n_simulations : int
            Number of simulations
        confidence_intervals : float
            Confidence level for CI (e.g., 0.95 = 95%)
        periods_per_year : int
            Trading periods per year
        risk_free_rate : float
            Risk-free rate
        random_state : int
            Random seed

        Returns:
        --------
        MonteCarloResult with all analyses
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]

        # Bootstrap paths
        paths = MonteCarloAnalyzer.bootstrap_returns(
            returns,
            n_simulations=n_simulations,
            random_state=random_state
        )

        # Sharpe ratio distribution
        sharpes = MonteCarloAnalyzer.compute_sharpe_distribution(
            paths,
            periods_per_year,
            risk_free_rate
        )

        sharpe_mean = np.mean(sharpes)
        sharpe_std = np.std(sharpes)

        # Confidence intervals
        alpha = (1 - confidence_intervals) / 2
        sharpe_ci_lower = np.percentile(sharpes, alpha * 100)
        sharpe_ci_upper = np.percentile(sharpes, (1 - alpha) * 100)

        # Max drawdown distribution
        max_dds = MonteCarloAnalyzer.compute_drawdown_distribution(paths)

        max_dd_mean = np.mean(max_dds)
        max_dd_ci_lower = np.percentile(max_dds, alpha * 100)
        max_dd_ci_upper = np.percentile(max_dds, (1 - alpha) * 100)

        # Probability of ruin
        prob_ruin = MonteCarloAnalyzer.compute_probability_of_ruin(paths)

        # Expected shortfall
        all_returns = paths.flatten()
        es = MonteCarloAnalyzer.compute_expected_shortfall(all_returns)

        return MonteCarloResult(
            sharpe_mean=sharpe_mean,
            sharpe_std=sharpe_std,
            sharpe_ci_lower=sharpe_ci_lower,
            sharpe_ci_upper=sharpe_ci_upper,
            max_drawdown_mean=max_dd_mean,
            max_drawdown_ci_lower=max_dd_ci_lower,
            max_drawdown_ci_upper=max_dd_ci_upper,
            probability_ruin=prob_ruin,
            expected_shortfall=es,
        )

    @staticmethod
    def format_results(result: MonteCarloResult) -> Dict:
        """Format Monte Carlo results for display."""
        return {
            "Sharpe Ratio": f"{result.sharpe_mean:.3f} +/- {result.sharpe_std:.3f}",
            "Sharpe 95% CI": f"[{result.sharpe_ci_lower:.3f}, {result.sharpe_ci_upper:.3f}]",
            "Max Drawdown": f"{result.max_drawdown_mean:.1%} +/- {result.max_drawdown_ci_upper - result.max_drawdown_mean:.1%}",
            "Max Drawdown 95% CI": f"[{result.max_drawdown_ci_lower:.1%}, {result.max_drawdown_ci_upper:.1%}]",
            "Probability of Ruin": f"{result.probability_ruin:.1%}",
            "Expected Shortfall": f"{result.expected_shortfall:.1%}",
        }
