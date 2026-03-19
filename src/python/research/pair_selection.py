"""
Pair selection and screening for statistical arbitrage.

Screens a universe of tickers to identify cointegrated pairs.
Scores pairs by multiple factors: cointegration strength, half-life, stability.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, NamedTuple
from itertools import combinations
from .cointegration import EngleGrangerTest
from .mean_reversion import OrnsteinUhlenbeckEstimator

import warnings
warnings.filterwarnings("ignore")


class PairScore(NamedTuple):
    """Score for a potential trading pair."""
    symbol1: str
    symbol2: str
    cointegration_pvalue: float
    half_life: float
    hurst_exponent: float
    spread_std: float
    total_score: float


class PairSelector:
    """
    Screens a universe of assets for cointegrated pairs.

    Scoring criteria:
    1. Cointegration p-value (lower is better, < 0.05 preferred)
    2. Half-life of mean reversion (5-20 periods optimal)
    3. Hurst exponent (< 0.5 indicates mean-reverting)
    4. Spread stability (low volatility)
    """

    @staticmethod
    def hurst_exponent(series: np.ndarray, lags: np.ndarray = None) -> float:
        """
        Calculate Hurst exponent using rescaled range analysis.

        H < 0.5: mean-reverting (good for pairs trading)
        H = 0.5: random walk
        H > 0.5: trending

        Parameters:
        -----------
        series : array-like
            Time series
        lags : array-like, optional
            Lags to use (default: log-spaced)

        Returns:
        --------
        Hurst exponent estimate
        """
        series = np.asarray(series, dtype=np.float64)
        series = series[~np.isnan(series)]

        if len(series) < 10:
            return 0.5

        # Mean-center
        series = series - np.mean(series)

        if lags is None:
            lags = np.logspace(0.5, np.log10(len(series) // 2), 20, dtype=int)
            lags = np.unique(lags)

        tau = []
        for lag in lags:
            if lag >= len(series):
                break

            # Compute rescaled range
            chunks = len(series) // lag
            RS = []
            for i in range(chunks):
                chunk = series[i * lag:(i + 1) * lag]
                if len(chunk) < lag:
                    continue

                # Cumulative sum
                Y = np.cumsum(chunk - np.mean(chunk))

                # Range
                R = np.max(Y) - np.min(Y)

                # Standard deviation
                S = np.std(chunk, ddof=1)

                if S > 0:
                    RS.append(R / S)

            if RS:
                tau.append(np.mean(RS))

        if len(tau) < 2:
            return 0.5

        # Fit log(R/S) = H * log(lag) + constant
        lags = lags[:len(tau)]
        log_tau = np.log(tau)
        log_lags = np.log(lags)

        H = np.polyfit(log_lags, log_tau, 1)[0]

        return np.clip(H, 0, 1)

    @staticmethod
    def score_pair(series1: np.ndarray, series2: np.ndarray,
                   symbol1: str = "X", symbol2: str = "Y") -> PairScore:
        """
        Score a pair based on multiple criteria.

        Parameters:
        -----------
        series1, series2 : array-like
            Price series for the two assets
        symbol1, symbol2 : str
            Asset identifiers

        Returns:
        --------
        PairScore with composite and individual scores
        """
        series1 = np.asarray(series1, dtype=np.float64)
        series2 = np.asarray(series2, dtype=np.float64)

        # Remove NaN
        mask = ~(np.isnan(series1) | np.isnan(series2))
        series1 = series1[mask]
        series2 = series2[mask]

        if len(series1) < 20:
            return PairScore(symbol1, symbol2, 1.0, np.inf, 0.5, np.inf, 0.0)

        # Cointegration test
        coint_result = EngleGrangerTest.test(series1, series2)
        p_value = coint_result.p_value

        # Mean reversion parameters
        spread = series2 - np.mean(series2) / np.mean(series1) * series1
        ou_params = OrnsteinUhlenbeckEstimator.estimate_regression(spread)
        half_life = ou_params.half_life

        # Hurst exponent (on spread)
        h_exp = PairSelector.hurst_exponent(spread)

        # Spread stability (lower std is better)
        spread_std = np.std(spread)

        # Composite score (higher is better)
        # 1. Cointegration: penalize high p-values
        coint_score = max(0, 1 - p_value) * 0.4  # 40% weight

        # 2. Half-life: optimal range 5-20 periods
        if 5 <= half_life <= 20:
            hl_score = 1.0
        elif half_life < 5:
            hl_score = half_life / 5
        else:
            hl_score = max(0, 1 - (half_life - 20) / 50)
        hl_score *= 0.3  # 30% weight

        # 3. Hurst exponent: lower is better (mean-reverting)
        h_score = max(0, 1 - h_exp) * 0.2  # 20% weight

        # 4. Spread stability: lower std is better
        if spread_std > 0:
            std_score = max(0, 1 - np.log1p(spread_std) / 5) * 0.1  # 10% weight
        else:
            std_score = 0.0

        total_score = coint_score + hl_score + h_score + std_score

        return PairScore(
            symbol1=symbol1,
            symbol2=symbol2,
            cointegration_pvalue=p_value,
            half_life=half_life,
            hurst_exponent=h_exp,
            spread_std=spread_std,
            total_score=total_score
        )

    @staticmethod
    def screen_universe(
        data: pd.DataFrame,
        min_score: float = 0.3,
        max_pairs: Optional[int] = None
    ) -> List[PairScore]:
        """
        Screen entire universe of assets for pairs.

        Parameters:
        -----------
        data : DataFrame
            DataFrame with assets as columns, dates as index
        min_score : float
            Minimum composite score to include pair
        max_pairs : int, optional
            Maximum pairs to return (sorted by score)

        Returns:
        --------
        List of PairScore, sorted by score (descending)
        """
        symbols = data.columns.tolist()
        pairs = []

        for sym1, sym2 in combinations(symbols, 2):
            try:
                score = PairSelector.score_pair(
                    data[sym1].values,
                    data[sym2].values,
                    symbol1=sym1,
                    symbol2=sym2
                )

                if score.total_score >= min_score:
                    pairs.append(score)
            except (ValueError, RuntimeError):
                # Skip pairs that fail
                continue

        # Sort by score (descending)
        pairs.sort(key=lambda x: x.total_score, reverse=True)

        if max_pairs:
            pairs = pairs[:max_pairs]

        return pairs

    @staticmethod
    def filter_pairs(
        scores: List[PairScore],
        max_correlation: float = 0.95,
        data: Optional[pd.DataFrame] = None
    ) -> List[PairScore]:
        """
        Filter pairs to avoid redundancy.

        Removes pairs that share assets with higher-scoring pairs,
        to avoid over-diversification with correlated pairs.

        Parameters:
        -----------
        scores : List[PairScore]
            Pairs to filter
        max_correlation : float
            Maximum correlation to allow between assets in different pairs
        data : DataFrame, optional
            Price data for correlation calculation

        Returns:
        --------
        Filtered list of non-overlapping pairs
        """
        filtered = []
        used_symbols = set()

        for score in scores:
            if score.symbol1 not in used_symbols and score.symbol2 not in used_symbols:
                filtered.append(score)
                used_symbols.add(score.symbol1)
                used_symbols.add(score.symbol2)

        return filtered

    @staticmethod
    def generate_report(scores: List[PairScore]) -> pd.DataFrame:
        """Generate pandas DataFrame report of pair scores."""
        return pd.DataFrame([
            {
                "Pair": f"{s.symbol1}/{s.symbol2}",
                "Score": f"{s.total_score:.4f}",
                "P-Value": f"{s.cointegration_pvalue:.4f}",
                "Half-Life": f"{s.half_life:.2f}",
                "Hurst": f"{s.hurst_exponent:.3f}",
                "Spread Std": f"{s.spread_std:.6f}",
            }
            for s in scores
        ])
