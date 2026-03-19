"""Research layer for statistical arbitrage analysis."""

from .cointegration import EngleGrangerTest, JohansenTest
from .kalman_filter import KalmanFilter
from .mean_reversion import OrnsteinUhlenbeckEstimator
from .pair_selection import PairSelector
from .regime_detection import RegimeDetector

__all__ = [
    "EngleGrangerTest",
    "JohansenTest",
    "KalmanFilter",
    "OrnsteinUhlenbeckEstimator",
    "PairSelector",
    "RegimeDetector",
]
