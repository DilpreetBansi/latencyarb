"""
Kalman filter for dynamic hedge ratio estimation.

Continuously updates the optimal hedge ratio between cointegrated pairs
as market conditions change. Essential for adaptive pairs trading.
"""

import numpy as np
from typing import Tuple, NamedTuple


class KalmanState(NamedTuple):
    """State of Kalman filter."""
    x: np.ndarray  # State (hedge ratio)
    P: np.ndarray  # Covariance of state estimate
    y: float       # Output (spread prediction)


class KalmanFilter:
    """
    1D Kalman filter for estimating hedge ratio between two cointegrated series.

    State equation: h_t = h_{t-1} + w_t,  w_t ~ N(0, Q)
    Observation: z_t = y_t - h_t * x_t + v_t,  v_t ~ N(0, R)

    Where:
    - h_t is the hedge ratio (state)
    - x_t, y_t are the two price series
    - z_t is the spread (observation)
    """

    def __init__(
        self,
        initial_hedge_ratio: float = 1.0,
        process_noise: float = 1e-6,
        measurement_noise: float = 0.01,
        initial_covariance: float = 1.0,
    ):
        """
        Initialize Kalman filter.

        Parameters:
        -----------
        initial_hedge_ratio : float
            Starting estimate of hedge ratio
        process_noise : float
            Process noise variance (Q) - higher = faster adaptation
        measurement_noise : float
            Measurement noise variance (R) - higher = less trust in observations
        initial_covariance : float
            Initial state covariance (P0)
        """
        self.Q = process_noise
        self.R = measurement_noise
        self.state = np.array([initial_hedge_ratio])
        self.P = initial_covariance
        self.history = []

    def predict(self) -> KalmanState:
        """
        Prediction step: predict next state and covariance.

        Returns:
        --------
        KalmanState with predicted values
        """
        # State prediction: h_{t|t-1} = h_{t-1}
        x_pred = self.state.copy()

        # Covariance prediction: P_{t|t-1} = P_{t-1} + Q
        P_pred = self.P + self.Q

        return KalmanState(x=x_pred, P=P_pred, y=0.0)

    def update(self, x: float, y: float) -> KalmanState:
        """
        Update step: incorporate new observation.

        Parameters:
        -----------
        x : float
            Price of first asset
        y : float
            Price of second asset

        Returns:
        --------
        KalmanState with updated values
        """
        # Innovation (measurement residual)
        spread = y - self.state[0] * x
        innovation = spread

        # Innovation covariance
        S = self.P * x * x + self.R

        # Kalman gain
        K = self.P * x / S

        # State update
        self.state = self.state + K * innovation

        # Covariance update
        self.P = (1 - K * x) * self.P

        # Store history
        self.history.append({
            "hedge_ratio": self.state[0],
            "spread": spread,
            "innovation": innovation,
            "covariance": self.P,
        })

        return KalmanState(
            x=self.state.copy(),
            P=self.P,
            y=spread
        )

    def filter(self, x_series: np.ndarray, y_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter entire time series.

        Parameters:
        -----------
        x_series : array-like
            Time series of first asset prices
        y_series : array-like
            Time series of second asset prices

        Returns:
        --------
        hedge_ratios : filtered hedge ratio estimates
        spreads : filtered spread estimates
        """
        x_series = np.asarray(x_series, dtype=np.float64)
        y_series = np.asarray(y_series, dtype=np.float64)

        hedge_ratios = np.zeros(len(x_series))
        spreads = np.zeros(len(x_series))

        for t in range(len(x_series)):
            self.predict()
            state = self.update(x_series[t], y_series[t])
            hedge_ratios[t] = state.x[0]
            spreads[t] = state.y

        return hedge_ratios, spreads

    def get_hedge_ratio(self) -> float:
        """Get current hedge ratio estimate."""
        return self.state[0]

    def get_covariance(self) -> float:
        """Get current estimate covariance."""
        return self.P

    def reset(self, initial_hedge_ratio: float = 1.0, initial_covariance: float = 1.0) -> None:
        """Reset filter to initial state."""
        self.state = np.array([initial_hedge_ratio])
        self.P = initial_covariance
        self.history = []

    def get_history(self) -> list:
        """Get history of filter updates."""
        return self.history


class MultivarianteKalmanFilter:
    """
    Multivariate Kalman filter for estimating hedge ratios across multiple pairs.

    Handles systems with multiple state variables and observations.
    """

    def __init__(self, n_assets: int, process_noise: float = 1e-6,
                 measurement_noise: float = 0.01):
        """
        Initialize multivariate Kalman filter.

        Parameters:
        -----------
        n_assets : int
            Number of assets (dimensions)
        process_noise : float
            Process noise variance
        measurement_noise : float
            Measurement noise variance
        """
        self.n = n_assets
        self.Q = process_noise * np.eye(n_assets)
        self.R = measurement_noise
        self.x = np.ones(n_assets)
        self.P = np.eye(n_assets)

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state."""
        x_pred = self.x.copy()
        P_pred = self.P + self.Q
        return x_pred, P_pred

    def update(self, price_vector: np.ndarray, target: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update with new observation.

        Parameters:
        -----------
        price_vector : array
            Vector of current prices
        target : float
            Target or observed spread

        Returns:
        --------
        Updated state and covariance
        """
        # Prediction
        x_pred, P_pred = self.predict()

        # Innovation
        y_hat = price_vector @ x_pred
        innovation = target - y_hat

        # Innovation covariance
        S = price_vector @ P_pred @ price_vector + self.R

        # Kalman gain
        K = P_pred @ price_vector / S

        # Update state
        self.x = x_pred + K * innovation

        # Update covariance
        self.P = (np.eye(self.n) - np.outer(K, price_vector)) @ P_pred

        return self.x.copy(), self.P.copy()

    def get_state(self) -> np.ndarray:
        """Get current state vector."""
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Get current covariance matrix."""
        return self.P.copy()
