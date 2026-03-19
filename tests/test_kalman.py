"""Unit tests for Kalman filter."""

import numpy as np
import pytest
import sys
sys.path.insert(0, '../src')

from python.research.kalman_filter import KalmanFilter


class TestKalmanFilter:
    """Test suite for Kalman filter."""

    def test_initialization(self):
        """Test Kalman filter initialization."""
        kf = KalmanFilter(initial_hedge_ratio=1.5, process_noise=1e-5)

        assert kf.get_hedge_ratio() == 1.5
        assert kf.Q == 1e-5

    def test_predict_step(self):
        """Test prediction step."""
        kf = KalmanFilter()

        state = kf.predict()

        assert state.x is not None
        assert state.P is not None

    def test_update_step(self):
        """Test update step with observation."""
        kf = KalmanFilter()

        state = kf.update(100.0, 150.0)

        assert state.x is not None
        assert state.y is not None  # Spread

    def test_filter_series(self):
        """Test filtering a complete time series."""
        np.random.seed(42)
        n = 100

        # Generate two price series with known relationship
        x_series = np.cumsum(np.random.randn(n) * 0.01) + 100
        y_series = 1.5 * x_series + np.cumsum(np.random.randn(n) * 0.005) + 50

        kf = KalmanFilter()
        hedge_ratios, spreads = kf.filter(x_series, y_series)

        assert len(hedge_ratios) == n
        assert len(spreads) == n
        assert not np.any(np.isnan(hedge_ratios))
        assert not np.any(np.isnan(spreads))

    def test_hedge_ratio_convergence(self):
        """Test that hedge ratio converges to true ratio."""
        np.random.seed(42)
        n = 200

        true_ratio = 2.0
        x_series = np.cumsum(np.random.randn(n) * 0.01) + 100
        y_series = true_ratio * x_series + np.cumsum(np.random.randn(n) * 0.1)

        kf = KalmanFilter(initial_hedge_ratio=1.0, process_noise=1e-4)
        hedge_ratios, _ = kf.filter(x_series, y_series)

        # Hedge ratio should converge towards true ratio
        final_ratio = hedge_ratios[-1]
        assert abs(final_ratio - true_ratio) < 0.3, \
            f"Final ratio {final_ratio} should be close to {true_ratio}"

    def test_reset(self):
        """Test filter reset."""
        kf = KalmanFilter()

        # Do some updates
        kf.update(100, 150)
        kf.update(101, 152)

        # Reset
        kf.reset(initial_hedge_ratio=1.0)

        assert kf.get_hedge_ratio() == 1.0
        assert len(kf.get_history()) == 0

    def test_history_tracking(self):
        """Test that history is properly tracked."""
        kf = KalmanFilter()

        for i in range(10):
            kf.predict()
            kf.update(100.0 + i, 150.0 + i * 1.5)

        history = kf.get_history()

        assert len(history) == 10
        assert 'hedge_ratio' in history[0]
        assert 'spread' in history[0]
        assert 'covariance' in history[0]

    def test_low_noise_environment(self):
        """Test filter behavior with low noise."""
        kf = KalmanFilter(process_noise=1e-8, measurement_noise=1e-6)

        # Nearly perfect relationship
        x_series = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        y_series = 1.5 * x_series

        _, spreads = kf.filter(x_series, y_series)

        # Spreads should be close to zero with low noise
        assert np.max(np.abs(spreads)) < 0.5

    def test_high_noise_environment(self):
        """Test filter behavior with high noise."""
        kf = KalmanFilter(process_noise=0.1, measurement_noise=1.0)

        x_series = np.random.randn(50) + 100
        y_series = 1.5 * x_series + np.random.randn(50) * 10

        hedge_ratios, _ = kf.filter(x_series, y_series)

        # Covariance should increase with high noise
        final_cov = kf.get_covariance()
        assert final_cov > 0.01, "Covariance should be large with high noise"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
