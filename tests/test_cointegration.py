"""Unit tests for cointegration module."""

import numpy as np
import pytest
import sys
sys.path.insert(0, '../src')

from python.research.cointegration import EngleGrangerTest, JohansenTest


class TestEngleGrangerTest:
    """Test suite for Engle-Granger cointegration test."""

    def test_cointegrated_series(self):
        """Test detection of cointegrated series."""
        # Create cointegrated series
        np.random.seed(42)
        n = 200
        t = np.arange(n)

        x = np.cumsum(np.random.randn(n)) + 100
        y = 2 * x + np.random.randn(n) * 0.1 + 50  # Highly cointegrated

        result = EngleGrangerTest.test(x, y)

        # Should detect cointegration
        assert result.p_value < 0.10, f"P-value {result.p_value} should be low"
        assert result.cointegration_vector is not None

    def test_non_cointegrated_series(self):
        """Test detection of non-cointegrated series."""
        # Create independent random walks
        np.random.seed(42)
        n = 200

        x = np.cumsum(np.random.randn(n))
        y = np.cumsum(np.random.randn(n))

        result = EngleGrangerTest.test(x, y)

        # Should NOT detect cointegration
        assert result.p_value > 0.05, f"P-value {result.p_value} should be high"

    def test_adf_test_stationary(self):
        """Test ADF test on stationary series."""
        # Stationary series
        residuals = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1

        t_stat, p_val = EngleGrangerTest.adf_test(residuals)

        # Should reject null hypothesis (series is stationary)
        assert p_val < 0.10, f"Stationary series should have low p-value"

    def test_adf_test_non_stationary(self):
        """Test ADF test on non-stationary series."""
        # Non-stationary (random walk)
        residuals = np.cumsum(np.random.randn(100))

        t_stat, p_val = EngleGrangerTest.adf_test(residuals)

        # Should not reject null hypothesis (series is non-stationary)
        assert p_val > 0.05, f"Non-stationary series should have high p-value"

    def test_short_series(self):
        """Test behavior on short series."""
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])

        result = EngleGrangerTest.test(x, y)

        # Should handle gracefully
        assert result.p_value == 1.0
        assert not result.is_cointegrated

    def test_with_nan(self):
        """Test handling of NaN values."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, 4, 6, np.nan, 10])

        result = EngleGrangerTest.test(x, y)

        # Should handle NaN gracefully
        assert isinstance(result.test_statistic, float)
        assert isinstance(result.p_value, float)


class TestJohansenTest:
    """Test suite for Johansen cointegration test."""

    def test_single_cointegrating_pair(self):
        """Test detection of single cointegrating pair."""
        np.random.seed(42)
        n = 200

        x1 = np.cumsum(np.random.randn(n))
        x2 = 2 * x1 + np.random.randn(n) * 0.1  # Cointegrated with x1
        x3 = np.cumsum(np.random.randn(n))  # Independent

        data = np.column_stack([x1, x2, x3])

        result = JohansenTest.test(data)

        # Should detect at least 1 cointegrating relationship
        assert result['n_cointegrating'] >= 1

    def test_three_variable_system(self):
        """Test on 3-variable system."""
        np.random.seed(42)
        n = 150

        data = np.column_stack([
            np.cumsum(np.random.randn(n)),
            np.cumsum(np.random.randn(n)),
            np.cumsum(np.random.randn(n)),
        ])

        result = JohansenTest.test(data)

        # Should return valid statistics
        assert 'eigenvalues' in result
        assert len(result['eigenvalues']) == 3
        assert 'n_cointegrating' in result

    def test_output_structure(self):
        """Test output structure of Johansen test."""
        np.random.seed(42)
        n = 100
        data = np.random.randn(n, 2)

        result = JohansenTest.test(data)

        # Check all expected keys
        expected_keys = [
            'trace_statistic',
            'eigenvalue_statistic',
            'eigenvalues',
            'critical_values_90',
            'critical_values_95',
            'n_cointegrating'
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
