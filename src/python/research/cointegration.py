"""
Cointegration testing for pairs trading.

Implements Engle-Granger two-step method and Johansen procedure.
Critical for identifying statistically valid pairs for mean reversion trading.
"""

import numpy as np
from typing import Tuple, Dict, Optional, NamedTuple
from scipy import stats
from scipy.linalg import eig
import warnings

warnings.filterwarnings("ignore")


class CointestResult(NamedTuple):
    """Result from cointegration test."""
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    is_cointegrated: bool
    cointegration_vector: Optional[np.ndarray] = None


class EngleGrangerTest:
    """
    Engle-Granger two-step cointegration test.

    Tests if two time series X and Y are cointegrated by:
    1. Running OLS regression: Y = a + b*X + e
    2. Testing residuals e for unit root (ADF test)

    If residuals are stationary, X and Y are cointegrated.
    """

    @staticmethod
    def adf_test(residuals: np.ndarray, max_lag: int = 1) -> Tuple[float, float]:
        """
        Simplified Augmented Dickey-Fuller test.

        Returns test_statistic, p_value.
        """
        # Remove NaN values
        residuals = residuals[~np.isnan(residuals)]

        if len(residuals) < 10:
            return 0.0, 1.0

        # Construct regression: Δy = α + β*y_{t-1} + Σγ_i*Δy_{t-i}
        n = len(residuals)
        dy = np.diff(residuals)

        # Build regressor matrix
        regressors = np.column_stack([
            np.ones(n - 1),
            residuals[:-1],
        ])

        # Add lagged differences
        for lag in range(1, max_lag + 1):
            if lag < len(dy):
                regressors = np.column_stack([
                    regressors,
                    np.concatenate([np.zeros(lag), dy[:-lag]])
                ])

        # OLS regression
        y = dy
        try:
            beta = np.linalg.lstsq(regressors, y, rcond=None)[0]
            residuals_reg = y - regressors @ beta
            rss = np.sum(residuals_reg ** 2)
            mse = rss / (len(y) - len(beta))

            # Standard error of beta[1] (coefficient on y_{t-1})
            var_covar = np.linalg.inv(regressors.T @ regressors)
            se_beta = np.sqrt(np.diag(var_covar) * mse)

            # t-statistic
            t_stat = beta[1] / se_beta[1]

            # Approximate p-value (crude but works)
            # Under null hypothesis of unit root, t_stat ~ N(0, 1) approximately
            p_val = stats.norm.sf(abs(t_stat))

            return t_stat, p_val
        except np.linalg.LinAlgError:
            return 0.0, 1.0

    @staticmethod
    def test(x: np.ndarray, y: np.ndarray, significance: float = 0.05) -> CointestResult:
        """
        Test cointegration between x and y using Engle-Granger method.

        Parameters:
        -----------
        x : array-like
            First time series
        y : array-like
            Second time series (dependent variable)
        significance : float
            Significance level for ADF test (default 5%)

        Returns:
        --------
        CointestResult with test statistics and cointegration vector
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Remove rows with NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        if len(x) < 10:
            return CointestResult(
                test_statistic=0.0,
                p_value=1.0,
                critical_values={"1%": -3.43, "5%": -2.86, "10%": -2.57},
                is_cointegrated=False,
                cointegration_vector=None
            )

        # Step 1: OLS regression y = a + b*x + e
        X_reg = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X_reg, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return CointestResult(0.0, 1.0, {"1%": -3.43, "5%": -2.86, "10%": -2.57}, False)

        residuals = y - X_reg @ beta

        # Step 2: ADF test on residuals
        adf_stat, p_value = EngleGrangerTest.adf_test(residuals)

        # Critical values for ADF test (Mackinnon's values)
        critical = {"1%": -3.43, "5%": -2.86, "10%": -2.57}

        is_coint = adf_stat < critical["5%"]  # Use 5% significance

        # Cointegration vector [1, -beta[1]]
        coint_vec = np.array([1.0, -beta[1]])

        return CointestResult(
            test_statistic=adf_stat,
            p_value=p_value,
            critical_values=critical,
            is_cointegrated=is_coint,
            cointegration_vector=coint_vec
        )


class JohansenTest:
    """
    Johansen cointegration test for multivariate systems.

    Tests for cointegrating relationships between N > 2 time series.
    More powerful than Engle-Granger for detecting multiple cointegrating pairs.
    """

    @staticmethod
    def test(data: np.ndarray, det_order: int = 0, k_ar_diff: int = 1) -> Dict:
        """
        Johansen cointegration test.

        Parameters:
        -----------
        data : ndarray, shape (T, N)
            Time series matrix, T observations, N variables
        det_order : int
            Order of deterministic terms (0=none, 1=constant)
        k_ar_diff : int
            Number of lags in AR model

        Returns:
        --------
        Dictionary with trace and eigenvalue test statistics
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("data must be 2-dimensional")

        T, N = data.shape

        if T < 2 * N:
            return {
                "trace_statistic": np.zeros(N),
                "eigenvalue_statistic": np.zeros(N),
                "critical_values_90": np.zeros(N),
                "critical_values_95": np.zeros(N),
                "n_cointegrating": 0,
            }

        # Difference the data
        diff_data = np.diff(data, axis=0)

        # Build lagged differences
        if k_ar_diff > 0:
            lags = np.column_stack([
                np.concatenate([np.zeros((lag, N)), diff_data[:-lag]])
                for lag in range(1, k_ar_diff + 1)
            ])
        else:
            lags = np.empty((len(diff_data), 0))

        # Deterministic terms
        if det_order == 1:
            det = np.ones((len(diff_data), 1))
        else:
            det = np.empty((len(diff_data), 0))

        # Construct regressor matrix
        X = np.column_stack([lags, det]) if lags.shape[1] > 0 or det.shape[1] > 0 else np.empty((len(diff_data), 0))

        y = diff_data
        y_lagged = data[k_ar_diff:-1]

        # Residuals from regressions
        if X.shape[1] > 0:
            R0 = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
            R1 = y_lagged - X @ np.linalg.lstsq(X, y_lagged, rcond=None)[0]
        else:
            R0 = y
            R1 = y_lagged

        # Covariance matrices
        S00 = R0.T @ R0 / len(R0)
        S11 = R1.T @ R1 / len(R1)
        S01 = R0.T @ R1 / len(R0)

        try:
            # Solve generalized eigenvalue problem
            S00_inv = np.linalg.inv(S00 + 1e-8 * np.eye(N))
            M = S11 @ S00_inv @ S01.T
            eigenvalues, _ = eig(M @ np.linalg.inv(S11), np.eye(N))
            eigenvalues = np.real(np.sort(eigenvalues)[::-1])
            eigenvalues = np.maximum(eigenvalues, 0)
        except np.linalg.LinAlgError:
            eigenvalues = np.zeros(N)

        # Trace statistic
        trace_stat = np.array([
            -len(y) * np.sum(np.log(1 - eigenvalues[i:])) for i in range(N)
        ])

        # Eigenvalue statistic
        eig_stat = -len(y) * np.log(1 - eigenvalues)

        # Critical values (Osterwald-Lenum, 1992)
        critical_90 = np.array([13.7, 19.9, 25.6, 30.9]) if N >= 4 else np.array([13.7, 19.9, 25.6])
        critical_95 = np.array([15.4, 21.9, 27.7, 33.4]) if N >= 4 else np.array([15.4, 21.9, 27.7])

        # Count cointegrating relationships
        n_coint = np.sum(eig_stat > critical_95[:min(N, len(critical_95))])

        return {
            "trace_statistic": trace_stat,
            "eigenvalue_statistic": eig_stat,
            "eigenvalues": eigenvalues,
            "critical_values_90": critical_90[:N],
            "critical_values_95": critical_95[:N],
            "n_cointegrating": min(n_coint, N - 1),
        }
