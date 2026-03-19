"""
Ornstein-Uhlenbeck process parameter estimation.

Estimates mean-reverting parameters for pairs. Essential for predicting
reversion times and setting stop losses based on signal strength.
"""

import numpy as np
from typing import NamedTuple
from scipy.optimize import minimize


class OUParameters(NamedTuple):
    """Ornstein-Uhlenbeck process parameters."""
    mu: float          # Long-term mean
    theta: float       # Mean reversion speed
    sigma: float       # Volatility
    half_life: float   # Half-life of mean reversion in periods


class OrnsteinUhlenbeckEstimator:
    """
    Estimate Ornstein-Uhlenbeck process parameters from time series.

    The OU process is:
    dX_t = θ(μ - X_t)dt + σ dW_t

    In discrete form:
    X_t = μ + e^(-θΔt) * (X_{t-1} - μ) + σ * √(1 - e^(-2θΔt)) * ε_t

    Where:
    - μ: long-term mean
    - θ: mean reversion speed (higher = faster reversion)
    - σ: volatility of innovations
    """

    @staticmethod
    def estimate_mle(series: np.ndarray, dt: float = 1.0) -> OUParameters:
        """
        Estimate OU parameters using Maximum Likelihood Estimation.

        Parameters:
        -----------
        series : array-like
            Time series to fit
        dt : float
            Time interval between observations

        Returns:
        --------
        OUParameters with estimated values
        """
        series = np.asarray(series, dtype=np.float64)

        # Remove NaN values
        series = series[~np.isnan(series)]

        if len(series) < 10:
            return OUParameters(
                mu=np.mean(series),
                theta=0.1,
                sigma=np.std(series),
                half_life=np.log(2) / 0.1
            )

        # Initial estimates
        mu_init = np.mean(series)
        sigma_init = np.std(series)

        def negative_log_likelihood(params: np.ndarray) -> float:
            """Compute negative log-likelihood."""
            mu, theta, sigma = params

            # Constraints
            if theta <= 0 or sigma <= 0:
                return 1e10

            # Likelihood
            nll = 0.0
            for i in range(1, len(series)):
                # Expected value and variance of X_t given X_{t-1}
                exp_neg_theta = np.exp(-theta * dt)
                mean_t = mu + exp_neg_theta * (series[i - 1] - mu)
                var_t = (sigma ** 2 / (2 * theta)) * (1 - exp_neg_theta ** 2)

                # Gaussian log-likelihood
                if var_t > 0:
                    nll += 0.5 * np.log(2 * np.pi * var_t)
                    nll += 0.5 * ((series[i] - mean_t) ** 2) / var_t

            return nll

        # Optimize
        result = minimize(
            negative_log_likelihood,
            x0=np.array([mu_init, 0.1, sigma_init]),
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )

        if result.success:
            mu, theta, sigma = result.x
        else:
            mu, theta, sigma = mu_init, 0.1, sigma_init

        # Ensure parameters are valid
        theta = max(theta, 1e-6)
        sigma = max(sigma, 1e-6)

        # Half-life of mean reversion
        half_life = np.log(2) / (theta * dt) if theta > 0 else np.inf

        return OUParameters(
            mu=mu,
            theta=theta,
            sigma=sigma,
            half_life=half_life
        )

    @staticmethod
    def estimate_regression(series: np.ndarray, dt: float = 1.0) -> OUParameters:
        """
        Estimate OU parameters using regression method.

        Faster but less accurate than MLE. Uses least-squares fitting of:
        X_t - X_{t-1} = α(μ - X_{t-1}) + noise

        Parameters:
        -----------
        series : array-like
            Time series to fit
        dt : float
            Time interval between observations

        Returns:
        --------
        OUParameters with estimated values
        """
        series = np.asarray(series, dtype=np.float64)
        series = series[~np.isnan(series)]

        if len(series) < 10:
            return OUParameters(
                mu=np.mean(series),
                theta=0.1,
                sigma=np.std(series),
                half_life=np.log(2) / 0.1
            )

        # Differences
        diffs = np.diff(series)

        # Regression: dX = -θ(X - μ) + noise
        # Rewrite as: dX/dt = -θ*X + θ*μ
        X_lag = series[:-1]

        # OLS: dX = β0 + β1*X
        X_mat = np.column_stack([np.ones(len(X_lag)), X_lag])
        beta = np.linalg.lstsq(X_mat, diffs / dt, rcond=None)[0]

        # θ*μ = β0, -θ = β1
        theta = -beta[1] * dt
        mu = beta[0] / beta[1] if beta[1] != 0 else np.mean(series)

        # Residual std
        residuals = diffs / dt - X_mat @ beta
        sigma = np.std(residuals) * np.sqrt(dt)

        theta = max(theta, 1e-6)
        sigma = max(sigma, 1e-6)

        half_life = np.log(2) / theta if theta > 0 else np.inf

        return OUParameters(
            mu=mu,
            theta=theta,
            sigma=sigma,
            half_life=half_life
        )

    @staticmethod
    def half_life_from_acf(series: np.ndarray, dt: float = 1.0) -> float:
        """
        Estimate half-life from autocorrelation function.

        The lag at which ACF drops to 0.5 approximates the half-life.
        """
        series = np.asarray(series, dtype=np.float64)
        series = series[~np.isnan(series)]

        if len(series) < 10:
            return np.inf

        # Mean-center
        series = series - np.mean(series)

        # Autocorrelations
        c0 = np.sum(series ** 2) / len(series)
        acfs = []
        for lag in range(1, min(len(series) // 2, 100)):
            c = np.sum(series[:-lag] * series[lag:]) / len(series)
            acfs.append(c / c0)

        acfs = np.array(acfs)

        # Find crossing of 0.5
        if len(acfs) > 0 and np.min(acfs) < 0.5:
            idx = np.argmax(acfs < 0.5)
            half_life = idx * dt
            return half_life

        return np.inf

    @staticmethod
    def simulate(
        params: OUParameters,
        n_steps: int,
        dt: float = 1.0,
        x0: float = 0.0,
        random_state: int = None
    ) -> np.ndarray:
        """
        Simulate OU process.

        Parameters:
        -----------
        params : OUParameters
            Process parameters
        n_steps : int
            Number of steps to simulate
        dt : float
            Time interval
        x0 : float
            Initial value
        random_state : int
            Random seed

        Returns:
        --------
        Simulated time series
        """
        if random_state is not None:
            np.random.seed(random_state)

        path = np.zeros(n_steps)
        path[0] = x0

        exp_neg_theta = np.exp(-params.theta * dt)
        std = params.sigma * np.sqrt((1 - exp_neg_theta ** 2) / (2 * params.theta))

        for t in range(1, n_steps):
            path[t] = (
                params.mu +
                exp_neg_theta * (path[t - 1] - params.mu) +
                std * np.random.randn()
            )

        return path
