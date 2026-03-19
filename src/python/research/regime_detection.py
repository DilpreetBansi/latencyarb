"""
Hidden Markov Model for market regime detection.

Detects whether market is in trending or mean-reverting regime.
Pairs trading works best in mean-reverting regime.
"""

import numpy as np
from typing import Tuple, NamedTuple, Optional

import warnings
warnings.filterwarnings("ignore")


class HMMRegime(NamedTuple):
    """Hidden Markov Model regime state."""
    regime: int               # 0 = mean-reverting, 1 = trending
    probability: float        # Probability of this regime
    log_likelihood: float     # Log-likelihood of model


class RegimeDetector:
    """
    2-state Hidden Markov Model for regime detection.

    States:
    - State 0: Mean-reverting (low volatility, mean-seeking)
    - State 1: Trending (high volatility, momentum)

    Detects which regime market is in for adaptive trading.
    """

    def __init__(
        self,
        n_states: int = 2,
        initial_state_prob: Optional[np.ndarray] = None,
        initial_transition: Optional[np.ndarray] = None,
    ):
        """
        Initialize HMM.

        Parameters:
        -----------
        n_states : int
            Number of hidden states (default 2)
        initial_state_prob : array, optional
            Initial state probabilities
        initial_transition : array, optional
            Transition matrix (n_states x n_states)
        """
        self.n_states = n_states
        self.state_prob = initial_state_prob or np.ones(n_states) / n_states
        self.transition = initial_transition or np.ones((n_states, n_states)) / n_states
        self.emission_mean = np.zeros(n_states)
        self.emission_std = np.ones(n_states)
        self.fitted = False

    def fit(self, observations: np.ndarray, max_iter: int = 100, tol: float = 1e-5) -> None:
        """
        Fit HMM using Baum-Welch algorithm (EM).

        Parameters:
        -----------
        observations : array-like
            Observation sequence (e.g., returns)
        max_iter : int
            Maximum iterations for EM
        tol : float
            Convergence tolerance
        """
        observations = np.asarray(observations, dtype=np.float64)
        observations = observations[~np.isnan(observations)]

        if len(observations) < 10:
            self.fitted = False
            return

        # Initialize parameters
        self._initialize_parameters(observations)

        prev_ll = -np.inf

        for iteration in range(max_iter):
            # Forward pass
            alpha, ll = self._forward(observations)

            # Check convergence
            if ll - prev_ll < tol:
                break

            prev_ll = ll

            # Backward pass
            beta = self._backward(observations)

            # E-step: compute state posteriors
            gamma = self._compute_gamma(alpha, beta, ll)
            xi = self._compute_xi(observations, alpha, beta, ll)

            # M-step: update parameters
            self.state_prob = gamma[0]
            self.transition = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0, keepdims=True)

            for s in range(self.n_states):
                gamma_s = gamma[:, s]
                self.emission_mean[s] = np.sum(gamma_s * observations) / np.sum(gamma_s)
                diff = observations - self.emission_mean[s]
                self.emission_std[s] = np.sqrt(np.sum(gamma_s * diff ** 2) / np.sum(gamma_s))
                self.emission_std[s] = max(self.emission_std[s], 1e-6)

        self.fitted = True

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict hidden states using Viterbi algorithm.

        Parameters:
        -----------
        observations : array-like
            Observation sequence

        Returns:
        --------
        Array of predicted hidden states
        """
        observations = np.asarray(observations, dtype=np.float64)

        if not self.fitted or len(observations) < 1:
            return np.zeros(len(observations), dtype=int)

        T = len(observations)
        viterbi = np.zeros((T, self.n_states))
        path = np.zeros((T, self.n_states), dtype=int)

        # Initialize
        viterbi[0] = np.log(self.state_prob) + self._emission_prob(observations[0])

        # Forward pass
        for t in range(1, T):
            for s in range(self.n_states):
                # Probability of reaching state s at time t
                trans_prob = viterbi[t - 1] + np.log(self.transition[:, s])
                path[t, s] = np.argmax(trans_prob)
                viterbi[t, s] = trans_prob[path[t, s]] + self._emission_prob(observations[t], s)

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(T - 2, -1, -1):
            states[t] = path[t + 1, states[t + 1]]

        return states

    def get_current_regime(self, observations: np.ndarray) -> HMMRegime:
        """
        Get current regime (last state in sequence).

        Parameters:
        -----------
        observations : array-like
            Observation sequence

        Returns:
        --------
        HMMRegime with current state and probability
        """
        states = self.predict(observations)

        if len(states) == 0:
            return HMMRegime(regime=0, probability=0.5, log_likelihood=0.0)

        current_state = states[-1]

        # Compute smoothed probability using forward-backward
        alpha, ll = self._forward(observations)
        beta = self._backward(observations)

        # Smoothed probability at last time step
        gamma = alpha[-1] * beta[-1]
        gamma = gamma / np.sum(gamma)

        return HMMRegime(
            regime=current_state,
            probability=gamma[current_state],
            log_likelihood=ll
        )

    def _initialize_parameters(self, observations: np.ndarray) -> None:
        """Initialize HMM parameters from data."""
        # Split observations into two groups for initial estimates
        split = len(observations) // 2
        group1 = observations[:split]
        group2 = observations[split:]

        self.emission_mean[0] = np.mean(group1)
        self.emission_mean[1] = np.mean(group2)

        self.emission_std[0] = np.std(group1)
        self.emission_std[1] = np.std(group2)

        # Uniform transition matrix
        self.transition = np.array([
            [0.95, 0.05],
            [0.05, 0.95]
        ])

    def _emission_prob(self, obs: float, state: Optional[int] = None) -> np.ndarray:
        """Compute emission probability."""
        if state is not None:
            # Single state
            diff = obs - self.emission_mean[state]
            return -0.5 * (diff / self.emission_std[state]) ** 2
        else:
            # All states
            diff = obs - self.emission_mean
            return -0.5 * (diff / self.emission_std) ** 2

    def _forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward algorithm."""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        # Initialize
        alpha[0] = np.log(self.state_prob) + self._emission_prob(observations[0])

        # Forward pass
        for t in range(1, T):
            for s in range(self.n_states):
                trans_prob = alpha[t - 1] + np.log(self.transition[:, s])
                alpha[t, s] = np.log(np.sum(np.exp(trans_prob - np.max(trans_prob)))) + \
                             np.max(trans_prob) + self._emission_prob(observations[t], s)

        # Log-likelihood
        ll = np.log(np.sum(np.exp(alpha[-1] - np.max(alpha[-1])))) + np.max(alpha[-1])

        return alpha, ll

    def _backward(self, observations: np.ndarray) -> np.ndarray:
        """Backward algorithm."""
        T = len(observations)
        beta = np.zeros((T, self.n_states))

        # Initialize
        beta[-1] = 0.0

        # Backward pass
        for t in range(T - 2, -1, -1):
            for s in range(self.n_states):
                trans_prob = np.log(self.transition[s]) + \
                            self._emission_prob(observations[t + 1]) + beta[t + 1]
                beta[t, s] = np.log(np.sum(np.exp(trans_prob - np.max(trans_prob)))) + \
                            np.max(trans_prob)

        return beta

    def _compute_gamma(self, alpha: np.ndarray, beta: np.ndarray, ll: float) -> np.ndarray:
        """Compute state posteriors (gamma)."""
        gamma = alpha + beta
        gamma = gamma - np.max(gamma, axis=1, keepdims=True)
        gamma = np.exp(gamma)
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        return gamma

    def _compute_xi(self, observations: np.ndarray, alpha: np.ndarray,
                   beta: np.ndarray, ll: float) -> np.ndarray:
        """Compute transition posteriors (xi)."""
        T = len(observations)
        xi = np.zeros((T - 1, self.n_states, self.n_states))

        for t in range(T - 1):
            for s1 in range(self.n_states):
                for s2 in range(self.n_states):
                    xi[t, s1, s2] = alpha[t, s1] + \
                                   np.log(self.transition[s1, s2]) + \
                                   self._emission_prob(observations[t + 1], s2) + \
                                   beta[t + 1, s2]

            xi[t] = xi[t] - np.max(xi[t])
            xi[t] = np.exp(xi[t])
            xi[t] = xi[t] / np.sum(xi[t])

        return xi

    def is_mean_reverting(self, observations: np.ndarray, threshold: float = 0.6) -> bool:
        """
        Check if market is currently in mean-reverting regime.

        Parameters:
        -----------
        observations : array-like
            Recent observation sequence
        threshold : float
            Confidence threshold

        Returns:
        --------
        True if mean-reverting regime with sufficient confidence
        """
        regime = self.get_current_regime(observations)
        return regime.regime == 0 and regime.probability >= threshold
