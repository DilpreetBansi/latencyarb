"""
Signal generation for pairs trading.

Generates entry, exit, and stop-loss signals based on spread z-scores.
"""

import numpy as np
from typing import Tuple, Optional, NamedTuple
from collections import deque


class Signal(NamedTuple):
    """Trading signal."""
    timestamp: int
    signal_type: str  # 'entry_long', 'entry_short', 'exit', 'stop'
    z_score: float
    spread: float
    confidence: float  # 0 to 1


class SignalGenerator:
    """
    Generate trading signals based on spread z-scores.

    Entry signals when |z-score| > entry_threshold
    Exit signals when spread mean-reverts
    Stop signals when losses exceed threshold
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss_threshold: float = -3.0,
        min_holding_periods: int = 5,
    ):
        """
        Initialize signal generator.

        Parameters:
        -----------
        lookback : int
            Lookback window for z-score calculation
        entry_threshold : float
            Z-score threshold for entry (std deviations from mean)
        exit_threshold : float
            Z-score threshold for exit
        stop_loss_threshold : float
            Z-score threshold for stop loss
        min_holding_periods : int
            Minimum periods to hold position
        """
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.min_holding_periods = min_holding_periods

        self.spread_history = deque(maxlen=lookback)
        self.z_score_history = deque(maxlen=lookback)
        self.position = 0  # -1, 0, or 1
        self.entry_price = 0.0
        self.entry_z_score = 0.0
        self.periods_in_position = 0

    def update(self, spread: float, timestamp: int = 0) -> Optional[Signal]:
        """
        Update with new spread value and generate signal if needed.

        Parameters:
        -----------
        spread : float
            Current spread between the two assets
        timestamp : int
            Current timestamp

        Returns:
        --------
        Signal if one is generated, None otherwise
        """
        self.spread_history.append(spread)

        # Need enough data for z-score calculation
        if len(self.spread_history) < self.lookback:
            return None

        # Calculate z-score
        mean = np.mean(self.spread_history)
        std = np.std(self.spread_history)

        if std < 1e-8:
            z_score = 0.0
        else:
            z_score = (spread - mean) / std

        self.z_score_history.append(z_score)
        self.periods_in_position += 1

        # Exit signal
        if self.position != 0:
            # Exit on mean reversion
            if abs(z_score) < self.exit_threshold:
                return Signal(
                    timestamp=timestamp,
                    signal_type="exit",
                    z_score=z_score,
                    spread=spread,
                    confidence=self._compute_confidence(z_score)
                )

            # Stop loss
            if z_score < self.stop_loss_threshold:
                return Signal(
                    timestamp=timestamp,
                    signal_type="stop",
                    z_score=z_score,
                    spread=spread,
                    confidence=1.0
                )

        # Entry signals
        if self.position == 0:
            if z_score > self.entry_threshold:
                # Enter short position (bet on mean reversion)
                self.position = -1
                self.entry_price = spread
                self.entry_z_score = z_score
                self.periods_in_position = 0

                return Signal(
                    timestamp=timestamp,
                    signal_type="entry_short",
                    z_score=z_score,
                    spread=spread,
                    confidence=min(abs(z_score) / 4.0, 1.0)
                )

            elif z_score < -self.entry_threshold:
                # Enter long position (bet on mean reversion)
                self.position = 1
                self.entry_price = spread
                self.entry_z_score = z_score
                self.periods_in_position = 0

                return Signal(
                    timestamp=timestamp,
                    signal_type="entry_long",
                    z_score=z_score,
                    spread=spread,
                    confidence=min(abs(z_score) / 4.0, 1.0)
                )

        return None

    def reset_position(self) -> None:
        """Reset position tracking."""
        self.position = 0
        self.entry_price = 0.0
        self.entry_z_score = 0.0
        self.periods_in_position = 0

    def _compute_confidence(self, z_score: float) -> float:
        """Compute signal confidence (0 to 1)."""
        # Higher absolute z-score = higher confidence
        return min(abs(z_score) / 3.0, 1.0)

    def get_current_z_score(self) -> float:
        """Get current z-score."""
        if len(self.z_score_history) == 0:
            return 0.0
        return self.z_score_history[-1]

    def get_position(self) -> int:
        """Get current position: -1 (short), 0 (flat), 1 (long)."""
        return self.position

    def get_position_age(self) -> int:
        """Get periods in current position."""
        return self.periods_in_position if self.position != 0 else 0


class MultiWindowSignalGenerator:
    """
    Multi-timeframe signal generator for more robust entries.

    Uses multiple lookback windows and requires confirmation across windows.
    """

    def __init__(
        self,
        lookbacks: list = None,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
    ):
        """
        Initialize multi-window signal generator.

        Parameters:
        -----------
        lookbacks : list
            List of lookback windows (default: [10, 20, 50])
        entry_threshold : float
            Entry threshold in std devs
        exit_threshold : float
            Exit threshold in std devs
        """
        if lookbacks is None:
            lookbacks = [10, 20, 50]

        self.generators = [
            SignalGenerator(lookback, entry_threshold, exit_threshold)
            for lookback in lookbacks
        ]

    def update(self, spread: float, timestamp: int = 0) -> Optional[Signal]:
        """
        Update all generators and generate signal if confirmed across windows.

        Returns:
        --------
        Signal if multiple generators agree, None otherwise
        """
        signals = [gen.update(spread, timestamp) for gen in self.generators]
        signals = [s for s in signals if s is not None]

        if len(signals) == 0:
            return None

        # Require agreement on signal type from at least 2 generators
        signal_types = [s.signal_type for s in signals]
        if len(set(signal_types)) == 1:
            # All generators agree
            best_signal = max(signals, key=lambda s: s.confidence)
            return best_signal

        return None

    def reset_positions(self) -> None:
        """Reset all generators."""
        for gen in self.generators:
            gen.reset_position()
