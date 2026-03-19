"""
Core pairs trading strategy implementation.

Combines cointegration analysis, signal generation, and risk management.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from .signal_generator import SignalGenerator, Signal
from .risk_manager import RiskManager
from ..research.kalman_filter import KalmanFilter


class PairsStrategy:
    """
    Complete pairs trading strategy.

    Workflow:
    1. Monitor spread between cointegrated pair
    2. Generate entry signals on extreme deviations
    3. Exit on mean reversion or stops
    4. Size positions based on risk and Kalman hedge ratio
    """

    def __init__(
        self,
        pair_id: str,
        symbol1: str,
        symbol2: str,
        hedge_ratio: float = 1.0,
        initial_capital: float = 1_000_000,
        entry_z_threshold: float = 2.0,
        exit_z_threshold: float = 0.5,
        use_kalman: bool = True,
        lookback: int = 20,
    ):
        """
        Initialize pairs strategy.

        Parameters:
        -----------
        pair_id : str
            Unique pair identifier
        symbol1 : str
            First asset symbol
        symbol2 : str
            Second asset symbol
        hedge_ratio : float
            Initial hedge ratio (price2 / price1)
        initial_capital : float
            Starting capital
        entry_z_threshold : float
            Entry signal z-score threshold
        exit_z_threshold : float
            Exit signal z-score threshold
        use_kalman : bool
            Whether to use Kalman filter for dynamic hedge ratio
        lookback : int
            Lookback window for spread calculation
        """
        self.pair_id = pair_id
        self.symbol1 = symbol1
        self.symbol2 = symbol2

        self.signal_generator = SignalGenerator(
            lookback=lookback,
            entry_threshold=entry_z_threshold,
            exit_threshold=exit_z_threshold,
        )

        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_drawdown=0.20,
            max_daily_loss=0.05,
        )

        # Kalman filter for dynamic hedge ratio
        self.use_kalman = use_kalman
        self.kalman = KalmanFilter(hedge_ratio) if use_kalman else None

        self.hedge_ratio = hedge_ratio
        self.prices1 = []
        self.prices2 = []
        self.spreads = []
        self.signals = []

        self.position_leg1 = 0.0
        self.position_leg2 = 0.0
        self.entry_price_spread = 0.0
        self.entry_time = None

    def update(
        self,
        price1: float,
        price2: float,
        timestamp: int = 0,
        volume1: float = 1.0,
        volume2: float = 1.0,
    ) -> Optional[Signal]:
        """
        Update strategy with new market data.

        Parameters:
        -----------
        price1 : float
            Price of first asset
        price2 : float
            Price of second asset
        timestamp : int
            Current timestamp
        volume1, volume2 : float
            Volume for position sizing

        Returns:
        --------
        Trading signal if generated
        """
        self.prices1.append(price1)
        self.prices2.append(price2)

        # Update Kalman filter hedge ratio
        if self.use_kalman and self.kalman and len(self.prices1) > 1:
            self.kalman.predict()
            _, spread = self.kalman.update(price1, price2)
        else:
            spread = price2 - self.hedge_ratio * price1

        self.spreads.append(spread)

        # Generate signal
        signal = self.signal_generator.update(spread, timestamp)

        if signal:
            signal = self._execute_signal(signal, price1, price2, timestamp)
            if signal:
                self.signals.append(signal)
                return signal

        return None

    def _execute_signal(
        self,
        signal: Signal,
        price1: float,
        price2: float,
        timestamp: int,
    ) -> Optional[Signal]:
        """
        Execute trading signal (position entry/exit).

        Parameters:
        -----------
        signal : Signal
            Generated signal
        price1, price2 : float
            Current prices
        timestamp : int
            Timestamp

        Returns:
        --------
        Executed signal or None if rejected by risk manager
        """
        if signal.signal_type == "entry_long":
            # Go long spread (long asset1, short asset2)
            if not self.risk_manager.check_daily_loss_limit():
                return None

            position_size = self._calculate_position_size(signal)
            self.position_leg1 = position_size
            self.position_leg2 = -position_size * self.hedge_ratio
            self.entry_price_spread = signal.spread
            self.entry_time = timestamp

            self.risk_manager.add_position(
                self.pair_id,
                self.position_leg1, price1,
                self.position_leg2, price2,
            )

            return signal

        elif signal.signal_type == "entry_short":
            # Go short spread (short asset1, long asset2)
            if not self.risk_manager.check_daily_loss_limit():
                return None

            position_size = self._calculate_position_size(signal)
            self.position_leg1 = -position_size
            self.position_leg2 = position_size * self.hedge_ratio
            self.entry_price_spread = signal.spread
            self.entry_time = timestamp

            self.risk_manager.add_position(
                self.pair_id,
                self.position_leg1, price1,
                self.position_leg2, price2,
            )

            return signal

        elif signal.signal_type in ["exit", "stop"]:
            if self.position_leg1 != 0:
                self.position_leg1 = 0.0
                self.position_leg2 = 0.0
                self.risk_manager.remove_position(self.pair_id)
                self.signal_generator.reset_position()

            return signal

        return None

    def _calculate_position_size(self, signal: Signal) -> float:
        """
        Calculate position size based on signal strength and risk limits.

        Parameters:
        -----------
        signal : Signal
            Trading signal

        Returns:
        --------
        Position size (notional in first asset)
        """
        capital = self.risk_manager.current_capital

        # Risk 1% of capital per trade
        risk_per_trade = capital * 0.01

        # Position size = risk / volatility
        volatility = np.std(self.spreads[-20:]) if len(self.spreads) >= 20 else 1.0

        if volatility < 1e-6:
            volatility = 1.0

        base_size = risk_per_trade / volatility

        # Scale by signal confidence
        scaled_size = base_size * signal.confidence

        # Cap at 5% of capital
        max_size = capital * 0.05

        return min(scaled_size, max_size)

    def get_current_pnl(self, price1: float, price2: float) -> float:
        """
        Calculate current unrealized PnL.

        Parameters:
        -----------
        price1, price2 : float
            Current prices

        Returns:
        --------
        Unrealized PnL in dollars
        """
        pnl = 0.0

        if self.position_leg1 != 0:
            pnl += self.position_leg1 * (price1 - self.prices1[-1])

        if self.position_leg2 != 0:
            pnl += self.position_leg2 * (price2 - self.prices2[-1])

        return pnl

    def get_position_metrics(self) -> Dict:
        """Get current position metrics."""
        is_long = self.position_leg1 > 0
        is_short = self.position_leg1 < 0

        return {
            'pair_id': self.pair_id,
            'is_positioned': self.position_leg1 != 0,
            'is_long': is_long,
            'is_short': is_short,
            'position_leg1': self.position_leg1,
            'position_leg2': self.position_leg2,
            'current_z_score': self.signal_generator.get_current_z_score(),
            'entry_spread': self.entry_price_spread,
            'current_spread': self.spreads[-1] if self.spreads else 0.0,
            'spread_pnl': (self.spreads[-1] - self.entry_price_spread) if self.spreads and self.position_leg1 != 0 else 0.0,
            'hedge_ratio': self.kalman.get_hedge_ratio() if self.kalman else self.hedge_ratio,
        }

    def get_strategy_stats(self) -> Dict:
        """Get strategy statistics."""
        if not self.signals:
            return {}

        signal_types = [s.signal_type for s in self.signals]
        entries = len([s for s in signal_types if 'entry' in s])
        exits = len([s for s in signal_types if s in ['exit', 'stop']])

        return {
            'pair_id': self.pair_id,
            'total_signals': len(self.signals),
            'entries': entries,
            'exits': exits,
            'current_exposure': self.risk_manager.get_total_notional_exposure(),
            'leverage': self.risk_manager.get_leverage(),
            'current_drawdown': self.risk_manager.get_current_drawdown(),
        }

    def reset(self) -> None:
        """Reset strategy to initial state."""
        self.position_leg1 = 0.0
        self.position_leg2 = 0.0
        self.signal_generator.reset_position()
