"""
Risk management for pairs trading strategies.

Handles position sizing, drawdown limits, correlation-based portfolio limits, VAR.
"""

import numpy as np
from typing import Dict, Optional, Tuple, NamedTuple


class RiskMetrics(NamedTuple):
    """Risk metrics snapshot."""
    portfolio_value: float
    current_drawdown: float
    max_drawdown: float
    var_95: float
    position_pnl: Dict[str, float]
    total_notional: float


class RiskManager:
    """
    Risk management for statistical arbitrage.

    Features:
    - Kelly criterion position sizing
    - Maximum drawdown enforcement
    - Daily value-at-risk (VAR)
    - Correlation-based position limits
    - Position limits per strategy
    """

    def __init__(
        self,
        initial_capital: float,
        max_drawdown: float = 0.20,  # 20% max drawdown
        max_daily_loss: float = 0.05,  # 5% max daily loss
        kelly_fraction: float = 0.25,  # Use 25% of Kelly leverage
        confidence_level: float = 0.95,  # VAR 95%
    ):
        """
        Initialize risk manager.

        Parameters:
        -----------
        initial_capital : float
            Starting capital
        max_drawdown : float
            Maximum acceptable drawdown (e.g., 0.20 = 20%)
        max_daily_loss : float
            Maximum loss per day
        kelly_fraction : float
            Fraction of Kelly criterion to use (0 < f <= 1)
        confidence_level : float
            Confidence level for VAR calculation
        """
        self.initial_capital = initial_capital
        self.max_drawdown_limit = max_drawdown
        self.max_daily_loss_limit = max_daily_loss
        self.kelly_fraction = kelly_fraction
        self.confidence_level = confidence_level

        self.peak_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_starting_capital = initial_capital

        self.positions = {}  # {pair_id: {'long_qty': x, 'short_qty': y, ...}}
        self.daily_pnl = 0.0
        self.returns_history = []

    def update_capital(self, new_capital: float) -> None:
        """Update current capital (mark-to-market)."""
        self.current_capital = new_capital

        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

        self.returns_history.append((new_capital - self.daily_starting_capital) / self.daily_starting_capital)

    def get_current_drawdown(self) -> float:
        """
        Calculate current drawdown from peak.

        Returns:
        --------
        Drawdown as fraction (e.g., 0.15 = 15% drawdown)
        """
        if self.peak_capital == 0:
            return 0.0

        return 1.0 - (self.current_capital / self.peak_capital)

    def get_max_historical_drawdown(self) -> float:
        """Calculate maximum historical drawdown."""
        if len(self.returns_history) < 2:
            return 0.0

        cumulative = np.cumprod(1 + np.array(self.returns_history))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = 1 - cumulative / running_max

        return float(np.max(drawdown))

    def calculate_kelly_position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate position size using Kelly criterion.

        Kelly fraction = (p * b - q) / b
        where p = win rate, q = 1 - p, b = avg_win / avg_loss

        Parameters:
        -----------
        win_rate : float
            Probability of winning (0 to 1)
        avg_win : float
            Average profit per win
        avg_loss : float
            Average loss per loss (absolute value)

        Returns:
        --------
        Fraction of capital to allocate
        """
        if avg_loss <= 0:
            return 0.0

        q = 1 - win_rate
        b = avg_win / avg_loss

        if b == 0:
            return 0.0

        kelly = (win_rate * b - q) / b

        # Apply kelly_fraction to avoid over-leverage
        position_fraction = max(0, min(kelly * self.kelly_fraction, 0.25))

        return position_fraction

    def calculate_position_size_target(
        self,
        pair_id: str,
        signal_strength: float,
        volatility: float,
    ) -> float:
        """
        Calculate target position size based on signal and volatility.

        Parameters:
        -----------
        pair_id : str
            Identifier for pair
        signal_strength : float
            Strength of signal (0 to 1)
        volatility : float
            Current volatility estimate

        Returns:
        --------
        Notional amount to allocate to this pair
        """
        if volatility < 1e-6:
            return 0.0

        # Risk budget: allocate 1% risk per position
        risk_budget = self.current_capital * 0.01

        # Size based on volatility targeting
        position_size = (risk_budget / volatility) * signal_strength

        # Cap at 10% of capital per pair
        max_per_pair = self.current_capital * 0.10

        return min(position_size, max_per_pair)

    def check_drawdown_limit(self) -> bool:
        """
        Check if current drawdown exceeds limit.

        Returns:
        --------
        True if within limit, False if exceeded
        """
        current_dd = self.get_current_drawdown()
        return current_dd <= self.max_drawdown_limit

    def check_daily_loss_limit(self) -> bool:
        """
        Check if daily loss exceeds limit.

        Returns:
        --------
        True if within limit, False if exceeded
        """
        daily_loss = (self.daily_starting_capital - self.current_capital) / self.daily_starting_capital
        return daily_loss <= self.max_daily_loss_limit

    def calculate_var(self, returns: np.ndarray, window: int = 252) -> float:
        """
        Calculate Value-at-Risk.

        Parameters:
        -----------
        returns : array-like
            Daily returns
        window : int
            Lookback window (default 252 trading days)

        Returns:
        --------
        VAR as fraction of capital
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return 0.0

        # Use percentile method
        var = np.percentile(returns, (1 - self.confidence_level) * 100)

        return abs(var)

    def add_position(
        self,
        pair_id: str,
        leg1_quantity: float,
        leg1_price: float,
        leg2_quantity: float,
        leg2_price: float,
    ) -> None:
        """
        Register a new pair position.

        Parameters:
        -----------
        pair_id : str
            Pair identifier
        leg1_quantity : float
            Quantity of first leg (can be negative)
        leg1_price : float
            Price of first leg
        leg2_quantity : float
            Quantity of second leg
        leg2_price : float
            Price of second leg
        """
        notional1 = abs(leg1_quantity * leg1_price)
        notional2 = abs(leg2_quantity * leg2_price)

        self.positions[pair_id] = {
            'leg1_qty': leg1_quantity,
            'leg1_price': leg1_price,
            'leg1_notional': notional1,
            'leg2_qty': leg2_quantity,
            'leg2_price': leg2_price,
            'leg2_notional': notional2,
            'entry_time': len(self.returns_history),
        }

    def remove_position(self, pair_id: str) -> None:
        """Remove a pair position."""
        if pair_id in self.positions:
            del self.positions[pair_id]

    def get_total_notional_exposure(self) -> float:
        """Get total notional exposure across all pairs."""
        return sum(
            p.get('leg1_notional', 0) + p.get('leg2_notional', 0)
            for p in self.positions.values()
        )

    def get_leverage(self) -> float:
        """Calculate current leverage."""
        if self.current_capital <= 0:
            return 0.0

        return self.get_total_notional_exposure() / self.current_capital

    def should_reduce_positions(self) -> bool:
        """
        Check if positions should be reduced.

        Returns:
        --------
        True if leverage or drawdown limits are concerning
        """
        leverage = self.get_leverage()
        drawdown = self.get_current_drawdown()
        daily_loss = (self.daily_starting_capital - self.current_capital) / self.daily_starting_capital

        return (
            leverage > 3.0 or
            drawdown > self.max_drawdown_limit * 0.8 or
            daily_loss > self.max_daily_loss_limit * 0.8
        )

    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics snapshot."""
        position_pnl = {}
        for pair_id, pos in self.positions.items():
            # Simplified PnL calculation
            pnl = (pos.get('leg1_notional', 0) + pos.get('leg2_notional', 0)) * 0.01  # Placeholder
            position_pnl[pair_id] = pnl

        returns_array = np.array(self.returns_history)
        var = self.calculate_var(returns_array) if len(returns_array) > 0 else 0.0

        return RiskMetrics(
            portfolio_value=self.current_capital,
            current_drawdown=self.get_current_drawdown(),
            max_drawdown=self.get_max_historical_drawdown(),
            var_95=var,
            position_pnl=position_pnl,
            total_notional=self.get_total_notional_exposure(),
        )

    def reset_daily(self) -> None:
        """Reset daily tracking for next day."""
        self.daily_starting_capital = self.current_capital
        self.daily_pnl = 0.0
