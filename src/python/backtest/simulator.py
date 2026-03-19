"""
Market impact and slippage simulation for realistic backtesting.

Simulates partial fills, slippage, commissions, and market impact.
"""

import numpy as np
from typing import Dict, Optional, NamedTuple
from enum import Enum


class SlippageModel(Enum):
    """Slippage modeling approach."""
    LINEAR = 1         # Fixed percentage
    SQUARE_ROOT = 2    # Proportional to sqrt of notional
    VOLUME_BASED = 3   # Based on market volume


class Fill(NamedTuple):
    """Execution fill details."""
    price: float
    quantity: int
    commission: float
    slippage: float


class MarketSimulator:
    """
    Simulate realistic market execution.

    Features:
    - Configurable slippage models
    - Commission schedules
    - Market impact estimation
    - Partial fill simulation
    - Time-in-force handling
    """

    def __init__(
        self,
        commission_rate: float = 0.001,  # 0.1% per side
        slippage_model: SlippageModel = SlippageModel.LINEAR,
        slippage_rate: float = 0.0001,  # 1 basis point for linear
        market_impact: float = 0.0001,  # 1 basis point per $1M
        partial_fill_rate: float = 1.0,  # 100% fill by default
    ):
        """
        Initialize market simulator.

        Parameters:
        -----------
        commission_rate : float
            Commission per side (e.g., 0.001 = 0.1%)
        slippage_model : SlippageModel
            How to model slippage
        slippage_rate : float
            Slippage rate or base rate
        market_impact : float
            Market impact coefficient
        partial_fill_rate : float
            Probability of full fill (0 to 1)
        """
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.slippage_rate = slippage_rate
        self.market_impact = market_impact
        self.partial_fill_rate = partial_fill_rate

    def execute_order(
        self,
        side: str,
        quantity: int,
        mid_price: float,
        bid_price: float,
        ask_price: float,
        volume: int,
        order_type: str = "market",
    ) -> Fill:
        """
        Simulate order execution.

        Parameters:
        -----------
        side : str
            'BUY' or 'SELL'
        quantity : int
            Order quantity
        mid_price : float
            Mid price
        bid_price : float
            Current bid
        ask_price : float
            Current ask
        volume : int
            Market volume
        order_type : str
            'market' or 'limit'

        Returns:
        --------
        Fill with actual execution price and quantity
        """
        # Determine execution price based on side
        if side.upper() == "BUY":
            execution_price = ask_price
            spread_cost = (ask_price - bid_price) / mid_price
        else:  # SELL
            execution_price = bid_price
            spread_cost = (ask_price - bid_price) / mid_price

        # Apply slippage
        slippage = self._calculate_slippage(quantity, mid_price, volume)
        if side.upper() == "BUY":
            execution_price += slippage
        else:
            execution_price -= slippage

        # Apply market impact
        impact = self._calculate_market_impact(quantity, mid_price)
        if side.upper() == "BUY":
            execution_price += impact
        else:
            execution_price -= impact

        # Calculate actual quantity filled
        fill_quantity = self._calculate_fill_quantity(quantity)

        # Calculate commission
        commission = execution_price * fill_quantity * self.commission_rate

        return Fill(
            price=execution_price,
            quantity=fill_quantity,
            commission=commission,
            slippage=slippage,
        )

    def _calculate_slippage(self, quantity: int, price: float, volume: int) -> float:
        """
        Calculate slippage based on model.

        Parameters:
        -----------
        quantity : int
            Order quantity
        price : float
            Current price
        volume : int
            Market volume

        Returns:
        --------
        Slippage in dollars per share
        """
        notional = quantity * price

        if self.slippage_model == SlippageModel.LINEAR:
            # Fixed slippage
            return self.slippage_rate * price

        elif self.slippage_model == SlippageModel.SQUARE_ROOT:
            # Proportional to sqrt of notional
            millions = notional / 1_000_000
            return self.slippage_rate * price * np.sqrt(max(millions, 1))

        elif self.slippage_model == SlippageModel.VOLUME_BASED:
            # Based on ratio of order to market volume
            if volume > 0:
                order_ratio = quantity / volume
                return self.slippage_rate * price * order_ratio
            else:
                return self.slippage_rate * price

        return 0.0

    def _calculate_market_impact(self, quantity: int, price: float) -> float:
        """
        Calculate market impact.

        Parameters:
        -----------
        quantity : int
            Order quantity
        price : float
            Current price

        Returns:
        --------
        Market impact in dollars per share
        """
        notional = quantity * price
        millions = notional / 1_000_000

        # Impact scales with order size
        impact_bps = self.market_impact * millions * 100  # Convert to bps

        return (impact_bps / 10000) * price

    def _calculate_fill_quantity(self, requested_quantity: int) -> int:
        """
        Calculate actual filled quantity.

        Parameters:
        -----------
        requested_quantity : int
            Requested order quantity

        Returns:
        --------
        Actual filled quantity
        """
        if np.random.random() < self.partial_fill_rate:
            # Full fill
            return requested_quantity
        else:
            # Partial fill (50-100%)
            fill_ratio = np.random.uniform(0.5, 1.0)
            return int(requested_quantity * fill_ratio)

    def simulate_order_book_impact(
        self,
        side: str,
        quantity: int,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
    ) -> Tuple[float, int]:
        """
        Simulate execution against order book levels.

        Parameters:
        -----------
        side : str
            'BUY' or 'SELL'
        quantity : int
            Order quantity
        bid_prices, bid_volumes : arrays
            Order book bid side
        ask_prices, ask_volumes : arrays
            Order book ask side

        Returns:
        --------
        (average_price, filled_quantity)
        """
        remaining = quantity
        total_cost = 0.0
        filled = 0

        if side.upper() == "BUY":
            # Take from ask side
            for price, vol in zip(ask_prices, ask_volumes):
                fill_size = min(remaining, vol)
                total_cost += fill_size * price
                filled += fill_size
                remaining -= fill_size

                if remaining == 0:
                    break

        else:  # SELL
            # Take from bid side
            for price, vol in zip(bid_prices, bid_volumes):
                fill_size = min(remaining, vol)
                total_cost += fill_size * price
                filled += fill_size
                remaining -= fill_size

                if remaining == 0:
                    break

        avg_price = total_cost / filled if filled > 0 else 0.0

        return avg_price, filled

    def set_slippage_model(
        self,
        model: SlippageModel,
        rate: float,
    ) -> None:
        """Update slippage model parameters."""
        self.slippage_model = model
        self.slippage_rate = rate

    def set_commission(self, rate: float) -> None:
        """Update commission rate."""
        self.commission_rate = rate

    def set_market_impact(self, coefficient: float) -> None:
        """Update market impact coefficient."""
        self.market_impact = coefficient
