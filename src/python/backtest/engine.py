"""
Event-driven backtesting engine.

Processes market data chronologically, executes strategy, tracks positions and PnL.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class BacktestEngine:
    """
    Event-driven backtesting framework.

    Architecture:
    - Process market events chronologically
    - Track positions, fills, orders
    - Calculate PnL and metrics
    - Support multiple strategy instances
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0001,
    ):
        """
        Initialize backtest engine.

        Parameters:
        -----------
        initial_capital : float
            Starting capital
        commission_rate : float
            Commission per side
        slippage_rate : float
            Slippage rate
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # State
        self.cash = initial_capital
        self.equity = initial_capital
        self.positions = {}  # {pair_id: {leg1_qty, leg2_qty, ...}}
        self.orders = []
        self.fills = []
        self.trades = []  # Closed trades

        # History
        self.equity_curve = []
        self.timestamps = []
        self.daily_returns = []

        self.peak_equity = initial_capital

    def process_market_data(
        self,
        timestamp: str,
        prices: Dict[str, float],
    ) -> None:
        """
        Process market data point.

        Parameters:
        -----------
        timestamp : str
            Timestamp (YYYY-MM-DD HH:MM:SS)
        prices : dict
            {pair_id: price}
        """
        # Update mark-to-market
        self._update_mtm(prices)

        # Record equity
        self.equity_curve.append(self.equity)
        self.timestamps.append(timestamp)

    def submit_order(
        self,
        pair_id: str,
        leg1_symbol: str,
        leg1_qty: float,
        leg1_price: float,
        leg2_symbol: str,
        leg2_qty: float,
        leg2_price: float,
        timestamp: str = None,
    ) -> bool:
        """
        Submit paired order.

        Parameters:
        -----------
        pair_id : str
            Pair identifier
        leg1_symbol, leg2_symbol : str
            Asset symbols
        leg1_qty, leg2_qty : float
            Quantities (can be negative)
        leg1_price, leg2_price : float
            Current prices
        timestamp : str
            Order timestamp

        Returns:
        --------
        True if order accepted, False otherwise
        """
        # Check cash
        notional = abs(leg1_qty * leg1_price) + abs(leg2_qty * leg2_price)
        commission = notional * self.commission_rate

        if self.cash < notional + commission:
            return False

        # Create order
        order = {
            'pair_id': pair_id,
            'leg1_symbol': leg1_symbol,
            'leg1_qty': leg1_qty,
            'leg1_price': leg1_price,
            'leg2_symbol': leg2_symbol,
            'leg2_qty': leg2_qty,
            'leg2_price': leg2_price,
            'timestamp': timestamp,
            'status': 'pending',
        }

        self.orders.append(order)

        # Execute immediately (market order simulation)
        return self._fill_order(order)

    def _fill_order(self, order: Dict) -> bool:
        """Fill an order."""
        pair_id = order['pair_id']

        # Apply slippage
        leg1_fill_price = order['leg1_price'] * (1 + self.slippage_rate)
        leg2_fill_price = order['leg2_price'] * (1 + self.slippage_rate)

        # Calculate costs
        leg1_cost = order['leg1_qty'] * leg1_fill_price
        leg2_cost = order['leg2_qty'] * leg2_fill_price
        total_cost = abs(leg1_cost) + abs(leg2_cost)
        commission = total_cost * self.commission_rate

        # Update cash
        self.cash -= (leg1_cost + leg2_cost + commission)

        # Update position
        if pair_id not in self.positions:
            self.positions[pair_id] = {
                'leg1_qty': 0,
                'leg1_entry_price': 0,
                'leg2_qty': 0,
                'leg2_entry_price': 0,
            }

        pos = self.positions[pair_id]

        # Average entry prices
        if pos['leg1_qty'] != 0:
            pos['leg1_entry_price'] = (
                (pos['leg1_entry_price'] * pos['leg1_qty'] + leg1_fill_price * order['leg1_qty']) /
                (pos['leg1_qty'] + order['leg1_qty'])
            )
        else:
            pos['leg1_entry_price'] = leg1_fill_price

        if pos['leg2_qty'] != 0:
            pos['leg2_entry_price'] = (
                (pos['leg2_entry_price'] * pos['leg2_qty'] + leg2_fill_price * order['leg2_qty']) /
                (pos['leg2_qty'] + order['leg2_qty'])
            )
        else:
            pos['leg2_entry_price'] = leg2_fill_price

        pos['leg1_qty'] += order['leg1_qty']
        pos['leg2_qty'] += order['leg2_qty']

        # Record fill
        fill = {
            'pair_id': pair_id,
            'timestamp': order['timestamp'],
            'leg1_qty': order['leg1_qty'],
            'leg1_price': leg1_fill_price,
            'leg2_qty': order['leg2_qty'],
            'leg2_price': leg2_fill_price,
            'commission': commission,
        }

        self.fills.append(fill)
        order['status'] = 'filled'

        return True

    def close_position(
        self,
        pair_id: str,
        leg1_price: float,
        leg2_price: float,
        timestamp: str = None,
    ) -> float:
        """
        Close a position and realize PnL.

        Parameters:
        -----------
        pair_id : str
            Position to close
        leg1_price, leg2_price : float
            Current prices
        timestamp : str
            Close timestamp

        Returns:
        --------
        Realized PnL
        """
        if pair_id not in self.positions:
            return 0.0

        pos = self.positions[pair_id]

        # Calculate PnL
        leg1_pnl = pos['leg1_qty'] * (leg1_price - pos['leg1_entry_price'])
        leg2_pnl = pos['leg2_qty'] * (leg2_price - pos['leg2_entry_price'])
        total_pnl = leg1_pnl + leg2_pnl

        # Commission
        notional = abs(pos['leg1_qty'] * leg1_price) + abs(pos['leg2_qty'] * leg2_price)
        commission = notional * self.commission_rate

        realized_pnl = total_pnl - commission

        # Update cash
        close_proceeds = (
            pos['leg1_qty'] * leg1_price +
            pos['leg2_qty'] * leg2_price
        )
        self.cash += close_proceeds

        # Record trade
        trade = {
            'pair_id': pair_id,
            'entry_time': None,
            'close_time': timestamp,
            'entry_price_leg1': pos['leg1_entry_price'],
            'entry_price_leg2': pos['leg2_entry_price'],
            'close_price_leg1': leg1_price,
            'close_price_leg2': leg2_price,
            'pnl': realized_pnl,
            'duration': 0,
        }

        self.trades.append(trade)

        # Remove position
        del self.positions[pair_id]

        return realized_pnl

    def _update_mtm(self, prices: Dict[str, float]) -> None:
        """Update mark-to-market valuation."""
        mtm_equity = self.cash

        for pair_id, pos in self.positions.items():
            if pair_id in prices:
                leg1_val = pos['leg1_qty'] * prices.get(pair_id + '_leg1', 0)
                leg2_val = pos['leg2_qty'] * prices.get(pair_id + '_leg2', 0)
                mtm_equity += leg1_val + leg2_val

        self.equity = mtm_equity
        self.peak_equity = max(self.peak_equity, self.equity)

    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak."""
        if self.peak_equity == 0:
            return 0.0

        return 1.0 - (self.equity / self.peak_equity)

    def get_returns(self) -> np.ndarray:
        """Get daily returns."""
        if len(self.equity_curve) < 2:
            return np.array([])

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        return returns

    def get_summary(self) -> Dict:
        """Get backtest summary statistics."""
        returns = self.get_returns()

        if len(returns) == 0:
            return {}

        total_return = (self.equity - self.initial_capital) / self.initial_capital
        annual_return = total_return * (252 / len(returns)) if len(returns) > 0 else 0

        sharpe = (
            (np.mean(returns) * 252 - 0.04) / (np.std(returns) * np.sqrt(252))
            if np.std(returns) > 0 else 0
        )

        max_dd = 0
        if len(self.equity_curve) > 0:
            eq = np.array(self.equity_curve)
            running_max = np.maximum.accumulate(eq)
            max_dd = np.min((eq - running_max) / running_max)

        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.equity,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': abs(max_dd),
            'total_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t['pnl'] > 0]),
            'losing_trades': len([t for t in self.trades if t['pnl'] < 0]),
        }

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(self.trades)

    def reset(self) -> None:
        """Reset engine to initial state."""
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.positions = {}
        self.orders = []
        self.fills = []
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        self.peak_equity = self.initial_capital
