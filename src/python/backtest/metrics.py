"""
Performance metrics calculation for strategy evaluation.

Computes: Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, NamedTuple


class PerformanceMetrics(NamedTuple):
    """Complete performance metrics set."""
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade_duration: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    consecutive_wins: int
    consecutive_losses: int


class MetricsCalculator:
    """Calculate performance metrics from backtest results."""

    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.04,
    ) -> float:
        """
        Calculate Sharpe ratio.

        Sharpe = (mean_return - rf_rate) / std_return * sqrt(periods_per_year)

        Parameters:
        -----------
        returns : array-like
            Daily returns
        periods_per_year : int
            Trading periods per year (252 for daily)
        risk_free_rate : float
            Annual risk-free rate

        Returns:
        --------
        Sharpe ratio
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret == 0:
            return 0.0

        # Annualize
        annual_mean = mean_ret * periods_per_year
        annual_std = std_ret * np.sqrt(periods_per_year)

        return (annual_mean - risk_free_rate) / annual_std

    @staticmethod
    def calculate_sortino_ratio(
        returns: np.ndarray,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.04,
    ) -> float:
        """
        Calculate Sortino ratio (only penalizes downside volatility).

        Parameters:
        -----------
        returns : array-like
            Daily returns
        periods_per_year : int
            Trading periods per year
        risk_free_rate : float
            Annual risk-free rate

        Returns:
        --------
        Sortino ratio
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
        else:
            downside_std = 0.0

        if downside_std == 0:
            return 0.0

        # Annualize
        annual_mean = mean_ret * periods_per_year
        annual_downside = downside_std * np.sqrt(periods_per_year)

        return (annual_mean - risk_free_rate) / annual_downside

    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Parameters:
        -----------
        equity_curve : array-like
            Portfolio value over time

        Returns:
        --------
        Max drawdown as fraction (e.g., 0.20 = 20%)
        """
        equity_curve = np.asarray(equity_curve, dtype=np.float64)

        if len(equity_curve) < 2:
            return 0.0

        # Running maximum
        running_max = np.maximum.accumulate(equity_curve)

        # Drawdown at each point
        drawdown = (equity_curve - running_max) / running_max

        return float(np.min(drawdown))

    @staticmethod
    def calculate_calmar_ratio(
        returns: np.ndarray,
        equity_curve: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).

        Parameters:
        -----------
        returns : array-like
            Daily returns
        equity_curve : array-like
            Portfolio values
        periods_per_year : int
            Trading periods per year

        Returns:
        --------
        Calmar ratio
        """
        annual_return = np.mean(returns) * periods_per_year
        max_dd = abs(MetricsCalculator.calculate_max_drawdown(equity_curve))

        if max_dd == 0:
            return 0.0

        return annual_return / max_dd

    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> float:
        """
        Calculate win rate from trade list.

        Parameters:
        -----------
        trades : DataFrame
            DataFrame with 'pnl' column

        Returns:
        --------
        Win rate (0 to 1)
        """
        if len(trades) == 0:
            return 0.0

        winning = len(trades[trades['pnl'] > 0])
        return winning / len(trades)

    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame) -> float:
        """
        Calculate profit factor (sum of wins / abs(sum of losses)).

        Parameters:
        -----------
        trades : DataFrame
            DataFrame with 'pnl' column

        Returns:
        --------
        Profit factor
        """
        if len(trades) == 0:
            return 0.0

        wins = trades[trades['pnl'] > 0]['pnl'].sum()
        losses = abs(trades[trades['pnl'] < 0]['pnl'].sum())

        if losses == 0:
            return float('inf') if wins > 0 else 1.0

        return wins / losses

    @staticmethod
    def calculate_avg_trade_metrics(trades: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate average win, average loss, and average trade duration.

        Parameters:
        -----------
        trades : DataFrame
            DataFrame with 'pnl' and 'duration' columns

        Returns:
        --------
        (avg_win, avg_loss, avg_duration)
        """
        if len(trades) == 0:
            return 0.0, 0.0, 0.0

        wins = trades[trades['pnl'] > 0]['pnl']
        losses = trades[trades['pnl'] < 0]['pnl']

        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
        avg_duration = trades['duration'].mean() if 'duration' in trades.columns else 0.0

        return avg_win, avg_loss, avg_duration

    @staticmethod
    def calculate_consecutive_trades(trades: pd.DataFrame) -> Tuple[int, int]:
        """
        Calculate consecutive wins and losses.

        Parameters:
        -----------
        trades : DataFrame
            DataFrame with 'pnl' column

        Returns:
        --------
        (max_consecutive_wins, max_consecutive_losses)
        """
        if len(trades) == 0:
            return 0, 0

        is_win = (trades['pnl'] > 0).values.astype(int)

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for win in is_win:
            if win:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    @staticmethod
    def compute_all_metrics(
        equity_curve: np.ndarray,
        trades: pd.DataFrame,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.04,
    ) -> PerformanceMetrics:
        """
        Compute all performance metrics.

        Parameters:
        -----------
        equity_curve : array-like
            Portfolio values over time
        trades : DataFrame
            Trade details with 'pnl' column
        periods_per_year : int
            Trading periods per year
        risk_free_rate : float
            Risk-free rate

        Returns:
        --------
        PerformanceMetrics with all calculated values
        """
        equity_curve = np.asarray(equity_curve, dtype=np.float64)
        returns = np.diff(equity_curve) / equity_curve[:-1]

        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        annual_return = (1 + total_return) ** (1 / (len(equity_curve) / periods_per_year)) - 1
        annual_vol = np.std(returns) * np.sqrt(periods_per_year)

        sharpe = MetricsCalculator.calculate_sharpe_ratio(returns, periods_per_year, risk_free_rate)
        sortino = MetricsCalculator.calculate_sortino_ratio(returns, periods_per_year, risk_free_rate)
        max_dd = MetricsCalculator.calculate_max_drawdown(equity_curve)
        calmar = MetricsCalculator.calculate_calmar_ratio(returns, equity_curve, periods_per_year)

        win_rate = MetricsCalculator.calculate_win_rate(trades)
        profit_factor = MetricsCalculator.calculate_profit_factor(trades)
        avg_win, avg_loss, avg_duration = MetricsCalculator.calculate_avg_trade_metrics(trades)
        max_wins, max_losses = MetricsCalculator.calculate_consecutive_trades(trades)

        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] < 0])

        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            annual_volatility=annual_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration=avg_duration,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            consecutive_wins=max_wins,
            consecutive_losses=max_losses,
        )
