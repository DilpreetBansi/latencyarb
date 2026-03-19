"""Unit tests for backtesting engine."""

import numpy as np
import pytest
import sys
sys.path.insert(0, '../src')

from python.backtest.engine import BacktestEngine
from python.backtest.metrics import MetricsCalculator


class TestBacktestEngine:
    """Test suite for backtesting engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = BacktestEngine(initial_capital=100_000)

        assert engine.equity == 100_000
        assert engine.cash == 100_000
        assert len(engine.positions) == 0

    def test_submit_order(self):
        """Test order submission."""
        engine = BacktestEngine(initial_capital=100_000)

        success = engine.submit_order(
            pair_id="TEST-1",
            leg1_symbol="ASSET1",
            leg1_qty=100,
            leg1_price=100.0,
            leg2_symbol="ASSET2",
            leg2_qty=-150,
            leg2_price=66.67,
        )

        assert success
        assert "TEST-1" in engine.positions

    def test_insufficient_cash(self):
        """Test rejection when insufficient cash."""
        engine = BacktestEngine(initial_capital=1_000)

        success = engine.submit_order(
            pair_id="TEST-1",
            leg1_symbol="ASSET1",
            leg1_qty=100,
            leg1_price=100.0,  # Requires 10k
            leg2_symbol="ASSET2",
            leg2_qty=-150,
            leg2_price=66.67,
        )

        assert not success

    def test_position_tracking(self):
        """Test position tracking."""
        engine = BacktestEngine(initial_capital=100_000)

        engine.submit_order(
            pair_id="PAIR1",
            leg1_symbol="A",
            leg1_qty=100,
            leg1_price=50.0,
            leg2_symbol="B",
            leg2_qty=-100,
            leg2_price=50.0,
        )

        pos = engine.positions["PAIR1"]
        assert pos['leg1_qty'] == 100
        assert pos['leg2_qty'] == -100

    def test_close_position(self):
        """Test position closing and PnL realization."""
        engine = BacktestEngine(initial_capital=100_000)

        # Open position
        engine.submit_order(
            pair_id="PAIR1",
            leg1_symbol="A",
            leg1_qty=100,
            leg1_price=100.0,
            leg2_symbol="B",
            leg2_qty=-100,
            leg2_price=100.0,
        )

        # Close at profit
        pnl = engine.close_position("PAIR1", 110.0, 90.0)

        assert pnl > 0  # Should have profit
        assert "PAIR1" not in engine.positions

    def test_mtm_update(self):
        """Test mark-to-market update."""
        engine = BacktestEngine(initial_capital=100_000)

        # Open position
        engine.submit_order(
            pair_id="PAIR1",
            leg1_symbol="A",
            leg1_qty=100,
            leg1_price=100.0,
            leg2_symbol="B",
            leg2_qty=-100,
            leg2_price=100.0,
        )

        initial_equity = engine.equity

        # Update prices (profit scenario)
        engine._update_mtm({
            "PAIR1_leg1": 110.0,
            "PAIR1_leg2": 90.0,
        })

        # Equity should increase
        assert engine.equity > initial_equity

    def test_drawdown_tracking(self):
        """Test drawdown calculation."""
        engine = BacktestEngine(initial_capital=100_000)

        # Simulate loss
        engine.equity = 80_000

        dd = engine.get_current_drawdown()

        assert dd == 0.2, f"Expected 20% drawdown, got {dd}"

    def test_returns_calculation(self):
        """Test returns calculation."""
        engine = BacktestEngine(initial_capital=100_000)

        # Simulate equity curve
        engine.equity_curve = [100_000, 105_000, 103_000, 108_000]

        returns = engine.get_returns()

        assert len(returns) == 3
        assert returns[0] == pytest.approx(0.05)  # 5%
        assert returns[1] == pytest.approx(-0.019047619, abs=1e-6)  # -1.9%

    def test_summary_stats(self):
        """Test summary statistics."""
        engine = BacktestEngine(initial_capital=100_000)

        # Simulate equity curve
        engine.equity_curve = np.linspace(100_000, 120_000, 100).tolist()

        summary = engine.get_summary()

        assert 'total_return' in summary
        assert 'annual_return' in summary
        assert 'sharpe_ratio' in summary
        assert 'max_drawdown' in summary


class TestMetricsCalculator:
    """Test suite for metrics calculator."""

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Synthetic returns with positive mean
        returns = np.random.randn(252) * 0.01 + 0.001

        sharpe = MetricsCalculator.calculate_sharpe_ratio(returns)

        assert isinstance(sharpe, float)
        assert sharpe > 0  # Positive return should give positive Sharpe

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = np.random.randn(252) * 0.01 + 0.001

        sortino = MetricsCalculator.calculate_sortino_ratio(returns)

        assert isinstance(sortino, float)
        assert sortino > 0

    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        equity = np.array([100, 120, 110, 130, 90, 100])

        max_dd = MetricsCalculator.calculate_max_drawdown(equity)

        # From peak 130 to low 90
        expected_dd = (90 - 130) / 130
        assert max_dd == expected_dd

    def test_win_rate(self):
        """Test win rate calculation."""
        import pandas as pd

        trades = pd.DataFrame({
            'pnl': [100, -50, 200, -100, 150]
        })

        wr = MetricsCalculator.calculate_win_rate(trades)

        assert wr == 0.6  # 3 wins out of 5

    def test_profit_factor(self):
        """Test profit factor calculation."""
        import pandas as pd

        trades = pd.DataFrame({
            'pnl': [100, -50, 200, -100, 150]
        })

        pf = MetricsCalculator.calculate_profit_factor(trades)

        # (100 + 200 + 150) / (50 + 100) = 450 / 150 = 3.0
        assert pf == 3.0

    def test_consecutive_trades(self):
        """Test consecutive win/loss tracking."""
        import pandas as pd

        trades = pd.DataFrame({
            'pnl': [100, 150, -50, -100, 200, 50]
        })

        max_wins, max_losses = MetricsCalculator.calculate_consecutive_trades(trades)

        assert max_wins == 2  # First two trades
        assert max_losses == 2  # Middle two trades


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
