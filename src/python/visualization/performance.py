"""Performance visualization (Plotly)."""

import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import Optional, Dict


class PerformanceVisualizer:
    """Create interactive performance charts."""

    @staticmethod
    def plot_equity_curve(
        equity: np.ndarray,
        timestamps: list = None,
        title: str = "Portfolio Equity Curve",
        benchmark: np.ndarray = None,
    ) -> go.Figure:
        """
        Plot equity curve with optional benchmark.

        Parameters:
        -----------
        equity : array-like
            Portfolio values over time
        timestamps : list
            Timestamps
        title : str
            Chart title
        benchmark : array-like, optional
            Benchmark values for comparison

        Returns:
        --------
        Plotly Figure
        """
        if timestamps is None:
            timestamps = range(len(equity))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=equity,
            name='Strategy',
            mode='lines',
            line=dict(color='blue', width=2),
            hovertemplate='<b>%{text}</b><br>Equity: $%{y:,.0f}<extra></extra>',
            text=timestamps,
        ))

        if benchmark is not None:
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=benchmark,
                name='Benchmark',
                mode='lines',
                line=dict(color='red', dash='dash', width=1),
                hovertemplate='<b>%{text}</b><br>Benchmark: $%{y:,.0f}<extra></extra>',
                text=timestamps,
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_white',
            hovermode='x unified',
            height=600,
        )

        return fig

    @staticmethod
    def plot_drawdown(
        equity: np.ndarray,
        timestamps: list = None,
        title: str = "Drawdown Chart",
    ) -> go.Figure:
        """Plot drawdown curve."""
        if timestamps is None:
            timestamps = range(len(equity))

        equity = np.asarray(equity)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=drawdown,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red'),
            hovertemplate='<b>%{text}</b><br>Drawdown: %{y:.2f}%<extra></extra>',
            text=timestamps,
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_white',
            height=500,
        )

        return fig

    @staticmethod
    def plot_returns_distribution(
        returns: np.ndarray,
        title: str = "Returns Distribution",
    ) -> go.Figure:
        """Plot histogram of returns."""
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns',
            marker=dict(color='blue', opacity=0.7),
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Daily Returns (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=500,
        )

        return fig

    @staticmethod
    def plot_monthly_returns(
        returns: np.ndarray,
        timestamps: pd.DatetimeIndex,
        title: str = "Monthly Returns Heatmap",
    ) -> go.Figure:
        """Plot monthly returns heatmap."""
        # Convert to monthly returns
        df = pd.DataFrame({
            'date': timestamps,
            'returns': returns,
        })

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Aggregate to monthly
        monthly = df.groupby(['year', 'month'])['returns'].sum() * 100

        # Create pivot for heatmap
        monthly_df = monthly.reset_index()
        pivot = monthly_df.pivot(index='month', columns='year', values='returns')

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlGn',
                zmid=0,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title='Year',
            yaxis_title='Month',
            height=400,
        )

        return fig

    @staticmethod
    def plot_cumulative_returns(
        returns: np.ndarray,
        timestamps: list = None,
        title: str = "Cumulative Returns",
    ) -> go.Figure:
        """Plot cumulative returns."""
        if timestamps is None:
            timestamps = range(len(returns))

        cumulative = np.cumprod(1 + returns) - 1

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=cumulative * 100,
            fill='tozeroy',
            name='Cumulative Return',
            line=dict(color='green'),
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            template='plotly_white',
            height=500,
        )

        return fig

    @staticmethod
    def create_summary_dashboard(
        equity: np.ndarray,
        returns: np.ndarray,
        timestamps: list = None,
        metrics: Dict = None,
    ) -> go.Figure:
        """Create multi-panel summary dashboard."""
        equity = np.asarray(equity)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100

        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('Equity Curve', 'Drawdown', 'Daily Returns', 'Cumulative Returns'),
            specs=[[{}, {}], [{}, {}]]
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(x=timestamps, y=equity, name='Equity', line=dict(color='blue')),
            row=1, col=1
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(x=timestamps, y=drawdown, fill='tozeroy', name='Drawdown',
                      line=dict(color='red')),
            row=1, col=2
        )

        # Daily returns
        fig.add_trace(
            go.Bar(x=timestamps, y=returns * 100, name='Daily Return',
                   marker=dict(color='steelblue')),
            row=2, col=1
        )

        # Cumulative returns
        cumulative = np.cumprod(1 + returns) - 1
        fig.add_trace(
            go.Scatter(x=timestamps, y=cumulative * 100, fill='tozeroy',
                      name='Cumulative', line=dict(color='green')),
            row=2, col=2
        )

        fig.update_yaxes(title_text='Value ($)', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown (%)', row=1, col=2)
        fig.update_yaxes(title_text='Daily Return (%)', row=2, col=1)
        fig.update_yaxes(title_text='Cumulative (%)', row=2, col=2)

        fig.update_layout(
            title_text="Strategy Performance Dashboard",
            height=900,
            showlegend=False,
            template='plotly_white',
        )

        return fig
