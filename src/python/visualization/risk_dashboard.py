"""Risk analytics dashboard."""

import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from typing import Dict, List


class RiskDashboard:
    """Create risk analytics dashboards."""

    @staticmethod
    def plot_var_distribution(
        returns: np.ndarray,
        confidence_level: float = 0.95,
        title: str = "Value-at-Risk Distribution",
    ) -> go.Figure:
        """
        Plot VaR distribution from returns.

        Parameters:
        -----------
        returns : array-like
            Return distribution
        confidence_level : float
            Confidence level (e.g., 0.95)
        title : str
            Chart title

        Returns:
        --------
        Plotly Figure
        """
        var = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = np.mean(returns[returns <= var])

        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name='Returns',
            marker=dict(color='lightblue', opacity=0.7),
        ))

        # VaR line
        fig.add_vline(
            x=var * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR ({confidence_level*100:.0f}%): {var*100:.2f}%",
        )

        # CVaR line
        fig.add_vline(
            x=cvar * 100,
            line_dash="dash",
            line_color="darkred",
            annotation_text=f"CVaR: {cvar*100:.2f}%",
        )

        fig.update_layout(
            title=title,
            xaxis_title='Daily Returns (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=500,
        )

        return fig

    @staticmethod
    def plot_correlation_matrix(
        correlation_matrix: np.ndarray,
        assets: List[str] = None,
        title: str = "Correlation Matrix",
    ) -> go.Figure:
        """Plot correlation heatmap."""
        n_assets = correlation_matrix.shape[0]

        if assets is None:
            assets = [f"Asset {i}" for i in range(n_assets)]

        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix,
                x=assets,
                y=assets,
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
            )
        )

        fig.update_layout(
            title=title,
            height=500 + n_assets * 20,
        )

        return fig

    @staticmethod
    def plot_regime_probability(
        regime_probs: np.ndarray,
        timestamps: list = None,
        regime_names: List[str] = None,
        title: str = "Market Regime Probability",
    ) -> go.Figure:
        """Plot regime probability over time."""
        if timestamps is None:
            timestamps = range(len(regime_probs))

        if regime_names is None:
            regime_names = [f"Regime {i}" for i in range(regime_probs.shape[1])]

        fig = go.Figure()

        for i, regime in enumerate(regime_names):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=regime_probs[:, i],
                name=regime,
                mode='lines',
                stackgroup='one',
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Probability',
            template='plotly_white',
            height=500,
            hovermode='x unified',
        )

        return fig

    @staticmethod
    def plot_rolling_metrics(
        equity: np.ndarray,
        window: int = 30,
        timestamps: list = None,
        title: str = "Rolling Sharpe & Volatility",
    ) -> go.Figure:
        """Plot rolling Sharpe ratio and volatility."""
        if timestamps is None:
            timestamps = range(len(equity))

        equity = np.asarray(equity)
        returns = np.diff(equity) / equity[:-1]

        # Rolling Sharpe
        rolling_mean = np.convolve(returns, np.ones(window) / window, mode='valid')
        rolling_std = np.array([
            np.std(returns[i:i+window])
            for i in range(len(returns) - window + 1)
        ])

        rolling_sharpe = (rolling_mean * 252 - 0.04) / (rolling_std * np.sqrt(252))

        fig = sp.make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility'),
        )

        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=timestamps[window:],
                y=rolling_sharpe,
                name='Rolling Sharpe',
                line=dict(color='blue'),
            ),
            row=1, col=1
        )

        # Rolling Volatility
        fig.add_trace(
            go.Scatter(
                x=timestamps[window:],
                y=rolling_std * np.sqrt(252) * 100,
                name='Rolling Volatility',
                line=dict(color='red'),
            ),
            row=2, col=1
        )

        fig.update_yaxes(title_text='Sharpe Ratio', row=1, col=1)
        fig.update_yaxes(title_text='Annualized Volatility (%)', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=1)

        fig.update_layout(
            title=title,
            height=700,
            template='plotly_white',
            hovermode='x unified',
        )

        return fig

    @staticmethod
    def create_risk_summary(
        metrics: Dict,
        title: str = "Risk Summary",
    ) -> go.Figure:
        """Create summary risk table."""
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color='paleturquoise',
                align='left',
                font=dict(color='black', size=12),
            ),
            cells=dict(
                values=[
                    list(metrics.keys()),
                    [f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics.values()],
                ],
                fill_color='lavender',
                align='left',
                font=dict(size=11),
            )
        )])

        fig.update_layout(
            title=title,
            height=500,
        )

        return fig
