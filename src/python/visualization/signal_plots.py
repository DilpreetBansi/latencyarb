"""Signal visualization for spread and z-score analysis."""

import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from typing import Optional, List


class SignalPlotter:
    """Visualize trading signals and spreads."""

    @staticmethod
    def plot_spread_with_signals(
        spread: np.ndarray,
        z_scores: np.ndarray,
        signals: List[dict] = None,
        timestamps: list = None,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        title: str = "Spread & Signals",
    ) -> go.Figure:
        """
        Plot spread with z-score and trading signals.

        Parameters:
        -----------
        spread : array-like
            Spread values
        z_scores : array-like
            Z-score values
        signals : list, optional
            List of signal dicts with 'index', 'type', 'price'
        timestamps : list, optional
            Timestamps
        entry_threshold : float
            Entry threshold line
        exit_threshold : float
            Exit threshold line
        title : str
            Chart title

        Returns:
        --------
        Plotly Figure
        """
        if timestamps is None:
            timestamps = range(len(spread))

        fig = sp.make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Spread', 'Z-Score'),
            row_heights=[0.6, 0.4],
        )

        # Spread
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=spread,
                name='Spread',
                line=dict(color='blue', width=2),
                hovertemplate='<b>%{text}</b><br>Spread: %{y:.4f}<extra></extra>',
                text=timestamps,
            ),
            row=1, col=1
        )

        # Z-score with thresholds
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=z_scores,
                name='Z-Score',
                line=dict(color='green', width=2),
            ),
            row=2, col=1
        )

        # Threshold lines
        fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red",
                     annotation_text="Entry", row=2, col=1)
        fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="red",
                     annotation_text="Entry", row=2, col=1)
        fig.add_hline(y=exit_threshold, line_dash="dot", line_color="gray",
                     annotation_text="Exit", row=2, col=1)
        fig.add_hline(y=-exit_threshold, line_dash="dot", line_color="gray",
                     annotation_text="Exit", row=2, col=1)

        # Plot signals
        if signals:
            for signal in signals:
                idx = signal.get('index', 0)
                sig_type = signal.get('type', 'unknown')
                price = signal.get('price', 0)

                color = 'green' if 'entry' in sig_type else 'red'
                symbol = 'triangle-up' if 'long' in sig_type else 'triangle-down'

                fig.add_trace(
                    go.Scatter(
                        x=[timestamps[idx]],
                        y=[spread[idx]],
                        mode='markers',
                        marker=dict(size=12, color=color, symbol=symbol),
                        name=sig_type,
                        hovertemplate=f'<b>{sig_type}</b><br>Spread: %{{y:.4f}}<extra></extra>',
                    ),
                    row=1, col=1
                )

        fig.update_yaxes(title_text='Spread', row=1, col=1)
        fig.update_yaxes(title_text='Z-Score', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=1)

        fig.update_layout(
            title=title,
            height=700,
            hovermode='x unified',
            template='plotly_white',
        )

        return fig

    @staticmethod
    def plot_pair_prices(
        price1: np.ndarray,
        price2: np.ndarray,
        symbol1: str = "Asset 1",
        symbol2: str = "Asset 2",
        timestamps: list = None,
        title: str = "Pair Prices",
    ) -> go.Figure:
        """Plot both assets in the pair."""
        if timestamps is None:
            timestamps = range(len(price1))

        fig = sp.make_subplots(
            specs=[[{"secondary_y": True}]]
        )

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=price1,
                name=symbol1,
                line=dict(color='blue'),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=price2,
                name=symbol2,
                line=dict(color='red'),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            hovermode='x unified',
            template='plotly_white',
            height=500,
        )

        fig.update_yaxes(title_text=f'{symbol1} Price', secondary_y=False)
        fig.update_yaxes(title_text=f'{symbol2} Price', secondary_y=True)

        return fig

    @staticmethod
    def plot_trade_analysis(
        spread: np.ndarray,
        z_scores: np.ndarray,
        entry_signals: List[int] = None,
        exit_signals: List[int] = None,
        timestamps: list = None,
        title: str = "Trade Analysis",
    ) -> go.Figure:
        """Plot trades on spread chart."""
        if timestamps is None:
            timestamps = range(len(spread))

        fig = sp.make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Spread with Trades', 'Z-Score'),
        )

        # Spread
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=spread,
                name='Spread',
                line=dict(color='blue', width=2),
                fill='tozeroy',
            ),
            row=1, col=1
        )

        # Entry signals
        if entry_signals:
            entry_prices = [spread[idx] for idx in entry_signals if idx < len(spread)]
            entry_times = [timestamps[idx] for idx in entry_signals if idx < len(spread)]

            fig.add_trace(
                go.Scatter(
                    x=entry_times,
                    y=entry_prices,
                    mode='markers',
                    marker=dict(size=12, color='green', symbol='triangle-up'),
                    name='Entry',
                ),
                row=1, col=1
            )

        # Exit signals
        if exit_signals:
            exit_prices = [spread[idx] for idx in exit_signals if idx < len(spread)]
            exit_times = [timestamps[idx] for idx in exit_signals if idx < len(spread)]

            fig.add_trace(
                go.Scatter(
                    x=exit_times,
                    y=exit_prices,
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='triangle-down'),
                    name='Exit',
                ),
                row=1, col=1
            )

        # Z-score
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=z_scores,
                name='Z-Score',
                line=dict(color='purple'),
            ),
            row=2, col=1
        )

        fig.update_yaxes(title_text='Spread', row=1, col=1)
        fig.update_yaxes(title_text='Z-Score', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=1)

        fig.update_layout(
            title=title,
            height=700,
            hovermode='x unified',
            template='plotly_white',
        )

        return fig
