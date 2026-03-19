"""
LatencyArb: Ultra-Low-Latency Statistical Arbitrage Engine

A high-performance pairs trading framework targeting hedge funds and proprietary trading firms.
Designed for sub-microsecond latency with production-grade backtesting and risk management.
"""

__version__ = "1.0.0"
__author__ = "Quant Team"

from . import research
from . import strategy
from . import backtest
from . import visualization

__all__ = [
    "research",
    "strategy",
    "backtest",
    "visualization",
]
