# LatencyArb Quick Start Guide

Get up and running in 10 minutes.

## Prerequisites

- Python 3.8+
- C++17 compiler (for optional C++ components)
- CMake 3.16+ (for building C++ components)

## Installation

### Step 1: Clone and Setup

```bash
cd /path/to/portfolio/projects
git clone https://github.com/yourname/latencyarb.git
cd latencyarb

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Test imports
python -c "import numpy, scipy, pandas, yfinance, plotly; print('OK')"
```

## Run Examples

All examples are in the `examples/` directory:

### Example 1: Pair Selection (5 min)

```bash
cd examples
python 01_pair_selection.py
```

**What it does:**
- Downloads historical data for a universe of stocks (AAPL, MSFT, GOOGL, AMZN)
- Screens for cointegrated pairs
- Displays cointegration test results, mean reversion parameters, Hurst exponent
- Analyzes the top pair in detail

**Expected output:**
```
==============================================================================
LATENCYARB: Pair Selection Example
==============================================================================

[*] Downloading data for 4 stocks...
    - AAPL... OK
    - MSFT... OK
    ...

[*] Screening universe for cointegrated pairs...

    Found 3 high-quality pairs:

        Pair  Score  P-Value  Half-Life  Hurst  Spread Std
  MSFT/GOOGL  0.6234  0.0123      12.34   0.421  0.005234
  ...

[*] Detailed Analysis: MSFT / GOOGL
```

### Example 2: Backtest Pairs Strategy (5 min)

```python
python 02_backtest_pairs.py
```

**What it does:**
- Backtests a pairs trading strategy on historical data
- Generates entry/exit signals based on z-scores
- Calculates performance metrics (Sharpe, max drawdown, etc.)
- Tracks positions and PnL

**Expected output:**
```
[*] Running backtest...
    - Backtest complete: 232 bars processed
    - Signals generated: 8
    - Final equity: $105,234

[*] Performance Analysis:
    Total Return: 5.23%
    Sharpe Ratio: 1.45
    Max Drawdown: -8.3%
```

### Example 3: Risk Analysis (5 min)

```python
python 03_risk_analysis.py
```

**What it does:**
- Runs Monte Carlo simulation (1000 paths)
- Calculates confidence intervals on Sharpe ratio
- Estimates probability of ruin
- Computes expected shortfall (CVaR)

**Expected output:**
```
[*] Monte Carlo Analysis (1000 simulations):
  - Sharpe ratio (mean): 1.45
  - Sharpe ratio (95% CI): [1.23, 1.67]
  - Max drawdown (mean): -9.2%
  - Probability of ruin: 2.3%
  - Expected shortfall: -1.8%
```

### Example 4: Full Pipeline (10 min)

```python
python 04_full_pipeline.py
```

**What it does:**
- Complete end-to-end workflow
- Pair selection → Strategy setup → Backtesting → Risk analysis
- Integrated example showing all components together

## Common Tasks

### Download Market Data

```python
from src.python.backtest.data_handler import DataHandler

# Download data
data = DataHandler.download_data(
    ['AAPL', 'MSFT', 'GOOGL'],
    start_date='2023-01-01',
    end_date='2024-01-01'
)

print(data)
```

### Test Cointegration

```python
from src.python.research.cointegration import EngleGrangerTest
import numpy as np

# Generate test data
x = np.cumsum(np.random.randn(100)) + 100
y = 2 * x + np.random.randn(100) * 0.1 + 50

# Run test
result = EngleGrangerTest.test(x, y)

print(f"P-value: {result.p_value:.4f}")
print(f"Cointegrated: {result.is_cointegrated}")
```

### Estimate Mean Reversion Parameters

```python
from src.python.research.mean_reversion import OrnsteinUhlenbeckEstimator
import numpy as np

# Spread series
spread = np.cumsum(np.random.randn(200) * 0.1)

# Estimate OU parameters
params = OrnsteinUhlenbeckEstimator.estimate_regression(spread)

print(f"Mean: {params.mu:.4f}")
print(f"Mean reversion speed (θ): {params.theta:.6f}")
print(f"Half-life: {params.half_life:.2f} periods")
```

### Run Kalman Filter

```python
from src.python.research.kalman_filter import KalmanFilter
import numpy as np

# Two price series
x_series = np.cumsum(np.random.randn(100) * 0.01) + 100
y_series = 1.5 * x_series + np.cumsum(np.random.randn(100) * 0.005)

# Filter
kf = KalmanFilter(initial_hedge_ratio=1.0)
hedge_ratios, spreads = kf.filter(x_series, y_series)

print(f"Final hedge ratio: {hedge_ratios[-1]:.4f}")
```

### Backtest Strategy

```python
from src.python.strategy.pairs_strategy import PairsStrategy
import numpy as np

# Create strategy
strategy = PairsStrategy(
    pair_id="TEST",
    symbol1="ASSET1",
    symbol2="ASSET2",
    initial_capital=100_000,
)

# Generate synthetic price data
prices1 = 100 + np.cumsum(np.random.randn(100) * 1)
prices2 = 100 + np.cumsum(np.random.randn(100) * 1)

# Run backtest
for i, (p1, p2) in enumerate(zip(prices1, prices2)):
    signal = strategy.update(p1, p2, timestamp=i)
    if signal:
        print(f"Signal at {i}: {signal.signal_type}")
```

### Calculate Performance Metrics

```python
from src.python.backtest.metrics import MetricsCalculator
import numpy as np

# Generate sample returns
returns = np.random.randn(252) * 0.01 + 0.001

# Calculate metrics
sharpe = MetricsCalculator.calculate_sharpe_ratio(returns)
sortino = MetricsCalculator.calculate_sortino_ratio(returns)

print(f"Sharpe: {sharpe:.2f}")
print(f"Sortino: {sortino:.2f}")
```

### Run Monte Carlo Analysis

```python
from src.python.backtest.monte_carlo import MonteCarloAnalyzer
import numpy as np

# Returns
returns = np.random.randn(252) * 0.01 + 0.001

# Monte Carlo
result = MonteCarloAnalyzer.analyze_strategy(
    returns,
    n_simulations=1000,
    confidence_intervals=0.95
)

print(f"Probability of Ruin: {result.probability_ruin:.2%}")
print(f"Expected Shortfall: {result.expected_shortfall:.2%}")
```

## Build C++ Components (Optional)

```bash
mkdir cmake_build
cd cmake_build
cmake ..
make
cd ..
```

This creates a static library with the C++ components. The Python layer can interface with it for ultra-low-latency execution.

## Run Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_cointegration.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Configuration

Edit these files to customize:

### `requirements.txt`
Python dependencies. Add any custom libraries here.

### `CMakeLists.txt`
C++ build configuration. Adjust compiler flags, optimization level.

### Strategy Parameters
In example scripts:
```python
strategy = PairsStrategy(
    entry_z_threshold=2.0,      # Entry at 2 std devs
    exit_z_threshold=0.5,       # Exit at 0.5 std devs
    initial_capital=100_000,
)
```

### Risk Parameters
In example scripts:
```python
risk_manager = RiskManager(
    max_drawdown=0.20,           # 20% max drawdown
    max_daily_loss=0.05,         # 5% max daily loss
    kelly_fraction=0.25,         # 25% of Kelly leverage
)
```

## Troubleshooting

### ImportError: No module named yfinance
```bash
pip install --upgrade yfinance
```

### Data download fails
- Check internet connection
- Yahoo Finance may have rate limits - wait a moment and retry
- Alternatively, use synthetic data (examples handle this)

### CMake build fails
- Ensure C++17 compiler available: `g++ --version` or `clang --version`
- CMake 3.16+: `cmake --version`

### Tests fail
```bash
# Run with verbose output
pytest tests/ -v -s

# Run single test
pytest tests/test_kalman.py::TestKalmanFilter::test_initialization -v
```

## Next Steps

1. **Study the math**: Read the mathematical foundations in README.md
2. **Explore components**: Review source code in `src/python/` and `src/cpp/`
3. **Modify examples**: Change parameters and see how results change
4. **Write your own**: Create custom strategies using the framework
5. **Deploy carefully**: Start with paper trading, use strict risk controls

## Documentation

- **README.md**: Comprehensive guide with all details
- **PROJECT_SUMMARY.md**: Project structure and statistics
- **Code docstrings**: All classes have detailed docstrings
- **Examples**: 4 runnable examples demonstrating key features

## Support & Contributing

This is an open-source educational project. Contributions welcome!

Areas to enhance:
- Additional cointegration tests
- More signal generation methods
- Alternative risk models
- Exchange integrations
- Visualization improvements

## License

MIT License - see LICENSE file

---

**Ready to trade?** Start with Example 1 (pair selection), then Example 2 (backtest), then Example 3 (risk analysis).

