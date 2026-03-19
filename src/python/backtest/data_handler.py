"""
Data handling for backtesting using yfinance (free Yahoo Finance data).

Supports downloading, cleaning, and aligning OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class DataHandler:
    """
    Handle market data loading and preprocessing.

    Features:
    - Free data from Yahoo Finance
    - Handle splits and dividends
    - Align multiple ticker timestamps
    - Fill missing data
    """

    @staticmethod
    def download_data(
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download price data from Yahoo Finance.

        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        interval : str
            Interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)

        Returns:
        --------
        DataFrame with adjusted close prices, index is date
        """
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
        )

        if len(tickers) == 1:
            # Single ticker returns Series
            return pd.DataFrame({"Close": data["Close"].values}, index=data.index)
        else:
            # Multiple tickers return DataFrame
            return data["Adj Close"]

    @staticmethod
    def get_ohlcv(
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a single ticker.

        Parameters:
        -----------
        ticker : str
            Ticker symbol
        start_date : str
            Start date
        end_date : str
            End date

        Returns:
        --------
        DataFrame with Open, High, Low, Close, Volume
        """
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        return data[["Open", "High", "Low", "Close", "Volume"]]

    @staticmethod
    def align_data(
        data_dict: Dict[str, pd.DataFrame],
        method: str = "ffill",
    ) -> pd.DataFrame:
        """
        Align multiple time series on common dates.

        Parameters:
        -----------
        data_dict : dict
            Dict of {ticker: DataFrame}
        method : str
            Forward fill ('ffill') or backward fill ('bfill')

        Returns:
        --------
        Aligned DataFrame with multi-level columns
        """
        # Concatenate all data
        combined = pd.concat(data_dict.values(), axis=1, keys=data_dict.keys())

        # Fill missing data
        combined = combined.fillna(method=method)

        # Drop rows with any NaN
        combined = combined.dropna()

        return combined

    @staticmethod
    def calculate_returns(
        prices: pd.Series,
        return_type: str = "log",
    ) -> pd.Series:
        """
        Calculate returns from price series.

        Parameters:
        -----------
        prices : Series
            Price time series
        return_type : str
            'log' for log returns, 'simple' for simple returns

        Returns:
        --------
        Returns series
        """
        if return_type == "log":
            return np.log(prices / prices.shift(1)).dropna()
        else:
            return (prices / prices.shift(1) - 1).dropna()

    @staticmethod
    def resample_data(
        data: pd.DataFrame,
        frequency: str = "D",
    ) -> pd.DataFrame:
        """
        Resample data to different frequency.

        Parameters:
        -----------
        data : DataFrame
            Price data with datetime index
        frequency : str
            Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)

        Returns:
        --------
        Resampled data
        """
        return data.resample(frequency).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna()

    @staticmethod
    def handle_splits_dividends(
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Handle stock splits and dividends by using adjusted close.

        Parameters:
        -----------
        ticker : str
            Ticker symbol
        start_date : str
            Start date
        end_date : str
            End date

        Returns:
        --------
        DataFrame with adjusted data
        """
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Calculate adjustment ratio
        adjustment = data["Adj Close"] / data["Close"]

        # Apply adjustment to OHLC
        data["Open"] = data["Open"] * adjustment
        data["High"] = data["High"] * adjustment
        data["Low"] = data["Low"] * adjustment
        data["Close"] = data["Adj Close"]

        return data[["Open", "High", "Low", "Close", "Volume"]]

    @staticmethod
    def fill_missing_dates(
        data: pd.DataFrame,
        method: str = "ffill",
    ) -> pd.DataFrame:
        """
        Fill missing trading dates.

        Parameters:
        -----------
        data : DataFrame
            Data with gaps
        method : str
            Fill method

        Returns:
        --------
        Data with filled dates
        """
        # Create date range
        date_range = pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq="D"  # Daily frequency
        )

        # Reindex
        data = data.reindex(date_range)

        # Fill NaN
        if method == "ffill":
            data = data.fillna(method="ffill")
        elif method == "bfill":
            data = data.fillna(method="bfill")

        return data.dropna()

    @staticmethod
    def get_benchmark_data(
        ticker: str = "SPY",
        start_date: str = None,
        end_date: str = None,
    ) -> pd.Series:
        """
        Get benchmark data (S&P 500 by default).

        Parameters:
        -----------
        ticker : str
            Benchmark ticker
        start_date : str
            Start date
        end_date : str
            End date

        Returns:
        --------
        Adjusted close price series
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        return data["Adj Close"]

    @staticmethod
    def get_risk_free_rate(
        start_date: str = None,
        end_date: str = None,
    ) -> float:
        """
        Get approximate risk-free rate (3-month Treasury).

        Parameters:
        -----------
        start_date : str
            Start date
        end_date : str
            End date

        Returns:
        --------
        Annual risk-free rate as fraction
        """
        # Download 3-month Treasury rate
        try:
            data = yf.download("^IRX", start=start_date, end=end_date, progress=False)
            if len(data) > 0:
                return data["Close"].iloc[-1] / 100  # Convert from basis points
        except:
            pass

        # Fallback to 4% annual
        return 0.04

    @staticmethod
    def validate_data(data: pd.DataFrame) -> bool:
        """
        Validate data quality.

        Parameters:
        -----------
        data : DataFrame
            Data to validate

        Returns:
        --------
        True if valid, False otherwise
        """
        # Check for NaN
        if data.isna().any().any():
            return False

        # Check for negative prices
        if (data < 0).any().any():
            return False

        # Check minimum data points
        if len(data) < 20:
            return False

        return True
