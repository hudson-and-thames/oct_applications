"""
Import price data using yfiance and yahoo_fin.

output price, return in pandas data frame form with either daily price data,
or 5 min interval price for upto within the past 60 days.
For daily price data, we only keep the opening price.

@author: Hansen
"""
import yfinance as yf
import yahoo_fin.stock_info as ys
import pandas as pd


class ImportData:
    """Wrapper class that imports data from yfinance and yahoo_fin."""

    def __init__(self):
        pass

    def get_sp500_tickers(self):
        """
        Get the S&P 500 stocks tickers.

        Returns
        -------
        tickers_sp500 : list of str
            S&P 500 tickers

        """
        tickers_sp500 = ys.tickers_sp500()

        return tickers_sp500

    def get_dow_tickers(self):
        """
        Get the Dow stocks tickers.

        Returns
        -------
        tickers_dow : list of str
            Dow tickers

        """
        tickers_dow = ys.tickers_dow()

        return tickers_dow

    def remove_nuns(self, df, threshold=100):
        """
        Remove tickers with nuns in value over a threshold.

        Parameters
        ----------
        df : pandas dataframe
            Price time series dataframe

        threshold: int, OPTIONAL
            The number of null values allowed
            Default is 100

        Returns
        -------
        df : pandas dataframe
            Updated price time series without any nuns
        """
        null_sum_each_ticker = df.isnull().sum()
        tickers_under_threshold = \
            null_sum_each_ticker[null_sum_each_ticker <= threshold].index
        df = df[tickers_under_threshold]

        return df

    def get_price_data(self,
                       tickers,
                       start_date,
                       end_date,
                       interval='5m'):
        """
        Get the price data with custom start and end date and interval.

        No Pre and Post market data.
        For daily price, only keep the closing price.

        Parameters
        ----------
        start_date : str
            Download start date string (YYYY-MM-DD) or _datetime.
        end_date : str
            Download end date string (YYYY-MM-DD) or _datetime.
        interval : str, OPTIONAL
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
            Default is '5m'
        tickers : str, list
            List of tickers to download

        Returns
        -------
        price_data : pandas dataframe
            The requested price_data
        """
        price_data = yf.download(tickers,
                                 start=start_date, end=end_date,
                                 interval=interval,
                                 group_by='column')['Close']

        return price_data

    def get_returns_data(self, price_data):
        """
        Calculate return data with custom start and end date and interval.

        Parameters
        ----------
        price_data : pandas dataframe
            The price data

        Returns
        -------
        returns_data : pandas dataframe
            The requested returns data.
        """
        returns_data = price_data.pct_change()
        returns_data = returns_data.iloc[1:]

        return returns_data
