import pickle

import pandas as pd
import yfinance as yf


class PriceData:
    """
    Parent class for retrieving and storing daily close price data for further analysis.

    Methods:
        set_start_date: Set the price series starting point to a specific date.
                        Users need to provide their own date
        load_sp500: Loading S&P 500 constituents information, including ticker name, industry, etc.
        get_price: Use yfinance to load price history of the tickers.

    """
    def __init__(self, start_date: str):
        """
        Constructor for the stock data class.

        Attributes:
        symbolinfo (DataFrame): A Dataframe including the tickers and their related information.
        start_date (string): Define the starting date of the price series.
                             Should be in "YYYY-mm-dd" format.
        price_history (DataFrame): A DataFrame including the price history of all the tickers

        :param str start_date: A string that indicates the starting date of the price series.
                                  Should be in "YYYY-mm-dd" format.
        """
        self.__symbols = None
        self.__symbolinfo = None
        self.__price_history = None
        self._start_date = pd.to_datetime(start_date)

    def set_start_date(self, start_date):
        """
        Setter. Set starting date to a user-defined date.
        String should be in "YYYY-mm-dd" format.
        :param start_date: A string in "YYYY-mm-dd" format
        :return: None
        """
        self._start_date = start_date

    def load_sp500(self, filepath="sp500.pkl"):
        """
        Retrieve sp500 stock symbols info from a pickle file.

        Check if the file exists, if not, try to retrieve it from the Wikipedia page and cache it.

        Meanwhile, set the attributes so that the class has the information
        :param filepath: A string that indicates the path of the sp500 ticker info pickle file
        :return:
            symbols (tuple): A read-only list of tickers
            symbolinfo (DataFrame): A DataFrame that includes
        """
        try:
            with open(filepath, 'rb') as infof:
                symbolinfo = pickle.load(infof)
                symbols = tuple(symbolinfo['Symbol'].values)
        except FileNotFoundError:
            print("{} not found! Re-downloading from Wikipedia".format(filepath))
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            symbolinfo = table[0]
            with open("sp500.pkl", 'wb') as writef:
                pickle.dump(symbolinfo, writef)
            symbols = tuple(symbolinfo['Symbol'].values)

        self.__symbolinfo = symbolinfo
        self.__symbols = symbols

    def get_price(self):
        """
        Use yfinance to get the price data.
        The price data would be cut such that the price series start from the date defined
            by _start_date
        :return:
            price_data (DataFrame): A dataframe containing the price data history starting
                                    from the start_date of all tickers
        """
        try:
            assert self.__symbols is not None and len(self.__symbols) > 0, \
                "No stocks to select from! Please reload the tickers."
        except AssertionError:
            self.load_sp500()

        # Unfortunately, yfinance could not download historical intraday data.
        # To still demonstrate the method, we'll use daily data here instead.
        # To ensure there's enough datapoints, a 5-year history would be used here.

        price_data = yf.download(
            tickers=self.__symbols,
            period='5y',
            interval='1d'
        )
        # Set the starting date to _start_date
        # price_data = price_data.loc[self._start_date:]['Adj Close']
        price_data = price_data['Adj Close'].loc[self._start_date:]

        # Front-fill the NaNs. This will yield the correct return (0%) as there was no trading
        # on these days.
        price_data.fillna(method='ffill', inplace=True)

        # If the data history is not long enough, just drop the ticker.
        price_data.dropna(axis=1, inplace=True)

        # Save price history for cointegration checks and spread building.
        self.__price_history = price_data

        return price_data

    def get_price_history(self):
        """
        Getter. Return price history.
        :return:
        price_history (DataFrame): Raw daily close price data.
        """
        return self.__price_history

    def get_symbol_info(self):
        """
        Getter. Return related info for symbols.
        :return:
        symbolinfo (DataFrame): Sector and Sub-industry information of stocks.
        """
        return self.__symbolinfo
