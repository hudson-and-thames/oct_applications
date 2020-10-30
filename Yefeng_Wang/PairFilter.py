import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.unitroot import engle_granger

from PairSelector import PairSelector
from PriceData import PriceData


class PairFilter(PairSelector):
    """
    This class perform the 4 filters.
    1. Cointegration
    2. Hurst Exponent
    3. Mean Reversion Half Life
    4. Minimum Mean Crossovers

    Methods:
        filter_pair: Apply all the filters and reduce the investing universe.
        plot_spread(pair): Plot the spread price action, and the price action of constituents.
        pair_summary: Output a summary table of the sector and sub-industries of pair constituents.

    Properties:
        final_pairs: Return a list of the final pair candidates.
    """
    def __init__(self, start_date):
        """
        Attributes:
            final_pairs: A set that stores the final selected pairs.
        :param start_date: A "YYYY-mm-dd" string that indicates the start of the price series.
        """
        super().__init__(start_date)
        self.__final_pairs = None
        self.__spread_dict = dict()

    @staticmethod
    def _engle_granger_test(price_x, price_y):
        """
        Here we use the arch package to do the Engle-Granger test.
        During the process, the hedge ratio is generated as well.
        We use p < 0.05 as the criterion to reject null hypothesis.

        :param price_x: pair component
        :param price_y: the other pair component
        :return:
        coint: Boolean that indicates if two series are cointegrated
        test_statistics: the smaller t-statistics of Engle_Granger(x,y) and Engle_Granger(y,x)
        hedge_ratio: the hedge ratio corresponding to the selected test
        """
        eg_test1 = engle_granger(price_x, price_y, trend='n')
        eg_test2 = engle_granger(price_y, price_x, trend='n')
        # No trend related regressors are included in the regression

        # Check if the p-value is less than 0.05
        if eg_test1.pvalue >= 0.05:
            if eg_test2.pvalue >= 0.05:
                # Can't reject null hypothesis. No cointegration. Skip
                return False, None, None
                # Only Coint(y,x) rejected null hypothesis, use this test to calculate
                # hedge ratio.
            return True, eg_test2.stat, eg_test2.cointegrating_vector
        else:
            # Coint(x,y) rejected null hypothesis. Now testing Coint(y,x)
            if eg_test2.pvalue >= 0.05:
                # Coint(y, x) didn't pass, so take the Coint(x, y) result for hedge ratio.
                return True, eg_test1.stat, eg_test1.cointegrating_vector
            # Coint(y, x) and Coint(x, y) both passed.
            # Choose the test that yielded smaller t-stat for hedge ratio.
            if eg_test1.stat < eg_test2.stat:
                return True, eg_test1.stat, eg_test1.cointegrating_vector
            return True, eg_test2.stat, eg_test2.cointegrating_vector

    @staticmethod
    def _hurst(price_df):
        """
        Calculate the Hurst Exponent of the time series.

        The Hurst Exponent calculation code reference:
        https://www.quantstart.com/articles/basics-of-statistical-mean-reversion-testing/

        :param price_df: spread price DataFrame
        :return:
        hurst_exponent (float): The hurst exponent of the time series.
        """
        price = price_df['spread'].to_numpy()

        lags = range(2, 100)

        # Calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(np.subtract(price[lag:], price[:-lag]))) for lag in lags]

        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        # Return the Hurst exponent from the polyfit output
        return poly[0] * 2.0

    @staticmethod
    def _half_life(price):
        """
        We'll assume the price follows an AR(1) process so as to
            estimate the mean reversion half-life.
        :param price: spread price DataFrame
        :return:
        halflife (float): half_life in days as we're using daily data
        """
        price['one_lagged'] = price['spread'].shift(1)
        price['return'] = price['spread'] - price['one_lagged']
        price_copy = price.dropna()
        autoreg = sm.OLS(price_copy['return'], sm.add_constant(price_copy['one_lagged']))
        res = autoreg.fit()

        halflife = -np.log(2) / res.params[1]
        return halflife

    @staticmethod
    def _mean_crossover(price):
        """
        Calculate how many times the price has crossed over mean.
        :param price: spread price DataFrame
        :return:
        times (int): The times the spread has crossed over mean
        """

        mean_level = price['spread'].mean()
        price['crossover'] = ((price['spread'] > mean_level) & (price['one_lagged'] <= mean_level)) | \
                             ((price['spread'] < mean_level) & (price['one_lagged'] >= mean_level))

        # Crossover means the previous day is below mean level,
        # but current day closed above mean level
        # Or: the previous day is above mean level, but current day closed below mean level
        price['Year'] = price.index.year
        xover_count = price[['crossover', 'Year']].groupby('Year').sum()
        return xover_count

    def _get_single_price(self, ticker):
        """
        Retrieve the history of a single ticker for spread building.
        :param ticker: Ticker Symbol
        :type ticker: str
        :return:
        price_series (pd.Series): The price history of a stock
        """
        raw_price = PriceData.get_price_history(self)
        return raw_price[ticker]

    def _build_spread(self, ticker_x, ticker_y):
        """
        Build the spread according to the hedge ratio.
        :param ticker_x: First stock symbol
        :type ticker_x: str
        :param ticker_y: Second stock symbol
        :type ticker_y: str
        :return:
        pd.DataFrame: The spread price series, if cointegrated.
        None: if not cointegrated.
        """
        price_x = self._get_single_price(ticker_x)
        price_y = self._get_single_price(ticker_y)
        coint, _, coint_vec = self._engle_granger_test(price_x, price_y)
        if coint:
            spread = price_x * coint_vec.loc[ticker_x] + price_y * coint_vec.loc[ticker_y]
            spread.name = "spread"
            return spread.to_frame()
        # No cointegration. Return None as a boolean flag so we will not deal with these pairs
        # in the future.
        return None

    def _hurst_filter(self, price_df):
        """
        Check if the price series satisfies the Hurst Exponent condition.
        :param price_df: Spread price series in DataFrame format, with column name 'spread'
        :return: bool, 0 < Hurst exponent < 0.5 => True, otherwise False
        """
        hurst_exp = self._hurst(price_df)
        return 0 < hurst_exp < 0.5

    def _hl_filter(self, price_df):
        """
        Check if the mean reversion half life is longer than 1 day but shorter than 1 year.
        :param price_df: Spread price series in DataFrame format, with column name 'spread'
        :return: bool, 1 <= half life <= 252 (252 trading days in 1 year) => True, otherwise False
        """
        half_life = self._half_life(price_df)
        return 1 <= half_life <= 252

    def _crossover_filter(self, price_df):
        """
        Check if the price action satisfies the following:
         1. It must cross over the mean by 11 times in 2020.
         2. It must cross over the mean by 12 times in 2019.
        :param price_df: Spread price series in DataFrame format, with column name 'spread'
        :return: bool, if both conditions are satisfied, otherwise False
        """
        crossover_df = self._mean_crossover(price_df)

        count_curr_year = crossover_df['crossover'].tail(1).sum()
        count_two_year = crossover_df['crossover'].tail(2).sum()

        return (count_curr_year >= 11) and (count_two_year >= 23)

    def filter_pair(self):
        """
        Use all four filters to select tradable pairs.
        :return:
        final_pairs (set): All tradable pairs in a single set.
        """
        final_pairs = []
        for candidate in self.pairs:
            # candidate is a tuple. Use asterisk to fill the argument.
            spread = self._build_spread(*candidate)

            if spread is not None:
                # Cointegration is satisfied. First, test Hurst Exponent.
                hurst_condition = self._hurst_filter(spread)

                # Then test mean-reversion half life
                hl_condition = self._hl_filter(spread)

                # Finally test mean crossover
                xover_condition = self._crossover_filter(spread)

                # If everything is good, add to the final selection
                if all([hurst_condition, hl_condition, xover_condition]):
                    final_pairs.append(candidate)
                    self.__spread_dict[candidate] = spread

        self.__final_pairs = final_pairs
        return final_pairs

    @property
    def final_pairs(self):
        """
        Getter for presenting the final pairs
        :return:
        final_pairs (set): the set that stores the selected pairs.
        """
        return self.__final_pairs

    def plot_spread(self, pair,
                    figw=15,
                    figh=10,
                    start_date=pd.Timestamp(2016, 1, 1),
                    end_date=pd.Timestamp(2020, 10, 31)):
        """
        Plot the spread price action as well as the price actions of the pair constituents.
        :param pair: The 2-tuple of the tickers that formed the pair
        :param start_date: x-axis left limit
        :param end_date: x-axis right limit
        :param figw: figure size - width
        :param figh: figure size - height
        :return: None
        """

        stock_one, stock_two = pair
        price1 = PriceData.get_price_history(self)[stock_one]
        price2 = PriceData.get_price_history(self)[stock_two]

        spread = self.__spread_dict[pair]

        # Set the x-ticks format
        # Make the ticks exactly at each month, and a bigger tick at year start
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y')

        _, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(figw, figh))
        ax1.plot(price1, label=price1.name)
        ax1.plot(price2, label=price2.name)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.tick_params(axis='y', labelsize=14)
        ax2.plot(spread['spread'], label='spread')
        ax2.legend(loc='best', fontsize=12)
        ax2.axhline(y=spread['spread'].mean(), color='black')
        ax2.xaxis.set_major_locator(years)
        ax2.xaxis.set_major_formatter(years_fmt)
        ax2.xaxis.set_minor_locator(months)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.set_xlim((start_date, end_date))

    def pair_summary(self):
        """
        Report the sector and sub-industry of the pair constituents
        :return:
        summary (DataFrame): A Dataframe that shows info of all the pairs.
        """
        stock_info_df = PriceData.get_symbol_info(self)[["Symbol", "GICS Sector", "GICS Sub-Industry"]]
        summary_list = []

        # Retrieve the information for each constituent
        for pair in self.__final_pairs:
            stock_one, stock_two = pair
            stock_one_info = stock_info_df[stock_info_df['Symbol'] == stock_one]
            stock_two_info = stock_info_df[stock_info_df['Symbol'] == stock_two]

            # Rename the columns for better representation
            stock_one_info.columns = [
                "Stock 1",
                "Stock 1 Sector",
                "Stock 1 Sub-Industry"
            ]

            stock_two_info.columns = [
                "Stock 2",
                "Stock 2 Sector",
                "Stock 2 Sub-Industry"
            ]

            # Before joining, reset the index so there would be no NaNs
            stock_one_info.reset_index(drop=True, inplace=True)
            stock_two_info.reset_index(drop=True, inplace=True)

            summary_list.append(pd.concat([stock_one_info, stock_two_info], axis=1))

        summary = pd.concat(summary_list)
        # Reset index again, so the index will be natural numbers.
        summary.reset_index(drop=True, inplace=True)

        return summary
