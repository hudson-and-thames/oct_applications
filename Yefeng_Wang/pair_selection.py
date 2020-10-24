import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from itertools import combinations
from arch.unitroot import engle_granger
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None


class PairSelection:
    def __init__(self):
        self.stockUniverse = tuple()
        self.pricedict = dict()
        self.returns = None
        self.pca_repr = None
        self.clusterings = None
        self.cluster_dict = None
        self.pairs = None
        self.stockinfo = None
        self.starting_point = "2016-01-04"
        self.spreaddict = dict()
        self.hurstdict = dict()
        self.halflifedict = dict()
        self.xoverdict = dict()
        self.selected_set = set()

    def get_sp500(self):
        """
        Get the constituents of S&P 500 index.

        We will use the Wikipedia page to extract the S&P 500 constituents.
        :return:
        None
        The method will edit the stockUniverse variable and save the tickers.
        """
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        self.stockinfo = df
        self.stockUniverse = tuple(df['Symbol'].values)

    def get_tickers(self):
        """
        Getter. Return list of tickers
        :return:
        """
        return self.stockUniverse

    def set_start_date(self, date):
        """
        Setter. Set starting date
        :param date:
        :return: None
        """
        self.starting_point = date

    def get_price(self):
        """
        Use yfinance to get the price data.
        :return:
        None

        The data would be directly saved to pricedict
        """
        assert len(self.stockUniverse) > 0, "No stocks to select from! Please reload the tickers."

        # Unfortunately, yfinance could not download historical intraday data.
        # To still demonstrate the method, we'll use daily data here instead.
        # To ensure there's enough datapoints, a 5-year history would be used here.

        price_data = yf.download(
            tickers=self.stockUniverse,
            period='5y',
            interval='1d'
        )
        # Data downloaded. Convert it into a dictionary for return calculation.
        price_data = price_data['Adj Close']
        # print(price_data['AAPL'])
        for stock in price_data.columns:
            tmp = price_data[[stock]]
            tmp.dropna(inplace=True)
            if len(tmp) == 0:
                # No data for this stock
                print("No data for {}!".format(stock))
                continue
            self.pricedict[stock] = tmp

    def calc_pct_return(self, ticker):
        """
        Calculate the percentage return of the stock
        :param ticker: stock ticker in stockUniverse
        :return:
        price_df: pd.DataFrame

        The dataframe includes date, adjusted price, and percent return
        Will return None if the ticker doesn't have sufficient price history.
        """
        try:
            price_df = self.pricedict[ticker]
            price_df['{}_pct_return'.format(ticker)] = price_df / price_df.shift(1) - 1
            return price_df
        except KeyError:
            print("Ticker doesn't exist!")
            return None

    def prepare_PCA_df(self, start_time="2016-01-04"):
        """
        Return a matrix of returns as the input to PCA
        Adding a starting point parameter to cutoff possible starting NAs

        :param start_time:
        :return:
        returns: pd.DataFrame

        return matrix for PCA
        """
        dflist = []
        for stock in self.stockUniverse:
            return_df = self.calc_pct_return(stock)
            if return_df is not None:
                # Cutoff at start_time
                return_df = return_df.loc[start_time:]

                # Remove stocks with shorter history
                return_df.dropna(inplace=True)
                df_start_time = return_df.head(1).index.date[0].strftime("%Y-%m-%d")

                # print(df_start_time)
                if df_start_time != start_time:
                    print("Stock {} has insufficient data!".format(stock))
                    continue
                return_df = return_df['{}_pct_return'.format(stock)]
                dflist.append(return_df)
        # combine all percent returns
        self.returns = pd.concat(dflist, axis=1)
        # Drop NaNs as sk-learn PCA cannot accept these null values
        # Preprocess for future convenience.
        self.returns.dropna(inplace=True)
        return self.returns

    def do_PCA(self, components=15):
        """
        :param components: Number of principal components (the paper set the upper bound to 15)
        :return:
        pca_repr: PCA representation of the return matrix
        """
        # Standardize the returns first
        scaler = StandardScaler()
        standardized_ret = scaler.fit_transform(self.returns)

        # Now do the PCA
        pca = PCA(n_components=components)
        pca.fit(standardized_ret)
        pca_repr = pca.components_.T

        # Get the new representation into a new DataFrame
        pca_df = pd.DataFrame(pca_repr)
        pca_df.columns = ['Feature {}'.format(x) for x in range(1, components+1)]
        pca_df.index = [x.split('_')[0] for x in list(self.returns.columns)]
        self.pca_repr = pca_df
        return self.pca_repr

    def do_OPTICS(self):
        """
        No optimal parameter was mentioned in the paper, so use default parameters here.
        :return:
        clusterings: The cluster label of each stock
        """
        cluster = OPTICS()
        cluster.fit(self.pca_repr)
        self.clusterings = cluster.labels_
        return self.clusterings

    def cluster_summary(self):
        """
        Give a summary on the clustering result.
        Stock tickers that were labeled as -1 will not be selected for further analysis.

        :return:
        cluster_dict: A dictionary which stores the tickers under each cluster label.
        """
        tickers = self.pca_repr.index
        cluster_dict = dict()
        num_classes = self.clusterings.max() + 1
        for i in range(num_classes):
            cluster_dict["Cluster {}".format(i)] = tuple(tickers[self.clusterings == i])
        self.cluster_dict = cluster_dict
        return self.cluster_dict

    def plot_PCA(self):
        """
        todo: Make a visualization for PCA results
        :return:
        """

    def plot_clusters(self):
        """
        todo: Make a visualization for clustering results
        :return:
        """

    def form_pairs(self, start_date='2016-01-04', components=8):
        """
        This is the method we use to form all the pairs when we create this class.
        We only get all the pairings within clusters.

        :param start_date: Cutoff to make sure the price history of every stock in the universe start at the same date
        :param components: Number of dimensions to reduce returns dataframe to
        :return:
        """
        self.get_sp500()
        self.get_price()
        self.prepare_PCA_df(start_time=start_date)
        self.do_PCA(components=components)
        self.do_OPTICS()
        self.cluster_summary()
        pair_list = []
        for cl in self.cluster_dict:
            constituents = self.cluster_dict[cl]
            pairings = combinations(constituents, 2)
            pair_list.extend(pairings)
        self.pairs = pair_list
        self.selected_set = set(self.pairs)
        return self.pairs

    """
    Up to this point, Part A and Part B has been finished.
    Now we need to further select the pairs to trade (Part C).
    
    Filters:
    1. Engle-Granger test.
    2. Hurst exponent.
    3. Mean-reversion Half-life.
    4. Minimum crosses over mean: 12 times.
    """

    @staticmethod
    def hurst(price_df):
        """
        Calculate the Hurst Exponent of the time series.

        While this method itself does not differentiate fractional Brownian motion (fBM) from
        fractional geometric Brownian motion (fGBM), it's important to understand that we're calculating
        the Hurst exponent under the assumption that the underlying dynamics follows a fBM, not a fGBM. This is
        due to the fact that the spread price can easily go negative and log price of the spread does not exist
        if this is the case.

        The Hurst Exponent calculation code reference:
        https://www.quantstart.com/articles/basics-of-statistical-mean-reversion-testing/

        :param price_df: spread price DataFrame
        :return:
        hurst_exponent: The hurst exponent of the time series.
        """
        price = price_df['spread'].to_numpy()
        # this algorithm requires an nparray input, or else np.subtract(price[lag:], price[:-lag]) would produce
        # incorrect results.

        lags = range(2, 100)

        # Calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(np.subtract(price[lag:], price[:-lag]))) for lag in lags]

        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        # Return the Hurst exponent from the polyfit output
        return poly[0] * 2.0

    @staticmethod
    def engle_granger_test(price_x, price_y):
        """
        Here we use the arch package to do the Engle-Granger test.
        During the process, the hedge ratio should be generated as well.
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
        if eg_test1.pvalue >= 0.05 and eg_test2.pvalue >= 0.05:
            # Can't reject null hypothesis. No cointegration. Skip
            return False, None, None
        elif eg_test1.pvalue >= 0.05 and eg_test2.pvalue < 0.05:
            return True, eg_test2.stat, eg_test2.cointegrating_vector
        elif eg_test1.pvalue < 0.05 and eg_test2.pvalue >= 0.05:
            return True, eg_test1.stat, eg_test1.cointegrating_vector
        else:
            if eg_test1.stat < eg_test2.stat:
                return True, eg_test1.stat, eg_test1.cointegrating_vector
            else:
                return True, eg_test2.stat, eg_test2.cointegrating_vector

    @staticmethod
    def half_life(price):
        """
        We'll assume the price follows an AR(1) process so as to estimate the mean reversion half-life.
        :param price: spread price DataFrame
        :return:
        halflife: half_life in days as we're using daily data
        """
        price_copy = deepcopy(price)
        price_copy['one_lagged'] = price_copy['spread'].shift(1)
        price_copy['return'] = price_copy['spread'] - price_copy['one_lagged']
        price_copy.dropna(inplace=True)
        autoreg = sm.OLS(price_copy['return'], sm.add_constant(price_copy['one_lagged']))
        res = autoreg.fit()

        halflife = -np.log(2) / res.params[1]
        return halflife

    @staticmethod
    def mean_crossover(price):
        """
        Calculate how many times the price has crossed over mean.
        :param price: spread price DataFrame
        :return:
        times: The times the spread has crossed over mean
        """

        price_copy = deepcopy(price)
        mean_level = price_copy['spread'].mean()
        price_copy['prev_day'] = price_copy['spread'].shift(1)
        price_copy['crossover'] = ((price_copy['spread'] > mean_level) & (price_copy['prev_day'] <= mean_level)) | \
                                  ((price_copy['spread'] < mean_level) & (price_copy['prev_day'] >= mean_level))
        # Crossover means the previous day is below mean level, but current day closed above mean level
        # Or: the previous day is above mean level, but current day closed below mean level
        xover = price_copy[['crossover']]
        xover['Year'] = xover.index.year
        xover_count = xover.groupby('Year').sum()
        return xover_count

    def cointegrate_filter(self):
        """
        Cointegration is the most important filter. This will determine the hedge ratio.
        Without hedge ratio, the spread cannot be properly calculated.
        If not cointegrated, the pair would not even be selected, and would be removed from selected_set
        The spread data was stored in the spread dictionary for further selection,
         as H and half-life require the spread formed by the cointegrated vector.
        :return:
        Spreaddict
        """
        coint_dict = {}
        for x, y in self.pairs:
            # pricedict[x][x], where x is ticker name
            # first x is dictionary key, second x is DataFrame column name
            x_price = self.pricedict[x][x].loc[self.starting_point:]
            y_price = self.pricedict[y][y].loc[self.starting_point:]
            coint, t_stat, coint_vec = self.engle_granger_test(x_price, y_price)
            if coint:
                tmp = (x_price * coint_vec.loc[x] + y_price * coint_vec.loc[y])
                tmp = tmp.to_frame()
                tmp.columns = ["spread"]
                coint_dict[(x, y)] = tmp
        self.spreaddict = coint_dict
        self.selected_set &= set(self.spreaddict.keys())
        return self.spreaddict

    def hurst_filter(self):
        """
        Use Hurst exponent to filter the pairs
        :return:
        hurst_dict: Hurst Exponent of the selected pairs

        Selected pairs are modified within this method.
        """
        hurst_dict = {}
        for pair in self.spreaddict:
            spread_data = self.spreaddict[pair]
            hurst_exp = self.hurst(spread_data)
            if 0 < hurst_exp < 0.5:
                hurst_dict[pair] = hurst_exp
        self.hurstdict = hurst_dict
        self.selected_set &= set(hurst_dict.keys())

        return self.hurstdict

    def hl_filter(self):
        """
        Use half-life to further select pairs
        :return:
        """
        halflife_dict = {}
        for pair in self.hurstdict:
            spread_data = self.spreaddict[pair]
            hl = self.half_life(spread_data)
            if 1 <= hl <= 252:
                # We'll use 252 days as one-year (trading day convention)
                halflife_dict[pair] = hl
        self.halflifedict = halflife_dict
        self.selected_set &= set(halflife_dict.keys())

        return self.halflifedict

    def crossover_filter(self):
        """
        Use minimal crossover to further select pairs
        However, looking at the selected pairs after the cointegration filter, Hurst filter, half-life filter,
         the times the spread crossed over the mean is not uniformly distributed.
        I consider when the spread became more actively crossing over the mean *recently*, we should be more attracted
         to take advantage of this and trade the pair more.
        e.g. If the pair met the minimum crossover requirement in 2016 and 2017, but not in 2019 and 2020, we would not
         consider the pair as a candidate for pair trading.

        Therefore, right now it's almost November in 2020, we'll use the filter as follows:
         1. It must cross over the mean by 11 times till now.
         2. It must cross over the mean by 12 times in 2019.

        This would be our final selected pairs.

        :return:
        """
        crossover_dict = {}
        for pair in self.halflifedict:
            crossover_count = self.mean_crossover(self.spreaddict[pair])
            count_curr_year = crossover_count['crossover'].tail(1).sum()
            count_two_year = crossover_count['crossover'].tail(2).sum()

            if count_curr_year >= 11 and count_two_year >= 23:
                crossover_dict[pair] = (count_curr_year, count_two_year)

        self.xoverdict = crossover_dict
        self.selected_set &= set(crossover_dict.keys())
        return self.xoverdict
