# -*- coding: utf-8 -*-.
"""
Module for Handling calculations for pairs selection.

@author: Hansen
"""
from itertools import permutations
from statsmodels.tsa.stattools import coint as EG
import numpy as np
from hurst import compute_Hc
from sklearn import linear_model


class Screen:
    """
    Class that houses the 4 screening mechanisms.

        1. Engle-Granger test for cointegration;
        2. Hurst Exponent for mean-reversion;
        3. Half life for mean_reversion;
        4. Min number of corssing the mean.
    """

    def __init__(self):
        pass

    def find_cluster_pairs(self, cluster_asset_dict,
                           return_all_pairs=True):
        """
        Find all pairs within each cluster.

        Parameters
        ----------
        cluster_asset_dict : dict
            key: cluster label, dtype = int
            value: asset label, dtype = int
        return_all_pairs : bool, optional
            Whether you want to return a list of all pairs formed. The default
            is True.

        Returns
        -------
        cluster_pairs_dict : dict
            key: cluster label, dtype = int
            value: available pairs, dtype = tuple
        all_pairs : list of tuples
            Optional. All available pairs stored in one list.
        """
        all_pairs = []
        cluster_pairs_dict = {}
        # Use permutation to get all possible pairs.
        # Note: pair (1,2) is not the same as (2,1)
        for cluster_label in cluster_asset_dict:
            cluster_pairs_dict[cluster_label] = \
                list(permutations(cluster_asset_dict[cluster_label], 2))
            all_pairs.append(cluster_pairs_dict[cluster_label])

        all_pairs = [pairs for sublist in all_pairs for pairs in sublist]

        if return_all_pairs:
            return cluster_pairs_dict, all_pairs
        return cluster_pairs_dict

    def find_pair_spread_dict(self, returns_dict, all_pairs):
        """
        For each pair, find their returns spread time series. Stores in a dict.

        Parameters
        ----------
        returns_dict : dict
            Returns time series dict for all assets.
            key: asset label, dtype = int
            value: returns time series, dtype = array like
        all_pairs : list of tuples
            All available pairs stored in one list.

        Returns
        -------
        pair_spread_dict : dict
            Spread time series for each pair. Stored in a dict.
            key: pair, dtype = tuple of int
            value: return spread time series, dtype = array like
        """
        pair_spread_dict = {}
        for pair in all_pairs:
            pair_spread_dict[pair] = returns_dict[:, pair[0]] \
                          - returns_dict[:, pair[1]]

        return pair_spread_dict

    def eg_screen(self, returns_dict, all_pairs, keep_percent=0):
        """
        Screen out pairs using Engle-Granger Test.

        Engle-Granger test will provide t-statistics for each pair of time
        series tested. Smaller t means higher cointegration but it does not
        commute. This function does the following two screening process:
        1. For each pair (a,b) and its reverse (b,a), only keep the pair with
        smaller t-stats.
        2. Only keeps the pairs of the lowest t-stats for a certain percentage.
        e.g. keep_percent=0 means keeping nothing, keep_percent=100 means
        keeping every pair.

        Parameters
        ----------
        returns_dict : dict
            Returns time series dict for all assets.
            key: asset label, dtype = int
            value: returns time series, dtype = array like
        all_pairs : list of tuples
            All available pairs stored in one list.
        keep_percent : int, [0, 100], optional
            The percentage of pairs that you want to keep ranked by their
            t-stats from Engle-Granger test.
            The default is 0.

        Returns
        -------
        pairs_left : list of tuples
            All available pairs left after the screening, stored in one list.

        """
        # 0. Conduct Engel-Granger test
        pairs_tstats_dict = {}
        for pair in all_pairs:
            pairs_tstats_dict[pair] = [EG(returns_dict[:, pair[0]],
                                          returns_dict[:, pair[1]])[0]
                                       ]

        # 1. Screen for the pair with smaller t stats compared to its reverse
        # Note: a pair's reverse pair may not be in all the pairs we are
        # calculating. In this case we skip this screening process.
        pair_small_t_dict = {}
        for pair in all_pairs:
            rvs_pair = tuple(reversed(pair))  # reverse pair
            if (rvs_pair in all_pairs) \
                and \
               (pairs_tstats_dict[pair][0] <= pairs_tstats_dict[rvs_pair][0]):
                pair_small_t_dict[pair] = pairs_tstats_dict[pair]
            else:
                pair_small_t_dict[pair] = pairs_tstats_dict[pair]

        # 2. Screen for the given percentage, by calculating the threshold
        # t-stats implied by the percentage. If a pair's t-stats < threshold
        # then we keep this pair.
        all_tstats = list(pair_small_t_dict.values())
        threshold = np.percentile(all_tstats, keep_percent)
        pairs_left = [pair for pair in pair_small_t_dict
                      if threshold >= pair_small_t_dict[pair]]

        return pairs_left

    def hurst_screen(self, all_pairs,
                     threshold=0.5, returns_dict=None, pair_spread_dict=None,
                     return_pair_spread_dict=False):
        """
        Use Hurst Exponent for the spread series of each pair to screen pairs.

        Hurst Exponent (H) measures whether a series is mean reverting. If
        H < 0.5 then it is mean-reverting, otherwise not.

        You must input either pair_spread_dict or returns_dict. If the input
        is returns_dict, then the function will calculate pair_spread_dict
        internally. You can toggle return_pair_spread_dict=True to return
        pair_spread_dict.

        Parameters
        ----------
        all_pairs : list of tuples
            All available pairs stored in one list.
        threshold : float, optional
            Hurst Exponent threshold. Pairs with H smaller than this value
            will not be kept. The default is 0.5.
        returns_dict : dict, optional
            Returns time series dict for all assets.
            key: asset label, dtype = int
            value: returns time series, dtype = array like
        pair_spread_dict : dict, optional
            Spread time series for each pair. Stored in a dict.
            key: pair, dtype = tuple of int
            value: return spread time series, dtype = array like
        return_pair_spread_dict : bool, optional
            Whether returning return_pair_spread_dict. The default is False.

        Returns
        -------
        pairs_left : list of tuples
            All available pairs left after the screening, stored in one list.
        pair_spread_dict : dict, optional
            Spread time series for each pair. Stored in a dict.
            key: pair, dtype = tuple of int
            value: return spread time series, dtype = array like
            Will not be returned by default.
        """
        # 1. Calculate spread for each pair, when input is just returns_dict
        if pair_spread_dict is None:
            pair_spread_dict = self.find_pair_spread_dict(returns_dict,
                                                          all_pairs)

        # 2. Calculate the Hurst Exponent value H. Store result in a dict
        # for each pair
        hurst_dict = {}
        for pair in all_pairs:
            hurst_dict[pair] = compute_Hc(pair_spread_dict[pair],
                                          kind='change',
                                          simplified=True)

        # 3. Screen for H < threshold(default 0.5)
        pairs_left = [pair for pair in hurst_dict
                      if hurst_dict[pair][0] < threshold]

        if return_pair_spread_dict:
            return pairs_left, pair_spread_dict
        return pairs_left

    def half_life_screen(self, lower_bound, upper_bound,
                         all_pairs, returns_dict=None,
                         pair_spread_dict=None,
                         return_pair_halflife_dict=False):
        """
        Use half life for the spread series of each pair to screen pairs.

        Half life is calculated by linearly regress the series (mean - lag(1))
        against the diff spread using least square. Call the slope value theta.
        Then the halflife is just:
        half_life = log(2)/theta

        You must input either pair_spread_dict or returns_dict. If the input
        is returns_dict, then the function will calculate pair_spread_dict
        internally. You can toggle return_pair_spread_dict=True to return
        pair_spread_dict.

        Parameters
        ----------
        lower_bound : float
            The lower bound of mean reversion half life for the pairs you
            plan to keep. Unit is in days.
        upper_bound : float
            The upper bound of mean reversion half life for the pairs you
            plan to keep. Unit is in days.
        all_pairs : list of tuples
            All available pairs stored in one list.
        returns_dict : dict, optional
            Returns time series dict for all assets.
            key: asset label, dtype = int
            value: returns time series, dtype = array like
        pair_spread_dict : dict, optional
            Spread time series for each pair. Stored in a dict.
            key: pair, dtype = tuple of int
            value: return spread time series, dtype = array like
        return_pair_halflife_dict : bool, optional
            Whether returning the half life for each pair.
            The default is False.

        Returns
        -------
        pairs_left : list of tuples
            All available pairs left after the screening, stored in one list.
        pair_halflife_dict : dict, optional
            Calculated half life for each pair. Stored in a dict.
            key: pair, dtype = tuple of int
            value: return spread time series, dtype = float
            Will not be returned by default.

        """
        # 0. Calculate spread series dict for each pair
        if pair_spread_dict is None:
            pair_spread_dict = self.find_pair_spread_dict(returns_dict,
                                                          all_pairs)

        # 1. Calculate half-life using linear regression
        lin_reg_model = linear_model.LinearRegression()
        # calculate the diff of spread of order 1
        diff_pair_spread_dict = {pair: np.diff(pair_spread_dict[pair])
                                 for pair in all_pairs}
        pair_halflife_dict = {}
        for pair in all_pairs:
            spread_series = np.array(pair_spread_dict[pair])
            spread_mean = np.mean(spread_series)
            spread_lag = np.roll(spread_series, 1)
            spread_lag[0] = 0
            spread_delta = diff_pair_spread_dict[pair]
            # sklearn needs this vertical matrix shape
            spread_delta = spread_delta.reshape(len(spread_delta), 1)
            spread_lag = spread_lag.reshape(len(spread_lag), 1)
            # 0th element of the lag series os not needed
            lin_reg_model.fit(spread_mean - spread_lag[1:], spread_delta)
            half_life = np.log(2) / lin_reg_model.coef_.item()
            pair_halflife_dict[pair] = half_life

        # 2. Screen for pairs within required half life range
        pairs_left = []
        for pair, halflife in pair_halflife_dict.items():
            if lower_bound < halflife < upper_bound:
                pairs_left.append(pair)

        if return_pair_halflife_dict:
            return pairs_left, pair_halflife_dict

        return pairs_left

    def min_cross_screen(self, all_pairs,
                         returns_dict=None, pair_spread_dict=None,
                         min_num_cross=12, intraday_amount=1,
                         days_per_year=250):
        """
        Use minimum numbers of crossing to screen pairs.

        For convenience, call the number of crossings c. This is a slightly
        crude estimation of c in 2 ways:
        1. The it may not strictly coincide with the calendar year, for
        multiple reasons. However, setting a year as roughly 250 days may
        be good enough to estimate c.
        2. If the data available does not integer number of years, then it
        uses the most recent integer amount of years. e.g. For 9.5 years
        of data it will use the most recent 9 years.

        Pairs to be selected has to have more than the min_num_cross each year.
        This method assumes the data is preprocessed and there is no missing
        dates.

        You must input either pair_spread_dict or returns_dict. If the input
        is returns_dict, then the function will calculate pair_spread_dict
        internally.

        Parameters
        ----------
        all_pairs : list of tuples
            All available pairs stored in one list.
        returns_dict : dict, optional
            Returns time series dict for all assets.
            key: asset label, dtype = int
        pair_spread_dict : dict, optional
            Spread time series for each pair. Stored in a dict.
            key: pair, dtype = tuple of int
            value: return spread time series, dtype = array like
        min_num_cross : int, optional
            Minimum number. The default is 12.
        intraday_amount : int, optional
            The amount of data per trading day. The default is 1.
        days_per_year : int, optional
            The amount of trading days per year in data. The default is 250.

        Returns
        -------
        pairs_left : list of tuples
            All available pairs left after the screening, stored in one list.
        """
        # 0. Calculate spread series dict for each pair
        if pair_spread_dict is None:
            pair_spread_dict = self.find_pair_spread_dict(returns_dict,
                                                          all_pairs)

        # 1. Get the index where each pair crosses its mean
        pair_xing_idx_dict = {}
        diff_spread_dict = {pair: np.diff(pair_spread_dict[pair])
                            for pair in pair_spread_dict}
        max_idx = 0
        for pair in all_pairs:
            spread_series = diff_spread_dict[pair][1:]
            spread_mean = np.mean(spread_series)
            spread_lag = np.roll(diff_spread_dict[pair], 1)[1:]
            # series of +1 and -1, indicating whether the original series is
            # positive or not.
            pm_spread = np.sign(spread_series - spread_mean)
            pm_lag = np.sign(spread_lag - spread_mean)
            # crossing happens when the spread and the lag(1) series has
            # different signs
            pm_xing = np.array(pm_spread * pm_lag)
            # find all the time index where the crossing happens
            pair_xing_idx_dict[pair] = np.argwhere(pm_xing == -1).ravel()
            # also calculate the max length of data across all pairs
            max_index = max(max_idx, len(pair_spread_dict[pair]))

        # 2. Bin counting crossings within each time interval
        pairs_left = []
        for pair in all_pairs:
            cross_each_year = \
                self.bincount_by_year(pair_xing_idx_dict[pair],
                                      minval=0,
                                      maxval=max_index,
                                      intraday_amount=intraday_amount,
                                      days_per_year=days_per_year)
            if np.min(cross_each_year) >= min_num_cross:
                pairs_left.append(pair)

        return pairs_left

    def bincount_by_year(self, array, minval, maxval,
                         intraday_amount=1, days_per_year=250):
        """
        Count number of crossings happening roughly per year.

        Parameters
        ----------
        array : 1D array like
            Time indices.
        minval : float
            Starting time index that you want to bin.
        maxval : float
            Ending time index that you want to bin.
        intraday_amount : int, optional
            The amount of data per trading day. The default is 1.
        days_per_year : int, optional
            The amount of trading days per year in data. The default is 250.

        Returns
        -------
        bin_counter : list of integers
            The number of time indices falling into each time interval.

        """
        bin_length = intraday_amount * days_per_year  # data per year
        residue = maxval % bin_length
        num_bins = (maxval - minval)//bin_length
        bin_counter = np.zeros(num_bins, dtype=int)
        for item in array:
            if item >= residue:
                bin_num = (item-residue)//bin_length
                bin_counter[bin_num] += 1

        return bin_counter
