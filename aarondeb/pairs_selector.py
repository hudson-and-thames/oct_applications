import pandas as pd
import numpy as np
import math

from sklearn.utils import *
from sklearn.decomposition import *
from sklearn.preprocessing import *
from sklearn.cluster import *

import itertools

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl

import stat_arb_utils


class PairsSelector():
    """
    Implementation of the Proposed Pairs Selection Framework in the following paper. The
    method consists of three parts; dimensionality reduction, clustering of features and
    finally the selection of pairs with the use of a set of heuristics.    

    http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf
    """

    def __init__(self, universe: pd.DataFrame):
        """
        Constructor

        Sets up the price series needed for the next step

        :param universe: (pd.DataFrame): Asset prices universe
        """

        self.prices_df = universe

    def dimensionality_reduction(self, num_features: int = 10):
        """
        Processes and scales the prices universe supplied in the constructor, into returns. Then reduces the resulting 
        data using pca down to the amount of dimensions needed to be used as a feature vector in the clustering step.
        Optimal ranges for the dimensions required in the feature vector should be <15.

        :param num_features: (int): Used to select pca n_components to be used in the feature vector
        """

        if self.prices_df is None:
            raise Exception(
                "Please input a valid price series before running this method")

        # cleaning
        returns_df = (self.prices_df - self.prices_df.shift(1))
        returns_df = returns_df / self.prices_df.shift(1)
        returns_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        returns_df.ffill(inplace=True)

        # scaling
        scaler = StandardScaler()
        scaled_returns_df = pd.DataFrame(scaler.fit_transform(returns_df))
        scaled_returns_df.columns = returns_df.columns
        scaled_returns_df.set_index(returns_df.index)
        scaled_returns_df.dropna(inplace=True)

        # reducing
        pca = PCA(n_components=num_features)
        pca.fit(scaled_returns_df)
        self.feature_vector = pd.DataFrame(pca.components_)
        self.feature_vector.columns = returns_df.columns
        self.feature_vector = self.feature_vector.T

    def plot_pca_matrix(self):
        """
        Plots the feature vector on a scatter matrix.
        """
        pd.plotting.scatter_matrix(
            self.feature_vector, alpha=0.2, figsize=(15, 15))

    def cluster(self):
        """
        Second step of the framework; Doing Unsupervised Learning on the feature vector supplied from the first step.
        The clustering method used is OPTICS, chosen mainly for it being basically parameterless
        """

        if self.feature_vector is None:
            raise Exception("The needed feature vector has not been computed yet",
                            "Please run dimensionality_reduction() before this method")

        clust = OPTICS()
        clust.fit(self.feature_vector)
        self.clust = clust

    def plot_clustering_info(self):
        """
        Plots the clusters found on a scatter plot.
        """

        if self.feature_vector is None:
            raise Exception("The needed feature vector has not been computed yet",
                            "Please run dimensionality_reduction() before this method")

        if self.clust is None:
            raise Exception("The needed clusters have not been computed yet",
                            "Please run cluster() before this method")

        space = np.arange(len(self.feature_vector))
        reachability = self.clust.reachability_[self.clust.ordering_]
        labels = self.clust.labels_[self.clust.ordering_]
        no_of_classes = len(np.unique(self.clust.labels_))

        plt.figure(figsize=(10, 5))
        ax1 = plt.subplot()

        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, no_of_classes))

        # OPTICS
        for klass, color in zip(range(0, no_of_classes), colors):
            Xk = self.feature_vector[self.clust.labels_ == klass]
            ax1.plot(Xk.loc[:, 0], Xk.loc[:, 1], alpha=0.7,
                     marker='.', linestyle='None')

        ax1.plot(self.feature_vector.iloc[self.clust.labels_ == -1, 0],
                 self.feature_vector.iloc[self.clust.labels_ == -1, 1], 'k+', alpha=0.1)
        ax1.set_title('Automatic Clustering\nOPTICS')

        plt.tight_layout()
        plt.show()

    def _generate_pairwise_combinations(self) -> list:
        """
        This method will loop through all generated clusters (except -1) and generate 
        pairwise combinations of the assets in each cluster.

        :return pair_combinations: (list) : list of asset name pairs
        """
        c_labels = np.unique(self.clust.labels_[self.clust.labels_ != -1])

        if len(c_labels) == 0:
            raise Exception("No clusters have been found")

        pair_combinations = []

        for c in c_labels:
            cluster_x = self.feature_vector[self.clust.labels_ == c].index
            cluster_x = cluster_x.tolist()

            for combination in list(itertools.combinations(cluster_x, 2)):
                pair_combinations.append(combination)

        return pair_combinations

    def _hurst_criterion(self, pairs: list, hurst_exp_threshold: int = 0.5) -> tuple:
        """
        This method will go through all the pairs given, calculate the needed spread and run
        the hurst exponent test against each one.

        :param pairs: (list) : List of asset name pairs to be analyzed
        :param hurst_exp_threshold: (int) : max hurst threshold value 
        :return (tuple) :  
            spreads_df: (pd.DataFrame) : Hedge ratio adjusted spreads DataFrame
            hurst_pass_pairs: (list) : tuple list of pairs that passed the hurst check
        """

        hurst_pass_pairs = []
        spreads_lst = []
        spreads_cols = []

        if len(pairs) != 0:
            for idx, ep in pairs.iterrows():
                asset_one = self.prices_df.loc[:, idx[1]].values
                asset_two = self.prices_df.loc[:, idx[0]].values

                spread_ts = (asset_one - asset_two*ep['hedge_ratio'])
                hurst_exp = self.hurst(spread_ts)

                if hurst_exp < hurst_exp_threshold:
                    hurst_pass_pairs.append(idx)
                    spreads_lst.append(spread_ts)
                    spreads_cols.append(str(idx))
        else:
            raise Exception("No pairs have been found")

        spreads_df = pd.DataFrame(data=spreads_lst).T
        spreads_df.columns = spreads_cols
        spreads_df.index = pd.to_datetime(self.prices_df.index)

        return spreads_df, hurst_pass_pairs

    def _final_criterions(self, spreads_df: pd.DataFrame, pairs: list, min_crossover_threshold_per_year: int = 12) -> tuple:
        """
        This method consists of the final two criterions checks in the third stage of the proposed
        framework which involves; the calculation and check, of the half life of the given pair spread 
        and the amount of mean crossovers throughout a set period, in this case in a year.  

        :param spreads_df: (pd.DataFrame) : Hedge ratio adjusted spreads DataFrame
        :param pairs: (list) : List of asset name pairs to be analyzed
        :param min_crossover_threshold_per_year: (int) : minimum amount of mean crossovers per year
        :return (tuple) :  
            hl_pass_pairs: (list) : tuple list of final pairs
            final_pairs: (list) : tuple list of final pairs
        """

        hl_pass_pairs = []
        final_pairs = []

        if len(pairs) != 0:
            ou_results = stat_arb_utils.run_ou_tests(
                spreads_df, pairs, test_period='2Y', cross_overs_per_delta=min_crossover_threshold_per_year)

            final_selection = ou_results[1 < ou_results['hl']]

            final_selection = final_selection[ou_results['hl'] < 365]

            hl_pass_pairs = final_selection.index.tolist()

            final_selection = final_selection[ou_results['crossovers'] == True]

            final_pairs = final_selection.index.tolist()

        else:
            raise Exception("No pairs have been found")

        return hl_pass_pairs, final_pairs

    def criterion_selector(self, pvalue_threshold: int = 0.01, hurst_exp_threshold: int = 0.5, min_crossover_threshold_per_year: int = 12) -> list:
        """
        Third step of the framework; The clusters found in step two are used to generate a list of possible pairwise 
        combinations. The combinations generated are then checked to see if they comply with the criteria supplied in the
        paper: the pair being cointegrated, the hurst exponent being <0.5, the spread moves within convenient periods and
        finally that the spread reverts to the mean with enough frequency.

        :param pvalue_threshold: (int) : max p-value threshold to be used in the cointegration tests
        :param hurst_exp_threshold: (int) : max hurst threshold value
        :param min_crossover_threshold_per_year: (int) : minimum amount of mean crossovers per year
        :return final_pairs: (list) : tuple list of final pairs
        """

        if self.clust is None:
            raise Exception("The needed clusters have not been computed yet",
                            "Please run cluster() before this method")

        # Generate needed pairwise combinations and remove unneccessary duplicates.

        cluster_x_cointegration_combinations = self._generate_pairwise_combinations()
        self.cluster_pairs_combinations = cluster_x_cointegration_combinations

        # Selection Criterion One: First, it is imposed that pairs are cointegrated, using a p-value of 1%.

        cointegration_results = stat_arb_utils.run_cointegration_tests(
            self.prices_df, cluster_x_cointegration_combinations)

        passing_pairs = cointegration_results.loc[cointegration_results['pvalue']
                                                  <= pvalue_threshold]

        self.coint_pass_pairs = passing_pairs

        # Selection Criterion Two: Then, the spread’s Hurst exponent, represented by H should be smaller than 0.5.

        spreads_df, hurst_pass_pairs = self._hurst_criterion(
            passing_pairs, hurst_exp_threshold)

        self.spreads_df = spreads_df

        self.hurst_pass_pairs = hurst_pass_pairs

        # Selection Criterion Three & Four: Additionally, the half-life period, represented by hl, should
        # lay between one day and one year. Finally, it is imposed that the spread crosses a mean at least
        # 12 times per year.

        hl_pass_pairs, final_pairs = self._final_criterions(
            spreads_df, hurst_pass_pairs, min_crossover_threshold_per_year)

        self.hl_pass_pairs = hl_pass_pairs

        self.final_pairs = final_pairs

        return final_pairs

    def plot_selected_pairs(self):
        """
        Plots the final selection of pairs.
        """

        if self.final_pairs is None:
            raise Exception("The needed pairs have not been computed yet",
                            "Please run criterion_selector() before this method")
        elif len(self.final_pairs) == 0:
            raise Exception("No valid pairs have been found!")

        fig, axs = plt.subplots(len(self.final_pairs),
                                figsize=(15, 3*len(self.final_pairs)))

        for i, ep in enumerate(self.final_pairs):
            rets_asset_one = np.log(self.prices_df.loc[:, ep[0]]).diff()
            rets_asset_two = np.log(self.prices_df.loc[:, ep[1]]).diff()

            axs[i].plot(rets_asset_one.cumsum())
            axs[i].plot(rets_asset_two.cumsum())
            axs[i].legend([ep[0], ep[1]])

    def print_info(self) -> pd.DataFrame:
        """
        Returns the Pairs Selector Summary statistics.
        The following statistics are included - the number of clusters, total possible pair combinations, 
        the number of pairs that passed the cointegration threshold, the number of pairs that passed the 
        hurst exponent threshold, the number of pairs that passed the half life threshold and the number 
        of final set of pairs.

        :return: (pd.DataFrame) Dataframe of summary statistics.
        """

        no_clusters = len(list(set(self.clust.labels_))) - 1
        no_paircomb = len(self.cluster_pairs_combinations)
        no_hurstpair = len(self.hurst_pass_pairs)
        no_hlpair = len(self.hl_pass_pairs)

        info = []

        info.append(("No. of Clusters", no_clusters))
        info.append(("Total Pair Combinations", no_paircomb))
        info.append(("Pairs passing Coint Test", len(self.coint_pass_pairs)))
        info.append(("Pairs passing Hurst threshold", no_hurstpair))
        info.append(("Pairs passing Half Life threshold", no_hlpair))
        info.append(("Final Set of Pairs", len(self.final_pairs)))
        return pd.DataFrame(info)

    @staticmethod
    def hurst(data: pd.DataFrame, max_lags: int = 100) -> int:
        """
        Hurst Exponent Calculation

        :param data: (pd.DataFrame) Time Series that is going to be analyzed.
        :param: max_lags: (int) Maximum amount of lags to be used calculating tau.
        :return: (int) hurst exponent.
        """
        lags = range(2, max_lags)
        tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag])))
               for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]*2.0
