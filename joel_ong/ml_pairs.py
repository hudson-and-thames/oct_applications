#!/usr/bin/env python3
# coding: utf-8
# author: joelowj

import logging
import logging.config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import statsmodels.api as sm

from itertools import combinations

from multiprocessing import Pool

from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.stattools import coint

from tqdm import tqdm
from typing import List
from typing import Tuple

from utils import istarmap


logging.config.fileConfig(fname="logger.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class MLPairs:
    """
        Enhancing a Pairs Trading strategy with the application of Machine Learning
        http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf

        This class implements Section III Pair Selection Framework, part
            A. Dimensionality Reduction
            B. Unsupervised Learning Clustering
            C. Pairs Selection Criteria
    """

    def __init__(self, universe: pd.DataFrame, seed: int = 42) -> None:
        """

        """
        self.universe = universe
        self._tickers = universe.columns.tolist()
        self.features = None
        self.cluster_ids = None
        self.valid_cluster_ids = None
        self.selected_pairs = None
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def dim_reduction(self, top_k: int, norm_window: int = None) -> None:
        """
            Find a compact representation for each security using PCA. Instead
            of analyzing the proportion of the total variance explained by each
            principal component, and then use the number of components that
            explain a fixed percentage. A different approach is adopted. Because
            an Unsupervised Learning algorithm is applied using these data features,
            there should be a consideration for the data dimensionality.

            Parameters

              top_k (int):
                Determine number of PCA dimensions, upper bounded at 15.

              norm_window (int):
                Determine the rolling window if we were to apply rolling
                normalization to the price series. If None, we will apply
                normalization using the global mean and standard deviation.

            Returns:

        """
        logger.info(f'Begining Part A. Dimensionality reduction.')
        if top_k <= 0 or top_k > 15:
            logging.info("")
            raise ValueError("""
                The recommended PCA dimensions is lower bounded at 1 and
                upper bounded at 15.
            """)
        universe_df = self.universe.copy(deep=True)
        universe_return_df = universe_df.pct_change() \
            .iloc[1:] \
            .copy(deep=True)
        universe_return_df.fillna(0, inplace=True)
        if norm_window:
            universe_norm_return_df = (
                universe_return_df -
                universe_return_df.rolling(window=norm_window).mean()
            ) / (universe_return_df.rolling(window=norm_window).std())
            universe_norm_return_df.dropna(how='all', axis=0, inplace=True)
        else:
            scaler = StandardScaler()
            universe_norm_return = scaler.fit_transform(universe_return_df)
        pca = PCA(
            n_components=top_k,
            random_state=self.seed,
        ).fit(
            universe_norm_return
        )
        self.features = pd.DataFrame(
            pca.components_.T,
            index=self._tickers
        )

    def plot_principle_components(self, figsize: tuple = (15, 10)) -> None:
        """
            Draw a matrix of scatter plots based on the principle components
            obtained from perform dimensionality reduction.

            Parameters
            ----------
                figsize: tuple
                    width, height of plot in inches.
        """
        if self.features is None:
            raise ValueError(
                """
                Please perform dimensionality reduction to generate feature
                representation for each assets in the universe before running
                this function.
                """
            )
        sm = pd.plotting.scatter_matrix(
            self.features, alpha=.2, figsize=figsize)
        # Hide all ticks
        [s.set_xticks(()) for s in sm.reshape(-1)]
        [s.set_yticks(()) for s in sm.reshape(-1)]

    def clustering(self, technique: str, params: dict) -> None:
        """
            Apply an appropriate clustering algorithm.

            Parameters
            ----------
              technique (str):
                Supported clustering technique,
                1. DBSCAN (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
                2. OPTICS (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html)

              params (dict):
                A dictionary containing the hyperparameters which we can use to
                specify for the clustering algorithm. Please refer to the sklearn
                website for the list of available hyperparameters for the respective
                clustering algorithm.

        """
        logger.info(
            f'Begining Part B. Unsupervised Learning Clustering using {technique}')
        if self.features is None:
            raise ValueError(
                """
                Please perform dimensionality reduction to generate feature
                representation for each assets in the universe before running
                this function.
                """
            )
        if technique == 'dbscan':
            func_ = DBSCAN
        elif technique == 'optics':
            func_ = OPTICS
        else:
            raise ValueError(
                f"The supported clustering techniques are dbscan and optics. Got {technique}.")
        clustering = func_(**params).fit(self.features)
        cluster_ids = pd.DataFrame(
            data=clustering.labels_,
            index=self._tickers,
            columns=['cluster_id']
        )
        self.cluster_ids = cluster_ids
        self.valid_cluster_ids = cluster_ids[cluster_ids['cluster_id'] != -1]
        num_of_pairs = int(len(self._tickers) * (len(self._tickers) - 1) / 2)
        logger.info(
            f'Total number of candidate pairs (before clustering): {num_of_pairs}.')
        num_of_pairs = int((self.valid_cluster_ids.value_counts(
        ) * (self.valid_cluster_ids.value_counts() - 1)).sum() / 2)
        logger.info(
            f'Total number of candidate pairs (after clustering): {num_of_pairs}.')

    def plot_cluster_members(self, figsize: tuple = (15, 10)) -> None:
        """

        Parameters
        ----------
            figsize: tuple
                width, height of plot in inches.
        """
        if self.valid_cluster_ids is None:
            raise ValueError(
                "Please perform clustering to generate clusters of assets before running this function."
            )
        fig, ax = plt.subplots(figsize=figsize)
        plt.barh(
            range(len(self.valid_cluster_ids.value_counts())),
            self.valid_cluster_ids.value_counts(),
        )
        plt.title('Cluster Members')
        plt.xlabel('Number of Stock')
        plt.ylabel('Cluster ID')
        for i, v in enumerate(self.valid_cluster_ids.value_counts().tolist()):
            ax.text(v + 1, i, str(v))
        plt.show()

    def plot_cluster_ts(
        self,
        cluster_id: int,
        figsize: tuple = (
            15,
            10)) -> None:
        """

        Parameters
        ----------
            figsize: tuple
                width, height of plot in inches.
        """
        valid_cluster_ids = self.valid_cluster_ids['cluster_id'].unique(
        ).tolist()
        if cluster_id not in valid_cluster_ids:
            raise ValueError(
                f'{cluster_id} is not a valid cluster id. Available ids are {valid_cluster_ids}')
        members = self.valid_cluster_ids[self.valid_cluster_ids['cluster_id']
                                         == cluster_id].index.tolist()
        (
            np.log(
                self.universe[members] /
                self.universe[members].mean())).plot(
            figsize=figsize,
            title=f'Demean Log Price for Stocks in Cluster {cluster_id}',
            legend=False)
        plt.show()

    def plot_tsne(self, params: dict, figsize: tuple = (15, 10)) -> None:
        """
        Plot t-SNE to visualize principal components obtained from dimensionality
        reduction. t-SNE converts similarities between data points to joint
        probabilities and tries to minimize the Kullback-Leibler divergence
        between the joint probabilities of the low-dimensional embedding and
        the high-dimensional data.
        
        Parameters
        ----------
            figsize: tuple
                width, height of plot in inches.
        """
        params['random_state'] = self.seed
        tsne = TSNE(**params).fit_transform(self.features)

        fig, ax = plt.subplots(figsize=figsize)

        scatter = ax.scatter(
            tsne[(self.cluster_ids.cluster_id != - 1).values.flatten(), 0],
            tsne[(self.cluster_ids.cluster_id != - 1).values.flatten(), 1],
            s=200,
            alpha=0.3,
            c=self.cluster_ids[(self.cluster_ids.cluster_id != -1)].values.flatten(),
        )

        legend1 = ax.legend(
            *scatter.legend_elements(),
            title="Cluster ID"
        )

        scatter = ax.scatter(
            tsne[(self.cluster_ids.cluster_id == - 1).values.flatten(), 0],
            tsne[(self.cluster_ids.cluster_id == - 1).values.flatten(), 1],
            s=200,
            alpha=.05,
        )

        plt.show()

    def pairs_selection(
            self,
            max_lag: int,
            convenient_periods: int,
            mp: bool) -> None:
        """
            A pair is selected if it complies with the four conditions described next:
            1. The pair’s constituents are cointegrated.
            2. The pair’s spread Hurst exponent reveals a mean-reverting character.
            3. The pair’s spread diverges and converges within convenient periods.
            4. The pair’s spread reverts to the mean with enough frequency

            Even after clustering, The number of candidate pairs could still be huge.
            This function support multiprocessing to ensure that we leverage on
            all available computation power to enumerate through the universe of
            candidate pairs.

            Parameters
            ----------
              max_lag (int):

              convenient_periods (int):
                Specify the trading period. This will be use to filter out pairs
                whereby its half-life is not coherent with the trading period.

              mp (bool):
                If True, multiprocessing will be use.

        """
        if self.valid_cluster_ids is None:
            raise ValueError(
                """
                Please perform clustering to generate clusters of assets before
                running this function.
                """
            )
        candidate_pairs = []
        for id in self.valid_cluster_ids.value_counts().index:
            assets = self.valid_cluster_ids[self.valid_cluster_ids['cluster_id'] == id].index.tolist(
            )
            candidate_pairs += [
                pair for pair in combinations(assets, 2)
            ]
        selected_pairs = []
        if mp:
            with Pool() as pool:
                iterable = [
                    (self.universe[asset1],
                     self.universe[asset2],
                        max_lag,
                        convenient_periods)
                    for asset1, asset2 in candidate_pairs
                ]
                for result in tqdm(
                        pool.istarmap(
                            pairs_selection_test,
                            iterable),
                        total=len(iterable)):
                    selected_pairs.append(result)
        else:
            for asset1, asset2 in candidate_pairs:
                selected_pairs.append(
                    pairs_selection_test(
                        self.universe[asset1],
                        self.universe[asset2],
                        max_lag,
                        convenient_periods
                    )
                )
        self.selected_pairs = np.array(candidate_pairs)[
            np.where(selected_pairs)[0]].tolist()
        logger.info(
            f'Total number of selected pairs: {len(self.selected_pairs)}.')


def pairs_selection_test(
        price1: pd.Series,
        price2: pd.Series,
        max_lag: int,
        convenient_periods: int) -> bool:
    """
    Parameters
    ----------
    price1 : pd.Series
        Price series of 1st security.
    price2 : pd.Series
        Price series of 2nd security.
    max_lag: int

    convenient_periods: int
        Specify the trading period. This will be use to filter out pairs
        whereby its half-life is not coherent with the trading period.

    Returns
    -------
    bool
        whether the pairs pass the selection test.
    """
    price1.replace([np.inf, -np.inf], np.nan, inplace=True)
    price1.dropna(inplace=True)
    price2.replace([np.inf, -np.inf], np.nan, inplace=True)
    price2.dropna(inplace=True)
    price1, price2 = price1.align(
        price2,
        join='inner'
    )
    # Cointegrated pairs; Propose that the Engle-Granger test is run for the
    # two possible selections of the dependent variable and that the combination
    # that generated the lowest t-statistic is selected.
    _, pval1, _ = coint(price1, price2)
    _, pval2, _ = coint(price2, price1)
    pval = min(pval1, pval2)
    if pval >= .01:
        return False
    spread = np.log(price1 / price2)
    # Mean-reverting Hurst exponent; It aims to constrain false positives,
    # possibly arising as an effect of the multiple comparisons problem. The
    # condition imposed is that the Hurst exponent associated with the spread
    # of a given pair is enforced to be smaller than 0.5, assuring the process
    # leans towards mean-reversion.
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(spread[lag:], spread[:-lag])))
           for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst_exponent = poly[0] * 2
    if hurst_exponent >= 0.5:
        return False
    # Suitable half-life; the value of theta which is obtained by running a
    # linear regression on the difference between mean of spread and spread,
    # and the difference between tomorrow's value of spread and today's value
    model = sm.OLS(
        (np.mean(spread) - spread).iloc[:-1],
        (spread.shift(-1) - spread).iloc[:-1]
    )
    results = model.fit()
    half_life = -np.log(2) / results.params[0]
    if half_life > convenient_periods:
        return False
    # Monthly Mean Crossing; Enforce that every spread crosses its mean at least
    # once per month, to provide enough liquidity.
    spread_mu = spread.resample('MS').transform('mean')
    delta_sign = (np.sign(spread - spread_mu).diff().dropna() != 0).astype(int)
    num_of_cross_per_year = delta_sign.resample('Y').sum()
    num_of_year = len(num_of_cross_per_year)
    if (num_of_cross_per_year >= 12).sum() / num_of_year < 1:
        return False
    return True
