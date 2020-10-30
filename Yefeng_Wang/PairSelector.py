from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from PriceData import PriceData


class PairSelector(PriceData):
    """
    Using PCA and OPTICS to select pairs. Stock price data were inherited from the PriceData class.

    Methods:
        set_pca_component: Set the number of PCs for PCA.
        plot_pca: Make the pair plot of PCs
        plot_clusters(figh, figw): Make the t-SNE plot of OPTICS
        form_pairs: Perform PCA and OPTICS to narrow down the pair selection range.

    Properties:
        pairs: read-only. PCA and OPTICS selection results.
        cluster_dict: read-only.
    """
    def __init__(self, start_date):
        """
        Constructor for Pair Selector class.

        Attribute:
            pca_component (int): Public attribute, define the number of PCs used in PCA.
            pca_repr (DataFrame): Private attribute, stores the compressed representations of
                                    percent returns
            clusterings (nparray): Private attribute, cluster label output of OPTICS.
            cluster_dict (dict): Private attribute, stores tickers under the same cluster labels.
            pairs (set): Private attribute, a set that stores all the pairs generated
                           by PCA and OPTICS.

        :param start_date: A "YYYY-mm-dd" string to indicate the starting point of price series
        :type start_date: str
        """
        super().__init__(start_date)
        self.pca_component = 8
        self.__pca_repr = None
        self.__clusterings = None
        self.__cluster_dict = None
        self.__pairs = None

    def set_pca_component(self, component):
        """
        Set the number of principal components (PC)
        :param component: number of PC
        :type component: int
        :return:
        None
        """
        self.pca_component = component

    def _calc_pct_return(self):
        """
        Calculate the daily percentage return of each stock
        :return:
        returns_data (DataFrame): Percentage returns of each stock
        """
        price_data = self.get_price()
        # Only the first row is NA.
        returns_data = price_data.pct_change().dropna(axis=0)
        return returns_data

    def _do_pca(self):
        """
        Perform PCA on percent returns matrix
        :return:
        pca_repr: PCA representation of the return matrix
        """
        # Standardize the returns first
        returns = self._calc_pct_return()
        scaler = StandardScaler()
        standardized_ret = scaler.fit_transform(returns)

        # Now do the PCA
        pca = PCA(n_components=self.pca_component)
        pca.fit(standardized_ret)
        pca_repr = pca.components_.T

        # Get the new representation into a new DataFrame
        pca_df = pd.DataFrame(pca_repr)
        pca_df.columns = ['Feature {}'.format(x) for x in range(1, self.pca_component+1)]
        pca_df.index = returns.columns
        self.__pca_repr = pca_df

    def _do_optics(self):
        """
        No optimal parameter was mentioned in the paper, so use default parameters here.
        :return:
        clusterings: The cluster label of each stock
        """
        cluster = OPTICS()
        cluster.fit(self.__pca_repr)
        self.__clusterings = cluster.labels_
        return cluster.labels_

    def plot_pca(self):
        """
        Plot a pair plot to show the distribution and pairwise correlation between the principal
            components.
        :return:
        None
        """
        sns.pairplot(self.__pca_repr, diag_kind='kde')

    def _cluster_summary(self):
        """
        Return the clustering of the tickers.
        :return:
        cluster_dict (dict): A dictionary where the keys are cluster labels, and the values are
                             a tuple of tickers assigned to the corresponding cluster label.

        OPTICS will return labels of -1. These tickers don't belong to any of the clusters and
            will be ignored in the future process.
        """
        tickers = self.__pca_repr.index
        cluster_dict = dict()
        num_classes = self.__clusterings.max() + 1
        for i in range(num_classes):
            cluster_dict["Cluster {}".format(i)] = tuple(tickers[self.__clusterings == i])
        self.__cluster_dict = cluster_dict
        return cluster_dict

    def plot_clusters(self, figw=20, figh=15):
        """
        Make a t-SNE plot to visualize the clustering of the tickers
        :param figw: Figure width
        :type figw: int
        :param figh: Figure height
        :type figh: int
        :return:
        None
        """
        # Calculate the TSNE 2-D projection.
        tsne = TSNE(n_components=2)
        tsne_pca = tsne.fit_transform(self.__pca_repr)

        # Convert the np array into a DataFrame
        tsne_pca_df = pd.DataFrame(tsne_pca)

        # Use tickers as index so we know the correspondence between the dots and the tickers
        tsne_pca_df.index = self.__pca_repr.index
        tsne_pca_df.columns = ['TSNE_1', 'TSNE_2']

        # Make sure the cluster variable is categorical not numerical.
        # This will affect the color palette.
        tsne_pca_df['cluster'] = pd.Categorical(self.__clusterings)

        # Plotting.
        plt.figure(figsize=(figw, figh))
        sns.set_style("ticks")
        sns.scatterplot(
            x='TSNE_1',
            y='TSNE_2',
            data=tsne_pca_df,
            hue='cluster',
            palette="cubehelix_r"
        )

    def form_pairs(self):
        """
        This is the API where the users call this function and obtain the pairings.
        :return:
        None.

        The method will save the pairs as an attribute for further filtering.
        """
        self._do_pca()
        self._do_optics()
        self._cluster_summary()
        pair_list = []
        for cl in self.__cluster_dict:
            constituents = self.__cluster_dict[cl]
            pairings = combinations(constituents, 2)
            pair_list.extend(pairings)
        self.__pairs = set(pair_list)

    @property
    def pairs(self):
        """
        Getter for pairings.
        :return:
        pairs (set): Class attribute pairs.
        """
        return self.__pairs

    @property
    def cluster_dict(self):
        """
        Getter for the clustered tickers.
        :return:
        cluster_dict (Dictionary): Class attribute cluster_dict
        """
        return self.__cluster_dict
