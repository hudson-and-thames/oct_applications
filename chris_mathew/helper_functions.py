import numpy as np
import pandas as pd
import datetime as dt

from johansen import coint_johansen

from statsmodels.tsa.stattools import coint

from sklearn.cluster import KMeans, DBSCAN, OPTICS

'''
tickers = ['DBA', 'JO', 'CORN', 'WEAT', 'SGG', 'SOYB', 'NIB', 'JJG', 'CANE', 'COW', 'BAL', 'JJA', 'JJS', 'FUD', 'UAG', 'TAGS', 'FUE', 'GRU', 'RJA', 'JJU', 'GLD', 'IAU', 'SLV', 'PDBC', 'DBC', 'USO', 'GSG', 'SGOL', 'DJP', 'GLDM', 'PPLT', 'UGAZ', 'BAR', 'USCI', 'UWT', 'UCO', 'GLTR', 'SIVR', 'DBO', 'UNG', 'DGAZ', 'PALL', 'USLV', 'AGQ', 'FTGC', 'DBB', 'BCI', 'DWT', 'OUNZ', 'OILU', 'GCC', 'UGLD', 'DBP', 'AAAU', 'OIL', 'BNO', 'DGL', 'SCO', 'DBE', 'DGP', 'UGL', 'UCI', 'USL', 'COMB', 'BCM', 'OILK', 'UGA', 'COM', 'CMDY', 'GSP', 'GLDI', 'DJCI', 'DSLV', 'GLDW', 'BOIL', 'USOU', 'ZSL', 'GLL', 'SLVO', 'OILD', 'WTIU', 'DGLD', 'DZZ', 'OILX', 'JJC', 'CPER', 'DBS', 'DTO', 'KOLD', 'DGZ', 'USOI', 'JJN', 'COMG', 'UBG', 'UCIB', 'IAUF', 'JJM', 'JJP', 'JJT', 'JJE', 'UNL', 'AOIL', 'PLTM', 'PGM', 'SDCI', 'USV', 'BCD', 'GAZ', 'OLEM', 'WTID', 'USOD', 'LD', 'RJI', 'RJN', 'RJZ', 'GSC', 'XLE', 'AMLP', 'VDE', 'AMJ', 'EMLP', 'XOP', 'MLPI', 'IXC', 'MLPA', 'MLPX', 'OIH', 'IYE', 'FENY', 'AMZA', 'ATMP', 'ERX', 'TPYP', 'IEO', 'AMU', 'GUSH', 'XES', 'RYE', 'FXN', 'IEZ', 'FCG', 'DIG', 'MLPC', 'MLPQ', 'ENFR', 'NRGU', 'DRIP', 'NRGO', 'KOL', 'PXI', 'ZMLP', 'FILL', 'YGRN', 'NRGZ', 'YMLP', 'NRGD', 'PXE', 'IMLP', 'GASL', 'MLPO', 'PSCE', 'MLPZ', 'ERY', 'FRAK', 'JHME', 'PYPE', 'MLPE', 'CRAK', 'YMLI', 'PXJ', 'DUG', 'AMJL', 'FTXN', 'USAI', 'MLPG', 'MLPB', 'AMUB', 'PPLN', 'GASX', 'XLEY', 'BMLP', 'CHIE', 'DDG', 'MLPY', 'GDX', 'GDXJ', 'NUGT', 'JNUG', 'RING', 'SGDM', 'DUST', 'SGDJ', 'JDST', 'GOEX', 'GOAU', 'GDXX', 'GDXS', 'LIT', 'XME', 'PICK', 'REMX', 'COPX', 'BATT', 'URA', 'NLR', 'SIL', 'SLVP', 'SILJ', 'PHO', 'CGW', 'FIW', 'PIO', 'TBLU']
'''

def get_prices_intra_day(start_date, end_date, tickers, freq):
    '''
    Function downloads intra-day price data using tiingo API and amalgamates in a single data frame with closing price data

    :param start_date: for example '2018-01-01'
    :param end_date: for example '2018-01-31'
    :param tickers: list of tickers
    :param freq: '1min', '5min' etc
    :return: prices =  data frame with closing price data, exceptions = list of tickers that tiingo failed to download data for
    '''
    token = '73de814eb72f01abf14025c3d3b204f0fe7753e3'
    exceptions = []
    frames = []
    prices = pd.DataFrame()

    for ticker in tickers:
        url = f"https://api.tiingo.com/iex/{ticker}/prices?startDate={start_date}&endDate={end_date}&resampleFreq={freq}&token={token}"
        try:
            df = pd.read_json(url)
            df = df.set_index('date')
            cols = list(df)
            cols.remove('close')
            df = df.drop(cols, axis=1)
            df = df.rename(columns={'close': ticker})
            frames.append(df)
        except:
            exceptions.append(ticker)

        if len(frames) > 0:
            prices = pd.concat(frames, axis=1, sort=False)

    return prices, exceptions


def get_prices_daily(start_date, end_date, tickers, price_type):
    '''
    Function downloads daily data and amalgamates into a single dataframe.  User can chose what price field is saved\
    : open, close, adjOpen, adjClose etc using price_type parameter.

    :param start_date: for example '2018-01-01'
    :param end_date: for example '2018-01-31'
    :param tickers:  list of tickers
    :param price_type: 'adjOpen' or 'adjClose' etc
    :return:
    '''


    token = '73de814eb72f01abf14025c3d3b204f0fe7753e3'
    exceptions = []
    frames = []
    prices = pd.DataFrame()

    for ticker in tickers:
        # url = f"https://api.tiingo.com/iex/{ticker}/prices?startDate={start_date}&endDate={end_date}&resampleFreq={freq}&token={token}"
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={token}"
        try:
            df = pd.read_json(url)
            df = df.set_index('date')
            cols = list(df)
            cols.remove(price_type)
            df = df.drop(cols, axis=1)
            df = df.rename(columns={price_type: ticker})
            frames.append(df)
        except:
            exceptions.append(ticker)

        if len(frames) > 0:
            prices = pd.concat(frames, axis=1, sort=False)

    return prices, exceptions


def get_coint_pairs(clusters, prices, pval_thresh=0.05):
    '''
    Function takes tickers in each cluster and generates pairs of tickers in each cluster.  Each pair is tested for \
    cointegration.

    :param clusters: list of list.  Each item in list is a list of tickers in each cluster
    :param pval_thresh: Threshold for significance level
    :param prices: dataframe with price data for tickers in the clusters
    :return: list of list where each item is list is a cluster.  And each cluster is a list of cointegrated pairs of tickers.
    '''
    num_clusters = len(clusters)

    # create buckets
    pairs = []
    for i in range(num_clusters):
        pairs.append([])

    # test each pair in each cluster for cointegration
    for i in range(num_clusters):
        for j in range(len(clusters[i])):
            for k in range(j + 1, len(clusters[i])):
                t1 = clusters[i][j]
                t2 = clusters[i][k]
                pair = (t1, t2)
                s1 = prices[t1]
                s2 = prices[t2]
                try:
                    score, pval, _ = coint(s1, s2)
                except:
                    pval = 100

                if pval < pval_thresh:
                    pairs[i].append(pair)

    return pairs

def get_clusters(data, tickers, min_samples=2):
    '''
    Function generates cluster using OPTICS from sklearn. Tickers are bucketed by cluster, as a list of lists.

    :param X: data for clustering.  It is assumed that this data has already been dimension-reduced.
    :param tickers: list of tickers, since X has been separated from its ticker labels
    :param min_samples: minimum sample size
    :return:
    '''
    op = OPTICS(min_samples)
    op.fit(data)
    labels = op.labels_
    num_clusters = np.max(labels) + 1


    clusters = []
    # create bukets: list of n empty lists
    for i in range(num_clusters):
        clusters.append([])

    # put tickers in buckets according to their cluster label
    for i in range(len(labels)):
        label = labels[i]
        if label != -1:
            clusters[int(label)].append(tickers[i])

    return clusters

def hurst(ts):
    """
    Returns the Hurst Exponent of the time series vector ts

    Source: https://medium.com/bluekiri/simple-stationarity-tests-on-time-series-ad227e2e6d48
    """
    # make sure we are working with an array, convert if necessary
    ts = np.asarray(ts)

    # Helper variables used during calculations
    lagvec = []
    tau = []
    # Create the range of lag values
    lags = range(2, 100)

    #  Step through the different lags
    for lag in lags:
        #  produce value difference with lag
        pdiff = np.subtract(ts[lag:],ts[:-lag])
        #  Write the different lags into a vector
        lagvec.append(lag)
        #  Calculate the variance of the difference vector
        tau.append(np.sqrt(np.std(pdiff)))

    #  linear fit to double-log graph
    m = np.polyfit(np.log10(np.asarray(lagvec)),
                   np.log10(np.asarray(tau).clip(min=0.0000000001)),
                   1)
    # return the calculated hurst exponent
    return m[0]*2.0

def half_life(ts):
    """
    Calculates the half life of a spread.

    Source: https://medium.com/bluekiri/simple-stationarity-tests-on-time-series-ad227e2e6d48
    """
    # make sure we are working with an array, convert if necessary
    ts = np.asarray(ts)

    # delta = p(t) - p(t-1)
    delta_ts = np.diff(ts)

    # calculate the vector of lagged values. lag = 1
    lag_ts = np.vstack([ts[1:], np.ones(len(ts[1:]))]).T

    # calculate the slope of the deltas vs the lagged values
    beta = np.linalg.lstsq(lag_ts, delta_ts, rcond=None)

    # compute and return half life
    return (np.log(2) / beta[0])[0]

def count_crossing(ts):
    '''
    Fucntion counts the number of mean crossing.
    Then finds average per year.
    '''
    mu = np.average(ts)
    num_years = len(ts)/250

    c = 0
    for t in range(1, len(ts)):
        p0=ts[t-1]
        p1=ts[t]
        if (p0-mu) * (p1-mu)<0:
            c+=1

    return c/num_years

def get_spread(pair, prices, coeff=None):
    '''
    Derives cointegrating spread. Cointegration coefficient can be generated on the fly using given pricing data OR pre-calculated coefficeint can be supplied (ie, for forward testing).

    :param pair: pair of tickers
    :param prices: df with prices series.  Can be from formation or validation periods.
    :param coeff: either None for on the fly calculation.  Or an already calculated value can be used
    :return: spread = t1 + coef * t2
    '''
    t1, t2 = pair

    s1 = prices[t1] #.iloc[sdx:edx]
    s2 = prices[t2] #.iloc[sdx:edx]

    data = prices[[t1, t2]]

    data.index = prices.index

    data = data.to_numpy()
    if coeff is None:
        try:
            result = coint_johansen(data, 0, 1)
        except:
            result = 'error in co_integration'
            print(result)

        if result != 'error':

            # Store the value of eigenvector. Using this eigenvector, you can create the spread
            ev = result.evec
            # print('evec', ev)
            eval = result.eig
            # print('eval', eval)
            # Take the transpose and use the first row of eigenvectors as it forms strongest cointegrating spread
            ev = result.evec.T[0]
            # print('evec 2 ',ev)

            # Normalise the eigenvectors by dividing the row with the first value of eigenvector
            ev = ev / ev[0]

            coeff = ev[1]

        spread = s1 + coeff * s2

        return coeff, spread

def get_mean_reverting(spreads, pairs):
    '''
    Function aggregates pairs that are mean reverting, ie have Hurst exponent <0.5
    :param spreads:
    :param pairs:
    :return: a list of mean reverting pairs.
    '''

    if len(spreads) != len(pairs):
        print('error in mean revesrion.  lists not of same length')

    mr_list=[]
    for i in range(len(spreads)):
        spread = spreads[i]
        pair =pairs[i]
        if hurst(spread)<0.5:
            mr_list.append(pair)
    return mr_list

def get_half_life(spreads, pairs):
    '''
    Fucntion aggregates pairs that have a half-life greater than 1 day and less than 250 days (ie, one trading year).
    :param spreads:
    :param pairs:
    :return: a list of pairs with half-lifes 1 < HL < 250
    '''

    if len(spreads) != len(pairs):
        print('error in half-life.  lists not of same length')

    hl_list=[]
    for i in range(len(spreads)):
        spread = spreads[i]
        pair = pairs[i]
        hl = half_life(spread)
        if hl > 1 and hl < 250:
        #if hl < 250:
            hl_list.append(pair)
    return hl_list

def get_mean_crossing(spreads, pairs):
    '''
    Function aggregates pairs that have at least 12 mean crossing per year
    :param spreads:
    :param pairs:
    :return: list of pairs whose spread crosses then mean at least 12 times per year, on average
    '''


    if len(spreads) != len(pairs):
        print('error in mean crossing.  lists not of same length')

    mx_list=[]
    for i in range(len(spreads)):
        spread = spreads[i]
        pair = pairs[i]
        mu=np.average(spread)
        num_years = len(spread)/250

        count = count_crossing(spread)

        if count >= 12:
            mx_list.append(pair)
    return mx_list

def get_back_test(pair, coeff, prices, LB=30):
    '''
    Function applies a simple Bollinger band based trading strategy, with the trigger hard-coded at 2 SDs.
    LB should be consistient with how validation data is captured to take into account the 'warm up window' in calculating rolling means and SD.
    Returns are not compounded.
    Returns do not include slippage or commission.

    :param pair:
    :param coeff:
    :param prices:
    :param LB:
    :return: df with results of backtest
    '''
    pd.options.mode.chained_assignment = None
    # from https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

    t1, t2 = pair
    df = prices[[t1, t2]]

    # calculate spread
    df['spread'] = df[t1] + coeff * df[t2]

    #Normalise spread = (spread - mu)/sd
    df['mu'] = df['spread'].rolling(LB).mean()
    df['sd'] = df['spread'].rolling(LB).std()
    df['Z'] = (df.spread - df.mu) / df.sd

    #placeholders for Trade decision and returns
    r, c = df.shape
    df['trade'] = np.zeros(r)
    df['ret'] = np.zeros(r)


    trade = np.zeros(r)
    ret = np.zeros(r)
    for i in range(r):
        z = df.Z.iloc[i]

        if abs(z) > 2: # place trade if spread is outside of 2 SDs
            trade[i] = -np.sign(z)
        elif i > 0 and df.Z.iloc[i - 1] * z < 0: # since normalized mean spread is zero crossing the mean will be +/- or -/+, hence product of subseqent terms will be negative.
            trade[i] = 0
        elif i > 0: # hold position otherwise.
            trade[i] = trade[i - 1]

        # calcualte return in %
        if i < r - 1:
            ret[i] = (df[t1].iloc[i + 1] / df[t1].iloc[i] - 1) + coeff * (df[t2].iloc[i + 1] / df[t2].iloc[i] - 1)

    df.trade = trade
    df.ret = ret
    df['ret X trade'] = df.trade * df.ret # product gives return from position implied by df['trade'] column
    df['cumsum'] = df['ret X trade'].cumsum()
    df['final return'] = df['cumsum']

    df.to_csv('./backtests/' + t1+ '_' +  t2 + '.csv')
    return df


def back_test_all(pairs, coeffs, prices):
    '''
    Returns dataframe that amalgamates results of all pairs

    :param pairs:
    :param coeffs:
    :param prices:
    :return: dataframe with 'final return' column for each pair
    '''

    frames=[]
    results = pd.DataFrame()

    for i in range(len(pairs)):
        pair = pairs[i]
        coeff = coeffs[i]
        df = get_back_test(pair, coeff, prices)
        cols = list(df)
        cols.remove('final return')
        df = df.drop(cols, axis=1)
        df = df.rename(columns={'final return': pair})
        frames.append(df)

    results = pd.concat(frames, axis=1, sort=False)

    results.to_csv('./backtests/all.csv')
    return results







