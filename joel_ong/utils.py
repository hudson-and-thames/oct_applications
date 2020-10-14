#!/usr/bin/env python3
# coding: utf-8
# author: joelowj

import multiprocessing.pool as mp
import pandas as pd
import yfinance as yf

from tqdm import tqdm


def download_dataset_yh(universe: list, filename: str = "universe") -> None:
    df = yf.download(
        tickers=universe,
        period="10y",
        interval="1d",
        group_by='ticker',
        auto_adjust=False,
        prepost=False,
        threads=True,
        proxy=None
    )
    df = df.iloc[:, df.columns.get_level_values(
        1) == 'Adj Close'].copy(deep=True)
    df.columns = df.columns.droplevel(level=1)
    df.to_csv(f"dataset/{filename}.csv")
    return


def download_dataset_av(
    ticker: str,
    interval: str,
    time_slices: list,
    api_key: str,
    usecols: list = [
        'time',
        'close']) -> pd.DataFrame:
    df = []
    for time_slice in time_slices:
        temp_df = pd.read_csv(
            f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval={interval}&slice={time_slice}&apikey={api_key}',
            usecols=usecols)
        temp_df['time'] = pd.to_datetime(temp_df['time'])
        temp_df.rename(
            columns={
                'time': 'datetime',
                'close': ticker},
            inplace=True)
        temp_df.set_index('datetime', inplace=True)
        df.append(temp_df)
    df = pd.concat(df, axis=0)
    return df


def mp_dataset_av(
        universe: list,
        interval: str,
        time_slices: list,
        api_key: str,
        usecols: list = [
            'time',
            'close'],
        filename: str = "universe") -> None:
    ticker_data = []
    with mp.Pool() as pool:
        iterable = [
            (ticker, interval, time_slices, api_key, usecols)
            for ticker in universe
        ]
        for result in tqdm(
                pool.istarmap(
                    download_dataset_av,
                    iterable),
                total=len(iterable)):
            ticker_data.append(result)
    df = pd.concat(ticker_data, axis=1)
    df.sort_values('datetime', inplace=True)
    df.to_csv(f"dataset/{filename}.csv")
    return


def istarmap(self, func, iterable, chunksize=1):
    """
        This is a hack to get tqdm working with starmap.
    """
    if self._state != mp.RUN:
        raise ValueError("Pool not running...")
    if chunksize < 1:
        raise ValueError(
            f"Expected chunksize to be equal or more than 1. Got {chunksize}."
        )
    task_batches = mp.Pool._get_tasks(func, iterable, chunksize)
    result = mp.IMapIterator(self._cache)
    self._taskqueue.put(
        (self._guarded_task_generation(
            result._job,
            mp.starmapstar,
            task_batches),
            result._set_length,
         ))
    return (item for chunk in result for item in chunk)


mp.Pool.istarmap = istarmap
