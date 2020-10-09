#!/usr/bin/env python3
# coding: utf-8
# author: joelowj

import multiprocessing.pool as mp
import pandas as pd
import yfinance as yf


def download_dataset(filename: str = "universe") -> None:
    tickers = list(
        pd.read_csv(
            "dataset/constituents.csv",
            usecols=["Symbol"]).values.flatten())
    df = yf.download(
        tickers=tickers,
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
