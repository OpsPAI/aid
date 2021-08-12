import numpy as np
import pandas as pd
from typing import List, Tuple


class TSTransform:
    @staticmethod
    def DIFF(ts: pd.Series):
        """differential"""
        if not len(ts.shape) == 1:
            raise NotImplementedError('Only support 1-d time series currently')
        new_ts = ts.diff()
        new_ts.iloc[0] = ts.iloc[0]
        return new_ts

    @staticmethod
    def OT(ts: pd.Series):
        """
        offset_translation
        """
        if not len(ts.shape) == 1:
            raise NotImplementedError('Only support 1-d time series currently')
        new_ts = ts - ts.mean()
        return new_ts

    @staticmethod
    def ZN(ts: pd.Series):
        """
        z_normalize
        """
        if not len(ts.shape) == 1:
            raise NotImplementedError('Only support 1-d time series currently')
        new_ts = ts - ts.mean()
        if ts.std() != 0:
            new_ts /= ts.std()
        return new_ts

    @staticmethod
    def MM(ts: pd.Series):
        """
        minmax_normalizem map to [0, 1]
        """
        if not len(ts.shape) == 1:
            raise NotImplementedError('Only support 1-d time series currently')
        new_ts = ts - ts.min()
        if (ts.max()-ts.min()) != 0:
            new_ts /= (ts.max()-ts.min())
        return new_ts

    @staticmethod
    def MA(ts: pd.Series, w=15, type=None):
        """Moving average on time series
        Args:
            ts: time series
            w: window size
            type: window type, average window by default
                See https://pandas.pydata.org/docs/reference/api/pandas.Series.rolling.html

        Returns:
            ts after moving average
        """
        if not len(ts.shape) == 1:
            raise NotImplementedError('Only support 1-d time series currently')
        new_ts = ts.rolling(window=w, win_type=type).mean()
        new_ts.iloc[:w-1] = ts.iloc[:w-1]
        return new_ts

    @staticmethod
    def EMA(ts: pd.Series, w=15):
        """Exponential moving average on time series
        Args:
            ts: time series
            w: window size

        Returns:
            ts after moving average
        """
        if not len(ts.shape) == 1:
            raise NotImplementedError('Only support 1-d time series currently')
        new_ts = ts.rolling(window=w, win_type='exponential').mean()
        new_ts.iloc[:w-1] = ts.iloc[:w-1]
        return new_ts


def CompoundTransform(ts: pd.Series, transforms: List[Tuple]):
    new_ts = ts.copy()
    for transform in transforms:
        operator = getattr(TSTransform, transform[0])
        new_ts = operator(new_ts, *transform[1:])
    return new_ts


def test():
    import matplotlib.pyplot as plt

    ts = pd.Series(np.array([333.53, 334.3, 340.98, 343.55, 338.55, 343.51, 347.64, 352.15, 354.87, 348, 353.54, 356.71, 357.55, 360.5,
                             356.52, 349.52, 337.72, 338.61, 338.37, 344.8, 351.12, 347.68, 348.4, 355.92, 357.75, 351.31, 352.25, 350.6,
                             344.9, 345]), index=range(100, 130))

    fig, ax = plt.subplots(2, 1)
    df = TSTransform.DIFF(ts)
    ax[0].plot(df, label='df')
    ot = TSTransform.OT(ts)
    ax[0].plot(ot, label='ot')
    zn = TSTransform.ZN(ts)
    ax[0].plot(zn, label='zn')
    mm = TSTransform.MM(ts)
    ax[0].plot(mm, label='mm')
    compound1 = TSTransform.MA(TSTransform.ZN(ts), 4)
    ax[0].plot(compound1, label='zn+ma')
    compound2 = CompoundTransform(ts, [('ZN',), ("MA", 4)])
    ax[0].plot(compound2, label='zn+ma 2')
    ax[0].legend()
    ax[1].plot(ts, label='original')
    ma = TSTransform.MA(ts, 4)
    ax[1].plot(ma, label='ma')
    ema = TSTransform.EMA(ts, 4)
    ax[1].plot(ema, label='ema')
    ax[1].legend()
    fig.show()


if __name__ == "__main__":
    # some tests
    test()
