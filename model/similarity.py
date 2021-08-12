from typing import List
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


class DTW:
    @staticmethod
    def naive_dtw_distance(ts_a, ts_b, mww, d=lambda x, y: abs(x-y)**2):
        """Computes dtw distance between two time series

        Args:
            ts_a: time series a
            ts_b: time series b
            mww: max warping window, int
            d: distance function

        Returns:
            dtw distance
        """

        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - mww), min(N, i + mww)):
                choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window
        return cost[-1, -1]

    @staticmethod
    def dsw_distance(ts_c, ts_p, mpw, delta=1, d=lambda x, y: abs(x-y)**2):
        """Computes dsw distance between parent and child

        Args:
            ts_c: time series child
            ts_p: time series parent
            mpw: max propagation window, int
            delta: allowed time shift in the system
            d: distance function

        Returns:
            dsw distance
        """

        # Create cost matrix via broadcasting with large int
        ts_p, ts_c = np.array(ts_p), np.array(ts_c)
        M, N = len(ts_c), len(ts_p)
        cost = np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_p[0], ts_c[0])
        for i in range(1, M):
            cost[i, 0] = cost[i-1, 0] + d(ts_c[i], ts_p[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j-1] + d(ts_c[0], ts_p[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - mpw - delta), min(N, i + delta)):
                choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + d(ts_c[i], ts_p[j])

        # Return DSW
        return cost[-1, -1]


class Correlation:
    @staticmethod
    def pearson(ts_a, ts_b):
        """Computes pearson correlation between two time series

        Args:
            ts_a: time series a
            ts_b: time series b

        Returns:
            r: correlation of a and b
            p: p value
        """
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        r, p = pearsonr(ts_a, ts_b)
        return r, p

    @staticmethod
    def spearman(ts_a, ts_b):
        """Computes spearman correlation between two time series

        Args:
            ts_a: time series a
            ts_b: time series b

        Returns:
            r: correlation of a and b
            p: p value
        """
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        r, p = spearmanr(ts_a, ts_b)
        return r, p

    @staticmethod
    def kendall(ts_a, ts_b):
        """Computes kendall correlation between two time series

        Args:
            ts_a: time series a
            ts_b: time series b

        Returns:
            r: correlation of a and b
            p: p value
        """
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        r, p = kendalltau(ts_a, ts_b)
        return r, p


class Aggregator:
    @staticmethod
    def mean_agg(metrics: List[float]):
        return np.mean(metrics)

    @staticmethod
    def max_agg(metrics: List[float]):
        return np.max(metrics)

    @staticmethod
    def min_agg(metrics: List[float]):
        return np.min(metrics)
