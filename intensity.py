import json
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy.special import softmax

from utils.time import TimestampAgg
from utils.ts import TSTransform, CompoundTransform
from utils.logger import setupLogging
from utils.dataloader import HuaweiDataset
from model.similarity import DTW, Aggregator


class AID:
    def __init__(self):
        # initialize logger
        loggerName = "AID"
        self._logger = setupLogging('logs', loggerName)
        # initialize data loader
        self._loader = HuaweiDataset()

    def _filterCandidate(self, candidateList):
        """Only return candidate calls whose parents appear as others' children
        Args:
            candidateList: a list of dict indicating calls

        Returns:
            candidateList: a list of filtered calls
        """
        childSet = set(map(lambda x: x['c'], candidateList))
        parentSet = set(map(lambda x: x['p'], candidateList))
        self._logger.info(f"No. of child services: {len(childSet)}")
        self._logger.info(f"No. of parent services: {len(parentSet)}")
        self._logger.info(
            f"No. of services in both child and parent: {len(childSet & parentSet)}")
        self._logger.info(
            f"No. of parent not in child: {len(parentSet-childSet)}")
        self._logger.info(
            f"No. of child not in parent: {len(childSet-parentSet)}")
        filteredCand = sorted(filter(lambda x: x['p'] in childSet, candidateList),
                              key=lambda x: x['cnt'],
                              reverse=True)
        return filteredCand

    def _calculateKPIDistance(self,
                              filteredCand,
                              TSDict,
                              kpiList,
                              rowIdx,
                              transformOperations,
                              mpw: int,
                              metricAggFunc=Aggregator.mean_agg,
                              kpiNorm: str = "minmax"):
        """Calculate the intensity of dependency
        Args:
            filteredCand: a list of filtered candidates, see self.eval()
            TSDict: kpi series, see self.eval()
            kpiList: name of kpis to use, see self.eval()
            rowIdx: the time index (bin index), see self.eval()
            transformOperations: the normalization of time series, see self.eval()
            mpw: max propagation window, check the DSW algorithm for details
            metricAggFunc: how to aggregate the metrics in each bin, default is mean aggregation
            kpiNorm: normalize the distances of the same kpi, can be "minmax" or "softmax"

        Returns:
            candidateList: a list of filtered calls
        """
        def transform(TSDict, cmdbId, kpi, rowIdx):
            srs = pd.Series(TSDict.loc[cmdbId][kpi], index=rowIdx).fillna(0)
            return CompoundTransform(srs, transformOperations)

        for item in filteredCand:
            for kpi in kpiList:
                # TODO
                # if the input array is constant (usually because we cannot detect any error)
                # then we should mark it as UNKNOWN
                item[f'dsw-{kpi}'] = DTW.dsw_distance(
                    transform(TSDict, item['c'], kpi, rowIdx),
                    transform(TSDict, item['p'], kpi, rowIdx),
                    mpw=mpw)

        if kpiNorm == "softmax":
            for kpi in kpiList:
                allValues = np.array(
                    list(map(lambda x: x[f'dsw-{kpi}'], filteredCand)))
                x = softmax(allValues)
                assert len(x) == len(filteredCand)
                for idx, candidate in enumerate(filteredCand):
                    candidate[f'normalized-dsw-{kpi}'] = x[idx]
        elif kpiNorm == "minmax":
            for kpi in kpiList:
                allValues = list(map(lambda x: x[f'dsw-{kpi}'], filteredCand))
                maxValue = np.max(allValues)
                minValue = np.min(allValues)
                for candidate in filteredCand:
                    candidate[f'normalized-dsw-{kpi}'] = candidate[f'dsw-{kpi}'] - minValue
                    if maxValue - minValue > 0:
                        candidate[f'normalized-dsw-{kpi}'] /= maxValue - minValue
        else:
            raise NotImplementedError

        # calculate intensity
        for candidate in filteredCand:
            sims_dsw = []
            for kpi in kpiList:
                sims_dsw.append(candidate[f'normalized-dsw-{kpi}'])
            # distance = 0 -> most similar, so need to use 1-agg
            candidate[f'intensity'] = 1-metricAggFunc(sims_dsw)

        filteredCand.sort(key=lambda x: x[f'intensity'], reverse=True)
        return filteredCand

    def eval(self,
             path: str,
             start: str,
             end: str,
             interval: int = 1,
             transformOperations: List[Tuple] = [('ZN',), ("MA", 15)],
             mpw: int = 5):
        """interface for evaluating dependency intensity

        Args:
            path: csv file name
            start: start date or time, eight-digit date YYYYMMDD
            end: end date or time, eight-digit date YYYYMMDD
            interval: aggregation interval. 1 minute is recommeneded.
            transformOperations

        Returns:
            intensity: a list of dicts, sorted by intensity value, higher
                value indicates higher dependency intensity
        """
        # 1. load file
        self._logger.info(f"File name: {path}")
        candidateList, TSDict, cmdbList, kpiList = self._loader.load(
            path,
            tsAggFunc=TimestampAgg.toFreqMinute,
            tsAggFreq=int(interval))
        self._logger.info(f"Finish loading dataset")

        # 2. preprocess
        # filter candidate
        self._logger.info(
            f"No. of candidates before filter: {len(candidateList)}")
        candidateList = self._filterCandidate(candidateList)
        self._logger.info(
            f"No. of candidates after filter: {len(candidateList)}")

        # filter data point
        def genDate(datestr):
            return f"{datestr[:4]}-{datestr[4:6]}-{datestr[6:8]}"

        rowIdx = pd.date_range(f"{genDate(start)} 00:00:00",
                               f"{genDate(end)} 23:59:00", freq=f'{interval}T')

        self._logger.info(f"Time start: {rowIdx[0]}")
        self._logger.info(f"Time end: {rowIdx[-1]}")

        # 3. Calculate intensity
        self._logger.info("Calculate inensity")
        self._logger.info(f"Applied Transformations: {transformOperations}")
        self._logger.info(f"DSW Max Propagation Window: {mpw}")
        intensityList = self._calculateKPIDistance(candidateList, TSDict, kpiList, rowIdx,
                                                   transformOperations=transformOperations,
                                                   mpw=mpw,
                                                   metricAggFunc=Aggregator.mean_agg)
        self._logger.info("Finish calculating intensity")

        # remove unnecessary attributes
        intensityList = list(map(
            lambda x: {"c": x["c"],
                       "p": x["p"],
                       "intensity": x["intensity"]},
            intensityList)
        )

        return intensityList


if __name__ == "__main__":
    # uasge example
    aid = AID()
    intensity = aid.eval("data/industry/status_1min_20210411.csv.xz",
                         start="20210411",
                         end='20210411')
    with open("intensity.json", 'w') as f:
        json.dump(intensity, f, indent=4)
