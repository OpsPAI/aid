import numpy as np
import pandas as pd

from .time import TimestampAgg


class TTDataset:
    def loadRawData(self, filename):
        df = pd.read_json(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / (10**9)
        return df

    def getCandidateListByDF(self, df):
        df_c = df[['parent_id', 'span_id', 'cmdb_id']]
        df_p = df[['span_id', 'cmdb_id']]
        df = pd.merge(df_c, df_p, how='left', left_on='parent_id',
                      right_on='span_id', suffixes=('_c', '_p'))
        df = df[['cmdb_id_c', 'cmdb_id_p']]
        df = df.dropna(axis=0, how='any')
        df = df[df['cmdb_id_c'] != df['cmdb_id_p']]
        df_g = df.groupby(['cmdb_id_c', 'cmdb_id_p'])
        candidateList = []
        for g in list(df_g):
            tmp = {}
            tmp['c'] = g[0][0]
            tmp['p'] = g[0][1]
            tmp['cnt'] = len(g[1])
            candidateList.append(tmp)
        return candidateList

    def countHttp(self, codes):
        cnt = 0
        for c in codes:
            cnt += (c != 200)
        return cnt

    def getTSDictByDF(self, trace_df, tsAggFunc=TimestampAgg.toMinute):
        cmdbList = list(trace_df['cmdb_id'].unique())

        TSDict = trace_df[['cmdb_id', 'timestamp',
                           'duration', 'httpCode']].copy(deep=True)
        TSDict['timestamp'] = TSDict['timestamp'].apply(
            tsAggFunc).apply(pd.to_datetime)
        TSDict = TSDict.groupby(['cmdb_id', 'timestamp']).agg(
            {'duration': [np.max, np.mean, np.std, 'count'], 'httpCode': [self.countHttp]})

        # TSDict.columns = ['_'.join(col) for col in TSDict.columns]
        TSDict.columns = ['duration_max',
                          'duration_avg',
                          'duration_std',
                          'call_cnt',
                          'http_err_cnt']
        TSDict['http_err_rate'] = TSDict['http_err_cnt'] / TSDict['call_cnt']
        TSDict.drop(columns=['http_err_cnt'], inplace=True)
        return TSDict, cmdbList

    def load(self, fileName, tsAggFunc=TimestampAgg.toMinute):
        trace = self.loadRawData(fileName)
        candidateList = self.getCandidateListByDF(trace)
        TSDict, cmdbList = self.getTSDictByDF(trace, tsAggFunc)
        kpiList = list(TSDict.columns)
        return candidateList, TSDict, cmdbList, kpiList


class HuaweiDataset:
    """
    Huawei Trace Data
    """

    def loadRawData(self, filename):
        trace = pd.read_csv(filename)
        trace['parent_csvc_name'].fillna("Source")
        trace['parent_cmpt_name'].fillna("Source")
        trace['parent_id'] = trace['parent_csvc_name'] + \
            "::" + trace['parent_cmpt_name']
        trace['child_id'] = trace['child_csvc_name'] + \
            "::" + trace['child_cmpt_name']
        trace = trace[trace['parent_id'] != trace['child_id']]
        trace.drop(columns=['parent_csvc_name', 'parent_cmpt_name',
                            'child_csvc_name', 'child_cmpt_name'],
                   inplace=True)
        return trace

    def getCandidateListByDF(self, df):
        df = df.groupby(['parent_id', 'child_id']).agg(
            {'call_num_sum': np.sum}).reset_index()
        candidateList = []
        for i in range(df.shape[0]):
            candidateList.append({
                'c': df.iloc[i]['child_id'],
                'p': df.iloc[i]['parent_id'],
                'cnt': df.iloc[i]['call_num_sum']
            })
        return candidateList

    def getTSDictByDF(self, df, tsAggFunc, tsAggFreq):
        # need to process trace
        cmdbList = list(df['child_id'].unique())

        df['from_duration_sum'] = df['from_duration_avg'] * df['call_num_sum']
        df['to_duration_sum'] = df['to_duration_avg'] * df['call_num_sum']
        df['from_err_num_sum'] = df['from_err_num_avg'] * df['call_num_sum']
        df['to_err_num_sum'] = df['to_err_num_avg'] * df['call_num_sum']
        # df['timeout_num_sum'] = df['timeout_num_avg'] * df['call_num_sum']
        df['ts'] = df['ts'].apply(tsAggFunc, args=(
            tsAggFreq,)).apply(pd.to_datetime)
        tmpdf = df.groupby(['child_id', 'ts']).agg({
            'call_num_sum': np.sum,
            'from_duration_sum': np.sum,
            'from_duration_max': np.max,
            'to_duration_sum': np.sum,
            'to_duration_max': np.max,
            'from_err_num_sum': np.sum,
            'from_err_num_max': np.max,
            'to_err_num_sum': np.sum,
            'to_err_num_max': np.max
        })
        tmpdf['from_duration_avg'] = tmpdf['from_duration_sum'] / \
            tmpdf['call_num_sum']
        tmpdf['to_duration_avg'] = tmpdf['to_duration_sum'] / \
            tmpdf['call_num_sum']
        tmpdf['from_err_rate'] = tmpdf['from_err_num_sum'] / \
            tmpdf['call_num_sum']
        tmpdf['to_err_rate'] = tmpdf['to_err_num_sum'] / \
            tmpdf['call_num_sum']
        tmpdf.drop(columns=['from_duration_sum',
                            'to_duration_sum',
                            'from_err_num_sum',
                            'to_err_num_sum'], inplace=True)

        TSDict = tmpdf
        return TSDict, cmdbList

    def load(self, fileName, tsAggFunc, tsAggFreq):
        trace = self.loadRawData(fileName)
        candidateList = self.getCandidateListByDF(trace)
        TSDict, cmdbList = self.getTSDictByDF(trace, tsAggFunc, tsAggFreq)
        kpiList = list(TSDict.columns)
        return candidateList, TSDict, cmdbList, kpiList
