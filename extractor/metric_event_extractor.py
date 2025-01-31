import json
import os

import pandas as pd
import utils.io_util as io_util
import utils.detect_util as d_util
from utils.time_util import cost_time, time2stamp

def findSubStr_clark(sourceStr, str, i):
    count = 0
    rs = 0
    for c in sourceStr:
        if c == str:
            count += 1
        if count == i:
            return rs
        rs += 1
    return -1

@cost_time
def extract_events(dfs: list, st_time):
    """extract events using 3-sigma from 
        different metric dataframe

    Args:
        dfs (list): metric dataframes

    Returns:
        events: extracted abnormal events
    """
    events = []
    for df in dfs:      
        full_name = [col for col in df.columns if col != 'timestamp'][0]
        split_idx = findSubStr_clark(full_name, '_', 2) # the second _
        svc_host = full_name[:split_idx]
        metric_name = full_name[split_idx+1:]
        svc, host = svc_host.split('_')[0], svc_host.split('_')[1]

        # if metric_name not in kpis:
        #     continue

        df.fillna(0, inplace=True)
        df.sort_values(by=['timestamp'], inplace=True, ascending=True)

        train_df=df[df['timestamp']<st_time]
        test_df=df[df['timestamp']>st_time]
        
        times = test_df['timestamp'].values
        if len(test_df)==0 or len(train_df)==0:
            continue

        # detect anomaly using 3-sigma
        cur_events, labels = d_util.k_sigma(
            train_arr=train_df[full_name].values,
            test_arr=test_df[full_name].values,
            k=3,
        )

        cur_events = cur_events.tolist()
        if len(cur_events) == 0:
            continue
        ab_t = times[labels==-1]
        
        events.append([ab_t[0], svc, host, metric_name])
    
    # sort by timestamp
    sorted_events = sorted(events, key=lambda e:e[0])
    # remove timestamp
    sorted_events = [e[1:] for e in sorted_events]
    return sorted_events


if __name__ == '__main__':
    res = {}
    out_dir = 'data/gaia/events/metric.json'
    split_dir = 'data/gaia/pkl/split'
    labels = pd.read_csv('data/gaia/gaia.csv')
    labels['st_time']=labels['st_time'].apply(time2stamp)

    for f in os.listdir(split_dir):
        print(f'Loading file {f} ...')
        data=io_util.load(os.path.join(split_dir, f))
        print(f'{split_dir} loaded.')
        for idx in data.keys():
            print(f'Processing failure {idx} ...')
            m_dfs = data[idx]['metric']
            events = extract_events(m_dfs, labels.loc[idx, 'st_time'])
            res[idx] = events
        del data

    print('Writing output ...')
    with open(out_dir, 'w') as f:
        json.dump(res, f)
    print("metric.json saved.")