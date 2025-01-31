import json
import pandas as pd

from drain.extract_log_templates import *
import utils.io_util as io_util
from utils.time_util import cost_time

def init_parser(logs: list, save_path: str, stats_path: str):
    """ Extract templates of logs
        Transform the logs into embeddings

    Args:
        L_data : The raw log data
    """
    drain_parser = extract_templates(
        log_list=logs,
        save_pth=save_path
    )

    # save templates and ID
    sorted_clusters = sorted(drain_parser.drain.clusters, key=lambda it: it.size, reverse=True)
    uq_tmps, uq_IDs, sizes = [], [], []
    for cluster in sorted_clusters:
        uq_tmps.append(cluster.get_template())
        uq_IDs.append(cluster.cluster_id)
        sizes.append(cluster.size)
    template_df = pd.DataFrame(data={"id": uq_IDs, "template": uq_tmps, 'count': sizes})
    template_df.to_csv(stats_path, index=False)


def processing_feature(svc, log, miner):   
    cluster = miner.match(log)
    if cluster is None:
        eventId = -1
    else:
        eventId = cluster.cluster_id
    res = {'service':svc,'id':eventId, 'count':1}
    return res


@cost_time
def extract_events(log_df: pd.DataFrame, miner: drain3.TemplateMiner, count_dic: dict, k: int):
    log_num = len(log_df)
    print(log_num)
    # templates_total_num = sum(count_dic.values())
    sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=False)

    select_events = []
    e_num=0
    for c in sorted_clusters:
        if 'ERROR' in c.get_template():
            select_events.append(c.cluster_id)
        elif e_num < k:
            select_events.append(c.cluster_id)
            e_num+=1
        else:
            break

    log_df.sort_values(by=['timestamp'], ascending=True, inplace=True)
    logs=log_df['message'].values
    svcs=log_df['service'].values

    events_dict = {'service':[], 'id': [], 'count':[]}
    for i,log in tqdm(enumerate(logs)):
        res=processing_feature(svcs[i], log, miner)
        events_dict['service'].append(res['service'])
        events_dict['id'].append(res['id'])
        events_dict['count'].append(res['count'])
    event_df=pd.DataFrame(events_dict)
    event_df = event_df[event_df['id'].isin(select_events)]
    event_gp = event_df.groupby(['id', 'service'])
    events=[[svc, str(event_id)] for (event_id, svc), _ in event_gp]

    return events


labels_path = 'data/gaia/gaia.csv'
pkl_split_dir = 'data/gaia/pkl/split'

drain_miner_path = 'data/gaia/drain/drain.pkl'
drain_stats_path = 'data/gaia/drain/statistics.csv'

nodes_pkl_path = 'data/gaia/raw/nodes.pkl'

events_output_path = 'data/gaia/events/log.json'

if __name__ == '__main__':
    labels = pd.read_csv(labels_path)
    
    #init drain using train dataset

    train_logs = []
    train_idxs = labels[labels['data_type']=='train']['index'].values.tolist()
    for f in os.listdir(pkl_split_dir):
        print(f'Loading {f} ...')
        data = io_util.load(os.path.join(pkl_split_dir, f))
        for idx in data.keys():
            if idx in train_idxs:
                log_df = data[idx]['log']
                train_logs.extend(log_df['message'].values.tolist())
        del data

    init_parser(train_logs, drain_miner_path, drain_stats_path) # creates drain.pkl and statistics.csv
    

    pods = io_util.load(nodes_pkl_path)

    count_df = pd.read_csv(drain_stats_path)
    count_dic=dict(zip(count_df['id'], count_df['count']))
    parser = io_util.load(drain_miner_path)
    k=20
    res = {}

    for f in os.listdir(pkl_split_dir):
        print(f"Processing {f} ...")
        data=io_util.load(os.path.join(pkl_split_dir, f))
        for idx in data.keys():
            log_df = data[idx]['log']
            events = extract_events(log_df, parser, count_dic, k)
            res[idx] = events
        del data

    with open(events_output_path, 'w') as f:
        json.dump(res, f)
    print('Saved log.json successfully!')
