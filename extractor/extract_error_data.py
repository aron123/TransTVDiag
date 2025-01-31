import os
from log_cache import LogCache
from utils import io_util
from utils.time_util import *
import pandas as pd
import time
import gc

trace_input_dir = "/mnt/d/GAIA-DataSet-release-v1.10/MicroSS/trace"
trace_output_dir = "data/gaia/csv/trace.csv"

log_input_dir = "/mnt/d/GAIA-DataSet-release-v1.10/MicroSS/business"
log_output_dir = "data/gaia/csv/log.csv"

log_csv_split_dir = "data/gaia/csv" 

metric_input_dir = "/mnt/d/GAIA-DataSet-release-v1.10/MicroSS/metric"

dataset_pkl_output_path = "data/gaia/pkl/gaia.pkl" 

splitted_dataset_path = 'data/gaia/pkl/split/'

def process_traces():
    def spans_df_left_join(spans_df_ori_1: pd.DataFrame) -> pd.DataFrame:
        spans_df_temp: pd.DataFrame = spans_df_ori_1
        spans_df_ori_1 = spans_df_ori_1.loc[:, ['span_id', 'service_name']]
        spans_df_ori_1.rename(columns={'service_name': 'parent_name'}, inplace=True)
        start_time = time.time()
        spans_df_temp = spans_df_temp.merge(spans_df_ori_1, left_on='parent_id', right_on='span_id', how='left')
        end_time = time.time()
        process_time = end_time - start_time
        print(fr"process time: {process_time}")
        del spans_df_ori_1

        spans_df_temp.rename(columns={'span_id_x': 'span_id'}, inplace=True)
        spans_df_temp.drop(columns=['span_id_y'], inplace=True)
        return spans_df_temp


    dfs = []
    for f in os.listdir(trace_input_dir):
        if f.endswith("2021-07.csv"):
            print(f'Appending {f} ...')
            dfs.append(pd.read_csv(os.path.join(trace_input_dir, f)))

    print('Formatting timestamps ...')
    trace_df = pd.concat(dfs)
    trace_df = spans_df_left_join(trace_df)
    trace_df['timestamp']=trace_df['timestamp'].apply(time2stamp)
    trace_df['start_time']=trace_df['start_time'].apply(time2stamp)
    trace_df['end_time']=trace_df['end_time'].apply(time2stamp)
    trace_df['duration']=trace_df['end_time']-trace_df['start_time']

    trace_df.to_csv(trace_output_dir)


def process_logs():
    def extract_Date(df: pd.DataFrame):
        df.dropna(axis=0, subset=['message'], inplace=True)
        df['timestamp'] = df['message'].map(lambda m: m.split(',')[0])
        df['timestamp'] = df['timestamp'].apply(lambda x: time2stamp(str(x)))
        return df

    dfs = []
    for f in os.listdir(log_input_dir):
        if f.endswith("2021-07.csv"):
            print(f'Processing {f} ...')
            df = pd.read_csv(os.path.join(log_input_dir, f))
            df = extract_Date(df)
            dfs.append(df)
    log_df = pd.concat(dfs)
    log_df.to_csv(log_output_dir)


# def process_metrics():
#     metric_dict = {}
#     for f in os.listdir("metric"):
#         metric_name = f.split("_2021")[0]
#         metric_df = pd.read_csv(f"metric/{f}")
#         # metric_df.set_index('timestamp', inplace=True)
#         metric_df.rename(columns={"value": metric_name}, inplace=True)
#         if metric_name in metric_dict.keys():
#             metric_dict[metric_name] = pd.concat([metric_dict[metric_name], metric_df])
#             # metric_dict[metric_name].sort_index(inplace=True)
#             metric_dict[metric_name].sort_values(by=['timestamp'], ascending=True)
#             metric_dict[metric_name].drop_duplicates(subset=['timestamp'], inplace=True)
#             metric_dict[metric_name].set_index('timestamp', inplace=True)
#         else:
#             metric_dict[metric_name] = metric_df
#
#     dfs = list(metric_dict.values())
#     i, id, n = 0, 0, int(len(dfs) / 10)
#
#     while i < len(dfs):
#         df = pd.concat(dfs[i:i+n], axis=1)
#         df.to_csv(f'metric{id}.csv')
#         i+=n
#         id+=1




def extract_traces(trace_df: pd.DataFrame, start_time, end_time):
    window=60*1000
    con1 = trace_df['timestamp'] > start_time
    con2 = trace_df['timestamp'] < end_time+1*window
    return trace_df[con1 & con2]

@cost_time
def extract_logs(log_df: pd.DataFrame, start_time, end_time):
    con1 = log_df['timestamp'] > start_time
    con2 = log_df['timestamp'] < end_time
    return log_df[con1 & con2]


def extract_metrics(metric_df: pd.DataFrame, start_time, end_time):
    window=1*60*1000
    con1 = metric_df['timestamp'] > start_time-40*window
    con2 = metric_df['timestamp'] < end_time+10*window
    return metric_df[con1 & con2]

if __name__ == '__main__':
    #trace_df = process_traces()
    #log_df = process_logs()

    print("Loading labels ...")

    label_df = pd.read_csv("data/gaia/gaia.csv")
    label_df['st_time'] = label_df['st_time'].apply(lambda x: time2stamp(str(x).split('.')[0]))
    label_df['ed_time'] = label_df['ed_time'].apply(lambda x: time2stamp(str(x).split('.')[0]))
    idxs = label_df.index.values.tolist()

    print("Loading traces, metrics, and logs ...")
    #trace_df = pd.read_csv("data/gaia/csv/trace.csv")
    
    log_cache = LogCache(log_csv_split_dir, 2)

    # reduce the IO count
    # metric_fs = {}
    # for f in os.listdir(metric_input_dir):
    #     if f.startswith("zookeeper") or f.startswith("system"):
    #         continue 

    #     print(f'Loading {f} ...')
    #     metric_fs[f] = pd.read_csv(os.path.join(metric_input_dir, f))
    #     metric_name = f.split("_2021")[0]
    #     metric_fs[f] = metric_fs[f].rename(columns={"value": metric_name})

    print("Loading preprocessed data ...")

    #data = io_util.load(dataset_pkl_output_path)

    print("Extracting traces, metrics, and logs ...")

    # saving logs as pandas DataFrames
    for filename in os.listdir(splitted_dataset_path):
        print(f'Loading {filename} ...')
        data = io_util.load(os.path.join(splitted_dataset_path, filename))

        for idx in data:
            st_time, ed_time = label_df.loc[int(idx)]['st_time'], label_df.loc[int(idx)]['ed_time']

            log_df = log_cache.get_logs_for_period(label_df.loc[int(idx)]['datetime'])
            tmp_log_df = extract_logs(log_df, st_time, ed_time)
            data[idx]['log'] = tmp_log_df #.compute()

        io_util.save(os.path.join(splitted_dataset_path, filename), data)
        print(f'{os.path.join(splitted_dataset_path, filename)} saved.')

        # free up memory
        del data
        gc.collect()


    # with mp.Pool(processes=18) as pool:
    #     for filename in os.listdir(splitted_dataset_path):
    #         print(f'Loading {filename} ...')
    #         data = io_util.load(os.path.join(splitted_dataset_path, filename))

    #         for idx in data:
    #             start_time = time.time()
    #             st_time, ed_time = label_df.loc[int(idx)]['st_time'], label_df.loc[int(idx)]['ed_time']
    #             #data[idx] = {}

    #             # tmp_trace_df = extract_traces(trace_df, st_time, ed_time)
    #             # data[idx] = {}
    #             # data[idx]['trace'] = tmp_trace_df

    #             # tmp_log_df = extract_logs(log_df, st_time, ed_time)
    #             # data[idx]['log'] = tmp_log_df
                
    #             # Process metrics in parallel
    #             results = []
    #             for f_name, metric_f in metric_fs.items():
    #                 result = pool.apply_async(extract_metrics, [metric_f, st_time, ed_time])
    #                 results.append(result)

    #             # Retrieve metric results without calling wait explicitly
    #             data[idx]['metric'] = [res.get() for res in results if not res.get().empty]

    #             end_time = time.time()
    #             process_time = end_time - start_time
    #             print(fr"Completed {idx}, Time taken: {process_time}")

    #         io_util.save(os.path.join(splitted_dataset_path, filename), data)
    #         print(f'{os.path.join(splitted_dataset_path, filename)} saved.')

    #         # free up memory
    #         del data
    #         gc.collect()