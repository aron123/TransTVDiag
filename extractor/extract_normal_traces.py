import pandas as pd
from utils import io_util
from utils.time_util import time2stamp

trace_csv_input_path = 'data/gaia/csv/trace.csv'
labels_input_path = 'data/gaia/gaia.csv'
output_file = 'data/gaia/pkl/normal_traces.pkl'

print('Loading labels ...')
label_df = pd.read_csv(labels_input_path)
label_df['st_time'] = label_df['st_time'].apply(time2stamp)
label_df['ed_time'] = label_df['ed_time'].apply(time2stamp)
print('Labels loaded.')

print('Loading traces ...')
df = pd.read_csv(trace_csv_input_path)
print(f'Traces loaded ({len(df)} rows).')

# Combine all label conditions into a single mask for efficiency
mask = pd.Series(False, index=df.index)  # Start with no failures
for idx, label_row in label_df.iterrows():
    print(f'Processing {idx} ...')
    condition = (df['start_time'] >= label_row['st_time']) & (df['end_time'] <= label_row['ed_time'])
    mask |= condition  # Accumulate failure masks for all labels

# Drop all failure traces at once
failure_traces = df[mask]
df = df[~mask]  # Keep only the "normal" traces
print(f'{len(failure_traces)} total traces removed (remaining: {len(df)}).')

io_util.save(output_file, df)
