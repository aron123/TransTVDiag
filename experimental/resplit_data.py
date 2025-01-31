from math import floor
import pandas as pd

input_file = 'data/gaia/label.csv'
output_file = 'data/gaia/label_40_60.csv'
split_ratio = 0.4

in_df = pd.read_csv(input_file)
in_df = in_df.sort_values('st_time')
split_idx = floor(len(in_df) * split_ratio)

in_df['data_type'] = ['train' if i <= split_idx else 'test' for i in range(len(in_df))]

in_df.to_csv(output_file, index=False)

print('Done.')