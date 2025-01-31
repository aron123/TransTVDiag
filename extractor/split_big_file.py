import os
from utils import io_util

input_file = 'data/gaia/pkl/gaia.pkl'
output_folder = 'data/gaia/pkl/split/'

print(f'Loading {input_file} ...')
data = io_util.load(input_file)
print('Input loaded.')

idxs=list(data.keys())
length = len(idxs)
i=0
while i < length:
    sub_len = int(length / 20)
    sub_idxs=idxs[i:i+sub_len]
    tmp_dict = {key: data[key] for key in sub_idxs}
    io_util.save(os.path.join(output_folder, f'{i}.pkl'), tmp_dict)
    print(f'{i}.pkl saved.')
    i = i+sub_len
    del tmp_dict
