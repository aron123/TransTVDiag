import os
import gc
import pickle

split_dir = 'data/gaia/pkl/split'

for filename in os.listdir(split_dir):
    with open(os.path.join(split_dir, filename), 'rb') as file:
        print(f'Loading {filename} ...')
        data = pickle.load(file, encoding='bytes')
        print(f'{filename} is not corrupted.')
        del data
        gc.collect()

print('All files are valid.')