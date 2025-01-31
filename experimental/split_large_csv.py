import dask.dataframe as dd
import os

input_file = 'data/gaia/csv/log.csv'
output_dir = 'data/gaia/csv/'

start_date = "2021-07-01"
end_date = "2021-07-31"

print('Reading logs ...')
df = dd.read_csv(input_file, dtype={'datetime': 'str'})

print('Processing csv ...')
current_date = start_date
end_date = end_date

while current_date <= end_date:
    print(f'Collecting logs on {current_date} ...')

    daily_data = df[df['datetime'] == current_date]
    
    if daily_data.shape[0].compute() > 0:
        output_file = os.path.join(output_dir, f"log-{current_date}.csv")
        daily_data.compute().to_csv(output_file, index=False, encoding='utf-8')
        print(f'Saved: {output_file}')

    year, month, day = map(int, current_date.split('-'))
    day += 1

    current_date = f"{year:04d}-{month:02d}-{day:02d}"
