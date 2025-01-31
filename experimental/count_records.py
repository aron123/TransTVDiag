import os
import csv
from concurrent.futures import ProcessPoolExecutor

def count_lines_in_file(file_path):
    try:
        with open(file_path, "r") as file:
            print(f'Processing {file_path} ...')
            reader = csv.reader(file)
            next(reader, None)  # Skip the header
            return sum(1 for _ in reader)
    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")
        return 0

def count_data_lines_in_csv(folder_path):
    total_lines = 0
    csv_files = []

    for filename in os.listdir(folder_path):
        if filename.startswith("business_table_2021-08"):
            continue
        if filename.endswith(".csv"):
            csv_files.append(os.path.join(folder_path, filename))
    
    with ProcessPoolExecutor() as executor:
        results = executor.map(count_lines_in_file, csv_files)
        total_lines = sum(results)

    return total_lines

if __name__ == "__main__":
    folder_path = '/mnt/d/RCA-datasets/GAIA-DataSet-release-v1.10/MicroSS/trace'
    total = count_data_lines_in_csv(folder_path)
    print(f"Total number of data lines across all files: {total}")