import os
import pandas as pd

def count_unique_trace_ids(folder_path):
    unique_trace_ids = set()

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)

            try:
                print(f'Processing {filename} ...')
                df = pd.read_csv(file_path)

                if 'trace_id' in df.columns:
                    unique_trace_ids.update(df['trace_id'].dropna().unique())

                    print(f"Processed '{filename}' with {len(df['trace_id'].unique())} unique trace_ids")
                else:
                    print(f"Column 'trace_id' not found in '{filename}'")

            except Exception as e:
                print(f"Error processing file '{filename}': {e}")

    return len(unique_trace_ids)

# Example usage
if __name__ == "__main__":
    folder_path = '/mnt/d/RCA-datasets/GAIA-DataSet-release-v1.10/MicroSS/trace'
    unique_ids = count_unique_trace_ids(folder_path)
    print(f"Unique 'trace_id' values across all files: {unique_ids}")
