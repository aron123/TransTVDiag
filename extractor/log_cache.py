import os
import pandas as pd

class LogCache:
    """
    LogCache supports the log entry extraction implemented in extract_error_data.py, by making it memory efficient
    (just the necessary log entries stays in memory).
    """

    def __init__(self, log_path, days_to_keep=2):
        self.log_path = log_path
        self.days_to_keep = days_to_keep
        self.cache = {}  # key: date (str), value: DataFrame with logs
    
    def _load_log_for_date(self, date):
        file_path = os.path.join(self.log_path, f'log-{date}.csv')
        if os.path.isfile(file_path):
            print(f'Loading log file for {date} ...')
            return pd.read_csv(file_path,
                header=0,
                engine='c',
                delimiter=',',
                quoting=0,
                quotechar='"',
                lineterminator="\n",
                dtype={
                    "datetime": str,
                    "service": str,
                    "message": str,
                    "timestamp": 'int64'
                },
                on_bad_lines='warn'
            )
        else:
            print(f'Log file for {date} not found.')
            return pd.DataFrame(columns=["datetime", "service", "message", "timestamp"])

    def get_logs_for_period(self, start_date):
        """
        Loads the two-day log period starting from the start_date.
        
        start_date: format 'YYYY-MM-DD'
        """

        dates_to_load = [pd.to_datetime(start_date), pd.to_datetime(start_date) + pd.Timedelta(days=1)]
        dates_to_load = [date.strftime('%Y-%m-%d') for date in dates_to_load]
        
        # Remove dates from the cache that are no longer needed
        for date in list(self.cache.keys()):
            if date not in dates_to_load:
                print(f'Removing log for {date} from cache ...')
                del self.cache[date]

        # Load logs from missing days to cache
        for date in dates_to_load:
            if date not in self.cache:
                self.cache[date] = self._load_log_for_date(date)
        
        combined_log_df = pd.concat([self.cache[date] for date in dates_to_load], ignore_index=True)
        return combined_log_df