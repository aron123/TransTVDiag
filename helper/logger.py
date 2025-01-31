import logging
import os
import sys

def get_logger(save_dir, task_name):
    LOG_FILE_NAME = '{}.log'.format(task_name)
    os.makedirs(f"{save_dir}", exist_ok=True)

    logger = logging.getLogger(task_name)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(save_dir, LOG_FILE_NAME), 'w+'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logger
