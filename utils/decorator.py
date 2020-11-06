from functools import wraps
import json
import time
import logging


def log_process(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f'{func.__name__} starting')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info('{} complete in {} seconds'.format(
            func.__name__, int(end-start)))
        return result
    return wrapper
