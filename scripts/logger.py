import logging
import sys


def init_logger(name, log_file=True):
    # log options
    log_format = '%(asctime)s [%(levelname)s]: %(message)s'
    date_format = '%d/%m/%Y %H:%M:%S'
    formatter = logging.Formatter(log_format, date_format)
    level = logging.DEBUG

    # handlers
    file_handler = logging.FileHandler(f'./log/{name}.log', mode='a')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # set logger
    logging.basicConfig(handlers=[], level=level)
    logger = logging.getLogger(name)
    logger.addHandler(console_handler)  # log to console
    if log_file:
        logger.addHandler(file_handler)  # log to file

    return logger