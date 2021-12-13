import logging
import sys
from pathlib import Path


def init_logger(name, log_file=True):
    """
    Initialize a logger

    Parameters:
        name (str): name for the logger and also to its file /log/{name}.log
        log_file (bool): whether to produce a written log or not

    Returns:
        logger (Logger): logging.getLogger() object
    """
    # log options
    log_format = '%(asctime)s [%(levelname)s]: %(message)s'
    date_format = '%d/%m/%Y %H:%M:%S'
    formatter = logging.Formatter(log_format, date_format)
    level = logging.DEBUG

    # handlers
    file_handler = logging.FileHandler(Path(f'log/{name}.log'), mode='a')
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
