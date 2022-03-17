import logging
import os.path as osp


def get_logger(experiment_dir):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logfile_handler = logging.FileHandler(osp.join(experiment_dir, "log.txt"))
    logfile_handler.setLevel(level=logging.INFO)
    logfile_handler.setFormatter(formatter)
    logger.addHandler(logfile_handler)

    return logger
