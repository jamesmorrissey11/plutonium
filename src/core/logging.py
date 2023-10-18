import json
import logging
import os
import warnings

import colorlog


def create_logger():
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")
    )
    logger = colorlog.getLogger("my_logger")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def mute_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)


def save_cli_args(args, logger, log_dir):
    with open(
        os.path.join(log_dir, "config.json"),
        "w",
    ) as f:
        json.dump(vars(args), f)
    logger.info(f"Config stored at {log_dir}/config.json")
