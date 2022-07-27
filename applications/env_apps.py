"""
    @file:              env_apps.py
    @Author:            Maxence Larose

    @Creation Date:     06/2022
    @Last modification: 07/2022

    @Description:       This file contains everything needed to configure the application environment. The file should
                        always be imported at the beginning of any application script.
"""

from datetime import datetime
import logging
import logging.config
import os
import sys
import yaml


# Append module root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def configure_logging(
        path_to_configuration_file: str
) -> None:
    now = datetime.now()
    logs_dir = f"logs/{now.strftime('%Y-%m-%d')}"
    logs_file = f"{logs_dir}/{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    os.makedirs(logs_dir, exist_ok=True)

    with open(path_to_configuration_file, 'r') as stream:
        config: dict = yaml.load(stream, Loader=yaml.FullLoader)

    config["handlers"]["file"]["filename"] = logs_file

    logging.config.dictConfig(config)
