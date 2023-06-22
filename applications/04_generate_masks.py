"""
    @file:              04_generate_masks.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 03/2023

    @Description:       This script is used to generate masks for hyperparameters tuning.
"""

import env_apps

from json import dump
from os.path import join

import pandas as pd

from constants import *
from src.data.datasets import TableDataset
from src.data.processing import Sampler


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                  Dataset                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    df = pd.read_csv(LEARNING_TABLE_PATH)

    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=PN_TASK,
        cont_features=CONTINUOUS_FEATURES,
        cat_features=CATEGORICAL_FEATURES
    )

    # ----------------------------------------------------------------------------------------------------------- #
    #                                                 Sampling                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    sampler = Sampler(
        dataset=table_dataset,
        n_out_split=5,
        n_in_split=5,
        random_state=SEED
    )

    masks = sampler()
    sampler.visualize_splits(masks)

    # Mask saving
    with open(join(MASKS_PATH, "masks.json"), "w") as file:
        dump(masks, file, indent=True)
