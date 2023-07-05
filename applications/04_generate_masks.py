"""
    @file:              04_generate_masks.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 03/2023

    @Description:       This script is used to generate masks for hyperparameters tuning.
"""

import env_apps

from json import dump

import pandas as pd

from constants import (
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    ID,
    LEARNING_TABLE_PATH,
    MASKS_PATH,
    PN_TASK,
    SEED
)
from src.data.datasets import TableDataset
from src.data.processing import Sampler


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                  Dataset                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    df = pd.read_csv(LEARNING_TABLE_PATH)

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=PN_TASK,
        continuous_features=CLINICAL_CONTINUOUS_FEATURES,
        categorical_features=CLINICAL_CATEGORICAL_FEATURES
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
    with open(MASKS_PATH, "w") as file:
        dump(masks, file, indent=True)
