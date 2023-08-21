"""
    @file:              03_generate_descriptive_analyses.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 04/2023

    @Description:       This script is used to generate a folder containing descriptive analyses of all data.
"""

import env_apps

import pandas as pd

from constants import (
    AGE,
    CLINICAL_STAGE,
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    DESCRIPTIVE_ANALYSIS_PATH,
    GLEASON_GLOBAL,
    GLEASON_PRIMARY,
    GLEASON_SECONDARY,
    HOLDOUT_TABLE_PATH,
    ID,
    LEARNING_TABLE_PATH,
    PSA,
    TABLE_TASKS
)
from src.data.datasets import TableDataset
from src.visualization import TableViewer


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                DataFrames                                                   #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df = pd.read_csv(LEARNING_TABLE_PATH)
    holdout_df = pd.read_csv(HOLDOUT_TABLE_PATH)

    df = pd.concat([learning_df, holdout_df], ignore_index=True)

    # ----------------------------------------------------------------------------------------------------------- #
    #                                            Descriptive analysis                                             #
    # ----------------------------------------------------------------------------------------------------------- #
    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=CLINICAL_CONTINUOUS_FEATURES,
        categorical_features=CLINICAL_CATEGORICAL_FEATURES
    )

    table_dataset.update_masks(
        train_mask=list(range(len(learning_df))),
        test_mask=list(range(len(learning_df), len(learning_df) + len(holdout_df)))
    )

    table_viewer = TableViewer(
        dataset=table_dataset,
        feature_names={
            AGE.column: "Age $($years$)$",
            CLINICAL_STAGE.column: "Clinical stage",
            GLEASON_GLOBAL.column: "Global Gleason",
            GLEASON_PRIMARY.column: "Primary Gleason",
            GLEASON_SECONDARY.column: "Secondary Gleason",
            PSA.column: "PSA $($ng/mL$)$"
        },
        target_names={
            "DEATH": "PCSS",
            "METASTASIS": "MFS",
            "PN": "LNI",
            "BCR": "BCR-FS",
            "CRPC": "CRPC-FS",
            "HTX": "dADT-FS",
        },
        crop={
            PSA.column: ((0, 50), (-0.8, 2.5)),
        }
    )
    table_viewer.save_descriptive_analysis(path_to_save=DESCRIPTIVE_ANALYSIS_PATH)
