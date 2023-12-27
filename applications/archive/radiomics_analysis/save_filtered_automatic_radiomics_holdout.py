"""
    @file:              radiomics.py
    @Author:            Maxence Larose

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file shows how to compute and save the data used for radiomics analysis.
"""

import env_apps

import os
import json

import pandas as pd

from constants import (
    AUTOMATIC_RADIOMICS_HOLDOUT_PATH,
    ID,
    LEARNING_TABLE_PATH,
    HOLDOUT_TABLE_PATH,
    HOLDOUT_MASKS_PATH,
    MASKS_PATH,
    TABLE_TASKS
)
from save_filtered_manual_radiomics import RadiomicsDataframe
from src.data.datasets import Feature, TableDataset, Mask
from src.data.transforms import MappingEncoding


if __name__ == "__main__":
    AGE = Feature(column="AGE")
    CLINICAL_STAGE = Feature(column="CLINICAL_STAGE", transform=MappingEncoding({"T1-T2": 0, "T3a": 1}))
    GLEASON_GLOBAL = Feature(column="GLEASON_GLOBAL")
    GLEASON_PRIMARY = Feature(column="GLEASON_PRIMARY")
    GLEASON_SECONDARY = Feature(column="GLEASON_SECONDARY")
    PSA = Feature(column="PSA")

    learning_df = pd.read_csv(LEARNING_TABLE_PATH)
    holdout_df = pd.read_csv(HOLDOUT_TABLE_PATH)

    df = pd.concat([learning_df, holdout_df], ignore_index=True)

    masks = json.load(open(HOLDOUT_MASKS_PATH, "r"))

    # Outer and inner split
    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=[AGE, PSA],
        categorical_features=[CLINICAL_STAGE, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY]
    )

    radiomics_dataframe = RadiomicsDataframe(table_dataset=table_dataset)

    ct_df = radiomics_dataframe.get_radiomics_dataframe(
        os.path.join(AUTOMATIC_RADIOMICS_HOLDOUT_PATH, f"ct.csv"), "CT"
    )
    pt_df = radiomics_dataframe.get_radiomics_dataframe(
        os.path.join(AUTOMATIC_RADIOMICS_HOLDOUT_PATH, f"pt.csv"), "PT"
    )
    radiomics_df = pd.concat([ct_df, pt_df], axis=1)

    radiomics_dataframe.save_final_dataframe(
        path_to_folder=AUTOMATIC_RADIOMICS_HOLDOUT_PATH,
        radiomics_df=radiomics_df,
        train_mask=masks[Mask.TRAIN] + masks[Mask.VALID],
        test_mask=masks[Mask.TEST],
        clinical_stage_column=CLINICAL_STAGE.column,
        mapping={0: "T1-T2", 1: "T3a"}
    )
