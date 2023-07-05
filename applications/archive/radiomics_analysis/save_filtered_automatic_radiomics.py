"""
    @file:              radiomics.py
    @Author:            Maxence Larose

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file shows how to compute and save the data used for radiomics analysis.
"""

import env_apps

import os

import pandas as pd

from constants import (
    AUTOMATIC_EXTRACTED_RADIOMICS_PATH,
    AUTOMATIC_FILTERED_RADIOMICS_PATH,
    ID,
    LEARNING_TABLE_PATH,
    MASKS_PATH,
    TABLE_TASKS
)
from save_filtered_manual_radiomics import RadiomicsDataframe
from src.data.datasets import Feature, TableDataset
from src.data.processing.sampling import extract_masks
from src.data.transforms import MappingEncoding


if __name__ == "__main__":
    AGE = Feature(column="AGE")
    CLINICAL_STAGE = Feature(column="CLINICAL_STAGE", transform=MappingEncoding({"T1-T2": 0, "T3a": 1}))
    GLEASON_GLOBAL = Feature(column="GLEASON_GLOBAL")
    GLEASON_PRIMARY = Feature(column="GLEASON_PRIMARY")
    GLEASON_SECONDARY = Feature(column="GLEASON_SECONDARY")
    PSA = Feature(column="PSA")

    masks = extract_masks(MASKS_PATH, k=5, l=5)

    # Outer and inner split
    table_dataset = TableDataset(
        dataframe=pd.read_csv(LEARNING_TABLE_PATH),
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=[AGE, PSA],
        categorical_features=[CLINICAL_STAGE, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY]
    )

    radiomics_dataframe = RadiomicsDataframe(table_dataset=table_dataset)

    radiomics_df = {}
    for k in range(5):
        ct_df = radiomics_dataframe.get_radiomics_dataframe(
            os.path.join(AUTOMATIC_EXTRACTED_RADIOMICS_PATH, f"ct_outer_split_{k}.csv"), "CT"
        )
        pt_df = radiomics_dataframe.get_radiomics_dataframe(
            os.path.join(AUTOMATIC_EXTRACTED_RADIOMICS_PATH, f"pt_outer_split_{k}.csv"), "PT"
        )
        radiomics_df[k] = pd.concat([ct_df, pt_df], axis=1)

    radiomics_dataframe.save_outer_and_inner_splits_dataframes(
        path_to_folder=AUTOMATIC_FILTERED_RADIOMICS_PATH,
        radiomics_df=radiomics_df,
        masks=masks,
        clinical_stage_column=CLINICAL_STAGE.column,
        mapping={0: "T1-T2", 1: "T3a"}
    )
