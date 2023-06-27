"""
    @file:              nomograms.py
    @Author:            Maxence Larose

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file shows how to compute and save the clinical data used for nomograms.
"""

import env_apps

import pandas as pd

from constants import *
from src.data.datasets import Mask, TableDataset
from src.data.processing.sampling import extract_masks


def save_outer_splits_dataframes(
        path_to_df: str,
        path_to_folder: str,
        stage_feature: Feature,
        masks: dict,
        mapping: dict = None
):
    learning_df = pd.read_csv(path_to_df)

    table_dataset = TableDataset(
        dataframe=learning_df,
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=[AGE, PSA],
        categorical_features=[stage_feature, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY]
    )

    for k, v in masks.items():
        train_mask, valid_mask, test_mask, inner_masks = v[Mask.TRAIN], v[Mask.VALID], v[Mask.TEST], v[Mask.INNER]
        table_dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
        dataframe = table_dataset.imputed_dataframe.copy()

        if mapping:
            dataframe[stage_feature.column] = dataframe[stage_feature.column].map(mapping)

        named_masks = {"train": train_mask, "valid": valid_mask, "test": test_mask}
        dataframe = pd.concat(
            objs=[dataframe.iloc[mask].assign(SETS=name) for name, mask in named_masks.items()],
            ignore_index=True
        )

        dataframe.to_csv(os.path.join(path_to_folder, f"outer_split_{k}.csv"), index=False)


def save_final_dataframe(
        path_to_learning_df: str,
        path_to_holdout_df: str,
        path_to_folder: str,
        stage_feature: Feature,
        mapping: dict = None
):
    learning_df = pd.read_csv(path_to_learning_df)
    holdout_df = pd.read_csv(path_to_holdout_df)

    df = pd.concat([learning_df, holdout_df], ignore_index=True)

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=[AGE, PSA],
        categorical_features=[stage_feature, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY]
    )

    train_mask = list(range(len(learning_df)))
    test_mask = list(range(len(learning_df), len(learning_df) + len(holdout_df)))

    table_dataset.update_masks(
        train_mask=train_mask,
        test_mask=test_mask
    )

    dataframe = table_dataset.imputed_dataframe.copy()
    if mapping:
        dataframe[stage_feature.column] = dataframe[stage_feature.column].map(mapping)

    named_masks = {"train": train_mask, "test": test_mask}
    dataframe = pd.concat(
        objs=[dataframe.iloc[mask].assign(SETS=name) for name, mask in named_masks.items()],
        ignore_index=True
    )

    dataframe.to_csv(os.path.join(path_to_folder, f"final_set.csv"), index=False)


if __name__ == "__main__":
    AGE = Feature(column="AGE")
    CLINICAL_STAGE = Feature(column="CLINICAL_STAGE", transform=MappingEncoding({"T1-T2": 0, "T3a": 1}))
    CLINICAL_STAGE_MSKCC = Feature(
        column="CLINICAL_STAGE_MSKCC_STYLE",
        transform=MappingEncoding(
            {"T1c": 0, "T2": 0.2, "T2a": 0.2, "T2b": 0.4, "T2c": 0.6, "T3": 0.8, "T3a": 0.8, "T3b": 1}
        )
    )
    GLEASON_GLOBAL = Feature(column="GLEASON_GLOBAL")
    GLEASON_PRIMARY = Feature(column="GLEASON_PRIMARY")
    GLEASON_SECONDARY = Feature(column="GLEASON_SECONDARY")
    PSA = Feature(column="PSA")

    masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=5, l=5)

    # CAPRA
    save_outer_splits_dataframes(
        path_to_df="local_data/learning_table.csv",
        path_to_folder="local_data/nomograms/CAPRA/",
        stage_feature=CLINICAL_STAGE,
        masks=masks,
        mapping={0: "T1-T2", 1: "T3a"}
    )
    save_final_dataframe(
        path_to_learning_df="local_data/learning_table.csv",
        path_to_holdout_df="local_data/holdout_table.csv",
        path_to_folder="local_data/nomograms/CAPRA/",
        stage_feature=CLINICAL_STAGE,
        mapping={0: "T1-T2", 1: "T3a"}
    )

    # MSKCC
    save_outer_splits_dataframes(
        path_to_df="local_data/mskcc_learning_table.csv",
        path_to_folder="local_data/nomograms/MSKCC/",
        stage_feature=CLINICAL_STAGE_MSKCC,
        masks=masks,
        mapping={0: "T1c", 0.2: "T2a", 0.4: "T2b", 0.6: "T2c", 0.8: "T3a", 1: "T3b"}
    )
    save_final_dataframe(
        path_to_learning_df="local_data/mskcc_learning_table.csv",
        path_to_holdout_df="local_data/mskcc_holdout_table.csv",
        path_to_folder="local_data/nomograms/MSKCC/",
        stage_feature=CLINICAL_STAGE_MSKCC,
        mapping={0: "T1c", 0.2: "T2a", 0.4: "T2b", 0.6: "T2c", 0.8: "T3a", 1: "T3b"}
    )

    # CUSTOM
    save_outer_splits_dataframes(
        path_to_df="local_data/learning_table.csv",
        path_to_folder="local_data/nomograms/CUSTOM/",
        stage_feature=CLINICAL_STAGE,
        masks=masks
    )
    save_final_dataframe(
        path_to_learning_df="local_data/learning_table.csv",
        path_to_holdout_df="local_data/holdout_table.csv",
        path_to_folder="local_data/nomograms/CUSTOM/",
        stage_feature=CLINICAL_STAGE
    )
