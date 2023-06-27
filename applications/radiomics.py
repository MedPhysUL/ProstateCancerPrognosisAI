"""
    @file:              radiomics.py
    @Author:            Maxence Larose

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file shows how to compute and save the data used for radiomics analysis.
"""

import env_apps

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from constants import *
from src.data.datasets import Mask, TableDataset
from src.data.processing.sampling import extract_masks


def plot_features_importance(forest, columns, figsize=(12, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    importances = forest.feature_importances_

    tree_importance_sorted_idx = np.argsort(importances)
    tree_indices = np.arange(0, len(importances)) + 0.5

    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)[tree_importance_sorted_idx][-6:]

    ax.barh(tree_indices[:6], importances[tree_importance_sorted_idx][-6:], height=0.7, xerr=std)
    ax.set_yticks(tree_indices[:6])
    ax.set_yticklabels(columns[tree_importance_sorted_idx][-6:])
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    ax.set_ylim((0, len(importances[-6:])))
    fig.tight_layout()
    plt.show()


def get_targets(path: str, target_column: str):
    df = pd.read_csv(path)
    targets = df.loc[:, [target_column]]
    targets = np.array(targets).ravel()

    return targets


def get_radiomics_df(path: str, modality: str):
    radiomics_df = pd.read_csv(path)
    radiomics_df = radiomics_df.sort_values(["ID"])

    columns = []
    for column in radiomics_df.columns:
        if modality == "PT" and column.startswith("original_shape"):
            continue
        elif column.startswith("diagnostics") or column == "ID":
            continue
        else:
            columns.append(column)

    radiomics_df = radiomics_df[columns]
    radiomics_df = radiomics_df.rename(columns=lambda x: f"{modality}_{x}")

    return radiomics_df


def save_outer_splits_dataframes(
        path_to_clinical_df: str,
        path_to_ct_radiomics_df: str,
        path_to_pt_radiomics_df: str,
        path_to_folder: str,
        masks: dict
):
    learning_df = pd.read_csv(path_to_clinical_df)

    table_dataset = TableDataset(
        dataframe=learning_df,
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=[AGE, PSA],
        categorical_features=[CLINICAL_STAGE, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY]
    )

    ct_radiomics_df = get_radiomics_df(path_to_ct_radiomics_df, "CT")
    pt_radiomics_df = get_radiomics_df(path_to_pt_radiomics_df, "PT")
    radiomics_df = pd.concat([ct_radiomics_df, pt_radiomics_df], axis=1)

    for k, v in masks.items():
        print(f"Outer split {k}")
        train_mask, valid_mask, test_mask, inner_masks = v[Mask.TRAIN], v[Mask.VALID], v[Mask.TEST], v[Mask.INNER]
        table_dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
        dataframe = table_dataset.imputed_dataframe.copy()

        dataframes = [dataframe]
        for task in TABLE_TASKS:
            print(task.target_column)
            targets = dataframe.loc[:, [task.target_column]]
            targets = np.array(targets).ravel()
            targets = targets[train_mask + valid_mask]
            nan_mask = np.isnan(targets)

            y_train = targets[~nan_mask]
            X_train = radiomics_df.iloc[train_mask + valid_mask][~nan_mask]

            forest = RandomForestClassifier(n_estimators=10000, random_state=0)
            forest.fit(X_train, y_train)
            plot_features_importance(forest, radiomics_df.columns)

            importances = forest.feature_importances_
            forest_importances = pd.Series(importances, index=radiomics_df.columns)
            most_important_features = list(forest_importances.nlargest(n=6).index)
            radiomics = radiomics_df[most_important_features].copy()
            radiomics = radiomics.rename(columns=lambda x: f"{task.target_column}_{x}")
            dataframes.append(radiomics)

        named_masks = {"train": train_mask, "valid": valid_mask, "test": test_mask}
        dataframe = pd.concat(dataframes, axis=1)
        dataframe = pd.concat(
            objs=[dataframe.iloc[mask].assign(SETS=name) for name, mask in named_masks.items()],
            ignore_index=True
        )

        dataframe.to_csv(os.path.join(path_to_folder, f"outer_split_{k}.csv"), index=False)


def save_final_dataframe(
        path_to_learning_df: str,
        path_to_learning_ct_radiomics_df: str,
        path_to_learning_pt_radiomics_df: str,
        path_to_holdout_df: str,
        path_to_holdout_ct_radiomics_df: str,
        path_to_holdout_pt_radiomics_df: str,
        path_to_folder: str
):
    learning_df = pd.read_csv(path_to_learning_df)
    holdout_df = pd.read_csv(path_to_holdout_df)

    df = pd.concat([learning_df, holdout_df], ignore_index=True)

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=TABLE_TASKS,
        continuous_features=[AGE, PSA],
        categorical_features=[CLINICAL_STAGE, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY]
    )

    learning_ct_radiomics_df = get_radiomics_df(path_to_learning_ct_radiomics_df, "CT")
    learning_pt_radiomics_df = get_radiomics_df(path_to_learning_pt_radiomics_df, "PT")
    learning_radiomics_df = pd.concat([learning_ct_radiomics_df, learning_pt_radiomics_df], axis=1)

    holdout_ct_radiomics_df = get_radiomics_df(path_to_holdout_ct_radiomics_df, "CT")
    holdout_pt_radiomics_df = get_radiomics_df(path_to_holdout_pt_radiomics_df, "PT")
    holdout_radiomics_df = pd.concat([holdout_ct_radiomics_df, holdout_pt_radiomics_df], axis=1)

    radiomics_df = pd.concat([learning_radiomics_df, holdout_radiomics_df], ignore_index=True)

    train_mask = list(range(len(learning_df)))
    test_mask = list(range(len(learning_df), len(learning_df) + len(holdout_df)))

    table_dataset.update_masks(
        train_mask=train_mask,
        test_mask=test_mask
    )

    dataframe = table_dataset.imputed_dataframe.copy()

    dataframes = [dataframe]
    for task in TABLE_TASKS:
        print(task.target_column)
        targets = dataframe.loc[:, [task.target_column]]
        targets = np.array(targets).ravel()
        targets = targets[train_mask]
        nan_mask = np.isnan(targets)

        y_train = targets[~nan_mask]
        X_train = radiomics_df.iloc[train_mask][~nan_mask]

        forest = RandomForestClassifier(n_estimators=10000, random_state=0)
        forest.fit(X_train, y_train)
        plot_features_importance(forest, radiomics_df.columns)

        importances = forest.feature_importances_
        forest_importances = pd.Series(importances, index=radiomics_df.columns)
        most_important_features = list(forest_importances.nlargest(n=6).index)
        radiomics = radiomics_df[most_important_features].copy()
        radiomics = radiomics.rename(columns=lambda x: f"{task.target_column}_{x}")
        dataframes.append(radiomics)

    named_masks = {"train": train_mask, "test": test_mask}
    dataframe = pd.concat(dataframes, axis=1)
    dataframe = pd.concat(
        objs=[dataframe.iloc[mask].assign(SETS=name) for name, mask in named_masks.items()],
        ignore_index=True
    )

    dataframe.to_csv(os.path.join(path_to_folder, f"final_set.csv"), index=False)


if __name__ == "__main__":
    AGE = Feature(column="AGE")
    CLINICAL_STAGE = Feature(column="CLINICAL_STAGE", transform=MappingEncoding({"T1-T2": 0, "T3a": 1}))
    GLEASON_GLOBAL = Feature(column="GLEASON_GLOBAL")
    GLEASON_PRIMARY = Feature(column="GLEASON_PRIMARY")
    GLEASON_SECONDARY = Feature(column="GLEASON_SECONDARY")
    PSA = Feature(column="PSA")

    masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=5, l=5)

    save_outer_splits_dataframes(
        path_to_clinical_df="local_data/learning_table.csv",
        path_to_ct_radiomics_df="local_data/learning_ct_radiomics.csv",
        path_to_pt_radiomics_df="local_data/learning_pt_radiomics.csv",
        path_to_folder="local_data/radiomics/",
        masks=masks
    )

    save_final_dataframe(
        path_to_learning_df="local_data/learning_table.csv",
        path_to_learning_ct_radiomics_df="local_data/learning_ct_radiomics.csv",
        path_to_learning_pt_radiomics_df="local_data/learning_pt_radiomics.csv",
        path_to_holdout_df="local_data/holdout_table.csv",
        path_to_holdout_ct_radiomics_df="local_data/holdout_ct_radiomics.csv",
        path_to_holdout_pt_radiomics_df="local_data/holdout_pt_radiomics.csv",
        path_to_folder="local_data/radiomics/"
    )
