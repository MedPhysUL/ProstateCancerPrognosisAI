"""
    @file:              radiomics.py
    @Author:            Maxence Larose

    @Creation Date:     06/2023
    @Last modification: 07/2023

    @Description:       This file shows how to compute and save the data used for radiomics analysis.
"""

import env_apps

import os

from delia.databases import PatientsDatabase
from monai.data import DataLoader
import numpy as np
import pandas as pd
import torch

from constants import (
    DEEP_FILTERED_RADIOMICS_PATH,
    ID,
    LEARNING_TABLE_PATH,
    MASKS_PATH,
    TABLE_TASKS,
    PROSTATE_SEGMENTATION_TASK,
    SEED
)
from src.data.processing.sampling import extract_masks, Mask
from src.data.datasets import ImageDataset, ProstateCancerDataset, TableDataset
from src.models.torch import UNEXtractor


if __name__ == "__main__":
    df = pd.read_csv(LEARNING_TABLE_PATH)

    masks = extract_masks(MASKS_PATH, k=5, l=5)

    path_to_folder = DEEP_FILTERED_RADIOMICS_PATH

    os.makedirs(path_to_folder, exist_ok=True)
    for task in TABLE_TASKS:
        table_dataset = TableDataset(
            dataframe=df,
            ids_column=ID,
            tasks=task
        )

        database = PatientsDatabase(path_to_database=r"local_data/learning_set.h5")

        image_dataset = ImageDataset(
            database=database,
            modalities={"PT", "CT"},
            tasks=PROSTATE_SEGMENTATION_TASK
        )

        dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=table_dataset)

        os.makedirs(os.path.join(path_to_folder, task.target_column), exist_ok=True)
        for k, v in masks.items():
            outer_split_path = os.path.join(path_to_folder, task.target_column, f"outer_split_{k}")
            os.makedirs(outer_split_path, exist_ok=True)

            dataset.update_masks(
                train_mask=v[Mask.TRAIN],
                test_mask=v[Mask.TEST],
                valid_mask=v[Mask.VALID]
            )

            model = UNEXtractor(
                image_keys=["CT", "PT"],
                dropout_cnn=0.2,
                dropout_fnn=0.2,
                num_res_units=2,
                device=torch.device("cuda"),
                seed=SEED
            ).build(dataset)

            loaded_state = torch.load(
                f"local_data/records/experiments/{task.target_column}(UNEXtractor - Deep radiomics)/outer_splits/"
                f"split_{k}/best_models/outer_split/best_model.pt"
            )

            model.load_state_dict(loaded_state)

            data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=None)
            rads = []
            for features, _ in data_loader:
                rads.append(model.extract_radiomics(features).deep_features.cpu().detach().numpy())
            rads = np.array(rads)[:, 0, :]

            dataframe = df.copy()
            for i in range(6):
                dataframe[f"RADIOMIC_{i + 1}"] = rads[:, i]

            dataframe.to_csv(os.path.join(outer_split_path, f"outer_split.csv"), index=False)

            path_to_inner_splits = os.path.join(outer_split_path, f"inner_splits")
            os.makedirs(path_to_inner_splits, exist_ok=True)
            for idx, inner_mask in v[Mask.INNER].items():
                loaded_state = torch.load(
                    f"local_data/records/experiments/{task.target_column}(UNEXtractor - Deep radiomics)/outer_splits/"
                    f"split_{k}/best_models/inner_splits/split_{idx}/best_model.pt"
                )

                model.load_state_dict(loaded_state)

                data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=None)
                rads = []
                for features, _ in data_loader:
                    rads.append(model.extract_radiomics(features).deep_features.cpu().detach().numpy())
                rads = np.array(rads)[:, 0, :]

                dataframe = df.copy()
                for i in range(6):
                    dataframe[f"RADIOMIC_{i + 1}"] = rads[:, i]

                dataframe.to_csv(os.path.join(path_to_inner_splits, f"inner_split_{idx}.csv"), index=False)
