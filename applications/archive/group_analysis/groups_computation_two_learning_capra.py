import env_apps

import json
import os

import pandas as pd
import torch

from constants import (
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    EXPERIMENTS_PATH,
    FINAL_BAYES_TABLE_PATH,
    MASKS_PATH,
    ID,
    NOMOGRAMS_PATH,
    SEED,
    BCR_TASK,
    CRPC_TASK,
    DEATH_TASK,
    PN_TASK,
    METASTASIS_TASK,
    HTX_TASK
)
from src.data.datasets import Feature
from src.data.transforms import Normalization
from src.data.processing.sampling import Mask
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.visualization.table import TableViewer
from src.models.torch.prediction import SequentialNet

import numpy as np
from sksurv.compare import compare_survival


if __name__ == '__main__':
    df = pd.read_csv(FINAL_BAYES_TABLE_PATH)

    RADIOMIC_1 = Feature(column="RADIOMIC_PN_1", transform=Normalization(), impute=False)
    RADIOMIC_2 = Feature(column="RADIOMIC_PN_2", transform=Normalization(), impute=False)
    RADIOMIC_3 = Feature(column="RADIOMIC_PN_3", transform=Normalization(), impute=False)
    RADIOMIC_4 = Feature(column="RADIOMIC_PN_4", transform=Normalization(), impute=False)
    RADIOMIC_5 = Feature(column="RADIOMIC_PN_5", transform=Normalization(), impute=False)
    RADIOMIC_6 = Feature(column="RADIOMIC_PN_6", transform=Normalization(), impute=False)
    RADIOMIC_7 = Feature(column="RADIOMIC_BCR_1", transform=Normalization(), impute=False)
    RADIOMIC_8 = Feature(column="RADIOMIC_BCR_2", transform=Normalization(), impute=False)
    RADIOMIC_9 = Feature(column="RADIOMIC_BCR_3", transform=Normalization(), impute=False)
    RADIOMIC_10 = Feature(column="RADIOMIC_BCR_4", transform=Normalization(), impute=False)
    RADIOMIC_11 = Feature(column="RADIOMIC_BCR_5", transform=Normalization(), impute=False)
    RADIOMIC_12 = Feature(column="RADIOMIC_BCR_6", transform=Normalization(), impute=False)

    PN_RADIOMICS = [RADIOMIC_1, RADIOMIC_2, RADIOMIC_3, RADIOMIC_4, RADIOMIC_5, RADIOMIC_6]
    BCR_RADIOMICS = [RADIOMIC_7, RADIOMIC_8, RADIOMIC_9, RADIOMIC_10, RADIOMIC_11, RADIOMIC_12]

    RADIOMICS = PN_RADIOMICS + BCR_RADIOMICS

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=[PN_TASK, BCR_TASK, METASTASIS_TASK, HTX_TASK, CRPC_TASK],
        continuous_features=CLINICAL_CONTINUOUS_FEATURES + RADIOMICS,
        categorical_features=CLINICAL_CATEGORICAL_FEATURES
    )

    dataset = ProstateCancerDataset(table_dataset=table_dataset)

    for i in range(5):
        masks = json.load(open(MASKS_PATH, "r"))

        dataset.update_masks(
            train_mask=masks[f"{i}"][Mask.TRAIN],
            test_mask=masks[f"{i}"][Mask.TEST],
            valid_mask=masks[f"{i}"][Mask.VALID]
        )

        for task in [DEATH_TASK]:
            df = dataset.table_dataset.dataframe.iloc[dataset.train_mask + dataset.valid_mask]
            df_copy = df.copy()
            df_copy.dropna(subset=[task.event_indicator_column, task.event_time_column], inplace=True)

            value_counts = df[task.target_column].value_counts()
            ratio_0 = value_counts[0] / (value_counts[0] + value_counts[1])
            round_ratio = int(round(ratio_0, 2)*100)
            if round_ratio > 95:
                round_ratio = 90

            print(f"{task.target_column} - ratio", round_ratio)
            percentiles = np.array(list(range(round_ratio - 20, min(round_ratio + 20, 95))))

            rows_not_dropped_indices = df_copy.index.tolist()
            capra_pred = pd.read_csv(os.path.join(NOMOGRAMS_PATH, "CAPRA", f"outer_split_{i}.csv"))["PREDICTED_PREOPERATIVE_PROSTATE_CANCER_DEATH_RISK"]
            pred = capra_pred.to_numpy()[rows_not_dropped_indices]

            p_values = []
            thresholds = []
            for percentile in percentiles:
                prediction = pred

                threshold_value = np.percentile(prediction, percentile)
                below_threshold = np.where(prediction < threshold_value)[0]
                above_threshold = np.where(prediction >= threshold_value)[0]

                final_group_indices = np.zeros_like(prediction, dtype=int)
                final_group_indices[above_threshold] = 1

                _, p_value = compare_survival(
                    y=TableViewer._get_structured_array(
                        event_indicator=df_copy[task.event_indicator_column],
                        event_time=df_copy[task.event_time_column]
                    ),
                    group_indicator=final_group_indices
                )
                p_values.append(p_value)
                thresholds.append(threshold_value)

            p_values = np.array(p_values)
            p_valus_neighbor_means = (p_values[:-2] + p_values[1:-1] + p_values[2:]) / 3.0

            print(f"{task.target_column} - p_value", p_values[np.argmin(p_valus_neighbor_means)])
            best_threshold = thresholds[np.argmin(p_valus_neighbor_means)]
            print(f"{task.target_column} - threshold", best_threshold)
            best_percentile = percentiles[np.argmin(p_valus_neighbor_means)]
            print(f"{task.target_column} - percentile", best_percentile)
