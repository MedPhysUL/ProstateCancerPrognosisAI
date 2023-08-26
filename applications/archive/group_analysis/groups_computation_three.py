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
    HOLDOUT_MASKS_PATH,
    ID,
    SEED,
    BCR_TASK,
    CRPC_TASK,
    PN_TASK,
    METASTASIS_TASK,
    HTX_TASK,
    DEATH_TASK
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
        tasks=[PN_TASK, BCR_TASK, METASTASIS_TASK, HTX_TASK, CRPC_TASK, DEATH_TASK],
        continuous_features=CLINICAL_CONTINUOUS_FEATURES + RADIOMICS,
        categorical_features=CLINICAL_CATEGORICAL_FEATURES
    )

    dataset = ProstateCancerDataset(table_dataset=table_dataset)

    masks = json.load(open(HOLDOUT_MASKS_PATH, "r"))

    dataset.update_masks(
        train_mask=masks[Mask.TRAIN],
        test_mask=masks[Mask.TEST],
        valid_mask=masks[Mask.VALID]
    )

    cont_features = [c.column for c in CLINICAL_CONTINUOUS_FEATURES]
    cat_features = [c.column for c in CLINICAL_CATEGORICAL_FEATURES]
    model = SequentialNet(
        sequence=[PN_TASK.name, BCR_TASK.name, METASTASIS_TASK.name, HTX_TASK.name, CRPC_TASK.name, DEATH_TASK.name],
        n_layers={PN_TASK.name: 2, BCR_TASK.name: 1, METASTASIS_TASK.name: 0, HTX_TASK.name: 2, CRPC_TASK.name: 1, DEATH_TASK.name: 2},
        n_neurons={PN_TASK.name: 10, BCR_TASK.name: 15, METASTASIS_TASK.name: 0, HTX_TASK.name: 10, CRPC_TASK.name: 10, DEATH_TASK.name: 10},
        features_columns={
            PN_TASK.name: cont_features + [c.column for c in PN_RADIOMICS] + cat_features,
            BCR_TASK.name: cont_features + [c.column for c in BCR_RADIOMICS] + cat_features,
            METASTASIS_TASK.name: cont_features + cat_features,
            HTX_TASK.name: cont_features + cat_features,
            CRPC_TASK.name: cont_features + cat_features,
            DEATH_TASK.name: cont_features + cat_features
        },
        dropout={
            PN_TASK.name: 0,
            BCR_TASK.name: 0,
            METASTASIS_TASK.name: 0,
            HTX_TASK.name: 0,
            CRPC_TASK.name: 0,
            DEATH_TASK.name: 0
        },
        bayesian=True,
        temperature={
            PN_TASK.name: 0.0001,
            BCR_TASK.name: 0.001,
            METASTASIS_TASK.name: 0.0001,
            HTX_TASK.name: 0.0001,
            CRPC_TASK.name: 0.001,
            DEATH_TASK.name: 0.001
        },
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    state = torch.load(
        os.path.join(
            EXPERIMENTS_PATH,
            r"HOLDOUT\DEATH(BayesSeqNet - Clinical data only)\best_model_checkpoint.pt"
        )
    )["model_state"]

    model.load_state_dict(state)
    model.fix_thresholds_to_optimal_values(dataset)
    model.fit_breslow_estimators(dataset)

    score = model.compute_score_on_dataset(dataset, dataset.train_mask, 100)
    print(score)
    score = model.compute_score_on_dataset(dataset, dataset.valid_mask, 100)
    print(score)
    score = model.compute_score_on_dataset(dataset, dataset.test_mask, 100)
    print(score)

    for task in [BCR_TASK, METASTASIS_TASK, HTX_TASK, CRPC_TASK, DEATH_TASK]:
        df = dataset.table_dataset.dataframe.iloc[list(range(250))]
        df_copy = df.copy()
        df_copy.dropna(subset=[task.event_indicator_column, task.event_time_column], inplace=True)
        value_counts = df[task.target_column].value_counts()
        ratio_0 = value_counts[0] / (value_counts[0] + value_counts[1])
        round_ratio = int(round(ratio_0, 2)*100)

        if round_ratio > 95:
            round_ratio = 90

        print("Ratio used", round_ratio)
        rows_not_dropped_indices = df_copy.index.tolist()
        pred = model.predict_on_dataset(dataset=dataset, mask=rows_not_dropped_indices, n_samples=100)

        low, intermediate, high = 5, round_ratio, 95
        low_percentiles = np.array(list(range(low, intermediate - 1)))
        high_percentiles = np.array(list(range(intermediate + 1, high)))
        percentiles_matrix = np.zeros((len(low_percentiles), len(high_percentiles), 2))
        p_values_matrix = np.zeros((len(low_percentiles), len(high_percentiles)))
        p_values_12 = np.zeros((len(low_percentiles), len(high_percentiles)))
        p_values_23 = np.zeros((len(low_percentiles), len(high_percentiles)))
        thresholds_value_matrix = np.zeros((len(low_percentiles), len(high_percentiles), 2))
        for p1 in low_percentiles:
            for p2 in high_percentiles:
                prediction = pred[task.name]
                prediction = prediction.cpu().detach().numpy()[:, 0]

                threshold_value1 = np.percentile(prediction, p1)
                threshold_value2 = np.percentile(prediction, p2)

                group1_indices = np.where(prediction < threshold_value1)[0]
                group2_indices = np.where((prediction >= threshold_value1) & (prediction < threshold_value2))[0]
                group3_indices = np.where(prediction >= threshold_value2)[0]

                group12_indices = np.zeros_like(prediction, dtype=int)
                group12_indices[group2_indices] = 1

                group23_indices = np.zeros_like(prediction, dtype=int)
                group23_indices[group3_indices] = 1

                idxs = group1_indices.tolist() + group2_indices.tolist()
                _, p_value12 = compare_survival(
                    y=TableViewer._get_structured_array(
                        event_indicator=df_copy[task.event_indicator_column].iloc[idxs],
                        event_time=df_copy[task.event_time_column].iloc[idxs]
                    ),
                    group_indicator=group12_indices[idxs]
                )

                idxs = group2_indices.tolist() + group3_indices.tolist()
                _, p_value23 = compare_survival(
                    y=TableViewer._get_structured_array(
                        event_indicator=df_copy[task.event_indicator_column].iloc[idxs],
                        event_time=df_copy[task.event_time_column].iloc[idxs]
                    ),
                    group_indicator=group23_indices[idxs]
                )

                percentiles_matrix[p1 - low, p2 - intermediate - 1] = [p1, p2]
                p_values_matrix[p1 - low, p2 - intermediate - 1] = np.mean([p_value12, p_value23])
                p_values_12[p1 - low, p2 - intermediate - 1] = p_value12
                p_values_23[p1 - low, p2 - intermediate - 1] = p_value23
                thresholds_value_matrix[p1 - low, p2 - intermediate - 1] = [threshold_value1, threshold_value2]

        idx_min = np.unravel_index(np.argmin(p_values_matrix), p_values_matrix.shape)
        best_p_value = p_values_matrix[idx_min]
        print(f"{task.target_column} - p_value mean", best_p_value)
        best_p_value = p_values_12[idx_min]
        print(f"{task.target_column} - p_value 1-2", best_p_value)
        best_p_value = p_values_23[idx_min]
        print(f"{task.target_column} - p_value 2-3", best_p_value)
        best_percentile = percentiles_matrix[idx_min]
        print(f"{task.target_column} - percentile", best_percentile)
        best_threshold = thresholds_value_matrix[idx_min]
        print(f"{task.target_column} - threshold", best_threshold)
