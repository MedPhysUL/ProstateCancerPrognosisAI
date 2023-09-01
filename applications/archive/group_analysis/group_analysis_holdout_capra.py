import env_apps

import json
import os
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
    DEATH_TASK,
    NOMOGRAMS_PATH
)
from src.data.datasets import Feature
from src.data.transforms import Normalization
from src.data.processing.sampling import Mask
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.models.torch.prediction import SequentialNet
from src.tools.plot import add_details_to_kaplan_meier_curve, terminate_figure
from src.visualization import TableViewer
from src.visualization.color import LightColor

from sklearn.metrics import roc_curve
from sksurv.compare import compare_survival

from src.visualization.tools import survival_table_from_events, add_at_risk_counts


def _plot_stratified_kaplan_meier_curve(
        pred,
        tasks,
        thresholds,
        path_to_folder: Optional[str] = None,
        show: Optional[bool] = False,
) -> None:

    light_colors = [c for c in LightColor]
    c_index = {"DEATH": "0.87"}
    for idx, task in enumerate(tasks):
        fig, axes = plt.subplots(figsize=(8, 6))
        df = dataset.table_dataset.dataframe.iloc[dataset.test_mask]
        df_copy = df.copy()
        df_copy.dropna(subset=[task.event_indicator_column, task.event_time_column], inplace=True)
        rows_not_dropped_indices = df_copy.index.tolist()
        prediction = pred[rows_not_dropped_indices]

        threshold_value = thresholds[idx]
        below_threshold = np.where(prediction < threshold_value)[0]
        above_threshold = np.where(prediction >= threshold_value)[0]

        threshs = [below_threshold, above_threshold]

        survival_tables, colors = [], []
        groups = ["Low-risk group", "High-risk group"]
        for idx, threshold in enumerate(threshs):
            subset = df_copy.iloc[threshold]
            event, time = subset[task.event_indicator_column].astype(bool), subset[task.event_time_column]
            TableViewer._build_kaplan_meier_curve(event, time, axes, light_colors[idx], groups[idx], 0.68, 120)

            survival_tables.append(survival_table_from_events(time, event))
            colors.append(light_colors[idx])

        add_details_to_kaplan_meier_curve(axes, True)
        add_at_risk_counts(survival_tables=survival_tables, colors=colors, axes=axes, figure=fig)

        final_group_indices = np.zeros_like(prediction, dtype=int)
        final_group_indices[above_threshold] = 1

        _, p_value, stats, covariance = compare_survival(
            y=TableViewer._get_structured_array(
                event_indicator=df_copy[task.event_indicator_column],
                event_time=df_copy[task.event_time_column]
            ),
            group_indicator=final_group_indices,
            return_stats=True
        )

        observed, expected = stats["observed"].tolist(), stats["expected"].tolist()

        if p_value < 0.0001:
            annotation = f"p-value$ < 0.0001$"
        else:
            annotation = f"p-value$ = {p_value:.4f}$"

        axes.annotate(
            annotation, xy=(0.16, 0.05), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 5), ha='center', fontsize=20
        )
        axes.annotate(
            f"HR$ = n/a$", xy=(0.17, 0.12), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 5), ha='center', fontsize=20
        )
        axes.annotate(
            f"C-index$ = {c_index[task.target_column]}$", xy=(0.138, 0.19), xycoords="axes fraction",
            textcoords="offset points", xytext=(0, 5), ha='center', fontsize=20
        )

        terminate_figure(path_to_save=os.path.join(path_to_folder, f"{task.target_column}.png"), show=show, fig=fig)


if __name__ == '__main__':
    matplotlib.rc('axes', edgecolor='k')
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

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

    capra_pred = pd.read_csv(os.path.join(NOMOGRAMS_PATH, "CAPRA", "final_set.csv"))["PREDICTED_PREOPERATIVE_PROSTATE_CANCER_DEATH_RISK"]

    _plot_stratified_kaplan_meier_curve(
        capra_pred,
        [DEATH_TASK],
        [0.4209128210524002],
        path_to_folder="local_data"
    )
