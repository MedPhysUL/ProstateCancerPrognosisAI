import env_apps

from copy import deepcopy

import json
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import torch

from constants import (
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    EXPERIMENTS_PATH,
    FINAL_BAYES_TABLE_PATH,
    HOLDOUT_MASKS_PATH,
    MASKS_PATH,
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
        preds: list,
        datasets: List[ProstateCancerDataset],
        tasks,
        thresholds: List[List[float]],
        path_to_folder: Optional[str] = None,
        show: Optional[bool] = False,
) -> None:

    light_colors = [c for c in LightColor]
    c_index = {"DEATH": "0.82\pm0.02"}
    for idx, task in enumerate(tasks):
        fig, axes = plt.subplots(figsize=(8, 6))

        below_thresholds = []
        above_thresholds = []
        dfs = []
        for i in range(5):
            dataset = datasets[i]
            df = dataset.table_dataset.dataframe.iloc[dataset.test_mask]
            df_copy = df.copy()
            df_copy.dropna(subset=[task.event_indicator_column, task.event_time_column], inplace=True)
            rows_not_dropped_indices = df_copy.index.tolist()
            prediction = preds[i][rows_not_dropped_indices]

            threshold_value = thresholds[i][idx]
            below_threshold = np.where(prediction < threshold_value)[0]
            above_threshold = np.where(prediction >= threshold_value)[0]

            below_thresholds.append(below_threshold)
            above_thresholds.append(above_threshold)
            dfs.append(df_copy)

        survival_tables, colors = [], []
        groups = ["Low-risk group", "High-risk group"]

        subsets = pd.DataFrame(columns=["event_indicator", "event_time", "groups"])
        for df, below_threshold, above_threshold in zip(dfs, below_thresholds, above_thresholds):
            final_group_indices = np.zeros(len(df), dtype=int)
            final_group_indices[above_threshold] = 1
            split = pd.DataFrame(
                data=np.array(
                    [df[task.event_indicator_column], df[task.event_time_column], final_group_indices]
                ).transpose(),
                columns=["event_indicator", "event_time", "groups"]
            )
            subsets = pd.concat([subsets, split], ignore_index=True)

        for n in range(2):
            TableViewer._build_kaplan_meier_curve(
                subsets[subsets["groups"] == n]["event_indicator"].astype(bool),
                subsets[subsets["groups"] == n]["event_time"], axes, light_colors[n], groups[n], 0.68
            )

            survival_tables.append(
                survival_table_from_events(
                    subsets[subsets["groups"] == n]["event_time"],
                    subsets[subsets["groups"] == n]["event_indicator"]
                )
            )

            colors.append(light_colors[n])

        add_details_to_kaplan_meier_curve(axes, True)
        add_at_risk_counts(survival_tables=survival_tables, colors=colors, axes=axes, figure=fig)

        _, p_value, stats, covariance = compare_survival(
            y=TableViewer._get_structured_array(
                event_indicator=subsets["event_indicator"],
                event_time=subsets["event_time"]
            ),
            group_indicator=subsets["groups"],
            return_stats=True
        )

        observed, expected = stats["observed"].tolist(), stats["expected"].tolist()
        hazard_ratio = (observed[1]/expected[1])/(observed[0]/expected[0])

        se = np.sqrt(1/observed[1] + 1/observed[0])
        log_hr = np.log(hazard_ratio)
        low_ci, high_ci = np.exp(log_hr - 1.96 * se), np.exp(log_hr + 1.96 * se)

        if p_value < 0.0001:
            annotation = f"p-value$ < 0.0001$"
        else:
            annotation = f"p-value$ = {p_value:.4f}$"

        axes.annotate(
            annotation, xy=(0.162, 0.05), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 5), ha='center', fontsize=20
        )
        axes.annotate(
            f"HR$ = {hazard_ratio:.2f}$ $(95$% CI; {low_ci:.2f}-{high_ci:.2f}$)$",
            xy=(0.349, 0.12), xycoords="axes fraction", textcoords="offset points", xytext=(0, 5), ha='center',
            fontsize=20
        )
        axes.annotate(
            f"C-index$ = {c_index[task.target_column]}$", xy=(0.198, 0.19), xycoords="axes fraction",
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
        tasks=[PN_TASK, BCR_TASK, METASTASIS_TASK, HTX_TASK, CRPC_TASK],
        continuous_features=CLINICAL_CONTINUOUS_FEATURES + RADIOMICS,
        categorical_features=CLINICAL_CATEGORICAL_FEATURES
    )

    dataset = ProstateCancerDataset(table_dataset=table_dataset)

    cont_features = [c.column for c in CLINICAL_CONTINUOUS_FEATURES]
    cat_features = [c.column for c in CLINICAL_CATEGORICAL_FEATURES]

    capra_preds = []
    datasets = []
    for i in range(5):
        masks = json.load(open(MASKS_PATH, "r"))

        dataset.update_masks(
            train_mask=masks[f"{i}"][Mask.TRAIN],
            test_mask=masks[f"{i}"][Mask.TEST],
            valid_mask=masks[f"{i}"][Mask.VALID]
        )

        capra_pred = pd.read_csv(os.path.join(NOMOGRAMS_PATH, "CAPRA", "final_set.csv"))["PREDICTED_PREOPERATIVE_PROSTATE_CANCER_DEATH_RISK"]
        capra_preds.append(capra_pred[dataset.test_mask])
        datasets.append(deepcopy(dataset))

    _plot_stratified_kaplan_meier_curve(
        capra_preds,
        datasets,
        [DEATH_TASK],
        [
            [0.7898446984177432],
            [0.4631912159491422],
            [0.5220771848607764],
            [0.3885712303426683],
            [0.07802921723869283]
        ],
        path_to_folder="local_data"
    )
