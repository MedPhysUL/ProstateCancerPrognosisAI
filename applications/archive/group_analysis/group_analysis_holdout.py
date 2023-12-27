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


def plot_roc_curve(
        y_true: List[int],
        y_pred: List[float],
        show: bool,
        path_to_save_folder: Optional[str] = None,
        **kwargs
) -> None:
    fig, arr = plt.subplots(figsize=(8, 6))
    # AUC = "0.663"
    arr.plot([1, 0], [1, 0], color="silver", linestyle="--", lw=2)
    fpr, tpr, threshold = roc_curve(
        y_true,
        y_pred,
        pos_label=kwargs.get("pos_label", None),
        sample_weight=kwargs.get("sample_weight", None),
        drop_intermediate=kwargs.get("drop_intermediate", True)
    )
    arr.plot(fpr, tpr, color="k", lw=3)

    arr.set_xlabel(kwargs.get("xlabel", f"False positive rate"), fontsize=18)
    arr.set_ylabel(kwargs.get("ylabel", f"True positive rate"), fontsize=18)
    arr.minorticks_on()
    arr.tick_params(axis="both", direction='in', color="k", which="major", labelsize=16, length=6)
    arr.tick_params(axis="both", direction='in', color="k", which="minor", labelsize=16, length=3)
    arr.set_xlim(0, 1)
    arr.set_ylim(-0.02, 1.02)
    arr.grid(False)

    if path_to_save_folder is not None:
        path = os.path.join(
            path_to_save_folder,
            f"{kwargs.get('filename', 'roc_curve.png')}"
        )
    else:
        path = None
    terminate_figure(fig=fig, show=show, path_to_save=path, **kwargs)


def _plot_stratified_kaplan_meier_curve(
        model,
        tasks,
        thresholds,
        path_to_folder: Optional[str] = None,
        show: Optional[bool] = False,
) -> None:

    light_colors = [c for c in LightColor]
    c_index = {"BCR": "0.59", "METASTASIS": "0.75", "HTX": "0.70", "CRPC": "0.66"}
    for idx, task in enumerate(tasks):
        fig, axes = plt.subplots(figsize=(8, 6))
        df = dataset.table_dataset.dataframe.iloc[dataset.test_mask]
        df_copy = df.copy()
        df_copy.dropna(subset=[task.event_indicator_column, task.event_time_column], inplace=True)
        rows_not_dropped_indices = df_copy.index.tolist()
        pred = model.predict_on_dataset(dataset=dataset, mask=rows_not_dropped_indices, n_samples=100)

        prediction = pred[task.name]
        prediction = prediction.cpu().detach().numpy()[:, 0]

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
        hazard_ratio = (observed[1]/expected[1])/(observed[0]/expected[0])

        if p_value < 0.0001:
            annotation = f"p-value$ < 0.0001$"
        else:
            annotation = f"p-value$ = {p_value:.4f}$"

        axes.annotate(
            annotation, xy=(0.16, 0.05), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 5), ha='center', fontsize=20
        )
        axes.annotate(
            f"HR$ = {hazard_ratio:.2f}$", xy=(0.1748, 0.12), xycoords="axes fraction", textcoords="offset points",
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

    cont_features = [c.column for c in CLINICAL_CONTINUOUS_FEATURES]
    cat_features = [c.column for c in CLINICAL_CATEGORICAL_FEATURES]
    model = SequentialNet(
        sequence=[PN_TASK.name, BCR_TASK.name, METASTASIS_TASK.name, HTX_TASK.name, CRPC_TASK.name, DEATH_TASK.name],
        n_layers={
            PN_TASK.name: 2, BCR_TASK.name: 1, METASTASIS_TASK.name: 0, HTX_TASK.name: 2, CRPC_TASK.name: 1,
            DEATH_TASK.name: 2
        },
        n_neurons={
            PN_TASK.name: 10, BCR_TASK.name: 15, METASTASIS_TASK.name: 0, HTX_TASK.name: 10, CRPC_TASK.name: 10,
            DEATH_TASK.name: 10
        },
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

    prediction = model.predict_on_dataset(dataset, dataset.test_mask, n_samples=100)

    mskcc_pred = pd.read_csv(os.path.join(NOMOGRAMS_PATH, "MSKCC", "final_set.csv"))["PREDICTED_LYMPH_NODE_INVOLVEMENT"]
    mskcc_pred = mskcc_pred.to_numpy()[dataset.test_mask]
    y_bayes_seq_net_pred = prediction[PN_TASK.name].cpu().numpy()[:, 0]
    true = dataset.table_dataset.y[PN_TASK.name][dataset.test_mask]

    plot_roc_curve(true, y_bayes_seq_net_pred, False, path_to_save_folder="local_data")

    _plot_stratified_kaplan_meier_curve(
        model,
        [BCR_TASK, METASTASIS_TASK, HTX_TASK, CRPC_TASK],
        [0.038691001236438725, 0.2502322018146515, 0.08327216058969497, 0.433767817914486],
        path_to_folder="local_data"
    )
