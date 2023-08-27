import env_apps

from copy import deepcopy

import json
import os
from typing import List, Optional, Tuple

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


def plot_roc_curve(
        y_true: np.ndarray,
        y_pred: Tuple[np.ndarray, np.ndarray],
        show: bool,
        path_to_save_folder: Optional[str] = None,
        **kwargs
) -> None:
    fig, arr = plt.subplots(figsize=(8, 6))
    models = ["MSKCC", "BSN"]
    AUC = ["$0.70\pm0.07$", "$0.71\pm0.04$"]
    arr.plot([1, 0], [1, 0], color="silver", linestyle="--", lw=2)
    for i, p in enumerate(y_pred):
        fpr, tpr, threshold = roc_curve(
            y_true,
            p,
            pos_label=kwargs.get("pos_label", None),
            sample_weight=kwargs.get("sample_weight", None),
            drop_intermediate=kwargs.get("drop_intermediate", True)
        )
        arr.plot(fpr, tpr, color=[c for c in LightColor][i], lw=3, label=f"{models[i]} (AUC = {AUC[i]})")

    arr.set_xlabel(kwargs.get("xlabel", f"False positive rate"), fontsize=18)
    arr.set_ylabel(kwargs.get("ylabel", f"True positive rate"), fontsize=18)
    arr.minorticks_on()
    arr.tick_params(axis="both", direction='in', color="k", which="major", labelsize=16, length=6)
    arr.tick_params(axis="both", direction='in', color="k", which="minor", labelsize=16, length=3)
    arr.legend(fontsize=16)
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
        models: list,
        datasets: List[ProstateCancerDataset],
        tasks,
        thresholds: List[List[float]],
        path_to_folder: Optional[str] = None,
        show: Optional[bool] = False,
) -> None:

    light_colors = [c for c in LightColor]
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
            pred = models[i].predict_on_dataset(dataset=dataset, mask=rows_not_dropped_indices, n_samples=100)

            prediction = pred[task.name]
            prediction = prediction.cpu().detach().numpy()[:, 0]

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

        if p_value < 0.0001:
            annotation = f"p-value$ < 0.0001$"
        else:
            annotation = f"p-value$ = {p_value:.4f}$"

        axes.annotate(
            annotation, xy=(0.15, 0.05), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 5), ha='center', fontsize=16
        )
        axes.annotate(
            f"HR$ = {hazard_ratio:.2f}$", xy=(0.1648, 0.12), xycoords="axes fraction", textcoords="offset points",
            xytext=(0, 5), ha='center', fontsize=16
        )

        terminate_figure(path_to_save=os.path.join(path_to_folder, f"{task.target_column}.png"), show=show, fig=fig)


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

    cont_features = [c.column for c in CLINICAL_CONTINUOUS_FEATURES]
    cat_features = [c.column for c in CLINICAL_CATEGORICAL_FEATURES]

    layers = [
        {
            PN_TASK.name: 3,
            BCR_TASK.name: 3,
            METASTASIS_TASK.name: 2,
            HTX_TASK.name: 2,
            CRPC_TASK.name: 2
        },
        {
            PN_TASK.name: 3,
            BCR_TASK.name: 2,
            METASTASIS_TASK.name: 2,
            HTX_TASK.name: 2,
            CRPC_TASK.name: 1
        },
        {
            PN_TASK.name: 3,
            BCR_TASK.name: 1,
            METASTASIS_TASK.name: 2,
            HTX_TASK.name: 2,
            CRPC_TASK.name: 2
        },
        {
            PN_TASK.name: 3,
            BCR_TASK.name: 3,
            METASTASIS_TASK.name: 1,
            HTX_TASK.name: 2,
            CRPC_TASK.name: 2
        },
        {
            PN_TASK.name: 3,
            BCR_TASK.name: 1,
            METASTASIS_TASK.name: 2,
            HTX_TASK.name: 2,
            CRPC_TASK.name: 2
        }
    ]

    neurons = [
        {
            PN_TASK.name: 20,
            BCR_TASK.name: 20,
            METASTASIS_TASK.name: 20,
            HTX_TASK.name: 20,
            CRPC_TASK.name: 10
        },
        {
            PN_TASK.name: 15,
            BCR_TASK.name: 20,
            METASTASIS_TASK.name: 20,
            HTX_TASK.name: 15,
            CRPC_TASK.name: 5
        },
        {
            PN_TASK.name: 15,
            BCR_TASK.name: 5,
            METASTASIS_TASK.name: 15,
            HTX_TASK.name: 20,
            CRPC_TASK.name: 15
        },
        {
            PN_TASK.name: 20,
            BCR_TASK.name: 20,
            METASTASIS_TASK.name: 20,
            HTX_TASK.name: 20,
            CRPC_TASK.name: 20
        },
        {
            PN_TASK.name: 20,
            BCR_TASK.name: 20,
            METASTASIS_TASK.name: 15,
            HTX_TASK.name: 15,
            CRPC_TASK.name: 5
        }
    ]

    mskcc_preds, y_bayes_seq_net_preds, trues = [], [], []
    models = []
    datasets = []
    for i in range(5):
        masks = json.load(open(MASKS_PATH, "r"))

        dataset.update_masks(
            train_mask=masks[f"{i}"][Mask.TRAIN],
            test_mask=masks[f"{i}"][Mask.TEST],
            valid_mask=masks[f"{i}"][Mask.VALID]
        )

        model = SequentialNet(
            sequence=[PN_TASK.name, BCR_TASK.name, METASTASIS_TASK.name, HTX_TASK.name, CRPC_TASK.name],
            n_layers={
                PN_TASK.name: layers[i][PN_TASK.name],
                BCR_TASK.name: layers[i][BCR_TASK.name],
                METASTASIS_TASK.name: layers[i][METASTASIS_TASK.name],
                HTX_TASK.name: layers[i][HTX_TASK.name],
                CRPC_TASK.name: layers[i][CRPC_TASK.name]
            },
            n_neurons={
                PN_TASK.name: neurons[i][PN_TASK.name],
                BCR_TASK.name: neurons[i][BCR_TASK.name],
                METASTASIS_TASK.name: neurons[i][METASTASIS_TASK.name],
                HTX_TASK.name: neurons[i][HTX_TASK.name],
                CRPC_TASK.name: neurons[i][CRPC_TASK.name]
            },
            features_columns={
                PN_TASK.name: cont_features + [c.column for c in PN_RADIOMICS] + cat_features,
                BCR_TASK.name: cont_features + [c.column for c in BCR_RADIOMICS] + cat_features,
                METASTASIS_TASK.name: cont_features + cat_features,
                HTX_TASK.name: cont_features + cat_features,
                CRPC_TASK.name: cont_features + cat_features
            },
            dropout={
                PN_TASK.name: 0,
                BCR_TASK.name: 0,
                METASTASIS_TASK.name: 0,
                HTX_TASK.name: 0,
                CRPC_TASK.name: 0
            },
            bayesian=True,
            temperature={
                PN_TASK.name: 0.0001,
                BCR_TASK.name: 0.001,
                METASTASIS_TASK.name: 0.0001,
                HTX_TASK.name: 0.0001,
                CRPC_TASK.name: 0.001
            },
            device=torch.device("cuda"),
            seed=SEED
        ).build(dataset)

        state = torch.load(
            os.path.join(
                EXPERIMENTS_PATH,
                fr"CRPC(BayesSeqNet - Clinical data only)\outer_splits\split_{i}\best_models\outer_split\best_model.pt"
            )
        )
        state_copy = state.copy()

        model.load_state_dict(state_copy)
        model.fix_thresholds_to_optimal_values(dataset)
        model.fit_breslow_estimators(dataset)

        prediction = model.predict_on_dataset(dataset, dataset.test_mask, n_samples=100)

        mskcc_pred = pd.read_csv(os.path.join(NOMOGRAMS_PATH, "MSKCC", "final_set.csv"))["PREDICTED_LYMPH_NODE_INVOLVEMENT"]
        mskcc_pred = mskcc_pred.to_numpy()[dataset.test_mask]
        mskcc_preds.append(mskcc_pred)

        y_bayes_seq_net_pred = prediction[PN_TASK.name].cpu().numpy()[:, 0]
        y_bayes_seq_net_preds.append(y_bayes_seq_net_pred)

        true = dataset.table_dataset.y[PN_TASK.name][dataset.test_mask]
        trues.append(true)

        models.append(model)
        datasets.append(deepcopy(dataset))

    mskcc_preds = np.concatenate(mskcc_preds)
    y_bayes_seq_net_preds = np.concatenate(y_bayes_seq_net_preds)
    trues = np.concatenate(trues)

    plot_roc_curve(
        trues,
        (mskcc_preds, y_bayes_seq_net_preds),
        False,
        path_to_save_folder="local_data"
    )

    _plot_stratified_kaplan_meier_curve(
        models,
        datasets,
        [BCR_TASK, METASTASIS_TASK, HTX_TASK, CRPC_TASK],
        [
            [0.2910183846950531, -0.06488897085189821, 0.5745063066482544, 1.2015027379989622],
            [0.023626287281513152, -0.023762829005718158, 0.36310250759124746, 0.012039108388125892],
            [0.05972794573754071, 0.269909538924694, 0.6318077754974366, 0.011043768376111984],
            [-0.0028230430372059344, 0.28542009696364407, -0.08477262414991873, 0.02855702117085457],
            [0.4450987964868545, 1.2470833015441898, 0.35523607581853867, 0.007185159367509187]
        ],
        path_to_folder="local_data"
    )
