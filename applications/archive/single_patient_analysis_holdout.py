import env_apps

import json
import os
from typing import Optional

import matplotlib.pyplot as plt
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
from src.models.torch.prediction import SequentialNet
from src.tools.plot import add_details_to_kaplan_meier_curve, terminate_figure
from src.visualization.color import LightColor

import numpy as np


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

    print(dataset[269])
    names = {
        "BCR": "BCR-FS", "METASTASIS": "MFS", "HTX": "dADT-FS", "CRPC": "CRPC-FS", "DEATH": "PCSS"
    }

    def get_survival_probability(task, time):
        survival_function = task.breslow_estimator.get_survival_function(pred[task.name].cpu())[0]
        return survival_function(time)

    def plot_survival_function(
            pred_mean: dict,
            pred_lower: dict,
            pred_upper: dict,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> None:
        fig, arr = plt.subplots(figsize=(8, 6))
        light_colors = [c for c in LightColor]
        survs_tasks = dataset.tasks.survival_analysis_tasks.tasks
        arr.axvline(x=24, lw=2, ls='--', color="silver")
        arr.axvline(x=60, lw=2, ls='--', color="silver")
        for idx, task in enumerate(survs_tasks):
            mean_survs = task.breslow_estimator.get_survival_function(np.array([pred_mean[task.name]]))
            lower_survs = task.breslow_estimator.get_survival_function(np.array([pred_lower[task.name]]))
            upper_survs = task.breslow_estimator.get_survival_function(np.array([pred_upper[task.name]]))
            for mean_surv, lower_surv, upper_surv in zip(mean_survs, lower_survs, upper_survs):
                name = task.target_column
                name = names[name] if name in names else name

                time_to_plot = np.concatenate(([0], mean_surv.x))
                survival_probability_to_plot = np.concatenate(([1], mean_surv(mean_surv.x)))

                arr.step(
                    time_to_plot, survival_probability_to_plot,
                    where="post", color=light_colors[idx], label=name, lw=3
                )
                arr.fill_between(
                    mean_surv.x, lower_surv(lower_surv.x), upper_surv(upper_surv.x),
                    alpha=0.3, step="post", color=light_colors[idx]
                )

        add_details_to_kaplan_meier_curve(arr)
        if path_to_save_folder is not None:
            path = os.path.join(
                path_to_save_folder,
                f"{kwargs.get('filename', 'survival_function.png')}"
            )
        else:
            path = None

        terminate_figure(fig=fig, show=show, path_to_save=path, **kwargs)

    pred_pn = []
    pred_bcr = []
    pred_mts = []
    pred_htx = []
    pred_crpc = []
    pred_death = []
    surv_bcr_2_years = []
    surv_bcr_5_years = []
    surv_mts_2_years = []
    surv_mts_5_years = []
    surv_htx_2_years = []
    surv_htx_5_years = []
    surv_crpc_2_years = []
    surv_crpc_5_years = []
    surv_death_2_years = []
    surv_death_5_years = []
    for i in range(100):
        pred = model.predict_on_dataset(dataset=dataset, mask=[269], n_samples=1)
        pred_pn.append(pred[PN_TASK.name].item())
        pred_bcr.append(pred[BCR_TASK.name].item())
        pred_mts.append(pred[METASTASIS_TASK.name].item())
        pred_htx.append(pred[HTX_TASK.name].item())
        pred_crpc.append(pred[CRPC_TASK.name].item())
        pred_death.append(pred[DEATH_TASK.name].item())

        surv_bcr_2_years.append(get_survival_probability(BCR_TASK, 24))
        surv_bcr_5_years.append(get_survival_probability(BCR_TASK, 60))
        surv_mts_2_years.append(get_survival_probability(METASTASIS_TASK, 24))
        surv_mts_5_years.append(get_survival_probability(METASTASIS_TASK, 60))
        surv_htx_2_years.append(get_survival_probability(HTX_TASK, 24))
        surv_htx_5_years.append(get_survival_probability(HTX_TASK, 60))
        surv_crpc_2_years.append(get_survival_probability(CRPC_TASK, 24))
        surv_crpc_5_years.append(get_survival_probability(CRPC_TASK, 60))
        surv_death_2_years.append(get_survival_probability(DEATH_TASK, 24))
        surv_death_5_years.append(get_survival_probability(DEATH_TASK, 60))

    print(f"PN", np.mean(pred_pn), np.std(pred_pn))
    print(f"BCR", np.mean(pred_bcr), np.std(pred_bcr))
    print(f"MTS", np.mean(pred_mts), np.std(pred_mts))
    print(f"HTX", np.mean(pred_htx), np.std(pred_htx))
    print(f"CRPC", np.mean(pred_crpc), np.std(pred_crpc))
    print(f"DEATH", np.mean(pred_death), np.std(pred_death))

    print(f"BCR 2 years", np.mean(surv_bcr_2_years), np.std(surv_bcr_2_years))
    print(f"BCR 5 years", np.mean(surv_bcr_5_years), np.std(surv_bcr_5_years))
    print(f"MTS 2 years", np.mean(surv_mts_2_years), np.std(surv_mts_2_years))
    print(f"MTS 5 years", np.mean(surv_mts_5_years), np.std(surv_mts_5_years))
    print(f"HTX 2 years", np.mean(surv_htx_2_years), np.std(surv_htx_2_years))
    print(f"HTX 5 years", np.mean(surv_htx_5_years), np.std(surv_htx_5_years))
    print(f"CRPC 2 years", np.mean(surv_crpc_2_years), np.std(surv_crpc_2_years))
    print(f"CRPC 5 years", np.mean(surv_crpc_5_years), np.std(surv_crpc_5_years))
    print(f"DEATH 2 years", np.mean(surv_death_2_years), np.std(surv_death_2_years))
    print(f"DEATH 5 years", np.mean(surv_death_5_years), np.std(surv_death_5_years))

    plot_survival_function(
        pred_mean={
            BCR_TASK.name: np.mean(pred_bcr),
            METASTASIS_TASK.name: np.mean(pred_mts),
            HTX_TASK.name: np.mean(pred_htx),
            CRPC_TASK.name: np.mean(pred_crpc),
            DEATH_TASK.name: np.mean(pred_death),
        },
        pred_lower={
            BCR_TASK.name: np.mean(pred_bcr) - 2 * np.std(pred_bcr),
            METASTASIS_TASK.name: np.mean(pred_mts) - 2 * np.std(pred_mts),
            HTX_TASK.name: np.mean(pred_htx) - 2 * np.std(pred_htx),
            CRPC_TASK.name: np.mean(pred_crpc) - 2 * np.std(pred_crpc),
            DEATH_TASK.name: np.mean(pred_death) - 2 * np.std(pred_death)
        },
        pred_upper={
            BCR_TASK.name: np.mean(pred_bcr) + 2 * np.std(pred_bcr),
            METASTASIS_TASK.name: np.mean(pred_mts) + 2 * np.std(pred_mts),
            HTX_TASK.name: np.mean(pred_htx) + 2 * np.std(pred_htx),
            CRPC_TASK.name: np.mean(pred_crpc) + 2 * np.std(pred_crpc),
            DEATH_TASK.name: np.mean(pred_death) + 2 * np.std(pred_death)
        },
        show=True,
        path_to_save_folder="local_data"
    )
