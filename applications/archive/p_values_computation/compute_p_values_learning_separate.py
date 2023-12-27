import numpy as np

import env_apps

import json
import os

import pandas as pd

from constants import (
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    FINAL_BAYES_TABLE_PATH,
    HOLDOUT_MASKS_PATH,
    MASKS_PATH,
    ID,
    BCR_TASK,
    CRPC_TASK,
    PN_TASK,
    NOMOGRAMS_PATH,
    METASTASIS_TASK,
    HTX_TASK,
    DEATH_TASK,
    TABLE_TASKS
)
from src.data.datasets import Feature
from src.data.transforms import Normalization
from src.data.processing.sampling import Mask
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.evaluation import PredictionComparator
from src.metrics.single_task import (
    BinaryBalancedAccuracy,
    ConcordanceIndexCensored,
    ConcordanceIndexIPCW,
    CumulativeDynamicAUC
)


if __name__ == '__main__':
    df = pd.read_csv(FINAL_BAYES_TABLE_PATH)

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=[PN_TASK, BCR_TASK, METASTASIS_TASK, HTX_TASK, CRPC_TASK, DEATH_TASK],
        continuous_features=CLINICAL_CONTINUOUS_FEATURES,
        categorical_features=CLINICAL_CATEGORICAL_FEATURES
    )

    dataset = ProstateCancerDataset(table_dataset=table_dataset)

    masks = json.load(open(MASKS_PATH, "r"))
    test_sets_mask = []
    all_c_index = {t.name: [] for t in TABLE_TASKS}
    all_auc = {PN_TASK.name: []}
    all_c_ipcw = {t.name: [] for t in TABLE_TASKS}
    all_cda = {t.name: [] for t in TABLE_TASKS}
    all_bba = {t.name: [] for t in TABLE_TASKS}
    for i in range(5):
        dataset.update_masks(
            train_mask=masks[str(i)][Mask.TRAIN],
            test_mask=masks[str(i)][Mask.TEST],
            valid_mask=masks[str(i)][Mask.VALID]
        )
        test_sets_mask += masks[str(i)][Mask.TEST]

        net_pred = json.load(open(rf"local_data\preds\split_{i}\BayesSeqNet.json", "r"))

        capra_pred = pd.read_csv(os.path.join(NOMOGRAMS_PATH, "CAPRA", f"outer_split_{i}.csv"))
        mskcc_pred = pd.read_csv(os.path.join(NOMOGRAMS_PATH, "MSKCC", f"outer_split_{i}.csv"))

        baseline_pred = {
            PN_TASK.name: mskcc_pred["PREDICTED_LYMPH_NODE_INVOLVEMENT"].to_numpy()[dataset.test_mask],
            BCR_TASK.name: mskcc_pred["PREDICTED_PREOPERATIVE_BCR_RISK"].to_numpy()[dataset.test_mask],
            METASTASIS_TASK.name: capra_pred["PREDICTED_METASTASIS_RISK"].to_numpy()[dataset.test_mask],
            HTX_TASK.name: capra_pred["PREDICTED_HORMONOTHERAPY_RISK"].to_numpy()[dataset.test_mask],
            CRPC_TASK.name: capra_pred["PREDICTED_CASTRATE_RESISTANT_RISK"].to_numpy()[dataset.test_mask],
            DEATH_TASK.name: capra_pred["PREDICTED_PREOPERATIVE_PROSTATE_CANCER_DEATH_RISK"].to_numpy()[
                dataset.test_mask]
        }

        if i == 1:
            _tasks = [PN_TASK, BCR_TASK, METASTASIS_TASK, HTX_TASK, CRPC_TASK]
        else:
            _tasks = [PN_TASK, BCR_TASK, METASTASIS_TASK, HTX_TASK, CRPC_TASK, DEATH_TASK]

        comparator = PredictionComparator(
            pred_1=net_pred,
            pred_2=baseline_pred,
            ground_truth=dataset.table_dataset[dataset.test_mask].y,
            tasks=_tasks
        )

        p_values = comparator.compute_c_index_p_value()
        for k, v in p_values.items():
            all_c_index[k].append(v)

        p_values = comparator.compute_auc_p_value()
        for k, v in p_values.items():
            all_auc[k].append(v)

        metrics = {}
        for task in _tasks[1:]:
            c_index_ipcw = ConcordanceIndexIPCW()
            c_index_ipcw.update_censoring_distribution(dataset.table_dataset[dataset.train_mask].y[task.name])
            metrics[task.name] = c_index_ipcw
        p_values = comparator.compute_any_metric_p_value(metrics)
        for k, v in p_values.items():
            all_c_ipcw[k].append(v)

        metrics = {}
        for task in _tasks[1:]:
            c_index_ipcw = CumulativeDynamicAUC()
            c_index_ipcw.update_censoring_distribution(dataset.table_dataset[dataset.train_mask].y[task.name])
            metrics[task.name] = c_index_ipcw
        p_values = comparator.compute_any_metric_p_value(metrics)
        for k, v in p_values.items():
            all_cda[k].append(v)

        bba = BinaryBalancedAccuracy()
        bba._threshold = 0.58
        bba.update_scaling_factor(dataset.table_dataset[dataset.train_mask].y[PN_TASK.name])
        p_values = comparator.compute_any_metric_p_value(bba)

        for k, v in p_values.items():
            all_bba[k].append(v)

    print("C-index", all_c_index)
    print("AUC", all_auc)
    print("CIPCW", all_c_ipcw)
    print("CDA", all_cda)
    print("BA", all_bba)

    print("C-index", {k: np.median(v) for k, v in all_c_index.items()})
    print("AUC", {k: np.median(v) for k, v in all_auc.items()})
    print("CIPCW", {k: np.median(v) for k, v in all_c_ipcw.items()})
    print("CDA", {k: np.median(v) for k, v in all_cda.items()})
    print("BA", {k: np.median(v) for k, v in all_bba.items()})
