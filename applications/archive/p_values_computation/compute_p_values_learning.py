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
    bayes_seq_net_pred = {
        PN_TASK.name: [],
        BCR_TASK.name: [],
        METASTASIS_TASK.name: [],
        HTX_TASK.name: [],
        CRPC_TASK.name: []
    }
    baseline_pred = {
        PN_TASK.name: [],
        BCR_TASK.name: [],
        METASTASIS_TASK.name: [],
        HTX_TASK.name: [],
        CRPC_TASK.name: []
    }
    test_sets_mask = []
    for i in range(5):

        dataset.update_masks(
            train_mask=masks[str(i)][Mask.TRAIN],
            test_mask=masks[str(i)][Mask.TEST],
            valid_mask=masks[str(i)][Mask.VALID]
        )
        test_sets_mask += masks[str(i)][Mask.TEST]

        net_pred = json.load(open(rf"local_data\preds\split_{i}\BayesSeqNet.json", "r"))
        bayes_seq_net_pred[PN_TASK.name].append(net_pred[PN_TASK.name])
        bayes_seq_net_pred[BCR_TASK.name].append(net_pred[BCR_TASK.name])
        bayes_seq_net_pred[METASTASIS_TASK.name].append(net_pred[METASTASIS_TASK.name])
        bayes_seq_net_pred[HTX_TASK.name].append(net_pred[HTX_TASK.name])
        bayes_seq_net_pred[CRPC_TASK.name].append(net_pred[CRPC_TASK.name])

        capra_pred = pd.read_csv(os.path.join(NOMOGRAMS_PATH, "CAPRA", f"outer_split_{i}.csv"))
        mskcc_pred = pd.read_csv(os.path.join(NOMOGRAMS_PATH, "MSKCC", f"outer_split_{i}.csv"))

        baseline_pred[PN_TASK.name].append(mskcc_pred["PREDICTED_LYMPH_NODE_INVOLVEMENT"].to_numpy()[dataset.test_mask].tolist())
        baseline_pred[BCR_TASK.name].append(mskcc_pred["PREDICTED_PREOPERATIVE_BCR_RISK"].to_numpy()[dataset.test_mask].tolist())
        baseline_pred[METASTASIS_TASK.name].append(capra_pred["PREDICTED_METASTASIS_RISK"].to_numpy()[dataset.test_mask].tolist())
        baseline_pred[HTX_TASK.name].append(capra_pred["PREDICTED_HORMONOTHERAPY_RISK"].to_numpy()[dataset.test_mask].tolist())
        baseline_pred[CRPC_TASK.name].append(capra_pred["PREDICTED_CASTRATE_RESISTANT_RISK"].to_numpy()[dataset.test_mask].tolist())

    comparator = PredictionComparator(
        pred_1={k: np.concatenate(np.array(v)) for k, v in bayes_seq_net_pred.items()},
        pred_2={k: np.concatenate(np.array(v)) for k, v in baseline_pred.items()},
        ground_truth=dataset.table_dataset[test_sets_mask].y,
        tasks=[PN_TASK, BCR_TASK, METASTASIS_TASK, HTX_TASK, CRPC_TASK]
    )

    p_values = comparator.compute_c_index_p_value()
    print("C-index", p_values)
    p_values = comparator.compute_auc_p_value()
    print("AUC", p_values)

    metrics = {}
    for task in [BCR_TASK, CRPC_TASK, METASTASIS_TASK, HTX_TASK]:
        c_index_ipcw = ConcordanceIndexIPCW()
        c_index_ipcw.update_censoring_distribution(dataset.table_dataset[dataset.train_mask].y[task.name])
        metrics[task.name] = c_index_ipcw
    p_values = comparator.compute_any_metric_p_value(metrics)
    print("CIPCW", p_values)

    metrics = {}
    for task in [BCR_TASK, CRPC_TASK, METASTASIS_TASK, HTX_TASK]:
        c_index_ipcw = CumulativeDynamicAUC()
        c_index_ipcw.update_censoring_distribution(dataset.table_dataset[dataset.train_mask].y[task.name])
        metrics[task.name] = c_index_ipcw
    p_values = comparator.compute_any_metric_p_value(metrics)
    print("CDA", p_values)

    bba = BinaryBalancedAccuracy()
    bba._threshold = 0.58
    bba.update_scaling_factor(dataset.table_dataset[dataset.train_mask].y[PN_TASK.name])
    p_values = comparator.compute_any_metric_p_value(bba)
    print("BA", p_values)


    bayes_seq_net_pred = {DEATH_TASK.name: []}
    baseline_pred = {DEATH_TASK.name: []}
    test_sets_mask = []
    for i in [0, 2, 3, 4]:

        dataset.update_masks(
            train_mask=masks[str(i)][Mask.TRAIN],
            test_mask=masks[str(i)][Mask.TEST],
            valid_mask=masks[str(i)][Mask.VALID]
        )
        test_sets_mask += masks[str(i)][Mask.TEST]

        net_pred = json.load(open(rf"local_data\preds\split_{i}\BayesSeqNet.json", "r"))
        bayes_seq_net_pred[DEATH_TASK.name].append(net_pred[DEATH_TASK.name])

        capra_pred = pd.read_csv(os.path.join(NOMOGRAMS_PATH, "CAPRA", f"outer_split_{i}.csv"))
        mskcc_pred = pd.read_csv(os.path.join(NOMOGRAMS_PATH, "MSKCC", f"outer_split_{i}.csv"))

        baseline_pred[DEATH_TASK.name].append(capra_pred["PREDICTED_PREOPERATIVE_PROSTATE_CANCER_DEATH_RISK"].to_numpy()[dataset.test_mask].tolist())

    comparator = PredictionComparator(
        pred_1={k: np.concatenate(np.array(v)) for k, v in bayes_seq_net_pred.items()},
        pred_2={k: np.concatenate(np.array(v)) for k, v in baseline_pred.items()},
        ground_truth=dataset.table_dataset[test_sets_mask].y,
        tasks=[DEATH_TASK]
    )

    p_values = comparator.compute_c_index_p_value()
    print("C-index", p_values)
    p_values = comparator.compute_auc_p_value()
    print("AUC", p_values)

    metrics = {}
    for task in [DEATH_TASK]:
        c_index_ipcw = ConcordanceIndexIPCW()
        c_index_ipcw.update_censoring_distribution(dataset.table_dataset[dataset.train_mask].y[task.name])
        metrics[task.name] = c_index_ipcw
    p_values = comparator.compute_any_metric_p_value(metrics)
    print("CIPCW", p_values)

    metrics = {}
    for task in [DEATH_TASK]:
        c_index_ipcw = CumulativeDynamicAUC()
        c_index_ipcw.update_censoring_distribution(dataset.table_dataset[dataset.train_mask].y[task.name])
        metrics[task.name] = c_index_ipcw
    p_values = comparator.compute_any_metric_p_value(metrics)
    print("CDA", p_values)

