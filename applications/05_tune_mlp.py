"""
    @file:              04_test.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This script is used to generate a folder containing descriptive analyses of all data.
"""

from time import time

import pandas as pd
from constants import *
from hps import MLP_HPS

from src.data.datasets import ProstateCancerDataset, TableDataset
from src.data.processing.sampling import extract_masks
from src.models.mlp import MLP
from src.evaluation import Evaluator
from src.losses.multi_task import MeanLoss


if __name__ == '__main__':
    # Start timer
    start = time()

    # Table dataset
    df = pd.read_csv(LEARNING_TABLE_PATH)

    feature_cols = [AGE, PSA, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    target_cols = [PN, BCR]

    df = df[[ID] + feature_cols + target_cols]

    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=TABLE_TASKS,
        cont_cols=[AGE, PSA],
        cat_cols=[GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    )

    # Creation of the dataset
    dataset = ProstateCancerDataset(image_dataset=None, table_dataset=table_dataset)

    # Saving of the fixed params of MLP
    fixed_params = {
        'max_epochs': 100,
        'output_size': len(dataset.tasks),
        'criterion': MeanLoss(),
        'early_stopper_type': EarlyStopperType.MULTITASK_LOSS,
        'path_to_model': CHECKPOINTS_PATH,
        'patience': 20
    }

    # Masks
    masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=2, l=2)

    # Creation of the evaluator
    evaluator = Evaluator(model_constructor=MLP,
                          dataset=dataset,
                          masks=masks,
                          evaluation_name=f"TEST",
                          hps=MLP_HPS,
                          n_trials=100,
                          feature_selector=None,
                          fixed_params=fixed_params,
                          path_to_experiment_records=EXPERIMENTS_PATH,
                          save_hps_importance=True,
                          save_optimization_history=True,
                          save_parallel_coordinates=True,
                          save_pareto_front=True,
                          seed=42)

    # Evaluation
    evaluator.evaluate()

    print(f"Time taken for MLP (minutes): {(time() - start) / 60:.2f}")

