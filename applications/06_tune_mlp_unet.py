"""
    @file:              06_tune_mlp_unet.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This script is used to tune an mlp and 3D Unet model.
"""

from time import time

import pandas as pd
from monai.transforms import (
    AddChanneld,
    CenterSpatialCropd,
    Compose,
    HistogramNormalized,
    KeepLargestConnectedComponentd,
    ThresholdIntensityd,
    ToTensord,
)
from torch import float32

from constants import *
from hps import MLP_HPS
from src.data.datasets.image_dataset import ImageDataset
from src.data.datasets.prostate_cancer_dataset import ProstateCancerDataset
from src.data.datasets.table_dataset import TableDataset
from src.data.extraction.local import LocalDatabaseManager
from src.data.processing.sampling import extract_masks

from src.models.mlp_unet import MLPUnet
from src.training.evaluator import Evaluator
from src.training.early_stopper import EarlyStopperType
from src.utils.multi_task_losses import MeanLoss


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

    database_manager = LocalDatabaseManager(
        path_to_database='D:/ProstateCancerData/learning_set.h5'
    )

    trans = Compose([
        AddChanneld(keys=['CT', 'Prostate']),
        CenterSpatialCropd(keys=['CT', 'Prostate'], roi_size=(1000, 120, 120)),
        ThresholdIntensityd(keys=['CT'], threshold=-250, above=True, cval=-250),
        ThresholdIntensityd(keys=['CT'], threshold=500, above=False, cval=500),
        HistogramNormalized(keys=['CT'], num_bins=751, min=0, max=1),
        KeepLargestConnectedComponentd(keys=['Prostate']),
        ToTensord(keys=['CT', 'Prostate'], dtype=float32)
    ])

    image_dataset = ImageDataset(
        database_manager=database_manager,
        tasks=IMAGE_TASKS,
        modalities={"CT"},
        transforms=trans
    )

    # Creation of the dataset
    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=table_dataset)

    # Saving of the fixed params of MLP
    fixed_params = {
        'max_epochs': 50,
        'output_size': len(dataset.tasks),
        'criterion': MeanLoss(),
        'early_stopper_type': EarlyStopperType.MULTITASK_LOSS,
        'path_to_model': CHECKPOINTS_PATH,
        'patience': 10
    }

    # Masks
    masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=1, l=1)

    # Creation of the evaluator
    evaluator = Evaluator(model_constructor=MLPUnet,
                          dataset=dataset,
                          masks=masks,
                          evaluation_name=f"TEST",
                          hps=MLP_HPS,
                          n_trials=3,
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

