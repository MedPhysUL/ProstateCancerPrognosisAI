"""
    @file:              09_train_unextractor.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This script is used to train a UNETextractor model.
"""

import env_apps

from delia.databases import PatientsDatabase
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import *
from src.data.processing.sampling import extract_masks, Mask
from src.data.datasets import ImageDataset, ProstateCancerDataset, TableDataset
from src.models.torch.extraction import UNEXtractor
from src.losses.multi_task import MeanLoss, WeightedSumLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper
from src.training.callbacks.learning_algorithm.regularizer import L2Regularizer


if __name__ == '__main__':
    df = pd.read_csv(LEARNING_TABLE_PATH)

    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=BCR_TASK
    )

    database = PatientsDatabase(path_to_database=r"local_data/learning_set.h5")

    image_dataset = ImageDataset(
        database=database,
        modalities={"PT", "CT"},
        tasks=PROSTATE_SEGMENTATION_TASK
    )

    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=table_dataset)

    masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=5, l=5)

    dataset.update_masks(
        train_mask=masks[0][Mask.TRAIN],
        test_mask=masks[0][Mask.TEST],
        valid_mask=masks[0][Mask.VALID]
    )

    model = UNEXtractor(
        image_keys=["CT", "PT"],
        dropout_cnn=0.2,
        dropout_fnn=0.2,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    optimizer = Adam(
        params=model.parameters(),
        lr=1e-3,
        weight_decay=0.001
    )

    learning_algorithm = LearningAlgorithm(
        criterion=WeightedSumLoss(weights=[1/2, 1/2], tasks=[BCR_TASK, PROSTATE_SEGMENTATION_TASK]),
        optimizer=optimizer,
        clip_grad_max_norm=EXTRACTOR_CLIP_GRAD_MAX_NORM_DICT[BCR_TASK.target_column],
        lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=0.99),
        early_stopper=MultiTaskLossEarlyStopper(criterion=MeanLoss(tasks=BCR_TASK), patience=20),
        regularizer=L2Regularizer(model.named_parameters(), lambda_=0.001)
    )
    trainer = Trainer(
        batch_size=8,
        # checkpoint=Checkpoint(),
        exec_metrics_on_train=True,
        n_epochs=100,
        seed=SEED
    )

    trained_model, history = trainer.train(
        model=model,
        dataset=dataset,
        learning_algorithms=learning_algorithm
    )

    history.plot(show=True)

    trained_model.fix_thresholds_to_optimal_values(dataset)
    score = trained_model.compute_score_on_dataset(dataset, dataset.train_mask)
    print(score)

    score = trained_model.compute_score_on_dataset(dataset, dataset.valid_mask)
    print(score)

    score = trained_model.compute_score_on_dataset(dataset, dataset.test_mask)
    print(score)
