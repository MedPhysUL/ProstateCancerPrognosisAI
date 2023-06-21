"""
    @file:              07_train_cnn.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This script is used to train a cnn model.
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
from src.models.torch.extraction import CNN
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper
from src.training.callbacks.learning_algorithm.regularizer import L2Regularizer


if __name__ == '__main__':
    df = pd.read_csv(LEARNING_TABLE_PATH)

    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=TABLE_TASKS,
        cont_features=CONTINUOUS_FEATURES,
        cat_features=CATEGORICAL_FEATURES
    )

    database = PatientsDatabase(path_to_database=r"local_data/learning_set.h5")

    image_dataset = ImageDataset(
        database=database,
        modalities={"PT", "CT"},
        organs={"CT": {"Prostate"}}
    )

    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=table_dataset)

    masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=5, l=5)

    dataset.update_masks(
        train_mask=masks[0][Mask.TRAIN],
        test_mask=masks[0][Mask.TEST],
        valid_mask=masks[0][Mask.VALID]
    )

    model = CNN(
        image_keys=["PT"],
        segmentation_key_or_task="CT_Prostate",
        model_mode="prediction",
        merging_method="multiplication",
        multi_task_mode="separated",
        dropout_cnn=0.1,
        dropout_fnn=0.1,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    optimizer = Adam(
        params=model.parameters(),
        lr=3e-5,
        weight_decay=0.01
    )

    learning_algorithm = LearningAlgorithm(
        criterion=MeanLoss(),
        optimizer=optimizer,
        lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=0.99),
        early_stopper=MultiTaskLossEarlyStopper(patience=10),
        regularizer=L2Regularizer(model.named_parameters(), lambda_=0.02)
    )
    trainer = Trainer(
        batch_size=16,
        checkpoint=Checkpoint(),
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
