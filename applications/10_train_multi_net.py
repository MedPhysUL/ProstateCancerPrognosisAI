"""
    @file:              10_train_multi_net.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This script is used to train a multi-net model.
"""

import env_apps

import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from delia.databases import PatientsDatabase

from constants import *
from src.data.processing.sampling import extract_masks, Mask
from src.data.datasets import ImageDataset, ProstateCancerDataset, TableDataset
from src.models.torch.combination import ModelSetup, MultiNet
from src.models.torch.extraction import CNN
from src.models.torch.segmentation import Unet
from src.models.torch.prediction import MLP
from src.losses.multi_task import MeanLoss
from src.losses.multi_task import WeightedMeanLoss
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
        tasks=PROSTATE_SEGMENTATION_TASK
    )

    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=table_dataset)
    masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=2, l=2)

    dataset.update_masks(
        train_mask=masks[0][Mask.TRAIN],
        test_mask=masks[0][Mask.TEST],
        valid_mask=masks[0][Mask.VALID]
    )

    cnn = CNN(
        image_keys=["PT"],
        segmentation_key_or_task=PROSTATE_SEGMENTATION_TASK,
        model_mode="extraction",
        merging_method="multiplication",
        multi_task_mode="separated",
        dropout=0.5
    )

    unet = Unet(
        image_keys="CT",
        spatial_dims=3,
        num_res_units=3,
        dropout=0.2
    )

    mlp = MLP(
        multi_task_mode="separated",
        dropout=0.2
    )

    multi_net = MultiNet(
        predictor_setup=ModelSetup(mlp),
        extractor_setup=ModelSetup(cnn),
        segmentor_setup=ModelSetup(unet),
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    optimizer_mlp = Adam(
        params=mlp.parameters(),
        lr=2e-4,
        weight_decay=0.02
    )

    learning_algorithm_mlp = LearningAlgorithm(
        criterion=MeanLoss(tasks=PN_TASK),
        optimizer=optimizer_mlp,
        lr_scheduler=ExponentialLR(optimizer=optimizer_mlp, gamma=0.99),
        early_stopper=MultiTaskLossEarlyStopper(patience=20),
        regularizer=L2Regularizer(mlp.named_parameters(), lambda_=0.01)
    )

    optimizer_cnn = Adam(
        params=cnn.parameters(),
        lr=3e-5,
        weight_decay=0.1
    )

    learning_algorithm_cnn = LearningAlgorithm(
        criterion=MeanLoss(tasks=PN_TASK),
        optimizer=optimizer_cnn,
        lr_scheduler=ExponentialLR(optimizer=optimizer_cnn, gamma=0.99),
        early_stopper=MultiTaskLossEarlyStopper(patience=20),
        regularizer=L2Regularizer(cnn.named_parameters(), lambda_=0.02)
    )

    optimizer_unet = Adam(
        params=unet.parameters(),
        lr=1e-4
    )

    learning_algorithm_unet = LearningAlgorithm(
        criterion=WeightedMeanLoss(tasks=[PROSTATE_SEGMENTATION_TASK, PN_TASK], weights=[1/2, 1/2]),
        optimizer=optimizer_unet,
        lr_scheduler=ExponentialLR(optimizer=optimizer_unet, gamma=0.99)
    )

    trainer = Trainer(
        batch_size=8,
        checkpoint=Checkpoint(),
        exec_metrics_on_train=True,
        n_epochs=100,
        seed=SEED
    )

    trained_model, history = trainer.train(
        model=multi_net,
        dataset=dataset,
        learning_algorithms=[learning_algorithm_unet, learning_algorithm_cnn, learning_algorithm_mlp]
    )

    history.plot(show=True)

    # The next part will be integrated in an evaluation tool in the near future.
    multi_net.fix_thresholds_to_optimal_values(dataset)
    score = multi_net.score_on_dataset(dataset, dataset.test_mask)
    print(score)
