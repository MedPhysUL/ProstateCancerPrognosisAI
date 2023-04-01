import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from delia.databases import PatientsDatabase

from constants import *
from src.data.processing.sampling import extract_masks, Mask
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.models.torch.prediction import TabularMLP
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper
from src.training.callbacks.learning_algorithm.regularizer import L2Regularizer


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(DATA_PATH, "midl2023_learning_table.csv"))

    feature_cols = [AGE, PSA, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    target_cols = [PN, BCR, BCR_TIME]

    df = df[[ID] + feature_cols + target_cols]

    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=TABLE_TASKS,
        cont_cols=[AGE, PSA],
        cat_cols=[GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    )

    database = PatientsDatabase(path_to_database=r"local_data/midl2023_learning_set.h5")

    dataset = ProstateCancerDataset(table_dataset=table_dataset)

    masks = extract_masks(os.path.join(MASKS_PATH, "midl2023_masks.json"), k=1, l=0)

    dataset.update_masks(
        train_mask=masks[0][Mask.TRAIN],
        valid_mask=masks[0][Mask.VALID],
        test_mask=[]
    )

    model = TabularMLP(
        hidden_channels=[30, 30, 30],
        activation="PRELU",
        dropout=0.2,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    optimizer = Adam(
        params=model.parameters(),
        lr=2e-4,
        weight_decay=0.02
    )

    learning_algorithm = LearningAlgorithm(
        criterion=MeanLoss(),
        optimizer=optimizer,
        lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=0.99),
        early_stopper=MultiTaskLossEarlyStopper(patience=20),
        regularizer=L2Regularizer(model.named_parameters(), lambda_=0.01)
    )

    trainer = Trainer(
        batch_size=8,
        checkpoint=Checkpoint(),
        exec_metrics_on_train=True,
        n_epochs=200,
        seed=SEED
    )

    trained_model, history = trainer.train(
        model=model,
        dataset=dataset,
        learning_algorithms=learning_algorithm
    )

    history.plot(show=True)
