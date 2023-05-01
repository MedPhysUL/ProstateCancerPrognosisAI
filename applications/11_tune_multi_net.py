"""
    @file:              07_tune_mlp.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 03/2023

    @Description:       This script is used to tune an MLP model.
"""

import env_apps
from optuna.samplers import TPESampler
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from delia.databases import PatientsDatabase

from constants import *
from src.data.datasets import ImageDataset, ProstateCancerDataset, TableDataset
from src.data.processing.sampling import extract_masks
from src.losses.multi_task import MeanLoss
from src.models.torch import CNN, MLP, MultiNet
from src.models.torch.combination import ModelSetup
from src.training.callbacks.learning_algorithm import L2Regularizer, MultiTaskLossEarlyStopper
from src.tuning import SearchAlgorithm, TorchObjective, Tuner
from src.tuning.callbacks import TuningRecorder

from src.tuning.hyperparameters.containers import (
    HyperparameterList
)
from src.tuning.hyperparameters.optuna import (
    CategoricalHyperparameter,
    FixedHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter
)
from src.tuning.hyperparameters.containers import HyperparameterObject
from src.tuning.hyperparameters.torch import (
    CheckpointHyperparameter,
    CriterionHyperparameter,
    EarlyStopperHyperparameter,
    LearningAlgorithmHyperparameter,
    LRSchedulerHyperparameter,
    OptimizerHyperparameter,
    RegularizerHyperparameter,
    TorchModelHyperparameter,
    TrainerHyperparameter,
    TrainMethodHyperparameter
)


if __name__ == '__main__':
    df = pd.read_csv(LEARNING_TABLE_PATH)

    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=PN_TASK,
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

    path_to_record_folder = os.path.join(
        EXPERIMENTS_PATH,
        f"{PN_TASK.target_column}(MultiNet - Clinical data + Deep radiomics)"
    )

    search_algo = SearchAlgorithm(
        sampler=TPESampler(
            n_startup_trials=10,
            multivariate=True,
            seed=SEED
        ),
        storage="sqlite:///" + os.path.join(path_to_record_folder, "tuning_history.db")
    )

    tuner = Tuner(
        search_algorithm=search_algo,
        recorder=TuningRecorder(path_to_record_folder=path_to_record_folder),
        n_trials=50,
        seed=SEED
    )

    model_hyperparameter = TorchModelHyperparameter(
        constructor=MultiNet,
        parameters={
            "predictor_setup": HyperparameterObject(
                constructor=ModelSetup,
                parameters={
                    "model": TorchModelHyperparameter(
                        constructor=MLP,
                        parameters={
                            "activation": FixedHyperparameter(name="activation", value="PReLU"),
                            "input_mode": "hybrid",
                            "multi_task_mode": "separated",
                            "hidden_channels": CategoricalHyperparameter(
                                name="hidden_channels",
                                choices=[
                                    "(10, 10)", "(15, 15)", "(20, 20)",
                                    "(10, 10, 10)", "(15, 15, 15)", "(20, 20, 20)"
                                ]
                            ),
                            "dropout": FloatHyperparameter(name="dropout_mlp", low=0.05, high=0.25)
                        }
                    )
                }
            ),
            "extractor_setup": HyperparameterObject(
                constructor=ModelSetup,
                parameters={
                    "model": TorchModelHyperparameter(
                        constructor=CNN,
                        parameters={
                            "image_keys": ["CT", "PT"],
                            "segmentation_key_or_task": "CT_Prostate",
                            "model_mode": "extraction",
                            "merging_method": "multiplication",
                            "multi_task_mode": "separated",
                            "channels": CategoricalHyperparameter(
                                name="channels",
                                choices=[
                                    "(4, 8, 16, 32)", "(8, 16, 32, 64)", "(16, 32, 64, 128)",
                                    "(4, 8, 16, 32, 64)", "(8, 16, 32, 64, 128)", "(16, 32, 64, 128, 256)"
                                ]
                            ),
                            "kernel_size": IntegerHyperparameter(name="kernel_size", low=3, high=7, step=2),
                            "num_res_units": IntegerHyperparameter(name="num_res_units", low=0, high=3),
                            "dropout": FloatHyperparameter(name="dropout_cnn", low=0.1, high=0.5)
                        }
                    ),
                }
            )
        }
    )

    predictor_learning_algorithm_hyperparameter = LearningAlgorithmHyperparameter(
        criterion=CriterionHyperparameter(
            constructor=MeanLoss
        ),
        optimizer=OptimizerHyperparameter(
            constructor=Adam,
            model_params_getter=lambda model: model.predictor.parameters(),
            parameters={"lr": FloatHyperparameter(name="lr_predictor", low=1e-4, high=1e-2, log=True)}
        ),
        early_stopper=EarlyStopperHyperparameter(
            constructor=MultiTaskLossEarlyStopper,
            parameters={"patience": 20}
        ),
        lr_scheduler=LRSchedulerHyperparameter(
            constructor=ExponentialLR,
            parameters={"gamma": CategoricalHyperparameter(name="gamma_predictor", choices=[0.9, 0.99, 0.999])}
        ),
        regularizer=RegularizerHyperparameter(
            constructor=L2Regularizer,
            model_params_getter=lambda model: model.predictor.parameters(),
            parameters={"lambda_": FloatHyperparameter(name="alpha_predictor", low=1e-4, high=1e-2, log=True)}
        )
    )

    extractor_learning_algorithm_hyperparameter = LearningAlgorithmHyperparameter(
        criterion=CriterionHyperparameter(
            constructor=MeanLoss
        ),
        optimizer=OptimizerHyperparameter(
            constructor=Adam,
            model_params_getter=lambda model: model.extractor.parameters(),
            parameters={"lr": FloatHyperparameter(name="lr_extractor", low=1e-5, high=1e-2, log=True)}
        ),
        early_stopper=EarlyStopperHyperparameter(
            constructor=MultiTaskLossEarlyStopper,
            parameters={"patience": 20}
        ),
        lr_scheduler=LRSchedulerHyperparameter(
            constructor=ExponentialLR,
            parameters={"gamma": CategoricalHyperparameter(name="gamma_extractor", choices=[0.9, 0.99, 0.999])}
        ),
        regularizer=RegularizerHyperparameter(
            constructor=L2Regularizer,
            model_params_getter=lambda model: model.extractor.parameters(),
            parameters={"lambda_": FloatHyperparameter(name="alpha_extractor", low=1e-5, high=1e-3, log=True)}
        )
    )

    trainer_hyperparameter = TrainerHyperparameter(
        batch_size=8,
        n_epochs=100
        # checkpoint=CheckpointHyperparameter(save_freq=20)
    )

    train_method_hyperparameter = TrainMethodHyperparameter(
        model=model_hyperparameter,
        learning_algorithms=HyperparameterList(
            [predictor_learning_algorithm_hyperparameter, extractor_learning_algorithm_hyperparameter]
        )
    )

    objective = TorchObjective(
        trainer_hyperparameter=trainer_hyperparameter,
        train_method_hyperparameter=train_method_hyperparameter
    )

    masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=5, l=3)

    tuner.tune(
        objective=objective,
        dataset=dataset,
        masks=masks
    )
