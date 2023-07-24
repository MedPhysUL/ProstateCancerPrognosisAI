"""
    @file:              14_tune_mlp_clinical_data_and_radiomics.py
    @Author:            Maxence Larose

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This script is used to tune an MLP model.
"""

import env_apps

import os
from typing import Dict, Union

from optuna.integration.botorch import BoTorchSampler
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import (
    AUTOMATIC_FILTERED_RADIOMICS_PATH,
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    DEEP_FILTERED_RADIOMICS_PATH,
    EXPERIMENTS_PATH,
    ID,
    MANUAL_FILTERED_RADIOMICS_PATH,
    MASKS_PATH,
    MLP_RAD_AND_CLIN_DATA_LR_HIGH_BOUND_DICT,
    PREDICTOR_CLIP_GRAD_MAX_NORM_DICT,
    RADIOMICS_FEATURES,
    SEED,
    TABLE_TASKS
)
from src.data.datasets import ProstateCancerDataset, TableDataset, Split
from src.data.processing.sampling import extract_masks
from src.losses.multi_task import MeanLoss
from src.models.torch.prediction import MLP
from src.training.callbacks.learning_algorithm import L2Regularizer, MultiTaskLossEarlyStopper
from src.tuning import SearchAlgorithm, TorchObjective, Tuner
from src.tuning.callbacks import TuningRecorder

from src.tuning.hyperparameters.optuna import (
    CategoricalHyperparameter,
    FixedHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter
)
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


def get_dataframes_dictionary(
        path_to_dataframes_folder: str,
        n_outer_loops: int,
        n_inner_loops: int
) -> Dict[int, Dict[str, Union[pd.DataFrame, Dict[int, pd.DataFrame]]]]:
    """
    This function returns a dictionary of dataframes. The keys of the first level are the outer split indices. The keys
    of the second level are the split indices. The values are the dataframes.

    Parameters
    ----------
    path_to_dataframes_folder : str
        The path to the folder containing the dataframes.
    n_outer_loops : int
        The number of outer loops.
    n_inner_loops : int
        The number of inner loops.

    Returns
    -------
    dataframes_dict : Dict[int, Dict[str, Union[pd.DataFrame, Dict[int, pd.DataFrame]]]]
        The dictionary of dataframes.
    """
    dataframes = {}
    for k in range(n_outer_loops):
        path_to_outer_split_folder = os.path.join(path_to_dataframes_folder, f"outer_split_{k}")

        dataframes[k] = {}
        dataframes[k][str(Split.OUTER)] = pd.read_csv(os.path.join(path_to_outer_split_folder, "outer_split.csv"))
        dataframes[k][str(Split.INNER)] = {}

        path_to_inner_splits_folder = os.path.join(path_to_outer_split_folder, "inner_splits")
        for l in range(n_inner_loops):
            path_to_inner_split = os.path.join(path_to_inner_splits_folder, f"inner_split_{l}.csv")
            dataframes[k][str(Split.INNER)][l] = pd.read_csv(path_to_inner_split)

    return dataframes


if __name__ == '__main__':
    for task in TABLE_TASKS:
        df_dict = get_dataframes_dictionary(
            path_to_dataframes_folder=os.path.join(DEEP_FILTERED_RADIOMICS_PATH, task.target_column),
            n_outer_loops=5,
            n_inner_loops=5
        )

        table_dataset = TableDataset(
            dataframe=df_dict[0][str(Split.OUTER)],  # Dummy dataset - Never used
            ids_column=ID,
            tasks=task,
            continuous_features=CLINICAL_CONTINUOUS_FEATURES + RADIOMICS_FEATURES,
            categorical_features=CLINICAL_CATEGORICAL_FEATURES
        )

        dataset = ProstateCancerDataset(image_dataset=None, table_dataset=table_dataset)

        path_to_record_folder = os.path.join(
            EXPERIMENTS_PATH,
            f"{task.target_column}(MLP - Clinical data and deep radiomics)"
        )

        search_algo = SearchAlgorithm(
            sampler=BoTorchSampler(
                n_startup_trials=5,
                seed=SEED
            ),
            storage="sqlite:///" + os.path.join(path_to_record_folder, "tuning_history.db")
        )

        tuner = Tuner(
            search_algorithm=search_algo,
            recorder=TuningRecorder(path_to_record_folder=path_to_record_folder),
            n_trials=25,
            seed=SEED
        )

        model_hyperparameter = TorchModelHyperparameter(
            constructor=MLP,
            parameters={
                "activation": FixedHyperparameter(name="activation", value="PReLU"),
                "n_layers": IntegerHyperparameter(name="n_layers", low=1, high=3),
                "n_neurons": IntegerHyperparameter(name="n_neurons", low=5, high=30),
                "dropout": FloatHyperparameter(name="dropout", low=0.05, high=0.25)
            }
        )

        learning_algorithm_hyperparameter = LearningAlgorithmHyperparameter(
            criterion=CriterionHyperparameter(
                constructor=MeanLoss
            ),
            optimizer=OptimizerHyperparameter(
                constructor=Adam,
                parameters={
                    "lr": FloatHyperparameter(
                        name="lr",
                        low=1e-4,
                        high=MLP_RAD_AND_CLIN_DATA_LR_HIGH_BOUND_DICT[task.target_column],
                        log=True
                    ),
                    "weight_decay": FloatHyperparameter(name="weight_decay", low=1e-4, high=1e-2, log=True)
                }
            ),
            clip_grad_max_norm=PREDICTOR_CLIP_GRAD_MAX_NORM_DICT[task.target_column],
            early_stopper=EarlyStopperHyperparameter(
                constructor=MultiTaskLossEarlyStopper,
                parameters={"patience": 20}
            ),
            lr_scheduler=LRSchedulerHyperparameter(
                constructor=ExponentialLR,
                parameters={"gamma": FixedHyperparameter(name="gamma", value=0.99)}
            ),
            regularizer=RegularizerHyperparameter(
                constructor=L2Regularizer,
                parameters={"lambda_": FloatHyperparameter(name="alpha", low=1e-5, high=1e-2, log=True)}
            )
        )

        trainer_hyperparameter = TrainerHyperparameter(
            batch_size=16,
            n_epochs=100
            # checkpoint=CheckpointHyperparameter(save_freq=20)
        )

        train_method_hyperparameter = TrainMethodHyperparameter(
            model=model_hyperparameter,
            learning_algorithms=learning_algorithm_hyperparameter
        )

        objective = TorchObjective(
            trainer_hyperparameter=trainer_hyperparameter,
            train_method_hyperparameter=train_method_hyperparameter
        )

        masks = extract_masks(MASKS_PATH, k=5, l=5)

        tuner.tune(
            objective=objective,
            dataset=dataset,
            masks=masks,
            dataframes=df_dict
        )
