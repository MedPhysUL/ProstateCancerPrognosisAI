import env_apps

from delia.databases import PatientsDatabase
from optuna.integration.botorch import BoTorchSampler
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import *
from src.data.datasets import ImageDataset, ProstateCancerDataset, TableDataset
from src.data.processing.sampling import extract_masks
from src.losses.multi_task import MeanLoss
from src.models.torch import CNN
from src.training.callbacks.learning_algorithm import L2Regularizer, MultiTaskLossEarlyStopper
from src.tuning import SearchAlgorithm, TorchObjective, Tuner
from src.tuning.callbacks import TuningRecorder

from src.tuning.hyperparameters.optuna import (
    CategoricalHyperparameter,
    FixedHyperparameter,
    FloatHyperparameter
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


if __name__ == '__main__':
    df = pd.read_csv(LEARNING_TABLE_PATH)
    task = BCR_TASK

    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=task
    )

    database = PatientsDatabase(path_to_database=r"local_data/learning_set.h5")

    image_dataset = ImageDataset(
        database=database,
        modalities={"PT", "CT"}
    )

    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=table_dataset)

    path_to_record_folder = os.path.join(
        EXPERIMENTS_PATH,
        f"{task.target_column}(CNN - Deep radiomics)"
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
        constructor=CNN,
        parameters={
                    "image_keys": ["CT", "PT"],
                    "model_mode": "prediction",
                    "channels": FixedHyperparameter(name="channels", value=(64, 128, 256, 512, 1024)),
                    "kernel_size": FixedHyperparameter(name="kernel_size", value=3),
                    "num_res_units": FixedHyperparameter(name="num_res_units", value=3),
                    "dropout_cnn": FloatHyperparameter(name="dropout_cnn", low=0.2, high=0.8),
                    "dropout_fnn": FloatHyperparameter(name="dropout_fnn", low=0.1, high=0.4)
        }
    )

    extractor_learning_algorithm_hyperparameter = LearningAlgorithmHyperparameter(
        criterion=CriterionHyperparameter(
            constructor=MeanLoss
        ),
        optimizer=OptimizerHyperparameter(
            constructor=Adam,
            model_params_getter=lambda model: model.extractor.parameters(),
            parameters={
                "lr": FloatHyperparameter(name="lr", low=1e-5, high=1e-3, log=True),
                "weight_decay": FloatHyperparameter(name="weight_decay", low=1e-3, high=1e-1, log=True)
            }
        ),
        clip_grad_max_norm=EXTRACTOR_CLIP_GRAD_MAX_NORM_DICT[task.target_column],
        early_stopper=EarlyStopperHyperparameter(
            constructor=MultiTaskLossEarlyStopper,
            parameters={"patience": 20}
        ),
        lr_scheduler=LRSchedulerHyperparameter(
            constructor=ExponentialLR,
            parameters={"gamma": FixedHyperparameter(name="gamma", value=0.95)}
        ),
        regularizer=RegularizerHyperparameter(
            constructor=L2Regularizer,
            model_params_getter=lambda model: model.extractor.named_parameters(),
            parameters={"lambda_": FloatHyperparameter(name="alpha", low=1e-4, high=1e-2, log=True)}
        )
    )

    trainer_hyperparameter = TrainerHyperparameter(
        batch_size=16,
        n_epochs=100,
        verbose=False
        # checkpoint=CheckpointHyperparameter(save_freq=20)
    )

    train_method_hyperparameter = TrainMethodHyperparameter(
        model=model_hyperparameter,
        learning_algorithms=extractor_learning_algorithm_hyperparameter
    )

    objective = TorchObjective(
        trainer_hyperparameter=trainer_hyperparameter,
        train_method_hyperparameter=train_method_hyperparameter
    )

    masks = extract_masks(os.path.join(MASKS_PATH, "masks.json"), k=5, l=5)

    tuner.tune(
        objective=objective,
        dataset=dataset,
        masks=masks
    )
