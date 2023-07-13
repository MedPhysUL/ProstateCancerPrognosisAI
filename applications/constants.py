"""
    @file:              constants.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 03/2023

    @Description:       This file stores helpful constants.
"""

import os

from src.losses.single_task import BCEWithLogitsLoss, DiceLoss, NegativePartialLogLikelihood
from src.metrics.single_task import (
    AUC, BinaryBalancedAccuracy, ConcordanceIndexCensored, DiceMetric, Sensitivity, Specificity
)
from src.tasks import BinaryClassificationTask, SegmentationTask, SurvivalAnalysisTask
from src.data.datasets import Feature
from src.data.transforms import Normalization, MappingEncoding

# SEED
SEED = 1837

# SIZE
HOLDOUT_SIZE = 0.15

# PATHS
DATA_PATH = "local_data"
CLINICAL_DATA_PATH = os.path.join(DATA_PATH, "clinical_data.xlsx")
IMAGES_FOLDER_PATH = os.path.join(DATA_PATH, "Images")

LEARNING_TABLE_PATH = os.path.join(DATA_PATH, "learning_table.csv")
HOLDOUT_TABLE_PATH = os.path.join(DATA_PATH, "holdout_table.csv")

MASKS_PATH: str = os.path.join(DATA_PATH, "masks")

RECORDS_PATH = os.path.join("local_data", "records")
OUTLIERS_RECORDS_PATH = os.path.join(RECORDS_PATH, "outliers")
DESCRIPTIVE_ANALYSIS_PATH = os.path.join(RECORDS_PATH, "descriptive_analyses")
EXPERIMENTS_PATH = os.path.join(RECORDS_PATH, "experiments")
CHECKPOINTS_PATH = os.path.join(EXPERIMENTS_PATH, "checkpoints")

# ID COLUMN
ID = "ID"

# FEATURES
AGE = Feature(column="AGE", transform=Normalization())
CLINICAL_STAGE = Feature(column="CLINICAL_STAGE", transform=MappingEncoding({"T1-T2": 0, "T3a": 1}))
GLEASON_GLOBAL = Feature(column="GLEASON_GLOBAL", transform=MappingEncoding({8: 0, 9: 0.5, 10: 1}))
GLEASON_PRIMARY = Feature(column="GLEASON_PRIMARY", transform=MappingEncoding({3: 0, 4: 0.5, 5: 1}))
GLEASON_SECONDARY = Feature(column="GLEASON_SECONDARY", transform=MappingEncoding({3: 0, 4: 0.5, 5: 1}))
PSA = Feature(column="PSA", transform=Normalization())

CONTINUOUS_FEATURES = [AGE, PSA]
CATEGORICAL_FEATURES = [CLINICAL_STAGE, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY]
FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

# TARGETS
BCR = "BCR"
BCR_TIME = "BCR_TIME"
CRPC = "CRPC"
CRPC_TIME = "CRPC_TIME"
DEATH = "DEATH"
DEATH_TIME = "DEATH_TIME"
HTX = "HTX"
HTX_TIME = "HTX_TIME"
METASTASIS = "METASTASIS"
METASTASIS_TIME = "METASTASIS_TIME"
PN = "PN"

# TABLE TASKS
BCR_TASK = SurvivalAnalysisTask(
    event_indicator_column=BCR,
    event_time_column=BCR_TIME,
    criterion=NegativePartialLogLikelihood(),
    hps_tuning_metric=ConcordanceIndexCensored(),
    temperature=...
)
CRPC_TASK = SurvivalAnalysisTask(
    event_indicator_column=CRPC,
    event_time_column=CRPC_TIME,
    criterion=NegativePartialLogLikelihood(),
    hps_tuning_metric=ConcordanceIndexCensored(),
    temperature=...
)
DEATH_TASK = SurvivalAnalysisTask(
    event_indicator_column=DEATH,
    event_time_column=DEATH_TIME,
    criterion=NegativePartialLogLikelihood(),
    hps_tuning_metric=ConcordanceIndexCensored(),
    temperature=...
)
HTX_TASK = SurvivalAnalysisTask(
    event_indicator_column=HTX,
    event_time_column=HTX_TIME,
    criterion=NegativePartialLogLikelihood(),
    hps_tuning_metric=ConcordanceIndexCensored(),
    temperature=...
)
METASTASIS_TASK = SurvivalAnalysisTask(
    event_indicator_column=METASTASIS,
    event_time_column=METASTASIS_TIME,
    criterion=NegativePartialLogLikelihood(),
    hps_tuning_metric=ConcordanceIndexCensored(),
    temperature=...
)
PN_TASK = BinaryClassificationTask(
    target_column=PN,
    decision_threshold_metric=BinaryBalancedAccuracy(),
    hps_tuning_metric=AUC(),
    evaluation_metrics=[Sensitivity(), Specificity()],
    criterion=BCEWithLogitsLoss(),
    temperature=...
)

PREDICTOR_CLIP_GRAD_MAX_NORM_DICT = {BCR: 3.0, CRPC: 3.0, DEATH: 1.0, HTX: 3.0, METASTASIS: 2.0, PN: 3.0}
EXTRACTOR_CLIP_GRAD_MAX_NORM_DICT = {BCR: 10.0, CRPC: 10.0, DEATH: 5.0, HTX: 10.0, METASTASIS: 8.0, PN: 10.0}

CNN_LR_HIGH_BOUND_DICT = {BCR: 1e-3, CRPC: 1e-3, DEATH: 5e-4, HTX: 1e-3, METASTASIS: 5e-4, PN: 1e-3}

UNEXTRACTOR_LR_LOW_BOUND_DICT = {BCR: 1e-4, CRPC: 1e-4, DEATH: 5e-5, HTX: 1e-4, METASTASIS: 5e-5, PN: 1e-4}
UNEXTRACTOR_LR_HIGH_BOUND_DICT = {BCR: 1e-3, CRPC: 1e-3, DEATH: 5e-4, HTX: 1e-3, METASTASIS: 5e-4, PN: 1e-3}

TABLE_TASKS = [BCR_TASK, CRPC_TASK, DEATH_TASK, HTX_TASK, METASTASIS_TASK, PN_TASK]

# IMAGE TASKS
PROSTATE_SEGMENTATION_TASK = SegmentationTask(
    criterion=DiceLoss(),
    evaluation_metrics=DiceMetric(),
    organ="Prostate",
    modality="CT"
)

IMAGE_TASKS = [PROSTATE_SEGMENTATION_TASK]
