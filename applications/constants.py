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
    AUC,
    BinaryBalancedAccuracy,
    ConcordanceIndexCensored,
    ConcordanceIndexIPCW,
    CumulativeDynamicAUC,
    DiceMetric,
    Sensitivity,
    Specificity
)
from src.tasks import BinaryClassificationTask, SegmentationTask, SurvivalAnalysisTask
from src.data.datasets import Feature
from src.data.transforms import Normalization, MappingEncoding

# SEED
SEED = 1837

# SIZE
HOLDOUT_SIZE = 0.15

# PATHS
LOGGING_CONFIG_PATH = "logging_conf.yaml"
DATA_PATH = "local_data"

HOLDOUT_SET_FOLDER_PATH = os.path.join(DATA_PATH, "Holdout_set")
HOLDOUT_SET_PATH = os.path.join(DATA_PATH, "holdout_set.h5")
LEARNING_SET_FOLDER_PATH = os.path.join(DATA_PATH, "Learning_set")
LEARNING_SET_PATH = os.path.join(DATA_PATH, "learning_set.h5")

MASKS_PATH = os.path.join(DATA_PATH, "masks", "masks.json")
NOMOGRAMS_PATH = os.path.join(DATA_PATH, "nomograms")
SERIES_DESCRIPTIONS_PATH = os.path.join(DATA_PATH, "series_descriptions.json")

LEARNING_TABLE_PATH = os.path.join(DATA_PATH, "learning_table.csv")
HOLDOUT_TABLE_PATH = os.path.join(DATA_PATH, "holdout_table.csv")
MSKCC_LEARNING_TABLE_PATH = os.path.join(DATA_PATH, "mskcc_learning_table.csv")
MSKCC_HOLDOUT_TABLE_PATH = os.path.join(DATA_PATH, "mskcc_holdout_table.csv")

_RADIOMICS_PATH = os.path.join(DATA_PATH, "radiomics")
_AUTOMATIC_RADIOMICS_PATH = os.path.join(_RADIOMICS_PATH, "automatic")
AUTOMATIC_EXTRACTED_RADIOMICS_PATH = os.path.join(_AUTOMATIC_RADIOMICS_PATH, "extracted")
AUTOMATIC_FILTERED_RADIOMICS_PATH = os.path.join(_AUTOMATIC_RADIOMICS_PATH, "filtered")
AUTOMATIC_RADIOMICS_MODELS_PATH = os.path.join(_AUTOMATIC_RADIOMICS_PATH, "models")
_AUTOMATIC_BAYESIAN_RADIOMICS_PATH = os.path.join(_RADIOMICS_PATH, "automatic_bayesian")
AUTOMATIC_BAYESIAN_EXTRACTED_RADIOMICS_PATH = os.path.join(_AUTOMATIC_RADIOMICS_PATH, "extracted")
AUTOMATIC_BAYESIAN_FILTERED_RADIOMICS_PATH = os.path.join(_AUTOMATIC_RADIOMICS_PATH, "filtered")
AUTOMATIC_BAYESIAN_RADIOMICS_MODELS_PATH = os.path.join(_AUTOMATIC_RADIOMICS_PATH, "models")
_DEEP_RADIOMICS_PATH = os.path.join(_RADIOMICS_PATH, "deep")
DEEP_FILTERED_RADIOMICS_PATH = os.path.join(_DEEP_RADIOMICS_PATH, "filtered")
_MANUAL_RADIOMICS_PATH = os.path.join(_RADIOMICS_PATH, "manual")
MANUAL_EXTRACTED_RADIOMICS_PATH = os.path.join(_MANUAL_RADIOMICS_PATH, "extracted")
MANUAL_FILTERED_RADIOMICS_PATH = os.path.join(_MANUAL_RADIOMICS_PATH, "filtered")
CT_FEATURES_EXTRACTOR_PARAMS_PATH = os.path.join(_RADIOMICS_PATH, "features_extractor_params_CT.yaml")
PT_FEATURES_EXTRACTOR_PARAMS_PATH = os.path.join(_RADIOMICS_PATH, "features_extractor_params_PT.yaml")

RECORDS_PATH = os.path.join(DATA_PATH, "records")
OUTLIERS_RECORDS_PATH = os.path.join(RECORDS_PATH, "outliers")
DESCRIPTIVE_ANALYSIS_PATH = os.path.join(RECORDS_PATH, "descriptive_analyses")
EXPERIMENTS_PATH = os.path.join(RECORDS_PATH, "experiments")
CHECKPOINTS_PATH = os.path.join(EXPERIMENTS_PATH, "checkpoints")

# ID COLUMN
ID = "ID"

# CLINICAL FEATURES
AGE = Feature(column="AGE", transform=Normalization())
CLINICAL_STAGE = Feature(column="CLINICAL_STAGE", transform=MappingEncoding({"T1-T2": 0, "T3a": 1}))
GLEASON_GLOBAL = Feature(column="GLEASON_GLOBAL", transform=MappingEncoding({8: 0, 9: 0.5, 10: 1}))
GLEASON_PRIMARY = Feature(column="GLEASON_PRIMARY", transform=MappingEncoding({3: 0, 4: 0.5, 5: 1}))
GLEASON_SECONDARY = Feature(column="GLEASON_SECONDARY", transform=MappingEncoding({3: 0, 4: 0.5, 5: 1}))
PSA = Feature(column="PSA", transform=Normalization())

CLINICAL_CONTINUOUS_FEATURES = [AGE, PSA]
CLINICAL_CATEGORICAL_FEATURES = [CLINICAL_STAGE, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY]
CLINICAL_FEATURES = CLINICAL_CONTINUOUS_FEATURES + CLINICAL_CATEGORICAL_FEATURES

# RADIOMICS FEATURES
RADIOMIC_1 = Feature(column="RADIOMIC_1", transform=Normalization(), impute=False)
RADIOMIC_2 = Feature(column="RADIOMIC_2", transform=Normalization(), impute=False)
RADIOMIC_3 = Feature(column="RADIOMIC_3", transform=Normalization(), impute=False)
RADIOMIC_4 = Feature(column="RADIOMIC_4", transform=Normalization(), impute=False)
RADIOMIC_5 = Feature(column="RADIOMIC_5", transform=Normalization(), impute=False)
RADIOMIC_6 = Feature(column="RADIOMIC_6", transform=Normalization(), impute=False)

RADIOMICS_FEATURES = [RADIOMIC_1, RADIOMIC_2, RADIOMIC_3, RADIOMIC_4, RADIOMIC_5, RADIOMIC_6]

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
    evaluation_metrics=[ConcordanceIndexIPCW(), CumulativeDynamicAUC()],
    temperature=1e-2
)
CRPC_TASK = SurvivalAnalysisTask(
    event_indicator_column=CRPC,
    event_time_column=CRPC_TIME,
    criterion=NegativePartialLogLikelihood(),
    hps_tuning_metric=ConcordanceIndexCensored(),
    evaluation_metrics=[ConcordanceIndexIPCW(), CumulativeDynamicAUC()],
    temperature=1e-2
)
DEATH_TASK = SurvivalAnalysisTask(
    event_indicator_column=DEATH,
    event_time_column=DEATH_TIME,
    criterion=NegativePartialLogLikelihood(),
    hps_tuning_metric=ConcordanceIndexCensored(),
    evaluation_metrics=[ConcordanceIndexIPCW(), CumulativeDynamicAUC()],
    temperature=1e-2
)
HTX_TASK = SurvivalAnalysisTask(
    event_indicator_column=HTX,
    event_time_column=HTX_TIME,
    criterion=NegativePartialLogLikelihood(),
    hps_tuning_metric=ConcordanceIndexCensored(),
    evaluation_metrics=[ConcordanceIndexIPCW(), CumulativeDynamicAUC()],
    temperature=1e-2
)
METASTASIS_TASK = SurvivalAnalysisTask(
    event_indicator_column=METASTASIS,
    event_time_column=METASTASIS_TIME,
    criterion=NegativePartialLogLikelihood(),
    hps_tuning_metric=ConcordanceIndexCensored(),
    evaluation_metrics=[ConcordanceIndexIPCW(), CumulativeDynamicAUC()],
    temperature=1e-2
)
PN_TASK = BinaryClassificationTask(
    target_column=PN,
    decision_threshold_metric=BinaryBalancedAccuracy(),
    hps_tuning_metric=AUC(),
    evaluation_metrics=[Sensitivity(), Specificity()],
    criterion=BCEWithLogitsLoss(),
    temperature=1e-2
)

PREDICTOR_CLIP_GRAD_MAX_NORM_DICT = {BCR: 3.0, CRPC: 3.0, DEATH: 1.0, HTX: 3.0, METASTASIS: 2.0, PN: 3.0}
EXTRACTOR_CLIP_GRAD_MAX_NORM_DICT = {BCR: 10.0, CRPC: 10.0, DEATH: 5.0, HTX: 10.0, METASTASIS: 8.0, PN: 10.0}

MLP_RAD_LR_HIGH_BOUND_DICT = {BCR: 1e-2, CRPC: 5e-3, DEATH: 5e-3, HTX: 1e-2, METASTASIS: 5e-3, PN: 1e-2}
MLP_RAD_AND_CLIN_DATA_LR_HIGH_BOUND_DICT = {BCR: 1e-2, CRPC: 5e-3, DEATH: 5e-3, HTX: 1e-2, METASTASIS: 5e-3, PN: 1e-2}
CNN_LR_HIGH_BOUND_DICT = {BCR: 1e-3, CRPC: 1e-3, DEATH: 5e-4, HTX: 1e-3, METASTASIS: 5e-4, PN: 1e-3}
UNEXTRACTOR_LR_LOW_BOUND_DICT = {BCR: 1e-4, CRPC: 1e-4, DEATH: 5e-5, HTX: 1e-4, METASTASIS: 5e-5, PN: 1e-4}
UNEXTRACTOR_LR_HIGH_BOUND_DICT = {BCR: 1e-3, CRPC: 1e-3, DEATH: 5e-4, HTX: 1e-3, METASTASIS: 5e-4, PN: 1e-3}

TABLE_TASKS = [BCR_TASK, CRPC_TASK, DEATH_TASK, HTX_TASK, METASTASIS_TASK, PN_TASK]

# IMAGE TASKS
PROSTATE_SEGMENTATION_TASK = SegmentationTask(
    criterion=DiceLoss(),
    evaluation_metrics=DiceMetric(),
    organ="Prostate",
    modality="CT",
    temperature=1e-2
)

IMAGE_TASKS = [PROSTATE_SEGMENTATION_TASK]
