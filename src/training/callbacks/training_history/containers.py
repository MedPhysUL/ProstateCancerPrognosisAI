"""
    @file:              containers.py
    @Author:            Maxence Larose

    @Creation Date:     12/2022
    @Last modification: 02/2023

    @Description:       This file is used to define containers for training history.
"""

from dataclasses import dataclass, field
from typing import Dict, List, TypeAlias, Union

MeasurementsType: TypeAlias = Dict[str, Dict[str, Union[float, List[float]]]]


@dataclass
class MeasurementsContainer:
    """
    A data class defining the data structure that contains the various measurements performed during a training process.

    Elements
    --------
    multi_task_losses: MeasurementsType
        A dictionary containing multi-task losses. The keys are the names of the learning algorithm used, the values
        are dictionaries whose keys are the names of the losses while its values are a list of the losses measured each
        epoch. This list of losses can also just be a float, when the current class is used to represent a single
        epoch state instead of the entire history for example.

        Example (2 learning algorithms, 3 epochs) :
            multi_task_losses = {
                "LearningAlgorithm(0)": {
                    "MeanLoss('regularization'=False)": [0.9, 0.7, 0.6],
                    "MeanLoss('regularization'=True)": [1.1, 0.8, 0.7]
                },
                "LearningAlgorithm(1)": {
                    "MedianLoss('regularization'=False)": [1.1, 0.8, 0.7],
                    "MedianLoss('regularization'=True)": [2, 1.5, 1.1]
                },
            }

    single_task_losses: MeasurementsType
        A dictionary containing single task losses. The keys are the names of the tasks, the values are dictionaries
        whose keys are the names of the losses while its values are a list of the losses measured each epoch. This list
        of losses can also just be a float, when the current class is used to represent a single epoch state instead
        of the entire history for example.

        Example (2 tasks, 3 epochs) :
            single_task_losses = {
                "ClassificationTask('target_column'='PN')": {
                    "BinaryBalancedAccuracy('reduction'='mean')": [0.6, 0.5, 0.9]
                },
                "SegmentationTask('modality'='CT', 'organ'='Prostate')": {
                    "DICELoss('reduction'='mean')": [0.7, 0.76, 0.85]
                },
            }

    single_task_metrics : MeasurementsType
        A dictionary containing metric values. The keys are the names of the tasks, the values are dictionaries whose
        keys are the names of the metrics while its values are a list of the metrics measured each epoch. This list of
        metrics can also just be a float, when the current class is used to represent a single epoch state instead of
        the entire history for example.

        Example (2 tasks, 3 epochs) :

            metrics = {
                "ClassificationTask('target_column'='PN')": {
                    "BinaryBalancedAccuracy('reduction'='mean')": [0.6, 0.5, 0.9],
                    "AUC('reduction'='none')": [0.6, 0.7, 0.75]
                },
                "SegmentationTask('modality'='CT', 'organ'='Prostate')": {
                    "DICEMetric('reduction'='mean')": [0.7, 0.76, 0.85]
                },
            }
    """
    multi_task_losses: MeasurementsType = field(default_factory=dict)
    single_task_losses: MeasurementsType = field(default_factory=dict)
    single_task_metrics: MeasurementsType = field(default_factory=dict)


@dataclass
class HistoryContainer:
    """
    A data class defining the data structure that contains the different sets measurements.

    Elements
    --------
    train : MeasurementsContainer
        Training set measurements.
    valid : MeasurementsContainer
        Validation set measurements.
    """
    train: MeasurementsContainer = field(default_factory=MeasurementsContainer)
    valid: MeasurementsContainer = field(default_factory=MeasurementsContainer)
