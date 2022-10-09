"""
    @file:              score_metrics.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file is used to define the metrics used to measure models' performance.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple, Union

from monai.metrics import DiceMetric
import numpy as np
from sklearn.metrics import roc_auc_score
from torch import from_numpy, isnan, is_tensor, mean, pow, prod, sum, Tensor, where, zeros

from src.utils.reductions import MetricReduction


class Direction(Enum):
    """
    Custom enum for optimization directions
    """
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

    def __iter__(self):
        return iter([self.MAXIMIZE, self.MINIMIZE])


class Metric(ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as optimization metrics.
    """

    def __init__(
            self,
            direction: Union[Direction, str],
            name: str,
            reduction: Union[MetricReduction, str],
            n_digits: int = 5
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        direction : Union[Direction, str]
            Whether the metric needs to be "maximize" or "minimize".
        name : str
            Name of the metric.
        reduction : Union[MetricReduction, str]
            Reduction method to use.
        n_digits : int
            Number of digits kept.
        """
        # Protected attributes
        self._direction = Direction(direction).value
        self._name = name
        self._reduction = MetricReduction(reduction).value
        self._n_digits = n_digits

    @abstractmethod
    def __call__(
            self,
            *args,
            **kwargs
    ) -> float:
        """
        Gets metric value.

        Returns
        -------
        metric : float
            Rounded metric score.
        """
        raise NotImplementedError

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def name(self) -> str:
        return f"{self._reduction}_{self._name}"

    def perform_reduction(
            self,
            x: Union[float, Tensor],
            reduction: Optional[Union[MetricReduction, str]] = None
    ) -> float:
        """
        Gets metric value.

        Parameters
        ----------
        x : Union[float, Tensor]
            Float or (N, 1) tensor.
        reduction : Optional[Union[MetricReduction, str]]
            Reduction method to use. If None, we use self.reduction.

        Returns
        -------
        metric : float
            Rounded metric score.
        """
        if reduction is None:
            reduction = self.reduction
        else:
            reduction = MetricReduction(reduction).value

        if reduction == MetricReduction.NONE.value:
            if isinstance(x, float):
                return x
            else:
                return x.item()
        elif reduction == MetricReduction.MEAN.value:
            return mean(x).item()
        elif reduction == MetricReduction.SUM.value:
            return sum(x).item()
        elif reduction == MetricReduction.GEOMETRIC_MEAN.value:
            return pow(prod(x), exponent=(1 / x.shape[0])).item()

    @property
    def reduction(self) -> str:
        return self._reduction

    @property
    def n_digits(self) -> int:
        return self._n_digits


class RegressionMetric(Metric):
    """
    An abstract class that represents the skeleton of callable classes to use as regression metrics.
    """

    def __init__(
            self,
            direction: Union[Direction, str],
            name: str,
            reduction: Union[MetricReduction, str],
            n_digits: int = 5
    ):
        """
        Sets protected attributes using parent's constructor

        Parameters
        ----------
        direction : Union[Direction, str]
            Whether the metric needs to be "maximize" or "minimize".
        name : str
            Name of the metric.
        reduction : Union[MetricReduction, str]
            Reduction method to use.
        n_digits : int
            Number of digits kept.
        """
        super().__init__(direction=direction, name=name, reduction=reduction, n_digits=n_digits)

    def __call__(
            self,
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor]
    ) -> float:
        """
        Converts inputs to tensors than computes the metric and applies rounding.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N,) tensor or array with predicted labels.
        targets : Union[np.array, Tensor]
            (N,) tensor or array with ground truth

        Returns
        -------
        metric : float
            Rounded metric score.
        """
        nonmissing_targets_idx = self.get_idx_of_nonmissing_targets(targets)
        targets, pred = targets[nonmissing_targets_idx], pred[nonmissing_targets_idx]

        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return round(self.perform_reduction(self._compute_metric(pred, targets)), self.n_digits)

    @staticmethod
    def get_idx_of_nonmissing_targets(
            y: Union[Tensor, np.array]
    ) -> List[int]:
        """
        Gets the idx of the nonmissing targets in the given array or tensor.

        Parameters
        ----------
        y : Union[Tensor, np.array]
            (N,) tensor or array with targets.

        Returns
        -------
        idx : List[int]
            Index.
        """
        if isinstance(y, Tensor):
            idx = where(~isnan(y))
        else:
            idx = np.where(~np.isnan(y))

        return idx[0].tolist()

    @staticmethod
    def convert_to_tensors(
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Converts inputs to tensors.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N,) tensor or array containing predictions.
        targets : Union[np.array, Tensor]
            (N,) tensor or array containing ground truth.

        Returns
        -------
        pred, targets : Tuple[Tensor, Tensor]
            (N,) tensor, (N,) tensor
        """
        if not is_tensor(pred):
            return from_numpy(pred).float(), from_numpy(targets).float()
        else:
            return pred, targets

    @abstractmethod
    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor
    ) -> Union[float, Tensor]:
        """
        Computes the metric score.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted labels
        targets : Tensor
            (N,) tensor with ground truth

        Returns
        -------
        metric_score : Union[float, Tensor]
            Score as a float or a (N, 1) tensor.
        """
        raise NotImplementedError


class BinaryClassificationMetric(Metric):
    """
    An Abstract class that represents the skeleton of callable classes to use as classification metrics
    """

    def __init__(
            self,
            direction: Union[Direction, str],
            name: str,
            reduction: Union[MetricReduction, str],
            threshold: float = 0.5,
            weight: float = 0.5,
            n_digits: int = 5
    ):
        """
        Sets protected attributes using parent's constructor

        Parameters
        ----------
        direction : Union[Direction, str]
            Whether the metric needs to be "maximize" or "minimize".
        name : str
            Name of the metric.
        reduction : Union[MetricReduction, str]
            Reduction method to use.
        threshold : float
            The threshold used to classify a sample in class 1.
        weight : float
            The weight attributed to class 1 (in [0, 1]).
        n_digits : int
            Number of digits kept.
        """
        if not (0 < weight < 1):
            raise ValueError("The weight parameter must be included in range [0, 1]")

        self._scaling_factor = None
        self._threshold = threshold
        self._weight = weight

        super().__init__(direction=direction, name=name, reduction=reduction, n_digits=n_digits)

    @property
    def scaling_factor(self) -> Optional[float]:
        return self._scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, scaling_factor: float):
        self._scaling_factor = scaling_factor

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    @property
    def weight(self) -> float:
        return self._weight

    def __call__(
            self,
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor],
            thresh: Optional[float] = None
    ) -> float:
        """
        Converts inputs to tensors, applies softmax if shape is different than expected and than computes the metric
        and applies rounding.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N,) tensor or array with predicted labels.
        targets : Union[np.array, Tensor]
            (N,) tensor or array with ground truth
        thresh : float
            The threshold used to classify a sample in class 1.

        Returns
        -------
        metric : float
            Rounded metric score.
        """
        nonmissing_targets_idx = self.get_idx_of_nonmissing_targets(targets)
        targets, pred = targets[nonmissing_targets_idx], pred[nonmissing_targets_idx]

        if thresh is None:
            thresh = self._threshold

        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return round(self.perform_reduction(self._compute_metric(pred, targets, thresh)), self.n_digits)

    @staticmethod
    def get_idx_of_nonmissing_targets(
            y: Union[Tensor, np.array]
    ) -> List[int]:
        """
        Gets the idx of the nonmissing targets in the given array or tensor.

        Parameters
        ----------
        y : Union[Tensor, np.array]
            (N,) tensor or array with targets.

        Returns
        -------
        idx : List[int]
            Index.
        """
        if isinstance(y, Tensor):
            idx = where(y >= 0)
        else:
            idx = np.where(y >= 0)

        return idx[0].tolist()

    def get_scaling_factor(
            self,
            y_train: Union[np.array, Tensor]
    ) -> float:
        """
        Computes the scaling factor that needs to be apply to the weight of samples in the class 1.

        We need to find alpha that satisfies :
            (alpha*n1)/n0 = w/(1-w)
        Which gives the solution:
            alpha = w*n0/(1-w)*n1

        Parameters
        ----------
        y_train : Union[Tensor, np.array]
            (N_train, ) tensor or array with targets used for training.

        Returns
        -------
        scaling_factor : float
            Positive scaling factors.
        """
        y_train = y_train[self.get_idx_of_nonmissing_targets(y_train)]

        # Otherwise we return samples' weights in the appropriate format
        n1 = y_train.sum()              # number of samples with label 1
        n0 = y_train.shape[0] - n1      # number of samples with label 0

        return (n0/n1)*(self.weight/(1-self.weight))

    @staticmethod
    def convert_to_tensors(
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Converts predictions to float (since they are probabilities) and ground truth to long.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N,) tensor with predicted probabilities of being in class 1.
        targets : Union[np.array, Tensor]
            (N,) tensor with ground truth.

        Returns
        -------
        pred, targets : Tuple[Tensor, Tensor]
            (N,) tensor, (N,) tensor
        """
        if not is_tensor(pred):
            return from_numpy(pred).float(), from_numpy(targets).long()
        else:
            return pred, targets

    def get_confusion_matrix(
            self,
            pred_proba: Tensor,
            targets: Tensor,
            thresh: Optional[float] = None
    ) -> Tensor:
        """
        Gets the confusion matrix.

        Parameters
        ----------
        pred_proba : Tensor
            (N,) tensor with with predicted probabilities of being in class 1.
        targets : Tensor
            (N,) tensor with ground truth.
        thresh : Optional[float]
            Probability threshold that must be reach by a sample to be classified into class 1.

        Returns
        -------
        confusion_matrix : Tensor
            (2,2) tensor
        """
        if thresh is None:
            thresh = self._threshold

        # We initialize an empty confusion matrix
        conf_matrix = zeros(2, 2)

        # We fill the confusion matrix
        pred_labels = (pred_proba >= thresh).long()
        for t, p in zip(targets, pred_labels):
            conf_matrix[t, p] += 1

        return conf_matrix

    @abstractmethod
    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor,
            thresh: float
    ) -> Union[float, Tensor]:
        """
        Computes the metric score.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted probabilities of being in class 1.
        targets : Tensor
            (N,) tensor with ground truth.
        thresh : float
            Probability threshold that must be reach by a sample to be classified into class 1.

        Returns
        -------
        metric_score : Union[float, Tensor]
            Score as a float or a (N, 1) tensor.
        """
        raise NotImplementedError


class SegmentationMetric(Metric):
    """
    An abstract class that represents the skeleton of callable classes to use as segmentation metrics.
    """

    def __init__(
            self,
            direction: Union[Direction, str],
            name: str,
            reduction: Union[MetricReduction, str],
            n_digits: int = 5
    ):
        """
        Sets protected attributes using parent's constructor

        Parameters
        ----------
        direction : Union[Direction, str]
            Whether the metric needs to be "maximize" or "minimize".
        name : str
            Name of the metric.
        reduction : Union[MetricReduction, str]
            Reduction method to use.
        n_digits : int
            Number of digits kept.
        """
        super().__init__(direction=direction, name=name, reduction=reduction, n_digits=n_digits)

    def __call__(
            self,
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor],
            reduction: Optional[Union[MetricReduction, str]] = None
    ) -> float:
        """
        Converts inputs to tensors than computes the metric and applies rounding.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N, X, Y, Z) tensor or array with predicted labels.
        targets : Union[np.array, Tensor]
            (N, X, Y, Z) tensor or array with ground truth
        reduction : Optional[Union[MetricReduction, str]]
            Reduction method to use. If None, we use self.reduction.

        Returns
        -------
        metric : float
            Rounded metric score.
        """
        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return round(self._compute_metric(pred, targets, reduction), self.n_digits)

    @staticmethod
    def convert_to_tensors(
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Converts inputs to tensors.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N, X, Y, Z) tensor or array containing predictions.
        targets : Union[np.array, Tensor]
            (N, X, Y, Z) tensor or array containing ground truth.

        Returns
        -------
        pred, targets : Tuple[Tensor, Tensor]
            (N, X, Y, Z) tensor, (N, X, Y, Z) tensor
        """
        if not is_tensor(pred):
            return from_numpy(pred).float(), from_numpy(targets).float()
        else:
            return pred, targets

    @abstractmethod
    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor,
            reduction: Optional[Union[MetricReduction, str]]
    ) -> float:
        """
        Computes the metric score.

        Parameters
        ----------
        pred : Tensor
            (B, C, X, Y, Z) tensor with predicted labels
        targets : Tensor
            (B, C, X, Y, Z) tensor with ground truth
        reduction : Optional[Union[MetricReduction, str]]
            Reduction method to use.

        Returns
        -------
        metric_score : float
            Score as a float.
        """
        raise NotImplementedError


class AUC(BinaryClassificationMetric):
    """
    Callable class that computes the AUC for ROC curve.
    """

    def __init__(
            self,
            n_digits: int = 5
    ) -> None:
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        n_digits : int
            Number of digits kept for the score.
        """
        super().__init__(direction=Direction.MAXIMIZE, name="AUC", reduction=MetricReduction.NONE, n_digits=n_digits)

    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor,
            thresh: float
    ) -> float:
        """
        Returns the AUC for ROC curve.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted probabilities of being in class 1
        targets : Tensor
            (N,) tensor with ground truth
        thresh : Tensor
            Probability threshold that must be reach by a sample to be classified into class 1 (Not used here)

        Returns
        -------
        metric : float
            Score.
        """
        return roc_auc_score(targets, pred)


class BinaryAccuracy(BinaryClassificationMetric):
    """
    Callable class that computes the binary accuracy.
    """

    def __init__(
            self,
            n_digits: int = 5,
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    ):
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        n_digits : int
            Number of digits kept for the score.
        reduction : Union[MetricReduction, str]
            Reduction method to use.
        """
        super().__init__(direction=Direction.MAXIMIZE, name="Accuracy", reduction=reduction, n_digits=n_digits)

        if self.reduction not in (MetricReduction.MEAN.value, MetricReduction.SUM.value):
            raise ValueError(f"Unsupported reduction: {self.reduction}, available options are ['mean', 'sum'].")

    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor,
            thresh: float
    ) -> Tensor:
        """
        Returns the binary accuracy.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted probabilities of being in class 1
        targets : Tensor
            (N,) tensor with ground truth
        thresh : Tensor
            Probability threshold that must be reach by a sample to be classified into class 1 (Not used here)

        Returns
        -------
        metric : Tensor
            (N, 1) tensor.
        """
        pred_labels = (pred >= thresh).float()

        return (pred_labels == targets).float()


class BinaryBalancedAccuracy(BinaryClassificationMetric):
    """
    Callable class that computes balanced accuracy using confusion matrix.
    """

    def __init__(
            self,
            n_digits: int = 5,
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    ):
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        n_digits : int
            Number of digits kept for the score.
        reduction : Union[MetricReduction, str]
            "Mean" for (TPR + TNR)/2 or "GeometricMean" for sqrt(TPR*TNR)
        """
        super().__init__(direction=Direction.MAXIMIZE, name="BalancedAcc", reduction=reduction, n_digits=n_digits)

        if self.reduction not in (MetricReduction.MEAN.value, MetricReduction.GEOMETRIC_MEAN.value):
            raise ValueError(f"Unsupported reduction: {self.reduction}, available options are "
                             f"['mean', 'geometric_mean'].")

    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor,
            thresh: float
    ) -> Tensor:
        """
        Returns the either (TPR + TNR)/2 or sqrt(TPR*TNR).

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted probabilities of being in class 1
        targets : Tensor
            (N,) tensor with ground truth
        thresh : Tensor
            Probability threshold that must be reach by a sample to be classified into class 1 (Not used here)

        Returns
        -------
        metric : Tensor
            (N, 1) tensor.
        """
        # We get confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets, thresh)

        # We get TNR and TPR
        correct_rates = conf_mat.diag() / conf_mat.sum(dim=1)

        return correct_rates


class DICEMetric(SegmentationMetric):
    """
    Callable class that computes the DICE.
    """

    def __init__(
            self,
            n_digits: int = 5,
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    ) -> None:
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        n_digits : int
            Number of digits kept for the score.
        reduction :  Union[MetricReduction, str]
            Reduction method to use.
        """
        super().__init__(direction=Direction.MAXIMIZE, name="DICEMetric", reduction=reduction, n_digits=n_digits)

        if self.reduction not in (MetricReduction.MEAN.value, MetricReduction.SUM.value):
            raise ValueError(f"Unsupported reduction: {self.reduction}, available options are ['mean', 'sum'].")

    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor,
            reduction: Optional[Union[MetricReduction, str]]
    ) -> float:
        """
        Returns the average of the DICE score.

        Parameters
        ----------
        pred : Tensor
            (B, C, X, Y, Z) tensor with predicted labels
        targets : Tensor
            (B, C, X, Y, Z) tensor with ground truth
        reduction : Optional[Union[MetricReduction, str]]
            Reduction method to use.

        Returns
        -------
        metric : float
            Score as a float.
        """
        metric = DiceMetric(reduction="mean")

        return self.perform_reduction(metric(y_pred=pred, y=targets), reduction)
