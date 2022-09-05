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
from torch import from_numpy, isnan, is_tensor, Tensor, where, zeros

from src.utils.reductions import GeometricMean, Identity, Mean, Reduction


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
            direction: Direction,
            name: str,
            reduction: Optional[Reduction],
            n_digits: int = 5
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        direction : Direction
            Whether the metric needs to be "maximize" or "minimize".
        name : str
            Name of the metric.
        reduction : Optional[Reduction]
            Reduction method to use.
        n_digits : int
            Number of digits kept.
        """
        if direction not in Direction:
            raise ValueError(f"direction must be in {list(i.value for i in Direction)}")

        if reduction is None:
            reduction = Identity()

        # Protected attributes
        self._direction = direction
        self._name = name
        self._reduction = reduction
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
    def direction(self) -> Direction:
        return self._direction

    @property
    def name(self) -> str:
        return f"{self._reduction.name}_{self._name}"

    @property
    def reduction(self) -> Optional[Reduction]:
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
            direction: Direction,
            name: str,
            reduction: Optional[Reduction] = None,
            n_digits: int = 5
    ):
        """
        Sets protected attributes using parent's constructor

        Parameters
        ----------
        direction : Direction
            Whether the metric needs to be "maximize" or "minimize".
        name : str
            Name of the metric.
        reduction : Optional[Reduction]
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

        return round(self._reduction(self._compute_metric(pred, targets)), self.n_digits)

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
            Score.
        """
        raise NotImplementedError


class BinaryClassificationMetric(Metric):
    """
    An Abstract class that represents the skeleton of callable classes to use as classification metrics
    """

    def __init__(
            self,
            direction: Direction,
            name: str,
            reduction: Optional[Reduction] = None,
            weight: float = None,
            n_digits: int = 5
    ):
        """
        Sets protected attributes using parent's constructor

        Parameters
        ----------
        direction : Direction
            Whether the metric needs to be "maximize" or "minimize".
        name : str
            Name of the metric.
        reduction : Optional[Reduction]
            Reduction method to use.
        weight : Optional[float]
            The weight attributed to class 1 (in [0, 1]).
        n_digits : int
            Number of digits kept.
        """
        self._threshold = 0.5
        if weight is not None:
            if not (0 < weight < 1):
                raise ValueError("The weight parameter must be included in range [0, 1]")

        self._weight = weight

        super().__init__(direction=direction, name=name, reduction=reduction, n_digits=n_digits)

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    @property
    def weight(self) -> Optional[float]:
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

        return round(self._reduction(self._compute_metric(pred, targets, thresh)), self.n_digits)

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
        scaling_factor : float]
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
            Metric score.
        """
        raise NotImplementedError


class SegmentationMetric(Metric):
    """
    An abstract class that represents the skeleton of callable classes to use as segmentation metrics.
    """

    def __init__(
            self,
            direction: Direction,
            name: str,
            reduction: Reduction = Mean(),
            n_digits: int = 5
    ):
        """
        Sets protected attributes using parent's constructor

        Parameters
        ----------
        direction : Direction
            Whether the metric needs to be "maximize" or "minimize".
        name : str
            Name of the metric.
        reduction : Reduction
            Reduction method to use.
        n_digits : int
            Number of digits kept.
        """
        super().__init__(direction=direction, name=name, reduction=reduction, n_digits=n_digits)

    def __call__(
            self,
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor],
            reduction: Optional[Reduction] = None,
    ) -> float:
        """
        Converts inputs to tensors than computes the metric and applies rounding.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N, X, Y, Z) tensor or array with predicted labels.
        targets : Union[np.array, Tensor]
            (N, X, Y, Z) tensor or array with ground truth
        reduction : Optional[Reduction]
            Reduction method to use.

        Returns
        -------
        metric : float
            Rounded metric score.
        """
        if reduction is None:
            reduction = self._reduction
        else:
            reduction = reduction

        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return round(reduction(self._compute_metric(pred, targets)), self.n_digits)

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
    ) -> Tensor:
        """
        Computes the metric score.

        Parameters
        ----------
        pred : Tensor
            (N, X, Y, Z) tensor with predicted labels
        targets : Tensor
            (N, X, Y, Z) tensor with ground truth

        Returns
        -------
        metric_score : Tensor
            Score.
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
        super().__init__(direction=Direction.MAXIMIZE, name="AUC", n_digits=n_digits)

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
            n_digits: int = 5
    ):
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        n_digits : int
            Number of digits kept for the score.
        """
        super().__init__(direction=Direction.MAXIMIZE, name="Accuracy", reduction=Mean(), n_digits=n_digits)

    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor,
            thresh: float
    ) -> Tensor:
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
        metric : Tensor
            Score.
        """
        pred_labels = (pred >= thresh).float()

        return (pred_labels == targets).float()


class BinaryBalancedAccuracy(BinaryClassificationMetric):
    """
    Callable class that computes balanced accuracy using confusion matrix.
    """

    def __init__(
            self,
            reduction: Reduction = Mean(),
            n_digits: int = 5
    ):
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        reduction : Reduction
            "Mean()" for (TPR + TNR)/2 or "GeometricMean()" for sqrt(TPR*TNR)
        n_digits : int
            Number of digits kept for the score.
        """
        if not isinstance(reduction, (Mean, GeometricMean)):
            raise ValueError(f"Reduction must be in {[Mean, GeometricMean]}.")

        super().__init__(direction=Direction.MAXIMIZE, name="BalancedAcc", reduction=reduction, n_digits=n_digits)

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
            Score.
        """
        # We get confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets, thresh)

        # We get TNR and TPR
        correct_rates = conf_mat.diag() / conf_mat.sum(dim=1)

        return correct_rates


class DICE(SegmentationMetric):
    """
    Callable class that computes the DICE.
    """

    def __init__(
            self,
            reduction: Reduction = Mean(),
            n_digits: int = 5
    ) -> None:
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        reduction : Reduction
            Reduction method to use.
        n_digits : int
            Number of digits kept for the score.
        """
        super().__init__(direction=Direction.MAXIMIZE, name="DICE", reduction=reduction, n_digits=n_digits)

    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor,
    ) -> Tensor:
        """
        Returns the average of the DICE score.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted probabilities of being in class 1
        targets : Tensor
            (N,) tensor with ground truth

        Returns
        -------
        metric : Tensor
            Score.
        """
        metric = DiceMetric()

        return metric(y_pred=pred, y=targets)  # .cpu().data.numpy().flatten().tolist()
