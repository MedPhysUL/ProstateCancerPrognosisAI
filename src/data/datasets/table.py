"""
    @file:              table.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file contains a custom torch dataset named TableDataset. We follow
                        https://ieeexplore.ieee.org/document/8892612 setting for multi-output learning. This class
                        allows to cover the cases where some patients have a missing label for one task (or should I
                        say, for one output) while it is available for another.
"""

from __future__ import annotations
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch import cat, from_numpy, stack, Tensor
from torch.utils.data import Dataset

from ..transforms.base import Identity, Transform
from ..transforms.enums import CategoricalTransform, ContinuousTransform
from ...tasks import BinaryClassificationTask, TaskList, SurvivalAnalysisTask
from ...tasks.base import TableTask


class TableDataModel(NamedTuple):
    """
    Data element named tuple. This tuple is used to separate features (x) and targets (y) where
        - x : D-dimensional dictionary containing (N, ) tensor or array where D is the number of features.
        - y : T-dimensional dictionary containing (N, ) tensor or array where T is the number of tasks. Note that it can
            also contain (N, 2) tensor or array in the case of survival analysis, where the first column corresponds to
            events indicators and the second to events times.
    """
    x: Dict[str, Union[np.ndarray, Tensor]]
    y: Dict[str, Union[np.ndarray, Tensor]]


class Feature(NamedTuple):
    """
    Feature column named tuple. This tuple is used to store the name of a column and its associated transform.

    Elements
    --------
    column : str
        Name of the column.
    transform : Optional[Transform]
        Transform to apply to the column. If None, the identity transform is used.
    """
    column: str
    transform: Optional[Transform] = None


class TableDataset(Dataset):
    """
    A custom dataset class used to perform experiments on tabular data. It is designed to be used with a
    torch.utils.data.DataLoader. It allows to cover the cases where some patients have a missing label for one task
    (or should I say, for one output) while it is available for another. It also allows to perform experiments with
    multiple tasks.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            ids_col: str,
            tasks: Union[TableTask, TaskList, List[TableTask]],
            cont_features: Optional[List[Feature]] = None,
            cat_features: Optional[List[Feature]] = None,
            to_tensor: bool = False
    ):
        """
        Sets protected and public attributes of our custom dataset class.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with the original data.
        ids_col : str
            Name of the column containing the patient ids.
        tasks : Union[TableTask, TaskList, List[TableTask]]
            List of tasks.
        cont_features : Optional[List[Feature]]
            List of column names associated with continuous feature data.
        cat_features : Optional[List[Feature]]
            List of column names associated with categorical feature data.
        to_tensor : bool
            Whether we want the features and targets in tensors. False for numpy arrays.
        """
        super(TableDataset).__init__()

        # Validation of inputs
        if cont_features is None and cat_features is None:
            raise ValueError("At least a list of continuous columns or a list of categorical columns must be provided.")

        for features, cont in zip([cont_features, cat_features], [True, False]):
            self._check_features_validity(df, features, cont)

        self._tasks = TaskList(tasks)
        assert all(isinstance(task, TableTask) for task in TaskList(self._tasks)), (
            f"All tasks must be instances of 'TableTask'."
        )

        # Set default protected attributes
        self._cont_features, self._cat_features = cont_features, cat_features
        self._cat_features_cols, self._cat_features_idx = [f.column for f in cat_features], []
        self._cont_features_cols, self._cont_features_idx = [f.column for f in cont_features], []
        self._ids_col = ids_col
        self._ids = list(df[ids_col].values)
        self._ids_to_row_idx = {id_: i for i, id_ in enumerate(self._ids)}
        self._n = df.shape[0]
        self._original_data = df
        self._target_cols = [task.target_column for task in self._tasks.table_tasks]
        self._to_tensor = to_tensor
        self._train_mask, self._valid_mask, self._test_mask = [], None, []
        self._x_cat, self._x_cont = None, None
        self._y = self._initialize_targets(self._tasks, to_tensor)

        # Define protected feature "getter" method
        self._x = self._define_feature_getter()

        # We set a "getter" method to get modes of categorical columns and we also extract encodings
        self._get_modes = self._define_categorical_stats_getter()

        # We set a "getter" method to get mu ans std of continuous columns
        self._get_mu_and_std = self._define_numerical_stats_getter()

        # We set two "setter" methods to preprocess available data after masks update
        self._set_numerical = self._define_numerical_data_setter()
        self._set_categorical = self._define_categorical_data_setter()

        # We update current training mask with all the data
        self.update_masks(list(range(self._n)), [], [])

    def __len__(self) -> int:
        return self._n

    def __getitem__(
            self,
            idx: Union[int, List[int]]
    ) -> TableDataModel:
        """
        Gets dataset item. In the multi-output learning setting, the output variables, i.e the targets, share the same
        training features (See https://ieeexplore.ieee.org/document/8892612).

        Parameters
        ----------
        idx : Union[int, List[int]]
            Indices of data to get.

        Returns
        -------
        item : TableDataModel
            A data element.
        """
        x = dict((col, self.x[idx, i]) for i, col in enumerate(self.features_cols))
        y = dict((col, y_task[idx]) for col, y_task in self.y.items())

        return TableDataModel(x=x, y=y)

    @property
    def cat_features(self) -> List[Feature]:
        return self._cat_features

    @property
    def cat_features_cols(self) -> List[str]:
        return self._cat_features_cols

    @property
    def cat_features_idx(self) -> List[int]:
        return self._cat_features_idx

    @property
    def columns(self) -> List[str]:
        return self.features_cols + self.target_cols

    @property
    def cont_features(self) -> List[Feature]:
        return self._cont_features

    @property
    def cont_features_cols(self) -> List[str]:
        return self._cont_features_cols

    @property
    def cont_features_idx(self) -> List[int]:
        return self._cont_features_idx

    @property
    def features_cols(self) -> List[str]:
        return self.cont_features_cols + self.cat_features_cols

    @property
    def ids(self) -> List[str]:
        return self._ids

    @property
    def ids_col(self) -> str:
        return self._ids_col

    @property
    def ids_to_row_idx(self) -> Dict[str, int]:
        return self._ids_to_row_idx

    @property
    def original_data(self) -> pd.DataFrame:
        return self._original_data

    @property
    def target_cols(self) -> List[str]:
        return self._target_cols

    @property
    def tasks(self) -> TaskList:
        return self._tasks

    @property
    def test_mask(self) -> List[int]:
        return self._test_mask

    @property
    def to_tensor(self) -> bool:
        return self._to_tensor

    @property
    def train_mask(self) -> List[int]:
        return self._train_mask

    @property
    def valid_mask(self) -> Optional[List[int]]:
        return self._valid_mask

    @property
    def x(self) -> Union[Tensor, np.array]:
        return self._x()

    @property
    def x_cat(self) -> Optional[Union[np.array, Tensor]]:
        return self._x_cat

    @property
    def x_cont(self) -> Optional[Union[np.array, Tensor]]:
        return self._x_cont

    @property
    def y(self) -> Dict[str, Union[np.array, Tensor]]:
        return self._y

    def _categorical_setter(
            self,
            modes: pd.Series
    ) -> None:
        """
        Fill missing values of categorical data according to the modes in the training set and then encodes categories
        using the same ordinal encoding as in the training set.

        Parameters
        ----------
        modes : pd.Series
            Modes in the training set.
        """
        # We apply an ordinal encoding to categorical columns
        x_cat = self._preprocess_categoricals(
            df=self._original_data[self._cat_features_cols].copy(),
            features=self._cat_features,
            mode=modes
        )
        self._x_cat = x_cat.to_numpy(dtype=float)

    @staticmethod
    def _preprocess_categoricals(
            df: pd.DataFrame,
            features: List[Feature],
            mode: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Applies all categorical transforms to a dataframe containing only continuous data

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing all data.
        features : List[Feature]
            List of categorical features.
        mode : Optional[pd.Series]
            Pandas series with modes of columns.

        Returns
        -------
        df : pd.DataFrame
            Pandas dataframe.
        """
        if mode is not None:
            df = df.fillna(mode)
        else:
            df = df.fillna(df.mode().iloc[0])

        for feature in features:
            df[feature.column] = feature.transform(df=df[feature.column])

        return df

    def _define_categorical_data_setter(self) -> Callable:
        """
        Defines the function used to set categorical data after masks update.

        Returns
        -------
        set_categorical : Callable
            The function used to set categorical data after masks update.
        """
        # If there is no categorical columns
        if self._cat_features_cols is None:

            def set_categorical(modes: Optional[pd.Series]) -> None:
                pass

            return set_categorical

        else:
            if self._to_tensor:

                def set_categorical(modes: Optional[pd.Series]) -> None:
                    self._categorical_setter(modes)
                    self._x_cat = from_numpy(self._x_cat).long()

                return set_categorical

            return self._categorical_setter

    def _define_categorical_stats_getter(self) -> Callable:
        """
        Defines the function used to extract the modes of categorical columns.

        Returns
        -------
        get_modes, encodings : Tuple[Callable, Dict[str, Dict[str, int]]]
            The function used to extract the modes of categorical columns paired with categorical variables encodings.
        """
        # If there is not categorical column
        if self._cat_features_cols is None:

            def get_modes(df: Optional[pd.DataFrame]) -> None:
                return None

        else:
            # Make sure that categorical data in the original dataframe is in the correct format
            cols = self._cat_features_cols
            self._original_data[cols] = self._original_data[cols].astype('category')

            def get_modes(df: pd.DataFrame) -> pd.Series:
                return df[cols].mode().iloc[0]

        return get_modes

    def _define_feature_getter(self) -> Callable:
        """
        Defines the method used to extract the features (processed data) for training.

        Returns
        -------
        feature_getter : Callable
            Function used to extract features from data.
        """
        if self._cont_features_cols is None:

            # Only categorical column idx
            self._cat_features_idx = list(range(len(self._cat_features_cols)))

            # Only categorical feature extracted by the getter
            def x() -> Union[Tensor, np.array]:
                return self.x_cat

        elif self._cat_features_cols is None:

            # Only continuous column idx
            self._cont_features_idx = list(range(len(self._cont_features_cols)))

            # Only continuous features extracted by the getter
            def x() -> Union[Tensor, np.array]:
                return self.x_cont

        else:

            # Continuous and categorical column idx
            nb_cont_features_cols = len(self._cont_features_cols)
            self._cont_features_idx = list(range(nb_cont_features_cols))
            self._cat_features_idx = list(range(
                nb_cont_features_cols,
                nb_cont_features_cols + len(self._cat_features_cols)
            ))

            # Continuous and categorical features extracted by the getter
            if not self._to_tensor:
                def x() -> Union[Tensor, np.array]:
                    return np.concatenate((self.x_cont, self.x_cat), axis=1)
            else:
                def x() -> Union[Tensor, np.array]:
                    return cat((self.x_cont, self.x_cat), dim=1)

        return x

    def _define_numerical_data_setter(self) -> Callable:
        """
        Defines the function used to set numerical continuous data after masks update.

        Returns
        -------
        numerical_data_setter : Callable
             Function used to set numerical continuous data after masks update.
        """
        # If there is no continuous column
        if self._cont_features_cols is None:

            def set_numerical(mu: Optional[pd.Series], std: Optional[pd.Series]) -> None:
                pass

            return set_numerical

        else:
            if self._to_tensor:

                def set_numerical(mu: Optional[pd.Series], std: Optional[pd.Series]) -> None:
                    self._numerical_setter(mu, std)
                    self._x_cont = from_numpy(self._x_cont).float()

                return set_numerical

            return self._numerical_setter

    def _define_numerical_stats_getter(self) -> Callable:
        """
        Defines the function used to extract the mean and the standard deviations of numerical columns in a dataframe.

        Returns
        -------
        numerical_stats_getter : Callable
            Function used to extract the mean and the standard deviations of numerical columns in a dataframe.
        """
        # If there is no continuous column
        if self._cont_features_cols is None:

            def get_mu_and_std(df: pd.DataFrame) -> Tuple[None, None]:
                return None, None
        else:
            # Make sure that numerical data in the original dataframe is in the correct format
            self._original_data[self._cont_features_cols] = self._original_data[self._cont_features_cols].astype(float)

            def get_mu_and_std(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
                return df[self._cont_features_cols].mean(), df[self._cont_features_cols].std()

        return get_mu_and_std

    def _set_scaling_factors(self):
        """
        Sets scaling factor of all binary classification tasks.
        """
        for task in self.tasks.binary_classification_tasks:
            # We set the scaling factors of all classification metrics
            for metric in task.metrics:
                metric.update_scaling_factor(y_train=self.y[task.name][self.train_mask])

            # We set the scaling factor of the criterion
            if task.criterion:
                task.criterion.update_scaling_factor(y_train=self.y[task.name][self.train_mask])

    def _numerical_setter(
            self,
            mu: pd.Series,
            std: pd.Series
    ) -> None:
        """
        Fills missing values of numerical continuous data according according to the means of the training mask and
        then normalizes continuous data using the means and the standard deviations of the training mask.

        Parameters
        ----------
        mu : pd.Series
            Means of the numerical column according to the training mask.
        std : pd.Series
            Standard deviations of the numerical column according to the training mask.
        """
        # We fill missing with means and normalize the data
        x_cont = self._preprocess_continuous(
            df=self._original_data[self._cont_features_cols].copy(),
            features=self._cont_features,
            mean=mu,
            std=std
        )

        # We apply the basis function
        self._x_cont = x_cont.to_numpy(dtype=float)

    @staticmethod
    def _preprocess_continuous(
            df: pd.DataFrame,
            features: List[Feature],
            mean: Optional[pd.Series] = None,
            std: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Applies all continuous transforms to a dataframe containing only continuous data.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing all data.
        features : List[Feature]
            List of continuous features.
        mean : Optional[pd.Series]
            Pandas series with mean.
        std : Optional[pd.Series]
            Pandas series with standard deviations

        Returns
        -------
        preprocessed_dataframe : pd.DataFrame
            Dataframe containing data on which all continuous transforms have been applied.
        """
        if mean is not None:
            df = df.fillna(mean)
        else:
            df = df.fillna(df.mean())

        for feature in features:
            column = feature.column
            df[column] = feature.transform(df=df[column], mean=mean[column], std=std[column])

        return df

    def get_imputed_dataframe(
            self
    ) -> pd.DataFrame:
        """
        Returns a copy of the original pandas dataframe where missing values are imputed according to the training mask.

        Returns
        -------
        imputed_df : pd.DataFrame
            Copy of the original pandas dataframe where missing values are imputed according to the training mask.
        """
        imputed_df = self.original_data.copy()

        if self._cont_features_cols is not None:
            imputed_df[self._cont_features_cols] = np.array(self._x_cont)
        if self._cat_features_cols is not None:
            imputed_df[self._cat_features_cols] = np.array(self._x_cat)

        return imputed_df

    def current_train_stats(
            self
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """
        Returns the current statistics and encodings related to the training data.

        Returns
        -------
        (mu, std, modes) : Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]
            Tuple containing the current statistics and encodings related to the training data :
                mu : Optional[pd.Series]
                    Means of the numerical columns according to the training mask.
                std : Optional[pd.Series]
                    Standard deviations of the numerical columns according to the training mask.
                modes : Optional[pd.Series]
                    Modes of the categorical columns according to the training mask.
        """
        # We extract the current training data
        train_data = self._original_data.iloc[self._train_mask]

        # We compute the current values of mu, std, modes and encodings
        mu, std = self._get_mu_and_std(train_data)
        modes = self._get_modes(train_data)

        return mu, std, modes

    def update_masks(
            self,
            train_mask: List[int],
            test_mask: List[int],
            valid_mask: Optional[List[int]] = None
    ) -> None:
        """
        Updates the train, valid and test masks and then preprocesses the data available according to the current
        statistics of the training data.

        Parameters
        ----------
        train_mask : List[int]
            List of idx in the training set.
        test_mask : List[int]
            List of idx in the test set.
        valid_mask : Optional[List[int]]
            List of idx in the valid set.
        """
        # We set the new masks values
        self._train_mask, self._test_mask = train_mask, test_mask
        self._valid_mask = valid_mask if valid_mask is not None else []

        # We compute the current values of mu, std, modes and encodings
        mu, std, modes = self.current_train_stats()

        # We update the data that will be available via __get_item__
        self._set_numerical(mu, std)
        self._set_categorical(modes)

        # We set the classification tasks scaling factors
        self._set_scaling_factors()

    def _initialize_targets(
            self,
            tasks: TaskList,
            target_to_tensor: bool
    ) -> Union[np.array, Tensor]:
        """
        Sets the targets according to the task and the choice of container.

        Parameters
        ----------
        tasks : TaskList
            List of tasks.
        target_to_tensor : bool
            True if we want the targets to be in a tensor, false for numpy array.

        Returns
        -------
        targets : Union[np.array, Tensor]
            Targets in a proper format.
        """
        targets = {}
        for task in tasks:
            t = self.original_data[task.target_column].to_numpy(dtype=float)

            if (not isinstance(task, BinaryClassificationTask)) and target_to_tensor:
                t = from_numpy(t).float()
            elif isinstance(task, (BinaryClassificationTask, SurvivalAnalysisTask)):
                if target_to_tensor:
                    t = from_numpy(t).long()
                else:
                    t = t.astype(int)

            if isinstance(task, SurvivalAnalysisTask):
                event_time = self.original_data[task.event_time_column].to_numpy(dtype=float)
                if target_to_tensor:
                    event_time = from_numpy(event_time).float()
                    t = stack([t, event_time], dim=1)
                else:
                    t = np.stack([t, event_time], axis=1)

            targets[task.name] = t

        return targets

    @staticmethod
    def _check_features_validity(
            df: pd.DataFrame,
            features: Optional[List[Feature]] = None,
            cont: bool = True
    ) -> None:
        """
        Checks if the columns are all in the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe with original data.
        features : Optional[List[Feature]]
            List of features.
        cont : bool
            Whether the features are continuous or not.
        """
        if features is not None:
            dataframe_columns = list(df.columns.values)
            for f in features:
                if f.column not in dataframe_columns:
                    raise ValueError(f"Column {f.column} is not part of the given dataframe")

                if f.transform:
                    if cont:
                        assert isinstance(f.transform, tuple(t.value for t in ContinuousTransform)), (
                            f"Transform {f.transform} is not a continuous transform. Available transforms are: "
                            f"{[t.name for t in ContinuousTransform]}."
                        )
                    else:
                        assert isinstance(f.transform, tuple(t.value for t in CategoricalTransform)), (
                            f"Transform {f.transform} is not a categorical transform. Available transforms are: "
                            f"{[t.name for t in CategoricalTransform]}."
                        )
                else:
                    f.transform = Identity()
