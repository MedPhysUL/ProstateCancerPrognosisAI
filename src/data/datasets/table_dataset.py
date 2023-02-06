"""
    @file:              table_dataset.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 01/2023

    @Description:       This file contains a custom torch dataset named TableDataset. We follow
                        https://ieeexplore.ieee.org/document/8892612 setting for multi-output learning. This class
                        allows to cover the cases where some patients have a missing label for one task (or should I
                        say, for one output) while it is available for another.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch import cat, from_numpy, stack, Tensor
from torch.utils.data import Dataset

from src.data.processing.preprocessing import preprocess_categoricals, preprocess_continuous
from src.data.processing.tools import MaskType
from src.data.processing.transforms import CategoricalTransform as CaT
from src.utils.tasks import ClassificationTask, TableTask
from src.utils.task_list import TaskList


class TableDataModel(NamedTuple):
    """
    Data element named tuple. This tuple is used to separate features (x) and targets (y) where
        - x : D-dimensional dictionary containing (N, ) tensor or array where D is the number of features.
        - y : T-dimensional dictionary containing (N, ) tensor or array where T is the number of tasks.
    """
    x: Dict[str, Union[np.ndarray, Tensor]]
    y: Dict[str, Union[np.ndarray, Tensor]]


class TableDataset(Dataset):
    """
    A custom dataset class used to perform experiments on tabular data.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            ids_col: str,
            tasks: Union[TableTask, TaskList, List[TableTask]],
            cont_cols: Optional[List[str]] = None,
            cat_cols: Optional[List[str]] = None,
            feature_selection_groups: Optional[List[List[str]]] = None,
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
        cont_cols : Optional[List[str]]
            List of column names associated with continuous data.
        cat_cols : Optional[List[str]]
            List of column names associated with categorical data.
        feature_selection_groups : Optional[List[List[str]]]
            List with list of column names to consider together in group-wise feature selection.
        to_tensor : bool
            Whether we want the features and targets in tensors. False for numpy arrays.
        """
        super(TableDataset).__init__()

        # Validation of inputs
        if cont_cols is None and cat_cols is None:
            raise ValueError("At least a list of continuous columns or a list of categorical columns must be provided.")

        for columns in [cont_cols, cat_cols]:
            self._check_columns_validity(df, columns)

        self._tasks = TaskList(tasks)
        assert all([isinstance(task, TableTask) for task in TaskList(self._tasks)]), (
            f"All tasks must be instances of 'TableTask'."
        )

        # Set default protected attributes
        self._cat_cols, self._cat_idx = cat_cols, []
        self._cont_cols, self._cont_idx = cont_cols, []
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
        self._x = self._define_feature_getter(cont_cols, cat_cols, to_tensor)

        # We set feature selection idx groups
        self._feature_selection_idx_groups = self._create_feature_selection_idx_groups(feature_selection_groups)

        # We set a "getter" method to get modes of categorical columns and we also extract encodings
        self._get_modes, self._encodings = self._define_categorical_stats_getter(cat_cols)

        # We set a "getter" method to get mu ans std of continuous columns
        self._get_mu_and_std = self._define_numerical_stats_getter(cont_cols)

        # We set two "setter" methods to preprocess available data after masks update
        self._set_numerical = self._define_numerical_data_setter(cont_cols, to_tensor)
        self._set_categorical = self._define_categorical_data_setter(cat_cols, to_tensor)

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
        y = dict((col, self.y[idx, i]) for i, col in enumerate(self.target_cols))

        return TableDataModel(x=x, y=y)

    @property
    def cat_cols(self) -> List[str]:
        return self._cat_cols

    @property
    def cat_idx(self) -> List[int]:
        return self._cat_idx

    @property
    def cat_sizes(self) -> Optional[List[int]]:
        if self._encodings is not None:
            return [len(self._encodings[c].items()) for c in self._cat_cols]
        return None

    @property
    def tasks(self) -> TaskList:
        return self._tasks

    @property
    def cont_cols(self) -> List[str]:
        return self._cont_cols

    @property
    def cont_idx(self) -> List[int]:
        return self._cont_idx

    @property
    def encodings(self) -> Dict[str, Dict[str, int]]:
        return self._encodings

    @property
    def features_cols(self) -> List[str]:
        return self.cont_cols + self.cat_cols

    @property
    def feature_selection_idx_groups(self) -> Dict[int, Dict[str, List]]:
        return self._feature_selection_idx_groups

    @property
    def ids_col(self) -> str:
        return self._ids_col

    @property
    def ids(self) -> List[str]:
        return self._ids

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
    def y(self) -> Union[np.array, Tensor]:
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
        x_cat, _ = preprocess_categoricals(
            self._original_data[self._cat_cols].copy(),
            mode=modes,
            encodings=self._encodings
        )

        self._x_cat = x_cat.to_numpy(dtype=int)

    def _create_feature_selection_idx_groups(
            self,
            groups: Optional[List[List[str]]]
    ) -> Dict[int, dict[str, list | Any]]:
        """
        Creates a list of lists with idx of features in the different groups. All the features not included in any group
        will be used to create an additional group.

        Parameters
        ----------
        groups: Optional[List[List[str]]]
            List of list with name of columns to use in group for feature selection.
            
        Returns
        -------
        feature_idx_groups : Dict[str]
            Dictionary of the features and indexes.
        """
        # We create an additional group with the features that are not already in a group
        groups = [] if (groups is None or groups[0] is None) else groups
        cat_cols = [] if self._cat_cols is None else self._cat_cols
        cont_cols = [] if self._cont_cols is None else self._cont_cols

        last_group = []
        for c in cat_cols + cont_cols:
            included = False
            for group in groups:
                if c in group:
                    included = True
                    break
            if not included:
                last_group.append(c)

        if len(last_group) > 0:
            groups.append(last_group)

        # We associate each feature to its index when data is extracted using the item getter
        feature_idx_groups = {}
        for i, group in enumerate(groups):
            group_idx = []
            for f in group:
                if f in cat_cols:
                    group_idx.append(self._cat_idx[cat_cols.index(f)])
                elif f in cont_cols:
                    group_idx.append(self._cont_idx[cont_cols.index(f)])
                else:
                    raise ValueError(f"{f} is not part of cont_cols or cat_cols")
            feature_idx_groups[i] = {'features': group, 'idx': group_idx}

        return feature_idx_groups

    def _define_categorical_data_setter(
            self,
            cat_cols: Optional[List[str]] = None,
            to_tensor: bool = False
    ) -> Callable:
        """
        Defines the function used to set categorical data after masks update.

        Parameters
        ----------
        cat_cols : Optional[List[str]]
            List with names of categorical columns.
        to_tensor : bool, default = False
            True if we want the data to be converted into tensor.

        Returns
        -------
        set_categorical : Callable
            The function used to set categorical data after masks update.
        """
        # If there is no categorical columns
        if cat_cols is None:

            def set_categorical(modes: Optional[pd.Series]) -> None:
                pass

            return set_categorical

        else:
            if to_tensor:

                def set_categorical(modes: Optional[pd.Series]) -> None:
                    self._categorical_setter(modes)
                    self._x_cat = from_numpy(self._x_cat).long()

                return set_categorical

            return self._categorical_setter

    def _define_categorical_stats_getter(
            self,
            cat_cols: Optional[List[str]] = None
    ) -> Tuple[Callable, Dict[str, Dict[str, int]]]:
        """
        Defines the function used to extract the modes of categorical columns.

        Parameters
        ----------
        cat_cols : Optional[List[str]]
            List of categorical column names

        Returns
        -------
        get_modes, encodings : Tuple[Callable, Dict[str, Dict[str, int]]]
            The function used to extract the modes of categorical columns paired with categorical variables encodings.
        """
        # If there is not categorical column
        if cat_cols is None:

            def get_modes(df: Optional[pd.DataFrame]) -> None:
                return None

            encodings = None

        else:
            # Make sure that categorical data in the original dataframe is in the correct format
            self._original_data[cat_cols] = self._original_data[cat_cols].astype('category')

            # We extract ordinal encodings
            encodings = {c: {v: k for k, v in enumerate(self._original_data[c].cat.categories)} for c in cat_cols}

            def get_modes(df: pd.DataFrame) -> pd.Series:
                return df[cat_cols].mode().iloc[0]

        return get_modes, encodings

    def _define_feature_getter(
            self,
            cont_cols: Optional[List[str]] = None,
            cat_cols: Optional[List[str]] = None,
            to_tensor: bool = False
    ) -> Callable:
        """
        Defines the method used to extract the features (processed data) for training.

        Parameters
        ----------
        cont_cols : Optional[List[str]]
            List of continuous column names.
        cat_cols : Optional[List[str]]
            list of categorical column names
        to_tensor : bool
            True if the data must be converted to tensor

        Returns
        -------
        feature_getter : Callable
            Function used to extract features from data.
        """
        if cont_cols is None:

            # Only categorical column idx
            self._cat_idx = list(range(len(cat_cols)))

            # Only categorical feature extracted by the getter
            def x() -> Union[Tensor, np.array]:
                return self.x_cat

        elif cat_cols is None:

            # Only continuous column idx
            self._cont_idx = list(range(len(cont_cols)))

            # Only continuous features extracted by the getter
            def x() -> Union[Tensor, np.array]:
                return self.x_cont

        else:

            # Continuous and categorical column idx
            nb_cont_cols = len(cont_cols)
            self._cont_idx = list(range(nb_cont_cols))
            self._cat_idx = list(range(nb_cont_cols, nb_cont_cols + len(cat_cols)))

            # Continuous and categorical features extracted by the getter
            if not to_tensor:
                def x() -> Union[Tensor, np.array]:
                    return np.concatenate((self.x_cont, self.x_cat), axis=1)
            else:
                def x() -> Union[Tensor, np.array]:
                    return cat((self.x_cont, self.x_cat), dim=1)

        return x

    def _define_numerical_data_setter(
            self,
            cont_cols: Optional[List[str]] = None,
            to_tensor: bool = False
    ) -> Callable:
        """
        Defines the function used to set numerical continuous data after masks update.

        Parameters
        ----------
        cont_cols : Optional[List[str]]
            List of continuous column names.
        to_tensor : False
            True if data needs to be converted into tensor.

        Returns
        -------
        numerical_data_setter : Callable
             Function used to set numerical continuous data after masks update.
        """
        # If there is no continuous column
        if cont_cols is None:

            def set_numerical(mu: Optional[pd.Series], std: Optional[pd.Series]) -> None:
                pass

            return set_numerical

        else:
            if to_tensor:

                def set_numerical(mu: Optional[pd.Series], std: Optional[pd.Series]) -> None:
                    self._numerical_setter(mu, std)
                    self._x_cont = from_numpy(self._x_cont).float()

                return set_numerical

            return self._numerical_setter

    def _define_numerical_stats_getter(
            self,
            cont_cols: Optional[List[str]] = None
    ) -> Callable:
        """
        Defines the function used to extract the mean and the standard deviations of numerical columns in a dataframe.

        Parameters
        ----------
        cont_cols : Optional[List[str]]
            List with names of continuous columns.

        Returns
        -------
        numerical_stats_getter : Callable
            Function used to extract the mean and the standard deviations of numerical columns in a dataframe.
        """
        # If there is no continuous column
        if cont_cols is None:

            def get_mu_and_std(df: pd.DataFrame) -> Tuple[None, None]:
                return None, None
        else:
            # Make sure that numerical data in the original dataframe is in the correct format
            self._original_data[cont_cols] = self._original_data[cont_cols].astype(float)

            def get_mu_and_std(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
                return df[self._cont_cols].mean(), df[self._cont_cols].std()

        return get_mu_and_std

    def _get_augmented_dataframe(
            self,
            data: pd.DataFrame,
            categorical: bool = False
    ) -> Tuple[pd.DataFrame, Optional[List[str]], Optional[List[str]]]:
        """
        Returns an augmented dataframe by concatenating original df and data.

        Parameters
        ----------
        data : pd.Dataframe
            Dataframe with 2 columns. First column must be PATIENT ids. Second column must be the feature we want to
            add.
        categorical : bool
            True if the new features are categorical.

        Returns
        -------
        df, cont_cols, cat_cols : Tuple[pd.DataFrame, Optional[List[str]], Optional[List[str]]]
            Pandas dataframe, list of cont cols, list of cat cols.
        """
        # Extraction of the original dataframe
        df = self._retrieve_subset_from_original(self._cont_cols, self._cat_cols)

        # We add the new feature
        df = pd.merge(df, data, on=[self._ids_col], how=MaskType.INNER)

        # We update the columns list
        feature_name = [f for f in data.columns if f != self._ids_col]
        if categorical:
            cat_cols = self._cat_cols + feature_name if self._cat_cols is not None else [feature_name]
            cont_cols = self._cont_cols
        else:
            cont_cols = self._cont_cols + feature_name if self._cont_cols is not None else [feature_name]
            cat_cols = self._cat_cols

        return df, cont_cols, cat_cols

    def _set_scaling_factors(self):
        """
        Sets scaling factor of all classification tasks.
        """
        for task in self.tasks.classification_tasks:
            idx = self.target_cols.index(task.target_column)

            # We set the scaling factors of all classification metrics
            for metric in task.metrics:
                metric.update_scaling_factor(y_train=self.y[self.train_mask, idx])

            # We set the scaling factor of the criterion
            if task.criterion:
                task.criterion.update_scaling_factor(y_train=self.y[self.train_mask, idx])

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
        x_cont = preprocess_continuous(self._original_data[self._cont_cols].copy(), mu, std)

        # We apply the basis function
        self._x_cont = x_cont.to_numpy(dtype=float)

    def _retrieve_subset_from_original(
            self,
            cont_cols: Optional[List[str]] = None,
            cat_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Returns a copy of a subset of the original dataframe.

        Parameters
        ----------
        cont_cols : Optional[List[str]]
            List of continuous columns.
        cat_cols : Optional[List[str]]
            List of categorical columns.

        Returns
        -------
        subset_df : pd.DataFrame
            Copy of a subset of the original dataframe.
        """
        selected_cols = []
        if cont_cols is not None:
            selected_cols += cont_cols
        if cat_cols is not None:
            selected_cols += cat_cols

        return self.original_data[[self._ids_col] + self._target_cols + selected_cols].copy()

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

        if self._cont_cols is not None:
            imputed_df[self._cont_cols] = np.array(self._x_cont)
        if self._cat_cols is not None:
            imputed_df[self._cat_cols] = np.array(self._x_cat)

        return imputed_df

    def create_subset(
            self,
            cont_cols: Optional[List[str]] = None,
            cat_cols: List[str] = None
    ) -> TableDataset:
        """
        Returns a subset of the current dataset using the given cont_cols and cat_cols.

        Parameters
        ----------
        cont_cols : Optional[List[str]]
            List of continuous columns.
        cat_cols : Optional[List[str]]
            List of categorical columns.

        Returns
        -------
        sub_dataset : TableDataset
            Instance of the TableDataset class.
        """
        subset = self._retrieve_subset_from_original(cont_cols, cat_cols)

        sub_dataset = TableDataset(
            df=subset,
            ids_col=self._ids_col,
            tasks=self._tasks,
            cont_cols=cont_cols,
            cat_cols=cat_cols,
            to_tensor=self._to_tensor
        )

        return sub_dataset

    def create_superset(
            self,
            data: pd.DataFrame,
            categorical: bool = False
    ) -> TableDataset:
        """
        Returns a superset of the current dataset by including the given data.

        Parameters
        ----------
        data : pd.Dataframe
            Dataframe with 2 columns. First column must be PATIENT ids. Second column must be the feature we want to
            add.
        categorical : bool
            True if the new features are categorical.

        Returns
        -------
        sub_dataset : TableDataset
            Instance of the TableDataset class.
        """
        # We build the augmented dataframe
        df, cont_cols, cat_cols = self._get_augmented_dataframe(data, categorical)

        super_dataset = TableDataset(
            df=df,
            ids_col=self._ids_col,
            tasks=self._tasks,
            cont_cols=cont_cols,
            cat_cols=cat_cols,
            to_tensor=self._to_tensor
        )

        return super_dataset

    def current_train_stats(
            self
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """
        Returns the current statistics and encodings related to the training data.
        """
        # We extract the current training data
        train_data = self._original_data.iloc[self._train_mask]

        # We compute the current values of mu, std, modes and encodings
        mu, std = self._get_mu_and_std(train_data)
        modes = self._get_modes(train_data)

        return mu, std, modes

    def get_one_hot_encodings(
            self,
            cat_cols: List[str]
    ) -> Union[np.array, Tensor]:
        """
        Returns one hot encodings associated to the specified categorical columns.

        Parameters
        ----------
        cat_cols : Optional[List[str]]
            List of categorical columns.

        Returns
        -------
        one_hot_encodings : Union[np.array, Tensor]
            One hot vector of categorical columns.
        """
        # We check if the column names specified are categorical
        self._validate_columns_type(cat_cols, categorical=True)

        # We extract one hot encodings
        e = CaT.one_hot_encode(self.get_imputed_dataframe()[cat_cols].astype('str'))

        # We return the good type of data
        if self._to_tensor:
            return CaT.to_tensor(e)
        else:
            return e.to_numpy(dtype=int)

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

    def _validate_columns_type(
            self,
            col_list: List[str],
            categorical: bool
    ) -> None:
        """
        Checks if all element in the column names list are either in the cat_cols list or the cont_cols list.

        Parameters
        ----------
        col_list : List[str]
            List of column names.
        categorical : bool
            Whether columns contain categorical variables or not.
        """
        if categorical:
            cols = self._cat_cols if self._cat_cols is not None else []
            col_type = 'categorical'
        else:
            cols = self._cont_cols if self._cont_cols is not None else []
            col_type = 'continuous'

        for c in col_list:
            if c not in cols:
                raise ValueError(f"Column name {c} is not part of the {col_type} columns")

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
        targets = []
        for task in tasks:
            # Set targets protected attribute according to task
            t = self.original_data[task.target_column].to_numpy(dtype=float)

            if (not isinstance(task, ClassificationTask)) and target_to_tensor:
                t = from_numpy(t).float()
            elif isinstance(task, ClassificationTask):
                if target_to_tensor:
                    t = from_numpy(t).long()
                else:
                    t = t.astype(int)

            targets.append(t)

        if target_to_tensor:
            return stack(targets, dim=1)
        else:
            return np.stack(targets, axis=1)

    @staticmethod
    def _check_columns_validity(
            df: pd.DataFrame,
            columns: Optional[List[str]] = None
    ) -> None:
        """
        Checks if the columns are all in the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas dataframe with original data.
        columns : Optional[List[str]]
            List of column names.
        """
        if columns is not None:
            dataframe_columns = list(df.columns.values)
            for c in columns:
                if c not in dataframe_columns:
                    raise ValueError(f"Column {c} is not part of the given dataframe")
