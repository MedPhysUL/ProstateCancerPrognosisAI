"""
    @file:              sampling.py
    @Author:            Maxence Larose, Nicolas Raymond, Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 06/2022

    @Description:       This file contains the RandomStratifiedSampler class used to separate test sets and valid sets
                        from train sets. Also contains few functions used to extract specific datasets.
"""

from copy import deepcopy
from itertools import product
from json import load
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import tensor
from tqdm import tqdm

from src.data.processing.single_task_table_dataset import MaskType, SingleTaskTableDataset
from src.data.processing.multi_task_table_dataset import MultiTaskTableDataset


class RandomStratifiedSampler:
    """
    A class used to generate lists of indexes to use as train, valid and test masks for outer and inner validation
    loops.
    """

    def __init__(
            self,
            dataset: Union[SingleTaskTableDataset, MultiTaskTableDataset],
            n_out_split: int,
            n_in_split: int,
            valid_size: float = 0.20,
            test_size: float = 0.20,
            random_state: Optional[int] = None,
            alpha: int = 4,
            patience: int = 100
    ):
        """
        Sets private and public attributes of the sampler.

        Parameters
        ----------
        dataset : Union[SingleTaskTableDataset, MultiTaskTableDataset]
            Custom dataset.
        n_out_split : int
            Number of outer splits to produce.
        n_in_split : int
            Number of inner splits to produce.
        valid_size : float
            Percentage of data taken to create the validation indexes set.
        test_size : float
            Percentage of data taken to create the test indexes set
        random_state : Optional[int]
            Random state integer.
        alpha : int
            IQR multiplier used to check numerical variable range validity of the masks created
        patience : int
            Number of tries that the sampler has to make a single valid split
        """
        if n_out_split <= 0:
            raise ValueError('Number of outer split must be greater than 0')
        if n_in_split < 0:
            raise ValueError('Number of inner split must be greater or equal to 0')
        if not (0 <= valid_size < 1):
            raise ValueError('Validation size must be in the range [0, 1)')
        if not (0 < test_size < 1):
            raise ValueError('Test size must be in the range (0, 1)')
        if valid_size + test_size >= 1:
            raise ValueError('Train size must be non null')

        self.__original_dataset = dataset

        # Private attributes
        self.__unique_encodings = []
        if isinstance(dataset, SingleTaskTableDataset):
            self.__datasets = [dataset]
            self.__targets = [dataset.y]
            if self.__datasets[0].encodings is not None:
                self.__unique_encodings = [{k: list(v.values()) for k, v in self.__datasets[0].encodings.items()}]
            else:
                self.__unique_encodings = [{}]
        elif isinstance(dataset, MultiTaskTableDataset):
            self.__datasets = dataset.datasets
            self.__targets = dataset.get_targets()
            for ds in self.__datasets:
                if ds.encodings is not None:
                    self.__unique_encodings.append({k: list(v.values()) for k, v in ds.encodings.items()})
                else:
                    self.__unique_encodings.append({})

        # Public attributes
        self.alpha = alpha
        self.n_out_split = n_out_split
        self.n_in_split = n_in_split
        self.patience = patience
        self.random_state = random_state

        # Public method
        self.split = self.__define_split_function(test_size, valid_size)

    def __call__(
            self
    ) -> Dict[int, Dict[str, Union[List[int], Dict[str, List[int]]]]]:
        """
        Returns lists of indexes to use as train, valid and test masks for outer and inner validation loops.
        The proportion of each class is conserved within each split.

        Returns
        -------
        masks : Dict[int, Dict[str, Union[List[int], Dict[str, List[int]]]]]
            Dictionary of dictionaries with list of indexes.

        Example
        -------
            {0:
                {
                    'train': [..],
                    'valid': [..],
                    'test': [..],
                    'inner':
                        {0:
                            {
                                'train': [..],
                                'valid': [..],
                                'test': [..]
                            }
                        }
                },
            ...
            }
        """
        # We set targets to use for stratification
        targets = [target if self.is_categorical(target) else self.mimic_classes(target) for target in self.__targets]

        # We set the random state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # We initialize the dict that will contain the results and the list of indexes to use
        masks, idx = {}, np.array(range(len(targets[0])))

        # We save a copy of targets in an array
        targets_c = deepcopy(targets)

        with tqdm(total=(self.n_out_split + self.n_out_split*self.n_in_split)) as bar:
            for i in range(self.n_out_split):

                # We create outer split masks
                masks[i] = {**self.split(idx, targets_c), MaskType.INNER: {}}
                bar.update()

                for j in range(self.n_in_split):

                    # We create the inner split masks
                    masks[i][MaskType.INNER][j] = self.split(masks[i][MaskType.TRAIN], targets_c)
                    bar.update()

        # We turn arrays of idx into lists of idx
        self.serialize_masks(masks)

        return masks

    def __define_split_function(
            self,
            test_size: float,
            valid_size: float
    ) -> Callable:
        """
        Defines the split function according to the valid size.

        Parameters
        ----------
        test_size : float
            Percentage of data taken to create the train indexes set.
        valid_size : float
            Percentage of data taken to create the validation indexes set.

        Returns
        -------
        split : Callable
            Split function.
        """

        if valid_size > 0:

            # Split must extract train, valid and test masks
            def split(idx: np.array, targets: List[np.array]) -> Dict[str, np.array]:
                train_mask, valid_mask, test_mask = None, None, None
                # We make a pseudo-multilabel
                targets_str = targets[0].astype(str)
                for i in range(len(targets)-1):
                    targets_str = np.char.add(targets_str, targets[i+1].astype(str))

                # We initialize loop important values
                mask_ok = False
                nb_tries_remaining = self.patience

                # We test multiple possibilities till we find one or the patience is achieved
                while not mask_ok and nb_tries_remaining > 0:
                    remaining_idx, test_mask = train_test_split(idx, stratify=targets_str[idx], test_size=test_size)
                    train_mask, valid_mask = train_test_split(remaining_idx, stratify=targets_str[remaining_idx],
                                                              test_size=valid_size)
                    mask_ok = self.check_masks_validity(train_mask, test_mask, valid_mask)
                    nb_tries_remaining -= 1

                if not mask_ok:
                    raise Exception("The sampler could not find a proper train, valid and test split.")

                return {MaskType.TRAIN: train_mask, MaskType.VALID: valid_mask, MaskType.TEST: test_mask}
        else:
            # Split must extract train and test masks only
            def split(idx: np.array, targets: List[np.array]) -> Dict[str, np.array]:
                train_mask, test_mask = None, None
                # We make a pseudo-multilabel
                targets_str = targets[0].astype(str)
                for i in range(len(targets)-1):
                    targets_str = np.char.add(targets_str, targets[i+1].astype(str))

                # We initialize loop important values
                mask_ok = False
                nb_tries_remaining = self.patience

                # We test multiple possibilities till we find one or the patience is achieved
                while not mask_ok and nb_tries_remaining > 0:
                    train_mask, test_mask = train_test_split(idx, stratify=targets_str[idx], test_size=test_size)
                    mask_ok = self.check_masks_validity(train_mask, test_mask)
                    nb_tries_remaining -= 1

                if not mask_ok:
                    raise Exception("The sampler could not find a proper train and test split")

                return {MaskType.TRAIN: train_mask, MaskType.VALID: None, MaskType.TEST: test_mask}

        return split

    def check_masks_validity(
            self,
            train_mask: List[int],
            test_mask: List[int],
            valid_mask: Optional[List[int]] = None
    ) -> bool:
        """
        Valid if categorical and numerical variables of other masks are out of the range of train mask.

        Parameters
        ----------
        train_mask : List[int]
            List of idx to use for training.
        test_mask : List[int]
            List of idx to use for test.
        valid_mask : Optional[List[int]]
            List of idx to use for validation.

        Returns
        -------
        valid : bool
            Whether the masks are valid.
        """
        self.__original_dataset.update_masks(train_mask, test_mask, valid_mask)

        for idx, ds in enumerate(self.__datasets):
            # We extract train dataframe
            imputed_df = ds.get_imputed_dataframe()
            train_df = imputed_df.iloc[ds.train_mask]

            # We check if all categories of categorical columns are in the training set
            for cat, values in self.__unique_encodings[idx].items():
                for c in train_df[cat].unique():
                    if c not in values:
                        return False

            # We save q1 and q3 of each numerical columns
            train_quantiles = {c: (train_df[c].quantile(0.25), train_df[c].quantile(0.75)) for c in ds.cont_cols}

            # We validate the other masks
            other_masks = [m for m in [ds.valid_mask, ds.test_mask] if m is not None]
            for mask in other_masks:

                # We extract the subset
                subset_df = imputed_df.iloc[mask]

                # We check if all numerical values are not extreme outliers according to the train mask
                for cont_col, (q1, q3) in train_quantiles.items():
                    iqr = q3 - q1
                    other_min, other_max = (subset_df[cont_col].min(), subset_df[cont_col].max())
                    if other_min < q1 - self.alpha*iqr or other_max > q3 + self.alpha*iqr:
                        return False

        return True

    @staticmethod
    def is_categorical(
            targets: Union[tensor, np.array]
    ) -> bool:
        """
        Checks if the number of unique values is lower than 15% of the length of the targets sequence.

        Parameters
        ----------
        targets : Union[tensor, np.array]
            Sequence of float/int used for stratification.

        Returns
        -------
        categorical : bool
            Whether the target is a categorical variable.
        """
        target_list_copy = list(targets)
        return len(set(target_list_copy)) < 0.15*len(target_list_copy)

    @staticmethod
    def mimic_classes(
            targets: Union[tensor, np.array, List[Any]]
    ) -> np.array:
        """
        Creates fake classes array out of real-valued targets sequence using quartiles.

        Parameters
        ----------
        targets : Union[tensor, np.array, List[Any]]
            Sequence of float/int used for stratification.

        Returns
        -------
        fake_classes : np.array
            Array with fake classes.
        """
        return pd.qcut(np.array(targets), 2, labels=False)

    @staticmethod
    def serialize_masks(
            masks: Dict[int, Dict[str, Union[np.array, Dict[str, np.array]]]]
    ) -> None:
        """
        Turns all numpy arrays of idx into lists of idx.

        Parameters
        ----------
        masks : Dict[int, Dict[str, Union[np.array, Dict[str, np.array]]]]
            Dictionary of masks
        """
        for k, v in masks.items():
            for t1 in MaskType():
                masks[k][t1] = v[t1].tolist() if v[t1] is not None else None
            for in_k, in_v in masks[k][MaskType.INNER].items():
                for t2 in MaskType():
                    masks[k][MaskType.INNER][in_k][t2] = in_v[t2].tolist() if in_v[t2] is not None else None

    @staticmethod
    def visualize_splits(
            datasets: dict
    ) -> None:
        """
        Details the data splits for the experiment.

        Parameters
        ----------
        datasets : dict
            Dict with all the masks obtained from the sampler.
        """
        print("#----------------------------------#")
        for k, v in datasets.items():
            print(f"Split {k+1} \n")
            print(f"Outer :")
            valid = v[MaskType.VALID] if v[MaskType.VALID] is not None else []
            print(f"Train {len(v[MaskType.TRAIN])} - Valid {len(valid)} - Test {len(v[MaskType.TEST])}")
            if v[MaskType.INNER]:
                print(f"{MaskType.INNER} :")
            for k1, v1 in v[MaskType.INNER].items():
                valid = v1[MaskType.VALID] if v1[MaskType.VALID] is not None else []
                print(f"{k+1}.{k1} -> Train {len(v1[MaskType.TRAIN])} - Valid {len(valid)} -"
                      f" Test {len(v1[MaskType.TEST])}")
            print("#----------------------------------#")


# ---------------------- THE FUNCTION extract_masks BE INCLUDED IN THE STRATIFIER SAMPLER CLASS ---------------------- #

# def extract_masks(
#         file_path: str,
#         k: int = 20,
#         l: int = 20
# ) -> Dict[int, Dict[str, Union[List[int], Dict[str, List[int]]]]]:
#     """
#     Extract masks saved in json file.
#
#     Parameters
#     ----------
#     file_path : str
#         Path of json file containing the masks.
#     k : int
#         Number of outer loops to extract.
#     l : int
#         Number of inner loops to extract.
#
#     Returns
#     -------
#     masks
#     """
#     # Opening JSON file
#     f = open(file_path)
#
#     # Extract complete masks
#     all_masks = load(f)
#
#     # Extraction of masks subset
#     masks = {}
#     for i in map(str, range(k)):
#         int_i = int(i)
#         masks[int_i] = {}
#         for t in MaskType():
#             masks[int_i][t] = all_masks[i][t]
#         masks[int_i][MaskType.INNER] = {}
#         for j in map(str, range(l)):
#             int_j = int(j)
#             masks[int_i][MaskType.INNER][int_j] = {}
#             for t in MaskType():
#                 masks[int_i][MaskType.INNER][int_j][t] = all_masks[i][MaskType.INNER][j][t]
#
#     # Closing file
#     f.close()
#
#     return masks


# ---------------------- THE FOLLOWING CODE SHOULD BE SOMEWHERE ELSE (kinda application side ------------------------- #
# ----------------------                        instead of library side)                     ------------------------- #


# def push_valid_to_train(masks: Dict[int, Dict[str, Union[List[int], Dict[str, List[int]]]]]
#                         ) -> Dict[int, Dict[str, Union[List[int], Dict[str, List[int]]]]]:
#     """
#     Pushes all index of validation masks into train masks
#     Args:
#         masks: dictionary with list of idx to use as train, valid and test masks
#     Returns: same masks with valid idx added to test idx
#     """
#     for k, v in masks.items():
#         masks[k][MaskType.TRAIN] += v[MaskType.VALID]
#         masks[k][MaskType.VALID] = None
#         for in_k, in_v in masks[k][MaskType.INNER].items():
#             masks[k][MaskType.INNER][in_k][MaskType.TRAIN] += in_v[MaskType.VALID]
#             masks[k][MaskType.INNER][in_k][MaskType.VALID] = None
#
#
# def get_learning_one_data(data_manager: ,
#                           genes: Optional[str],
#                           baselines: bool = True,
#                           classification: bool = False,
#                           dummy: bool = False,
#                           holdout: bool = False) -> Tuple[pd.DataFrame, str, List[str], List[str]]:
#     """
#     Extracts dataframe needed to proceed to "learning one" experiments and turn it into a dataset
#     Args:
#         data_manager: data manager to communicate with the database
#         genes: One choice among ("None", "significant", "all")
#         baselines: if True, baselines variables are included
#         classification: if True, targets returned are obesity classes instead of Total Body Fat values
#         dummy: if True, includes dummy variable combining sex and Total Body Fat quantile
#         holdout: if True, holdout data is included at the bottom of the dataframe
#     Returns: dataframe, target, continuous columns, categorical columns
#     """
#
#     # We initialize empty lists for continuous and categorical columns
#     cont_cols, cat_cols = [], []
#
#     # We add baselines
#     if baselines:
#         cont_cols += [AGE_AT_DIAGNOSIS, DT, DOX, METHO, CORTICO]
#         cat_cols += [SEX, RADIOTHERAPY_DOSE, DEX, BIRTH_AGE]
#
#     if dummy:
#         cat_cols.append(WARMUP_DUMMY)
#
#     # We extract the dataframe
#     target = TOTAL_BODY_FAT
#     df = data_manager.get_table(LEARNING_1, columns=[PARTICIPANT, TOTAL_BODY_FAT] + cont_cols + cat_cols)
#
#     if holdout:
#         h_df = data_manager.get_table(LEARNING_1_HOLDOUT,
#                                       columns=[PARTICIPANT, TOTAL_BODY_FAT] + cont_cols + cat_cols)
#         df = df.append(h_df, ignore_index=True)
#
#     if classification:
#         target = OBESITY
#         ob_df = data_manager.get_table(OBESITY_TARGET, columns=[PARTICIPANT, OBESITY])
#         df = merge(df, ob_df, on=[PARTICIPANT], how=INNER)
#         df.drop([TOTAL_BODY_FAT], axis=1, inplace=True)
#
#     # We replace wrong categorical values
#     if baselines:
#         df.loc[(df[DEX] != "0"), [DEX]] = ">0"
#     else:
#         cont_cols = None
#
#     return df, target, cont_cols, cat_cols
#
#
# def get_learning_two_data(data_manager: PetaleDataManager,
#                           genes: Optional[str],
#                           baselines: bool = True,
#                           **kwargs) -> Tuple[DataFrame, str, List[str], List[str]]:
#     """
#     Extracts dataframe needed to proceed to "learning two" experiments and turn it into a dataset
#     Args:
#         data_manager: data manager to communicate with the database
#         genes: One choice among ("None", "significant", "all")
#         baselines: if True, baselines variables are included
#     Returns: dataframe, target, continuous columns, categorical columns
#     """
#
#     # We initialize empty lists for continuous and categorical columns
#     cont_cols, cat_cols = [], []
#
#     # We add baselines
#     if baselines:
#         cont_cols += [AGE_AT_DIAGNOSIS, DT, DOX, METHO, CORTICO]
#         cat_cols += [SEX, RADIOTHERAPY_DOSE, DEX, BIRTH_AGE]
#
#     # We check for genes
#     if genes is not None:
#
#         if genes not in GeneChoice():
#             raise ValueError(f"genes value must be in {GeneChoice()}")
#
#         if genes == GeneChoice.SIGNIFICANT:
#             cat_cols += SIGNIFICANT_CHROM_POS_REF
#
#         else:
#             cat_cols += ALL_CHROM_POS_REF
#
#     # We extract the dataframe
#     df = data_manager.get_table(LEARNING_2, columns=[PARTICIPANT, EF] + cont_cols + cat_cols)
#
#     # We replace wrong categorical values
#     if baselines:
#         df.loc[(df[DEX] != "0"), [DEX]] = ">0"
#     else:
#         cont_cols = None
#
#     if genes is not None:
#         df.replace("0/2", "0/1", inplace=True)
#         df.replace("1/2", "1/1", inplace=True)
#
#     return df, EF, cont_cols, cat_cols
#
#
# def generate_multitask_labels(df: DataFrame,
#                               target_columns: List[str]) -> Tuple[array, Dict[int, tuple]]:
#     """
#     Generates single array of class labels using all possible combinations of unique values
#     contained within target_columns.
#     For example, for 3 binary columns we will generate 2^3 = 8 different class labels and assign them
#     to the respective rows.
#     Args:
#         df: dataframe with items to classify
#         target_columns: names of the columns to use for multitask learning
#     Returns: array with labels, dict with the meaning of each label
#     """
#     # We extract unique values of each target column
#     possible_targets = [list(df[target].unique()) for target in target_columns]
#
#     # We generate all possible combinations of these unique values and assign them a label
#     labels_dict = {combination: i for i, combination in enumerate(product(*possible_targets))}
#
#     # We associate labels to the items in the dataframe
#     item_labels_union = list(zip(*[df[t].values for t in target_columns]))
#     multitask_labels = array([labels_dict[item] for item in item_labels_union])
#
#     # We rearrange labels_dict for visualization purpose
#     labels_dict = {v: k for k, v in labels_dict.items()}
#
#     return multitask_labels, labels_dict
#
#
# def get_warmup_data(data_manager: PetaleDataManager,
#                     baselines: bool = True,
#                     genes: Optional[str] = None,
#                     sex: bool = False,
#                     dummy: bool = False,
#                     holdout: bool = False) -> Tuple[DataFrame, str, Optional[List[str]], Optional[List[str]]]:
#     """
#     Extracts dataframe needed to proceed to warmup experiments
#     Args:
#         data_manager: data manager to communicate with the database
#         baselines: true if we want to include variables from original equation
#         genes: One choice among ("None", "significant", "all")
#         sex: true if we want to include sex variable
#         dummy: true if we want to include dummy variable combining sex and VO2 quantile
#         holdout: if true, holdout data is included at the bottom of the dataframe
#     Returns: dataframe, target, continuous columns, categorical columns
#     """
#
#     # We make sure few variables were selected
#     if not (baselines or genes or sex):
#         raise ValueError("At least baselines, genes or sex must be selected")
#
#     # We save participant and VO2 max column names
#     all_columns = [PARTICIPANT, VO2R_MAX]
#
#     # We save the name of continuous columns in a list
#     if baselines:
#         cont_cols = [WEIGHT, TDM6_HR_END, TDM6_DIST, DT, AGE, MVLPA]
#         all_columns += cont_cols
#     else:
#         cont_cols = None
#
#     # We check for genes
#     cat_cols = []
#     if genes is not None:
#
#         if genes not in GeneChoice():
#             raise ValueError(f"Genes value must be in {list(GeneChoice())}")
#
#         if genes == GeneChoice.ALL:
#             cat_cols += ALL_CHROM_POS_WARMUP
#
#         elif genes == GeneChoice.SIGNIFICANT:
#             cat_cols += SIGNIFICANT_CHROM_POS_WARMUP
#
#     if sex:
#         cat_cols.append(SEX)
#     if dummy:
#         cat_cols.append(WARMUP_DUMMY)
#
#     all_columns += cat_cols
#     cat_cols = cat_cols if len(cat_cols) != 0 else None
#
#     # We extract the dataframe
#     df = data_manager.get_table(LEARNING_0_GENES, columns=all_columns)
#
#     # We add the holdout data
#     if holdout:
#         h_df = data_manager.get_table(LEARNING_0_GENES_HOLDOUT, columns=all_columns)
#         df = df.append(h_df, ignore_index=True)
#
#     return df, VO2R_MAX, cont_cols, cat_cols
