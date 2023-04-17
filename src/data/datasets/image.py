"""
    @file:              image.py
    @Author:            Raphael Brodeur, Maxence Larose

    @Creation Date:     05/2022
    @Last modification: 04/2023

    @Description:       This file contains a class used to create a dataset containing multiple patient images and
                        segmentations from a given DELIA database.
"""

import collections.abc
from itertools import chain
from typing import Dict, List, Optional, Union, Sequence, Set, Tuple

from delia.databases.patients_database import PatientsDatabase
from monai.data import MetaTensor
from monai.transforms import apply_transform, Compose, EnsureChannelFirstd, MapTransform, ToTensord
import numpy as np
from torch import float32
from torch.utils.data import Dataset, Subset

from .modality import Modality
from ...tasks import SegmentationTask, TaskList


class ImageDataset(Dataset):
    """
    A class used to create a dataset containing multiple patient images and segmentations from a given DELIA database.
    The rendered images are in shape (Z, X, Y) by default.
    """

    def __init__(
            self,
            database: PatientsDatabase,
            modalities: Set[str],
            organs: Optional[Dict[str, Set[str]]] = None,
            tasks: Optional[Union[SegmentationTask, TaskList, List[SegmentationTask]]] = None,
            transforms: Optional[Union[Compose, MapTransform]] = None,
            transposition: Tuple[int, int, int] = (2, 0, 1),
            **kwargs
    ):
        """
        Creates a dataset containing multiple patient images and segmentations from a given DELIA database.

        Parameters
        ----------
        database : PatientsDatabase
            A DELIA database that is used to interact with the HDF5 file that contains all the patients' folders.
        modalities : Set[str]
            Set of image modalities to include in the dataset. These images are added to the dataset as features.
                    Example : {"CT", "PT", "MR"}.
        organs : Dict[str, Set[str]]
            Dictionary of organs to include in the dataset. Keys are modality names and values are sets of organs. The
            keys of the dictionary, i.e. modality images (CT scan for example), will NOT be added to the dataset. This
            parameter is only used to add the organs. These label maps are added to the dataset as features.
                    Example : {
                        "CT": {"Prostate", "Bladder"},
                        "PT": {"Prostate"},
                        "MR": {"Brain"}
                    }.
        tasks : Optional[Union[SegmentationTask, TaskList, List[SegmentationTask]]]
            Segmentation tasks to perform. These label maps are added to the dataset as targets.
        transforms : Optional[Union[Compose, MapTransform]]
            A single or a sequence of transforms to apply to images and segmentations (depending on transform keys).
        transposition : Tuple[int, int, int]
            The transposition to apply to images before applying transforms. The rendered images are in shape (Z, X, Y)
            by default.
        **kwargs : dict
            Keywords arguments controlling images and segmentations format, and segmentations series to use.
        """
        self._tasks = TaskList(tasks)
        assert all(isinstance(task, SegmentationTask) for task in self._tasks), (
            f"All tasks must be instances of 'SegmentationTask'."
        )

        self._database = database
        self._modalities = modalities
        self._organs = organs if organs else {}
        self._modalities_to_iterate_over = set(
            chain(
                modalities,
                self._organs.keys(),
                [t.modality for t in self._tasks]
            )
        )
        self._validate_modalities()
        self._transforms = transforms
        self._transposition = transposition

        self._organ_key_getter = kwargs.get("organ_key_getter", lambda modality, organ: f"{modality}_{organ}")
        self._seg_series = kwargs.get("seg_series", "0")

    def __len__(self) -> int:
        """
        Gets dataset length.

        Returns
        -------
        length : int
            Length of the database.
        """
        return len(self._database)

    def __getitem__(
            self,
            index: Union[int, slice, Sequence[int]]
    ) -> Union[Dict[str, Union[np.array, MetaTensor]], Subset]:
        """
        Gets specific items in the dataset.

        Parameters
        ----------
        index : Union[int, slice, Sequence[int]]
            The index of the items to get.

        Returns
        -------
        item : Union[Dict[str, Union[np.array, MetaTensor]], Subset]
            A single patient data or a subset of the dataset.
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        elif isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)
        elif isinstance(index, int):
            data = self._get_patient_data(index)
            return self._transform(data)
        else:
            raise AssertionError(f"'index' must be of type 'int', 'slice' or 'Sequence[int]'. Found type {type(index)}")

    @property
    def tasks(self) -> TaskList:
        return self._tasks

    def _validate_modalities(self) -> None:
        """
        Validates modalities by checking if all given modalities are found in the list of available modalities.
        """
        available_modalities = [str(modality) for modality in Modality]
        unknown_modalities = list(self._modalities_to_iterate_over - set(available_modalities))

        assert not unknown_modalities, (
            f"Found {len(unknown_modalities)} unknown modalities in the given modalities, i.e. {unknown_modalities}. "
            f"Available modalities are {available_modalities}."
        )

    def _get_patient_data(self, index: int) -> Dict[str, np.ndarray]:
        """
        Gets a single patient images data from its index in the dataset.

        Parameters
        ----------
        index : int
            The index of the patient whose data needs to be retrieved

        Returns
        -------
        data : Dict[str, np.ndarray]
            A patient images data.
        """
        patient_group = self._database[index]

        # print(f"Loading {patient_group.name[1:]}.")  # TODO : Use logging instead of a print.
        img_dict, seg_dict = {}, {}
        for series_number in patient_group.keys():
            series = patient_group[series_number]
            for modality in self._modalities_to_iterate_over:
                if series.attrs[self._database.MODALITY] == modality:
                    if modality in self._modalities:
                        img_dict[modality] = self._transpose(series[self._database.IMAGE])

                    if modality in self._organs.keys():
                        for organ in self._organs[modality]:
                            seg_array = series[self._seg_series][organ]
                            seg_dict[self._organ_key_getter(modality, organ)] = self._transpose(seg_array)

                    for task in self.tasks.segmentation_tasks:
                        if modality == task.modality:
                            seg_array = series[self._seg_series][task.organ]
                            seg_dict[task.name] = self._transpose(seg_array)

        return dict(img_dict, **seg_dict)

    def _transform(self, data: Dict[str, np.ndarray]) -> Dict[str, Union[np.array, MetaTensor]]:
        """
        Transforms images and segmentations.

        Parameters
        ----------
        data : Dict[str, np.ndarray]
            A dictionary containing 3D images or segmentation maps of a single patient.

        Returns
        -------
        transformed_data : Dict[str, Union[np.array, MetaTensor]]
            The dictionary of transformed images and segmentation maps.
        """
        if self._transforms is None:
            keys = list(data.keys())
            transforms = Compose([
                EnsureChannelFirstd(keys=keys),
                ToTensord(keys=keys, dtype=float32)
            ])
        else:
            transforms = self._transforms

        return apply_transform(transforms, data) if transforms else data

    def _transpose(self, array: np.ndarray) -> np.array:
        """
        Gets transposed data.

        Parameters
        ----------
        array : np.array
            A 3D image or segmentation map.

        Returns
        -------
        transposed_array : np.array
            The transposed array.
        """
        return np.transpose(np.array(array), self._transposition)
