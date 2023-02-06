"""
    @file:              image_dataset.py
    @Author:            Raphael Brodeur, Maxence Larose

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file contains a class used to create a dataset containing multiple patient images and
                        segmentations from a given DELIA database.
"""

import collections.abc
from typing import Dict, List, Optional, Union, Sequence, Set, Tuple

from delia.databases.patients_database import PatientsDatabase
from monai.data import MetaTensor
from monai.transforms import apply_transform, Compose, MapTransform
import numpy as np
from torch.utils.data import Dataset, Subset

from src.utils.tasks import SegmentationTask
from src.utils.task_list import TaskList


class ImageDataset(Dataset):
    """
    A class used to create a dataset containing multiple patient images and segmentations from a given DELIA database.
    The rendered images are in shape (Z, X, Y) by default.
    """

    def __init__(
            self,
            database: PatientsDatabase,
            tasks: Union[SegmentationTask, TaskList, List[SegmentationTask]],
            modalities: Set[str],
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
        tasks : Union[SegmentationTask, TaskList, List[SegmentationTask]]
            Segmentation tasks to perform.
        modalities : Set[str]
            Set of modalities to include in the dataset. Ex : {CT, PT, MR}.
        transforms : Optional[Union[Compose, MapTransform]]
            A single or a sequence of transforms to apply to images and segmentations (depending on transform keys).
        transposition : Tuple[int, int, int]
            The transposition to apply to images before applying transforms. The rendered images are in shape (Z, X, Y)
            by default.
        **kwargs : dict
            Keywords arguments controlling images and segmentations format, and segmentations series to use.
        """
        self.database = database
        self.modalities = modalities
        self.tasks = TaskList(tasks)
        self.transforms = transforms
        self.transposition = transposition

        self.img_format = kwargs.get("img_format", np.float16)
        self.seg_format = kwargs.get("seg_format", np.int8)
        self.seg_series = kwargs.get("seg_series", "0")

    def __len__(self) -> int:
        """
        Gets dataset length.

        Returns
        -------
        length : int
            Length of the database.
        """
        return len(self.database)

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
        patient_group = self.database[index]

        print(f"Loading {patient_group.name[1:]}.")  # TODO : Use logging instead of a print.
        img_dict, seg_dict = {}, {}
        for series_number in patient_group.keys():
            series = patient_group[series_number]
            for modality in self.modalities:
                if series.attrs[self.database.MODALITY] == modality:
                    img_dict[modality] = self._transpose(series[self.database.IMAGE]).astype(self.img_format)

                    for task in self.tasks.segmentation_tasks:
                        if modality == task.modality:
                            seg_array = series[self.seg_series][task.organ]
                            seg_dict[task.organ] = self._transpose(seg_array).astype(self.seg_format)

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
        return apply_transform(self.transforms, data) if self.transforms else data

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
        return np.transpose(np.array(array), self.transposition)
