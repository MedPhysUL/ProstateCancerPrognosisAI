"""
    @file:              image_dataset.py
    @Author:            Raphael Brodeur, Maxence Larose

    @Creation Date:     05/2022
    @Last modification: 07/2022

    @Description:       This file contains a class used to create a dataset of various patients and their respective CT
                        and segmentation map from a given local HDF5 file. The foreground is cropped and a crop along Z
                        can be specified.
"""

from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple

from monai.data import Dataset

from monai.transforms import CropForeground, SpatialCrop
import numpy as np

from src.data.extraction.local import LocalDatabaseManager
from src.utils.tasks import SegmentationTask


class ImageDataset(Dataset):
    """
    A class used to create a dataset of various patients and their respective CT and segmentation map from a given local
    HDF5 file. The rendered images are in shape (Z, X, Y).
    """

    class ZDimension(NamedTuple):
        """
        A tuple that specify the z-dimension crop.
        """
        start: int
        stop: int

    def __init__(
            self,
            database_manager: LocalDatabaseManager,
            tasks: List[SegmentationTask],
            modalities: Set[str],
            transforms: Optional[Callable] = None,
            z_dim: ZDimension = ZDimension(start=50, stop=210)
    ) -> None:
        """
        Creates a dataset of various patients and their respective CT and segmentation map from a given local HDf5 file.
        Images and segmentation maps are rendered in shape (Z, X, Y).

        Parameters
        ----------
        database_manager : LocalDatabaseManager
            A database manager that is used to interact with the HDF5 file that contains all the patients' folders.
        tasks : List[SegmentationTask]
            Task to perform.
        transforms : Optional[Callable]
            A single or a sequence of transforms to apply to images and segmentations (depending on transform keys).
        z_dim : ZDimension
            A tuple that specify the z-dimension crop.
        """
        self._tasks = tasks

        db = database_manager.get_database()
        
        img_list, seg_list = [], []
        for patient in db.keys():
            print(f"Loading {patient}.")
            img_dict, seg_dict = {}, {}
            for series_number in db[patient].keys():
                series = db[patient][series_number]
                for modality in modalities:
                    if series.attrs[database_manager.MODALITY] == modality:
                        img_dict[modality] = self._transpose(series[database_manager.IMAGE])

                        for task in tasks:
                            if modality == task.modality:
                                seg_dict[task.organ] = self._transpose(series["0"][f"{task.organ}_label_map"])

            img_dict, seg_dict = self._crop(img_dict=img_dict, seg_dict=seg_dict, z_dim=z_dim)

            img_list.append(img_dict)
            seg_list.append(seg_dict)

        super().__init__(
            data=[dict(img_dict, **seg_dict) for img_dict, seg_dict in zip(img_list, seg_list)],
            transform=transforms
        )

    @property
    def tasks(self) -> List[SegmentationTask]:
        return self._tasks

    @staticmethod
    def _transpose(data) -> np.array:
        return np.transpose(np.array(data), (2, 0, 1))

    @staticmethod
    def _crop(
            img_dict: Dict[str, np.array],
            seg_dict: Dict[str, np.array],
            z_dim: ZDimension = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Crops the foreground. A crop along Z can also be specified.

        Parameters
        ----------
         img_dict: Dict[str, np.array]
            An image array in shape (Z, X, Y).
        seg_dict: Dict[str, np.array]
            A segmentation map array in shape (Z, X, Y).
        z_dim : ZDimension
            Lower bound and upper bound of the crop to apply along Z.

        Returns
        -------
        img_cropped, seg_cropped : Tuple[np.ndarray, np.ndarray]
            A cropped image array and a cropped segmentation map array.
        """
        img_dict["CT"], start, end = CropForeground(return_coords=True)(img_dict["CT"])

        for name, seg in seg_dict.items():
            seg_dict[name] = SpatialCrop(roi_start=start, roi_end=end)(seg)

        for name, img in img_dict.items():
            if name != "CT":
                img_dict[name] = SpatialCrop(roi_start=start, roi_end=end)(img)

        if z_dim:
            for name, img in img_dict.items():
                img_dict[name] = img[z_dim[0]: z_dim[1]]

            for name, seg in seg_dict.items():
                seg_dict[name] = seg[z_dim[0]: z_dim[1], :, :]

        return img_dict, seg_dict
