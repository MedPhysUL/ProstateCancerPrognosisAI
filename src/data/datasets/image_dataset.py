"""
    @file:              image_dataset.py
    @Author:            Raphael Brodeur, Maxence Larose

    @Creation Date:     05/2022
    @Last modification: 07/2022

    @Description:       This file contains a class used to create a dataset of various patients and their respective CT
                        and segmentation map from a given local HDF5 file. The foreground is cropped and a crop along Z
                        can be specified.
"""

from typing import Callable, NamedTuple, Optional, Tuple, Union

from monai.data import Dataset, ZipDataset
from monai.transforms import CropForeground, SpatialCrop
import numpy as np

from src.data.extraction.local import LocalDatabaseManager


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
            tep: bool = False,
            transform: Optional[Callable] = None,
            z_dim: ZDimension = ZDimension(start=50, stop=210)
    ) -> None:
        """
        Creates a dataset of various patients and their respective CT and segmentation map from a given local HDf5 file.
        Images and segmentation maps are rendered in shape (Z, X, Y).

        Parameters
        ----------
        database_manager : LocalDatabaseManager
            A database manager that is used to interact with the HDF5 file that contains all the patients' folders.
        img_transform : Optional[Callable]
            A single or a sequence of transforms to apply to the image.
        seg_transform : Optional[Callable]
            A single or a sequence of transforms to apply to the segmentation.
        z_dim : ZDimension
            A tuple that specify the z-dimension crop.
        """
        db = database_manager.get_database()
        img_list, seg_list, tep_list = [], [], []
        for patient in db.keys():
            if db[patient]['0'].attrs[database_manager.MODALITY] == "CT":
                img = np.transpose(np.array(db[patient]['0'][database_manager.IMAGE]), (2, 0, 1))
                seg = np.transpose(np.array(db[patient]['0']['0']["Prostate_label_map"]), (2, 0, 1))

                if tep:
                    tep = np.transpose(np.array(db[patient]['1'][database_manager.IMAGE]), (2, 0, 1))

            else:
                img = np.transpose(np.array(db[patient]['1'][database_manager.IMAGE]), (2, 0, 1))
                seg = np.transpose(np.array(db[patient]['1']['0']["Prostate_label_map"]), (2, 0, 1))

                if tep:
                    tep = np.transpose(np.array(db[patient]['0'][database_manager.IMAGE]), (2, 0, 1))

            if tep:
                img_cropped, seg_cropped, tep_cropped = self._crop(img=img, seg=seg, tep=tep, z_dim=z_dim)

                img_list.append(img_cropped)
                seg_list.append(seg_cropped)
                tep_list.append(tep_cropped)

            else:
                img_cropped, seg_cropped = self._crop(img=img, seg=seg, z_dim=z_dim)

                img_list.append(img_cropped)
                seg_list.append(seg_cropped)

        if tep:
            super().__init__(
                data=[{'img': img, 'seg': seg, 'tep': tep} for img, seg, tep in zip(img_list, seg_list, tep_list)],
                transform=transform
            )

        else:
            super().__init__(
                data=[{'img': img, 'seg': seg} for img, seg in zip(img_list, seg_list)],
                transform=transform
            )

    @staticmethod
    def _crop(
            img: np.ndarray,
            seg: np.ndarray,
            tep: Optional[np.ndarray] = None,
            z_dim: ZDimension = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Crops the foreground. A crop along Z can also be specified.

        Parameters
        ----------
        img : np.ndarray
            An image array in shape (Z, X, Y).
        seg : np.ndarray
            A segmentation map array in shape (Z, X, Y).
        z_dim : ZDimension
            Lower bound and upper bound of the crop to apply along Z.

        Returns
        -------
        img_cropped, seg_cropped : Tuple[np.ndarray, np.ndarray]
            A cropped image array and a cropped segmentation map array.
        """
        img_cropped, start, end = CropForeground(return_coords=True)(img)
        seg_cropped = SpatialCrop(roi_start=start, roi_end=end)(seg)

        if z_dim:
            img_cropped, seg_cropped = img_cropped[z_dim[0]: z_dim[1]], seg_cropped[z_dim[0]: z_dim[1]]

        if tep:
            tep_cropped = SpatialCrop(roi_start=start, roi_end=end)(tep)

            if z_dim:
                tep_cropped = tep_cropped[z_dim[0]: z_dim[1]]

            return img_cropped, seg_cropped, tep_cropped

        return img_cropped, seg_cropped
