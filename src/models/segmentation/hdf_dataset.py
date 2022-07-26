"""
    @file:              hdf_dataset.py
    @Author:            Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 06/2022

    @Description:       This file contains a class used to create a dataset of various patients and their respective CT
                        and segmentation map from a given local HDF5 file. The foreground is cropped and a crop along Z
                        can be specified.
"""

import numpy as np
from typing import Callable, List, Optional

import h5py
from monai.data import ArrayDataset
from monai.transforms import CropForeground, SpatialCrop


PATIENTS_TO_DISCARD = [
    # "TEP-051",
    # "TEP-093",
    # "TEP-094",
    # "TEP-090",
    # "TEP-102",
    # "TEP-119",
    # "TEP-133", # celle qui est a l'envers
    # "TEP-175",
    # "TEP-176",
    # "TEP-296",
    # "TEP-306",
    # "TEP-358",
    # "TEP-363",
    # "TEP-377",
    # "TEP-407",
    # "TEP-444",
    # "TEP-482",
]


class HDFDataset(ArrayDataset):
    """
    A class used to create a dataset of various patients and their respective CT and segmentation map from a given local
    HDF5 file. The rendered images are in shape (Z, X, Y).
    """

    def __init__(
            self,
            path: str,
            img_transform: Optional[Callable] = None,
            seg_transform: Optional[Callable] = None
    ):
        """
        Creates a dataset of various patients and their respective CT and segmentation map from a given local HDf5 file.
        Images and segmentation maps are rendered in shape (Z, X, Y).

        Parameters
        ----------
        path : str
            The path to the HDF5 file that contains all the patients' folders.
        img_transform : Optional[Callable]
            A single or a sequence of transforms to apply to the image.
        seg_transform : Optional[Callable]
            A single or a sequence of transforms to apply to the segmentation.
        """
        file = h5py.File(path)
        img_list, seg_list = [], []
        for patient in file.keys():
            if patient in PATIENTS_TO_DISCARD:
                print(f"Scuffed patient named {patient} avoided.")
            elif file[patient]['0'].attrs['Modality'] == "CT":
                img = np.transpose(np.array(file[patient]['0']['image']), (2, 0, 1))
                seg = np.transpose(np.array(file[patient]['0']['0']['Prostate_label_map']), (2, 0, 1))

                img_cropped, seg_cropped = self._crop(img=img, seg=seg, z_dim=[50, 210])

                img_list.append(img_cropped)
                seg_list.append(seg_cropped)

            else:
                img = np.transpose(np.array(file[patient]['1']['image']), (2, 0, 1))
                seg = np.transpose(np.array(file[patient]['1']['0']['Prostate_label_map']), (2, 0, 1))

                img_cropped, seg_cropped = self._crop(img=img, seg=seg, z_dim=[50, 210])

                img_list.append(img_cropped)
                seg_list.append(seg_cropped)

        self._img_list = img_list
        super().__init__(img=img_list, seg=seg_list, img_transform=img_transform, seg_transform=seg_transform)

    @staticmethod
    def _crop(
            img: np.ndarray,
            seg: np.ndarray,
            z_dim: Optional[List] = None
    ):
        """
        Crops the foreground. A crop along Z can be specified.

        Parameters
        ----------
        img : np.ndarray
            An image array in shape (Z, X, Y).
        seg : np.ndarray
            A segmentation map array in shape (Z, X, Y).
        z_dim : Optional[List]
            Lower bound and upper bound of the crop to apply along Z.

        Returns
        -------
        img_cropped : np.ndarray
            A cropped image array.
        seg_cropped : np.ndarray
            A cropped segmentation map array.
        """
        img_cropped, start, end = CropForeground(return_coords=True)(img)
        seg_cropped = SpatialCrop(roi_start=start, roi_end=end)(seg)

        if z_dim:
            img_cropped, seg_cropped = img_cropped[z_dim[0]:z_dim[1]], seg_cropped[z_dim[0]:z_dim[1]]
            return img_cropped, seg_cropped

        return img_cropped, seg_cropped
