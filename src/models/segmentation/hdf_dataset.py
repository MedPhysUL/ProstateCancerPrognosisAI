# NOTES
# Cleaner le try-except et le tasse-467 une fois qu'on a regler le fait quon a des 467 et des 333
#
"""
    @file:              hdf_dataset.py
    @Author:            Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 05/2022

    @Description:       This file contains a class used to create a dataset of various patients and their respective CT
                        and segmentation map from a given local HDF5 file.
"""

import numpy as np
from typing import Callable, Optional, Sequence

import h5py
from torch.utils.data import Dataset
from monai.data import ArrayDataset


class HDFDataset(ArrayDataset):
    """
    A class used to create a dataset of various patients and their respective CT and segmentation map from a given local
    HDF5 file. The rendered dataset is (Patients x Channels x Z x X x Y).
    """
    def __init__(
            self,
            path: str,
            img_transform: Optional[Callable] = None,
            seg_transform: Optional[Callable] = None
    ):
        """
        Creates a dataset of various patients and their respective CT and segmentation map from a given local HDf5 file.

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
        img, seg = [], []
        for patient in file.keys():
            try: #va enelever  ca
                if file[patient]['0'].attrs['Modality'] == "CT" and file[patient]['0']['image'].shape == (333, 333, 573): # va enlver le and
                    img.append(np.transpose(np.array(file[patient]['0']['image']), (2, 0, 1)))
                    seg.append(np.transpose(np.array(file[patient]['0']['Prostate_label_map']), (2, 0, 1)))
                if file[patient]['1'].attrs['Modality'] == "CT" and file[patient]['1']['image'].shape == (333, 333, 573): #va remettre else
                    img.append(np.transpose(np.array(file[patient]['1']['image']),(2, 0, 1)))
                    seg.append(np.transpose(np.array(file[patient]['1']['Prostate_label_map']), (2, 0, 1)))
            except KeyError:                                # va enlever ca
                print(f"Patient {patient} ignored.")        # pis ca

        super().__init__(img=img, seg=seg, img_transform=img_transform, seg_transform=seg_transform)
