"""
    @file:              local.py
    @Author:            Maxence Larose

    @Creation Date:     06/2022
    @Last modification: 07/2022

    @Description:       This file contains the LocalDatabaseManager class which is used to create and/or interact with
                        an HDF5 file containing the patients' images and their segmentations.
"""

import os
from typing import Dict, List, Optional, Sequence, Union

from dicom2hdf import PatientsDatabase, PatientWhoFailed
from dicom2hdf.processing.transforms import BaseTransform
import h5py


class LocalDatabaseManager(PatientsDatabase):
    """
    A LocalDatabaseManager class which is used to extract images contained in DICOM files and aggregate these images and
    some metadata into an HDF5 file. This file is then easier to parse than the original DICOM files to retrieve images
    and segmentations.
    """

    def __init__(
            self,
            path_to_database: str
    ):
        """
        Initializes database.

        Parameters
        ----------
        path_to_database : str
            Path to dataset.
        """
        super().__init__(path_to_database=path_to_database)

    @property
    def _database_exists(
            self
    ) -> bool:
        """
        Whether database exists.

        Returns
        -------
        database_existence : bool
            Whether database exists.
        """
        return os.path.exists(self.path_to_database)

    def get_database(
            self
    ) -> h5py.File:
        """
        Gets h5py.File representing the database.

        Returns
        -------
        database : h5py.File
            Images and segmentations database.
        """
        if self._database_exists:
            return h5py.File(self.path_to_database, mode="r")
        else:
            raise FileExistsError(
                f"HDF5 file with path {self.path_to_database} doesn't exists. Use create_database before get_database.")

    def create_database(
            self,
            path_to_patients_folder: str,
            series_descriptions: Optional[Union[str, Dict[str, List[str]]]] = None,
            transformations: Optional[Sequence[BaseTransform]] = None,
            erase_unused_dicom_files: bool = False,
            overwrite_database: bool = False
    ) -> List[PatientWhoFailed]:
        """
        Creates an HDF dataset from multiple patients DICOM files.

        Parameters
        ----------
        path_to_patients_folder : str
            The path to the folder that contains all the patients' folders.
        series_descriptions : Optional[Union[str, Dict[str, List[str]]]], default = None.
            A dictionary that contains the series descriptions of the images that needs to be extracted from the
            patient's file. Keys are arbitrary names given to the images we want to add and values are lists of
            series descriptions. The images associated with these series descriptions do not need to have a
            corresponding segmentation. Note that it can be specified as a path to a json dictionary that contains the
            series descriptions.
        transformations : Optional[Sequence[BaseTransform]], default = None.
            A sequence of transformations to apply to images and segmentations.
        erase_unused_dicom_files: bool, default = False
            Whether to delete unused DICOM files or not. Use with caution.
        overwrite_database : bool, default = False.
            Overwrite existing database.

        Returns
        -------
        patients_who_failed : List[PatientWhoFailed]
            List of patients with one or more images not added to the HDF5 dataset due to the absence of the series in
            the patient record.
        """
        patients_who_failed = self.create(
            path_to_patients_folder=path_to_patients_folder,
            tags_to_use_as_attributes=[(0x0008, 0x103E), (0x0020, 0x000E), (0x0008, 0x0060)],
            series_descriptions=series_descriptions,
            transforms=transformations,
            erase_unused_dicom_files=erase_unused_dicom_files,
            overwrite_database=overwrite_database
        )

        return patients_who_failed
