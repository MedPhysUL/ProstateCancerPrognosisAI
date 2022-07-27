"""
    @file:              local.py
    @Author:            Maxence Larose

    @Creation Date:     06/2022
    @Last modification: 07/2022

    @Description:       This file contains the LocalDatabaseManager class which is used to create and/or interact with
                        an HDF5 file containing the patients' images and their segmentations.
"""

from typing import Dict, List, Optional, Sequence, Union

from dicom2hdf import PatientsDataset, PatientWhoFailed
from dicom2hdf.processing.transforms import BaseTransform


class LocalDatabaseManager(PatientsDataset):
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
        super().__init__(path_to_dataset=path_to_database)

    def create_database(
            self,
            path_to_patients_folder: str,
            series_descriptions: Optional[Union[str, Dict[str, List[str]]]] = None,
            transformations: Optional[Sequence[BaseTransform]] = None,
            erase_unused_dicom_files: bool = False,
            overwrite_dataset: bool = False
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
        overwrite_dataset : bool, default = False.
            Overwrite existing dataset.

        Returns
        -------
        patients_who_failed : List[PatientWhoFailed]
            List of patients with one or more images not added to the HDF5 dataset due to the absence of the series in
            the patient record.
        """
        patients_who_failed = self.create_hdf5_dataset(
            path_to_patients_folder=path_to_patients_folder,
            tags_to_use_as_attributes=[(0x0008, 0x103E), (0x0020, 0x000E), (0x0008, 0x0060)],
            series_descriptions=series_descriptions,
            transforms=transformations,
            erase_unused_dicom_files=erase_unused_dicom_files,
            overwrite_dataset=overwrite_dataset
        )

        return patients_who_failed
