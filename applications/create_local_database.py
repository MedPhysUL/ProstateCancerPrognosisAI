"""
    @file:              create_local_database.py
    @Author:            Maxence Larose
    @Creation Date:     06/2022
    @Last modification: 06/2022
    @Description:       This file contains the LocalExtractor class which is used to create an HDF5 database containing
                        the images and their segmentations.
"""

import env_apps

from dicom2hdf import transforms

from src.data.extraction.local import LocalExtractor


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------- #
    #                                               Logs Setup                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    env_apps.configure_logging("logging_conf.yaml")

    # ----------------------------------------------------------------------------------------------------------- #
    #                                             Create Database                                                 #
    # ----------------------------------------------------------------------------------------------------------- #
    extractor = LocalExtractor(path_to_database=r"local_data/learning_set.h5")

    patients_who_failed = extractor.create_database(
        path_to_patients_folder=r"local_data/Learning_set",
        series_descriptions=r"local_data/series_descriptions.json",
        transformations=[transforms.Resample((1.5, 1.5, 1.5))]
    )
