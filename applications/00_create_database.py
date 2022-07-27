"""
    @file:              create_database.py
    @Author:            Maxence Larose

    @Creation Date:     06/2022
    @Last modification: 07/2022

    @Description:       This file shows how to create an HDF5 file database using a LocalDatabaseManager.
"""

import env_apps

from dicom2hdf import transforms

from src.data.extraction.local import LocalDatabaseManager


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------- #
    #                                               Logs Setup                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    env_apps.configure_logging("logging_conf.yaml")

    # ----------------------------------------------------------------------------------------------------------- #
    #                                             Create Database                                                 #
    # ----------------------------------------------------------------------------------------------------------- #
    database_manager = LocalDatabaseManager(path_to_database=r"local_data/learning_set.h5")

    database_manager.create_database(
        path_to_patients_folder=r"local_data/Learning_set",
        series_descriptions=r"local_data/series_descriptions.json",
        transformations=[transforms.Resample((1.5, 1.5, 1.5))],
        erase_unused_dicom_files=True
    )
