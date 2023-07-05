"""
    @file:              00_create_database.py
    @Author:            Maxence Larose

    @Creation Date:     06/2022
    @Last modification: 11/2022

    @Description:       This file shows how to create an HDF5 file database using DELIA.
"""

import env_apps

from delia.databases import PatientsDatabase
from delia.extractors import PatientsDataExtractor
from delia.transforms import (
    MatchingCentroidSpatialCropD,
    MatchingCropForegroundD,
    MatchingResampleD,
    PETtoSUVD,
    ResampleD
)
from monai.transforms import (
    Compose,
    KeepLargestConnectedComponentD,
    ScaleIntensityRangeD,
    SpatialCropD
)

from constants import (
    LEARNING_SET_FOLDER_PATH,
    LEARNING_SET_PATH,
    LOGGING_CONFIG_PATH,
    SERIES_DESCRIPTIONS_PATH
)


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------- #
    #                                               Logs Setup                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    env_apps.configure_logging(LOGGING_CONFIG_PATH)

    # ----------------------------------------------------------------------------------------------------------- #
    #                                               Transforms                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    transforms = Compose([
        ResampleD(keys=["CT"], out_spacing=(1.0, 1.0, 1.0)),
        MatchingResampleD(reference_image_key="CT", matching_keys=["PT", "Prostate"]),
        MatchingCropForegroundD(reference_image_key="CT", matching_keys=["PT", "Prostate"]),
        SpatialCropD(keys=["CT", "PT", "Prostate"], roi_slices=[slice(30, 740), slice(None), slice(None)]),
        KeepLargestConnectedComponentD(keys=["Prostate"]),
        MatchingCentroidSpatialCropD(segmentation_key="Prostate", matching_keys=["CT", "PT"], roi_size=(128, 128, 128)),
        PETtoSUVD(keys=["PT"]),
        ScaleIntensityRangeD(keys=["CT"], a_min=-200, a_max=250, b_min=0, b_max=1, clip=True),
        ScaleIntensityRangeD(keys=["PT"], a_min=0, a_max=25, b_min=0, b_max=1, clip=True)
    ])

    # ----------------------------------------------------------------------------------------------------------- #
    #                                      Create patients data extractor                                         #
    # ----------------------------------------------------------------------------------------------------------- #
    patients_data_extractor = PatientsDataExtractor(
        path_to_patients_folder=LEARNING_SET_FOLDER_PATH,
        tag_values=SERIES_DESCRIPTIONS_PATH,
        transforms=transforms
    )

    # ----------------------------------------------------------------------------------------------------------- #
    #                                                Create database                                              #
    # ----------------------------------------------------------------------------------------------------------- #
    database = PatientsDatabase(path_to_database=LEARNING_SET_PATH)

    database.create(
        patients_data_extractor=patients_data_extractor,
        organs_to_keep="Prostate",
        tags_to_use_as_attributes=[(0x0008, 0x103E), (0x0020, 0x000E), (0x0008, 0x0060)]
    )
