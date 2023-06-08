from delia.databases import PatientsDatabase
from delia.extractors import PatientsDataExtractor
import h5py
from delia.transforms import ResampleD, MatchingResampleD
from monai.transforms import (
    CenterSpatialCropD,
    ScaleIntensityD,
)

import matplotlib.pyplot as plt
import numpy as np

from monai.transforms import Compose

# Variables to be used in function arguments, can be directly added within the function, but reduces redundancy
image_keys = ["CT"]
series = {"CT": [
            "THORAX 1.0 B45f",
            "CHEST 1.25 MM",
            "LUNG WINDOW",
            "THIN LUNG WINDOW",
            "Thorax  1.0  B45f",
            ".625 mm Chest",
            "THINS",
            "Recon 2: CHEST",
            "Recon 3: CT CHEST W/O",
            "0.625  DMPR On + SS50",
            "Recon 2: CAP",
            "1.25MM CHEST BONE",
            "CHEST 1.0  B45f",
            "LUNG 1MM B45f",
            "1.25MM CHEST BONE PLUS",
            "CT Thick Axials 2.5mm",
            "0.625MM CHEST NO PACS",
            "CHEST LUNG",
            "THIN CHEST LUNG",
            "SUPER",
            "CHEST NON-CON",
            "AX 1.25",
            "THIN",
            "IN REACH",
            "CHEST 1.25MM",
            "LUNG",
            "Chest wo  3.0  B40f"
        ]}
seg_keys = ["Heart", "Segmentation", "Tissue"]
keys = image_keys + seg_keys
path_to_patients_folder = "/Users/felixdesroches/Desktop/Stages et notes/Stage -E23/brain_segmentation_E23/lung_data/manifest-1685649685383/patients"

# Variable to decide if I want to apply transforms (True), use all data (False) or use an already created file (None).
need_transform = None

# Variable to decide if I want to plot some of the scans (True) or not (False).
graph_scans = True

if need_transform is None:  # Skips file creation, uses pre-existing file with the desired name
    pass
elif need_transform:  # Creates a file with transforms and selected data
    # Creates a list of transformations to compose later.
    transform_list = [
                ResampleD(keys=image_keys, out_spacing=(1.5, 1.5, 1.5)),
                MatchingResampleD(reference_image_key="CT", matching_keys=seg_keys),
                CenterSpatialCropD(keys=keys, roi_size=(192, 192, 192)),
                ScaleIntensityD(keys=image_keys, minv=-1, maxv=1)]

    # Creates a PatientsDataExtractor object using the path to the "patients" folder.
    patients_data_extractor = PatientsDataExtractor(
        path_to_patients_folder=path_to_patients_folder,
        series_descriptions=series,
        transforms=Compose(transform_list))

    # Creates a PatientsDatabase object to store the data, the path_to_database will decide where the file is created.
    # In this case the database will be located within the same folder.
    database = PatientsDatabase(path_to_database="patients_database.h5")

    # Creates the database
    database.create(
        patients_data_extractor=patients_data_extractor,
        overwrite_database=True,
        tags_to_use_as_attributes=[(0x0008, 0x103E), (0x0020, 0x000E), (0x0008, 0x0060)]
    )

    # Closes the database to allow reopening in any mode.
    database.close()

    # The following lines allow for the segmentation files to be renamed "lung" as that was the case for my data, the
    # name can be changed or the whole section can be removed to keep the original names. Some errors may appear in
    # these lines if all patients do not have a segmentation or if only a part of the patients have the desired
    # Series Description. However, the database is still created beforehand and should work without any issues.
    with h5py.File("patients_database_2.h5", "r+") as f:
        for key in list(f.keys()):
            group = f[key]["0"]["0"]
            group.move(list(group.keys())[0], "Lung")

elif not need_transform:  # Creates a file with all data and no transforms
    patients_data_extractor = PatientsDataExtractor(
        path_to_patients_folder=path_to_patients_folder,
        series_descriptions=series
    )
    database = PatientsDatabase(path_to_database="patients_database.h5")
    database.create(
        patients_data_extractor=patients_data_extractor,
        overwrite_database=True,
        tags_to_use_as_attributes=[(0x0008, 0x103E), (0x0020, 0x000E), (0x0008, 0x0060)]
    )


if graph_scans:  # Shows a selected portion of the images of a patient for visualisation.

    # Name of the file to view. If need_transforms is not None, then the same name will show the created database.
    filename = 'patients_database_2.h5'

    # Name of the array to show, "Image" will show the patient and "SEG" will show the segmentation
    array_name = "Image"
    if array_name == "SEG":  # extracts the segmentation as a numpy array
        with h5py.File(filename, "r") as f:
            for key in list(f.keys()):  # Iterates over all patients.
                patient_data = f[key]["0"]["0"]
                patient_seg = patient_data[list(patient_data.keys())[0]]
                data_4 = np.array(patient_seg)
                print(data_4.shape)

                precision = 2  # Spacing between each slice viewed, 1 means all slices are selected.
                subplot_x = 10  # Number of slices ploted per line.
                subplot_y = (data_4.shape[2] // precision) // subplot_x  # Number of lines of plots.
                fig, arr = plt.subplots(subplot_y, subplot_x)

                for i in range(subplot_y * subplot_x):  # Iterates over all elements of the figure.
                    if precision * i > data_4.shape[2]:  # If the position is out of bounds for that patient, stops.
                        continue
                    arr[i % subplot_y, i // subplot_y].imshow(data_4[:, :, i * precision], cmap="gray")
                    arr[i % subplot_y, i // subplot_y].axis("off")
                plt.show()

    if array_name == "Image":  # extracts the patient as a numpy array
        with h5py.File(filename, "r") as f:
            for key in list(f.keys()):  # Iterates over all patients.
                patient_data = f[key]["0"]["Image"]
                data_4 = np.array(patient_data)
                print(data_4.shape)

                precision = 2  # Spacing between each slice viewed, 1 means all slices are selected.
                subplot_x = 10  # Number of slices ploted per line.
                subplot_y = (data_4.shape[2] // precision) // subplot_x  # Number of lines of plots.
                fig, arr = plt.subplots(subplot_y, subplot_x)

                for i in range(subplot_y * subplot_x):  # Iterates over all elements of the figure.
                    if precision * i > data_4.shape[2]:  # If the position is out of bounds for that patient, stops.
                        continue
                    arr[i % subplot_y, i // subplot_y].imshow(data_4[:, :, i * precision], cmap="gray")
                    arr[i % subplot_y, i // subplot_y].axis("off")
                plt.show()
