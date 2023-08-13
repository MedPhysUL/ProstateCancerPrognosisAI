
import env_apps
import json
import os

from delia.databases import PatientsDatabase
from delia.radiomics import RadiomicsDataset, RadiomicsFeatureExtractor
from monai.data import DataLoader
import SimpleITK as sitk
import torch

from constants import (
    AUTOMATIC_RADIOMICS_HOLDOUT_PATH,
    CT_FEATURES_EXTRACTOR_PARAMS_PATH,
    HOLDOUT_MASKS_PATH,
    LEARNING_SET_PATH,
    PROSTATE_SEGMENTATION_TASK,
    PT_FEATURES_EXTRACTOR_PARAMS_PATH,
    SEED
)
from src.data.datasets import ImageDataset, ProstateCancerDataset
from src.data.processing.sampling import Mask
from src.models.torch.segmentation import Unet


def save_radiomics(dataset: ProstateCancerDataset, radiomics_dataset: RadiomicsDataset, modality: str):
    radiomics_dataset.extractor = RadiomicsFeatureExtractor(
        path_to_params=CT_FEATURES_EXTRACTOR_PARAMS_PATH if modality == "CT" else PT_FEATURES_EXTRACTOR_PARAMS_PATH,
        geometryTolerance=1e-4
    )

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=None)
    radiomics_features = {}
    for idx, (features, _) in enumerate(data_loader):
        pred = model.predict(features=features, probability=False)
        mask = pred["SegmentationTask('modality'='CT', 'organ'='Prostate')"]

        radiomics = radiomics_dataset.extractor.execute(
            imageFilepath=sitk.GetImageFromArray(
                arr=features.image[f"{modality}"].cpu().detach().numpy()[0, 0, ...],
                isVector=False
            ),
            maskFilepath=sitk.GetImageFromArray(
                arr=mask.cpu().detach().numpy()[0, 0, ...],
                isVector=False
            ),
            label=None,
            label_channel=None,
            voxelBased=False
        )

        radiomics_features[f"{idx}"] = radiomics

    radiomics_dataset.save(radiomics_features)


if __name__ == '__main__':
    database = PatientsDatabase(path_to_database=LEARNING_SET_PATH)

    image_dataset = ImageDataset(
        database=database,
        modalities={"CT", "PT"},
        tasks=PROSTATE_SEGMENTATION_TASK
    )

    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=None)

    model_state = torch.load(os.path.join(AUTOMATIC_RADIOMICS_HOLDOUT_PATH, "model.pt"))["model_state"]

    masks = json.load(open(HOLDOUT_MASKS_PATH, "r"))
    dataset.update_masks(
        train_mask=masks[Mask.TRAIN],
        valid_mask=masks[Mask.VALID]
    )

    model = Unet(
        image_keys="CT",
        num_res_units=3,
        dropout=0.2,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    model.load_state_dict(model_state)

    radiomics_dataset = RadiomicsDataset(
        path_to_dataset=os.path.join(AUTOMATIC_RADIOMICS_HOLDOUT_PATH, f"ct_learning.csv")
    )
    save_radiomics(dataset, radiomics_dataset, "CT")

    radiomics_dataset = RadiomicsDataset(
        path_to_dataset=os.path.join(AUTOMATIC_RADIOMICS_HOLDOUT_PATH, f"pt_learning.csv")
    )
    save_radiomics(dataset, radiomics_dataset, "PT")
