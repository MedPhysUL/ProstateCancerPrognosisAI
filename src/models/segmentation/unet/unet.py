"""
    @file:              hdf_dataset.py
    @Author:            Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 05/2022

    @Description:       This file contains an implementation of a U-Net.

"""
import numpy as np

from src.models.segmentation.hdf_dataset import HDFDataset

import matplotlib.pyplot as plt
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import AddChannel, Compose, ToTensor
from monai.utils import first, progress_bar, set_determinism
import torch


if __name__ == "__main__":

    # Set determinism
    set_determinism(seed=1010710)

    # Setting up
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = 0 # semble runner sans se locker lorsque l'on met plus que 0 workers
    num_val = 25
    batch_size = 1
    num_epochs = 2
    lr = 5e-3

    # Setting up data

    img_trans = Compose([AddChannel(), ToTensor(dtype=torch.float32)])
    seg_trans = Compose([AddChannel(), ToTensor(dtype=torch.float32)])

    ds = HDFDataset('C:/Users/MALAR507/Documents/GitHub/ProstateCancerPrognosisAI/examples/local_data/patients_dataset.h5',  img_transform=img_trans, seg_transform=seg_trans)
    # ds = HDFDataset('C:/Users/rapha/Desktop/patients_dataset.h5', img_transform=img_trans, seg_transform=seg_trans)
    # Visualise
    # print('ds shape:', np.shape(ds))
    # w/ aucune transformation : (Patient, Channels, Z, X,Y). ZXY est en array
    # w/ AddChannel() : (140, 2, 1, 573, 333, 333) -- (P, C, Added, Z, X, Y). AZXY est en array



    train_ds = ds[:-num_val]
    val_ds = ds[-num_val:]
    # Visualise
    # print('train_ds shape:', np.shape(train_ds))
    # (Patient, Channels, Z, X, Y)
    # w/ AddChannel() : (115, 2, 1, Z, X, Y) -- (P, C, Added, Z, X, Y)

    train_loader = DataLoader(
         dataset=train_ds,
         num_workers=num_workers,
         batch_size=batch_size
    )
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        num_workers=num_workers
    )
    #Visualize
    # print('val_loader shape:', np.shape(val_loader))
    # () -- marche pas
    # w/ AddChannel() : () -- marche pas
    # print('val_loader length:', len(val_loader))
    # 7 -- il s'agit du nombres de batches
    # w/ AddChannel() : 7 -- idem
    '''
    for i in val_loader:
        # print('length of i', len(i))
        # 2 -- image et seg
        # w/ AddChannel() : 2 -- idem
        for j in i:
            # print('length of j', len(j))
            # 4 -- nbr de patients dans la batch (pcq la 7e length est 1, comme on peut s'y attendre)
            # w/ AddChannel() :  4 -- idem
            for k in j:
                # print('shape de k', np.shape(k))
                # (Z, X, Y) -- dimensions de nos images
                # w/ AddChannel() : (1, 573, 333, 333) -- (Added, Z, X, Y)
                pass
    '''
    # Donc post DataLoader on a :
    # val_loader est de shape (Batch, Channels, Patients de la batch, Z, X, Y)
    # w/ AddChannel() : val_loader est de shape (Batch, Channels, Patients de la batch, Added, Z, X, Y)

    # Visualisation d'une image post DataLoader. Aussi, demonstration d'une utilisation de first() :
    # check_image, check_segmentation = first(val_loader)
    # print(check_image.shape)
    # plt.imshow(check_image[0, 120], cmap='gray')       # w/ AddChannel() : doit s'occuper du added donc serait par ex
    # plt.imshow(check_segmentation[0, 120], alpha=0.1)  # plt.imshow(check_image[0, 0, 120]) et idem pour segmentation
    # plt.show()
    # Donne une belle image et sa segmentation en overlay

    batch = first(train_loader)       # Donc batch est de shape (Channels, Patient de la batch, Z, X, Y)
    # print(batch[0][0, 115].shape) # (333, 333) -- (X, Y). En effet, batch[img/seg][patient de la batch, Z, X, Y]
    # plt.imshow(batch[0][0, 115])  # Donne la CT
    # plt.imshow(batch[1][0, 115])  # Donne la seg
    # plt.show()

    # Training

    net = UNet(
        dimensions=3,
        in_channels=1,                  # lors de la prediction, tu feed une image (1 channel)
        out_channels=1,                 # et tu obtiens une seg (1 channel. Ce que eux nomment channels est features.
        channels=(8, 16, 32, 64, 128),     # default : (8, 16, 32, 64, 128)
        strides=(1, 1, 1, 1)
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr)
    loss = DiceLoss(sigmoid=True)
    metric = DiceMetric(
        include_background=True, reduction='mean'
    )
    # jusque ici pas de bugs lorsque roulé

    step_losses = []
    epoch_metrics = []
    total_step = 0

    for epoch in range(num_epochs):
        net.train()

        # train network with training images
        for batch in train_loader:
            batch_images = batch[0].to(device)
            batch_segs = batch[1].to(device)
            # print(batch_images.shape)
            #
            # w/ AddChannel() : (2, 1, 40, 233, 233) -- (patients de la batch, channel(i/s), Z, X, Y)

            opt.zero_grad()
            ############################# RuntimeError: Given groups=1, weight of size [8, 1, 3, 3, 3],
            y_pred = net(batch_images)  #  expected input[1, 4, 573, 333, 333] to have 1 channels,testé: le 8 correspond au premier feature nbr dans UNET
            #############################  but got 4 channels instead. 4 est pour les 4 patients d'une batch et 1 est channel(img/seg)

            # binarizing
            y_pred = torch.sigmoid(y_pred)

            loss_val = loss(y_pred, batch_segs)
            # print(loss_val)
            loss_val.backward()
            opt.step()

            step_losses.append((total_step, loss_val.item()))
            total_step += 1

        net.eval()
        metric_vals = []

        # test our network using the validation dataset
        with torch.no_grad():
            for batch_images, batch_segs in val_loader:
                batch_images = batch_images.to(device)
                batch_segs = batch_segs.to(device)
                #Visualize
                # print('batch_image de val_loader est de shape :', batch_images.shape)
                # (3, 573, 333, 333) sur THE WORST -- (Patients de la batch?, Z, X, Y)
                # w/ AddChannel() : (3, 1, 573, 333, 333) sur THE WORST -- (Patients de la batch, Added, Z, X, Y)

                y_pred = net(batch_images)

                #binarizing
                y_pred = torch.sigmoid(y_pred)
                y_pred = torch.round(y_pred)

                pred_metric = metric(y_pred=y_pred, y=batch_segs)
                metric_vals += [i for i in pred_metric.cpu().data.numpy().flatten().tolist()]
                # print('metric_vals :', metric_vals)

        print(np.mean(metric_vals))
        epoch_metrics.append((total_step, np.average(metric_vals)))
        progress_bar(epoch + 1, num_epochs, f'Validation metric: {epoch_metrics[-1][1]}')































