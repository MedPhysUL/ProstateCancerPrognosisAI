from src.models.segmentation.hdf_dataset import HDFDataset
import matplotlib.pyplot as plt
from monai.transforms import AddChannel
import numpy as np
import h5py


ds = HDFDataset('C:/Users/rapha/Desktop/patients_dataset.h5')

print(len(ds))

print(len(ds[:-1]))





"""
a = [1,2,3,4,5,6]
print(a[-2:])   # [5,6]
print(a[:-2])   # [1,2,3,4]
"""


