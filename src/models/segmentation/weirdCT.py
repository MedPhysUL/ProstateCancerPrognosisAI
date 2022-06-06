import h5py
import matplotlib.pyplot as plt
import numpy as np




file = h5py.File('C:/Users/MALAR507/Documents/GitHub/ProstateCancerPrognosisAI/examples/local_data/patients_dataset.h5')

plt.imshow(file['TEP-047']['0']['image'][167:300, 167:300, 100], cmap='gray')
plt.show()
plt.imshow(file['TEP-047']['0']['image'][:, :, 100], cmap='gray')
plt.show()
for patient in file.keys():
    try:
        if file[patient]['0'].attrs['Modality'] == 'CT':
            if np.amax(file[patient]['0']['Prostate_label_map'][:, :, 0:50]) > 0:
                print('patient', patient, 'a une seg proche de 0')
            if np.amax(file[patient]['0']['Prostate_label_map'][:, :, 300:]) > 0:
                print('patient', patient, 'a une prostate haute')
        if file[patient]['1'].attrs['Modality'] == 'CT':
            if np.amax(file[patient]['1']['Prostate_label_map'][:, :, 0:50]) > 0:
                print('patient', patient, 'a une seg proche de 0')
            if np.amax(file[patient]['1']['Prostate_label_map'][:, :, 300:]) > 0:
                print('patient', patient, 'a une prostate haute')
    except KeyError:
        print(f'Patient {patient} marche po')



