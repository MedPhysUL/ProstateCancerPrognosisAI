#
# TO BE DELETED
# Purpose is to check how the segs are distributed along z in order to make an informed crop
#
import h5py
import matplotlib.pyplot as plt
import numpy as np

file = h5py.File('C:/Users/MALAR507/Documents/GitHub/ProstateCancerPrognosisAI/examples/local_data/patients_dataset.h5')

# en Z
hist_chill = []
hist_biz = []
patient_z_weird = []
z_seg_max_list = []
z_seg_min_list = []
z_seg_len_list = []

for idx, patient in enumerate(file.keys()):
    try:
        if file[patient]['0'].attrs['Modality'] == 'CT':
            print(f'----Patient {patient}----')
            print('Shape :', file[patient]['0']['image'].shape)
            # print('Max value :', np.amax(file[patient]['0']['image']))
            # print('Min value :', np.amin(file[patient]['0']['image']))
            z_seg = []
            for z in range(len(file[patient]['0']['image'][0, 0])):
                if np.amax(file[patient]['0']['Prostate_label_map'][:, :, z]) == 1:
                    z_seg.append(z)
            z_seg_max = max(z_seg)
            z_seg_min = min(z_seg)
            z_seg_len = z_seg_max - z_seg_min + 1
            z_seg_max_list.append(z_seg_max)
            z_seg_min_list.append(z_seg_min)
            z_seg_len_list.append(z_seg_len)
            print('z w/ prostate : de', z_seg_min, 'jusque a', z_seg_max)
            print('longueur de la prostate :', z_seg_len)
            if len(file[patient]['0']['image'][0, 0]) == 573:
                for i in z_seg:
                    hist_chill.append(i)
            if not len(file[patient]['0']['image'][0, 0]) == 573:
                patient_z_weird.append(patient)
                for i in z_seg:
                    hist_biz.append(i)
        if file[patient]['1'].attrs['Modality'] == 'CT':
            print(f'----Patient {patient}----')
            print('Shape :', file[patient]['1']['image'].shape)
            #print('Max value :', np.amax(file[patient]['1']['image']))
            #print('Min value :', np.amin(file[patient]['1']['image']))
            z_seg = []
            for z in range(len(file[patient]['1']['image'][0, 0])):
                if np.amax(file[patient]['1']['Prostate_label_map'][:, :, z]) == 1:
                    z_seg.append(z)

            z_seg_max = max(z_seg)
            z_seg_min = min(z_seg)
            z_seg_len = z_seg_max - z_seg_min + 1
            z_seg_max_list.append(z_seg_max)
            z_seg_min_list.append(z_seg_min)
            z_seg_len_list.append(z_seg_len)
            print('z w/ prostate : de', min(z_seg), 'jusque a', max(z_seg))
            print('longueur de la prostate :', z_seg_len)
            if len(file[patient]['1']['image'][0, 0]) == 573:
                for i in z_seg:
                    hist_chill.append(i)
            if not len(file[patient]['1']['image'][0, 0]) == 573:
                patient_z_weird.append(patient)
                for i in z_seg:
                    hist_biz.append(i)
    except KeyError:
        print(f'Patient {patient} ignored. pk y marche po?')
    # if idx == 5:
    #     break

print('patients avec un z biz :', patient_z_weird)

plt.hist(hist_biz, bins=25)
plt.show()
plt.hist(hist_chill, bins=25)
plt.show()
print('en moyenne la prostate commence a :', np.mean(z_seg_min_list), 'et termine a :', np.mean(z_seg_max_list))
print('la plus basses prostate commence a :', min(z_seg_min_list), 'et la plus haute termine a :', max(z_seg_max_list))
print('en moyenne la longueur est :', np.mean(z_seg_len_list))
print('la prostate la plus courte est de length', min(z_seg_len_list), 'et la plus longue est de length', max(z_seg_len_list))

'''

# en X, Y
hist_x = []
hist_y = []
for patient in file.keys():
    try:
        if file[patient]['0'].attrs['Modality'] == 'CT':
            print(f'----Patient {patient}----')
            seg_x = []
            seg_y = []
            for x in range(len(file[patient]['0']['image'][:, 0, 0])):
                if np.amax(file[patient]['0']['Prostate_label_map'][x, :, :]) == 1:
                    seg_x.append(x)
            for i in seg_x:
                hist_x.append(i)
            if min(seg_x) < 50:
                print(patient, 'a un x trop petit')
            if max(seg_x) > 400:
                print(patient, 'a un x trop grand')
            for y in range(len(file[patient]['0']['image'][0, :, 0])):
                if np.amax(file[patient]['0']['Prostate_label_map'][:, y, :]) == 1:
                    seg_y.append(y)
            for i in seg_y:
                hist_y.append(i)
        if file[patient]['1'].attrs['Modality'] == 'CT':
            print(f'----Patient {patient}----')
            seg_x = []
            seg_y = []
            for x in range(len(file[patient]['1']['image'][:, 0, 0])):
                if np.amax(file[patient]['1']['Prostate_label_map'][x, :, :]) == 1:
                    seg_x.append(x)
            for i in seg_x:
                hist_x.append(i)
            for y in range(len(file[patient]['1']['image'][0, :, 0])):
                if np.amax(file[patient]['1']['Prostate_label_map'][:, y, :]) == 1:
                    seg_y.append(y)
            if min(seg_y) < 50:
                print(patient, 'a un y trop petit')
            if max(seg_y) > 400:
                print(patient, 'a un y trop grand')
            for i in seg_y:
                hist_y.append(i)
    except KeyError:
        print(patient, 'marche po:(')
plt.title('x')
plt.hist(hist_x, bins=60)
plt.show()
plt.title('y')
plt.hist(hist_y, bins=60)
plt.show()

'''

'''
print(np.nonzero(file['TEP-404']['0']['Prostate_label_map']))

print(file['TEP-404']['0']['Prostate_label_map'][0, 0, 572])

plt.imshow(file['TEP-404']['0']['Prostate_label_map'][:, :, 572], cmap="gray")
plt.show()
'''