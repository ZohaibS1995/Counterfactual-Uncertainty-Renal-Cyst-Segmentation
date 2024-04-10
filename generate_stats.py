import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label, binary_dilation
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_normal_histogram(data, bins=100):
    """
    Plot a normal histogram of the data.

    Parameters:
    - data: array-like, the main dataset to plot.
    - bins: int, the number of bins to use in the histogram.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, color='lightcoral')
    plt.title('Normal Scale Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def plot_histogram_with_log_scale(data, noise_level=0, bins=100):
    """
    Plot a histogram of the data with a logarithmic scale in terms of value,
    considering additive constant noise near 0.

    Parameters:
    - data: array-like, the main dataset to plot.
    - noise_level: float, the level of additive noise to apply near 0.
    - bins: int, the number of bins to use in the histogram.
    """
    # Apply additive constant noise near 0
    noise = np.random.uniform(-noise_level, noise_level, size=data.shape)
    data_with_noise = data + noise
    data_with_noise[data_with_noise < 0] = 0.01  # Replace negative values with a small positive constant

    # Plotting
    plt.figure(figsize=(8, 6))
    # Using log scale for values by applying log1p (log(x+1)) to avoid log(0)
    plt.hist(np.log1p(data_with_noise), bins=bins, color='darkturquoise')
    plt.title('Histogram with Logarithmic Scale in Terms of Value')
    plt.xlabel('Log(Value + 1)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

## labels path
orig_lab_path = r"/data/kidnAI/DynUnet/data/labelsTs"

# getting the label paths
orig_lab_files = os.listdir(orig_lab_path)

print("############# Printing results on Amsterdam files ################")

total_cysts = 0
thresh_detect = 200
voxel_volume = 0.77 * 0.77 * 2.5
vol_ams = []
for idx, file in enumerate(orig_lab_files):
    if "RADIOMICS" in file:
        lab_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(orig_lab_path, file))) == 1) * 1
        labeled_mask, num_features = label(lab_tmp)

        for i in range(1, num_features + 1):
            single_lesion = labeled_mask == i

            # Only consider lesions with pixel count greater than 10
            if np.sum(single_lesion) > thresh_detect:
                total_cysts += 1
                vol_ams.append(np.sum(single_lesion))

print(total_cysts)
print(np.mean(vol_ams)*voxel_volume)

print("############# Printing results on MUMC files ################")

total_cysts = 0
vol_mumc = []
for idx, file in enumerate(orig_lab_files):
    if "BOSNIAK" in file:
        lab_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(orig_lab_path, file))) == 1) * 1
        labeled_mask, num_features = label(lab_tmp)

        for i in range(1, num_features + 1):
            single_lesion = labeled_mask == i

            # Only consider lesions with pixel count greater than 10
            if np.sum(single_lesion) > thresh_detect:
                total_cysts += 1
                vol_mumc.append(np.sum(single_lesion))

print(total_cysts)
print(np.mean(vol_mumc)*voxel_volume)

orig_lab_path = r"/data/kidnAI/DynUnet/data/labelsTr_s"

# getting the label paths
orig_lab_files = os.listdir(orig_lab_path)

print("############# Printing results on KITS files ################")

total_cysts = 0
vol_kits = []
for idx, file in enumerate(orig_lab_files):
    if "KITS" in file:
        lab_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(orig_lab_path, file))) == 2) * 1
        labeled_mask, num_features = label(lab_tmp)

        for i in range(1, num_features + 1):
            single_lesion = labeled_mask == i

            # Only consider lesions with pixel count greater than 10
            if np.sum(single_lesion) > thresh_detect:
                total_cysts += 1
                vol_kits.append(np.sum(single_lesion))

print(total_cysts)
print(np.mean(vol_kits)*voxel_volume)

print("############# Printing results on UCSF files ################")

vol_ucsf = []
total_cysts = 0
for idx, file in enumerate(orig_lab_files):
    if "BOSNIAK" in file:
        lab_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(orig_lab_path, file))) == 2) * 1
        labeled_mask, num_features = label(lab_tmp)

        for i in range(1, num_features + 1):
            single_lesion = labeled_mask == i

            # Only consider lesions with pixel count greater than 10
            if np.sum(single_lesion) > thresh_detect:
                total_cysts += 1
                vol_ucsf.append(np.sum(single_lesion))

print(total_cysts)
print(np.mean(vol_ucsf)*voxel_volume)

vol = []
vol.extend(vol_ucsf)
vol.extend(vol_ams)
vol.extend(vol_kits)
vol.extend(vol_mumc)

vol = np.array(vol) * 0.77 * 0.77 * 2.5

print(np.percentile(vol, 33))
print(np.percentile(vol, 66))
