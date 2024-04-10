import os
import numpy as np
import glob
import matplotlib.pyplot as plt


def plot_slices_with_contours(array_3d, mask_3d, save_path="output.png"):
    # Determine the number of slices
    num_slices = array_3d.shape[2]

    # Create a figure for plotting
    fig, axes = plt.subplots(6, 8, figsize=(16, 12)) # Adjust this for different grid sizes
    axes = axes.flatten()

    for i in range(num_slices):
        ax = axes[i]
        slice_2d = array_3d[:, :, i]
        mask_slice = mask_3d[:, :, i]

        # Plot the slice
        ax.imshow(slice_2d, cmap='gray')

        # Plot contours for labels 1 and 2
        for label in [1, 2]:
            contour = mask_slice == label
            ax.contour(contour, colors=['red' if label == 1 else 'blue'], levels=[0.5])

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


fold_n = glob.glob(os.path.join(r"/data/kidnAI/DynUnet/data/uncertainty/predTs", "fold0_uncertain_*_*"))
store_path = os.path.join(r"/data/kidnAI/DynUnet/data/uncertainty/predTs", r"fold0_uncertain_var")
store_check_data_path = os.path.join(r"/data/kidnAI/DynUnet/data/uncertainty/predTs", r"check_data")
label_path = os.path.join(r"/data/kidnAI/DynUnet/data/uncertainty", r"labelsTs")

files_npy = os.listdir(fold_n[0])

for file_n in files_npy:
    file_names_npy = [os.path.join(x, file_n) for x in fold_n]
    np_array = np.stack([np.load(x)[2, ...] for x in file_names_npy])
    var_array = np.std(np_array, axis=0)
    np.save(os.path.join(store_path, file_n), var_array)

    # loading your masks
    mask_array = np.load(os.path.join(label_path, file_n))
    plot_slices_with_contours(var_array,
                              mask_array,
                              os.path.join(store_check_data_path, file_n.split(".")[0] + ".png"))

