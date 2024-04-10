from scipy.ndimage import label, find_objects
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label, find_objects
import math
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm

def calculate_bounding_box(image):
    # Find the indices where the image is non-zero
    indices = np.nonzero(image)

    # Calculate the minimum and maximum indices along each dimension
    bbox = tuple(slice(indices[dim].min(), indices[dim].max() + 1) for dim in range(3))

    return bbox

def extract_rois(lab_arr, box_max=[48, 128, 128]):
    # Label the connected components
    labeled_array, num_features = label(lab_arr)

    offset_rois = []
    size_rois = []

    for idx in range(1, num_features + 1):
        temp_label = (labeled_array == idx) * 1
        box = calculate_bounding_box(temp_label)

        bbox = []
        for i in range(3):
            center = (box[i].start + box[i].stop) // 2
            half_size = min(box_max[i] // 2, temp_label.shape[i] // 2)

            i_bbox_min = max(center - half_size, 0)
            i_bbox_max = min(center + half_size, temp_label.shape[i])

            # Adjust if the box is larger than the image
            if i_bbox_max - i_bbox_min > box_max[i]:
                if i_bbox_min == 0:
                    i_bbox_max = box_max[i]
                else:
                    i_bbox_min = i_bbox_max - box_max[i]

            bbox.append(slice(int(i_bbox_min), int(i_bbox_max)))

        bbox = tuple(bbox)  # the final bounding box

        offset = np.array([bbox[i].start for i in range(3)], dtype=np.int32)
        size = np.array([bbox[i].stop - bbox[i].start for i in range(3)], dtype=np.int32)

        offset_rois.append(offset)
        size_rois.append(size)

    return offset_rois, size_rois


from scipy.ndimage import label
import numpy as np

def extract_rois_with_arr(lab_arr, box_max=[48, 128, 128]):
    # Label the connected components
    labeled_array, num_features = label(lab_arr)

    offset_rois = []
    size_rois = []
    connected_components = []  # List to store cropped connected components

    for idx in range(1, num_features + 1):
        temp_label = (labeled_array == idx) * 1
        box = calculate_bounding_box(temp_label)

        bbox = []
        for i in range(3):
            center = (box[i].start + box[i].stop) // 2
            half_size = min(box_max[i] // 2, temp_label.shape[i] // 2)

            i_bbox_min = max(center - half_size, 0)
            i_bbox_max = min(center + half_size, temp_label.shape[i])

            # Adjust if the box is larger than the image
            if i_bbox_max - i_bbox_min > box_max[i]:
                if i_bbox_min == 0:
                    i_bbox_max = box_max[i]
                else:
                    i_bbox_min = i_bbox_max - box_max[i]

            bbox.append(slice(int(i_bbox_min), int(i_bbox_max)))

        cropped_component = temp_label[bbox[0], bbox[1], bbox[2]]
        connected_components.append(cropped_component)  # Add the cropped component to the list

        bbox = tuple(bbox)  # the final bounding box

        offset = np.array([bbox[i].start for i in range(3)], dtype=np.int32)
        size = np.array([bbox[i].stop - bbox[i].start for i in range(3)], dtype=np.int32)

        offset_rois.append(offset)
        size_rois.append(size)

    return offset_rois, size_rois, connected_components


def save_slices_as_grid(array_3d, filename='slices_grid.png'):
    # Determine the number of slices
    num_slices = array_3d.shape[2]

    # Calculate the grid size
    grid_cols = math.ceil(math.sqrt(num_slices))
    grid_rows = math.ceil(num_slices / grid_cols)

    # Create a blank array to hold the grid
    slice_height, slice_width = array_3d.shape[0], array_3d.shape[1]
    grid_height = slice_height * grid_rows
    grid_width = slice_width * grid_cols
    grid_array = np.zeros((grid_height, grid_width), dtype=array_3d.dtype)

    # Copy slices into the grid array
    for i in range(num_slices):
        row = i // grid_cols
        col = i % grid_cols
        grid_array[row * slice_height:(row + 1) * slice_height, col * slice_width:(col + 1) * slice_width] = array_3d[:,
                                                                                                             :, i]

    # Normalize the grid array to the range [0, 1] if it's not already
    if np.amin(grid_array) < 0 or np.amax(grid_array) > 1:
        grid_array = (grid_array - np.amin(grid_array)) / (np.amax(grid_array) - np.amin(grid_array))

    # Use matplotlib to save the array as a PNG image
    plt.imshow(grid_array, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def pad_array_to_size(arr, target_size=[128, 128, 48]):
    # Calculate the padding needed for each dimension
    padding = [(max(0, (target_size[i] - arr.shape[i]) // 2),
                max(0, (target_size[i] - arr.shape[i] + 1) // 2)) for i in range(len(target_size))]

    # Pad the array
    padded_arr = np.pad(arr, padding, mode='constant', constant_values=0)

    return padded_arr

img_path = r"/data/kidnAI/DynUnet/data/imagesTs"
gt_path = r"/data/kidnAI/DynUnet/data/labelsTs"
store_img_path = r"/data/kidnAI/DynUnet/data/uncertainty/imagesTs"
store_gt_path = r"/data/kidnAI/DynUnet/data/uncertainty/labelsTs"
base_path_img = r"/home/zohaib/Desktop/check_data"

for f in tqdm(os.listdir(gt_path)):

    gt_sitk = sitk.ReadImage(os.path.join(gt_path, f))
    img_sitk = sitk.ReadImage(os.path.join(img_path, f.replace(".nii.gz", "_0000.nii.gz")))

    gt_roi = sitk.GetArrayFromImage(gt_sitk)
    offset_rois, size_rois, connected_comp = extract_rois_with_arr(gt_roi)

    for idx in range(len(offset_rois)):

        offset_t = offset_rois[idx][::-1].tolist()
        size_rois_t = size_rois[idx][::-1].tolist()

        patch_gt = sitk.RegionOfInterest(gt_sitk, size_rois_t, offset_t)
        patch_img = sitk.RegionOfInterest(img_sitk, size_rois_t, offset_t)

        patch_img_arr = sitk.GetArrayFromImage(patch_img).transpose()
        patch_gt_arr = sitk.GetArrayFromImage(patch_gt).transpose()

        patch_img_arr = pad_array_to_size(patch_img_arr)
        patch_gt_arr = pad_array_to_size(patch_gt_arr)#connected_comp[idx].transpose())

        f_t = f.split(".")[0]

        #save_slices_as_grid(patch_img_arr, os.path.join(base_path_img, f"img_{f_t}_{idx}.png"))
        #save_slices_as_grid(patch_gt_arr, os.path.join(base_path_img, f"lab_{f_t}_{idx}.png"))
        np.save(os.path.join(store_img_path, f"{f_t}_{idx}"), patch_img_arr)
        np.save(os.path.join(store_gt_path, f"{f_t}_{idx}"), patch_gt_arr)


img_path = r"/data/kidnAI/DynUnet/data/imagesTr"
gt_path = r"/data/kidnAI/DynUnet/data/labelsTr"
store_img_path = r"/data/kidnAI/DynUnet/data/uncertainty/imagesTr"
store_gt_path = r"/data/kidnAI/DynUnet/data/uncertainty/labelsTr"
fold_path = r"/data/kidnAI/DynUnet/data/pred_fold0"

fold_files = os.listdir(fold_path)

for f in tqdm(fold_files):

    gt_sitk = sitk.ReadImage(os.path.join(gt_path, f))
    img_sitk = sitk.ReadImage(os.path.join(img_path, f.replace(".nii.gz", "_0000.nii.gz")))

    gt_roi = sitk.GetArrayFromImage(gt_sitk)
    offset_rois, size_rois, connected_comp = extract_rois_with_arr(gt_roi == 2)

    for idx in range(len(offset_rois)):

        offset_t = offset_rois[idx][::-1].tolist()
        size_rois_t = size_rois[idx][::-1].tolist()

        patch_gt = sitk.RegionOfInterest(gt_sitk, size_rois_t, offset_t)
        patch_img = sitk.RegionOfInterest(img_sitk, size_rois_t, offset_t)

        patch_img_arr = sitk.GetArrayFromImage(patch_img).transpose()
        patch_gt_arr = sitk.GetArrayFromImage(patch_gt).transpose()

        patch_img_arr = pad_array_to_size(patch_img_arr)
        patch_gt_arr = pad_array_to_size(patch_gt_arr)#connected_comp[idx].transpose())

        f_t = f.split(".")[0]

        #save_slices_as_grid(patch_img_arr, os.path.join(base_path_img, f"img_{f_t}_{idx}.png"))
        #save_slices_as_grid(patch_gt_arr, os.path.join(base_path_img, f"lab_{f_t}_{idx}.png"))
        np.save(os.path.join(store_img_path, f"{f_t}_{idx}"), patch_img_arr)
        np.save(os.path.join(store_gt_path, f"{f_t}_{idx}"), patch_gt_arr)
