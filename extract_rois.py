from scipy.ndimage import label, find_objects
import os
import SimpleITK as sitk
import numpy as np
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


def extract_two_largest_components(array):
    # Label the connected components
    labeled_array, num_features = label(array)

    # Skip if no features found
    if num_features == 0:
        return None, None

    # Find the sizes of the components
    sizes = np.bincount(labeled_array.ravel())
    # Set size of background (label 0) to 0
    sizes[0] = 0

    # Find the labels of the two largest components
    largest_labels = sizes.argsort()[-2:][::-1]

    # Extract slices corresponding to the two largest components
    largest_components = [labeled_array[find_objects(labeled_array == label)[0]] for label in largest_labels]

    return largest_components


def extract_largest_rois(lab_arr, box_min=[128, 128, 48]):

    # Label the connected components
    labeled_array, num_features = label(lab_arr)
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0
    largest_labels = sizes.argsort()[-2:][::-1]

    # Combine kidney and cyst labels
    offset_rois = []
    size_rois = []

    for labels_idx in largest_labels:

        temp_label = (labeled_array == labels_idx) * 1

        box = calculate_bounding_box(temp_label)

        bbox = []
        for i in range(3):
            i_size = box[i].stop - box[i].start
            if i_size < box_min[i]:
                diff = box_min[i] - i_size
                if diff % 2 == 0:
                    diff_left = diff / 2
                    diff_right = diff / 2
                else:
                    diff_left = (diff - 1) / 2
                    diff_right = (diff - 1) / 2 + 1

                i_bbox_min = np.max([box[i].start - diff_left, 0])
                i_bbox_max = np.min([box[i].stop + diff_right, temp_label.shape[i]])
                bbox.append(slice(int(i_bbox_min), int(i_bbox_max)))
            else:
                bbox.append(box[i])
        bbox = tuple(bbox)  # the final cutting off box

        offset = np.array([bbox[i].start for i in range(3)], dtype=np.int32)
        size = np.array([bbox[i].stop - bbox[i].start for i in range(3)], dtype=np.int32)

        offset_rois.append(offset)
        size_rois.append(size)

    return offset_rois, size_rois

def pad_array_to_size(arr, target_size=[128, 128, 48]):
    # Calculate the padding needed for each dimension
    padding = [(max(0, (target_size[i] - arr.shape[i]) // 2),
                max(0, (target_size[i] - arr.shape[i] + 1) // 2)) for i in range(len(target_size))]

    # Pad the array
    padded_arr = np.pad(arr, padding, mode='constant', constant_values=0)

    return padded_arr


def detect_regions(gt, pred):
    """
    Detects regions in a 3D ground truth array 'gt' and classifies them based on sensitivity with respect to a prediction array 'pred'.

    Parameters:
    gt (np.array): The ground truth array.
    pred (np.array): The prediction array.

    Returns:
    tuple: A tuple containing two lists - (detected, non_detected)
           detected: List of labels for regions detected with a sensitivity >= 0.95.
           non_detected: List of labels for regions detected with a sensitivity < 0.95.
    """

    # Label the regions in the ground truth array
    labeled_gt, num_features = label(gt)

    if num_features == 0:
        return [], []

    # Initialize lists for detected and non-detected regions
    detected = []
    non_detected = []

    # Function to calculate sensitivity
    def calculate_sensitivity(true_positive, total_true):
        return true_positive / total_true if total_true > 0 else 0

    # Iterate through each region
    for idx in range(1, num_features+1):

        # Create masks for the current region in both gt and pred
        region_mask_gt = (labeled_gt == idx) * 1
        region_mask_pred = pred

        # Calculate true positives and total true instances
        true_positive = np.sum(region_mask_gt & region_mask_pred)
        total_true = np.sum(region_mask_gt)

        # Calculate sensitivity
        sensitivity = calculate_sensitivity(true_positive, total_true)

        # Classify the region based on sensitivity
        if sensitivity > 0.1:
            detected.append(region_mask_gt)
        else:
            print(sensitivity)
            non_detected.append(region_mask_gt)

    return detected, non_detected


pred_pth = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\pred_labelsTs"
img_path = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\imagesTs"
gt_path = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\labelsTs"

base_path_img = r"C:\Users\P70074460\PycharmProjects\dynunet\images"

store_img_nii = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\interpretability\vae_gan\check_data\analysis\images"
store_gt_nii = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\interpretability\vae_gan\check_data\analysis\gt"
store_pred_orig_nii = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\interpretability\vae_gan\check_data\analysis\pred_orig"

files = os.listdir(pred_pth)

size_patch = []
for f in tqdm(files):
    img_f = f.replace(".nii.gz", "_0000.nii.gz")

    lab_img = nib.load(os.path.join(pred_pth, f)).get_fdata()
    lab_img_all = (lab_img > 0) * 1
    offset_rois, size_rois = extract_largest_rois(lab_img_all)

    pred_sitk = sitk.ReadImage(os.path.join(pred_pth, f))
    img_sitk = sitk.ReadImage(os.path.join(img_path, img_f))
    gt_sitk = sitk.ReadImage(os.path.join(gt_path, f))

    for idx in range(len(offset_rois)):
        patch_img = sitk.RegionOfInterest(img_sitk, size_rois[idx].tolist(), offset_rois[idx].tolist())
        patch_pred = sitk.RegionOfInterest(pred_sitk, size_rois[idx].tolist(), offset_rois[idx].tolist())
        patch_gt = sitk.RegionOfInterest(gt_sitk, size_rois[idx].tolist(), offset_rois[idx].tolist())

        patch_img_arr = sitk.GetArrayFromImage(patch_img).transpose()
        patch_pred_arr = sitk.GetArrayFromImage(patch_pred).transpose()
        patch_gt_arr = sitk.GetArrayFromImage(patch_gt).transpose()

        patch_img_arr = pad_array_to_size(patch_img_arr)
        patch_pred_arr = pad_array_to_size(patch_pred_arr)
        patch_gt_arr = pad_array_to_size(patch_gt_arr)

        print(np.unique(patch_pred_arr))
        size_patch.append(patch_pred_arr.shape)

        np.save(os.path.join(store_img_nii, f.replace(".nii.gz", f"_{idx}.npy")), patch_img_arr)
        np.save(os.path.join(store_gt_nii, f.replace(".nii.gz", f"_{idx}.npy")), patch_gt_arr)
        np.save(os.path.join(store_pred_orig_nii, f.replace(".nii.gz", f"_{idx}.npy")), patch_pred_arr)

        #save_slices_as_grid(patch_img_arr, os.path.join(base_path_img, f"img_{f}_{idx}.png"))
        #save_slices_as_grid(patch_pred_arr, os.path.join(base_path_img, f"pred_{f}_{idx}.png"))
        #save_slices_as_grid(patch_gt_arr, os.path.join(base_path_img, f"gt_{f}_{idx}.png"))


store_img_nii = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\interpretability\vae_gan\check_data\analysis\images"
store_gt_nii = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\interpretability\vae_gan\check_data\analysis\gt"
store_pred_orig_nii = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\interpretability\vae_gan\check_data\analysis\pred_orig"

store_gt_cor_pred = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\interpretability\vae_gan\check_data\analysis\gt_cor_pred"
store_gt_miss_pred = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\interpretability\vae_gan\check_data\analysis\gt_miss_pred"


for f in tqdm(os.listdir(store_gt_nii)):
    print(f)
    gt = np.load(os.path.join(store_gt_nii, f))
    pred = np.load(os.path.join(store_pred_orig_nii, f))

    pred = (pred == 2) * 1

    detected_regions, non_detected_regions = detect_regions(gt, pred)

    case_no = f.split(".npy")[0]
    for idx in range(len(detected_regions)):
        np.save(os.path.join(store_gt_cor_pred, f"{case_no}_{idx}.npy"), detected_regions[idx])
        #save_slices_as_grid(detected_regions[idx], os.path.join(base_path_img, f"{case_no}_{idx}.png"))

    for idx in range(len(non_detected_regions)):
        np.save(os.path.join(store_gt_miss_pred, f"{case_no}_{idx}.npy"), non_detected_regions[idx])
        #save_slices_as_grid(detected_regions[idx], os.path.join(base_path_img, f"{case_no}_{idx}.png"))





