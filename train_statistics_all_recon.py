import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label, binary_dilation
from scipy.spatial.distance import directed_hausdorff
import monai
import seg_metrics.seg_metrics as sg
import pandas as pd


labels = [0,1 ]
def calculate_dice_score(array1, array2):

    if array1.shape != array2.shape:
        raise ValueError("Input arrays must have the same shape.")

    intersection = np.sum(array1 * array2)
    volume_sum = np.sum(array1) + np.sum(array2)

    if volume_sum == 0:
        return 1.0  # both arrays are empty, hence identical

    return 2 * intersection / volume_sum


def find_highest_dice_per_lesion(ground_truth, prediction):
    # Label each connected component (lesion) with a unique label
    labeled_ground_truth, num_features_gt = label(ground_truth)
    labeled_prediction, num_features_pred = label(prediction)

    # Prepare a dictionary to store the highest Dice score for each ground truth lesion
    highest_dice_scores = {}

    # Iterate through each lesion in the ground truth
    for gt_label in range(1, num_features_gt + 1):
        gt_lesion_mask = (labeled_ground_truth == gt_label)

        # Temporarily store the highest Dice score found for this lesion
        highest_dice = 0

        # Compare against each lesion in the prediction
        for pred_label in range(1, num_features_pred + 1):
            pred_lesion_mask = (labeled_prediction == pred_label)

            # Calculate the Dice score between the ground truth lesion and this predicted lesion
            dice_score = calculate_dice_score(pred_lesion_mask, gt_lesion_mask)

            # Update the highest Dice score if this score is higher
            if dice_score > highest_dice:
                highest_dice = dice_score

        # Store the highest Dice score for this ground truth lesion
        highest_dice_scores[gt_label] = highest_dice

    return highest_dice_scores


def find_false_positives(ground_truth, prediction, threshold=0.1, min_volume_threshold=200):
    labeled_ground_truth, num_features_gt = label(ground_truth)
    labeled_prediction, num_features_pred = label(prediction)

    unmatched_count = 0

    for pred_label in range(1, num_features_pred + 1):
        pred_lesion_mask = (labeled_prediction == pred_label)

        match_found = False

        if np.sum(pred_lesion_mask) > min_volume_threshold:
            for gt_label in range(1, num_features_gt + 1):
                gt_lesion_mask = (labeled_ground_truth == gt_label)

                dice_score = calculate_dice_score(pred_lesion_mask, gt_lesion_mask)

                if dice_score > threshold:
                    match_found = True
                    break

            if not match_found:
                unmatched_count += 1

    return unmatched_count


def remove_small_lesions(prediction, min_voxels=10):
    labeled_prediction, num_features_pred = label(prediction)
    output_prediction = np.zeros_like(prediction)

    for pred_label in range(1, num_features_pred + 1):
        lesion_mask = (labeled_prediction == pred_label)
        if np.sum(lesion_mask) >= min_voxels:
            output_prediction[lesion_mask] = 1

    return output_prediction


def calculate_95th_hausdorff_distance_handling_empty(ground_truth, prediction, max_distance=np.nan):
    # Check if the prediction is entirely zero
    if not np.any(prediction):
        return max_distance

    # Proceed with the normal calculation if the prediction is not empty
    gt_coords = np.argwhere(ground_truth == 1)
    pred_coords = np.argwhere(prediction == 1)

    if gt_coords.size == 0 and pred_coords.size == 0:
        # Case where both ground truth and prediction are empty
        return 0.0

    hausdorff_dist_gt_to_pred = [directed_hausdorff(gt_coords, pred_coords)[0]]
    hausdorff_dist_pred_to_gt = [directed_hausdorff(pred_coords, gt_coords)[0]]

    all_distances = np.concatenate((hausdorff_dist_gt_to_pred, hausdorff_dist_pred_to_gt))
    hausdorff_95th = np.percentile(all_distances, 95)

    return hausdorff_95th



overall_dsc = []
overall_h95 = []
lesion_dsc = []
lesion_gt_vol = []
false_postives_per_image = []

# constants
min_vol_threshold = 200
voxel_volume = 0.77 * 0.77 * 2.5
false_postive_lesions = 0
cyst_small_size = 1806
cyst_medium_size = 11494


for fold in range(0,5):
    # Paths for computing the results
    orig_lab_path = r"/data/kidnAI/DynUnet/data/labelsTr_s"
    pred_lab_path = f"/data/kidnAI/DynUnet/data/results/pred_fold{fold}_recon"
    orig_lab_files = os.listdir(pred_lab_path)

    print("# Results on UCSF Dataset")
    for idx, file in enumerate(orig_lab_files):
        if "BOSNIAK" in file:
            lab_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(orig_lab_path, file))) == 2) * 1
            img_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_lab_path, file))) == 2) * 1
            img_tmp = remove_small_lesions(img_tmp)

            print(lab_tmp.shape)
            # overall dice metrics
            dsc_tmp = calculate_dice_score(img_tmp, lab_tmp)

            h95_tmp = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                      gdth_img=lab_tmp,
                      pred_img=img_tmp,
                      spacing=[2.5, 0.77, 0.77],
                      metrics=['dice', 'hd95'])
            print(h95_tmp)
            h95_tmp = h95_tmp[0]["hd95"]
            print(f"{file}: dsc_tmp: {dsc_tmp}, h95_tmp: {h95_tmp}")

            overall_dsc.append(dsc_tmp)
            overall_h95.append(h95_tmp)

            # Label each connected component (lesion) with a unique label
            labeled_ground_truth, num_features_gt = label(lab_tmp)
            labeled_prediction, num_features_pred = label(img_tmp)


            # Iterate through each lesion in the ground truth
            for gt_label in range(1, num_features_gt + 1):

                gt_lesion_mask = (labeled_ground_truth == gt_label)

                gt_vol_voxel = np.sum(gt_lesion_mask)
                highest_dice = 0

                if gt_vol_voxel > min_vol_threshold:

                    # Compare against each lesion in the prediction
                    for pred_label in range(1, num_features_pred + 1):
                        pred_lesion_mask = (labeled_prediction == pred_label)

                        # Calculate the Dice score between the ground truth lesion and this predicted lesion
                        dice_score = calculate_dice_score(pred_lesion_mask, gt_lesion_mask)

                        # Update the highest Dice score if this score is higher
                        if dice_score > highest_dice:
                            highest_dice = dice_score

                    lesion_dsc.append(highest_dice)
                    lesion_gt_vol.append(gt_vol_voxel)

            # find number of false positive
            false_pos_tmp = find_false_positives(lab_tmp, img_tmp)
            false_postive_lesions += false_pos_tmp
            false_postives_per_image.append(false_pos_tmp)
            print(false_pos_tmp)


for fold in range(0, 5):
    # Paths for computing the results
    orig_lab_path = r"/data/kidnAI/DynUnet/data/labelsTr_s"
    pred_lab_path = f"/data/kidnAI/DynUnet/data/results/pred_fold{fold}_recon"
    orig_lab_files = os.listdir(pred_lab_path)

    for idx, file in enumerate(orig_lab_files):
        if "KITS" in file:
            lab_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(orig_lab_path, file))) == 2) * 1
            img_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_lab_path, file))) == 2) * 1
            img_tmp = remove_small_lesions(img_tmp)

            print(lab_tmp.shape)
            # overall dice metrics
            dsc_tmp = calculate_dice_score(img_tmp, lab_tmp)

            h95_tmp = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                      gdth_img=lab_tmp,
                      pred_img=img_tmp,
                      spacing=[2.5, 0.77, 0.77],
                      metrics=['dice', 'hd95'])
            print(h95_tmp)
            h95_tmp = h95_tmp[0]["hd95"]
            print(f"{file}: dsc_tmp: {dsc_tmp}, h95_tmp: {h95_tmp}")

            overall_dsc.append(dsc_tmp)
            overall_h95.append(h95_tmp)

            # Label each connected component (lesion) with a unique label
            labeled_ground_truth, num_features_gt = label(lab_tmp)
            labeled_prediction, num_features_pred = label(img_tmp)


            # Iterate through each lesion in the ground truth
            for gt_label in range(1, num_features_gt + 1):

                gt_lesion_mask = (labeled_ground_truth == gt_label)

                gt_vol_voxel = np.sum(gt_lesion_mask)
                highest_dice = 0

                if gt_vol_voxel > min_vol_threshold:

                    # Compare against each lesion in the prediction
                    for pred_label in range(1, num_features_pred + 1):
                        pred_lesion_mask = (labeled_prediction == pred_label)

                        # Calculate the Dice score between the ground truth lesion and this predicted lesion
                        dice_score = calculate_dice_score(pred_lesion_mask, gt_lesion_mask)

                        # Update the highest Dice score if this score is higher
                        if dice_score > highest_dice:
                            highest_dice = dice_score

                    lesion_dsc.append(highest_dice)
                    lesion_gt_vol.append(gt_vol_voxel)

            # find number of false positive
            false_pos_tmp = find_false_positives(lab_tmp, img_tmp)
            false_postive_lesions += false_pos_tmp
            false_postives_per_image.append(false_pos_tmp)
            print(find_false_positives(lab_tmp, img_tmp))

total_cysts = len(lesion_dsc)
detected_cysts = np.sum(np.array(lesion_dsc)>0.1)
lesion_gt_vol_real = np.mean(lesion_gt_vol) * voxel_volume
vol_lesions = np.array(lesion_gt_vol) * voxel_volume

h95_arr = np.array(overall_h95)
h95_arr = h95_arr[~np.isnan(h95_arr)]
h95_arr = h95_arr[h95_arr<1000]

small_lesion = np.array(vol_lesions<cyst_small_size)
medium_lesion = np.array(vol_lesions>cyst_small_size) & (vol_lesions<cyst_medium_size)
large_lesion = np.array(vol_lesions>cyst_medium_size)

lesion_dsc = np.array(lesion_dsc)
lesion_dsc_small = np.sum(lesion_dsc[small_lesion])/np.sum(small_lesion)
lesion_dsc_medium = np.sum(lesion_dsc[medium_lesion])/np.sum(medium_lesion)
lesion_dsc_large = np.sum(lesion_dsc[large_lesion])/np.sum(large_lesion)

lesion_detect_small = np.sum(lesion_dsc[small_lesion]>0.1)/np.sum(small_lesion)
lesion_detect_medium = np.sum(lesion_dsc[medium_lesion]>0.1)/np.sum(medium_lesion)
lesion_detect_large = np.sum(lesion_dsc[large_lesion]>0.1)/np.sum(large_lesion)


print("Train Mean result: ", np.mean(overall_dsc))
print("Train Mean H95: ", np.mean(h95_arr))
print("Train Detected: ", detected_cysts)
print("Train Not Detected: ", total_cysts-detected_cysts)
print("Train Mean Volume: ", lesion_gt_vol_real)
print("Train False Positive Lesions: ", false_postive_lesions)
print("Train Small Cyst DSC: ", lesion_dsc_small)
print("Train Medium Cyst DSC: ", lesion_dsc_medium)
print("Train Large Cyst DSC: ", lesion_dsc_large)
print("Train Small Cyst detect: ", lesion_detect_small)
print("Train Medium Cyst detect: ", lesion_detect_medium)
print("Train Large Cyst detect: ", lesion_detect_large)

lesion_dsc_small = lesion_dsc[small_lesion]
lesion_dsc_medium = lesion_dsc[medium_lesion]
lesion_dsc_large = lesion_dsc[large_lesion]
cyst_dsc = overall_dsc

cyst_sensitivity = np.array(lesion_dsc)>0.1
cyst_small_sensitivity = lesion_dsc[small_lesion]>0.1
cyst_medium_sensitivity = lesion_dsc[medium_lesion]>0.1
cyst_large_sensitivity = lesion_dsc[large_lesion]>0.1


# Find the maximum length
max_len = max(len(overall_dsc), len(lesion_dsc_small), len(lesion_dsc_medium), len(lesion_dsc_large),
              len(cyst_sensitivity), len(cyst_small_sensitivity), len(cyst_medium_sensitivity),
              len(cyst_large_sensitivity), len(false_postives_per_image))

arrays = [overall_dsc, lesion_dsc_small, lesion_dsc_medium, lesion_dsc_large,
          cyst_sensitivity, cyst_small_sensitivity, cyst_medium_sensitivity,
          cyst_large_sensitivity, false_postives_per_image]
extended_arrays = [
    np.pad(
        np.array(arr).astype(float),  # Convert array to float type
        (0, max_len - len(arr)),
        'constant',
        constant_values=np.nan
    )
    for arr in arrays
]

# Creating the DataFrame
df = pd.DataFrame({
    'DSC': extended_arrays[0],
    'Small Cysts DSC': extended_arrays[1],
    'Medium Cysts DSC': extended_arrays[2],
    'Large Cysts DSC': extended_arrays[3],
    'Sensitivity': extended_arrays[4],
    'Small Cysts Sensitivity': extended_arrays[5],
    'Medium Cysts Sensitivity': extended_arrays[6],
    'Large Cysts Sensitivity': extended_arrays[7],
    'False Positive Per Image': extended_arrays[8]
})

# Saving the DataFrame to an Excel file
excel_path = "/home/zohaib/Desktop/train_stats_recon.xlsx"
df.to_excel(excel_path, index=False)

