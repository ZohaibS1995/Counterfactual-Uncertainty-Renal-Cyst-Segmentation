import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import entropy
from datetime import datetime
from sklearn.metrics import mutual_info_score
from datetime import datetime
import pandas as pd
from skimage.measure import label, regionprops


def DSC(prediction, target):
    smooth = 1
    numerator = (prediction * target).sum()
    denominator = prediction.sum() + target.sum()

    dice_v = (2 * numerator + smooth) / (denominator + smooth)
    return dice_v


def expected_calibration_error(y_true, y_pred_prob, n_bins=4):
    bin_limits = np.linspace(0, 1, n_bins+1)
    bin_midpoints = (bin_limits[:-1] + bin_limits[1:]) / 2

    bin_lowers = bin_limits[:-1]
    bin_uppers = bin_limits[1:]

    accuracies = np.zeros(n_bins)
    avg_confidences = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    y_pred = np.argmax(y_pred_prob, axis=0) # Convert probabilities to labels
    confidences = np.max(y_pred_prob, axis=0) # Confidence of prediction

    for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = np.logical_and(bin_lower <= confidences, confidences < bin_upper) # Indices of samples in bin

        if np.sum(in_bin) > 0:
            accuracies[bin_idx] = np.mean(y_true[in_bin] == y_pred[in_bin])
            avg_confidences[bin_idx] = np.mean(confidences[in_bin])
            counts[bin_idx] = np.sum(in_bin)

    ece = np.sum(np.abs(accuracies - avg_confidences) * counts) / np.sum(counts)
    return ece


f_test_path = f'/data/kidnAI/DynUnet/data/uncertainty_fold_all/predTr_img'
f_test_label_path = r'/data/kidnAI/DynUnet/data/labelsTr_s'  # labelsTr_s labelsTs
save_path = f'/home/zohaib/Desktop/kidnAI_proj/SQC_figures_fold_all/train'
os.makedirs(save_path, exist_ok=True)
# sample_serial = 2

f_test_list = os.listdir(f_test_path)
f_test_list.sort()
current_round_folds = []
for sub_fold in f_test_list:
    if bool(re.search('^fold\d+_uncertain_\d_\d+$', sub_fold)):
        current_round_folds.append(sub_fold)

dice_per_patient_list = []
variance_per_patient_list = []

dice_per_lesion_list = []
variance_per_lesion_list = []


subject_list = os.listdir(os.path.join(f_test_path, current_round_folds[0]))
for sub in subject_list:
    segmentation_list = []
    label_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(f_test_label_path, sub)))
    if np.unique(label_img).shape[0] > 2:
        label_img = np.array(label_img == 2, dtype=int)
    labeled_cyst_total = label(label_img)

    # per patient
    for t_cf in current_round_folds:
        file_path = os.path.join(f_test_path, t_cf, sub)
        # print(file_path, os.path.exists(file_path))
        pred = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        segmentation_list.append(pred)

    segmentation_all_soft = np.stack(segmentation_list)
    print(segmentation_all_soft.shape, label_img.shape)
    # calculating mean prediction
    mean_pred_prob = np.mean(segmentation_all_soft, axis=0)
    mean_pred = np.array(mean_pred_prob > 0.5, dtype=int)

    # calculate mean metric
    dice_score = DSC(mean_pred, label_img)
    num_ori_pixels = np.sum(mean_pred)
    variance_values = np.sum(np.var(segmentation_all_soft, axis=0)) / (num_ori_pixels)
    dice_per_patient_list.append(dice_score)
    variance_per_patient_list.append(variance_values)
    print(datetime.now(), sub, 'DSC: ', dice_score, ', Var: ', variance_values)

    for i_ind, region in enumerate(regionprops(labeled_cyst_total)):
        # print(region.bbox)
        min_row, min_col, min_depth, max_row, max_col, max_depth = region.bbox
        cyst_cube = labeled_cyst_total[min_row:max_row, min_col:max_col, min_depth:max_depth] / (i_ind + 1)
        pred_cube = segmentation_all_soft[:, min_row:max_row, min_col:max_col, min_depth:max_depth]
        mean_pred_cube_prob = np.mean(pred_cube, axis=0)
        mean_pred_cube = np.array(mean_pred_cube_prob > 0.5, dtype=int)

        dice_score_cube = DSC(mean_pred_cube, cyst_cube)
        num_ori_pixels_cube = np.sum(mean_pred_cube)
        num_ori_pixels_cube = 1 if num_ori_pixels_cube == 0 else num_ori_pixels_cube
        variance_values_cube = np.sum(np.var(pred_cube, axis=0)) / num_ori_pixels_cube
        dice_per_lesion_list.append(dice_score_cube)
        variance_per_lesion_list.append(variance_values_cube)
        print(datetime.now(), sub, 'cyst: ', i_ind, 'DSC: ', dice_score_cube, ', Var: ', variance_values_cube)

df_patient = pd.DataFrame({'dice': dice_per_patient_list, 'UE': variance_per_patient_list})
df_lesion = pd.DataFrame({'dice': dice_per_lesion_list, 'UE': variance_per_lesion_list})


poor_thre = np.linspace(0.7, 0.8, 3)
for i in poor_thre:
    poor_quality_dice_thre = i

    poor_segmentation_fraction_l_patient = []
    flagged_for_manual_correction_l_patient = []
    poor_segmentation_fraction_l_lesion = []
    flagged_for_manual_correction_l_lesion = []

    df_patient['UE'].replace([np.inf, -np.inf], 0, inplace=True)
    thre = np.linspace(0, 1, 100000) * df_patient['UE'].max()#np.percentile(df_patient['UE'], 97)#df_patient['UE'].max()
    for t in thre:
        t_manual_correction_patient = df_patient[df_patient['UE'] > t]
        t_remaining_patient = df_patient[df_patient['UE'] <= t]
        flagged_for_manual_correction_l_patient.append(t_manual_correction_patient.shape[0])
        poor_segmentation_fraction_l_patient.append(t_remaining_patient[t_remaining_patient['dice'] < poor_quality_dice_thre].shape[0])

        t_manual_correction_lesion = df_lesion[df_lesion['UE'] > t]
        t_remaining_lesion = df_lesion[df_lesion['UE'] <= t]
        flagged_for_manual_correction_l_lesion.append(t_manual_correction_lesion.shape[0])
        poor_segmentation_fraction_l_lesion.append(t_remaining_lesion[t_remaining_lesion['dice'] < poor_quality_dice_thre].shape[0])

    poor_segmentation_fraction_l_patient = np.array(poor_segmentation_fraction_l_patient) / df_patient.shape[0]
    flagged_for_manual_correction_l_patient = np.array(flagged_for_manual_correction_l_patient) / df_patient.shape[0]

    poor_segmentation_fraction_l_lesion = np.array(poor_segmentation_fraction_l_lesion) / df_lesion.shape[0]
    flagged_for_manual_correction_l_lesion = np.array(flagged_for_manual_correction_l_lesion) / df_lesion.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(flagged_for_manual_correction_l_patient, poor_segmentation_fraction_l_patient)
    ax.plot([np.max(poor_segmentation_fraction_l_patient), 0], [0, np.max(poor_segmentation_fraction_l_patient)])
    ax.plot([1, 0], [0, np.max(poor_segmentation_fraction_l_patient)])
    ax.set_title('Patient-poor_threshold={}'.format(i))
    #plt.show()
    save_file_patient = os.path.join(save_path, '{}_patient.png'.format(i))
    plt.savefig(save_file_patient)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(flagged_for_manual_correction_l_lesion, poor_segmentation_fraction_l_lesion)
    ax.plot([np.max(poor_segmentation_fraction_l_lesion), 0], [0, np.max(poor_segmentation_fraction_l_lesion)])
    ax.plot([1, 0], [0, np.max(poor_segmentation_fraction_l_lesion)])
    ax.set_title('Lesion-poor_threshold={}'.format(i))
    #plt.show()
    save_file_patient = os.path.join(save_path, 'lesion_{}_patient.png'.format(i))
    plt.savefig(save_file_patient)
    plt.close()




