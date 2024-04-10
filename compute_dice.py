import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label, binary_dilation

def dice_score(array1, array2):
    """
    Compute the Dice score between two 3D numpy arrays.

    Parameters:
    - array1: 3D numpy array
    - array2: 3D numpy array

    Returns:
    - Dice score (float)
    """

    if array1.shape != array2.shape:
        raise ValueError("Input arrays must have the same shape.")

    intersection = np.sum(array1 * array2)
    volume_sum = np.sum(array1) + np.sum(array2)

    if volume_sum == 0:
        return 1.0  # both arrays are empty, hence identical

    return 2 * intersection / volume_sum

def compute_sensitivity(true_mask, pred_mask):
    TP = np.sum(true_mask & pred_mask)  # True positives
    FN = np.sum(true_mask & ~pred_mask)  # False negatives
    return TP / (TP + FN)


# Assuming gt_mask is the ground truth 3D label and pred_mask is the 3D prediction mask
#
def detect_lesions(gt_mask, pred_mask, thresh_detect=10):
    # Label the connected components in the ground truth mask
    labeled_mask, num_features = label(gt_mask)

    detected_count = 0

    for i in range(1, num_features + 1):
        single_lesion = labeled_mask == i

        # Only consider lesions with pixel count greater than 10
        if np.sum(single_lesion) > thresh_detect:
            sensitivity = compute_sensitivity(single_lesion, pred_mask)

            if sensitivity > 0.05:
                detected_count += 1

    # Calculate the total number of lesions with pixel count > 10
    total_lesions = np.sum([(labeled_mask == i).sum() > thresh_detect for i in range(1, num_features + 1)])
    non_detected_count = total_lesions - detected_count
    return detected_count, non_detected_count

## labels path
orig_lab_path = r"/data/kidnAI/DynUnet/data/labelsTs"
pred_lab_path = r"/data/kidnAI/DynUnet/data/pred_labelsTs"
#pred_lab_path_recon = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\task001_KidnAI\pred_labelsTs_recon"
# getting the label paths
orig_lab_files = os.listdir(orig_lab_path)

print("############# Printing results on MUMC files ################")


mumc_dsc_lst = []
mumc_detected_lesions = 0
mumc_non_detected_lesions = 0

mumc_dsc_lst_2 = []
mumc_detected_lesions_2 = 0
mumc_non_detected_lesions_2 = 0

for idx, file in enumerate(orig_lab_files):
    if "BOSNIAK" in file:
        lab_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(orig_lab_path, file))) == 1) * 1
        img_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_lab_path, file))) == 2) * 1
        #img_tmp_2 = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_lab_path_recon, file))) == 2) * 1

        # type 1
        dsc_tmp = dice_score(img_tmp, lab_tmp)
        detected_l_n, n_detected_l_n = detect_lesions(lab_tmp, img_tmp)
        mumc_detected_lesions += detected_l_n
        mumc_non_detected_lesions += n_detected_l_n

        mumc_dsc_lst.append(dsc_tmp)

        # type 2
        #dsc_tmp_2 = dice_score(img_tmp_2, lab_tmp)
        #detected_l_n_2, n_detected_l_n_2 = detect_lesions(lab_tmp, img_tmp_2)
        #mumc_detected_lesions_2 += detected_l_n_2
        #mumc_non_detected_lesions_2 += n_detected_l_n_2

        #mumc_dsc_lst_2.append(dsc_tmp_2)
        print(file, " : ", dsc_tmp, "-", detected_l_n, "/", detected_l_n+n_detected_l_n)
        #print(file, "_recon : ", dsc_tmp_2, "-", detected_l_n_2, "/", detected_l_n_2+n_detected_l_n_2)
        #if detected_l_n != detected_l_n_2:
        #    print(f"################################################### \n")


print("MUMC Mean result: ", np.mean(mumc_dsc_lst))
print("MUMC Detected: ", mumc_detected_lesions)
print("MUMC Not Detected: ", mumc_non_detected_lesions)
print("MUMC Percentage: ", (mumc_detected_lesions/(mumc_non_detected_lesions + mumc_detected_lesions))*100)

#print("MUMC_recon Mean result: ", np.mean(mumc_dsc_lst_2))
#print("MUMC_recon Detected: ", mumc_detected_lesions_2)
#print("MUMC_recon Not Detected: ", mumc_non_detected_lesions_2)
#print("MUMC_recon Percentage: ", (mumc_detected_lesions_2/(mumc_non_detected_lesions_2 + mumc_detected_lesions_2))*100)
#
#
print("############# Printing results on AMC files ################")

mumc_dsc_lst = []
mumc_detected_lesions = 0
mumc_non_detected_lesions = 0

#mumc_dsc_lst_2 = []
#mumc_detected_lesions_2 = 0
#mumc_non_detected_lesions_2 = 0

for idx, file in enumerate(orig_lab_files):
    if "RADIOMICS" in file:
        lab_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(orig_lab_path, file))) == 1) * 1
        img_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_lab_path, file))) == 2) * 1
 #       img_tmp_2 = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_lab_path_recon, file))) == 2) * 1

        # type 1
        dsc_tmp = dice_score(img_tmp, lab_tmp)
        detected_l_n, n_detected_l_n = detect_lesions(lab_tmp, img_tmp)
        mumc_detected_lesions += detected_l_n
        mumc_non_detected_lesions += n_detected_l_n

        mumc_dsc_lst.append(dsc_tmp)

        # type 2
        #dsc_tmp_2 = dice_score(img_tmp_2, lab_tmp)
        #detected_l_n_2, n_detected_l_n_2 = detect_lesions(lab_tmp, img_tmp_2)
        #mumc_detected_lesions_2 += detected_l_n_2
        #mumc_non_detected_lesions_2 += n_detected_l_n_2

        #mumc_dsc_lst_2.append(dsc_tmp_2)
        print(file, " : ", dsc_tmp, "-", detected_l_n, "/", detected_l_n + n_detected_l_n)
        #print(file, "_recon : ", dsc_tmp_2, "-", detected_l_n_2, "/", detected_l_n_2 + n_detected_l_n_2)
        #if detected_l_n != detected_l_n_2:
        #    print("################################################### \n")

print("AMC Mean result: ", np.mean(mumc_dsc_lst))
print("AMC Detected: ", mumc_detected_lesions)
print("AMC Not Detected: ", mumc_non_detected_lesions)
print("AMC Percentage: ", (mumc_detected_lesions / (mumc_non_detected_lesions + mumc_detected_lesions)) * 100)

#print("AMC_recon Mean result: ", np.mean(mumc_dsc_lst_2))
#print("AMC_recon Detected: ", mumc_detected_lesions_2)
#print("AMC_recon Not Detected: ", mumc_non_detected_lesions_2)
#print("AMC_recon Percentage: ",
#      (mumc_detected_lesions_2 / (mumc_non_detected_lesions_2 + mumc_detected_lesions_2)) * 100)



# labels path
orig_lab_path = r"/data/kidnAI/DynUnet/data/labelsTr_s"
pred_lab_path = r"/data/kidnAI/DynUnet/data/pred_fold0"

# getting the label paths
orig_lab_files = os.listdir(pred_lab_path)

print("############# Printing results on MUMC files ################")


mumc_dsc_lst = []
mumc_detected_lesions = 0
mumc_non_detected_lesions = 0
count_kits = 0
for idx, file in enumerate(orig_lab_files):
    if "KITS" in file:
        lab_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(orig_lab_path, file))) == 2) * 1
        img_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_lab_path, file))) == 2) * 1

        dsc_tmp = dice_score(img_tmp, lab_tmp)
        detected_l_n, n_detected_l_n = detect_lesions(lab_tmp, img_tmp)
        mumc_detected_lesions += detected_l_n
        mumc_non_detected_lesions += n_detected_l_n

        mumc_dsc_lst.append(dsc_tmp)
        print(file, " : ", dsc_tmp, "-", detected_l_n, "/", detected_l_n+n_detected_l_n)
        count_kits += 1

print("Total KITS: ", count_kits)
print("Fold Mean result: ", np.mean(mumc_dsc_lst))
print("Fold Detected: ", mumc_detected_lesions)
print("Fold Not Detected: ", mumc_non_detected_lesions)
print("Fold Percentage: ", (mumc_detected_lesions/(mumc_non_detected_lesions + mumc_detected_lesions))*100)

# getting the label paths
orig_lab_files = os.listdir(pred_lab_path)

print("############# Printing results on MUMC files ################")


mumc_dsc_lst = []
mumc_detected_lesions = 0
mumc_non_detected_lesions = 0
count_bosniak = 0
for idx, file in enumerate(orig_lab_files):
    if "BOSNIAK" in file:
        lab_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(orig_lab_path, file))) == 2) * 1
        img_tmp = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_lab_path, file))) == 2) * 1

        dsc_tmp = dice_score(img_tmp, lab_tmp)
        detected_l_n, n_detected_l_n = detect_lesions(lab_tmp, img_tmp)
        mumc_detected_lesions += detected_l_n
        mumc_non_detected_lesions += n_detected_l_n

        mumc_dsc_lst.append(dsc_tmp)
        print(file, " : ", dsc_tmp, "-", detected_l_n, "/", detected_l_n+n_detected_l_n)
        count_bosniak += 1

print("Total BOSNIAK: ", count_bosniak)
print("Fold Mean result: ", np.mean(mumc_dsc_lst))
print("Fold Detected: ", mumc_detected_lesions)
print("Fold Not Detected: ", mumc_non_detected_lesions)
print("Fold Percentage: ", (mumc_detected_lesions/(mumc_non_detected_lesions + mumc_detected_lesions))*100)