import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
import glob
import os

# Paths to your image and mask files
base_pth = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\interpretability\vae_gan\check_data\analysis\dice_analysis_detected_cyst_kid_gt"
gt_pth = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\interpretability\vae_gan\check_data\analysis\gt_cor_pred"
image_paths = glob.glob(os.path.join(base_pth, "*", "recon*.npy" ))
mask_paths = [os.path.join(gt_pth, x.split("\\")[-2] + ".npy") for x in image_paths]

# Initialize the feature extractor
params = {}  # Use default parameters, or define your own
extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(**params)

# Function to extract features
def extract_features(image_path, mask_path):
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    features = extractor.execute(image, mask)
    return {
        'First Order Mean': features['firstorder_Mean'],
        'First Order Standard Deviation': features['firstorder_StandardDeviation'],
        'GLCM Entropy': features['glcm_Entropy'],
        'GLCM Homogeneity': features['glcm_Homogeneity'],
        'GLRLM Short Run Emphasis': features['glrlm_ShortRunEmphasis'],
        'Histogram Kurtosis': features['firstorder_Kurtosis']
    }

# Extract features for each image-mask pair
feature_data = [extract_features(img_path, mask_path) for img_path, mask_path in zip(image_paths, mask_paths)]

# Create a DataFrame and save to CSV
df = pd.DataFrame(feature_data)
df.to_csv('radiomics_features.csv', index=False)
