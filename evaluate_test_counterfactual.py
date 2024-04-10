# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import configparser

import ignite.distributed as idist
import torch
import torch.distributed as dist
import torch.nn.functional as F

from monai.config import print_config
from monai.handlers import CheckpointSaver, LrScheduleHandler, MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.inferers import SimpleInferer, SlidingWindowInferer, sliding_window_inference
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel
from monai.losses.dice import *

from create_dataset import get_data
from create_network import get_network
from evaluator import DynUNetEvaluator
from task_params import data_loader_params, patch_size
from trainer import DynUNetTrainer
from monai.utils import first
from tqdm import tqdm
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist, load_decathlon_properties, partition_dataset
from task_params import task_name
import SimpleITK as sitk
from utils import EarlyStopping

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which does not require a GUI
import matplotlib.pyplot as plt
import math

import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label, binary_dilation

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from matplotlib.colors import ListedColormap

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

def dice_score_v(array1, array2):
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

def load_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config

def convert_options(config_section):
    options = type('Options', (), {})()
    for key, value in config_section.items():
        if value.lower() == 'true':
            setattr(options, key, True)
        elif value.lower() == 'false':
            setattr(options, key, False)
        elif value.isdigit():
            setattr(options, key, int(value))
        elif value.replace('.', '', 1).isdigit():
            setattr(options, key, float(value))
        else:
            setattr(options, key, value)
    return options

def print_config(config):
    for section in config.sections():
        print(f"[{section}]")
        for key in config[section]:
            print(f"{key} = {config[section][key]}")
        print()



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


def dice_score(preds, targets, epsilon=1e-6):
    preds = preds.float()
    targets = targets.float()

    intersection = torch.sum(preds * targets, dim=(0, 2, 3, 4))
    union = torch.sum(preds, dim=(0, 2, 3, 4)) + torch.sum(targets, dim=(0, 2, 3, 4))

    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice


def reparameterize(mu, sigma):
    epsilon = torch.randn_like(sigma)
    return mu + sigma * epsilon


if __name__ == "__main__":

    # model save directory
    model_save_dir = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\interpretability\vae_gan\check_data\models"
    check_data_test_snippets = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\Task001_KidnAI\interpretability\vae_gan\check_data\test"

    # reading configurations
    config_filename = 'config.ini'
    config = load_config(config_filename)
    options = convert_options(config['Settings'])
    options.task_id = f"{options.task_id:02}"
    args = options

    # load hyper-parameters
    task_id = args.task_id
    fold = args.fold
    val_output_dir = os.path.join(args.datalist_path, "/runs_{}_fold{}_{}/".format(task_id, fold, args.expr_name))
    log_filename = "nnunet_task{}_fold{}.log".format(task_id, fold)
    log_filename = os.path.join(val_output_dir, log_filename)
    interval = args.interval
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    multi_gpu_flag = args.multi_gpu
    amp_flag = args.amp
    lr_decay_flag = args.lr_decay
    sw_batch_size = args.sw_batch_size
    tta_val = args.tta_val
    batch_dice = args.batch_dice
    window_mode = args.window_mode
    eval_overlap = args.eval_overlap
    local_rank = args.local_rank
    determinism_flag = args.determinism_flag
    determinism_seed = args.determinism_seed

    # loading batch size
    train_batch_size = data_loader_params[task_id]["batch_size"]

    # setting the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")

    # loading validation dataloader
    properties, val_loader = get_data(args, mode="validation")

    # initializing early stopping
    net2 = get_network(properties, task_id, args.datalist_path,
                       "fold0_1000epoch_patch_size.pth")  # os.path.join(args.datalist_path, 'final_model.pth'))
    net2 = net2.to(device)
    net2.eval()

    # defining the model
    # setting up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Define Autoencoder KL network
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 64),
        latent_channels=3,
        num_res_blocks=1,
        norm_num_groups=16,
        attention_levels=(False, False, True),
    )
    autoencoder.to(device)

    # loading the model
    model = torch.load(os.path.join(model_save_dir, "ct_checkpoint_train_2.pt"))
    autoencoder.load_state_dict(model)
    autoencoder.eval()

    val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")

    count_t = 0
    count_c = 0
    for batch_idx, val_batch in val_progress_bar:
        print(f"IDX: {batch_idx} \n")
        val_inputs, val_targets = val_batch["image"], val_batch["label"]
        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

        for idx in range(4):
            count_t += 1
            input_val = val_inputs[idx, ...][None, ...]
            label_val = val_targets[idx, ...][None, ...]

            test_pred = net2(input_val)
            sf_test_pred = np.argmax(np.squeeze(test_pred.detach().cpu().numpy()), axis=0)

            # Generator part
            reconstruction, z_mu, z_sigma = autoencoder(input_val)
            z = reparameterize(z_mu, z_sigma)
            #z = z.clone().detach().requires_grad_(True)

            reconstruction = autoencoder.decode(z)

            recon_test_pred = net2(reconstruction)
            recon_sf_test_pred = np.argmax(np.squeeze(recon_test_pred.detach().cpu().numpy()), axis=0)

            #save_slices_as_grid(input_val.detach().cpu().numpy()[0, 0, ...],
            #                    os.path.join(check_data_test_snippets, f"train_image_sample_{batch_idx}.png"))
            #save_slices_as_grid(reconstruction.detach().cpu().numpy()[0, 0, ...],
            #                    os.path.join(check_data_test_snippets, f"train_reconstructed_sample_{batch_idx}.png"))

            #save_slices_as_grid(sf_test_pred,
            #                    os.path.join(check_data_test_snippets, f"pred_orig_{batch_idx}.png"))
            #save_slices_as_grid(recon_sf_test_pred,
            #                    os.path.join(check_data_test_snippets, f"pred_recons_{batch_idx}.png"))

            pred_orig = (sf_test_pred == 2) * 1
            pred_recon = (recon_sf_test_pred == 2) * 1
            orig_lab = (label_val.detach().cpu().numpy()[0, 0, ...] == 2) * 1

            print(f" Dice (pred_orig and pred_recon): {dice_score_v(pred_orig, orig_lab)}")
            print(f" Dice (orig_lab and pred_recon): {dice_score_v(pred_recon, orig_lab)}")
            print(f" Dice (pred_orig and pred_recon): {dice_score_v(pred_orig, pred_recon)}")

            if dice_score_v(pred_orig, pred_recon) < 0.9:
                plot_slices_with_contours(input_val.detach().cpu().numpy()[0, 0, ...],
                                          label_val.detach().cpu().numpy()[0, 0, ...],
                                          os.path.join(check_data_test_snippets, f'gt_{batch_idx}_{idx}.png'))

                plot_slices_with_contours(input_val.detach().cpu().numpy()[0, 0, ...],
                                          sf_test_pred,
                                          os.path.join(check_data_test_snippets, f'pred_{batch_idx}_{idx}.png'))

                plot_slices_with_contours(reconstruction.detach().cpu().numpy()[0, 0, ...],
                                          recon_sf_test_pred,
                                          os.path.join(check_data_test_snippets, f'recon_pred_{batch_idx}_{idx}.png'))
                count_c += 1

    val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")

    for batch_idx, val_batch in val_progress_bar:
        print(f"IDX: {batch_idx} \n")
        val_inputs, val_targets = val_batch["image"], val_batch["label"]
        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

        input_val = val_inputs[0, ...][None, ...]
        label_val = val_targets[0, ...][None, ...]

        test_pred = net2(input_val)
        sf_test_pred = np.argmax(np.squeeze(test_pred.detach().cpu().numpy()), axis=0)

        # Generator part
        reconstruction, z_mu, z_sigma = autoencoder(input_val)
        z = reparameterize(z_mu, z_sigma)
        #z = z.clone().detach().requires_grad_(True)

        reconstruction = autoencoder.decode(z)

        recon_test_pred = net2(reconstruction)
        recon_sf_test_pred = np.argmax(np.squeeze(recon_test_pred.detach().cpu().numpy()), axis=0)

        #save_slices_as_grid(input_val.detach().cpu().numpy()[0, 0, ...],
        #                    os.path.join(check_data_test_snippets, f"train_image_sample_{batch_idx}.png"))
        #save_slices_as_grid(reconstruction.detach().cpu().numpy()[0, 0, ...],
        #                    os.path.join(check_data_test_snippets, f"train_reconstructed_sample_{batch_idx}.png"))

        #save_slices_as_grid(sf_test_pred,
        #                    os.path.join(check_data_test_snippets, f"pred_orig_{batch_idx}.png"))
        #save_slices_as_grid(recon_sf_test_pred,
        #                    os.path.join(check_data_test_snippets, f"pred_recons_{batch_idx}.png"))

        pred_orig = (sf_test_pred == 2) * 1
        pred_recon = (recon_sf_test_pred == 2) * 1
        orig_lab = (label_val.detach().cpu().numpy()[0, 0, ...] == 2) * 1

        print(f" Dice (pred_orig and pred_recon): {dice_score_v(pred_orig, orig_lab)}")
        print(f" Dice (orig_lab and pred_recon): {dice_score_v(pred_recon, orig_lab)}")
        print(f" Dice (pred_orig and pred_recon): {dice_score_v(pred_orig, pred_recon)}")

        if dice_score_v(pred_orig, pred_recon) < 0.05:
            break

        recon_pred_tensor = (torch.argmax(recon_test_pred.squeeze(), dim=0) == 1) * 1
        cyst_pred = recon_test_pred[:, 2, :, :] * recon_pred_tensor
        kid_pred = recon_test_pred[:, 1, :, :] * recon_pred_tensor

        cyst_pred_sum = torch.sum(cyst_pred)
        kid_pred_sum = torch.sum(kid_pred)

        latent_gradients = torch.autograd.grad((cyst_pred_sum), z, retain_graph=True)[0]
        latent_gradients_kid = torch.autograd.grad((kid_pred_sum), z)[0]

        alphas = np.linspace(-0.001, 0, 20)#[-0.005, -0.004, -0.003, -0.002, -0.0001, -0.00005,  -0.00004, -0.00003, 0]  # Example range of alpha values

        for idx, alpha in enumerate(alphas):
            # Modify the latent space
            alpha = torch.tensor(alpha)
            modified_z = z - (alpha * latent_gradients) #+ (2*alpha * latent_gradients_kid)

            # Reconstruct image from modified latent space
            modified_reconstruction = autoencoder.decode(modified_z)

            # Generate prediction from modified reconstruction
            modified_pred = net2(modified_reconstruction)

            # Optional: Convert to class predictions, if needed
            modified_sf_pred = np.argmax(modified_pred.detach().cpu().numpy(), axis=1)

            #save_slices_as_grid(modified_reconstruction.detach().cpu().numpy()[0, 0, ...],
            #                     os.path.join(check_data_test_snippets, f'reconstructed_image_alpha_{idx}.png'))
            #save_slices_as_grid(modified_sf_pred[0],
            #                     os.path.join(check_data_test_snippets, f'prediction_alpha_{idx}.png'))

            plot_slices_with_contours(modified_reconstruction.detach().cpu().numpy()[0, 0, ...],
                                modified_sf_pred[0],
                                os.path.join(check_data_test_snippets, f'overlayed_alpha_{idx}.png'))
        dsadsa
        break
        if dice_score_v(pred_orig, pred_recon) > 0.9:
            break
    ## getting list containing test dataset name
    #list_key =  "{}_fold{}".format("validation", 0) # "test"
    #args.label_save_dir = r"D:\Zohaib\KIPA23\kidnAI_segmentation\nnUNet_raw_data_base\nnUNet_raw_data\task001_KidnAI\pred_fold0"
    #datalist_name = "dataset_task{}.json".format(args.task_id)
    #dataset_path = os.path.join(args.root_dir, task_name[task_id])
    #datalist = load_decathlon_datalist(os.path.join(args.datalist_path, datalist_name), True, list_key, dataset_path)
    #
    #for item in datalist:
    #    item['image'] = item['image'].replace(".nii.gz", "_0000.nii.gz")
    #
    #
    #for t_img in tqdm(datalist):
    #    test_img = t_img["image"]
    #    test_img_sitk = sitk.ReadImage(test_img)
    #    test_img_tensor = torch.tensor(
    #        np.transpose(sitk.GetArrayFromImage(test_img_sitk), (1, 2, 0))[None, None, ...]).to(device)
    #    test_img_tensor = test_img_tensor.type("torch.cuda.FloatTensor")
    #    test_pred = sliding_window_inference(inputs=test_img_tensor, roi_size=(128, 128, 48), sw_batch_size=3,
    #                                         predictor=net2, overlap=0.25, device=device, progress=True)
    #
    #    sf_test_pred = np.argmax(np.squeeze(test_pred.detach().cpu().numpy()),
    #                             axis=0)  # np.squeeze(F.softmax(test_pred, dim=1).detach().cpu().numpy()), axis=0)
    #    test_pred_lab = np.transpose(sf_test_pred, (2, 0, 1)).astype("uint8")
    #    test_pred_lab_sitk = sitk.GetImageFromArray(test_pred_lab)
    #    test_pred_lab_sitk.CopyInformation(test_img_sitk)
    #    sitk.WriteImage(test_pred_lab_sitk,
    #                    os.path.join(args.label_save_dir, test_img.split("\\")[-1].replace("_0000.nii.gz", ".nii.gz")),
    #                    useCompression=True)
    #    break
    #