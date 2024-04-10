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
import time

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
import matplotlib.pyplot as plt
import math
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator

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

class CombinedModel(nn.Module):
    def __init__(self, autoencoder, unet):
        super(CombinedModel, self).__init__()

        # loading the autoencoder model
        self.autoencoder = autoencoder
        # loading the unet model
        self.unet = unet

    def forward(self, x):
        with torch.no_grad():
            reconstruction, _, _ = self.autoencoder(x)
            recon_test_pred = self.unet(reconstruction)
            del reconstruction
        return recon_test_pred

class CombinedModel_orig(nn.Module):
    def __init__(self, autoencoder, unet):
        super(CombinedModel_orig, self).__init__()

        # loading the autoencoder model
        self.autoencoder = autoencoder
        # loading the unet model
        self.unet = unet

    def forward(self, x):
        with torch.no_grad():
            #reconstruction, _, _ = self.autoencoder(x)
            recon_test_pred = self.unet(x)
            #del reconstruction
        return recon_test_pred

if __name__ == "__main__":

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
    num_cycles = args.num_cycles
    uncertain_dir = args.uncertain_dir

    # loading batch size
    train_batch_size = data_loader_params[task_id]["batch_size"]

    # setting the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda")

    # loading validation dataloader
    properties, val_loader = get_data(args, mode="validation")

    # loading training dataloader
    _, train_loader = get_data(args, batch_size=train_batch_size, mode="train")

    # loading the network
    checkpoint = args.checkpoint
    net = get_network(properties, task_id, val_output_dir, "no_checkpoint.pth") # os.path.join(args.datalist_path, 'final_model.pth'))
    net = net.to(device)


    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.99,
        weight_decay=3e-5,
        nesterov=True,
    )

    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, batch=batch_dice)

    print(os.path.join(args.datalist_path, uncertain_dir, 'best_model_uncertain_fold0.pth'))
    # initializing early stopping
    early_stopping = EarlyStopping(patience=400, verbose=True,
                                   path=os.path.join(args.datalist_path, uncertain_dir, 'best_model_uncertain_fold0.pth'))

    # initialization of the learning rate scheduler
    budget_per_cycle = (max_epochs // num_cycles)
    print(f"Budget per cycle: {budget_per_cycle}")

    ## parameters related to the learning rate scheduler
    gamma = 0.8
    epsilon = 0.9
    alpha_r = 0.1
    alpha_o = 0.01
    T_c = budget_per_cycle
    T = max_epochs

    # Training loop
    for epoch in range(max_epochs):
        net.train()
        running_loss = 0.0
        t_running_dice_bg = 0.0
        t_running_dice_kd = 0.0
        t_running_dice_cst = 0.0

        # setting the learning rate
        cycle_no = (epoch // budget_per_cycle)
        t_small_c = epoch % T_c
        poly_frac = (1 - (np.minimum(t_small_c, gamma * T_c) / T_c)) ** epsilon

        if t_small_c == 0:
            optimizer.param_groups[0]['lr'] = alpha_r
        else:
            optimizer.param_groups[0]['lr'] = alpha_o * poly_frac

        # Wrap your data loader with tqdm for a progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{max_epochs}")
        for batch_idx, batch in progress_bar:
            inputs, targets = batch["image"], batch["label"]
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            preds = torch.unbind(outputs, dim=1)
            loss = sum([0.5 ** i * loss_function(p, targets) for i, p in enumerate(preds)])

            # calculating dice score approximations
            target_one_hot = one_hot(targets, num_classes=3)
            final_preds = preds[0]
            sf_final_preds = F.softmax(final_preds, dim=1)
            train_dices = dice_score(sf_final_preds, target_one_hot).detach().cpu().numpy()

            t_running_dice_bg += train_dices[0]
            t_running_dice_kd += train_dices[1]
            t_running_dice_cst += train_dices[2]

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Update the progress bar with the current loss
            progress_bar.set_postfix({'loss': running_loss / (batch_idx + 1)})


        #scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_train_bg_loss = t_running_dice_bg/ len(train_loader)
        epoch_train_kd_loss = t_running_dice_kd/ len(train_loader)
        epoch_train_cst_loss = t_running_dice_cst/ len(train_loader)

        print(f"Epoch {epoch + 1}/{max_epochs}, Training Loss: {epoch_loss:.4f},"
              f"Background Dice: {epoch_train_bg_loss:.4f}, Kidney Dice: {epoch_train_kd_loss:.4f},"
              f" Cyst Dice: {epoch_train_cst_loss:.4f}")

        # Validation loop
        net.eval()
        val_running_loss = 0.0
        v_running_dice_bg = 0.0
        v_running_dice_kd = 0.0
        v_running_dice_cst = 0.0

        # Wrap your validation data loader with tqdm as well
        val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")
        for batch_idx, val_batch in val_progress_bar:
            val_inputs, val_targets = val_batch["image"], val_batch["label"]
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

            val_preds = net(val_inputs)
            val_loss = loss_function(val_preds, val_targets)
            val_running_loss += val_loss.item()
            # calculating dice score approximations
            target_one_hot = one_hot(val_targets, num_classes=3)
            final_preds = val_preds
            sf_final_preds = F.softmax(final_preds, dim=1)
            val_dices = dice_score(sf_final_preds, target_one_hot).detach().cpu().numpy()

            v_running_dice_bg += val_dices[0]
            v_running_dice_kd += val_dices[1]
            v_running_dice_cst += val_dices[2]

            # Update the validation progress bar
            val_progress_bar.set_postfix({'val_loss': val_running_loss / (batch_idx + 1)})

        epoch_val_loss = val_running_loss/ len(val_loader)
        epoch_val_bg_loss = v_running_dice_bg/ len(val_loader)
        epoch_val_kd_loss = v_running_dice_kd/ len(val_loader)
        epoch_val_cst_loss = v_running_dice_cst/ len(val_loader)

        print(f"Epoch {epoch + 1}/{max_epochs}, Validation Loss: {epoch_val_loss:.4f}, "
              f"Background Dice: {epoch_val_bg_loss:.4f}, Kidney Dice: {epoch_val_kd_loss:.4f}, Cyst Dice: {epoch_val_cst_loss:.4f}")

        early_stopping(epoch_val_loss, net)
        print(os.path.join(args.datalist_path, uncertain_dir, f'fold0_uncertain_{cycle_no}_{t_small_c}.pth'))
        if t_small_c > (gamma * T_c):
            print(f"t_small_c : {t_small_c}, gamma * T_c : {gamma * T_c}")
            torch.save(net.state_dict(), os.path.join(args.datalist_path, uncertain_dir, f'fold0_uncertain_{cycle_no}_{t_small_c}.pth'))
        else:
            print(f"The current learning rate is {optimizer.param_groups[0]['lr']}")
            print(f"t_small_c : {t_small_c}, gamma * T_c : {gamma * T_c}")

    torch.save(net.state_dict(), os.path.join(args.datalist_path, uncertain_dir, 'final_model_cycleAtEnd.pth'))


