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
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator

import numpy as np
import matplotlib.pyplot as plt
import math

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


def validation(args):
    # load hyper parameters
    task_id = args.task_id
    sw_batch_size = args.sw_batch_size
    tta_val = args.tta_val
    window_mode = args.window_mode
    eval_overlap = args.eval_overlap
    multi_gpu_flag = args.multi_gpu
    local_rank = args.local_rank
    amp = args.amp

    # produce the network
    checkpoint = args.checkpoint
    val_output_dir = "./runs_{}_fold{}_{}/".format(task_id, args.fold, args.expr_name)

    if multi_gpu_flag:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda")

    properties, val_loader = get_data(args, mode="validation")
    net = get_network(properties, task_id, val_output_dir, checkpoint)
    net = net.to(device)

    if multi_gpu_flag:
        net = DistributedDataParallel(module=net, device_ids=[device])

    num_classes = len(properties["labels"])

    net.eval()

    evaluator = DynUNetEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        num_classes=num_classes,
        inferer=SlidingWindowInferer(
            roi_size=patch_size[task_id],
            sw_batch_size=sw_batch_size,
            overlap=eval_overlap,
            mode=window_mode,
        ),
        postprocessing=None,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=False,
                output_transform=from_engine(["pred", "label"]),
            )
        },
        additional_metrics=None,
        amp=amp,
        tta_val=tta_val,
    )

    evaluator.run()
    if local_rank == 0:
        print(evaluator.state.metrics)
        results = evaluator.state.metric_details["val_mean_dice"]
        if num_classes > 2:
            for i in range(num_classes - 1):
                print("mean dice for label {} is {}".format(i + 1, results[:, i].mean()))

    if multi_gpu_flag:
        dist.destroy_process_group()


def train(args):
    # load hyper parameters
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


    if determinism_flag:
        set_determinism(seed=determinism_seed)
        if local_rank == 0:
            print("Using deterministic training.")

    # transforms
    train_batch_size = data_loader_params[task_id]["batch_size"]
    if multi_gpu_flag:
        dist.init_process_group(backend="nccl", init_method="env://")

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda")

    properties, val_loader = get_data(args, mode="validation")
    _, train_loader = get_data(args, batch_size=train_batch_size, mode="train")

    ## produce the network
    checkpoint = args.checkpoint
    net = get_network(properties, task_id, val_output_dir, checkpoint)
    net = net.to(device)

    #
    if multi_gpu_flag:
        net = DistributedDataParallel(module=net, device_ids=[device])

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.99,
        weight_decay=3e-5,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9)
    ## produce evaluator
    val_handlers = (
        [
            StatsHandler(output_transform=lambda x: None),
            CheckpointSaver(save_dir=val_output_dir, save_dict={"net": net}, save_key_metric=True),
        ]
        if idist.get_rank() == 0
        else None
    )

    evaluator = DynUNetEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        num_classes=len(properties["labels"]),
        inferer=SlidingWindowInferer(
            roi_size=patch_size[task_id],
            sw_batch_size=sw_batch_size,
            overlap=eval_overlap,
            mode=window_mode,
        ),
        postprocessing=None,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=False,
                output_transform=from_engine(["pred", "label"]),
            )
        },
        val_handlers=val_handlers,
        amp=amp_flag,
        tta_val=tta_val,
    )
    #
    ## produce trainer
    loss = DiceCELoss(to_onehot_y=True, softmax=True, batch=batch_dice)
    train_handlers = [ValidationHandler(validator=evaluator, interval=interval, epoch_level=True)]
    if lr_decay_flag:
        train_handlers += [LrScheduleHandler(lr_scheduler=scheduler, print_lr=True)]
    if idist.get_rank() == 0:
        train_handlers += [
            StatsHandler(
                tag_name="train_loss",
                output_transform=from_engine(["loss"], first=True),
            )
        ]

    trainer = DynUNetTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        postprocessing=None,
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp_flag,
    )
    #
    if local_rank > 0:
        evaluator.logger.setLevel(logging.WARNING)
        trainer.logger.setLevel(logging.WARNING)
    #
    trainer.run()
    if multi_gpu_flag:
        dist.destroy_process_group()

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

class CombinedModel_orig(nn.Module):
    def __init__(self, unet):
        super(CombinedModel_orig, self).__init__()
        # loading the unet model
        self.unet = unet

    def forward(self, x):
        with torch.no_grad():
            recon_test_pred = self.unet(x)
        return recon_test_pred


class CombinedModel_orig(nn.Module):
    def __init__(self, unet, recon_net):
        super(CombinedModel_orig, self).__init__()
        # loading the unet model
        self.unet = unet
        self.recon_net = recon_net

    def forward(self, x):
        with torch.no_grad():
            recon_x, _ , _ = self.recon_net(x)
            recon_test_pred = self.unet(recon_x)
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

    model_save_dir = r"/data/kidnAI/interpretability/check_data/models"

    # loading batch size
    train_batch_size = data_loader_params[task_id]["batch_size"]

    # setting the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda")

    # loading validation dataloader
    properties, val_loader = get_data(args, mode="validation")

    checkpoint = args.checkpoint
    fold=4
    list_key = "{}_fold{}".format("validation", fold) #"test" #
    args.label_save_dir = f"/data/kidnAI/DynUnet/data/results/pred_fold{fold}_recon"
    net = get_network(properties, task_id, f"/data/kidnAI/DynUnet/data/model_uncertain_fold{fold}", 'fold0_uncertain_2_399.pth')    # fold2, 1_392  #alternate fold4 2_390  fold1 0_385
    net = net.to(device)
    net.eval()

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
    model = torch.load(os.path.join(model_save_dir, "ct_checkpoint_train_cor.pt"))
    autoencoder.load_state_dict(model)
    autoencoder.eval()

    net2 = CombinedModel_orig(net, autoencoder)
    # getting the result for the validation set
    #list_key = "test" #"{}_fold{}".format("validation", 0) #
    #args.label_save_dir = r"/data/kidnAI/DynUnet/data/results/pred_labelsTs_recon"
    #datalist_name = "dataset_task{}.json".format(args.task_id)
    #dataset_path = os.path.join(args.root_dir, task_name[task_id])
    #datalist = load_decathlon_datalist(os.path.join(args.datalist_path, datalist_name), True, list_key, dataset_path)
    #for item in datalist:
    #    item['image'] = item['image'].replace(".nii.gz", "_0000.nii.gz")
#
    #for t_img in tqdm(datalist):
    #    test_img = t_img["image"]
    #    test_img_sitk = sitk.ReadImage(test_img)
    #    test_img_tensor = torch.tensor(np.transpose(sitk.GetArrayFromImage(test_img_sitk), (1,2,0))[None, None, ...]).to(device)
    #    test_img_tensor = test_img_tensor.type("torch.cuda.FloatTensor")
    #    test_pred = sliding_window_inference(inputs= test_img_tensor, roi_size = (128, 128, 48), sw_batch_size=1,
    #                             predictor = net2, overlap = 0.25, device = device, progress = True)
#
    #    sf_test_pred = np.argmax(np.squeeze(test_pred.detach().cpu().numpy()), axis=0)#np.squeeze(F.softmax(test_pred, dim=1).detach().cpu().numpy()), axis=0)
    #    test_pred_lab = np.transpose(sf_test_pred, (2, 0, 1)).astype("uint8")
    #    test_pred_lab_sitk = sitk.GetImageFromArray(test_pred_lab)
    #    test_pred_lab_sitk.CopyInformation(test_img_sitk)
    #    sitk.WriteImage(test_pred_lab_sitk, os.path.join(args.label_save_dir, test_img.split("/")[-1].replace("_0000.nii.gz", ".nii.gz")), useCompression=True)


    # getting list containing test dataset name

    datalist_name = "dataset_task{}.json".format(args.task_id)
    dataset_path = os.path.join(args.root_dir, task_name[task_id])
    datalist = load_decathlon_datalist(os.path.join(args.datalist_path, datalist_name), True, list_key, dataset_path)
    for item in datalist:
        item['image'] = item['image'].replace(".nii.gz", "_0000.nii.gz")

    for t_img in tqdm(datalist):
        test_img = t_img["image"]
        test_img_sitk = sitk.ReadImage(test_img)
        test_img_tensor = torch.tensor(np.transpose(sitk.GetArrayFromImage(test_img_sitk), (1,2,0))[None, None, ...]).to(device)
        test_img_tensor = test_img_tensor.type("torch.cuda.FloatTensor")
        test_pred = sliding_window_inference(inputs= test_img_tensor, roi_size = (128, 128, 48), sw_batch_size=1,
                                 predictor = net2, overlap = 0.25, device = device, progress = True)

        sf_test_pred = np.argmax(np.squeeze(test_pred.detach().cpu().numpy()), axis=0)#np.squeeze(F.softmax(test_pred, dim=1).detach().cpu().numpy()), axis=0)
        test_pred_lab = np.transpose(sf_test_pred, (2, 0, 1)).astype("uint8")
        test_pred_lab_sitk = sitk.GetImageFromArray(test_pred_lab)
        test_pred_lab_sitk.CopyInformation(test_img_sitk)
        sitk.WriteImage(test_pred_lab_sitk, os.path.join(args.label_save_dir, test_img.split("/")[-1].replace("_0000.nii.gz", ".nii.gz")), useCompression=True)
