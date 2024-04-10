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


class CombinedModel_orig(nn.Module):
    def __init__(self, unet):
        super(CombinedModel_orig, self).__init__()
        # loading the unet model
        self.unet = unet

    def forward(self, x):
        with torch.no_grad():
            recon_test_pred = self.unet(x)
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
    train_batch_size = data_loader_params[task_id]["batch_size"]

    # setting the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda")

    # loading validation dataloader
    properties, val_loader = get_data(args, mode="validation")

    # model names
    fold_no = 2
#    model_names = [r"fold0_uncertain_0", r"fold0_uncertain_1", r"fold0_uncertain_2"]
#    base_dir = f"/data/kidnAI/DynUnet/data/uncertainty_fold{fold_no}/predTs_img"
#
#    for model_fold in model_names:
#        for idx in range(390, 400):
#            net = get_network(properties, task_id, f"/data/kidnAI/DynUnet/data/model_uncertain_fold{fold_no}", f"{model_fold}_{idx}.pth")
#            net = net.to(device)
#            net.eval()
#
#            # combined model
#            net2 = CombinedModel_orig(net)
#
#            # getting the result for the validation set
#            list_key = "test"
#            args.label_save_dir = os.path.join(base_dir, f"{model_fold}_{idx}")
#            os.makedirs(args.label_save_dir, exist_ok=True)
#
#            datalist_name = "dataset_task{}.json".format(args.task_id)
#            dataset_path = os.path.join(args.root_dir, task_name[task_id])
#            datalist = load_decathlon_datalist(os.path.join(args.datalist_path, datalist_name), True, list_key,
#                                               dataset_path)
#            for item in datalist:
#                item['image'] = item['image'].replace(".nii.gz", "_0000.nii.gz")
#
#            for t_img in tqdm(datalist):
#                test_img = t_img["image"]
#                test_img_sitk = sitk.ReadImage(test_img)
#                test_img_tensor = torch.tensor(
#                    np.transpose(sitk.GetArrayFromImage(test_img_sitk), (1, 2, 0))[None, None, ...]).to(device)
#                test_img_tensor = test_img_tensor.type("torch.cuda.FloatTensor")
#                test_pred = sliding_window_inference(inputs=test_img_tensor, roi_size=(128, 128, 48), sw_batch_size=1,
#                                                     predictor=net2, overlap=0.25, device=device, progress=True)
#
#                sf_test_pred = np.squeeze(F.softmax(test_pred, dim=1).detach().cpu().numpy())[2, ...]
#                test_pred_lab = np.transpose(sf_test_pred, (2, 0, 1))
#                test_pred_lab_sitk = sitk.GetImageFromArray(test_pred_lab)
#                test_pred_lab_sitk.CopyInformation(test_img_sitk)
#                sitk.WriteImage(test_pred_lab_sitk, os.path.join(args.label_save_dir, test_img.split("/")[-1].replace("_0000.nii.gz", ".nii.gz")), useCompression=True)
#

    # model names
    model_names = [r"fold0_uncertain_0", r"fold0_uncertain_1", r"fold0_uncertain_2"]
    base_dir = f"/data/kidnAI/DynUnet/data/uncertainty_fold{fold_no}/predTr_img"

    for model_fold in model_names:
        for idx in range(390, 400):
            net = get_network(properties, task_id, f"/data/kidnAI/DynUnet/data/model_uncertain_fold{fold_no}", f"{model_fold}_{idx}.pth")
            net = net.to(device)
            net.eval()

            # combined model
            net2 = CombinedModel_orig(net)

            # getting the result for the validation set
            list_key = "{}_fold{}".format("validation", fold_no)  # "test" #
            args.label_save_dir = os.path.join(base_dir, f"{model_fold}_{idx}")
            os.makedirs(args.label_save_dir, exist_ok=True)

            datalist_name = "dataset_task{}.json".format(args.task_id)
            dataset_path = os.path.join(args.root_dir, task_name[task_id])
            datalist = load_decathlon_datalist(os.path.join(args.datalist_path, datalist_name), True, list_key,
                                               dataset_path)
            for item in datalist:
                item['image'] = item['image'].replace(".nii.gz", "_0000.nii.gz")

            for t_img in tqdm(datalist):
                test_img = t_img["image"]
                test_img_sitk = sitk.ReadImage(test_img)
                test_img_tensor = torch.tensor(
                    np.transpose(sitk.GetArrayFromImage(test_img_sitk), (1, 2, 0))[None, None, ...]).to(device)
                test_img_tensor = test_img_tensor.type("torch.cuda.FloatTensor")
                test_pred = sliding_window_inference(inputs=test_img_tensor, roi_size=(128, 128, 48), sw_batch_size=1,
                                                     predictor=net2, overlap=0.25, device=device, progress=True)

                sf_test_pred = np.squeeze(F.softmax(test_pred, dim=1).detach().cpu().numpy())[2, ...]
                test_pred_lab = np.transpose(sf_test_pred, (2, 0, 1))
                test_pred_lab_sitk = sitk.GetImageFromArray(test_pred_lab)
                test_pred_lab_sitk.CopyInformation(test_img_sitk)
                sitk.WriteImage(test_pred_lab_sitk, os.path.join(args.label_save_dir, test_img.split("/")[-1].replace("_0000.nii.gz", ".nii.gz")), useCompression=True)
