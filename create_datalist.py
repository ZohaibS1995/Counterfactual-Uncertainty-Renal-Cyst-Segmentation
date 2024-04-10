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

import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from sklearn.model_selection import KFold


def create_datalist(
    dataset_file_path: str,
    output_dir: str,
    task_name: str,
    num_folds: int,
    seed: int,
):

    with open(dataset_file_path, "r") as f:
        dataset = json.load(f)

    dataset_with_folds = dataset.copy()

    keys = [line["image"].split("/")[-1].split(".")[0] for line in dataset["training"]]
    dataset_train_dict = dict(zip(keys, dataset["training"]))
    all_keys_sorted = np.sort(keys)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
        val_data = []
        train_data = []
        train_keys = np.array(all_keys_sorted)[train_idx]
        test_keys = np.array(all_keys_sorted)[test_idx]
        for key in test_keys:
            val_data.append(dataset_train_dict[key])
        for key in train_keys:
            train_data.append(dataset_train_dict[key])

        dataset_with_folds["validation_fold{}".format(i)] = val_data
        dataset_with_folds["train_fold{}".format(i)] = train_data
    del dataset

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "dataset_{}.json".format(task_name)), "w") as f:
        json.dump(dataset_with_folds, f)
        print("data list for {} has been created!".format(task_name))
        f.close()


if __name__ == "__main__":
    base_path = r"/data/kidnAI/DynUnet/data"

    create_datalist(
        dataset_file_path=os.path.join(base_path, r"dataset.json"),
        output_dir= base_path,
        task_name=r"task001_KidnAI",
        num_folds=5,
        seed=42,
    )
