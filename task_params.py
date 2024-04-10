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

task_name = {
    "01": "Task01_KidnAI",
}
patch_size = {
    "01": [128, 128, 48]
}

spacing = {
    "01": [0.77, 0.77, 2.5],
}

clip_values = {
    "01": [-3.23, 3.64]
}

normalize_values = {
    "01": [-0.12, 0.91],
}

data_loader_params = {
    "01": {"batch_size": 1}
}

deep_supr_num = {
    "01": 4
}
