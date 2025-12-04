
# Copyright (C) 2023-2025 Andes Technology Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import os
import yaml
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from .widerface_dataset import WiderfaceDataset, WiderFaceDetection, preproc, detection_collate


"""
Loading model_cfg.yaml in the current directory
"""
now_dir = os.path.dirname(__file__)
with open(now_dir + "/model_cfg.yaml", "r") as f:
    input_yaml = yaml.load(f, Loader=yaml.FullLoader)

"""
Template for return data pair
"""


def return_dataset():
    rgb_mean = (104, 117, 123)
    tra_dataset = WiderFaceDetection(
        input_yaml["tra_dataset_path"] + "/label.txt", preproc(input_yaml["image_size"], rgb_mean)
    )
    val_dataset = WiderfaceDataset(
        input_yaml["val_dataset_path"], image_size=input_yaml["image_size"], steps=input_yaml["steps"], cache=True
    )
    with open(now_dir + "/golden.txt", "rb") as fp:
        cal_samples = pickle.load(fp)
    cal_dataset = Subset(val_dataset, cal_samples)
    tra_dataloader = DataLoader(
        tra_dataset, batch_size=input_yaml["batch_size"], shuffle=True, num_workers=1, collate_fn=detection_collate
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    cal_dataloader = DataLoader(cal_dataset, batch_size=1, shuffle=False, num_workers=4)
    cos_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    return tra_dataloader, val_dataloader, cal_dataloader, cos_dataloader


"""
Dataset value max/min in FP32
"""


def dataset_cfg():
    return input_yaml


def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    interpreter.allocate_tensors()
    device = "cpu"
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]["quantization_parameters"]["scales"][0]
    input_zp = input_details[0]["quantization_parameters"]["zero_points"][0]
    input_dtype = input_details[0]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    correct_count = 0
    data_count = 0
    for sample in tqdm(dataloader):
        img, pad_params, img_path = sample[0].to(device), sample[1], sample[2][0]
        input_data = img.div(127.5).sub(1)
        input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
        input_data = input_dtype(input_data.numpy().transpose(0, 2, 3, 1))
        input_data.tofile(save_path + "/test_" + str(data_count) + ".bin")
        data_count += 1


"""
Load non-default dataset.yaml
"""


def load_dataset_cfg(data_cfg_path="none"):
    if data_cfg_path == "none":
        print(f"Load default yaml from %s " % now_dir + "/model_cfg.yaml")
    with open(data_cfg_path, "r") as f:
        input_yaml = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Load assigned yaml from %s " % data_cfg_path)
