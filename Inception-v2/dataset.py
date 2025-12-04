
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


import os
import yaml
import torch
import random
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset


"""
Loading model_cfg.yaml in the current directory
"""
now_dir = os.path.dirname(__file__)
with open(now_dir + "/model_cfg.yaml", 'r') as f:
    input_yaml = yaml.load(f, Loader=yaml.FullLoader)


"""
Dataset value max/min in FP32
"""
def dataset_cfg():
    return input_yaml


"""
Template for return data pair
"""
def return_dataset():
    tra_dataset = datasets.ImageFolder(
        input_yaml["tra_dataset_path"],
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )
    val_dataset = datasets.ImageFolder(
        input_yaml["val_dataset_path"],
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        ),
    )
    cal_dataset = datasets.ImageFolder(
        input_yaml["tra_dataset_path"],
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        ),
    )
    cal_samples = np.fromfile(now_dir+"/samples.txt", dtype=int, count = 10240).tolist()
    cal_dataset = Subset(cal_dataset, cal_samples) # extract specific calibration subset for quantiztation
    cos_dataset = Subset(val_dataset, [0])
    # Define training_dataset\validation_dataset\calibration_dataset(Subset of training_dataset)
    tra_dataloader = DataLoader(tra_dataset, batch_size=input_yaml['batch_size'], shuffle=True, num_workers=4, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=input_yaml['batch_size'], shuffle=False, num_workers=4, pin_memory=False)
    cal_dataloader = DataLoader(cal_dataset, batch_size=input_yaml['batch_size'], shuffle=False, num_workers=4, pin_memory=False)
    cos_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    del tra_dataset,val_dataset,cal_dataset,cos_dataset
    return tra_dataloader, val_dataloader, cal_dataloader, cos_dataloader


def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    device="cpu"
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    input_dtype = input_details[0]["dtype"]
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    data_count = 0
    for inputs, labels in dataloader:
        for i in range(inputs.shape[0]):
            if len(inputs[i].unsqueeze(0).numpy().shape) == 4:
                input_data = inputs[i].unsqueeze(0)
                input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin,input_qmax)
                input_data = input_dtype(input_data.numpy().transpose(0, 2, 3, 1))
                input_data.tofile(save_path+"/test_" + str(data_count) + ".bin")
                data_count += 1
            else:
                raise Exception("Error shape number")
