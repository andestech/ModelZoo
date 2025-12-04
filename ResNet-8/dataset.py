
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
import numpy as np
from .cifar10_dataset import cifar10_train_dataset, cifar10_test_dataset
from .cls_dataset import ClassificationRGBDataset


"""
Loading model_cfg.yaml in the current directory
"""
now_dir=os.path.dirname(__file__)
with open(now_dir+"/model_cfg.yaml", 'r') as f:
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
    train_dataset = cifar10_train_dataset(input_yaml['tra_dataset_path'])
    val_dataset = cifar10_test_dataset(input_yaml['val_dataset_path'])
    cal_dataset = ClassificationRGBDataset(input_yaml['cal_dataset_path'])
    tra_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    cos_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    return tra_dataloader,val_dataloader,cal_dataloader,cos_dataloader


def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    interpreter.allocate_tensors()
    device = "cpu"
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    input_dtype = input_details[0]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    correct_count = 0
    data_count = 0

    for inputs, labels in dataloader:
        for i in range(inputs.shape[0]):
            if len(inputs[i].unsqueeze(0).numpy().shape) == 4:
                input_data = inputs[i].unsqueeze(0)
                input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
                input_data = input_dtype(input_data.numpy().transpose(0, 2, 3, 1))
                input_data.tofile(save_path+"/test_" + str(data_count) + ".bin")
                data_count += 1
