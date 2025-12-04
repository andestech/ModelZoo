
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
from tqdm import tqdm
from torchvision import transforms
from .utils import SpeechCommand, Padding, Preprocess


"""
Loading model_cfg.yaml in the current directory
"""
now_dir = os.path.dirname(__file__)
with open(now_dir + "/model_cfg.yaml", "r") as f:
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
    transform = transforms.Compose([Padding()])
    train_dataset = SpeechCommand(input_yaml["tra_dataset_path"], 2, transform=transform)
    val_data = SpeechCommand(input_yaml["val_dataset_path"], 2, transform=transform)
    test_data = SpeechCommand(input_yaml["test_dataset_path"], 2, transform=transform)

    tra_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=input_yaml["batch_size"], shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=input_yaml["batch_size"], shuffle=False, num_workers=4, pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=input_yaml["batch_size"], shuffle=False, num_workers=4, pin_memory=True
    )
    cos_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )
    return tra_dataloader, test_dataloader, val_dataloader, cos_dataloader


def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    interpreter.allocate_tensors()
    device = "cpu"
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]["quantization_parameters"]["scales"][0]
    input_zp = input_details[0]["quantization_parameters"]["zero_points"][0]
    input_dtype = input_details[0]["dtype"]
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    data_count = 0
    noise_dir = dataset_cfg()["noise_dataset_path"]
    preprocess_test = Preprocess(noise_dir, device)
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            input_data = preprocess_test(inputs, labels=labels, is_train=False, augment=False)
            input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
            input_data = input_dtype(input_data.numpy().transpose(0, 2, 3, 1))
            for idx in range(input_data.shape[0]):
                np.expand_dims(input_data[idx], axis=0).tofile(save_path + "/test_" + str(data_count) + ".bin")
                data_count += 1
