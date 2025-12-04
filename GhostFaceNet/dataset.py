
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
import pickle
import numpy as np
from . import data
from .LFW_dataset import LFWDataset
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import Dataset, Subset


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


class WiderfaceDataset(Dataset):
    def __init__(self, config, train=True,img_shape=(112, 112),random_status=0, random_crop=(100, 100, 3),random_cutout_mask_area=0.0) -> None:
        super().__init__()
        image_names, image_classes, embeddings, classes, _ = data.pre_process_folder(
        config['data_path'], None, None
        )
        self.train=train
        self.random_process_image = data.RandomProcessImage(img_shape, random_status, random_crop, random_cutout_mask_area)
        self.len=len(image_names)
        self.image_names = image_names
        self.image_classes = image_classes
        self.classes= classes

    def __len__(self):
        return self.len
    def __getitem__(self, index):
        image = data.tf_imread(self.image_names[index])
        if self.train:
            image = self.random_process_image.process(image)
        image = (image - 127.5) * 0.0078125
        label = self.image_classes[index]
        return torch.tensor(np.array(image)).permute(2,0,1),torch.tensor(label)


"""
Template for return data pair
"""
def return_dataset():
    dataset_params = {
        "data_path": input_yaml["tra_dataset_path"],
        "batch_size": input_yaml["batch_size"],
        "random_status": 0,
        "random_cutout_mask_area": 0,
        "image_per_class": 0,
        "mixup_alpha": 0,
        "teacher_model_interf": None,
    }
    tra_dataset = WiderfaceDataset(dataset_params)
    val_dataset = LFWDataset(input_yaml['val_dataset_path'])
    cal_samples = random.sample(range(0, len(val_dataset)), 500)
    with open(now_dir+'/golden.txt', 'rb') as fp:
        cal_samples = pickle.load(fp)
    cal_dataset = Subset(val_dataset, cal_samples)
    tra_dataloader = torch.utils.data.DataLoader(tra_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    cos_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    return tra_dataloader,val_dataloader,cal_dataloader,cos_dataloader


def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    interpreter.allocate_tensors()
    device="cpu"
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    input_dtype = input_details[0]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    correct_count = 0
    data_count = 0
    for sample in dataloader:
        image, label = sample[0].to(device), sample[1].numpy()
        input_data = image.permute(0, 1, 2, 3)
        input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
        for batch_idx in range(input_data.size(0)):
            input_data_sub = input_dtype(input_data[batch_idx].unsqueeze(0).numpy())
            input_data_sub.tofile(save_path+"/test_" + str(data_count) + ".bin")
            data_count += 1
