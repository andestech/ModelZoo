
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
import cv2
import yaml
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from .Model.blazeface.config import cfg_blaze
from .Model.blazeface.data.transform.data_augment import preproc
from .Model.blazeface.data.dataset.wider_face import WiderFaceDetection, detection_collate


"""
Loading model_cfg.yaml in the current sdirectory
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
    trainset_folder = input_yaml['trainset_folder']
    training_dataset = trainset_folder[:-7] + "label.txt"
    rgb_mean = (104, 117, 123)
    train_dataset = WiderFaceDetection(training_dataset,preproc(cfg_blaze['image_size'], rgb_mean))
    with open(now_dir + "/sample.txt", "rb") as fp:
        cal_samples = pickle.load(fp)
    cal_dataset = Subset(train_dataset, cal_samples)
    cal_loader = data.DataLoader(cal_dataset, input_yaml['batch_size'], 
                            shuffle=False, num_workers=4, 
                            collate_fn=detection_collate, pin_memory=True)
    train_loader = data.DataLoader(train_dataset, input_yaml['batch_size'], 
                            shuffle=True, num_workers=4, 
                            collate_fn=detection_collate, pin_memory=True)
    dataset_folder = input_yaml['testset_folder']
    testset_list = dataset_folder[:-7] + "wider_val.txt"
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    return train_loader, test_dataset, cal_loader, test_dataset


def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start inference Backend")
    model_cfg = dataset_cfg()
    save_folder = './results/'
    testset_folder = model_cfg['testset_folder']
    cfg = cfg_blaze
    num_correct = 0
    num_total = 0
    data_count = 0
    device="cpu"

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    input_dtype = input_details[0]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    output_scale = output_details[0]['quantization_parameters']['scales'][0]
    output_zp = output_details[0]['quantization_parameters']['zero_points'][0]
    
    for img_name in tqdm(dataloader):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = 640
        max_size = 640
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        height, width, _ = im_shape
        image_t = np.empty((im_size_max, im_size_max, 3), dtype=img.dtype)
        image_t[:, :] = (104, 117, 123)
        image_t[0:0 + height, 0:0 + width] = img
        img = cv2.resize(image_t, (max_size, max_size))
        resize = float(target_size) / float(im_size_max)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        img.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)

        input_data = input_dtype(img.numpy().transpose(0, 2, 3, 1))
        input_data.tofile(save_path + "/test_" + str(data_count) + ".bin")
        data_count += 1
