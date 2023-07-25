import torch
import numpy as np
import os
import yaml
import random
import importlib
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from .data_utils import ImageDataset
from .cifar10_dataset import cifar10_train_dataset,cifar10_val_dataset,cifar10_test_dataset
from .cls_dataset import ClassificationRGBDataset


"""
Loading model_cfg.yaml in the current directory
"""
now_dir=os.path.dirname(__file__)
with open(now_dir+"/model_cfg.yaml", 'r') as f:
    input_yaml = yaml.load(f, Loader=yaml.FullLoader)

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

"""
Dataset value max/min in FP32
"""
def dataset_cfg():
    return input_yaml

def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    interpreter.allocate_tensors()
    device="cpu"
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    correct_count = 0
    data_count = 0
    for inputs, labels in dataloader:
        for i in range(inputs.shape[0]):
            if len(inputs[i].unsqueeze(0).numpy().shape) == 4:
                input_data = inputs[i].unsqueeze(0)
                input_data.div_(input_scale).round_().add_(input_zp).clamp_(-128, 127)
                input_data = np.int8(input_data.numpy().transpose(0, 2, 3, 1))
                input_data.tofile(save_path+"/test_" + str(data_count) + ".bin")
                data_count += 1



"""
Load non-default dataset.yaml
"""

def load_dataset_cfg(data_cfg_path = "none"):
    if data_cfg_path=="none":
        print(f"Load default yaml from %s " % now_dir+"/model_cfg.yaml")
    with open(data_cfg_path, 'r') as f:
        input_yaml = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Load assigned yaml from %s " % data_cfg_path)
