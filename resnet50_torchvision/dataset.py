import torch
import numpy as np
import os
import yaml
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import random

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
    tra_dataset = datasets.ImageFolder(
        input_yaml["tra_dataset_path"],
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
    cal_samples = np.fromfile(now_dir + "/samples.txt", dtype=int, count=10240).tolist()
    cal_dataset = Subset(cal_dataset, cal_samples)  # extract specific calibration subset for quantiztation
    cos_dataset = Subset(val_dataset, [0])
    """
    Define training_dataset\validation_dataset\calibration_dataset(Subset of training_dataset)
    """
    tra_dataloader = DataLoader(tra_dataset, batch_size=input_yaml["batch_size"], shuffle=True, num_workers=6, pin_memory=False, prefetch_factor=5)
    val_dataloader = DataLoader(val_dataset, batch_size=input_yaml["batch_size"], shuffle=False, num_workers=6, pin_memory=False, prefetch_factor=5)
    cal_dataloader = DataLoader(cal_dataset, batch_size=input_yaml["batch_size"], shuffle=False, num_workers=6, pin_memory=False, prefetch_factor=5)
    cos_dataloader = DataLoader(cos_dataset, batch_size=1, shuffle=False, num_workers=6, pin_memory=False, prefetch_factor=5)
    del tra_dataset, val_dataset, cal_dataset, cos_dataset
    return tra_dataloader, val_dataloader, cal_dataloader, cos_dataloader


"""
Dataset value max/min in FP32
"""


def dataset_cfg():
    return input_yaml


"""
Load non-default dataset.yaml
"""


def load_dataset_cfg(data_cfg_path="none"):
    if data_cfg_path == "none":
        print(f"Load default yaml from %s " % now_dir + "/model_cfg.yaml")
    with open(data_cfg_path, "r") as f:
        input_yaml = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Load assigned yaml from %s " % data_cfg_path)


def prepare_golden(save_sample=False):
    import pickle

    val_dataset = datasets.ImageFolder(
        input_yaml["val_dataset_path"],
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
    if save_sample:
        cal_samples = random.sample(range(0, len(val_dataset)), 200)
        with open(now_dir + "/golden.txt", "wb") as fp:
            pickle.dump(cal_samples, fp)
    else:
        pass
    with open(now_dir + "/golden.txt", "rb") as fp:
        cal_samples = pickle.load(fp)
    print(cal_samples)
    cal_dataset = Subset(val_dataset, cal_samples)
    val_dataloader = DataLoader(cal_dataset, batch_size=input_yaml["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    return val_dataloader

def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    device="cpu"
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
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
            else:
                raise Exception("Error shape number")

