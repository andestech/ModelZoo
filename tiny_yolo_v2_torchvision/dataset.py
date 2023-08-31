import os
import yaml
import torch
from ModelZoo.tiny_yolo_v2_torchvision.voc import *
import random
from torch.utils.data import DataLoader, Subset
import numpy as np
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
    dataloader = voc_detection_yolov2(root=input_yaml["VOCdevkit_root_path"], batch_size=input_yaml["batch_size"])
    cos_dataset = Subset(dataloader["test"].dataset, [0])
    cos_dataloader=DataLoader(cos_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    return dataloader["train"], dataloader["test"], dataloader["train"], dataloader["cos"]


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

    if save_sample:
        idx_list = []
        data_id_path = input_yaml["val_dataset_path"] + "/ImageSets/Main/test.txt"
        f = open(data_id_path, "r")
        for line in f.readlines():
            print(line)
            idx_list.append(int(line))
        cal_samples = random.sample(idx_list, 200)
        print(cal_samples)
        f = open(now_dir + "/golden.txt", "w")
        for data in cal_samples:
            f.write(str(int(data)).zfill(6))
            f.write("\n")
        f.close()
        os.system(f"sudo cp " + now_dir + "/golden.txt %s//ImageSets/Main/golden.txt" % input_yaml["val_dataset_path"])
    if not os.path.exists(input_yaml["val_dataset_path"] + "/ImageSets/Main/golden.txt"):
        os.system(f"sudo cp " + now_dir + "/golden.txt %s//ImageSets/Main/golden.txt" % input_yaml["val_dataset_path"])

    val_dataset = voc_detection_yolov2()["val"].dataset
    return val_dataset

def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    device="cpu"
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_scale = input_details[0]["quantization_parameters"]["scales"][0]
    input_zp = input_details[0]["quantization_parameters"]["zero_points"][0]
    data_count = 0
    for id in range(len(dataloader.dataset)):
        id = torch.tensor(id).to(device)
        input, label = dataloader.dataset[id]

        if len(input.numpy().shape) == 3:
            input_data = input.unsqueeze(0)
            input_data.div_(input_scale).round_().add_(input_zp).clamp_(-128, 127)
            input_data = np.int8(input_data.numpy().transpose(0, 2, 3, 1))
            input_data.tofile(save_path+"/test_" + str(data_count ) + ".bin")
            data_count += 1
