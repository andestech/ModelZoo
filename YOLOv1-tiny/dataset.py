import os
import yaml
from .voc import *
import random
import torch
import numpy as np
from .utils import decode_v1


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
    dataloader = voc_detection_yolov1(root= input_yaml["VOCdevkit_root_path"],batch_size=input_yaml["batch_size"])
    cos_dataloader = {}
    return dataloader["train"], dataloader["test"], dataloader["val"], dataloader["test"]


"""
Dataset value max/min in FP32
"""


def dataset_cfg():
    return input_yaml



def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    device="cpu"
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]["quantization_parameters"]["scales"][0]
    input_zp = input_details[0]["quantization_parameters"]["zero_points"][0]
    input_dtype = input_details[0]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    output_scale = output_details[0]["quantization_parameters"]["scales"][0]
    output_zp = output_details[0]["quantization_parameters"]["zero_points"][0]
    results = []
    a, b, c, d, e, f, g, h = [], [], [], [], [], [], [], []
    decode = decode_v1
    correct_count = 0
    data_count = 0
    for id in range(len(dataloader.dataset)):
        id = torch.tensor(id).to(device)
        input, label = dataloader.dataset[id]

        if len(input.numpy().shape) == 3:
            input_data = input.unsqueeze(0)
            input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
            input_data = input_dtype(input_data.numpy().transpose(0, 2, 3, 1))
            input_data.tofile(save_path+"/test_" + str(data_count ) + ".bin")
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

    val_dataset = voc_detection_yolov1()["val"].dataset
    return val_dataset
