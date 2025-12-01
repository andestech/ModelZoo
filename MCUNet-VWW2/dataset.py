import os
import yaml
import torch
import pyvww
import numpy as np
from torchvision import transforms
from torch.utils.data import Subset


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
    channel=input_yaml['channel']
    if channel==3:
        data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(144),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(36./255.), saturation=(0.5, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5],[0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize((144, 144)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5],[0.5])
        ]),
        }
    else:
        data_transforms = {
        'train': transforms.Compose([
                transforms.RandomResizedCrop(96),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=(36./255.), saturation=(0.5, 1.5)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
        ]),
        'val': transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(100),
                transforms.CenterCrop(96),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
        }
    train_dataset = pyvww.pytorch.VisualWakeWordsClassification(root=input_yaml['data_source_tra'], annFile=input_yaml['ann_file_tra'], transform=data_transforms['train'])
    tra_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=input_yaml['batch_size'], shuffle=True, num_workers=4)
    val_dataset = pyvww.pytorch.VisualWakeWordsClassification(root=input_yaml['data_source_val'], annFile=input_yaml['ann_file_val'], transform=data_transforms['val'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=input_yaml['batch_size'], shuffle=False, num_workers=4)
    cal_dataloader = val_dataloader
    cos_dataset = Subset(val_dataset, [0])
    cos_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    return tra_dataloader,val_dataloader,cal_dataloader,cos_dataloader


def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    device="cpu"
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_dtype = input_details[0]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    data_count = 0

    for inputs, labels in dataloader:
        for i in range(inputs.shape[0]):
            if len(inputs[i].unsqueeze(0).numpy().shape) == 4:
                input_data = inputs[i].unsqueeze(0)
                input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
                input_data = input_dtype(input_data.numpy().transpose(0, 2, 3, 1))
                input_data.tofile(save_path+"/test_" + str(data_count) + ".bin")
                data_count += 1
