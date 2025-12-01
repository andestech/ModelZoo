import os
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


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


def order(x):
    return int(x)


class NumpyDataset(Dataset):
	def __init__(self, data_dir, transform=None):
		self.transform = transform
		class_folder = sorted(os.listdir(data_dir), key=order)
		self.labels = []
		self.datas = []
		for i in class_folder:
			file_list = sorted(os.listdir(os.path.join(data_dir, i)))
			for j in file_list:
				self.datas.append(os.path.join(data_dir, i, j))
				self.labels.append(int(i))

	def __getitem__(self, index):
		file_path = self.datas[index]
		# img = Image.open(imgpath).convert('RGB')
		data = np.load(file_path)
		lbl = int(self.labels[index])
		if self.transform is not None:
			# import pdb; pdb.set_trace()
			data = self.transform(data)
		return data, lbl

	def __len__(self):
		return len(self.datas)


"""
Template for return data pair
"""
def return_dataset():
    train_dataset = NumpyDataset(input_yaml['tra_dataset_path'], transform=transforms.ToTensor())
    tra_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=input_yaml['batch_size'], shuffle=True, num_workers=4, pin_memory=False)
    # cal == val
    val_dataset = NumpyDataset(input_yaml['val_dataset_path'], transform=transforms.ToTensor())
    cal_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=input_yaml['batch_size'], shuffle=False, num_workers=4, pin_memory=False)
    # test == val
    test_dataset = NumpyDataset(input_yaml['test_dataset_path'], transform=transforms.ToTensor())
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=input_yaml['batch_size'], shuffle=False, num_workers=4, pin_memory=False)
    cos_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
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
    dataloader.reset = True
    for inputs, labels in dataloader:
        for i in range(inputs.shape[0]):
            if len(inputs[i].unsqueeze(0).numpy().shape) == 4:
                input_data = inputs[i].unsqueeze(0)
                input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
                input_data = input_data.permute(0, 2, 3, 1)
                input_data = input_dtype(input_data.numpy())
                input_data.tofile(save_path+"/test_" + str(data_count) + ".bin")
                data_count += 1
