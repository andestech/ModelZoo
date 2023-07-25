import torch
import torchvision
from torchvision import transforms
import numpy as np
import os
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import datasets
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class NumpyDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, data_dir, transform=None):
		# --------------------------------------------
		# Initialize paths, transforms, and so on
		# --------------------------------------------
		self.transform = transform
		# Load image path and annotations
		class_folder = os.listdir(data_dir)
		self.labels = []
		self.datas = []
		for i in class_folder:
			file_list = os.listdir(os.path.join(data_dir, i))
			for j in file_list:
				self.datas.append(os.path.join(data_dir, i, j))
				self.labels.append(int(i))

	def __getitem__(self, index):
		# --------------------------------------------
		# 1. Read from file (using numpy.fromfile, PIL.Image.open)
		# 2. Preprocess the data (torchvision.Transform)
		# 3. Return the data (e.g. image and label)
		# --------------------------------------------
		file_path = self.datas[index]
		# img = Image.open(imgpath).convert('RGB')
		data = np.load(file_path)
		lbl = int(self.labels[index])
		if self.transform is not None:
			# import pdb; pdb.set_trace()
			data = self.transform(data)
		return data, lbl

	def __len__(self):
		# --------------------------------------------
		# Indicate the total size of the dataset
		# --------------------------------------------
		return len(self.datas)

class ImageDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, data_dir, transform=None):
		# --------------------------------------------
		# Initialize paths, transforms, and so on
		# --------------------------------------------
		self.transform = transform
		# Load image path and annotations
		os.path.join(data_dir)
		class_folder = sorted(os.listdir(data_dir))
		self.labels = []
		self.datas = []
		for i, name in enumerate(class_folder):
			file_list = os.listdir(os.path.join(data_dir, name))
			for j in file_list:
				self.datas.append(os.path.join(data_dir, name, j))
				self.labels.append(int(i))

	def __getitem__(self, index):
		# --------------------------------------------
		# 1. Read from file (using numpy.fromfile, PIL.Image.open)
		# 2. Preprocess the data (torchvision.Transform)
		# 3. Return the data (e.g. image and label)
		# --------------------------------------------
		file_path = self.datas[index]
		img = Image.open(file_path).convert('RGB')
		lbl = int(self.labels[index])		
		if self.transform is not None:
			# import pdb; pdb.set_trace()
			img = self.transform(img)
			img = np.transpose(img, (2, 0, 1)).astype(np.float32)
			img = torch.Tensor(img)		

		else:
			img = np.array(img)
			img = np.transpose(img, (2, 0, 1)).astype(np.float32)
			img = torch.Tensor(img)
		return img, lbl

	def __len__(self):
		# --------------------------------------------
		# Indicate the total size of the dataset
		# --------------------------------------------
		return len(self.datas)

def vww_Data():
    dataloaders = {}
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(96),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(brightness=(36./255.), saturation=(0.5, 1.5)), #0.5, 1.5
			transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((96, 96)),
			transforms.ToTensor(),
        ]),
    }
    
    train_dataset = pyvww.pytorch.VisualWakeWordsClassification(root="/local1/dataset/visualwakewords/coco/all", annFile="/local1/dataset/visualwakewords/coco/annotations/instances_train.json", transform=data_transforms['train'])
    dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    #dataloaders['train'] = []
    val_dataset = pyvww.pytorch.VisualWakeWordsClassification(root="/local1/dataset/visualwakewords/coco/all", annFile="/local1/dataset/visualwakewords/coco/annotations/instances_val.json", transform=data_transforms['val'])
    dataloaders['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    #train_dataset = None
    #val_dataset = None
    #del train_dataset, val_dataset 
    
    return train_dataset, val_dataset, dataloaders

def kws_Data():
    train_dataset = NumpyDataset("/local1/dataset/kws_processed_data/train", transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = NumpyDataset("/local1/dataset/kws_processed_data/val", transform=transforms.ToTensor())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = NumpyDataset("/local1/dataset/kws_processed_data/test", transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader
    dataloaders['test'] = test_loader
    
    train_loader = None
    val_loader = None
    test_loader = None
    del test_loader, train_loader, val_loader
    
    return train_dataset, val_dataset, dataloaders

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
def cifar10_Data():
	#print()
	dataloaders = {}
	mean = {
	'cifar10': (0.4914, 0.4822, 0.4465),
	'cifar100': (0.4914, 0.4822, 0.4465), }

	std = {
	'cifar10': (0.2470, 0.2435, 0.2616),
	'cifar100': (0.2023, 0.1994, 0.2010), }
    
	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
		]),

		'test': transforms.Compose([
			transforms.ToTensor(),
		]),
    }
	
	train_dataset = ImageDataset(os.path.join('/local1/dataset/cifar10/DATA/cifar10/cifar10/','train'))#, transform=data_transforms['train'])
	val_dataset = ImageDataset(os.path.join('/local1/dataset/cifar10/DATA/cifar10/cifar10/','test'))

	dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
	dataloaders['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
	return train_dataset, val_dataset, dataloaders
	"""
	max_value = torch.tensor(0.)
	min_value = torch.tensor(999.)
	for inputs, labels in dataloaders['val']:
		max_value = max(max_value,inputs.max())
		min_value = min(min_value,inputs.min())

	for inputs, labels in dataloaders['train']:
		max_value = max(max_value,inputs.max())
		min_value = min(min_value,inputs.min())

	scale_Q = max(max(abs(max_value),abs(min_value))/127,  1e-20)
	scale_uQ = max((max_value-min_value)/255,1e-20)
	zp = min_value.div(scale_uQ).round()
	torch.set_printoptions(precision=10)
	print(max_value, min_value)
	print(scale_Q, scale_uQ, zp)
	"""
	#return train_dataset, val_dataset, dataloaders
    
    
    





