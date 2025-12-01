import numpy as np
import os
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset, Subset
import torch
import pickle
import sys
import random
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

class TensorFlowDataset(Dataset):
    def __init__(self, tensorflow_dataset, train=False) -> None:
        super().__init__()
        self.dataset = list(tensorflow_dataset[0].as_numpy_iterator())
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.train and index == 0:
            self.randind = random.sample(range(0, len(self.dataset)), len(self.dataset))
            audio, labels = self.dataset[self.randind[index]]
        else:
            audio, labels = self.dataset[index]
        return torch.tensor(audio).unsqueeze(0).permute(0, 3, 1, 2), torch.tensor(labels)


"""
Loading model_cfg.yaml in the current directory
"""
now_dir = os.path.dirname(__file__)
with open(now_dir + "/model_cfg.yaml", "r") as f:
    input_yaml = yaml.load(f, Loader=yaml.FullLoader)

"""
Template for return data pair
"""

sys.path.append(now_dir + "/recreate_code")
from .recreate_code.train_model import get_data


def return_dataset():
    datasets = get_data(input_yaml["tra_dataset_path"], input_yaml["batch_size"])
    train_dataset = ConcatDataset(
        [
            # TensorFlowDataset(datasets["train_full_size"], True),
            TensorFlowDataset(datasets["train_fluent_speech"], True),
            # TensorFlowDataset(datasets["train_reduced_size"], True),
        ]
    )
    val_dataset = TensorFlowDataset(datasets["test_fluent_speech"])
    cal_samples = random.sample(range(0, len(val_dataset)), 500)
    with open(now_dir + "/samples.txt", "rb") as fp:
        cal_samples = pickle.load(fp)
    cal_dataset = TensorFlowDataset(datasets["val_fluent_speech"])
    return train_dataset, val_dataset, cal_dataset, val_dataset


"""
Dataset value max/min in FP32
"""


def dataset_cfg():
    return input_yaml


def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    device = "cpu"
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]["quantization_parameters"]["scales"][0]
    input_zp = input_details[0]["quantization_parameters"]["zero_points"][0]
    input_dtype = input_details[0]["dtype"]
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    data_count = 0

    input_window_length = 296
    dataloader.reset = True
    for inputs, targets in tqdm(dataloader):
        targets = targets.to(device, non_blocking=True).cpu().detach().numpy()
        inputs = inputs.numpy()
        while inputs.shape[3] < input_window_length:
            inputs = np.append(inputs, inputs[:, :, :, -2:-1], axis=3)
        if inputs.shape[3] % 2 == 1:
            inputs = np.concatenate([inputs, np.zeros((1, inputs.shape[1], 1, 1), dtype=inputs.dtype)], axis=3)
        context = 24 + 2 * (7 * 3 + 16)  # = 98 - theoretical max receptive field on each side
        size = 296
        inner = size - 2 * context
        data_end = inputs.shape[3]
        data_pos = 0
        num_samples = 0
        os.makedirs(save_path + f"/test_{data_count}", exist_ok=True)
        while data_pos < data_end:
            if data_pos == 0:
                # Align inputs from the first window to the start of the data and include the intial context in the output
                start = data_pos
                end = start + size
                y_start = 0
                y_end = y_start + (size - context) // 2
                data_pos = end - context
            elif data_pos + inner + context >= data_end:
                # Shift left to align final window to the end of the data and include the final context in the output
                shift = (data_pos + inner + context) - data_end
                start = data_pos - context - shift
                end = start + size
                assert start >= 0
                y_start = (shift + context) // 2  # Will be even because we assert it above
                y_end = size // 2
                data_pos = data_end
            else:
                # Capture only the inner region from mid-input inferences, excluding output from both context regions
                start = data_pos - context
                end = start + size
                y_start = context // 2
                y_end = y_start + inner // 2
                data_pos = end - context
            input = torch.from_numpy(inputs[:, :, :, start:end].copy())
            input.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
            input = input.permute(0, 2, 3, 1)
            input = input_dtype(input.numpy())
            input.tofile(save_path + f"/test_{data_count}/feature_{num_samples}.bin")
            num_samples += 1

        np.array(num_samples, dtype=np.int32).tofile(save_path + f"/test_{data_count}/num_samples.bin")
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
