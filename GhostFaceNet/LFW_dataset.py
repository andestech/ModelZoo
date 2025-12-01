import torch
import numpy as np
import tensorflow as tf
from pathlib import Path
from torch.utils.data import Dataset


def tf_imread(input):
    return (tf.cast(tf.image.decode_image(input, channels=3), 'float32') - 127.5) * 0.0078125


class LFWDataset(Dataset):
    def __init__(self, torch_data_path, lfw_bin_path=None):
        if lfw_bin_path!=None:
            bins, issame_list = np.load(lfw_bin_path, encoding="bytes", allow_pickle=True)
            data_path = Path(torch_data_path)
            if data_path.is_dir():
                pass
            else:
                data_path.mkdir(parents=True, exist_ok=True)
            image_index = 0
            for idx, bin in enumerate(bins):
                image = torch.from_numpy(tf_imread(bin).numpy())
                data_file = f'input_{image_index}.pt'
                torch.save(image,data_path.joinpath(data_file))
                image_index = image_index + 1
            data_file = f'label.pt'
            torch.save(issame_list,data_path.joinpath(data_file))
            self.issame_list = issame_list + issame_list
        else:
            data_path = Path(torch_data_path)
            if data_path.is_file():
                print("not yet crate torch data from lfw.bin, pleae give lfw_bin_path")
            else:
                self.issame_list = torch.load(data_path.joinpath('label.pt')) + torch.load(data_path.joinpath('label.pt'))
        self.data_path = data_path

    def __len__(self):
        return len(self.issame_list)

    def __getitem__(self, idx):
        data_file = f'input_{idx}.pt'
        image = torch.load(self.data_path.joinpath(data_file))
        label =  self.issame_list[idx]
        return (image, label)
