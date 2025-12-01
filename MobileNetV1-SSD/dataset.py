import os
import yaml
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from .Model.SSDs.ssd import MatchPrior
from .Model.SSDs.config import mobilenetv1_ssd_config
from .Model.SSDs.data_preprocessing import PredictionTransform
from .Model.SSDs.data_preprocessing import TrainAugmentation, TestTransform
from .Model.voc_utils.data_utils.voc_utils import load_class_names
from .Model.voc_utils.data_utils.dataset.voc_dataset import VOCDataset


"""
Loading model_cfg.yaml in the current directory
"""
now_dir=os.path.dirname(__file__)
with open(now_dir + "/model_cfg.yaml", 'r') as f:
    input_yaml = yaml.load(f, Loader=yaml.FullLoader)


"""
Dataset value max/min in FP32
"""
def dataset_cfg():
    return input_yaml


def VOC_classnames():
    class_names = load_class_names(now_dir + '/Model/voc_utils/voc.names')
    class_names = ["BACKGROUND"] + class_names
    return class_names


"""
Template for return data pair
"""
def return_dataset():
    config = mobilenetv1_ssd_config
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    training_dataset1 = VOCDataset(input_yaml['tra_dataset_path'], transform=train_transform, target_transform=target_transform, is_test='train')
    training_dataset2 = VOCDataset(input_yaml['val_dataset_path'], transform=train_transform, target_transform=target_transform, is_test='train')
    training_dataset = ConcatDataset([training_dataset1, training_dataset2])
    tra_dataloader = DataLoader(training_dataset, batch_size=input_yaml['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    cal_dataloader = VOCDataset(input_yaml['val_dataset_path'], is_test='train')
    val_dataloader = VOCDataset(input_yaml['val_dataset_path'], is_test='val')
    cos_dataloader = VOCDataset(input_yaml['val_dataset_path'], is_test='val')
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
    transform = PredictionTransform(300, np.array([127, 127, 127]), 128.0)

    for i in range(len(dataloader)):
        image = dataloader.get_image(i)
        height, width, _ = image.shape
        image = transform(image)
        inputs = image
        if len(inputs.unsqueeze(0).numpy().shape) == 4:
            input_data = inputs.unsqueeze(0)
            input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
            input_data = input_dtype(input_data.numpy().transpose(0, 2, 3, 1))
            input_data.tofile(save_path+"/test_" + str(data_count) + ".bin")
            data_count += 1
