import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from .tinystories import Task
from functools import partial

"""
Loading model_cfg.yaml in the current directory
"""
now_dir = os.path.dirname(__file__)
with open(now_dir + "/model_cfg.yaml", 'r') as f:
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
    batch_size = 1
    max_seq_len = 1024
    vocab_size = 32000
    vocab_source = 'llama2'

    iter_batches = partial(
        Task.iter_batches,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        vocab_source=vocab_source,
        device="cpu",
        num_workers=0,
    )

    tra_dataloader = iter_batches(split='train')
    val_dataloader = iter_batches
    cal_dataloader = iter_batches
    val_per_sample_dataloader = iter_batches
    return tra_dataloader, val_dataloader, cal_dataloader, val_per_sample_dataloader


def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    interpreter.allocate_tensors()
    device = "cpu"
    input_details = interpreter.get_input_details()

    input_ids_index = 0
    attention_mask_index = 1
    position_ids_index = 2

    input_ids_dtype = input_details[input_ids_index]['dtype']
    assert input_ids_dtype is np.int32, "Input id dtype must be int32 for Tiny Llama models, but got %s" % input_ids_dtype
    input_ids_qmin = np.iinfo(input_ids_dtype).min
    input_ids_qmax = np.iinfo(input_ids_dtype).max

    attention_mask_scale = input_details[attention_mask_index]['quantization_parameters']['scales'][0]
    attention_mask_zp = input_details[attention_mask_index]['quantization_parameters']['zero_points'][0]
    attention_mask_dtype = input_details[attention_mask_index]['dtype']
    attention_mask_qmin = np.iinfo(attention_mask_dtype).min
    attention_mask_qmax = np.iinfo(attention_mask_dtype).max

    position_ids_scale = input_details[position_ids_index]['quantization_parameters']['scales'][0]
    position_ids_zp = input_details[position_ids_index]['quantization_parameters']['zero_points'][0]
    position_ids_dtype = input_details[position_ids_index]['dtype']
    position_ids_qmin = np.iinfo(position_ids_dtype).min
    position_ids_qmax = np.iinfo(position_ids_dtype).max

    data_pair = dataloader(split='val')
    eval_iters = 5000

    with torch.no_grad():
        for count in tqdm(range(eval_iters)):
            input_ids, targets = next(data_pair)
            attention_mask = torch.ones(input_ids.size(-1)).reshape(1,-1).to(device).to(torch.float32)
            position_ids = torch.arange(0, input_ids.size(-1)).reshape(1,-1).to(device).to(torch.float32)

            input_ids_data = input_ids.to(device).to(torch.int32)
            input_ids_data.clamp_(input_ids_qmin, input_ids_qmax)
            input_ids_data = input_ids_dtype(input_ids_data.numpy())

            attention_mask_data = attention_mask.to(device).to(torch.float32)
            attention_mask_data.div_(attention_mask_scale).round_().add_(attention_mask_zp).clamp_(attention_mask_qmin, attention_mask_qmax)
            attention_mask_data = attention_mask_dtype(attention_mask_data.numpy())

            position_ids_data = position_ids.to(device).to(torch.float32)
            position_ids_data.div_(position_ids_scale).round_().add_(position_ids_zp).clamp_(position_ids_qmin, position_ids_qmax)
            position_ids_data = position_ids_dtype(position_ids_data.numpy())

            input_ids_data.tofile(save_path + "/test_" + str(count) + "_0.bin")
            attention_mask_data.tofile(save_path + "/test_" + str(count) + "_1.bin")
            position_ids_data.tofile(save_path + "/test_" + str(count) + "_2.bin")
