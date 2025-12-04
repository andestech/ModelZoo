
# Copyright (C) 2023-2025 Andes Technology Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from jiwer import wer
import librosa
"""
Pre_qunat model input tools
"""


def QuantStub(input, min_val=-1.0, max_val=1.0, symm=True, bits=8, isHW=False):
    assert max_val > min_val, "max_val must larger than min_val"
    if symm:
        clamp_min = -((2 ** (bits - 1)))  # for bits=8 -128
        clamp_max = 2 ** (bits - 1) - 1  # for bits=8 127
        scale = torch.max(torch.tensor(min_val).abs(), torch.tensor(max_val).abs()).div((2 ** (bits - 1)) - 1)
        zero_point = torch.tensor(0.0)
    else:
        clamp_min = 0
        clamp_max = (2**bits) - 1
        scale = (torch.tensor(max_val) - torch.tensor(min_val)).div((2**bits) - 1)
        zero_point = torch.tensor(min_val).div(scale).round()
    if isHW:
        if symm:
            input.div_(scale).sub_(zero_point).round_().clamp_(clamp_min, clamp_max)
        else:
            input.div_(scale).sub_(zero_point).sub_(128).round_().clamp_(-128, 127).add_(128).add_(zero_point)
    else:
        input.div_(scale).sub_(zero_point).round_().clamp_(clamp_min, clamp_max).add_(zero_point).mul_(scale)


"""
Define Floating point 32(FP32) inference
"""


def ctc_preparation(tensor, y_predict):
    if len(y_predict.shape) == 4:
        y_predict = tf.squeeze(y_predict, axis=1)
    y_predict = tf.transpose(y_predict, (1, 0, 2))
    sequence_lengths, labels = tensor[:, 0], tensor[:, 1:]
    idx = tf.where(tf.not_equal(labels, 28))
    sparse_labels = tf.SparseTensor(idx, tf.gather_nd(labels, idx), tf.shape(labels, out_type=tf.int64))
    return sparse_labels, sequence_lengths, y_predict


def ctc_ler(y_true, y_predict):
    sparse_labels, logit_length, y_predict = ctc_preparation(y_true, y_predict)
    decoded, log_probabilities = tf.nn.ctc_greedy_decoder(y_predict, tf.cast(logit_length, tf.int32), merge_repeated=True)
    return tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), tf.cast(sparse_labels, tf.int32))).numpy()


def ctc_wer(y_true, y_predict):
    """Calculate CTC WER (Word Error Rate) only for batch size = 1."""

    def trans_int_to_string(trans_int):
        # create dictionary int -> string (0 -> a 1 -> b)
        string = ""
        alphabet = "abcdefghijklmnopqrstuvwxyz' @"
        alphabet_dict = {}
        count = 0
        for x in alphabet:
            alphabet_dict[count] = x
            count += 1
        for letter in trans_int:
            letter_np = np.array(letter).item(0)
            if letter_np != 28:
                string += alphabet_dict[letter_np]
        return string

    sparse_labels, logit_length, y_predict = ctc_preparation(y_true, y_predict)
    decoded, log_probabilities = tf.nn.ctc_greedy_decoder(y_predict, tf.cast(logit_length, tf.int32), merge_repeated=True)
    true_sentence = tf.cast(sparse_labels.values, tf.int32)
    wer_value = wer(str(trans_int_to_string(true_sentence)), str(trans_int_to_string(decoded[0].values)))
    # debug
    # if wer_value > 0.9:
    #     print(f"Target: {str(trans_int_to_string(true_sentence))}")
    #     print(f"Output: {str(trans_int_to_string(decoded[0].values))}")
    return wer_value


def inference_FP32(model, dataloader, device):
    lers = []
    model.to(device)
    # wers = []
    for inputs, targets in tqdm(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).cpu().detach().numpy()
        outputs = model(inputs).permute(0,  1, 2,3).cpu().detach().numpy()
        lers.append(ctc_ler(targets, outputs))
        # wers.append(ctc_wer(targets, outputs))

    ler = sum(lers) / len(lers) * 100
    # wer = sum(wers) / len(wers) * 100
    print(f"LER: {ler}")
    return 100 - ler

def normalize(values):
    """
    Normalize values to mean 0 and std 1
    """
    return (values - np.mean(values)) / np.std(values)

def transform_audio_to_mfcc(audio_file, transcript, n_mfcc=13, n_fft=512, hop_length=160):
    audio_data, sample_rate = librosa.load(audio_file, sr=16000)

    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # add derivatives and normalize
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc = np.concatenate((normalize(mfcc), normalize(mfcc_delta), normalize(mfcc_delta2)), axis=0)

    seq_length = mfcc.shape[1] // 2

    sequences = np.concatenate([[seq_length], transcript]).astype(np.int32)
    sequences = np.expand_dims(sequences, 0)
    mfcc_out = mfcc.T.astype(np.float32)
    mfcc_out = np.expand_dims(mfcc_out, 0)
    return mfcc_out, sequences

'''def inference_FP32(model, dataloader, device):
    lers = []
    # wers = []
    input_window_length = 296
    for inputs, targets in tqdm(dataloader):
        #inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).cpu().detach().numpy()
        inputs = inputs.numpy()
        while inputs.shape[3] < input_window_length:
            inputs = np.append(inputs, inputs[:, :,:, -2:-1], axis=3)
        if inputs.shape[3] % 2 == 1:
            inputs = np.concatenate([inputs, np.zeros((1, inputs.shape[1], 1, 1), dtype=inputs.dtype)], axis=3)
        context = 24 + 2 * (7 * 3 + 16)  # = 98 - theoretical max receptive field on each side
        size = 296
        inner = size - 2 * context
        data_end = inputs.shape[3]
        data_pos = 0
        outputs = []
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
            input=torch.from_numpy(inputs[:, :,:,start:end])
            output = model(input)[:,:,:,y_start:y_end]
            outputs.append(output)
        result=torch.cat(outputs,3).permute(0,2,3,1)
        lers.append(ctc_ler( targets,result.cpu().detach().numpy()))

    ler = sum(lers) / len(lers) * 100
    print(f"LER: {ler}")
    return 100 - ler
'''

"""
define Fake Quantization (FQ) model inferences
"""


def inference_FQ(model, dataloader, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    print("Start FQ inference")
    lers = []
    # wers = []
    print(calibration)
    input_window_length = 296
    dataloader.reset = True
    for inputs, targets in tqdm(dataloader):
        #inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).cpu().detach().numpy()
        inputs = inputs.numpy()
        while inputs.shape[3] < input_window_length:
            inputs = np.append(inputs, inputs[:, :,:, -2:-1], axis=3)
        if inputs.shape[3] % 2 == 1:
            inputs = np.concatenate([inputs, np.zeros((1, inputs.shape[1], 1, 1), dtype=inputs.dtype)], axis=3)
        context = 24 + 2 * (7 * 3 + 16)  # = 98 - theoretical max receptive field on each side
        size = 296
        inner = size - 2 * context
        data_end = inputs.shape[3]
        data_pos = 0
        outputs = []
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
            input=torch.from_numpy(inputs[:, :,:,start:end])#.permute(0,2,3,1).reshape(1,-1,39)
            output = model(input.to(device))[:,:,y_start:y_end,:]
            outputs.append(output)
        result=torch.cat(outputs,2)
        lers.append(ctc_ler( targets,result.cpu().detach().numpy()))
    if calibration:
        print("no calibration")
        return 0.0
    ler = sum(lers) / len(lers) * 100
    print(f"LER: {ler}")
    return 100 - ler


"""
Define Hardware(HW) Quantization model inference
"""


def inference_HW(model, dataloader, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    print("Start acuracy estimator inference")
    lers = []
    # wers = []
    print(calibration)
    input_window_length = 296
    dataloader.reset = True
    for inputs, targets in tqdm(dataloader):
        #inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).cpu().detach().numpy()
        inputs = inputs.numpy()
        while inputs.shape[3] < input_window_length:
            inputs = np.append(inputs, inputs[:, :,:, -2:-1], axis=3)
        if inputs.shape[3] % 2 == 1:
            inputs = np.concatenate([inputs, np.zeros((1, inputs.shape[1], 1, 1), dtype=inputs.dtype)], axis=3)
        context = 24 + 2 * (7 * 3 + 16)  # = 98 - theoretical max receptive field on each side
        size = 296
        inner = size - 2 * context
        data_end = inputs.shape[3]
        data_pos = 0
        outputs = []
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
            input=torch.from_numpy(inputs[:, :,:,start:end])#.permute(0,2,3,1).reshape(1,-1,39)
            output = model(input.to(device))[:,:,y_start:y_end,:]
            outputs.append(output)
        result=torch.cat(outputs,2)
        lers.append(ctc_ler( targets,result.cpu().detach().numpy()))
    if calibration:
        print("no calibration")
        return 0.0
    ler = sum(lers) / len(lers) * 100
    print(f"LER: {ler}")
    return 100 - ler


def inference_Backend(interpreter, dataloader, data_config, device, symm=True, bits=8):
    print("Start inference Backend")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]["quantization_parameters"]["scales"][0]
    input_zp = input_details[0]["quantization_parameters"]["zero_points"][0]
    input_dtype = input_details[0]["dtype"]
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    output_scale = output_details[0]["quantization_parameters"]["scales"][0]
    output_zp = output_details[0]["quantization_parameters"]["zero_points"][0]
    lers = []

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
        outputs = []
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
            interpreter.set_tensor(input_details[0]["index"], input)
            interpreter.invoke()
            tflite_out = interpreter.get_tensor(output_details[0]["index"]).astype(np.float32)
            output_data = output_scale * (tflite_out.astype(np.float32) - output_zp)
            output = torch.from_numpy(output_data)
            output = output[:, :, y_start:y_end, :]
            outputs.append(output)
        result = torch.cat(outputs, 2)
        lers.append(ctc_ler(targets, result.cpu().detach().numpy()))

    ler = sum(lers) / len(lers) * 100
    print(f"LER: {ler}")
    return 100 - ler


def inference_c(interpreter, dataloader, out_path):
    print("Start inference c")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    output_scale = output_details[0]["quantization_parameters"]["scales"][0]
    output_zp = output_details[0]["quantization_parameters"]["zero_points"][0]
    output_dtype = output_details[0]["dtype"]
    output_shape = output_details[0]["shape"]
    lers = []
    data_count = 0

    input_window_length = 296
    dataloader.reset = True
    for inputs, targets in tqdm(dataloader):
        targets = targets.cpu().detach().numpy()
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
        outputs = []
        num_samples = 0
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

            with open(out_path + f"/out_{data_count}_{num_samples}.bin", "rb") as fi:
                res = np.fromfile(fi, output_dtype).reshape(output_shape)
            tflite_out = res
            output_data = output_scale * (tflite_out.astype(np.float32) - output_zp)
            output = torch.from_numpy(output_data)
            output = output[:, :, y_start:y_end, :]
            outputs.append(output)
            num_samples += 1
        result = torch.cat(outputs, 2)
        lers.append(ctc_ler(targets, result.cpu().detach().numpy()))
        data_count += 1

    ler = sum(lers) / len(lers) * 100
    print(f"LER: {ler}")
    return 100 - ler


"""
Define Quantization model inference for metrics
"""


def forward_one(model, dataloaders, device):
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        output = None
        del output


def forward_one_Q(model, dataloader, data_config, device, symm=True, bits=8):
    for inputs, labels in dataloader:
        image = inputs.to(device)
        QuantStub(image, data_config["fp32_min"], data_config["fp32_max"], symm, bits, isHW=False)
        labels = labels.to(device)
        output = model(image)
        output = None
        del output
