
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


import os
import copy
import math
import yaml
import glob
import torch
import numpy as np
from tqdm import tqdm
from libdf import DF
from .df.io import load_audio
from .df.enhance import df_features

"""
Loading model_cfg.yaml in the current directory
"""
now_dir=os.path.dirname(__file__)
with open(now_dir+"/model_cfg.yaml", 'r') as f:
    input_yaml = yaml.load(f, Loader=yaml.FullLoader)

"""
Template for return data pair
"""
def return_dataset():
    sr = 48000
    RESAMPLE_METHOD = "sinc_fast"
    clean_speech_audio = []
    noisy_speech_audio = []
    clean_path = input_yaml['clean_wave_path']
    noisy_path = input_yaml['noisy_wave_path']
    clean_files = sorted(glob.glob(clean_path + "/*.wav"))
    noisy_files = sorted(glob.glob(noisy_path + "/*.wav"))
    for cleanfn, noisyfn in zip(clean_files, noisy_files):
        clean, _ = load_audio(cleanfn, sr, method=RESAMPLE_METHOD)
        noisy, _ = load_audio(noisyfn, sr, method=RESAMPLE_METHOD)
        clean_speech_audio.append(clean)
        noisy_speech_audio.append(noisy)
    val_dataloader = zip(clean_speech_audio, noisy_speech_audio)
    return val_dataloader,val_dataloader,val_dataloader,val_dataloader

"""
Dataset value max/min in FP32
"""
def dataset_cfg():
    return input_yaml

def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    dataloader=copy.deepcopy(dataloader)
    # interpreter.allocate_tensors() # model includes custom op cannot allocate tensors
    feature_input_1_idx = 0
    feature_input_2_idx = 1
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_1_scale = input_details[feature_input_1_idx]['quantization_parameters']['scales'][0]
    input_1_zp = input_details[feature_input_1_idx]['quantization_parameters']['zero_points'][0]
    input_1_dtype = input_details[feature_input_1_idx]['dtype']
    input_1_qmin = np.iinfo(input_1_dtype).min
    input_1_qmax = np.iinfo(input_1_dtype).max
    input_2_scale = input_details[feature_input_2_idx]['quantization_parameters']['scales'][0]
    input_2_zp = input_details[feature_input_2_idx]['quantization_parameters']['zero_points'][0]
    input_2_dtype = input_details[feature_input_2_idx]['dtype']
    input_2_qmin = np.iinfo(input_2_dtype).min
    input_2_qmax = np.iinfo(input_2_dtype).max

    input_yaml=dataset_cfg()
    test_wav_num=input_yaml['test_wav_num']
    df_state = DF(
        sr=48000,
        fft_size=960,
        hop_size=480,
        nb_bands=32,
        min_nb_erb_freqs=2,
    )
    print("state initialize")
    nb_df = 96
    data_length = 96000
    data_count = 0
    test_wav_num=input_yaml['test_wav_num']
    with torch.no_grad():
        for clean_speech, noisy_speech in tqdm(dataloader):
            final_denoised_audio=[]
            os.makedirs(save_path + f"/test_{data_count}", exist_ok=True)
            num_samples = math.ceil(noisy_speech.shape[1] / data_length)
            np.array(num_samples, dtype=np.int32).tofile(save_path + f"/test_{data_count}/num_samples.bin")

            # encoder_gru_input_idx = 2
            # dfdecoder_gru_input_idx = 3
            # encoder_gru_scale = input_details[encoder_gru_input_idx]['quantization_parameters']['scales'][0]
            # encoder_gru_zp = input_details[encoder_gru_input_idx]['quantization_parameters']['zero_points'][0]
            # input_3_dtype = input_details[encoder_gru_input_idx]['dtype']
            # input_3_qmin = np.iinfo(input_3_dtype).min
            # input_3_qmax = np.iinfo(input_3_dtype).max
            # encoder_gru_shape = tuple(input_details[encoder_gru_input_idx]['shape'])
            # encoder_gru_state = torch.zeros(encoder_gru_shape).div_(encoder_gru_scale).round_().add_(encoder_gru_zp).clamp_(input_3_qmin, input_3_qmax)
            # encoder_gru_state = input_3_dtype(encoder_gru_state.numpy())
            # dfdecoder_gru_scale = input_details[dfdecoder_gru_input_idx]['quantization_parameters']['scales'][0]
            # dfdecoder_gru_zp = input_details[dfdecoder_gru_input_idx]['quantization_parameters']['zero_points'][0]
            # input_4_dtype = input_details[dfdecoder_gru_input_idx]['dtype']
            # input_4_qmin = np.iinfo(input_4_dtype).min
            # input_4_qmax = np.iinfo(input_4_dtype).max
            # dfdecoder_gru_shape = tuple(input_details[dfdecoder_gru_input_idx]['shape'])
            # dfdecoder_gru_state = torch.zeros(dfdecoder_gru_shape).div_(dfdecoder_gru_scale).round_().add_(dfdecoder_gru_zp).clamp_(input_4_qmin, input_4_qmax)
            # dfdecoder_gru_state = input_4_dtype(dfdecoder_gru_state.numpy())

            for i in range(num_samples):
                noisy_data = noisy_speech[:, data_length * i : data_length * (i+1)]
                if noisy_data.shape[1] <= data_length:
                    zeros = torch.zeros(1, data_length - noisy_data.shape[1])
                    noisy_data = torch.cat((noisy_data, zeros), dim=1)
                    clean_speech = torch.cat((clean_speech, zeros), dim=1)
                spec, erb_feat, spec_feat = df_features(noisy_data, df_state, nb_df)
                erb_feat.div_(input_1_scale).round_().add_(input_1_zp).clamp_(input_1_qmin, input_1_qmax)
                erb_feat = input_1_dtype(erb_feat.numpy())
                erb_feat = erb_feat.transpose(0, 2, 3, 1)
                erb_feat.tofile(save_path + f"/test_{data_count}/feature_{i}_0.bin")
                spec_feat.div_(input_2_scale).round_().add_(input_2_zp).clamp_(input_2_qmin, input_2_qmax)
                spec_feat = input_2_dtype(spec_feat.numpy())
                spec_feat = spec_feat.transpose(0, 4, 2, 3, 1)
                spec_feat.tofile(save_path + f"/test_{data_count}/feature_{i}_1.bin")
                # encoder_gru_state.tofile(save_path + f"/test_{data_count}/feature_{i}_2.bin")
                # dfdecoder_gru_state.tofile(save_path + f"/test_{data_count}/feature_{i}_3.bin")
            data_count += 1
            if data_count == test_wav_num:
                break


"""
Load non-default dataset.yaml
"""

def load_dataset_cfg(data_cfg_path = "none"):
    if data_cfg_path=="none":
        print(f"Load default yaml from %s " % now_dir+"/model_cfg.yaml")
    with open(data_cfg_path, 'r') as f:
        input_yaml = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Load assigned yaml from %s " % data_cfg_path)
