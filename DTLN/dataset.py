
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
import yaml
import torch
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path


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


def load_clean_noisy_wavs(clean_speech_folder, noisy_speech_folder, isolate_noise):
    """Return lists of loaded audio for folders of clean and noisy audio.

    It is expected the clean speech folder and noisy speech folder have wav files with the same names.
    The wav files in the noisy speech folder will have the same speech but overlaid with some noise.

    If isolate_noise is True then this function will also isolate and return the noise along with the
    clean and noisy speech."""
    clean_speech_audio = []
    noise_speech_audio = []
    isolated_noise_audio = []

    for clean_file in sorted(clean_speech_folder.rglob('*.wav')):
        clean_wav = librosa.load(clean_file, sr=16000)[0]
        noisy_file_path = noisy_speech_folder / clean_file.name
        noisy_wav = librosa.load(noisy_file_path, sr=16000)[0]
        clean_speech_audio.append(clean_wav)
        noise_speech_audio.append(noisy_wav)
    return clean_speech_audio, noise_speech_audio, isolated_noise_audio


"""
Template for return data pair
"""
def return_dataset():
    batch_size = 32 
    window_size = 2000
    clean_wavs = Path(input_yaml['clean_wave_path'])
    noisy_wavs = Path(input_yaml['noisy_wave_path'])
    clean_speech_audio, noisy_speech_audio, _ = load_clean_noisy_wavs(clean_wavs, noisy_wavs, isolate_noise=False)
    val_dataloader = zip(clean_speech_audio, noisy_speech_audio)
    clean_wavs_train = Path(input_yaml['train_clean_wave_path'])
    noisy_wavs_train = Path(input_yaml['train_noisy_wave_path'])
    clean_speech_audio, _, noise_audio_list = load_clean_noisy_wavs(clean_wavs_train, noisy_wavs_train, isolate_noise=True)
    train_dataloader = zip(clean_speech_audio, noise_audio_list)
    return train_dataloader,val_dataloader,val_dataloader,val_dataloader


def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    def stftLayer(x, blockLen, block_shift):
        frames = x.unfold(-1, blockLen, block_shift)
        return frames
        print(frames.size())
        stft_dat = torch.fft.rfft(frames)
        mag = stft_dat.abs()
        phase = stft_dat.angle()
        return mag, phase

    print("Start generate test_bin")
    # interpreter.allocate_tensors()
    device="cpu"
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    blockLen = 512
    block_shift = 128
    data_count = 0

    with torch.no_grad():
        for clean_speech, noisy_speech in tqdm(dataloader):
            len_orig = len(noisy_speech)
            zero_pad = np.zeros(384)
            noisy_speech_pad = np.concatenate((zero_pad, noisy_speech, zero_pad), axis=0)
            # preprocess
            x = stftLayer(torch.from_numpy(noisy_speech_pad.astype(np.float32)), blockLen, block_shift)
            num_samples = 0
            os.makedirs(save_path + f"/test_{data_count}", exist_ok=True)
            for i in range(x.shape[0]):
                input_data = (x[i,:].reshape(1, 1, 512).clone()).to(device)
                input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
                input_data = input_dtype(input_data.numpy())
                input_data.tofile(save_path + f"/test_{data_count}/feature_{i}.bin")
                num_samples += 1
            np.array(num_samples, dtype=np.int32).tofile(save_path + f"/test_{data_count}/num_samples.bin")
            data_count += 1
