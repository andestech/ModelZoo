import torch
import numpy as np
import os
import yaml
import random
import importlib
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import librosa
from pathlib import Path
from tqdm import tqdm
from .rnnoise_pre_processing import RNNoisePreProcess


"""
Loading model_cfg.yaml in the current directory
"""
now_dir=os.path.dirname(__file__)
with open(now_dir+"/model_cfg.yaml", 'r') as f:
    input_yaml = yaml.load(f, Loader=yaml.FullLoader)


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
        try:
            clean_wav = librosa.load(clean_file, sr=48000)[0]
            noisy_file_path = noisy_speech_folder / clean_file.name
            noisy_wav = librosa.load(noisy_file_path, sr=48000)[0]

            if isolate_noise:
                noise_sample = noisy_wav - clean_wav
                noise_sample *= 2**15
                isolated_noise_audio.append(noise_sample)

            # Convert audio from fp32 range into int16 range (but as fp32 type).
            clean_wav *= 2**15
            clean_speech_audio.append(clean_wav)
            noisy_wav *= 2**15
            noise_speech_audio.append(noisy_wav)
        except:
            logging.warning(f"Could not process {clean_file}, make sure it exists in both clean and noisy folders.")
            pass

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

"""
Dataset value max/min in FP32
"""
def dataset_cfg():
    return input_yaml

def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    interpreter.allocate_tensors()
    device="cpu"
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    input_dtype = input_details[0]["dtype"]
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    data_count = 0
    for clean_speech, noisy_speech in tqdm(dataloader):
        preprocess = RNNoisePreProcess(training=False)
        num_samples = len(noisy_speech) // preprocess.FRAME_SIZE
        silence_list = []
        os.makedirs(save_path + f"/test_{data_count}", exist_ok=True)
        np.array(num_samples, dtype=np.int32).tofile(save_path + f"/test_{data_count}/num_samples.bin")
        for i in range(num_samples):
            audio_window = noisy_speech[i * RNNoisePreProcess.FRAME_SIZE:
                                        i * RNNoisePreProcess.FRAME_SIZE + RNNoisePreProcess.FRAME_SIZE]
            silence, features, X, P, Ex, Ep, Exp = preprocess.process_frame(audio_window)
            if not silence:
                features = np.expand_dims(features, (0, 1)).astype(np.float32)
                input_data = torch.from_numpy(features).to(device)
                input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
                input_data = input_dtype(input_data.numpy())
            
                input_data.tofile(save_path + f"/test_{data_count}/feature_{i}.bin")
            silence_list.append(silence)
        silence_list = np.array(silence_list, dtype=input_dtype)
        silence_list.tofile(save_path + f"/test_{data_count}/silence_table.bin")

        data_count += 1


"""
Load non-default dataset.yaml
"""

def load_dataset_cfg(data_cfg_path = "none"):
    if data_cfg_path=="none":
        print(f"Load default yaml from %s " % now_dir+"/model_cfg.yaml")
    with open(data_cfg_path, 'r') as f:
        input_yaml = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Load assigned yaml from %s " % data_cfg_path)
