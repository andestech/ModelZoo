import os
import onnx
import torch
import torch.nn as nn
from common import fx_utils
from common.onnx2torch.converter import convert

now_dir = os.path.dirname(__file__)
with torch.no_grad():
    onnx_model_path_1 = now_dir + "/Model/DTLN_model1_fp32.onnx"
    model_1 = onnx.load(onnx_model_path_1)
    model_1 = convert(model_1)
    model_1 = fx_utils.dag_process(model_1)
    onnx_model_path_2 = now_dir + "/Model/DTLN_model2_fp32.onnx"
    model_2 = onnx.load(onnx_model_path_2)
    model_2 = convert(model_2)
    model_2 = fx_utils.dag_process(model_2)


class fftLayer(nn.Module):
    def __init__(self):
        super(fftLayer, self).__init__()
        self.cust_module = True
        self.leaf_m_register=True

    def forward(self, x):
        stft_dat = torch.fft.rfft(x)
        mag = stft_dat.abs()
        phase = stft_dat.angle()
        return mag,phase


class ifftLayer(nn.Module):
    def __init__(self):
        super(ifftLayer, self).__init__()
        self.cust_module = True
        self.leaf_m_register=True

    def forward(self, mag, phase):
        s1_stft = torch.polar(mag, phase)
        time_domain_frames = torch.fft.irfft(s1_stft)
        return time_domain_frames


class DTLN_model(nn.Module):
    def __init__(self):
        super(DTLN_model, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.fftLayer = fftLayer()
        self.ifftLayer = ifftLayer()

    def forward(self, x, state1, state2):
        mag, angle = self.fftLayer(x)
        out1, state1 = self.model_1(mag, state1)
        estimated_mag = mag * out1
        estimated_frames_1 = self.ifftLayer(estimated_mag, angle)
        out2, state2 = self.model_2(estimated_frames_1, state2)
        return out2, state1, state2


def return_fp32_model():
    with torch.no_grad():
        model_test = DTLN_model()
    model_test.eval()
    return model_test