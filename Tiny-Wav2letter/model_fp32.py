import torch
import torchaudio.models as models
import os
import onnx
from common.fx_utils import andes_preprocessing
from common.onnx2torch.converter import convert

now_dir = os.path.dirname(__file__)


def return_fp32_model():
    with torch.no_grad():
        onnx_model_path = now_dir + "/Model/tiny_wav2letter.onnx"
        model_test = onnx.load(onnx_model_path)
        model_test = convert(model_test)
        model_test = andes_preprocessing(model_test)

    model_test.eval()  # keep it to ensure the model mode
    return model_test
