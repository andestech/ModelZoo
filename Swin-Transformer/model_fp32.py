import os
import ssl
import onnx
import torch
from common.onnx2torch.converter import convert

ssl._create_default_https_context = ssl._create_unverified_context
now_dir = os.path.dirname(__file__)


def return_fp32_model():
    with torch.no_grad():
        onnx_model_path = now_dir + "/Model/swin_transformer.onnx"
        model_test = onnx.load(onnx_model_path)
        model_test = convert(model_test)
    model_test.eval()   #keep it to ensure the model mode
    return model_test
