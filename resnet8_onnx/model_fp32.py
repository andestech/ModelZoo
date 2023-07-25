import torch
import torchvision.models as models
import onnx
import sys
import os
from common import fx_utils
from common.onnx2torch.converter import convert
now_dir=os.path.dirname(__file__)


def return_fp32_model():
    with torch.no_grad():
        print("Cifar10_ResNet8")
        onnx_model_path=now_dir+'/ONNX_models/Cifar10/ResNet8.onnx'
        model_test = onnx.load(onnx_model_path)
        model_test = convert(model_test)
        model_test = fx_utils.dag_process(model_test)
    model_test.eval()#keep it to ensure the model mode
    return model_test
