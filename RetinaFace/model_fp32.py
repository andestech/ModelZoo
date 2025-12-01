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
        print("RestinaFace 1.0 480x480")
        onnx_model_path=now_dir+'/Model/simp_480.onnx'
        model_test = onnx.load(onnx_model_path)
        model_test = convert(model_test)
    model_test=fx_utils.andes_preprocessing(model_test)
    model_test=fx_utils.add_idd(model_test)
    model_test.eval()#keep it to ensure the model mode
    return model_test
