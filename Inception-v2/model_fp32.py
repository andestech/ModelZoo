import os
import sys
import onnx
import torch
from common import fx_utils
from common.onnx2torch.converter import convert

sys.path.append('../')
sys.path.append('../../')
now_dir=os.path.dirname(__file__)


def return_fp32_model():
    with torch.no_grad():
        onnx_model_path = now_dir + '/Model/Inception_v2_fp32.onnx'
        model_test = onnx.load(onnx_model_path)
        model_test = convert(model_test)
        model_test = fx_utils.dag_process(model_test)
    model_test.eval()
    model_test = fx_utils.andes_preprocessing(model_test)
    model_test = fx_utils.add_idd(model_test)
    return model_test
