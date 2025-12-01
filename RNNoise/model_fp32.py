import torch
import torchvision.models as models
import onnx
import sys
import torch.fx as fx
import os
from common.onnx2torch.converter import convert
from common.fx_utils import andes_preprocessing, add_idd
now_dir=os.path.dirname(__file__)
mapping_insert_idd=['<built-in function add>','<built-in function mul>', '<built-in function sub>']
mapping_insert_idd_general=['built-in method exp of type object','built-in method log of type object']


def return_fp32_model():
    with torch.no_grad():
        print("rnnnoise")
        onnx_model_path=now_dir+'/Model/rnnoise_fp32.onnx'
        model_test = onnx.load(onnx_model_path)
        model_test = convert(model_test)
    model_test.eval()
    print(model_test)
    return model_test
