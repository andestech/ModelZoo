import os
import ssl
import onnx
import torch
from common.fx_utils import andes_preprocessing
from common.mask_attention import mask_simplify
from common.onnx2torch.converter import convert

ssl._create_default_https_context = ssl._create_unverified_context
now_dir = os.path.dirname(__file__)


def return_fp32_model():
    with torch.no_grad():
        onnx_model_path = now_dir + "/Model/model_squadv2.onnx"
        model_test = onnx.load(onnx_model_path)
        model_test = convert(model_test)
        model_test = mask_simplify(model_test, ['/bert/Expand', '/bert/Where_1'], torch.ones(1, 1, 384, 384))
        model_test(torch.ones(1, 384, dtype=torch.int32), torch.ones(1, 384, dtype=torch.int32), torch.ones(1, 384, dtype=torch.int32))
        model_test = andes_preprocessing(model_test)
    model_test.eval()  # keep it to ensure the model mode
    return model_test
