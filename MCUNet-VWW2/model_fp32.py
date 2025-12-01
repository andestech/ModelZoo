import os
import sys
import ssl
import onnx
from common import fx_utils
from .Model.mcunet.model_zoo import build_model

sys.path.append('../')
sys.path.append('../../')
now_dir=os.path.dirname(__file__)
ssl._create_default_https_context = ssl._create_unverified_context


def return_fp32_model():
    model_test, resolution, description = build_model('mcunet-vww2', pretrained=True)
    model_test.eval()
    model_test = fx_utils.add_idd(model_test)
    return model_test
