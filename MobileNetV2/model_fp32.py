import os
import ssl
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

ssl._create_default_https_context = ssl._create_unverified_context
now_dir = os.path.dirname(__file__)


def return_fp32_model():
    with torch.no_grad():
        model_test = ptcv_get_model('mobilenetv2_w1', pretrained=True)
    model_test.eval()
    return model_test
