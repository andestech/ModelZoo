import os
import torch
from common import fx_utils
from .Model.bcresnet import BCResNets

now_dir = os.path.dirname(__file__)


def return_fp32_model():
    with torch.no_grad():
        print("BCResNets")
        tau = 8
        model_test = BCResNets(int(tau * 8))
        model_test.load_state_dict(torch.load(now_dir + '/Model/bcresnet_fp32.pth', map_location = "cpu"))
    model_test = fx_utils.andes_preprocessing(model_test)
    model_test.eval()
    return model_test
