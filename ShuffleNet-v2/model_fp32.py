import torch
import torchvision.models as models
from common.fx_utils import add_idd,andes_preprocessing
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def return_fp32_model():
    with torch.no_grad():
        print("torch_vision shufflenet_v2_x1_0")
        model_test = models.shufflenet_v2_x1_0(pretrained=True)
        # model_test = andes_preprocessing(model_test)
        # model_test = add_idd(model_test)

    model_test.eval()   #keep it to ensure the model mode
    return model_test
