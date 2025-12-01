import torch
import torchvision.models as models
from common.fx_utils import add_idd,andes_preprocessing
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def return_fp32_model():
    with torch.no_grad():
        print("torch_vision squeezenet1_1")
        model_test = models.squeezenet1_1(pretrained=True)
        # model_test = andes_preprocessing(model_test)
        # model_test = add_idd(model_test)

    model_test.eval()   #keep it to ensure the model mode
    print(model_test)
    return model_test
