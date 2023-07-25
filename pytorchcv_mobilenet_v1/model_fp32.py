import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

def return_fp32_model():
    with torch.no_grad():
        print("pytorchcv mobilenet v1")
        model_test = ptcv_get_model('mobilenet_w1', pretrained=True)
    model_test.eval()   #keep it to ensure the model mode
    return model_test
