import torch
import torchvision.models as models
from .fx_utils import add_idd

def return_fp32_model():
    with torch.no_grad():
        print("torch_vision resnet50")
        model_test = models.resnet50(pretrained=True)

    model_test.eval()   #keep it to ensure the model mode
    return model_test
