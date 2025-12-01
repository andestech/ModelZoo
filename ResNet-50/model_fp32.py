import ssl
import torch
import torchvision.models as models

ssl._create_default_https_context = ssl._create_unverified_context


def return_fp32_model():
    with torch.no_grad():
        model_test = models.resnet50(pretrained=True)
    model_test.eval()
    return model_test
