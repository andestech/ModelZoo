import os
from .dataset import VOC_classnames
from .Model.SSDs.mobilenetv1_ssd import create_mobilenetv1_ssd


def return_fp32_model():
    model_test = create_mobilenetv1_ssd(len(VOC_classnames()), is_test=False)
    model_test.load(os.path.dirname(__file__) + '/Model/SSDs/weights/mobilenet-v1-ssd-mp-0_675.pth')
    model_test.eval()
    return model_test
