import torch
from .yolov2 import Tiny_YOLOv2
from .utils import load_weights


def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer
def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)

def return_fp32_model():
    # tiny-YOLOv2
    print('Tiny_YOLO_V2')
    model_test = Tiny_YOLOv2("ModelZoo/tiny_yolo_v2_torchvision/yolov2-tiny-voc.cfg")
    load_weights(model_test, "ModelZoo/tiny_yolo_v2_torchvision/yolov2-tiny-voc.weights", version="v2")
    a=torch.nn.Sequential(
        torch.nn.ZeroPad2d(padding=(0,1,0,1)),
        torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
    )
    set_layer(model_test,"models.11",a)
    # YOLOv2
    # print('YOLO_V2')
    # model_test = Tiny_YOLOv2("ModelZoo/tiny_yolo_v2_torchvision/yolov2.cfg")
    # load_weights(model_test, "ModelZoo/tiny_yolo_v2_torchvision/yolov2-voc.weights", version="v3")
    model_test.eval()   #keep it to ensure the model mode
    return model_test
