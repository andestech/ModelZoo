import torch
from .yolov1 import Tiny_YOLOv1


def return_fp32_model():
    print('Tiny_YOLO_V1')
    model_test = Tiny_YOLOv1("model_ws_package/tiny_yolo_v1_torchvision/yolov1-tiny.cfg")
    model_test.to('cpu')
    checkpoint = torch.load("model_ws_package/tiny_yolo_v1_torchvision/weights/yolov1.pt", map_location='cpu')
    model_test.load_state_dict(checkpoint)
    model_test.eval()   #keep it to ensure the model mode
    return model_test
