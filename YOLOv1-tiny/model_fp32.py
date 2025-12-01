import torch
from .yolov1 import Tiny_YOLOv1
import os
from common.fx_utils import andes_preprocessing, add_idd
now_dir=os.path.dirname(__file__)

def return_fp32_model():
    print('Tiny_YOLO_V1')
    model_test = Tiny_YOLOv1(now_dir + "/Model/yolov1-tiny.cfg")
    model_test.to('cpu')
    checkpoint = torch.load(now_dir + "/Model/yolov1.pt", map_location='cpu')
    model_test.load_state_dict(checkpoint)
    model_test.eval()   #keep it to ensure the model mode
    # model_test=andes_preprocessing(model_test)
    # model_test=add_idd(model_test)
    # model_test.eval()
    return model_test
