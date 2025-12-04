
# Copyright (C) 2023-2025 Andes Technology Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from .yolov2 import Tiny_YOLOv2
from .utils import load_weights
import os
from common.fx_utils import andes_preprocessing, add_idd
now_dir=os.path.dirname(__file__)

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
    model_test = Tiny_YOLOv2(now_dir+"/Model/yolov2-tiny-voc.cfg")
    load_weights(model_test, now_dir+"/Model/yolov2-tiny-voc.weights", version="v2")
    a=torch.nn.Sequential(
        torch.nn.ZeroPad2d(padding=(0,1,0,1)),
        torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
    )
    set_layer(model_test,"models.11",a)
    # YOLOv2
    # print('YOLO_V2')
    # model_test = Tiny_YOLOv2("model_ws_package/tiny_yolo_v2_torchvision/yolov2.cfg")
    # load_weights(model_test, "model_ws_package/tiny_yolo_v2_torchvision/yolov2-voc.weights", version="v3")
    # model_test=andes_preprocessing(model_test)
    # model_test=add_idd(model_test)
    model_test.eval()   #keep it to ensure the model mode
    return model_test
