
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
