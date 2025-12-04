
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


import os
import onnx
import torch
from common import fx_utils
from common.onnx2torch.converter import convert

now_dir=os.path.dirname(__file__)


def return_fp32_model():
    with torch.no_grad():
        print("Cifar10_ResNet8")
        onnx_model_path=now_dir+'/Model/ResNet8_fp32.onnx'
        model_test = onnx.load(onnx_model_path)
        model_test = convert(model_test)
        model_test = fx_utils.dag_process(model_test)
    model_test=fx_utils.andes_preprocessing(model_test)
    model_test=fx_utils.add_idd(model_test)
    model_test.eval()
    return model_test
