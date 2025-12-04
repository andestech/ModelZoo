
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
import torchvision.models as models
from common.fx_utils import add_idd,andes_preprocessing
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def return_fp32_model():
    with torch.no_grad():
        print("torch_vision squeezenet1_1")
        model_test = models.squeezenet1_1(pretrained=True)
        # model_test = andes_preprocessing(model_test)
        # model_test = add_idd(model_test)

    model_test.eval()   #keep it to ensure the model mode
    print(model_test)
    return model_test
