
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
import sys
import ssl
import onnx
from common import fx_utils
from .Model.mcunet.model_zoo import build_model

sys.path.append('../')
sys.path.append('../../')
now_dir=os.path.dirname(__file__)
ssl._create_default_https_context = ssl._create_unverified_context


def return_fp32_model():
    model_test, resolution, description = build_model('mcunet-vww2', pretrained=True)
    model_test.eval()
    model_test = fx_utils.add_idd(model_test)
    return model_test
