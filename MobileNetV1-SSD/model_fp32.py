
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
from .dataset import VOC_classnames
from .Model.SSDs.mobilenetv1_ssd import create_mobilenetv1_ssd


def return_fp32_model():
    model_test = create_mobilenetv1_ssd(len(VOC_classnames()), is_test=False)
    model_test.load(os.path.dirname(__file__) + '/Model/SSDs/weights/mobilenet-v1-ssd-mp-0_675.pth')
    model_test.eval()
    return model_test
