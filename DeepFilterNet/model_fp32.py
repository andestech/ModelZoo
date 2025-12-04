
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
import onnx
import sys
import torch.fx as fx
import os
from common.onnx2torch.converter import convert
from common.fx_utils import andes_preprocessing, add_idd
from common import fx_utils
now_dir=os.path.dirname(__file__)
mapping_insert_idd=['<built-in function add>','<built-in function mul>', '<built-in function sub>']
mapping_insert_idd_general=['built-in method exp of type object','built-in method log of type object']


def return_fp32_model():
    with torch.no_grad():
        print("deepfilter")
        onnx_model_path=now_dir+'/Model/sim_deepfilter_200.onnx'
        model_test = onnx.load(onnx_model_path)
        model_test = convert(model_test)

        # Symbolically trace an instance of the module
        trace = fx_utils.CustomTracer()
        fx_trace = trace.trace(model_test)
        fx_model = fx.graph_module.GraphModule(trace.root, fx_trace)
        graph = fx_model.graph
        model_test = dict(fx_model.named_modules())[""]
        for n in graph.nodes:
            if(n.name == 'reshape'):
                n.args = (n.args[0], (200, -1, 3072))
            if(n.name == 'reshape_1'):
                n.args = (n.args[0], (-1, 64, 8))
            if(n.name == 'reshape_2'):
                n.args = (n.args[0], (200, -1, 512))
            if(n.name == 'reshape_3'):
                n.args = (n.args[0], (200, -1, 512))
            if(n.name == 'reshape_4'):
                n.args = (n.args[0], (200, -1, 64, 8))
            if(n.name == 'reshape_5'):
                n.args = (n.args[0], (200, -1, 512))
            if(n.name == 'reshape_6'):
                n.args = (n.args[0], (200, -1, 64, 8))
            if(n.name == 'reshape_7'):
                n.args = (n.args[0], (200, -1, 512))
            if(n.name == 'reshape_8'):
                n.args = (n.args[0], (-1, 64, 8))
            if(n.name == 'reshape_9'):
                n.args = (n.args[0], (-1, 200, 512))
            if(n.name == 'reshape_10'):
                n.args = (n.args[0], (-1, 200, 64, 8))
            if(n.name == 'reshape_11'):
                n.args = (n.args[0], (200, -1, 64, 8))
            if(n.name == 'reshape_12'):
                n.args = (n.args[0], (200, -1, 512))
            if(n.name == 'reshape_13'):
                n.args = (n.args[0], (-1, 200, 10, 96))
            if(n.name == 'reshape_14'):
                n.args = (n.args[0], (-1, 200, 5, 2, 96))

        graph.lint()
        model_test = torch.fx.GraphModule(model_test, graph)
        model_test.recompile()

        model_test = fx_utils.andes_preprocessing(model_test)
        model_test = fx_utils.dag_process(model_test)
    model_test = fx_utils.add_idd(model_test)


    model_test.eval()
    # print(model_test)
    return model_test
