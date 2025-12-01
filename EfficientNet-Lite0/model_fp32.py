import os
import sys
import onnx
import torch
from common import fx_utils
from common.onnx2torch.converter import convert

sys.path.append("../")
sys.path.append("../../")
now_dir = os.path.dirname(__file__)


def return_fp32_model():
    with torch.no_grad():
        print("EfficientNet_lite")
        onnx_model_path = now_dir + "/Model/efficient_lite0_fp32.onnx"
        model_test = onnx.load(onnx_model_path)
        model_test = convert(model_test)
        model_test = fx_utils.dag_process(model_test)
    model_test.eval()

    gm_o = torch.fx.symbolic_trace(model_test)
    model_dict = dict(gm_o.named_modules())
    new_vertices = gm_o.graph
    for node in new_vertices.nodes:
        if node.name == "reshape":
            node.args = (node.args[0], (-1, 1280))
        if isinstance(model_dict.get(node.target), torch.nn.Softmax):
            model_dict.pop(node.target)
            node.replace_all_uses_with(node.prev)
            gm_o.graph.erase_node(node)
    new_vertices.lint()
    model_test = torch.fx.graph_module.GraphModule(model_dict, new_vertices)
    model_test.recompile()
    model_test.eval()
    return model_test
