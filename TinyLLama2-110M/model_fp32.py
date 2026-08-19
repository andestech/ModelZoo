import os
import onnx
import torch
from common import fx_utils
from common.onnx2torch.converter import convert

now_dir=os.path.dirname(__file__)


def return_fp32_model(phase='full_context'):
    if phase == 'full_context':
        onnx_model_path = now_dir + '/Model/full_context.onnx'
    elif phase == 'decode':
        onnx_model_path = now_dir + '/Model/decode.onnx'
    elif phase == 'prefill_128':
        onnx_model_path = now_dir + '/Model/prefill_128.onnx'
    else:
        assert False,'no phase defined'
    with torch.no_grad():
        print("tinyllama2_110M")
        model_test = onnx.load(onnx_model_path)
        model_test = convert(model_test)
        model_test = fx_utils.dag_process(model_test)
    model_test = fx_utils.andes_preprocessing(model_test)
    model_test = fx_utils.add_idd(model_test)
    model_test.eval()
    return model_test
