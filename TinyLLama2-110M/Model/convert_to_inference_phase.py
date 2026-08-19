import onnx
from onnxsim import simplify
from llm_utils import set_shape, find_value_info, remove_KV_input
from align_output import align_prefill_output,align_decode_output

max_sequence_length = 1024
onnx_path = 'tinyllama-110M.onnx'
"""
Prepare phase name
"""
signatures={}
signatures['full_context'] = {'batch_size': 1,'sequence_length': 1024,'past_sequence_length':0}
signatures['prefill_128'] = {'batch_size': 1,'sequence_length': 128,'past_sequence_length':0}
signatures['decode'] = {'batch_size': 1,'sequence_length': 1,'past_sequence_length':1024}
for key in signatures:
    print(f'Now dumping phase {key}')
    model = onnx.load(onnx_path)
    graph = model.graph
    inp_names  = [i.name for i in graph.input]
    out_names  = [o.name for o in graph.output]
    """
    Set shape for each input in differ signatures
    """
    set_shape(find_value_info(graph, 'input_ids'),[signatures[key]['batch_size'],signatures[key]['sequence_length']])
    set_shape(find_value_info(graph, 'attention_mask'),[signatures[key]['batch_size'],signatures[key]['sequence_length']+signatures[key]['past_sequence_length']])
    set_shape(find_value_info(graph, 'position_ids'),[signatures[key]['batch_size'],signatures[key]['sequence_length']])
    """
    transformer kv-cache inputs
    """
    for input_name in inp_names:
        if not input_name in ('input_ids','attention_mask','position_ids'):
            set_shape(find_value_info(graph, input_name),[signatures[key]['batch_size'],12,signatures[key]['past_sequence_length'],64])

    """
    remove kv-cache inputs for prefill model and full_context
    """
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    model,_ = simplify(model)
    if key != 'decode':
        model = remove_KV_input(model)
    if key=='decode':
        model=align_decode_output(model,max_context=max_sequence_length)
    elif 'prefill' in key:
        model=align_prefill_output(model,max_context=max_sequence_length,prefill_context=signatures[key]['sequence_length'])
    model,_ = simplify(model)
    onnx.save(model,f'{key}.onnx')

