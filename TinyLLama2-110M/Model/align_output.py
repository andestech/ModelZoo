import onnx
from onnxsim import simplify
from onnx import helper, TensorProto


def add_init(graph,name, vals):
    tensor = helper.make_tensor(
        name=name,
        data_type=TensorProto.INT64,
        dims=[len(vals)],
        vals=vals
    )
    graph.initializer.append(tensor)
    return name


def align_prefill_output(model,max_context=1024,prefill_context=128):
    pad = max_context-prefill_context
    if pad==0:
        return model
    elif pad<0:
        assert False,'prefill length > max context length is invalid'
    graph = model.graph
    outputs = graph.output
    output_list=[node.name for node in outputs][1:]
    tensor = helper.make_tensor(
            name='kv_cache_pad',
            data_type=TensorProto.FLOAT,
            dims=[1,12,pad,64],
            vals=[0.0]*(1 * 12 * pad * 64)
            )
    graph.initializer.append(tensor)
    for target_output in output_list:
        concate_after_node = None
        for node in graph.node:
            if target_output in node.output:
                concate_after_node = node
        if not concate_after_node == None:
            old_output = target_output
            concate_output = old_output + "_kv_align"

            concat_node = helper.make_node(
            "Concat",
            inputs=['kv_cache_pad',old_output],
            outputs=[concate_output],
            axis=2,
            name=old_output + "_ConcatPadTo1024",
            )
            idx = list(graph.node).index(concate_after_node)
            graph.node.insert(idx + 1, concat_node)
            for o in graph.output:
                if o.name == target_output:
                    o.name = concate_output
                    o.type.tensor_type.ClearField("shape")
                    shape = o.type.tensor_type.shape
                    shape.dim.add().dim_value = 1
                    shape.dim.add().dim_value = 12
                    shape.dim.add().dim_value = max_context
                    shape.dim.add().dim_value = 64
    new_model = onnx.shape_inference.infer_shapes(model,data_prop=True)
    return new_model

def align_decode_output(model,max_context=1024):
    graph = model.graph
    outputs = graph.output
    output_list=[node.name for node in outputs][1:]
    for target_output in output_list:
        slice_after_node = None
        for node in graph.node:
            if target_output in node.output:
                slice_after_node = node
        if not slice_after_node == None:
            old_output = target_output
            slice_output = old_output + "_kv_align"
            starts = [1]
            ends   = [max_context+1]
            axes   = [2]
            steps  = [1]
            starts_name = add_init(graph,slice_after_node.name + "_starts", starts)
            ends_name   = add_init(graph,slice_after_node.name + "_ends",   ends)
            axes_name   = add_init(graph,slice_after_node.name + "_axes",   axes)
            steps_name  = add_init(graph,slice_after_node.name + "_steps",  steps)
            slice_node = helper.make_node(
            "Slice",
            inputs=[old_output, starts_name, ends_name, axes_name, steps_name],
            outputs=[slice_output],
            name=slice_after_node.name + "_slice4d"
            )
            idx = list(graph.node).index(slice_after_node)
            graph.node.insert(idx + 1, slice_node)
            for o in graph.output:
                if o.name == target_output:
                    o.name = slice_output
                    o.type.tensor_type.ClearField("shape")
                    shape = o.type.tensor_type.shape
                    shape.dim.add().dim_value = 1
                    shape.dim.add().dim_value = 12
                    shape.dim.add().dim_value = max_context
                    shape.dim.add().dim_value = 64
    new_model = onnx.shape_inference.infer_shapes(model,data_prop=True)
    return new_model
