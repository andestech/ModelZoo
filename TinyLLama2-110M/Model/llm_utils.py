import onnx

def find_value_info(graph, name):
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.name == name:
            return vi
    return None

def set_shape(vi, dims):
    t = vi.type.tensor_type
    while len(t.shape.dim) < len(dims):
        t.shape.dim.add()
    for i, v in enumerate(dims):
        d = t.shape.dim[i]
        d.ClearField("dim_param")
        if isinstance(v, int):
            d.dim_value = v
        elif v is None:
            pass
        else:
            d.dim_param = str(v)

def remove_KV_input(model):
    graph = model.graph
    for n in graph.node:
        if n.op_type == "Concat" and 'past_key_values' in n.input[0]:
            keep_input = n.input[1]  #keep other branch input
            n.ClearField("input")
            n.input.extend([keep_input])
    input_list = [i for i in graph.input if all(d.dim_value != 0 for d in i.type.tensor_type.shape.dim)]
    del graph.input[:]
    graph.input.extend(input_list)
    return model
